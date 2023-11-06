from abc import ABC
from copy import deepcopy
import pdb

import torch
import torch.nn as nn

from dataclasses import dataclass
from transformers import AutoModel, AutoConfig

from triplet_mask import construct_mask

from graph_models import GCN, GraphSAGE, GAT, StandardGCN,VariationalGCNEncoder,VariationalGATEncoder
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import VGAE
import random

def build_model(args) -> nn.Module:
    if args.model.startswith('g'):
        return CustomGraphBertModel(args)
    if args.model.startswith('vgae'):
        return CustomVGAEBertModel(args)
    else:
         return CustomBertModel(args)


@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor
    # vgae_loss: torch.tensor # only valid for vage

# this is the original model
class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch # number of pre-batch used for negatives
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size  # 600 (300 X 2)
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size) # 600 X 768
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        if self.args.freeze_lm:
            for name,param in self.hr_bert.named_parameters():
                if name.startswith("bert"):
                    param.requires_grad = False
                
        self.tail_bert = deepcopy(self.hr_bert) # this copies an object, not a pointer
        

    def _encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)

        # print ('entering modeling')
        
        
        return cls_output # shape: 512x768

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                only_ent_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids)
        
        # hr_token_ids[512,50] 
        # hr_vector, contains head_name, head_desc, relation
        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids)
        # tail_vector, contains tail_name, tail_desc
        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        # head_vector, contains head_name, head_desc
        head_vector = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)


        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        # hr_vector shape: (2048,768) (batch_size, 768)

        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)# 300
        labels = torch.arange(batch_size).to(hr_vector.device) # 0..299

        

        # prediction part so simple: e^{hr} * e^{t}
        logits = hr_vector.mm(tail_vector.t()) # (batch_size, batch_size)
        if self.training:# smooth the logits? minus a small margin (0.02) on the diagonal
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()
        
        
        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)
        
        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)
        # logits shape [300,900], 900:self 300, and 2x300 pre_batch
        
        if self.args.use_self_negative and self.training: # default true for training
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp() # shape [300]
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)
        # logits shape: [300, 901], the last dim is torch.Size([300, 1]) ????wtm
        
        
        return {'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach()}

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        
        
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data'] # a list of data objects
        # batch_size x num_neg
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight # pre_batch_weight default 0.5
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)
        # import pdb;pdb.set_trace()
        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}

class CustomGraphBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        # if self.args.freeze_lm:
        #     for name,param in self.hr_bert.named_parameters():
        #         if name.startswith("bert"):
        #             param.requires_grad = False
        
        self.neighbor_bert = deepcopy(self.hr_bert)
        self.tail_bert = deepcopy(self.hr_bert)

        if self.args.model=='gcn':
            self.gcn = GCN(768, 768)
        elif self.args.model=='gcns':
            self.gcn = StandardGCN(768, 768)
        elif self.args.model=='gat':
            self.gcn = GAT(768, 768)
        else:
            self.gcn = GraphSAGE(768,768)

       

    def _encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)
        
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)

    
        return cls_output # shape: 512x768

    # region
    # def _encode_neighbors(self, encoder, token_ids, mask, token_type_ids, hr_token_ids, hr_mask, hr_token_type_ids):
        # this is to encode neighbors
        # neighbor_token_ids shape: [512, 50, 50] -> [batch_size, neighbor_counts, mx_len]
        # TODO: we can only done this by each sample..
        n_neighbors = token_ids.shape[1]
        hr_token_ids = hr_token_ids.unsqueeze(1)
        hr_mask = hr_mask.unsqueeze(1)
        hr_token_type_ids = hr_token_type_ids.unsqueeze(1)

        neighbor_output = []
        for sample_token_ids, sample_mask, sample_token_type_ids, head_token_ids, head_mask, head_token_type_ids in zip(token_ids, mask, token_type_ids,hr_token_ids, hr_mask, hr_token_type_ids):     
            
            token_ids = torch.cat((head_token_ids,sample_token_ids),0)
            mask = torch.cat((head_mask,sample_mask),0)
            token_type_ids = torch.cat((head_token_type_ids,sample_token_type_ids),0)
            
            # pdb.set_trace()
            # BERT asks for [batch_size, seq_len] shape as input
            outputs = encoder(input_ids=token_ids,attention_mask=mask,token_type_ids=token_type_ids,return_dict=True)
            last_hidden_state = outputs.last_hidden_state
            node_embedding = last_hidden_state[:, 0, :] # [51, 768]

            # generate star graph
            generated_adj = nx.star_graph(n_neighbors)
            nx_adj = from_networkx(generated_adj)['edge_index'].to(self.device)

            graph_output = self.gcn(node_embedding,nx_adj) # 51, 768
            neighbor_output.append(graph_output)
  

        return torch.stack(neighbor_output) # shape: 512x768 [batch_size, 768]
    # endregion
    
    
    def _encode_neighbors(self, encoder, token_ids, mask, token_type_ids, h_token_ids, h_mask, h_token_type_ids, 
                          neighbor_rel_ids=None, neighbor_rel_mask=None, neighbor_rel_type_ids=None # new
                          ):
        # this is to encode neighbors
        # neighbor_token_ids shape: [512, 50, 50] -> [batch_size, neighbor_counts, mx_len]
        n_neighbors = token_ids.shape[1]
        
        h_token_ids = h_token_ids.unsqueeze(1)
        h_mask = h_mask.unsqueeze(1)
        h_token_type_ids = h_token_type_ids.unsqueeze(1)
        
        
        # make parallel
        neighbor_token_ids = torch.cat((h_token_ids,token_ids),1)
        neighbor_mask = torch.cat((h_mask,mask),1)
        neighbor_token_type_ids = torch.cat((h_token_type_ids,token_type_ids),1)
        
        
        max_token=neighbor_token_ids.shape[-1]
        batch_size = neighbor_token_ids.shape[0]
        max_neighbor = neighbor_token_ids.shape[1]
        bert_output = encoder(input_ids=neighbor_token_ids.reshape(-1,max_token),attention_mask=neighbor_mask.reshape(-1,max_token),token_type_ids=neighbor_token_type_ids.reshape(-1,max_token),return_dict=True)
        bert_output = bert_output.last_hidden_state
        node_embedding = bert_output[:, 0, :].reshape(batch_size,max_neighbor,-1) # torch.Size([100, 6, 768]) -> (batch_size, max_neighbor, dim)
        # import pdb;pdb.set_trace()
        
        #build star graph
        generated_adj = nx.star_graph(n_neighbors)
        nx_adj = from_networkx(generated_adj)['edge_index'].to(self.device)
        # build batch for GCN
        graph_data_list = [Data(x=node_features,edge_index=nx_adj) for node_features in node_embedding]
        
        
        my_loader = DataLoader(graph_data_list, batch_size=batch_size, shuffle=False)
        batch=next(iter(my_loader))
        neighbor_output = self.gcn(batch.x,batch.edge_index) # (600,768)
        neighbor_output = neighbor_output.reshape(batch_size,max_neighbor,-1).sum(1) # add among the neighbor dimension
        
        # region
        # for sample_token_ids, sample_mask, sample_token_type_ids, head_token_ids, head_mask, head_token_type_ids in zip(token_ids, mask, token_type_ids,hr_token_ids, hr_mask, hr_token_type_ids):     
            
        #     token_ids = torch.cat((head_token_ids,sample_token_ids),0)
        #     mask = torch.cat((head_mask,sample_mask),0)
        #     token_type_ids = torch.cat((head_token_type_ids,sample_token_type_ids),0)
            
        #     # pdb.set_trace()
        #     # BERT asks for [batch_size, seq_len] shape as input
        #     outputs = encoder(input_ids=token_ids,attention_mask=mask,token_type_ids=token_type_ids,return_dict=True)
        #     last_hidden_state = outputs.last_hidden_state
        #     node_embedding = last_hidden_state[:, 0, :] # [51, 768]

        #     # generate star graph
        #     generated_adj = nx.star_graph(n_neighbors)
        #     nx_adj = from_networkx(generated_adj)['edge_index'].to(self.device)

        #     graph_output = self.gcn(node_embedding,nx_adj) # 51, 768
        #     neighbor_output.append(graph_output)
        # endregion

        return neighbor_output # shape: 512x768 [batch_size, 768]

 
        

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                neighbor_token_ids, neighbor_mask, neighbor_token_type_ids, # new
                neighbor_rel_ids=None, neighbor_rel_mask=None, neighbor_rel_type_ids=None, # new
                only_ent_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids)
        
        # hr_token_ids[512,50]
        # set device
        self.device=hr_token_ids.device

        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids)
        
        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)

        head_vector = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)
        
        if neighbor_token_ids is not None:
         
            
            neighbor_vector = self._encode_neighbors(self.neighbor_bert,
                                    token_ids=neighbor_token_ids,
                                    mask=neighbor_mask,
                                    token_type_ids=neighbor_token_type_ids, 
                                    h_token_ids=head_token_ids, 
                                    h_mask=head_mask, 
                                    h_token_type_ids=head_token_type_ids, 
                                    neighbor_rel_ids=neighbor_rel_ids, #new
                                    neighbor_rel_mask=neighbor_rel_mask,#new
                                    neighbor_rel_type_ids=neighbor_rel_type_ids,#new
                                    )
            
            # update hr_vector (hr_vector shape 100x768)
            hr_vector += neighbor_vector
            # hr_vector = neighbor_vector

        # neighbor_token_ids shape [512,50,50]
        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        # hr_vector shape: (2048,768) (batch_size, 768)

        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)

        # import pdb;pdb.set_trace()

        # prediction part so simple: e^{hr} * e^{t}
        logits = hr_vector.mm(tail_vector.t())
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

        return {'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach()}

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}

class CustomVGAEBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        # if self.args.freeze_lm:
        #     for name,param in self.hr_bert.named_parameters():
        #         if name.startswith("bert"):
        #             param.requires_grad = False
        
        self.neighbor_bert = deepcopy(self.hr_bert)
        self.tail_bert = deepcopy(self.hr_bert)

        self.gcn = GCN(768, 768) # this is to encode neighbors
        # self.vgae = VGAE(VariationalGCNEncoder(768, 768)) # this is to predict edges
        self.vgae = VGAE(VariationalGATEncoder(768, 768))
       

    def _encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)
        
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)

    
        return cls_output # shape: 512x768

    
    
    
    def _encode_neighbors(self, encoder, token_ids, mask, token_type_ids, h_token_ids, h_mask, h_token_type_ids, 
                          neighbor_rel_ids=None, neighbor_rel_mask=None, neighbor_rel_type_ids=None # new
                          ):
        # this is to encode neighbors
        # neighbor_token_ids shape: [512, 50, 50] -> [batch_size, neighbor_counts, mx_len]
        n_neighbors = token_ids.shape[1]
        
        h_token_ids = h_token_ids.unsqueeze(1)
        h_mask = h_mask.unsqueeze(1)
        h_token_type_ids = h_token_type_ids.unsqueeze(1)
        
        
        # make parallel
        neighbor_token_ids = torch.cat((h_token_ids,token_ids),1)
        neighbor_mask = torch.cat((h_mask,mask),1)
        neighbor_token_type_ids = torch.cat((h_token_type_ids,token_type_ids),1)
        
        
        max_token=neighbor_token_ids.shape[-1]
        batch_size = neighbor_token_ids.shape[0]
        max_neighbor = neighbor_token_ids.shape[1]
        bert_output = encoder(input_ids=neighbor_token_ids.reshape(-1,max_token),attention_mask=neighbor_mask.reshape(-1,max_token),token_type_ids=neighbor_token_type_ids.reshape(-1,max_token),return_dict=True)
        bert_output = bert_output.last_hidden_state
        node_embedding = bert_output[:, 0, :].reshape(batch_size,max_neighbor,-1) # torch.Size([100, 6, 768]) -> (batch_size, max_neighbor, dim)
        
        
        #build star graph
        generated_adj = nx.star_graph(n_neighbors)
        nx_adj = from_networkx(generated_adj)['edge_index'].to(self.device)
        # build batch for GCN
        graph_data_list = [Data(x=node_features,edge_index=nx_adj) for node_features in node_embedding]
        
        
        my_loader = DataLoader(graph_data_list, batch_size=batch_size, shuffle=False)
        batch=next(iter(my_loader))
        
        # uncomment for others
        # neighbor_output = self.gcn(batch.x,batch.edge_index) # (600,768)
        
        # uncomment for vgae
        # mu, neighbor_output = self.gcn(batch.x,batch.edge_index) # mu shape (600,hidden_dim=100) and output shape(600,768)
        # NOTE: random sample a few positive relations
        
        _,dim1=batch.edge_index.shape
        sample_dim = int(dim1*0.9)
        vgae_sample_indice = random.sample(range(dim1), sample_dim)
        vgae_sample_indice = torch.tensor(vgae_sample_indice).to(self.device)
        # sampled_values = batch.edge_index[vgae_sample_indice]
        sampled_dim0 = batch.edge_index[0][vgae_sample_indice]
        sampled_dim1 = batch.edge_index[1][vgae_sample_indice]
        sampled_edges = torch.stack([sampled_dim0,sampled_dim1],0)
        
        neighbor_output = self.gcn(batch.x,batch.edge_index) # output shape(600,768)
        # compute reconstruction
        # pdb.set_trace()
        
        # vgae original 
        neighbor_output = self.gcn(batch.x,batch.edge_index) # output shape(600,768)
        neighbor_output_for_edge = self.vgae.encode(batch.x,batch.edge_index)
        recon_loss = self.vgae.recon_loss(neighbor_output_for_edge, sampled_edges)
        n_nodes = batch.x.shape[0]
        kl_loss = (1 / n_nodes) * self.vgae.kl_loss()
        vgae_loss = recon_loss + kl_loss
        
        
        # pdb.set_trace()
        # recon_loss = model.recon_loss(z, train_data.pos_edge_label_index)
        neighbor_output = neighbor_output.reshape(batch_size,max_neighbor,-1).sum(1) # add among the neighbor dimension
        
        # region
        # for sample_token_ids, sample_mask, sample_token_type_ids, head_token_ids, head_mask, head_token_type_ids in zip(token_ids, mask, token_type_ids,hr_token_ids, hr_mask, hr_token_type_ids):     
            
        #     token_ids = torch.cat((head_token_ids,sample_token_ids),0)
        #     mask = torch.cat((head_mask,sample_mask),0)
        #     token_type_ids = torch.cat((head_token_type_ids,sample_token_type_ids),0)
            
        #     # pdb.set_trace()
        #     # BERT asks for [batch_size, seq_len] shape as input
        #     outputs = encoder(input_ids=token_ids,attention_mask=mask,token_type_ids=token_type_ids,return_dict=True)
        #     last_hidden_state = outputs.last_hidden_state
        #     node_embedding = last_hidden_state[:, 0, :] # [51, 768]

        #     # generate star graph
        #     generated_adj = nx.star_graph(n_neighbors)
        #     nx_adj = from_networkx(generated_adj)['edge_index'].to(self.device)

        #     graph_output = self.gcn(node_embedding,nx_adj) # 51, 768
        #     neighbor_output.append(graph_output)
        # endregion

        return neighbor_output,vgae_loss # shape: 512x768 [batch_size, 768]

 
        

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                neighbor_token_ids, neighbor_mask, neighbor_token_type_ids, # new
                neighbor_rel_ids=None, neighbor_rel_mask=None, neighbor_rel_type_ids=None, # new
                only_ent_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids)
        
        # hr_token_ids[512,50]
        # set device
        self.device=hr_token_ids.device

        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids)
        
        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)

        head_vector = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)
        
        vgae_loss = 0
        if neighbor_token_ids is not None:
         
            
            neighbor_vector, vgae_loss = self._encode_neighbors(self.neighbor_bert,
                                    token_ids=neighbor_token_ids,
                                    mask=neighbor_mask,
                                    token_type_ids=neighbor_token_type_ids, 
                                    h_token_ids=head_token_ids, 
                                    h_mask=head_mask, 
                                    h_token_type_ids=head_token_type_ids, 
                                    neighbor_rel_ids=neighbor_rel_ids, #new
                                    neighbor_rel_mask=neighbor_rel_mask,#new
                                    neighbor_rel_type_ids=neighbor_rel_type_ids,#new
                                    )
            
            # update hr_vector (hr_vector shape 100x768)
            hr_vector += neighbor_vector
            # hr_vector = neighbor_vector

        # neighbor_token_ids shape [512,50,50]
        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector,
                'vgae_loss':vgae_loss}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        # hr_vector shape: (2048,768) (batch_size, 768)

        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)

        # import pdb;pdb.set_trace()

        # prediction part so simple: e^{hr} * e^{t}
        logits = hr_vector.mm(tail_vector.t())
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

        
        
        
        return {'logits': logits, # logits shape (100, 301)
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach(),
                'vgae_loss':output_dict['vgae_loss']}

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}
    
def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector
