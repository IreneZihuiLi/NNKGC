import pdb
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

from math import ceil
from torch_geometric.nn import dense_diff_pool
from torch_geometric.nn import DenseGCNConv



# from torch_geometric.nn.models.autoencoder import VGAE

class StandardGCN(torch.nn.Module):
    def __init__(self,num_in_features,num_out_features, hidden=300):
        super().__init__()
        self.conv1 = GCNConv(num_in_features, hidden)
        self.conv2 = GCNConv(hidden, num_out_features)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        # return F.log_softmax(x, dim=1)
        # return x # shape [51, 768]
        return x

class GCN(torch.nn.Module):
    def __init__(self,num_in_features,num_out_features):
        super().__init__()
        
        self.conv = GCNConv(num_in_features, num_out_features)

    def forward(self, x, edge_index):

        x = self.conv(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return x
    
class GCNRel(torch.nn.Module):
    def __init__(self,num_in_features,num_out_features):
        super().__init__()
        
        self.conv = GCNConv(num_in_features, num_out_features)

    def forward(self, x, edge_index):
        
        x = self.conv(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return x



class GAT(torch.nn.Module):
    def __init__(self,input_dim,output_dim, hidden_dim=150):
        super().__init__()
        # self.conv1 = GATConv(input_dim, hidden_dim, heads=2)
        # self.conv2 = GATConv(hidden_dim * 2, output_dim,heads=1)
        
        self.conv1 = GATConv(in_channels=input_dim, out_channels=768, heads=3, concat=False)
        self.conv2 = GATConv(in_channels=input_dim, out_channels=768, heads=3, concat=False)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        output = F.dropout(x, training=self.training)
        output = self.conv2(output, edge_index)
        return output
        # return F.log_softmax(output, dim=1)  # shape: [num_node/x.shape[0], output_dim/num_class]

# add classic methods
class GraphSAGE(torch.nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim=32):
        super().__init__()
        self.conv1 = SAGEConv(input_dim,hidden_dim)
        self.conv2 = SAGEConv(hidden_dim,output_dim)

    def forward(self,x,edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x
        # return F.log_softmax(x, dim=1)
        
# 20221117: add VGAE encoder
class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, output_dim)
        self.conv_mu = GCNConv(output_dim, output_dim)
        self.conv_logstd = GCNConv(output_dim, output_dim)

    def forward(self, x, edge_index):
        
        # pdb.set_trace()
        x = self.conv1(x, edge_index).relu() # x shape torch.Size([600, 2*hidden_dim])
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index) # return shape (600,100) and (600,768)


class VariationalGATEncoder(torch.nn.Module):
    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.conv1 = GATConv(in_channels=input_dim, out_channels=768, heads=3, concat=False)
        self.conv_mu = GATConv(in_channels=768, out_channels=output_dim, heads=3,concat=False)
        self.conv_logstd = GATConv(in_channels=768, out_channels=output_dim, heads=3,concat=False)

    def forward(self, x, edge_index):
        
        # pdb.set_trace()
        x = self.conv1(x, edge_index).relu() # x shape torch.Size([600, 2*hidden_dim])
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index) # return shape (600,100) and (600,768)

