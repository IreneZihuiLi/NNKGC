import os
import json
import pdb
import random
import torch
import torch.utils.data.dataset

from typing import Optional, List
import pdb
from config import args
from triplet import reverse_triplet
from triplet_mask import construct_mask, construct_self_negative_mask
from dict_hub import get_entity_dict, get_link_graph, get_tokenizer
from logger_config import logger
import pprint

import time
# vectorize function applies linkgraph, defined in triplet.py


entity_dict = get_entity_dict()
if args.use_link_graph:
    # make the lazy data loading happen
    get_link_graph()


def _custom_tokenize(text: str,
                     text_pair: Optional[str] = None) -> dict:
    tokenizer = get_tokenizer()
    encoded_inputs = tokenizer(text=text,
                               text_pair=text_pair if text_pair else None,
                               add_special_tokens=True,
                               max_length=args.max_num_tokens,
                               return_token_type_ids=True,
                               truncation=True)
    # applying PLMs to encode a given text
    # return is a dict {'input_ids', 'token_type_ids','attention_mask'}
    
    return encoded_inputs

def _custom_list_tokenize(text: list,text_pair: Optional[str] = None) -> dict:
    tokenizer = get_tokenizer()
    # if empty, then we set special tokens (this means the node has no neighbors)
    # we tokenize in batches
    if len(text) < 1:
        text = ['[CLS]']
        text_pair = None
    encoded_inputs = tokenizer(text=text,
                               text_pair=text_pair if text_pair else None,
                            add_special_tokens=True,
                            max_length=args.max_num_tokens,
                            return_token_type_ids=True,
                            truncation=True)
    # encode neighbor info as a list of token ids, results: {'input_ids': [[101, 3247, 9338, 1024, 2619, ...
    return encoded_inputs

def _custom_list_tokenize_entity_rel(text: list,rel_text: Optional[str] = None) -> dict:
    tokenizer = get_tokenizer()
    # if empty, then we set special tokens (this means the node has no neighbors)
    # we tokenize in batches
    if len(text) < 1:
        text = ['[CLS]']
        rel_text = ['[CLS]']
        
    entity_inputs = tokenizer(text=text,
                            add_special_tokens=True,
                            max_length=args.max_num_tokens,
                            return_token_type_ids=True,
                            truncation=True)
    relation_inputs = tokenizer(text=rel_text,
                            add_special_tokens=True,
                            max_length=args.max_num_tokens,
                            return_token_type_ids=True,
                            truncation=True)
    # encode neighbor info as a list of token ids, results: {'input_ids': [[101, 3247, 9338, 1024, 2619, ...
    return (entity_inputs,relation_inputs)


def _parse_entity_name(entity: str) -> str:
    if args.task.lower() == 'wn18rr':
        # family_alcidae_NN_1
        entity = ' '.join(entity.split('_')[:-2])
        return entity
    # a very small fraction of entities in wiki5m do not have name

    return entity or ''


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    # for wn18rr data, check the wordnet-mlj12-definitions.txt file, it has
    # the definition for each relations. 
    # encode (relation name, relation description)
    
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    
    return entity


def get_neighbor_desc(head_id: str, tail_id: str = None) -> str:
    # original version: this only returns neighbor names
    neighbor_ids = get_link_graph().get_neighbor_ids(head_id)
    # avoid label leakage during training
    if not args.is_test:
        neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]
    entities = [entity_dict.get_entity_by_id(n_id).entity for n_id in neighbor_ids]
    entities = [_parse_entity_name(entity) for entity in entities]
    # import pdb;pdb.set_trace()
    return ' '.join(entities)

def get_neighbor_info(head_id: str, tail_id: str = None, n_hop=args.n_hop_graph) -> str:
    '''
    This is model 2
    This is to fetch neighbor info: names and descriptions, return a list of string
    '''
    # neighbor_ids = get_link_graph().get_neighbor_ids(head_id) # this is a list of string (neighbor ids)
    
    neighbor_ids_and_rels = get_link_graph().get_neighbor_ids_and_rels(head_id)[:args.n_neighbor] # this is a list of tuple (id,rel_in_string), i.e., ('/m/04ztj', 'location of c..')
    
    # print (neighbor_ids)
    # print (head_id)
    if n_hop ==2:
        # TODO: fix the relation part, using get_neighbor_ids_and_rels()
        # if two-hop, we iterate into neighbors
        # neighbor_set=set()
        # for n_id in neighbor_ids:
        #     neighbor_set.update(get_link_graph().get_neighbor_ids(n_id))
        # neighbor_set.update(neighbor_ids)
        # neighbor_ids = list(neighbor_set)
        neighbor_ids = [x[0] for x in neighbor_ids_and_rels]
        neighbor_rels = [x[1] for x in neighbor_ids_and_rels]   
        
        # TODO: Two many neighbors
        for n_id in neighbor_ids:
            # neighbor_ids_and_rels_2hop = get_link_graph().get_neighbor_ids_and_rels(n_id)[:2]
            neighbor_ids_and_rels_2hop = get_link_graph().get_neighbor_ids_and_rels_with_condition(n_id,neighbor_rels,max_to_keep=10)
            neighbor_ids_and_rels += neighbor_ids_and_rels_2hop
            # import pdb;pdb.set_trace()
            # 20221117 start: only keep same rels
            # neighbor_ids_2hop = [x[0] for x in neighbor_ids_and_rels_2hop]
            # neighbor_rels_2hop = [x[1] for x in neighbor_ids_and_rels_2hop]
            
            # keep_count = len(neighbor_ids_2hop)
            # total_count = len(neighbor_ids_and_rels_2hop)
            # if keep_count>0 and total_count>0:
            #     ratio = float(keep_count)/float(total_count)
                # print (ratio)
                # print (head_id)
            # neighbor_ids+=neighbor_ids_2hop
            # neighbor_rels+=neighbor_rels_2hop
            
            # 20221117 ends
            
            # before change, 20221117
            # neighbor_ids_2hop = [x[0] for x in neighbor_ids_and_rels_2hop]
            # neighbor_rels_2hop = [x[1] for x in neighbor_ids_and_rels_2hop]
            # neighbor_ids+=neighbor_ids_2hop
            # neighbor_rels+=neighbor_rels_2hop
            # before change ends
    # pdb.set_trace()    
            
    neighbor_ids = [x[0] for x in neighbor_ids_and_rels]
    neighbor_rels = [x[1] for x in neighbor_ids_and_rels]  

        
    # avoid label leakage during training
    if not args.is_test:
        # if testing: This is to remove the tail_id from the neighbors
        neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]
        neighbor_rels = [tuple[1] for tuple in neighbor_ids_and_rels if tuple[0] != tail_id]
    entities = [entity_dict.get_entity_by_id(n_id).entity for n_id in neighbor_ids]
    entities = [_parse_entity_name(entity) for entity in entities]
    
    # model 2
    neighbor_info = []
    # neighbor_rels = []
    
    
    # print ('neighbor nb', len(neighbor_ids))
    for i,n_id in enumerate(neighbor_ids):
        # old content, contains entity name: entity desc
        # content = _parse_entity_name(entity_dict.get_entity_by_id(n_id).entity) +':'+entity_dict.get_entity_by_id(n_id).entity_desc 
        # print (len(neighbor_rels),len(neighbor_ids))
        # new content, contains entity name [SEP] relation 
        # print (len(neighbor_ids),len(neighbor_rels))
        # print(neighbor_rels[i])
        # print (entity_dict.get_entity_by_id(n_id).entity)
        content = _parse_entity_name(entity_dict.get_entity_by_id(n_id).entity)+' [SEP] '+ neighbor_rels[i]
        
        # optimize starts (202211): content [SEP] relation
        # content += ' [SEP] '
        # content += neighbor_rels[i]
        # end of optimize
        neighbor_info.append(content)

    # print ('****') 
    # import pdb;pdb.set_trace()
    # pprint.pprint(neighbor_info)   
     
    # return neighbor_info,neighbor_rels
    return neighbor_info,None

# region
# def get_neighbor_desc(head_id: str, tail_id: str = None) -> str:
      # this is model 1
#     # notes: the original version does not include descriptions, only names of the neighbors!
#     '''
#     head id 11575425
#     neighbors  ['11567411', '11864906', '11866078', '11867070', '11869890', '11870212', '11870607', '11871294', '11871916', '11872850']
#     entities .. ['genus capparis', 'genus cleome', 'polanisia', 'genus aethionema', 'genus alliaria', 'genus alyssum', 'genus arabidopsis', 'genus arabis', 'genus armoracia']
#     This is to remove the tail_id from the neighbors. 
#     '''
#     # print ('start..', head_id)
#     neighbor_ids = get_link_graph().get_neighbor_ids(head_id)
#     # print ('head id', head_id)
#     # print ('tail id', tail_id)
#     # print ('neighbors ',neighbor_ids)
#     # avoid label leakage during training
#     if not args.is_test:
#         # if testing: This is to remove the tail_id from the neighbors
#         neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]
#     entities = [entity_dict.get_entity_by_id(n_id).entity for n_id in neighbor_ids]
#     entities = [_parse_entity_name(entity) for entity in entities]
    
#     # model 1
#     entitiy_desc = [_parse_entity_name(entity_dict.get_entity_by_id(n_id).entity) +' '+entity_dict.get_entity_by_id(n_id).entity_desc for n_id in neighbor_ids]
    
#     # model 2: repeat number_neighbor times
#     # if len(head_id) > 0:
#     #     head_desc = _parse_entity_name(entity_dict.get_entity_by_id(head_id).entity)+':'+entity_dict.get_entity_by_id(head_id).entity_desc+';'
#     # else:
#     #     head_desc = ' '
#     # print ('Head id is ',head_id)
#     # head_desc = _parse_entity_name(entity_dict.get_entity_by_id(head_id).entity)
#     # print (head_desc)
#     # entitiy_desc = [head_desc +_parse_entity_name(entity_dict.get_entity_by_id(n_id).entity) +' '+entity_dict.get_entity_by_id(n_id).entity_desc for n_id in neighbor_ids]

#     # model 1, 2
#     return ' '.join(entitiy_desc)
# endregion


class Example:

    def __init__(self, head_id, tail_id, relation, **kwargs):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation
        
    @property
    def head_desc(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity_desc

    @property
    def tail_desc(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity_desc

    @property
    def head(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity

    @property
    def tail(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity

    #region
    # def vectorize(self) -> dict:
    #     # original version
    #     head_desc, tail_desc = self.head_desc, self.tail_desc

        
    #     if args.use_link_graph: # default to be true
    #         if len(head_desc.split()) < 20:
    #             head_desc += ' ' + get_neighbor_desc(head_id=self.head_id, tail_id=self.tail_id)
    #         if len(tail_desc.split()) < 20:
    #             tail_desc += ' ' + get_neighbor_desc(head_id=self.tail_id, tail_id=self.head_id)
 
    #     # print ('--h--',head_desc)
    #     # print ('--t--',tail_desc)
    #     # time.sleep(7)
    #     head_word = _parse_entity_name(self.head)
    #     head_text = _concat_name_desc(head_word, head_desc)

    #     # print ('--before enc--',head_text)
    #     # time.sleep(7)
    #     hr_encoded_inputs = _custom_tokenize(text=head_text,
    #                                          text_pair=self.relation)
        
    #     head_encoded_inputs = _custom_tokenize(text=head_text)

    #     tail_word = _parse_entity_name(self.tail)
    #     tail_encoded_inputs = _custom_tokenize(text=_concat_name_desc(tail_word, tail_desc))

    #     '''
    #     hr_token_ids [101, 4087, 1024, 1997, 2030, 8800, 2000, 6028, 2030, 26293, 1999, 1037, 6742, 8066, 1025, 1000, 2010, 4087, 8144, 2001, 2010, 8248, 6198, 1000, 1025, 1000, 1996, 4087, 4830, 17644, 1997, 2014, 5613, 1000, 102, 29280, 3973, 3141, 2433, 102] 
    #     hr_token_type_ids [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1] 
    #     tail_token_ids [101, 6028, 1024, 8066, 20938, 1999, 1996, 3094, 1997, 8050, 2015, 4315, 14966, 2013, 3218, 1998, 24666, 1025, 1000, 3218, 6551, 24840, 26293, 1000, 8066, 20938, 16661, 102] 
    #     tail_token_type_ids [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
    #     head_token_ids [101, 4087, 1024, 1997, 2030, 8800, 2000, 6028, 2030, 26293, 1999, 1037, 6742, 8066, 1025, 1000, 2010, 4087, 8144, 2001, 2010, 8248, 6198, 1000, 1025, 1000, 1996, 4087, 4830, 17644, 1997, 2014, 5613, 1000, 102] 
    #     head_token_type_ids [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    #     '''
    #     # print (len(hr_encoded_inputs['input_ids']))
    #     return {'hr_token_ids': hr_encoded_inputs['input_ids'],
    #             'hr_token_type_ids': hr_encoded_inputs['token_type_ids'],
    #             'tail_token_ids': tail_encoded_inputs['input_ids'],
    #             'tail_token_type_ids': tail_encoded_inputs['token_type_ids'],
    #             'head_token_ids': head_encoded_inputs['input_ids'],
    #             'head_token_type_ids': head_encoded_inputs['token_type_ids'],
    #             'obj': self}
    #endregion
                
    def vectorize(self) -> dict:
        # model 2: add a new field "neighbor_token_ids"
        head_desc, tail_desc = self.head_desc, self.tail_desc

        
        if args.use_link_graph: # default to be true (if the length is too small, concate with neighbor NAMEs)
            if len(head_desc.split()) < 20:
                head_desc += ' ' + get_neighbor_desc(head_id=self.head_id, tail_id=self.tail_id)
            if len(tail_desc.split()) < 20:
                tail_desc += ' ' + get_neighbor_desc(head_id=self.tail_id, tail_id=self.head_id)
        
        
        # print ('--h--',head_desc)
        # print ('--t--',tail_desc)
        # time.sleep(7)
        head_word = _parse_entity_name(self.head)
        head_text = _concat_name_desc(head_word, head_desc)

       
      
        # hr_encoded_inputs: contains head_name, head_desc, relation
        hr_encoded_inputs = _custom_tokenize(text=head_text,text_pair=self.relation)
        
        # head_encoded_inputs: only head_name, head_desc
        head_encoded_inputs = _custom_tokenize(text=head_text)

        
        # add new 
        neighbor_word,neighbor_rel = get_neighbor_info(head_id=self.head_id, tail_id=self.tail_id)
        neighbor_rel_encoded_inputs = None
        # add new
        # neighbor_encoded_inputs = _custom_list_tokenize(text=neighbor_word,text_pair=neighbor_rel)
        # 20221107: extract relation as seperate relation
        # neighbor_encoded_inputs, neighbor_rel_encoded_inputs = _custom_list_tokenize_entity_rel(text=neighbor_word,rel_text=neighbor_rel)
        neighbor_encoded_inputs = _custom_list_tokenize(text=neighbor_word)
       
        # print ('--before enc--',neighbor_word)
        # print (neighbor_encoded_inputs['input_ids'])
        # time.sleep(7)
        
        # tail_encoded_inputs: contains tail_name, tail_desc
        tail_word = _parse_entity_name(self.tail)
        tail_encoded_inputs = _custom_tokenize(text=_concat_name_desc(tail_word, tail_desc))

        
        
        '''
        hr_token_ids [101, 4087, 1024, 1997, 2030, 8800, 2000, 6028, 2030, 26293, 1999, 1037, 6742, 8066, 1025, 1000, 2010, 4087, 8144, 2001, 2010, 8248, 6198, 1000, 1025, 1000, 1996, 4087, 4830, 17644, 1997, 2014, 5613, 1000, 102, 29280, 3973, 3141, 2433, 102] 
        hr_token_type_ids [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1] 
        tail_token_ids [101, 6028, 1024, 8066, 20938, 1999, 1996, 3094, 1997, 8050, 2015, 4315, 14966, 2013, 3218, 1998, 24666, 1025, 1000, 3218, 6551, 24840, 26293, 1000, 8066, 20938, 16661, 102] 
        tail_token_type_ids [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
        head_token_ids [101, 4087, 1024, 1997, 2030, 8800, 2000, 6028, 2030, 26293, 1999, 1037, 6742, 8066, 1025, 1000, 2010, 4087, 8144, 2001, 2010, 8248, 6198, 1000, 1025, 1000, 1996, 4087, 4830, 17644, 1997, 2014, 5613, 1000, 102] 
        head_token_type_ids [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        '''
        # import pdb;pdb.set_trace()
        return {'hr_token_ids': hr_encoded_inputs['input_ids'], # this is a list of ids
                'hr_token_type_ids': hr_encoded_inputs['token_type_ids'], # this is a list of ids
                'tail_token_ids': tail_encoded_inputs['input_ids'], # this is a list of ids
                'tail_token_type_ids': tail_encoded_inputs['token_type_ids'], # this is a list of ids
                'head_token_ids': head_encoded_inputs['input_ids'], # this is a list of ids
                'head_token_type_ids': head_encoded_inputs['token_type_ids'], # this is a list of ids
                'neighbor_token_ids':neighbor_encoded_inputs['input_ids'], # new, this is a list of list 
                'neighbor_token_type_ids':neighbor_encoded_inputs['token_type_ids'], # new, this is a list of list 
                'neighbor_rel_ids': None if neighbor_rel_encoded_inputs is None else neighbor_rel_encoded_inputs['input_ids'], # new
                'neighbor_rel_type_ids':None  if neighbor_rel_encoded_inputs is None else neighbor_rel_encoded_inputs['token_type_ids'] , # new, this is a list of list 
                'obj': self}

# region
# class Example:
    # original Example
#     def __init__(self, head_id, relation, tail_id, **kwargs):
#         self.head_id = head_id
#         self.tail_id = tail_id
#         self.relation = relation

#     @property
#     def head_desc(self):
#         if not self.head_id:
#             return ''
#         return entity_dict.get_entity_by_id(self.head_id).entity_desc

#     @property
#     def tail_desc(self):
#         return entity_dict.get_entity_by_id(self.tail_id).entity_desc

#     @property
#     def head(self):
#         if not self.head_id:
#             return ''
#         return entity_dict.get_entity_by_id(self.head_id).entity

#     @property
#     def tail(self):
#         return entity_dict.get_entity_by_id(self.tail_id).entity

#     def vectorize(self) -> dict:
#         head_desc, tail_desc = self.head_desc, self.tail_desc

        
#         if args.use_link_graph:
#             if len(head_desc.split()) < 20:
#                 head_desc += ' ' + get_neighbor_desc(head_id=self.head_id, tail_id=self.tail_id)
#             if len(tail_desc.split()) < 20:
#                 tail_desc += ' ' + get_neighbor_desc(head_id=self.tail_id, tail_id=self.head_id)
        
        
#         head_word = _parse_entity_name(self.head)
#         head_text = _concat_name_desc(head_word, head_desc)
#         hr_encoded_inputs = _custom_tokenize(text=head_text,
#                                              text_pair=self.relation)
        
#         head_encoded_inputs = _custom_tokenize(text=head_text)

#         tail_word = _parse_entity_name(self.tail)
#         tail_encoded_inputs = _custom_tokenize(text=_concat_name_desc(tail_word, tail_desc))

#         '''
#         hr_token_ids [101, 4087, 1024, 1997, 2030, 8800, 2000, 6028, 2030, 26293, 1999, 1037, 6742, 8066, 1025, 1000, 2010, 4087, 8144, 2001, 2010, 8248, 6198, 1000, 1025, 1000, 1996, 4087, 4830, 17644, 1997, 2014, 5613, 1000, 102, 29280, 3973, 3141, 2433, 102] 
#         hr_token_type_ids [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1] 
#         tail_token_ids [101, 6028, 1024, 8066, 20938, 1999, 1996, 3094, 1997, 8050, 2015, 4315, 14966, 2013, 3218, 1998, 24666, 1025, 1000, 3218, 6551, 24840, 26293, 1000, 8066, 20938, 16661, 102] 
#         tail_token_type_ids [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
#         head_token_ids [101, 4087, 1024, 1997, 2030, 8800, 2000, 6028, 2030, 26293, 1999, 1037, 6742, 8066, 1025, 1000, 2010, 4087, 8144, 2001, 2010, 8248, 6198, 1000, 1025, 1000, 1996, 4087, 4830, 17644, 1997, 2014, 5613, 1000, 102] 
#         head_token_type_ids [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#         '''
#         # print (len(hr_encoded_inputs['input_ids']))
#         return {'hr_token_ids': hr_encoded_inputs['input_ids'],
#                 'hr_token_type_ids': hr_encoded_inputs['token_type_ids'],
#                 'tail_token_ids': tail_encoded_inputs['input_ids'],
#                 'tail_token_type_ids': tail_encoded_inputs['token_type_ids'],
#                 'head_token_ids': head_encoded_inputs['input_ids'],
#                 'head_token_type_ids': head_encoded_inputs['token_type_ids'],
#                 'obj': self}
# endregion

class Dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, path, task, more_path=None, examples=None):
        self.path_list = path.split(',')
        self.task = task
        assert all(os.path.exists(path) for path in self.path_list) or examples
        if examples:
            self.examples = examples
        else:
            self.examples = []
            
            for path in self.path_list:
                # import pdb;pdb.set_trace()
                # print ('more_path',more_path)
                if not self.examples:
                    if more_path is '' or more_path is None:
                        
                        self.examples = load_data(path)
                    else:
                        self.examples = load_data_with_entity_extraction(path,more_path)
                        
                else:
                    if more_path is '' or more_path is None:
                        self.examples.extend(load_data(path))
                    else:
                        self.examples.extend(load_data_with_entity_extraction(path,more_path))
                
              
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index].vectorize()


def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = True) -> List[Example]:
    # load all triplets from the txt or json file, only read head tail and relation
    # /home/irene/Projects/2022Fall/simkgc_data/data/WN18RR/train.txt.json 
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_triplet or add_backward_triplet
    logger.info('In test mode: {}'.format(args.is_test))

    
    # loading from txt
    data = json.load(open(path, 'r', encoding='utf-8'))
    logger.info('Load {} examples from {}'.format(len(data), path))

    cnt = len(data)
    examples = []

    # packup as Example object
    for i in range(cnt):
        obj = data[i]
        if add_forward_triplet:
            examples.append(Example(**obj))
        if add_backward_triplet:
            # triplet.py search for reverse_triplet()
            examples.append(Example(**reverse_triplet(obj)))
        data[i] = None
    # vars(examples[0]): {'head_id': '00260881', 'tail_id': '00260622', 'relation': 'hypernym'}
    

    return examples


def load_data_with_entity_extraction(path: str,more_path:str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = True) -> List[Example]:
    # load all triplets from the txt.json, as well as we extracted relations, but we only build simple connection from the node name only, ignoring long text descriptions
    # /home/irene/Projects/2022Fall/simkgc_data/data/WN18RR/train.txt.json and our generated v2.extraction.desc.neg.pos.pt.json
    # the two data has different headers. 
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_triplet or add_backward_triplet
    logger.info('In test mode: {}'.format(args.is_test))

    
    # loading from txt
    

    data = json.load(open(path, 'r', encoding='utf-8'))
    logger.info('Load {} examples from {}'.format(len(data), path))

    cnt = len(data)
    examples = []

    # packup as Example object
    for i in range(cnt):
        obj = data[i]
        if add_forward_triplet:
            examples.append(Example(**obj))
        if add_backward_triplet:
            # triplet.py search for reverse_triplet()
            examples.append(Example(**reverse_triplet(obj)))
        data[i] = None
    # vars(examples[0]): {'head_id': '00260881', 'tail_id': '00260622', 'relation': 'hypernym'}
    
    
    # loading extracted relations 'v{123}.json'
    '''
    every data:
    >> /home/irene/Projects/2022Fall/simkgc_data/data/WN18RR/v2.extraction.desc.neg.pos.pt.json
    {'head_id': '05998052', 'head': 'frontier_NN_3', 'head_desc': 'an undeveloped field of study; a topic inviting research and development; "he worked at the frontier of brain science"', 'tail_id': '05996646', 'tail': 'subject_field_NN_1', 'tail_desc': 'a branch of knowledge; "in what discipline is his doctorate?"; "teachers should be well trained in their subject"; "anthropology is the study of human beings"', 'label': 1, 'relation': 'co-occurr'}
    '''
    
    data_extract = json.load(open(more_path, 'r', encoding='utf-8'))
    logger.info('Load {} examples from {}'.format(len(data_extract), more_path))
    cnt_extract = len(data_extract)
    examples_extract = []
    # import pdb;pdb.set_trace()
    # combine?
    for i in range(cnt_extract):
        obj = data_extract[i]
        if add_forward_triplet:
            examples_extract.append(Example(**obj))
        if add_backward_triplet:
            # triplet.py search for reverse_triplet()
            examples_extract.append(Example(**reverse_triplet(obj)))
        data_extract[i] = None
    
    return examples+examples_extract

def collate(batch_data: List[dict]) -> dict:

    
    hr_token_ids, hr_mask, mx_len = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    hr_token_type_ids, mx_len = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_type_ids']) for ex in batch_data],
        need_mask=False)

    tail_token_ids, tail_mask, mx_len = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    tail_token_type_ids, mx_len = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_type_ids']) for ex in batch_data],
        need_mask=False)


    
    head_token_ids, head_mask, mx_len = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    head_token_type_ids, mx_len = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_type_ids']) for ex in batch_data],
        need_mask=False)

    
  
    # 10252022
    if not args.is_test:
        # add new: process neighbor ids
        neighbor_token_ids, neighbor_mask = to_indices_and_mask_3d(
            batch_data, var_name='neighbor_token_ids', mx_len=mx_len,
            pad_token_id=get_tokenizer().pad_token_id,mx_neighbor=args.n_neighbor)
        neighbor_token_type_ids = to_indices_and_mask_3d(
            batch_data, var_name='neighbor_token_type_ids', mx_len=mx_len,
            need_mask=False,mx_neighbor=args.n_neighbor)
        
        # pdb.set_trace()
        # 20221107 new: add relation ids
        # neighbor_rel_ids, neighbor_rel_mask = to_indices_and_mask_3d(
        #     batch_data, var_name='neighbor_rel_ids', mx_len=mx_len,
        #     pad_token_id=get_tokenizer().pad_token_id,mx_neighbor=args.n_neighbor)
        # # pdb.set_trace()
        # neighbor_rel_type_ids = to_indices_and_mask_3d(
        #     batch_data, var_name='neighbor_rel_type_ids', mx_len=mx_len,
        #     need_mask=False,mx_neighbor=args.n_neighbor)
        
        
    # region
    # pdb.set_trace()

    # (Pdb) neighbor_token_ids.shape
    # torch.Size([512, 50, 50])
    # (Pdb) neighbor_mask.shape
    # torch.Size([512, 50, 50])
    
    
    # print(head_token_ids.shape)
    # print(head_mask.shape)
    # print(head_token_type_ids.shape)
    # torch.Size([512, 50])
    # torch.Size([512, 50])
    # torch.Size([512, 50])
    # endregion

    
    batch_exs = [ex['obj'] for ex in batch_data] # 300 objects
    batch_dict = {
        'hr_token_ids': hr_token_ids,
        'hr_mask': hr_mask,
        'hr_token_type_ids': hr_token_type_ids,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'head_token_ids': head_token_ids,
        'head_mask': head_mask,
        'head_token_type_ids': head_token_type_ids,
        'neighbor_token_ids':neighbor_token_ids if not args.is_test else None, # new
        'neighbor_mask':neighbor_mask if not args.is_test else None, # new
        'neighbor_token_type_ids':neighbor_token_type_ids if not args.is_test else None, # new
        'batch_data': batch_exs,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
        # 'neighbor_rel_ids':neighbor_rel_ids if not args.is_test else None, # new
        # 'neighbor_rel_mask':neighbor_rel_mask if not args.is_test else None, # new
        # 'neighbor_rel_type_ids':neighbor_rel_type_ids if not args.is_test else None, # new
    }

    return batch_dict

def collate_orgi(batch_data: List[dict]) -> dict:

    
    hr_token_ids, hr_mask, mx_len = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    hr_token_type_ids, mx_len = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_type_ids']) for ex in batch_data],
        need_mask=False)

    tail_token_ids, tail_mask, mx_len = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    tail_token_type_ids, mx_len = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_type_ids']) for ex in batch_data],
        need_mask=False)



    
    head_token_ids, head_mask, mx_len = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    head_token_type_ids, mx_len = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_type_ids']) for ex in batch_data],
        need_mask=False)


  
    
    if not args.is_test:
        # add new: process neighbor ids
        neighbor_token_ids, neighbor_mask = to_indices_and_mask_3d(
            batch_data, var_name='neighbor_token_ids', mx_len=mx_len,
            pad_token_id=get_tokenizer().pad_token_id,mx_neighbor=args.n_neighbor)
        neighbor_token_type_ids = to_indices_and_mask_3d(
            batch_data, var_name='neighbor_token_type_ids', mx_len=mx_len,
            need_mask=False,mx_neighbor=args.n_neighbor)
        
        
    # region
    # pdb.set_trace()

    # (Pdb) neighbor_token_ids.shape
    # torch.Size([512, 50, 50])
    # (Pdb) neighbor_mask.shape
    # torch.Size([512, 50, 50])
    
    
    # print(head_token_ids.shape)
    # print(head_mask.shape)
    # print(head_token_type_ids.shape)
    # torch.Size([512, 50])
    # torch.Size([512, 50])
    # torch.Size([512, 50])
    # endregion


    
    batch_exs = [ex['obj'] for ex in batch_data]
    batch_dict = {
        'hr_token_ids': hr_token_ids,
        'hr_mask': hr_mask,
        'hr_token_type_ids': hr_token_type_ids,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'head_token_ids': head_token_ids,
        'head_mask': head_mask,
        'head_token_type_ids': head_token_type_ids,
        'neighbor_token_ids':neighbor_token_ids if not args.is_test else None, # new
        'neighbor_mask':neighbor_mask if not args.is_test else None, # new
        'neighbor_token_type_ids':neighbor_token_type_ids if not args.is_test else None, # new
        'batch_data': batch_exs,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
    }

    return batch_dict


def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor]) # 50
    batch_size = len(batch_tensor) # 512
    indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id) # [512, 50]
    # For BERT, mask value of 1 corresponds to a valid position

    # pdb.set_trace()
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(0)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(1)
    if need_mask:
        # both shape: (batch_size, max_len)
        return indices, mask, mx_len
    else:
        return indices, mx_len


def to_indices_and_mask_3d(batch_data, var_name='neighbor_token_ids', pad_token_id=0, need_mask=True, mx_len=1,mx_neighbor=5):
    # this is to deal with neighbors, return is a 3D tensor
    # TODO: max_neighbor is fixed
    batch_size = len(batch_data) # okay
    
    # for batch in batch_data:
    #     # find max neighbor
    #     neighbor_count = max([len(x) for x in batch[var_name]])
    #     mx_neighbor=max(mx_neighbor,neighbor_count)


    # pad each neighbor to be mx_len
    # batch_tensor = []
    batch_indices = []
    batch_mask=[]



    for batch in batch_data:
        each_tensor = [torch.LongTensor(x) for x in batch[var_name]] # a list of neighbor tokens [n_neighbor]
        
        # pad 
        each_indices = torch.LongTensor(mx_neighbor, mx_len).fill_(pad_token_id) # [n_neighbor, mx_sequence_len]
        each_mask = torch.ByteTensor(mx_neighbor, mx_len).fill_(0) # [n_neighbor, mx_sequence_len]
        
        # pdb.set_trace()
        for i, t in enumerate(each_tensor):
            # print (i, each_indices.shape[0])
            if i < each_indices.shape[0]:
          
                each_indices[i, :len(t)].copy_(t)
                if need_mask:
                    each_mask[i, :len(t)].fill_(1)
            
        
        batch_indices.append(each_indices)
        batch_mask.append(each_mask)      
        # indices and mask: both shape [mx_neighbor, mx_len] i.e., [9,50]
    
    

    indices = torch.stack(batch_indices)
    mask = torch.stack(batch_mask)
    
    
    # For BERT, mask value of 1 corresponds to a valid position
    if need_mask:
        # return shape [batch_size, mx_neighbor, mx_len], i.e., [512,50,50]
        return indices, mask
    else:
        return indices
