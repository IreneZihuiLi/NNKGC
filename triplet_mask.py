import torch

from typing import List

from config import args
from dict_hub import get_train_triplet_dict, get_entity_dict, EntityDict, TripletDict

entity_dict: EntityDict = get_entity_dict()
train_triplet_dict: TripletDict = get_train_triplet_dict() if not args.is_test else None


def construct_mask(row_exs: List, col_exs: List = None) -> torch.tensor:
    '''
    mask out repeated tail ids, and neighbors
    '''
    positive_on_diagonal = col_exs is None # true during training: col_exs is None
    num_row = len(row_exs) # 300
    col_exs = row_exs if col_exs is None else col_exs # a list of objects, during training, this is a copy of row_exs
    num_col = len(col_exs) # 300
    
    # exact match
    row_entity_ids = torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in row_exs]) # tail id
    col_entity_ids = row_entity_ids if positive_on_diagonal else torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in col_exs])
    # it looks like row_entity_ids and col_entity_ids are the same (yes, during training)
        
    
    # num_row x num_col != element-wise comparison
    triplet_mask = (row_entity_ids.unsqueeze(1) != col_entity_ids.unsqueeze(0))
    if positive_on_diagonal: # true during training, this makes the same element as False
        triplet_mask.fill_diagonal_(True)
    
    # mask out other possible neighbors
    for i in range(num_row):
        head_id, relation = row_exs[i].head_id, row_exs[i].relation
        neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation)
        # exact match is enough, no further check needed
        if len(neighbor_ids) <= 1:
            continue

        for j in range(num_col):
            if i == j and positive_on_diagonal:
                continue
            tail_id = col_exs[j].tail_id
            if tail_id in neighbor_ids:
                triplet_mask[i][j] = False
    # import pdb;pdb.set_trace()
    return triplet_mask


def construct_self_negative_mask(exs: List) -> torch.tensor:
    mask = torch.ones(len(exs))
    for idx, ex in enumerate(exs):
        
        head_id, relation = ex.head_id, ex.relation # head id in string, and relation in string
        neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation) # a list of neighbor ids in string
        if head_id in neighbor_ids: # remove head ids from neighbors
            mask[idx] = 0
    
    return mask.bool() # size [300], bool
