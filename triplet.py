import os
import json

from typing import List
from dataclasses import dataclass
from collections import deque

from logger_config import logger
import random,pdb

@dataclass
class EntityExample:
    entity_id: str
    entity: str
    entity_desc: str = ''


class TripletDict:

    def __init__(self, path_list: List[str]):
        self.path_list = path_list
        logger.info('Triplets path: {}'.format(self.path_list))
        self.relations = set()
        self.hr2tails = {}
        self.triplet_cnt = 0

        for path in self.path_list:
            self._load(path)
        logger.info('Triplet statistics: {} relations, {} triplets'.format(len(self.relations), self.triplet_cnt))

    def _load(self, path: str):
        # train.txt.json
        # ex: {'head_id': '00260881', 'head': 'land_reform_NN_1', 'relation': 'hypernym', 'tail_id': '00260622', 'tail': 'reform_NN_1'}
        print ('openning...',path)
        examples = json.load(open(path, 'r', encoding='utf-8'))
        examples += [reverse_triplet(obj) for obj in examples]
        for ex in examples:
            # import pdb;pdb.set_trace()
            self.relations.add(ex['relation'])
            key = (ex['head_id'], ex['relation'])
            if key not in self.hr2tails:
                self.hr2tails[key] = set()
            self.hr2tails[key].add(ex['tail_id'])

            # self.hr2tails: <tuple(head_id,relation_name) :set(tail_id)> i.e., {('00260881', 'hypernym'): {'00260622'}}
        self.triplet_cnt = len(examples) # counts

    def get_neighbors(self, h: str, r: str) -> set:
        return self.hr2tails.get((h, r), set())


class EntityDict:

    def __init__(self, entity_dict_dir: str, inductive_test_path: str = None):
        path = os.path.join(entity_dict_dir, 'entities.json')
        assert os.path.exists(path)
        self.entity_exs = [EntityExample(**obj) for obj in json.load(open(path, 'r', encoding='utf-8'))]
        # self.entity_exs is a list
        # for each element: EntityExample(entity_id='00260881', entity='land_reform_NN_1', entity_desc='a redistribution of agricultural land (especially by government action)')
        if inductive_test_path:
            
            examples = json.load(open(inductive_test_path, 'r', encoding='utf-8'))
            valid_entity_ids = set()
            for ex in examples:
                valid_entity_ids.add(ex['head_id'])
                valid_entity_ids.add(ex['tail_id'])
            self.entity_exs = [ex for ex in self.entity_exs if ex.entity_id in valid_entity_ids]
        # import pdb;pdb.set_trace()
        self.id2entity = {ex.entity_id: ex for ex in self.entity_exs} # {id, EntityExample object} id is in string
        self.entity2idx = {ex.entity_id: i for i, ex in enumerate(self.entity_exs)} # {id_from_0, data_original_id_string}
        logger.info('Load {} entities from {}'.format(len(self.id2entity), path))

    def entity_to_idx(self, entity_id: str) -> int:
        return self.entity2idx[entity_id]

    def get_entity_by_id(self, entity_id: str) -> EntityExample:
        return self.id2entity[entity_id]

    def get_entity_by_idx(self, idx: int) -> EntityExample:
        return self.entity_exs[idx]

    def __len__(self):
        return len(self.entity_exs)


class LinkGraph:

    def __init__(self, train_path: str):
        logger.info('Start to build link graph from {}'.format(train_path))
        # id -> set(id)

        # this is to load from train.txt.json
        # each ex is {'head_id': '00260881', 'head': 'land_reform_NN_1', 'relation': 'hypernym', 'tail_id': '00260622', 'tail': 'reform_NN_1'}
        # this is an undirected graph: {head:t, t: head}, each node has a set that shows its neighbors
        # TODO: the relation is lost...
        self.graph = {} # <tail:[a list of heads]>
        self.relgraph={} # <tail:[a list of (head,relation)>
        
        self.hrelgraph={} # 20221117: <head+"|"+relation]: tail>
        examples = json.load(open(train_path, 'r', encoding='utf-8'))
        for ex in examples:
            head_id, tail_id = ex['head_id'], ex['tail_id']
            if head_id not in self.graph:
                self.graph[head_id] = set()
            self.graph[head_id].add(tail_id)
            if tail_id not in self.graph:
                self.graph[tail_id] = set()
            self.graph[tail_id].add(head_id)
            
            # we also load relations too
            if head_id not in self.relgraph:
                self.relgraph[head_id] = set()
            self.graph[head_id].add(tail_id)
            if tail_id not in self.relgraph:
                self.relgraph[tail_id] = set()
            self.relgraph[tail_id].add((head_id,ex['relation']))
            # i.e., ('/m/027rn','form of government country location ')
            

            # build an <h,r> graph
            # TODO: add tail rel? 
            hr=head_id+'|'+ex['relation']
            if hr not in self.hrelgraph:
                self.hrelgraph[hr]= set()
            self.hrelgraph[hr].add((tail_id,ex['relation']))    
            # <headid|relation: (tail,relation)>
        
        logger.info('Done build link graph with {} nodes'.format(len(self.graph)))
        

    def get_neighbor_ids(self, entity_id: str, max_to_keep=10) -> List[str]:
        # make sure different calls return the same results
        neighbor_ids = list(self.graph.get(entity_id, set()))
        random.shuffle(neighbor_ids)
        return neighbor_ids[:max_to_keep]
    
    def get_neighbor_ids_original(self, entity_id: str, max_to_keep=10) -> List[str]:
        # make sure different calls return the same results
        # note that here this is fixed
        neighbor_ids = self.graph.get(entity_id, set())
        return sorted(list(neighbor_ids))[:max_to_keep]

    def get_neighbor_ids_and_rels(self, entity_id: str, max_to_keep=10) -> List[str]:
        # return neighbor ids, as well as the corresponding relations in a list
        neighbor_tuples = list(self.relgraph.get(entity_id, set()))
        random.shuffle(neighbor_tuples)
        return neighbor_tuples[:max_to_keep]
    
    def get_neighbor_ids_and_rels_with_condition(self, entity_id: str, rel: list, max_to_keep=10) -> List[str]:
        # return neighbor ids, as well as the corresponding relations in a list
        
        hrs = [entity_id+'|'+x for x in rel]
        results = []
        for hr in hrs:
            # pdb.set_trace()
            # list(self.relgraph.get(hr, set()))
            results += list(self.hrelgraph.get(hr, set()))
        # neighbor_tuples = list(self.relgraph.get(entity_id, set()))
        random.shuffle(results)
        return results[:max_to_keep]
    
    
    def get_n_hop_entity_indices(self, entity_id: str,
                                 entity_dict: EntityDict,
                                 n_hop: int = 2,
                                 # return empty if exceeds this number
                                 max_nodes: int = 100000) -> set:
        if n_hop < 0:
            return set()

        seen_eids = set()
        seen_eids.add(entity_id)
        queue = deque([entity_id])
        for i in range(n_hop):
            len_q = len(queue)
            for _ in range(len_q):
                tp = queue.popleft()
                for node in self.graph.get(tp, set()):
                    if node not in seen_eids:
                        queue.append(node)
                        seen_eids.add(node)
                        if len(seen_eids) > max_nodes:
                            return set()
        return set([entity_dict.entity_to_idx(e_id) for e_id in seen_eids])


def reverse_triplet(obj):
    return {
        'head_id': obj['tail_id'],
        'head': obj['tail'],
        'relation': 'inverse {}'.format(obj['relation']),
        'tail_id': obj['head_id'],
        'tail': obj['head']
    }
