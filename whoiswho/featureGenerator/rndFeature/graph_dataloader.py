import sys
sys.path.append('../../../')
from whoiswho.utils import load_json, save_json, load_pickle, save_pickle,double_map,read_txt
import os
from os.path import join,dirname
import torch
import re
import copy
import argparse
import numpy as np
from typing import List
from tqdm import tqdm
import random
from torch_geometric.data import HeteroData,Data
from typing import List
from whoiswho.config import RNDFilePathConfig,version2path
from whoiswho.utils import load_json, save_json

def rev_edges(x):
    return torch.stack((x[1], x[0]), dim=0)

def get_embedding(graph_path,all_embed_dict):
    subgraph_embedding = {'author_emb':{},
                           'paper_emb':{},
                           'org_emb':{}   }
    if 'truncate_edge.json' in os.listdir(graph_path):
        edge_file = os.path.join(graph_path, 'truncate_edge.json')
        edges = load_json(edge_file)
    else:
        edge_file = os.path.join(graph_path, 'all_edge.json')
        edges = load_json(edge_file)
    paper_refpaper=edges["paper--refpaper"]
    author_paper =edges["author--paper"]
    author_org = edges["author--org"]

    delete_index = []
    for i, edge in enumerate(paper_refpaper):
        try:
            # Ensure that the center of the paper's subgraph will be in the first place
            subgraph_embedding['paper_emb'][edge[0]] = all_embed_dict[edge[0]].reshape(1,-1)
            subgraph_embedding['paper_emb'][edge[1]] = all_embed_dict[edge[1]].reshape(1,-1)
        except:
            delete_index.append(i)
    for i in sorted(delete_index, reverse=True):
        del paper_refpaper[i]

    delete_index = []
    for i, edge in enumerate(author_paper):
        try:
            # Ensure that the center of the author's subgraph will be in the first place
            subgraph_embedding['author_emb'][edge[0]] = all_embed_dict[edge[0]].reshape(1,-1)
            subgraph_embedding['paper_emb'][edge[1]] = all_embed_dict[edge[1]].reshape(1,-1)
        except:
            delete_index.append(i)
    for i in sorted(delete_index, reverse=True):
        del author_paper[i]

    delete_index = []
    for i,edge in enumerate(author_org):
        try:
            subgraph_embedding['author_emb'][edge[0]] = all_embed_dict[edge[0]].reshape(1,-1)
            subgraph_embedding['org_emb'][edge[1]] = all_embed_dict[edge[1]].reshape(1,-1)
        except:
            delete_index.append(i)
    for i in sorted(delete_index, reverse=True):
        del author_org[i]

    edges={"paper--refpaper":paper_refpaper,"author--paper":author_paper,"author--org":author_org}

    return subgraph_embedding,edges

def nodename2index(all_dict):
    index = list(range(len(all_dict)))
    nodename2index = dict(zip(all_dict.keys(), index))

    return nodename2index

def load_all_node(emb_dict,node_type: List[str]):
    first_emb_len=0
    all_emb ={}
    # Specify the order of merging
    for i,type in enumerate(node_type):
        if i==0:
            first_emb_len=len(emb_dict[type])
        all_emb.update(emb_dict[type] )

    if all_emb:
        node2index = nodename2index(all_emb)
        x = double_map(node2index,all_emb) #numpy
        x = torch.from_numpy(x).float()
    else:
        return None,None,None

    return x, node2index,first_emb_len

def load_all_edge(edge_dict,node_type: List[str],nodename2index: dict):#同构
    all_edge = []
    for type in node_type:
        all_edge.extend(edge_dict[type])

    paper_emb_len = len(edge_dict['paper--refpaper'])+1

    if all_edge:
        src = [nodename2index[edge[0]] for edge in all_edge]
        dst = [nodename2index[edge[1]] for edge in all_edge]
        edge_index = torch.tensor([src, dst])
    else:
        return []

    return edge_index,paper_emb_len

def save_homo(emb_dict: dict,edges: dict,emb_order: list,need_more_paper=False):
    all_emb, nodename2index,first_emb_len = load_all_node(emb_dict, emb_order)
    edge,paper_emb_len = load_all_edge(edges, ['author--paper', 'author--org','paper--refpaper'], nodename2index)
    data = Data(x=all_emb, edge_index=edge)

    if need_more_paper:
        return data,paper_emb_len
    else:
        return data

def save_graph_pair(args,author_nums_limit):
    dataset = GraphPairDataset(args,type='path',author_nums_limit=author_nums_limit)
    pair_datas = []
    for i in tqdm(range(len(dataset))):
        if dataset[i]:
            pair_datas.append(dataset[i])

    save_json(pair_datas,args.pathpair_path)


def map_path_num(args):
    path_pairs = load_json(args.pathpair_path)
    idx_to_path = []
    for pair in path_pairs:
        idx_to_path.extend(pair)
    idx_to_path = list(set(idx_to_path))

    path_to_idx = {path:i for i,path in enumerate(idx_to_path)}
    save_json(path_to_idx,args.path_to_idx)
    save_json(idx_to_path,args.idx_to_path)
    #map
    num_pairs = []
    for pair in path_pairs:
        num_pair=[]
        for path in pair:
            num_pair.append(path_to_idx[path])
        num_pairs.append(num_pair)
    save_json(num_pairs,args.numpair_path)


def split_num_pairs(args,ratio=0.7):
    num_pairs = load_json(args.numpair_path)
    random.shuffle(num_pairs)
    train_pairs = num_pairs[:int(len(num_pairs)*ratio)]
    valid_pairs = num_pairs[int(len(num_pairs)*ratio):]
    print(f'train pairs {len(train_pairs)} \n valid pairs {len(valid_pairs)}')
    save_json(train_pairs, args.train_pair_path)
    save_json(valid_pairs, args.valid_pair_path)


def prepare_graph_files(version):
    v2path = version2path(version)

    parser = argparse.ArgumentParser(description='Prepare files related to the graph')
    # Used for parsing num_pair
    parser.add_argument('--paper_ids_path', type=str,
                        default=join(v2path['whoiswhograph_data_root'], 'train_graph', 'paper_ids_ft.txt'))
    parser.add_argument('--positive_pair_path', type=str,
                        default=join(v2path['whoiswhograph_data_root'], 'train_graph', 'train_ground_truth.json'))
    parser.add_argument('--samename_path', type=str,
                        default=join(v2path['whoiswhograph_data_root'], 'train_graph', 'samename_authors.json'))

    parser.add_argument('--paper_graph_path', type=str,
                        default=join(v2path['whoiswhograph_data_root'], 'train_graph', 'paper_graph'))
    parser.add_argument('--author_graph_path', type=str,
                        default=join(v2path['whoiswhograph_data_root'], 'train_graph', 'author_graph'))

    parser.add_argument('--pathpair_path', type=str,
                        default=join(v2path['whoiswhograph_data_root'], 'train_graph', 'pair_path.json'))
    # Used for parsing num_pair
    parser.add_argument('--path_to_idx', type=str,
                        default=join(v2path['whoiswhograph_data_root'], 'train_graph', 'path_to_idx.json'))
    parser.add_argument('--idx_to_path', type=str,
                        default=join(v2path['whoiswhograph_data_root'], 'train_graph', 'idx_to_path.json'))
    parser.add_argument('--numpair_path', type=str,
                        default=join(v2path['whoiswhograph_data_root'], 'train_graph', 'pair_num.json'))

    # split_num_pairs
    parser.add_argument('--train_pair_path', type=str,
                        default=join(v2path['whoiswhograph_data_root'], 'train_graph', 'train_pair.json'))
    parser.add_argument('--valid_pair_path', type=str,
                        default=join(v2path['whoiswhograph_data_root'], 'train_graph', 'valid_pair.json'))
    args = parser.parse_args()

    # Prepare graph related files
    save_graph_pair(args, author_nums_limit=False)  # GraphPairDataset use type='path'
    map_path_num(args)
    split_num_pairs(args)

class GraphPairDataset:
    def __init__(self,args=None,type='graph',numpair_path=None,all_emb=None,author_nums_limit=True,
                 need_more_paper=False,need_label_paper=False,het=False):
        self.type = type
        self.author_nums_limit = author_nums_limit
        self.args = args
        self.need_more_paper=need_more_paper
        self.need_label_paper=need_label_paper
        self.het=het
        self.all_emb = all_emb

        if self.type == 'path':
            self.data = self.get_path_pair(args)
            print(len(self.data))

        # Called by gnn_train and loaded with graph related files
        if self.type == 'graph'and numpair_path and args:
            self.idx_to_path = load_json(self.args.idx_to_path)
            self.data = load_json(numpair_path)
            print(len(self.data))
            if self.args.all_emb_path:
                self.all_emb = np.load(self.args.all_emb_path, allow_pickle=True).item()

    def load_graph_pair(self, path_pair: list):
        pair = []
        if self.type == 'graph':
            graph_names = []
            for i, graph_path in enumerate(path_pair):
                graph_name = graph_path.split("/", 100)[-1]
                graph_names.append(graph_name)
                # Obtain the edges and embedding required for the graph
                graph_emb, edges = get_embedding(graph_path, self.all_emb)
                # The index 0 of path_pair is papergraph, paper_emb should be put in front.
                # After index 0 is authorgraph, author_emb should be put in front.
                if i == 0:
                    graph = save_homo(graph_emb, edges, emb_order=['paper_emb', 'author_emb', 'org_emb'],
                                      need_more_paper=self.need_more_paper)
                    if isinstance(graph, tuple):
                        pair.append(graph[0])
                        paper_emb_len = graph[1]
                    else:
                        pair.append(graph)
                        paper_emb_len = 0
                else:
                    graph = save_homo(graph_emb, edges, emb_order=['author_emb', 'paper_emb', 'org_emb'])
                    pair.append(graph)

        return pair, graph_names, paper_emb_len


    def __getitem__(self, idx: int):
        if self.author_nums_limit :
            nums = 12
        else:
            nums = 1000

        pair = []
        if self.type == 'path' :
            for graph_path in self.data[idx][:nums]:
                pair += (graph_path,)
            return pair

        if self.type == 'graph'and self.het==False:
            graph_names=[]
            for i,num_path in enumerate(self.data[idx][:nums]):
                graph_path = self.idx_to_path[num_path]
                graph_name = graph_path.split("/",100)[-1]
                graph_names.append(graph_name)
                # Obtain the edges and embedding required for the graph
                graph_emb, edges = get_embedding(graph_path, self.all_emb)
                # The index 0 of pair is papergraph, paper_emb should be put in front.
                # After index 0 is authorgraph, author_emb should be put in front.
                if i == 0:
                    graph = save_homo(graph_emb, edges,emb_order=['paper_emb','author_emb','org_emb'],
                                      need_more_paper=self.need_more_paper)
                    if isinstance(graph,tuple):
                        pair.append(graph[0])
                        paper_emb_len=graph[1]
                    else:
                        pair.append(graph)
                        paper_emb_len = 0
                else:
                    graph = save_homo(graph_emb, edges, emb_order=['author_emb', 'paper_emb', 'org_emb'])
                    pair.append(graph)

            return pair, graph_names, paper_emb_len

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_path_pair(args):
        data_info = list()
        unass_paper = read_txt(args.paper_ids_path)
        ground_truth = load_json(args.positive_pair_path)
        samename_author = load_json(args.samename_path)

        for paper in unass_paper:
            paper_path = os.path.join(args.paper_graph_path, paper)
            true_author = ground_truth.get(paper, [])
            true_author = list(set(true_author))
            if not true_author:
                continue
            for authorid in true_author:
                if re.search(r'new#.*\d+',authorid):
                    split_string = re.split(r'new#.*\d+',authorid)[1]
                    name = split_string.split('-',1)[1]
                elif re.search(r'---.*\d+',authorid):
                    split_string = re.split(r'---.*\d+', authorid)[1]
                    name = split_string.split('-', 1)[1]
                else:
                    name = authorid.split('-',1)[1]
                name_authors = copy.deepcopy(samename_author[name])

                if len(name_authors) == 0:
                    print(name, authorid)

                if authorid in name_authors:
                    name_authors.remove(authorid)

                positive_author_path = os.path.join(args.author_graph_path, authorid)
                negative_author_path = [os.path.join(args.author_graph_path, id) for id in name_authors]
                random.shuffle(negative_author_path)

                sample = [paper_path, positive_author_path]
                sample += negative_author_path

                if len(sample)>2:
                    data_info.append(sample)

        return data_info

if __name__ == '__main__':
    version = {"name": 'v3', "task": 'RND', "type": 'train'}
    prepare_graph_files(version)





