import sys
sys.path.append('../../../')
from whoiswho.utils import load_json, save_json, load_pickle, save_pickle,double_map,read_txt
import os
import torch
import re
import copy
import numpy as np
from typing import List
from tqdm import tqdm
import random
from torch_geometric.data import HeteroData,Data
from typing import List

'''
两步走
1.生成path pair 并转换为num pair
2.返回定制信息的graph
'''

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
            # 保证论文子图的中心论文能在第一位
            subgraph_embedding['paper_emb'][edge[0]] = all_embed_dict[edge[0]].reshape(1,-1)
            subgraph_embedding['paper_emb'][edge[1]] = all_embed_dict[edge[1]].reshape(1,-1)
        except:
            delete_index.append(i)
    for i in sorted(delete_index, reverse=True):
        del paper_refpaper[i]

    delete_index = []
    for i, edge in enumerate(author_paper):
        try:
            # 保证作者子图中心论文能在第一位
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

    return subgraph_embedding,edges  #返回图中三种emb的emb_dict 三种边的edge_dict

def nodename2index(all_dict):
    index = list(range(len(all_dict)))
    nodename2index = dict(zip(all_dict.keys(), index))

    return nodename2index

def load_node(emb_dict,node_type: str):#异构
    hetero_emb = emb_dict[node_type]  #node_type这种节点类型的向量
    if hetero_emb:   #可能会加载到空字典
        node2index = nodename2index(hetero_emb)
        x = double_map(node2index,hetero_emb) #numpy
        x = torch.from_numpy(x)
    else:
        return None,None

    return x, node2index #为映射字典 用于边映射

def load_edge(edge_dict,node_type: str,src_mapping: dict, dst_mapping: dict):#异构
    hetero_edge = edge_dict[node_type]
    if hetero_edge:
        src = [src_mapping[edge[0]] for edge in hetero_edge]
        dst = [dst_mapping[edge[1]] for edge in hetero_edge]
        edge_index = torch.tensor([src, dst])
    else:
        return []

    return edge_index

def save_hetero(emb_dict: dict,edges: dict,need_more_paper=False):
    #edges emb_dict 由get_embedding方法同步生成
    # 三种节点
    author_emb, author_map = load_node(emb_dict, 'author_emb')
    paper_emb, paper_map = load_node(emb_dict, 'paper_emb')
    org_emb, org_map = load_node(emb_dict, 'org_emb')

    # 三种边 根据source/target的type 对边进行编码
    author_paper = load_edge(edges, "author--paper", author_map, paper_map)
    author_org = load_edge(edges, "author--org", author_map, org_map)
    paper_paper = load_edge(edges, "paper--refpaper", paper_map, paper_map)


    data = HeteroData()
    data["paper"].x = paper_emb
    data["author"].x = author_emb
    data["affiliation"].x = org_emb
    if torch.is_tensor(author_paper):
        data["author", "writes", "paper"].edge_index = author_paper
    if torch.is_tensor(author_org):
        data["author", "in", "affiliation"].edge_index = author_org

    if torch.is_tensor(paper_paper):  # 有些论文查不到引用论文 不添加这种边
        data["paper", "cites", "paper"].edge_index = paper_paper  # A cites B

    # reverse
    if torch.is_tensor(author_paper):
        data["paper", "write_rev", "author"].edge_index = rev_edges(author_paper)
    if torch.is_tensor(author_org):
        data["affiliation", "in_rev", "author"].edge_index = rev_edges(author_org)
    if torch.is_tensor(paper_paper):
        data["paper", "cite_rev", "paper"].edge_index = rev_edges(paper_paper)

    # paper_graph构图的同时 给引用论文的数量 用于取平均表达target_unass
    if need_more_paper:
        paper_emb_len = len(edges["paper--refpaper"]) + 1
        return data, paper_emb_len
    else:
        return data
#用于save_homo 按papergraph/authorgraph指定的emb顺序合并emb_dict
def load_all_node(emb_dict,node_type: List[str]):#同构
    first_emb_len=0
    all_emb ={}
    #指定各类emb合并的顺序
    for i,type in enumerate(node_type):
        if i==0:
            first_emb_len=len(emb_dict[type])
        all_emb.update(emb_dict[type] )

    if all_emb:#可能会加载到空字典
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

 #生成同构图
def save_homo(emb_dict: dict,edges: dict,emb_order: list,need_more_paper=False):
    '''
    同构图要作者子图、论文子图分别考虑 分别要将author、unass放在emb首位
    :param emb_dict:
    :param edges:
    :param emb_order: 参数emb_order来配置各类emb组合为all_emb的顺序
    :param need_more_paper: 仅paper_graph会激活 统计引用论文数量
    :return:
    '''
    #对papergraph  first_emb_len就是paper_emb长度
    #对authorgraph first_emb_len就是author_emb长度
    all_emb, nodename2index,first_emb_len = load_all_node(emb_dict, emb_order)
    edge,paper_emb_len = load_all_edge(edges, ['author--paper', 'author--org','paper--refpaper'], nodename2index)
    data = Data(x=all_emb, edge_index=edge)
    # data = Graph(x=all_emb, edge_index=edge)
    # paper_graph构图的同时 给引用论文的数量 用于取平均表达target_unass
    if need_more_paper:
        return data,paper_emb_len
    else:
        return data

class GraphPairDataset:
    def __init__(self,args,type='graph',numpair_path=None,all_emb=None,test=False,
                 need_more_paper=False,need_label_paper=False,profile_path=None,het=False):
        '''
        type: 配置getitem所返回的pair元素类型
        numpair_path： 配置pair路径
        need_more_paper： paper_graph是否需要提供引用论文数
        need_label_paper: author_graph是否提供一阶labelpaper
        het：是否使用异构图
        '''
        self.type = type
        self.test = test #在测试集上 负例取所有负例 不限制负例数量
        self.args = args
        self.need_more_paper=need_more_paper
        self.need_label_paper=need_label_paper
        self.het=het

        # 通过pair路径 加载pair
        if self.type == 'graph':
            self.idx_to_path = load_json(self.args.idx_to_path)
            self.data = load_json(numpair_path)
            print(len(self.data))
            if all_emb:
                self.all_emb = all_emb
            else:
                self.all_emb = np.load(self.args.all_emb_path, allow_pickle=True).item()

        else: #如果没有生成好的pair get_pair去组织pair
            self.data = self.get_pair(args)
            print(len(self.data))

    def load_graph_pair(self, path_pair):
        pair = []
        # pair索引0是papergraph 要paper_emb放前 [0]之后是authorgraph 要author_emb放前
        if self.type == 'graph':
            # 获取graph名称
            graph_names = []

            # 用于逐个记录authorgraph信息
            author_emb_len_list = list()
            label_paper_len_list = list()
            for i, graph_path in enumerate(path_pair):  # paper 和 1:10的author正负例
                graph_name = graph_path.split("/", 100)[-1]
                graph_names.append(graph_name)
                # 即时生成子图所需emb与边信息
                graph_emb, edges = get_embedding(graph_path, self.all_emb)
                if i == 0:
                    # 构建paper的同构图 paperemb放前
                    '''
                    paper graph构图时可以选择是否需要`引用论文数` 如不需要paper_emb_len为0
                    '''
                    graph = save_homo(graph_emb, edges, emb_order=['paper_emb', 'author_emb', 'org_emb'],
                                      need_more_paper=self.need_more_paper)
                    if isinstance(graph, tuple):
                        pair.append(graph[0])
                        paper_emb_len = graph[1]
                    else:
                        pair.append(graph)
                        paper_emb_len = 0
                else:
                    # 构建author的同构图 author_emb在all_emb首位
                    # 相比论文图 save_homo传入need_label_paper 构图的同时可以获取`引用论文数` 返回tuple
                    graph = save_homo(graph_emb, edges, emb_order=['author_emb', 'paper_emb', 'org_emb'])

                    pair.append(graph)

        return pair, graph_names, paper_emb_len

    def __getitem__(self, idx):
        '''
        getitem返回值pair分为三类 path_pair / homograph_pair  / heterograph_pair
        :param idx:
        :return:
        '''
        pair = []
        if self.test==False:
            nums=12
        else:
            nums=1000
        # 仅用来第一次产生path_pair 之后使用映射后的pair信息
        if self.type == 'path' :
            for graph_path in self.data[idx][:nums]:  # paper 和 1:10的author正负例
                pair += (graph_path,)  # 直接存图路径
                #这种写法 效果同append
            return pair

        #用于同构图pair
        # pair索引0是papergraph 要paper_emb放前 [0]之后是authorgraph 要author_emb放前
        if self.type == 'graph'and self.het==False:
            # graph_path为num
            # 获取graph名称
            graph_names=[]
            # 用于逐个记录authorgraph信息
            author_emb_len_list = list()
            label_paper_len_list = list()
            for i,graph_path in enumerate(self.data[idx][:nums]):  # paper 和 1:10的author正负例
                graph_path = self.idx_to_path[graph_path]
                graph_name = graph_path.split("/",100)[-1]
                graph_names.append(graph_name)
                # 即时生成子图所需emb与边信息
                graph_emb, edges = get_embedding(graph_path, self.all_emb)

                if i == 0:
                    #构建paper的同构图 paperemb放前
                    #相比作者图 save_homo传入need_more_paper 构图的同时可以获取`引用论文数` 返回tuple
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

        # papergraph的引用论文加自身
        return pair, graph_names, paper_emb_len

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_pair(args):  #本方法构建pair各元素路径
        data_info = list()
        unass_paper = read_txt(args.paper_ids_path)
        ground_truth = load_json(args.positive_pair_path)
        samename_author = load_json(args.samename_path)

        for paper in unass_paper:
            paper_path = os.path.join(args.paper_graph_path, paper)
            true_author = ground_truth.get(paper, [])  # ground_truth论文对应的作者可能不止一个 列表
            true_author = list(set(true_author))
            if not true_author: #NIL论文无法构建正例
                continue      # valid '53e997ecb7602d9701fea3e0' 不在ground_truth中 是NIL
            for authorid in true_author:
                # 有些解码后含new#的aminerid含'-' 会导致'-'分隔后拿不到正确的名字 导致samename_author查不到
                if re.search(r'new#.*\d+',authorid):
                    split_string = re.split(r'new#.*\d+',authorid)[1]
                    name = split_string.split('-',1)[1]
                elif re.search(r'---.*\d+',authorid):
                    split_string = re.split(r'---.*\d+', authorid)[1]
                    name = split_string.split('-', 1)[1]
                else:
                    name = authorid.split('-',1)[1]
                name_authors = copy.deepcopy(samename_author[name])
                # 有些ground_truth的论文作者 不在samename_author中 没有负例则无法构建pair
                if len(name_authors) == 0:
                    print(name, authorid)

                if authorid in name_authors:
                    name_authors.remove(authorid)  # 去除正例 剩下的都是负例

                positive_author_path = os.path.join(args.author_graph_path, authorid)
                negative_author_path = [os.path.join(args.author_graph_path, id) for id in name_authors]
                random.shuffle(negative_author_path)

                sample = [paper_path, positive_author_path]
                sample += negative_author_path
                # print(len(sample))
                # if len(sample) == 2:
                #     print(authorid)
                if len(sample)>2: #同名下作者不止1个 才能构建pair
                    data_info.append(sample)



        return data_info







