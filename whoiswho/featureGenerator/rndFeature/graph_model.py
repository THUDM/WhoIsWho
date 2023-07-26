import pickle
import sys
sys.path.append('../../../')
from whoiswho.utils import load_json, save_json, load_pickle, save_pickle
import random
import json
import os
from os.path import join,dirname
import copy
import numpy as np
from functools import reduce
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import argparse
from torch_geometric.data.batch import Batch
from whoiswho.featureGenerator.rndFeature.graph_dataloader import GraphPairDataset
from whoiswho.featureGenerator.rndFeature.model import SAGE, GAT,infonce,matchingModel,matchingModel_gnn,gat_parameters,sage_parameters,HR,MRR
from whoiswho.config import RNDFilePathConfig,version2path

def get_batch(dataset, BATCH_SIZE):
    n_batches = len(dataset) // BATCH_SIZE
    index = list(range(n_batches * BATCH_SIZE))
    for idx in range(0, len(index), BATCH_SIZE):
        present_batch = [dataset[i] for i in range(idx, idx + BATCH_SIZE)]
        yield present_batch


def train(model,score_model,train_data,optimizer,device):
    batch_size=8
    log_interval=10
    correct = 0.
    hit_3 = 0.
    mrr_score = 0.
    total = 0.
    loss_mean = 0.
    loss_log = []
    model.train()
    for i, batch in enumerate(tqdm(get_batch(train_data, BATCH_SIZE=batch_size))):
        loss = []
        optimizer.zero_grad()
        for pair,graph_names,paper_emb_len in batch:
            data_list = copy.deepcopy(pair)
            length_list = [0] + [len(data_list[i].x) for i in range(len(data_list))]

            graph_index = [reduce(lambda a, b: a + b, length_list[:i]) for i in range(2, len(length_list)+1)]
            batch_graph = Batch.from_data_list(data_list)
            batch_graph.to(device)
            out=model(batch_graph.x,batch_graph.edge_index)

            if paper_emb_len == 0:
                paper_emb = out[0].unsqueeze(0)  # (1,128)
            else:
                paper_emb = out[:paper_emb_len]
                paper_emb = torch.mean(paper_emb, dim=0, keepdim=True)

            authors_emb=[]
            for idx in range(len(graph_index)-1):
                author_emb = out[graph_index[idx]:graph_index[idx+1]]
                authors_emb.append(author_emb)
            pos_author = authors_emb[0]  #Correct author subgraph node embeddings
            random.shuffle(authors_emb)
            for id,item in enumerate(authors_emb):
                if item.equal(pos_author):
                    label= id   #The correct author's index after shuffling
            logits = score_model(paper_emb, authors_emb)
            loss.append(infonce(logits, 0.1,label))  # one pair loss

            _, predicted = torch.max(logits, 1)
            if predicted == label:
                correct += 1

            label = torch.tensor([label]).to(device)
            p = torch.argsort(logits, dim=1, descending=True)
            hit_3 += HR(label, p, 3)
            mrr_score += MRR(label, p, 30)

            total += 1


        # batch loss
        batch_loss = loss[0]
        for num in range(1, len(loss)):
            batch_loss += loss[num]
        loss = batch_loss / batch_size
        loss.backward()
        optimizer.step()

        loss_mean += loss.item()
        if (i + 1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            loss_log.append(loss_mean)
            print("Iteration[{:0>3}] Train Loss: {:.4f} TrainBatch HR@1:{:.2%}".format(
                i + 1, loss_mean,correct / total))
            loss_mean = 0.

    HR_1 = correct / total
    HR_3 = hit_3 / total
    MRR_score = mrr_score / total
    print("Train HR@1:{:.2%} HR@3:{:.2%} MRR@1:{:.2%}correct:{} total{}".format(HR_1, HR_3, MRR_score, correct, total))

    return loss_log

def valid(model,score_model,valid_data,device):
    batch_size=8
    log_interval=50

    loss_mean = 0.
    loss_log=[]
    correct = 0.
    hit_3 = 0.
    mrr_score = 0.
    total = 0.
    model.eval()
    score_model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(get_batch(valid_data, BATCH_SIZE=batch_size))):
            loss = []
            for pair,graph_names,paper_emb_len in batch:
                data_list = copy.deepcopy(pair)
                length_list = [0] + [len(data_list[i].x) for i in range(len(data_list))]
                graph_index = [reduce(lambda a, b: a + b, length_list[:i]) for i in range(2, len(length_list) + 1)]
                batch_graph = Batch.from_data_list(data_list)
                batch_graph.to(device)
                out = model(batch_graph.x, batch_graph.edge_index)

                if paper_emb_len == 0:
                    paper_emb = out[0].unsqueeze(0)  # (1,128)
                else:
                    paper_emb=out[:paper_emb_len]
                    paper_emb = torch.mean(paper_emb,dim=0,keepdim=True)

                authors_emb = []
                for idx in range(len(graph_index) - 1):
                    author_emb = out[graph_index[idx]:graph_index[idx + 1]]
                    authors_emb.append(author_emb)

                pos_author = authors_emb[0]  #Correct author subgraph node embeddings
                random.shuffle(authors_emb)

                for id, item in enumerate(authors_emb):
                    if item.equal(pos_author):
                        label = id    #The correct author's index after shuffling

                logits = score_model(paper_emb, authors_emb)
                loss.append(infonce(logits, 0.1, label))  # one pair loss

                _, predicted = torch.max(logits, 1)
                if predicted == label:
                    correct += 1

                label = torch.tensor([label]).to(device)
                p = torch.argsort(logits, dim=1, descending=True)
                hit_3 += HR(label, p, 3)
                mrr_score += MRR(label, p, 30)
                total += 1

            # batch loss
            batch_loss = loss[0]
            for num in range(1, len(loss)):
                batch_loss += loss[num]
            loss = batch_loss / batch_size
            loss_mean += loss.item()
            if (i + 1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                loss_log.append(loss_mean)
                print("Iteration[{:0>3}] VALID Loss: {:.4f}".format(
                     i + 1, loss_mean))
                loss_mean = 0.

    HR_1 = correct / total
    HR_3 = hit_3 / total
    MRR_score = mrr_score / total
    print("Valid HR@1:{:.2%} HR@3:{:.2%} MRR@1:{:.2%}correct:{} total{}".format(HR_1, HR_3, MRR_score, correct, total))
    return HR_1,loss_log


def gnn_train():
    version = {"name": 'v3', "task": 'RND', "type": 'train'}
    v2path = version2path(version)

    parser = argparse.ArgumentParser(description='gnn_models')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model', type=str, default='gat')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--channels', type=int, default=128)

    # Whether to use scheduler in training
    parser.add_argument('--scheduler',  default=True)
    # On paper graph , whether to use the average of the referenced papers of the central node as embedding for the central node.
    parser.add_argument('--need_more_paper', default=True)
    # Used for parsing num_pair
    parser.add_argument('--idx_to_path', type=str,
                        default=join(v2path['whoiswhograph_data_root'],'train_graph','idx_to_path.json'))
    # Load prepared node embeddings
    parser.add_argument('--all_emb_path', type=str,
                        default = join(v2path['whoiswhograph_emb_root'],'train','all_train_emb_sim.npy'))
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    all_emb=np.load(args.all_emb_path, allow_pickle=True).item()

    #Load graph by num_pair
    train_data = GraphPairDataset(args=args, numpair_path=join(v2path['whoiswhograph_data_root'],'train_graph','train_pair.json'),
                                  all_emb=all_emb, need_more_paper=args.need_more_paper)
    valid_data = GraphPairDataset(args=args, numpair_path=join(v2path['whoiswhograph_data_root'],'train_graph','valid_pair.json'),
                                  all_emb=all_emb, need_more_paper=args.need_more_paper)

    if args.model == 'gat':
        para_dict = gat_parameters
        model_para = gat_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = GAT(in_channels=768, out_channels=args.channels, **model_para).to(device)
    if args.model == 'sage':
        para_dict = sage_parameters
        model_para = sage_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = SAGE(in_channels=768, out_channels=args.channels, **model_para).to(device)

    print(f'Model {args.model} initialized')

    print(sum(p.numel() for p in model.parameters()))
    torch.cuda.manual_seed(1)
    torch.manual_seed(1)
    model.reset_parameters()
    score_model = matchingModel_gnn(device).to(device)

    optimizer = torch.optim.Adam([
        {'params':model.parameters() },
        {'params': score_model.parameters()}
                        ],lr=args.lr, weight_decay=para_dict['l2'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,  gamma=0.9)


    maxscore = 0
    train_process_loss_log,valid_process_loss_log=[],[]

    for epoch in range(1, args.epochs + 1):
        #Check the performance of the initial model on valid set
        if epoch==1:
            hr_1, loss_log = valid(model,score_model,valid_data,device)

        print(f"epoch:{epoch} lr:{optimizer.param_groups[0]['lr']}")
        #train
        loss_log=train(model,score_model,train_data,optimizer,device)
        train_process_loss_log.append(loss_log)
        #valid
        hr_1,loss_log=valid(model,score_model,valid_data,device)
        valid_process_loss_log.append(loss_log)

        if args.scheduler:
            scheduler.step()

        if hr_1 > maxscore:
            maxscore = hr_1
            state = {'Gat': model.state_dict(),
                     'Linear': score_model.state_dict()}
            torch.save(state, '{}_model_oagbert_sim.pt'.format(args.model))

if __name__ == '__main__':
    #Train gnn based on whoiswhograph.
    gnn_train()