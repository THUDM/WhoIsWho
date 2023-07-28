import time
import os
from os.path import join,dirname
from typing import Tuple, List, Union, Dict, Callable, Any, Optional


log_time = time.strftime("%m%d_%H%M%S")

def version2path(version: dict) -> dict:
    """
    Map the dataset information to the corresponding path
    """
    name = version.get("name","")
    task = version.get("task","")
    type = version.get("type","")

    # print(os.path.abspath(dirname(__file__)))
    data_root =  join(os.path.abspath(dirname(__file__)), f'dataset/data/',name, task,'')
    feat_root = join(os.path.abspath(dirname(__file__)), f'featureGenerator/feat/',name, task,'')

    #data
    raw_data_root = data_root
    processed_data_root = os.path.join(data_root, 'processed_data/')
    whoiswhograph_data_root =  os.path.join(data_root, 'graph_data/')
    whoiswhograph_extend_processed_data = os.path.join(data_root,'whoiswhograph_extend_processed_data/')
    whoiswhograph_emb_root = os.path.join(data_root, 'graph_embedding/')

    #feat
    hand_feat_root = os.path.join(feat_root, 'hand/')
    bert_feat_root = os.path.join(feat_root, 'bert/')
    graph_feat_root = os.path.join(feat_root, 'graph/')

    v2path={
        'name': name,
        'task': task,
        'type': type,
        'raw_data_root': raw_data_root,
        'processed_data_root': processed_data_root,
        'whoiswhograph_data_root': whoiswhograph_data_root,
        'whoiswhograph_extend_processed_data': whoiswhograph_extend_processed_data,
        'whoiswhograph_emb_root': whoiswhograph_emb_root,
        'hand_feat_root': hand_feat_root,
        'bert_feat_root': bert_feat_root,
        'graph_feat_root': graph_feat_root}
    return v2path


paper_idf_path =  join(os.path.abspath(dirname(__file__)), 'saved/paper-tf-idf/')
snd_embs_path  =  join(os.path.abspath(dirname(__file__)), 'saved/snd-embs/')
pretrained_oagbert_path = join(os.path.abspath(dirname(__file__)), 'saved/oagbert-v2-sim/')
pretrained_word2vec_path = join(os.path.abspath(dirname(__file__)), 'saved/word2vec/')
uuid_path = join(os.path.abspath(dirname(__file__)),'saved/used_uuid_v3+v4.json')

configs = {

    "train_neg_sample"              : 19,
    "test_neg_sample"               : 19,


    "train_max_papers_each_author"  : 100,
    "train_min_papers_each_author"  : 5,

    "train_max_semi_len"            : 24,
    "train_max_whole_semi_len"      : 256,
    "train_max_per_len"             : 128,

    "train_max_semantic_len"        : 64,
    "train_max_whole_semantic_len"  : 512,
    "train_max_whole_len"           : 512,
    "raw_feature_len"               : 41,
    "feature_len"                   : 36 + 41,
    "bertsimi_graph_handfeature_len": 36 + 41 + 41 + 41,
    "str_len"                       : 36,
    "dl_len"                        : 44,
    # "train_knrm_learning_rate"    : 6e-5,
    "train_knrm_learning_rate"      : 2e-3,
    "local_accum_step"              : 32,

    "hidden_size"                   : 768,
    "n_epoch"                       : 15,
    "show_step"                     : 1,
    "padding_num"                   : 1,
}


class RNDFilePathConfig:
    # offline data
    train_name2aid2pid = "train/train_author.json"
    train_pubs = "train/train_pub.json"

    # valid
    database_name2aid2pid = "valid/whole_author_profiles.json"
    database_pubs = "valid/whole_author_profiles_pub.json"

    # train+valid  data_process.py  get_name2aid2pid
    whole_name2aid2pid = 'database/name2aid2pid.whole.json'
    whole_pubsinfo = 'database/pubs.info.json'

    unass_candi_offline_path = 'train/unass_candi.whole.json'
    unass_candi_v1_path = 'onlinev1/unass_candi.json'
    unass_candi_v2_path = 'onlinev2/unass_candi.json'

    unass_pubs_info_v1_path = 'valid/cna_valid_unass_pub.json'
    unass_pubs_info_v2_path = 'test/cna_test_unass_pub.json'

    # feat_dict
    feat_dict_path = 'feat/'


