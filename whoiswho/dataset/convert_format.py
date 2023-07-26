'''
#将帆进姐数据转为whoiswho 转为立即删
1.对训练集profile中出现的valid test论文删除
2.重写split_train2dev  不需要找作者索引 得到train/offline_unass.json  train/offline_profile.json  train/unass_candi.whole.json
3.train/offline_unass.json  train/offline_profile.json  划分k折
4.合并 不用变 直接把全量profile换名
5.pretreat_unass valid test的pid与name拼接
'''
import os
import codecs
import pickle
import re
import random
import copy
from collections import defaultdict
import multiprocessing
from tqdm import tqdm
from pprint import pprint
from whoiswho import logger

from whoiswho.utils import load_json, save_json, get_author_index, dname_l_dict,unify_name_order
from whoiswho.character.name_match.tool.is_chinese import cleaning_name
from whoiswho.character.name_match.tool.interface import FindMain
from whoiswho.character.match_name import  match_name
from whoiswho.config import RNDFilePathConfig, configs, version2path
from whoiswho.dataset import load_utils

puncs = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
stopwords = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with',
            'the', 'by', 'we', 'be', 'is', 'are', 'can']
stopwords_extend = ['university', 'univ', 'china', 'department', 'dept', 'laboratory', 'lab',
                    'school', 'al', 'et', 'institute', 'inst', 'college', 'chinese', 'beijing',
                    'journal', 'science', 'international', 'key', 'sciences', 'research',
                    'academy', 'state', 'center']
stopwords_check = ['a', 'was', 'were', 'that', '2', 'key', '1', 'technology', '0', 'sciences', 'as',
                    'from', 'r', '3', 'academy', 'this', 'nanjing', 'shanghai', 'state', 's', 'research',
                    'p', 'results', 'peoples', '4', 'which', '5', 'high', 'materials', 'study', 'control',
                    'method', 'group', 'c', 'between', 'or', 'it', 'than', 'analysis', 'system',  'sci',
                    'two', '6', 'has', 'h', 'after', 'different', 'n', 'national', 'japan', 'have', 'cell',
                    'time', 'zhejiang', 'used', 'data', 'these']



def split_train2dev(data: list,processed_data_root: str, unass_ratio=0.2):
    def _get_last_n_paper(name, paper_ids, paper_info, ratio=0.2):
        cnt_unfind_author_num = 0  # 未找到作者 index 的数量
        name = cleaning_name(name)
        years = set()
        now_years = defaultdict(list)
        for pid in paper_ids:
            paperid = pid.split('-')[0]
            year = paper_info[paperid].get('year', '0')
            if year == '':
                year = 0
            else:
                year = int(year)
            if year < 1500 or year > 2023:
                year = 0
            years.add(year)
            # authors = paper_info[paperid].get('authors', [])
            # author_names = [a['name'] for a in authors]
            # author_res = FindMain(name, author_names)[0]
            # if len(author_res) > 0:
            #     aids = author_res[0][1]
            # else:
            #     aids = get_author_index(name, author_names, False)
            #     if aids < 0:
            #         aids = len(authors)
            #         cnt_unfind_author_num += 1
            # assert aids >= 0
            # if aids == len(authors):
            #     cnt_unfind_author_num += 1
            # assert aids >= 0, f"{name} 's paper {pid}"
            now_years[year].append((pid))

        #papers_list : sort paperid by year
        years = list(years)
        years.sort(reverse=False)
        papers_list = []
        assert len(years) > 0
        for y in years:
            papers_list.extend(now_years[y])

        split_gap = int((1 - ratio) * len(papers_list))
        unass_list = papers_list[split_gap:]
        prof_list = papers_list[0:split_gap]
        assert len(unass_list) > 0
        assert len(prof_list) > 0
        assert len(unass_list) + len(prof_list) == len(papers_list)
        return prof_list, unass_list

    def _split_unass(names, authors_info, papers_info, unass_info, dump_info):
        sum_unfind_author_num = 0
        unass_candi_list = []
        for name in names:
            unass_info[name] = {}
            dump_info[name] = {}
            for aid in authors_info[name]:
                # paper ids
                papers = authors_info[name][aid]
                try:
                    prof_list, unass_list = _get_last_n_paper(name, papers, papers_info, unass_ratio) #返回pid-index

                    #Create profile about train_unass
                    unass_info[name][aid] = unass_list
                    #Create profile about train_dump
                    dump_info[name][aid] = prof_list
                    # train_unass list
                    for pid in unass_info[name][aid]:
                        unass_candi_list.append((pid, name))
                except:
                    continue

        return unass_candi_list

    authors_info = data[0]
    papers_info = data[1]
    names = []
    for name in authors_info:
        names.append(name)
    random.shuffle(names)

    train_unass_info = {} #profile about train_unass
    train_dump_info = {}  #profile about train_dump
    # train_unass list
    train_unass_candi = _split_unass(names, authors_info, papers_info, train_unass_info, train_dump_info)
    save_json(train_unass_info, processed_data_root, "train/offline_unass.json")
    save_json(train_dump_info, processed_data_root, "train/offline_profile.json")
    save_json(train_unass_candi, processed_data_root, 'train/unass_candi.whole.json')



def split_list2kfold(s_list, k, start_index=0):
    # Partition the input list into k parts
    num = len(s_list)
    each_l = int(num / k)
    result = []
    remainer = num % k
    random.shuffle(s_list)
    last_index = 0
    for i in range(k):
        if (k + i - start_index) % k < remainer:
            result.append(s_list[last_index:last_index + each_l + 1])
            last_index += each_l + 1
        else:
            result.append(s_list[last_index:last_index + each_l])
            last_index += each_l
    return result, (start_index + remainer) % k


def kfold_main_func(processed_data_root,offline_whole_profile, offline_whole_unass, k=5):
    kfold_path = f"{processed_data_root}/train/kfold_dataset/"
    os.makedirs(kfold_path, exist_ok=True)
    # Get the list of author names in the training set and the number of candidates
    name_weight = []
    for name, aid2pids in offline_whole_profile.items():
        assert len(aid2pids.keys()) == len(offline_whole_unass[name].keys())
        name_weight.append((name, len(aid2pids.keys())))
    # name_weight.sort(key=lambda x: x[1])

    both_name_weight = []
    unused_name_weight = []
    for name, weight in name_weight:
        if weight < 20:
            unused_name_weight.append((name, weight))
        else:
            both_name_weight.append((name, weight))
    # Partition the set of names into k groups
    start_index = 0
    split_res = [[] for i in range(k)]
    tmp, start_index = split_list2kfold(unused_name_weight, k, start_index)
    for i in range(k):
        split_res[i].extend(tmp[i])
    tmp, start_index = split_list2kfold(both_name_weight, k, start_index)
    for i in range(k):
        split_res[i].extend(tmp[i])

    # Generate the training data set of four-tuples
    for i in range(k):
        this_root = os.path.join(kfold_path, f'kfold_v{i + 1}')
        os.makedirs(this_root, exist_ok=True)
        dev_names = split_res[i]
        train_names = []
        for j in range(k):
            if j != i:
                train_names.extend(split_res[j])

        train_ins = []
        for na_w in train_names:
            name = na_w[0]
            whole_candi_aids = list(offline_whole_unass[name].keys())
            if len(whole_candi_aids) < configs['train_neg_sample'] + 1:
                continue
            for pos_aid, pids in offline_whole_unass[name].items():
                for pid in pids:
                    neg_aids = copy.deepcopy(whole_candi_aids)
                    neg_aids.remove(pos_aid)
                    neg_aids = random.sample(neg_aids, configs['train_neg_sample'])
                    train_ins.append((name, pid, pos_aid, neg_aids))
        save_json(train_ins, this_root, 'train_ins.json')

        dev_ins = []
        for na_w in dev_names:
            name = na_w[0]
            whole_candi_aids = list(offline_whole_unass[name].keys())
            if len(whole_candi_aids) < configs['test_neg_sample'] + 1:
                continue
            for pos_aid, pids in offline_whole_unass[name].items():
                for pid in pids:
                    neg_aids = copy.deepcopy(whole_candi_aids)
                    neg_aids.remove(pos_aid)
                    neg_aids = random.sample(neg_aids, configs['test_neg_sample'])
                    dev_ins.append((name, pid, pos_aid, neg_aids))
        save_json(dev_ins, this_root, 'test_ins.json')
    print(name_weight)

#从profile中删除 valid test  帆进姐说不用删
def delete_papers(whole_profile,valid_data,test_data,raw_data_root,train_profile_path):
    valid_delete=0
    test_delete=0

    for item in valid_data:
        name = item['name']
        aid  = item['aid1']
        pid = item['pid']
        try:
            whole_profile[name][aid].remove(pid)
            valid_delete+=1
        except:
            print("Cannot find {} ".format(pid))
    print(f"valid has {len(valid_data)} papers\n delete {valid_delete} papers") #4848

    for item in test_data:
        name = item['name']
        aid  = item['aid1']
        pid = item['pid']
        try:
            whole_profile[name][aid].remove(pid)
            test_delete+=1
        except:
            print("Cannot find {} ".format(pid))
    print(f"test has {len(test_data)} papers\n delete {test_delete} papers") #5443

    save_json(whole_profile,raw_data_root,train_profile_path)
    return whole_profile

def refactor_content(paper_content,raw_data_root,train_data_path):
    new_content = defaultdict(dict)
    for pid, pinfo in paper_content.items():
        id = pid
        title = pinfo["title"]
        # There is no abstract in fanjin data
        abstract = ''
        try:
            keywords = [field["name"] for field in pinfo["fos"]]
        except:
            keywords = []
        authors= [ {"name":author["OriginalAuthor"],"org":author["OriginalAffiliation"],"idx":author["AuthorSequenceNumber"]-1}
                   for author in pinfo["authors"]]
        try:
            venue= pinfo["venue_id"]
        except:
            venue=''
        year = pinfo["year"]

        new_content[pid]={
            "id":id,
            "title":title,
            "abstract":abstract,
            "keywords":keywords,
            "authors":authors,
            "venue":venue,
            "year":year
        }
    save_json(new_content, raw_data_root, train_data_path)
    return new_content

def pretreat_unass(unass_data,processed_data_root,save_path):
    unass_list=[[item["pid"],item["name"]] for item in unass_data]
    save_json(unass_list,processed_data_root,save_path)


def get_name2aid2pid(raw_data_root, processed_data_root, name2aid2pids_path):
    ''' Merge all the information from the train set and valid set '''

    whole_pros = load_json(raw_data_root, RNDFilePathConfig.database_name2aid2pid)
    whole_pubs_info = load_json(raw_data_root, RNDFilePathConfig.database_pubs)

    train_pros = load_json(raw_data_root, RNDFilePathConfig.train_name2aid2pid)
    train_pubs_info = load_json(raw_data_root, RNDFilePathConfig.train_pubs)

    whole_pubs_info.update(train_pubs_info)
    save_json(whole_pubs_info, processed_data_root, RNDFilePathConfig.whole_pubsinfo)

    this_year = 2022

def processdata_RND(ret,version):
    v2path = version2path(version)
    pprint(v2path)
    raw_data_root = v2path['raw_data_root']
    processed_data_root = v2path["processed_data_root"]

    # Partition train set by year
    split_train2dev(data=ret,
                    processed_data_root=processed_data_root,
                    unass_ratio=0.2)

    offline_profile = load_json(processed_data_root, "train/offline_profile.json")
    offline_unass = load_json(processed_data_root, "train/offline_unass.json")
    kfold_main_func(processed_data_root,offline_profile, offline_unass, 5)




if __name__ == '__main__':
    # version = {"name": 'kdd_part', "task": 'RND', "type": 'train'}
    # version = {"name": 'kdd_part', "task": 'RND', "type": 'valid'}
    # version = {"name": 'kdd_part', "task": 'RND', "type": 'test'}
    version = {"name": 'whoiswho_part', "task": 'RND', "type": 'train'}
    v2path = version2path(version)

    #train profile  delete papers
    # whole_profile = load_json('/home/share/crossnd-202307/kddcup/name_aid_to_pids_in.json')
    # valid_data = load_json('/home/share/crossnd-202307/whoiswho/eval_na_checking_triplets_valid.json')
    # test_data = load_json('/home/share/crossnd-202307/whoiswho/eval_na_checking_triplets_test.json')
    # train_profile = delete_papers(whole_profile,valid_data,test_data,
    #                               v2path['raw_data_root'],RNDFilePathConfig.train_name2aid2pid)

    #train data
    # train_data = load_json('/home/share/crossnd-202307/whoiswho/paper_dict_mag.json')
    # train_data = refactor_content(train_data,v2path['raw_data_root'],RNDFilePathConfig.train_pubs)

    #process train data
    # train_profile = load_json('/home/hantianyi/whoiswho_thudm/whoiswho/dataset/data/whoiswho_part/RND/train/train_author.json')
    # train_data = load_json(
    #     '/home/hantianyi/whoiswho_thudm/whoiswho/dataset/data/whoiswho_part/RND/train/train_pub.json')
    # train = [train_profile , train_data]
    # processdata_RND(train,version)

    #valid test unss paper
    # pretreat_unass(valid_data,v2path['processed_data_root'],RNDFilePathConfig.unass_candi_v1_path)
    # pretreat_unass(test_data, v2path['processed_data_root'], RNDFilePathConfig.unass_candi_v2_path)

    # feature = pickle.load(open('/home/hantianyi/whoiswho_thudm/whoiswho/featureGenerator/feat/kdd_part/RND/bert/pid2aid2bert_feat.onlinev1.pkl', 'rb'))
    # print()

    # unass = load_json(v2path['processed_data_root'],RNDFilePathConfig.unass_candi_v1_path)
    # score = load_json('/home/hantianyi/whoiswho_thudm/whoiswho/training/RND_result/result_score_vote.valid.json')
    #
    # print(len(unass),len(set([i[0] for i in unass])))
    # for item in unass:
    #     pid = item[0]
    #     try:
    #         author_score = score[pid]
    #     except:
    #         print(item)






