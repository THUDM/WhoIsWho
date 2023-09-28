import os
import codecs
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


def read_pubs(raw_data_root,mode):
    if mode == 'train':
        pubs = load_json(os.path.join(raw_data_root, "train", "train_pub.json"))
    elif mode == 'valid':
        pubs = load_json(os.path.join(raw_data_root, "valid", "sna_valid_pub.json"))
    elif mode == 'test':
        pubs = load_json(os.path.join(raw_data_root, 'test', 'sna_test_pub.json'))
    else:
        raise ValueError('choose right mode')

    return pubs


def read_raw_pubs(raw_data_root,mode):
    if mode == 'train':
        raw_pubs = load_json(os.path.join(raw_data_root, "train", "train_author.json"))
    elif mode == 'valid':
        raw_pubs = load_json(os.path.join(raw_data_root, "valid", "sna_valid_raw.json"))
    elif mode == 'test':
        raw_pubs = load_json(os.path.join(raw_data_root, 'test', 'sna_test_raw.json'))
    else:
        raise ValueError('choose right mode')

    return raw_pubs


def dump_name_pubs(raw_data_root,processed_data_root):
    for mode in ['train', 'valid', 'test']:
        # train valid format: name2aid2pid
        # test format: name2pid
        try:
            raw_pubs = read_raw_pubs(raw_data_root,mode)
            pubs = read_pubs(raw_data_root,mode)
            save_path = os.path.join(processed_data_root, 'names_pub',mode)
        except:
            continue

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            for name in tqdm(raw_pubs):
                name_pubs_raw = {}
                if mode == "test" or mode == 'valid':
                    for i, pid in enumerate(raw_pubs[name]):
                        name_pubs_raw[pid] = pubs[pid]
                else:
                    pids = []
                    for aid in raw_pubs[name]:
                        pids.extend(raw_pubs[name][aid])
                    for pid in pids:
                        name_pubs_raw[pid] = pubs[pid]
                save_json(name_pubs_raw, os.path.join(save_path, name+'.json'))


def dump_features_relations_to_file(raw_data_root,processed_data_root):
    """
    Generate paper features and relations by raw publication data and dump to files.
    Paper features consist of title, org, keywords. Paper relations consist of author_name, org, venue.
    """
    texts_dir = os.path.join(processed_data_root, 'extract_text')

    os.makedirs(texts_dir, exist_ok=True)
    wf = codecs.open(os.path.join(texts_dir, 'paper_features.txt'), 'w', encoding='utf-8')
    r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'

    for mode in ['train', 'valid', 'test']:
        try:
            raw_pubs = read_raw_pubs(raw_data_root,mode)
        except:
            continue

        for n, name in enumerate(tqdm(raw_pubs)): #author name
            file_path = os.path.join(processed_data_root, 'relations', mode, name)
            os.makedirs(file_path, exist_ok=True)
            coa_file = open(os.path.join(file_path, 'paper_author.txt'), 'w', encoding='utf-8')
            cov_file = open(os.path.join(file_path, 'paper_venue.txt'), 'w', encoding='utf-8')
            cot_file = open(os.path.join(file_path, 'paper_title.txt'), 'w', encoding='utf-8')
            coo_file = open(os.path.join(file_path, 'paper_org.txt'), 'w', encoding='utf-8')

            authorname_dict = {}  # maintain a author-name-dict
            pubs_dict = load_json(os.path.join(processed_data_root, 'names_pub', mode, name + '.json'))

            ori_name = name
            name, name_reverse = unify_name_order(name)

            for i, pid in enumerate(pubs_dict):
                paper_features = []
                pub = pubs_dict[pid]
                # Save title (relations)
                title = pub["title"]
                pstr = title.strip()
                pstr = pstr.lower()
                pstr = re.sub(r, ' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                pstr = pstr.split(' ')
                pstr = [word for word in pstr if len(word) > 1]
                pstr = [word for word in pstr if word not in stopwords]
                pstr = [word for word in pstr if word not in stopwords_check]
                for word in pstr:
                    cot_file.write(pid + '\t' + word + '\n')

                # Save keywords
                keyword = ""
                word_list = []
                if "keywords" in pub:
                    for word in pub["keywords"]:
                        word_list.append(word)
                    pstr = " ".join(word_list)
                    pstr = re.sub(' +', ' ', pstr)
                    keyword = pstr

                # Save org (relations)
                org = ""
                find_author = False
                for author in pub["authors"]:
                    authorname = ''.join(filter(str.isalpha, author['name'])).lower()
                    token = authorname.split(" ")

                    if len(token) == 2:
                        authorname = token[0] + token[1]
                        authorname_reverse = token[1] + token[0]
                        if authorname not in authorname_dict:
                            if authorname_reverse not in authorname_dict:
                                authorname_dict[authorname] = 1
                            else:
                                authorname = authorname_reverse
                    else:

                        authorname = authorname.replace(" ", "")

                    if authorname != name and authorname != name_reverse:
                        coa_file.write(pid + '\t' + authorname + '\n')  # current name is a name of co-author
                    else:
                        if "org" in author:
                            org = author["org"]  # current name is a name for disambiguating
                            find_author = True

                if not find_author:
                    for author in pub['authors']:
                        if match_name(author['name'], ori_name):
                            org = author['org']
                            break

                pstr = org.strip()
                pstr = pstr.lower()
                pstr = re.sub(puncs, ' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                pstr = pstr.split(' ')
                pstr = [word for word in pstr if len(word) > 1]
                pstr = [word for word in pstr if word not in stopwords]
                pstr = [word for word in pstr if word not in stopwords_extend]
                pstr = set(pstr)
                for word in pstr:
                    coo_file.write(pid + '\t' + word + '\n')

                # Save venue (relations)
                if pub["venue"]:
                    pstr = pub["venue"].strip()
                    pstr = pstr.lower()
                    pstr = re.sub(puncs, ' ', pstr)
                    pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                    pstr = pstr.split(' ')
                    pstr = [word for word in pstr if len(word) > 1]
                    pstr = [word for word in pstr if word not in stopwords]
                    pstr = [word for word in pstr if word not in stopwords_extend]
                    pstr = [word for word in pstr if word not in stopwords_check]
                    for word in pstr:
                        cov_file.write(pid + '\t' + word + '\n')
                    if len(pstr) == 0:
                        cov_file.write(pid + '\t' + 'null' + '\n')

                paper_features.append(
                    title + keyword + org
                )
                wf.write(pid + '\t' + ' '.join(paper_features) + '\n')

            coa_file.close()
            cov_file.close()
            cot_file.close()
            coo_file.close()
        print(f'Finish {mode} data extracted.')
    print(f'All paper features extracted.')
    wf.close()





def dump_plain_texts_to_file(raw_data_root,processed_data_root):
    """
    Dump raw publication data to files.
    Plain texts consist of all paper attributes and the authors' names and organizations (except year).
    """
    train_pubs_dict = load_json(os.path.join(raw_data_root, 'train', 'train_pub.json'))
    valid_pubs_dict = load_json(os.path.join(raw_data_root, 'valid', 'sna_valid_pub.json'))

    pubs_dict = {}
    pubs_dict.update(train_pubs_dict)
    pubs_dict.update(valid_pubs_dict)

    try:
        test_pubs_dict = load_json(os.path.join(raw_data_root, 'test', 'sna_test_pub.json'))
        pubs_dict.update(test_pubs_dict)
    except:
        pass

    texts_dir = os.path.join(processed_data_root, 'extract_text')
    os.makedirs(texts_dir, exist_ok=True)
    wf = codecs.open(os.path.join(texts_dir, 'plain_text.txt'), 'w', encoding='utf-8')
    r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'


    for i, pid in enumerate(tqdm(pubs_dict)):
        paper_features = []
        pub = pubs_dict[pid]

        # Save title
        title = pub["title"]
        pstr = title.strip()
        pstr = pstr.lower()
        pstr = re.sub(r, ' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        title_features = pstr

        # Save keywords
        keywd_features = ""
        word_list = []
        if "keywords" in pub:
            for word in pub["keywords"]:
                word_list.append(word)
            pstr = " ".join(word_list)
            keywd_features = pstr

        org_list = []
        for author in pub["authors"]:
            # Save org (every author's organization)
            if "org" in author:
                org = author["org"]
                pstr = org.strip()
                pstr = pstr.lower()
                pstr = re.sub(r, ' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                if pstr:
                    org_list.append(pstr)


        pstr = " ".join(org_list)
        org_features = pstr

        # Save venue
        if "venue" in pub and type(pub["venue"]) is str:
            venue = pub["venue"]
            pstr = venue.strip()
            pstr = pstr.lower()
            pstr = re.sub(r, ' ', pstr)
            pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
            venue_features = pstr

        # Save abstract
        if "abstract" in pub and type(pub["abstract"]) is str:
            abstract = pub["abstract"]
            pstr = abstract.strip()
            pstr = pstr.lower()
            pstr = re.sub(r, ' ', pstr)
            pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
            pstr = pstr.replace('\n', '')
            abstract_features = pstr

        paper_features.append(
            org_features + title_features + keywd_features +
            venue_features + abstract_features
        )
        wf.write(' '.join(paper_features) + '\n')

    print(f'All paper texts extracted.')
    wf.close()



def printInfo(dicts):
    aNum = 0
    pNum = 0
    for name, aidPid in dicts.items():
        aNum += len(aidPid)
        for aid, pids in aidPid.items():
            pNum += len(pids)

    print("#Name %d, #Author %d, #Paper %d" % (len(dicts), aNum, pNum))


def split_train2dev(data: list,processed_data_root: str, unass_ratio=0.2):
    def _get_last_n_paper(name, paper_ids, paper_info, ratio=0.2):
        cnt_unfind_author_num = 0  # 未找到作者 index 的数量
        name = cleaning_name(name)
        years = set()
        now_years = defaultdict(list)
        for pid in paper_ids:
            year = paper_info[pid].get('year', '0')
            if year == '':
                year = 0
            else:
                year = int(year)
            if year < 1500 or year > 2023:
                year = 0
            years.add(year)
            authors = paper_info[pid].get('authors', [])
            author_names = [a['name'] for a in authors]
            author_res = FindMain(name, author_names)[0]
            if len(author_res) > 0:
                aids = author_res[0][1]
            else:
                aids = get_author_index(name, author_names, False)
                if aids < 0:
                    aids = len(authors)
                    cnt_unfind_author_num += 1
            assert aids >= 0
            # if aids == len(authors):
            #     cnt_unfind_author_num += 1
            # assert aids >= 0, f"{name} 's paper {pid}"
            now_years[year].append((pid, aids,))

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
        return prof_list, unass_list, cnt_unfind_author_num

    def _split_unass(names, authors_info, papers_info, unass_info, dump_info):
        sum_unfind_author_num = 0
        unass_candi_list = []
        for name in names:
            unass_info[name] = {}
            dump_info[name] = {}
            for aid in authors_info[name]:
                # paper ids
                papers = authors_info[name][aid]
                prof_list, unass_list, cnt_unfind_num = _get_last_n_paper(name, papers, papers_info, unass_ratio)
                sum_unfind_author_num += cnt_unfind_num
                #Create profile about train_unass
                unass_info[name][aid] = [f"{p[0]}-{p[1]}" for p in unass_list if
                                         'authors' in papers_info[p[0]] and
                                         0 <= p[1] < len(papers_info[p[0]]['authors'])]
                #Create profile about train_dump
                dump_info[name][aid] = [f"{p[0]}-{p[1]}" for p in prof_list]
                # train_unass list
                for pid in unass_info[name][aid]:
                    unass_candi_list.append((pid, name))
        print('The number of papers that could not find the author name : ', sum_unfind_author_num)
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



def get_author_index_father(params):
    ''' Functions wrapped by multiprocessing  '''
    unass_pid, name, dnames = params
    author_res = FindMain(name, dnames)[0]
    if len(author_res) > 0:
        return unass_pid, author_res[0][1], 'find', name
    res = get_author_index(name, dnames, True)
    return unass_pid, res, 'doudi', name


def get_name2aid2pid(raw_data_root,processed_data_root,name2aid2pids_path):
    ''' Merge all the information from the train set and valid set '''

    whole_pros = load_json(raw_data_root, RNDFilePathConfig.database_name2aid2pid)
    whole_pubs_info = load_json(raw_data_root, RNDFilePathConfig.database_pubs)

    train_pros = load_json(raw_data_root, RNDFilePathConfig.train_name2aid2pid)
    train_pubs_info = load_json(raw_data_root, RNDFilePathConfig.train_pubs)

    whole_pubs_info.update(train_pubs_info)
    save_json(whole_pubs_info, processed_data_root, RNDFilePathConfig.whole_pubsinfo)

    this_year = 2022
    # Merge all authors under the same name.
    name_aid_pid = defaultdict(dict)
    for aid, ainfo in whole_pros.items():
        name = ainfo['name']
        pubs = ainfo['pubs']
        name_aid_pid[name][aid] = pubs
    # print(name_aid_pid)
    # Find the main author index for each paper
    key_names = list(name_aid_pid.keys())
    new_name2aid2pids = defaultdict(dict)

    for i in range(len(key_names)):
        name = key_names[i]
        aid2pid = name_aid_pid[name]
        for aid, pids in aid2pid.items():
            tmp_pubs = []
            for pid in pids:
                coauthors = [tmp['name'] for tmp in whole_pubs_info[pid]['authors']]
                coauthors = [n.replace('.', ' ').lower() for n in coauthors]
                if 'year' in whole_pubs_info[pid]:
                    year = whole_pubs_info[pid]['year']
                    year = int(year) if year != '' else this_year
                else:
                    year = this_year

                aidx = get_author_index_father((pid, name, coauthors))[1]
                if aidx < 0:
                    aidx = len(coauthors)
                new_pid = pid + '-' + str(aidx)
                tmp_pubs.append((new_pid, year))
            tmp_pubs.sort(key=lambda x: x[1], reverse=True)
            tmp_pubs = [p[0] for p in tmp_pubs]
            new_name2aid2pids[name][aid] = tmp_pubs
    printInfo(new_name2aid2pids)

    for name, aid2pid in train_pros.items():
        assert name not in key_names
        for aid, pids in aid2pid.items():
            tmp_pubs = []
            for pid in pids:
                coauthors = [tmp['name'].lower() for tmp in train_pubs_info[pid]['authors']]
                coauthors = [n.replace('.', ' ').lower() for n in coauthors]
                if 'year' in train_pubs_info[pid]:
                    year = train_pubs_info[pid]['year']
                    year = int(year) if year != '' else this_year
                else:
                    year = this_year

                aidx = get_author_index_father((pid, name, coauthors))[1]
                if aidx < 0:
                    aidx = len(coauthors)
                new_pid = pid + '-' + str(aidx)

                tmp_pubs.append((new_pid, year))
            tmp_pubs.sort(key=lambda x: x[1], reverse=True)
            tmp_pubs = [p[0] for p in tmp_pubs]
            new_name2aid2pids[name][aid] = tmp_pubs
    new_name2aid2pids = dict(new_name2aid2pids)
    printInfo(new_name2aid2pids)

    save_json(new_name2aid2pids, processed_data_root, name2aid2pids_path)


def pretreat_unass(raw_data_root,processed_data_root,unass_candi_path, unass_list_path, unass_paper_info_path):

    name_aid_pid = load_json(processed_data_root, RNDFilePathConfig.whole_name2aid2pid)

    unass_list = load_json(raw_data_root, unass_list_path)
    unass_paper_info = load_json(raw_data_root, unass_paper_info_path)
    whole_candi_names = list(name_aid_pid.keys())
    print("#Unass: %d #candiNames: %d" % (len(unass_list), len(whole_candi_names)))

    unass_candi = []
    not_match = 0

    num_thread = int(multiprocessing.cpu_count() / 1.3)
    pool = multiprocessing.Pool(num_thread)

    ins = []
    for unass_pid in unass_list:
        pid, aidx = unass_pid.split('-')
        candi_name = unass_paper_info[pid]['authors'][int(aidx)]['name']

        ins.append((unass_pid, candi_name, whole_candi_names))

    multi_res = pool.map(get_author_index_father, ins)
    pool.close()
    pool.join()
    for i in multi_res:
        pid, aidx, typ, name = i
        if aidx >= 0:
            unass_candi.append((pid, whole_candi_names[aidx]))
        else:
            not_match += 1
            print(i)   #Print the information of papers that were not found ambiguous names
    print("Matched: %d Not Match: %d" % (len(unass_candi), not_match))

    # print(unass_candi_path)
    save_json(unass_candi, processed_data_root, unass_candi_path)
    save_json(whole_candi_names, processed_data_root, 'whole_candi_names.json')


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

def processdata_SND(version: dict):
    v2path = version2path(version)
    # pprint(v2path)
    raw_data_root = v2path['raw_data_root']
    processed_data_root = v2path["processed_data_root"]

    dump_name_pubs(raw_data_root,processed_data_root)
    logger.info('Finish names')
    dump_plain_texts_to_file(raw_data_root,processed_data_root)
    logger.info('Finish extract infos')
    dump_features_relations_to_file(raw_data_root,processed_data_root)
    logger.info('Finish extract relations')


def processdata_RND(version: dict,train_data: list = []):
    v2path = version2path(version)
    # pprint(v2path)
    raw_data_root = v2path['raw_data_root']
    processed_data_root = v2path["processed_data_root"]

    # train_data stores the content of train_author.json and train_pub.json
    if not train_data:
        train_pros = load_json(raw_data_root, RNDFilePathConfig.train_name2aid2pid)
        train_pubs_info = load_json(raw_data_root, RNDFilePathConfig.train_pubs)
        train_data.append(train_pros)
        train_data.append(train_pubs_info)

    # Partition train set by year
    split_train2dev(data=train_data,
                    processed_data_root=processed_data_root,
                    unass_ratio=0.2)

    offline_profile = load_json(processed_data_root, "train/offline_profile.json")
    offline_unass = load_json(processed_data_root, "train/offline_unass.json")
    kfold_main_func(processed_data_root,offline_profile, offline_unass, 5)

    logger.info('Begin Combine Data')
    # train+valid name2aid2pid
    get_name2aid2pid(raw_data_root,processed_data_root,name2aid2pids_path=RNDFilePathConfig.whole_name2aid2pid)
    logger.info('Finish Combine Data')

    # Papers that have not been assigned
    try:
        pretreat_unass(raw_data_root,processed_data_root,RNDFilePathConfig.unass_candi_v1_path, "valid/cna_valid_unass.json",
                       "valid/cna_valid_unass_pub.json")
    except:
        logger.error('Error in Pretreat Valid')

    try:
        pretreat_unass(raw_data_root,processed_data_root,RNDFilePathConfig.unass_candi_v2_path, "test/cna_test_unass.json",
                       "test/cna_test_unass_pub.json")
    except:
        logger.error('Error in Pretreat Test')


if __name__ == '__main__':
    # train, version = load_utils.LoadData(name="v3", type="train", task='RND')
    # processdata_RND(version=version)

    version = load_utils.LoadData(name="v3", type="train", task='SND',just_version=True)
    processdata_SND(version=version)

