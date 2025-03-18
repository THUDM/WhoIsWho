import os
import re
from tqdm import tqdm
from os.path import join

from params import set_params
from dataset.dump_graph import build_graph
from dataset.load_data import load_json
from dataset.save_results import dump_json, check_mkdir
from character.match_name import match_name

args = set_params()

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


def read_pubinfo(mode):
    """
    Read pubs' meta-information.
    """
    base = join(args.save_path, "src")

    if mode == 'train':
        pubs = load_json(join(base, "train", "train_pub.json"))
    elif mode == 'valid':
        pubs = load_json(join(base, "sna-valid", "sna_valid_pub.json"))
    elif mode == 'test':
        pubs = load_json(join(base, 'sna-test', 'sna_test_pub.json'))
    else:
        raise ValueError('choose right mode')
    
    return pubs


def read_raw_pubs(mode):
    """
    Read raw pubs.
    """
    base = join(args.save_path, "src")

    if mode == 'train':
        raw_pubs = load_json(join(base, "train", "train_author.json"))
    elif mode == 'valid':
        raw_pubs = load_json(join(base, "sna-valid", "sna_valid_raw.json"))
    elif mode == 'test':
        raw_pubs = load_json(join(base, "sna-test", "sna_test_raw.json"))
    else:
        raise ValueError('choose right mode')
    
    return raw_pubs


def dump_name_pubs():
    """
    Split publications informations by {name} and dump files as {name}.json

    """
    for mode in ['train', 'valid', 'test']:
        raw_pubs = read_raw_pubs(mode)
        pub_info = read_pubinfo(mode)
        file_path = join(args.save_path, 'names_pub', mode)
        if not os.path.exists(file_path):
            check_mkdir(file_path)
        for name in tqdm(raw_pubs):
            name_pubs_raw = {}
            if mode != "train":
                for i, pid in enumerate(raw_pubs[name]):
                    name_pubs_raw[pid] = pub_info[pid]
            else:
                pids = []
                for aid in raw_pubs[name]:
                    pids.extend(raw_pubs[name][aid])
                for pid in pids:
                    name_pubs_raw[pid] = pub_info[pid]

            dump_json(name_pubs_raw, join(file_path, name+'.json'), indent=4)



def unify_name_order(name):
    """
    unifying different orders of name.
    Args:
        name
    Returns:
        name and reversed name
    """
    token = name.split("_")
    name = token[0] + token[1]
    name_reverse = token[1] + token[0]
    if len(token) > 2:
        name = token[0] + token[1] + token[2]
        name_reverse = token[2] + token[0] + token[1]

    return name, name_reverse


def dump_features_relations_to_file():
    """
    Generate paper features and relations by raw publication data and dump to files.
    Paper features consist of title, org, keywords. Paper relations consist of author_name, org, venue.
    """
    r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'

    for mode in ['train', 'valid', 'test']:
        raw_pubs = read_raw_pubs(mode)
        for n, name in tqdm(enumerate(raw_pubs)):

            file_path = join(args.save_path, 'relations', mode, name)
            check_mkdir(file_path)
            coa_file = open(join(file_path, 'paper_author.txt'), 'w', encoding='utf-8')
            cov_file = open(join(file_path, 'paper_venue.txt'), 'w', encoding='utf-8')
            cot_file = open(join(file_path, 'paper_title.txt'), 'w', encoding='utf-8')
            coo_file = open(join(file_path, 'paper_org.txt'), 'w', encoding='utf-8')

            authorname_dict = {} # maintain a author-name-dict
            pubs_dict = load_json(join(args.save_path, 'names_pub', mode, name+'.json'))

            ori_name = name
            name, name_reverse = unify_name_order(name)

            for i, pid in enumerate(pubs_dict):
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

            coa_file.close()
            cov_file.close()
            cot_file.close()
            coo_file.close()
        print(f'Finish {mode} data extracted.')
    print(f'All paper features extracted.')


if __name__ == "__main__":
    """
    some pre-processing
    """
    dump_name_pubs()
    # dump_features_relations_to_file()
    # build_graph()
