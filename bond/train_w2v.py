import re
import params
import numpy as np
from tqdm import tqdm
from os.path import join
from params import set_params
from gensim.models import word2vec
from datetime import datetime
from dataset.load_data import load_json, dump_data
from dataset.save_results import check_mkdir
from character.match_name import match_name
from dataset.preprocess_SND import read_raw_pubs

start_time = datetime.now()
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



def extract_text_save(pub_files, out_file):
    """
    extract [org, title, abstract, venue, keywords] from train/valid/test files.
    """
    r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
    f_out = open(out_file, 'w', encoding='utf-8')
    for file in pub_files:
        pubs = load_json(file)
        for pub in tqdm(pubs.values()):
            for author in pub["authors"]:
                if "org" in author:
                    org = author["org"]
                    pstr = org.strip()
                    pstr = pstr.lower()
                    pstr = re.sub(r, ' ', pstr)
                    pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                    f_out.write(pstr + '\n')

            title = pub["title"]
            pstr = title.strip()
            pstr = pstr.lower()
            pstr = re.sub(r, ' ', pstr)
            pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
            f_out.write(pstr + '\n')

            if "abstract" in pub and type(pub["abstract"]) is str:
                abstract = pub["abstract"]
                pstr = abstract.strip()
                pstr = pstr.lower()
                pstr = re.sub(r, ' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                f_out.write(pstr + '\n')

            if "venue" in pub and type(pub["venue"]) is str:
                venue = pub["venue"]
                pstr = venue.strip()
                pstr = pstr.lower()
                pstr = re.sub(r, ' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                f_out.write(pstr + '\n')
            
            word_list = []
            if "keywords" in pub:
                for word in pub["keywords"]:
                    word_list.append(word)
                pstr = " ".join(word_list)
                f_out.write(pstr + '\n')

        print(f'File {file} text extracted.')
    f_out.close()


def dump_corpus():
    """
    dump texts for word2vec trainning.
    """
    train_pub = join(args.save_path, 'src', 'train', 'train_pub.json')
    valid_pub = join(args.save_path, 'src', 'sna-valid', 'sna_valid_pub.json')
    test_pub = join(args.save_path, 'src', 'sna-test', 'sna_test_pub.json')
    pub_files = [train_pub, valid_pub, test_pub]
    texts_dir = join(args.save_path, 'extract_text')
    check_mkdir(texts_dir)
    extract_text_save(pub_files, join(texts_dir, 'train_valid_test.txt'))


def train_w2v_model(ft_dim):
    model_path = join(args.save_path, 'w2v_model')
    check_mkdir(model_path)
    texts_dir = join(args.save_path, 'extract_text')
    sentences = word2vec.Text8Corpus(join(texts_dir, 'train_valid_test.txt'))
    model = word2vec.Word2Vec(sentences, vector_size=ft_dim, negative=5, min_count=5, window=5)
    model.save(join(model_path, f'w2v_{ft_dim}.model'))
    print(f'Finish word2vec training.')


def dump_paper_emb(model_name, ft_dim):
    """
    dump paper's [title, org, keywords] average word-embedding as semantic feature.
    """
    model_path = join(args.save_path, 'w2v_model')
    w2v_model = word2vec.Word2Vec.load(join(model_path, f'{model_name}.model'))

    for mode in ['train', 'valid', 'test']:
        raw_pubs = read_raw_pubs(mode)
        for n, name in tqdm(enumerate(raw_pubs)):
            name_pubs = load_json(join(args.save_path, 'names_pub', mode, name + '.json'))
            text_feature_path = join(args.save_path, f'paper_emb', mode, name)
            check_mkdir(text_feature_path)

            ori_name = name
            taken = name.split("_")
            name = taken[0] + taken[1]
            name_reverse = taken[1] + taken[0]
            if len(taken) > 2:
                name = taken[0] + taken[1] + taken[2]
                name_reverse = taken[2] + taken[0] + taken[1]

            authorname_dict = {}

            ptext_emb = {}
            tcp = set()

            for i, pid in enumerate(name_pubs):

                pub = name_pubs[pid]
                # save authors
                org = ""
                find_author = False
                for author in pub["authors"]:
                    authorname = ''.join(filter(str.isalpha, author['name'])).lower()

                    taken = authorname.split(" ")
                    if len(taken) == 2:
                        authorname = taken[0] + taken[1]
                        authorname_reverse = taken[1] + taken[0]

                        if authorname not in authorname_dict:
                            if authorname_reverse not in authorname_dict:
                                authorname_dict[authorname] = 1
                            else:
                                authorname = authorname_reverse
                    else:
                        authorname = authorname.replace(" ", "")

                    if authorname != name and authorname != name_reverse:
                        pass
                    else:
                        if "org" in author:
                            org = author["org"]
                            find_author = True
                if not find_author:
                    for author in pub['authors']:
                        if match_name(author['name'], ori_name):
                            org = author['org']
                            break

                pstr = ""
                keyword = ""
                if "keywords" in pub:
                    for word in pub["keywords"]:
                        keyword = keyword + word + " "


                pstr = pub["title"] + " " + keyword + " " + org
                pstr = pstr.strip()
                pstr = pstr.lower()
                pstr = re.sub(puncs, ' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                pstr = pstr.split(' ')
                pstr = [word for word in pstr if len(word) > 2]
                pstr = [word for word in pstr if word not in stopwords]
                pstr = [word for word in pstr if word not in stopwords_extend]

                pstr = [word for word in pstr if word not in stopwords_check]


                words_vec = []
                for word in pstr:
                    # if word in w2v_model:
                    if word in w2v_model.wv:
                        # words_vec.append(w2v_model[word])
                        words_vec.append(w2v_model.wv[word])
                if len(words_vec) < 1:
                    words_vec.append(np.zeros(ft_dim))
                    tcp.add(i)

                ptext_emb[pid] = np.mean(words_vec, 0)

            dump_data(ptext_emb, join(text_feature_path, 'ptext_emb.pkl'))
            dump_data(tcp, join(text_feature_path, 'tcp.pkl'))


if __name__ == "__main__":
    """
    train w2v model and save paper-embedding.
    """
    
    ft_dim = 256
    
    # dump_corpus()
    # train_w2v_model(ft_dim)
    dump_paper_emb(model_name=f"w2v_{ft_dim}", ft_dim=ft_dim)
    
    print('done', datetime.now()-start_time)