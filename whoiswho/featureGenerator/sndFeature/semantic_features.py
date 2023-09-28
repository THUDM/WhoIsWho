from sklearn.metrics.pairwise import pairwise_distances
import sys
sys.path.append('../../../')
import os
import re
from os.path import join
from gensim.models import word2vec
from datetime import datetime
from tqdm import tqdm
import numpy as np
from whoiswho.dataset.data_process import read_raw_pubs
from whoiswho.utils import load_json, save_pickle
from whoiswho.character.match_name import  match_name
from whoiswho.config import version2path,snd_embs_path
from whoiswho.utils import load_json,load_pickle

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


def train_w2v_model(processed_data_root):
    texts_dir = join(processed_data_root, 'extract_text')
    sentences = word2vec.Text8Corpus(join(texts_dir, 'plain_text.txt'))

    model_path = join(processed_data_root, 'out', 'model')
    os.makedirs(model_path, exist_ok=True)
    model = word2vec.Word2Vec(sentences, size=100, negative=5, min_count=5, window=5)
    model.save(join(model_path, 'tvt.model'))
    print(f'Finish word2vec training.')


def dump_paper_emb(raw_data_root,processed_data_root,model_name):
    """
    dump paper's [title, org, keywords] average word-embedding as semantic feature.
    """
    model_path = join(processed_data_root, 'out','model')
    w2v_model = word2vec.Word2Vec.load(join(model_path, f'{model_name}.model'))

    for mode in ['train', 'valid', 'test']:
        raw_pubs = read_raw_pubs(raw_data_root,mode)
        for n, name in enumerate(tqdm(raw_pubs)):
            name_pubs = load_json(join(processed_data_root, 'names_pub', mode, name + '.json'))
            text_feature_path = join(processed_data_root, 'snd-embs', name)
            os.makedirs(text_feature_path, exist_ok=True)

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
                    if word in w2v_model:
                        words_vec.append(w2v_model[word])
                if len(words_vec) < 1:
                    words_vec.append(np.zeros(100))
                    tcp.add(i)

                ptext_emb[pid] = np.mean(words_vec, 0)

            save_pickle(ptext_emb, join(text_feature_path, 'ptext_emb.pkl'))
            save_pickle(tcp, join(text_feature_path, 'tcp.pkl'))


class SemanticFeatures:
    def __init__(self):
        self.snd_embs_path = snd_embs_path

    def cal_semantic_similarity(self, pubs, name):
        """Calculate semantic matrix of paper's by semantic feature.
        Args:
            Disambiguating name.
        Returns:
            Papers' similarity matrix (Numpy Array).
        """
        paper_embs = []
        ptext_emb = load_pickle(join(self.snd_embs_path, name, 'ptext_emb.pkl'))
        for i, pid in enumerate(pubs):
            paper_embs.append(ptext_emb[pid])
        emb_dis = pairwise_distances(paper_embs, metric="cosine")

        sem_outliers = load_pickle(join(self.snd_embs_path, name, 'tcp.pkl'))
        return emb_dis, sem_outliers


if __name__ == "__main__":
    """
    train w2v model and save paper-embedding.
    """
    version = {"name": 'v3', "task": 'SND', "type": 'train'}
    v2path = version2path(version)
    raw_data_root = v2path['raw_data_root']
    processed_data_root = v2path["processed_data_root"]
    # train_w2v_model(processed_data_root)
    dump_paper_emb(raw_data_root,processed_data_root,model_name="tvt")