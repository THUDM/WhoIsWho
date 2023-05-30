import os
from os.path import join
import random
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from gensim.models import word2vec

from whoiswho.config import version2path

class MetaPathGenerator:
    def __init__(self):
        self.paper_author = dict()
        self.author_paper = dict()
        self.paper_org = dict()
        self.org_paper = dict()
        self.paper_conf = dict()
        self.conf_paper = dict()

    def read_data(self, dirpath):
        temp = set()

        with open(dirpath + "/paper_org.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in self.paper_org:
                    self.paper_org[p] = []
                self.paper_org[p].append(a)
                if a not in self.org_paper:
                    self.org_paper[a] = []
                self.org_paper[a].append(p)
        temp.clear()

        with open(dirpath + "/paper_author.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in self.paper_author:
                    self.paper_author[p] = []
                self.paper_author[p].append(a)
                if a not in self.author_paper:
                    self.author_paper[a] = []
                self.author_paper[a].append(p)
        temp.clear()

        with open(dirpath + "/paper_venue.txt", encoding='utf-8') as pcfile:
            for line in pcfile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in self.paper_conf:
                    self.paper_conf[p] = []
                self.paper_conf[p].append(a)
                if a not in self.conf_paper:
                    self.conf_paper[a] = []
                self.conf_paper[a].append(p)
        temp.clear()

        print("#papers ", len(self.paper_conf))
        print("#authors", len(self.author_paper))
        print("#org_words", len(self.org_paper))
        print("#confs  ", len(self.conf_paper))

    def generate_WMRW(self, outfilename, numwalks, walklength, add_a, add_o, add_v):
        outfile = open(outfilename, 'w')
        for paper0 in self.paper_conf:
            for j in range(0, numwalks):  # wnum walks
                paper = paper0
                outline = ""
                i = 0
                while i < walklength:
                    i = i + 1
                    if add_a and paper in self.paper_author:
                        authors = self.paper_author[paper]
                        numa = len(authors)
                        authorid = random.randrange(numa)
                        author = authors[authorid]

                        papers = self.author_paper[author]
                        nump = len(papers)
                        # if nump == 1 --> self-loop
                        if nump > 1:
                            paperid = random.randrange(nump)
                            paper1 = papers[paperid]
                            while paper1 == paper:
                                paperid = random.randrange(nump)
                                paper1 = papers[paperid]
                            paper = paper1
                            outline += " " + paper

                    if add_o and paper in self.paper_org:
                        words = self.paper_org[paper]
                        numw = len(words)
                        wordid = random.randrange(numw)
                        word = words[wordid]

                        papers = self.org_paper[word]
                        nump = len(papers)
                        if nump > 1:
                            paperid = random.randrange(nump)
                            paper1 = papers[paperid]
                            while paper1 == paper:
                                paperid = random.randrange(nump)
                                paper1 = papers[paperid]
                            paper = paper1
                            outline += " " + paper

                    r_index = random.random()
                    if add_v and r_index >= 0.9:
                        if paper in self.paper_conf:
                            words = self.paper_conf[paper]
                            numw = len(words)
                            wordid = random.randrange(numw)
                            word = words[wordid]

                            papers = self.conf_paper[word]
                            nump = len(papers)
                            if nump > 1:
                                paperid = random.randrange(nump)
                                paper1 = papers[paperid]
                                while paper1 == paper:
                                    paperid = random.randrange(nump)
                                    paper1 = papers[paperid]
                                paper = paper1
                                outline += " " + paper

                outfile.write(outline + "\n")
        outfile.close()
        # print("walks done")


class RelationalFeatures:
    def __init__(self, version, processed_data_root = None , repeat_num: int = 10, num_walk: int = 5, walk_len: int = 20,
                 rw_dim: int = 100, w2v_neg: int = 25, w2v_window: int = 10):
        self.v2path = version2path(version)
        self.processed_data_root = processed_data_root

        self.repeat_num = repeat_num
        self.num_walk = num_walk
        self.walk_len = walk_len
        self.rw_dim = rw_dim
        self.w2v_neg = w2v_neg
        self.w2v_window = w2v_window

        if not processed_data_root:
            # self.raw_data_root = '../../dataset/' + self.v2path['raw_data_root']
            self.processed_data_root =  self.v2path['processed_data_root']

    def cal_relational_similarity(self, pubs, name, mode, add_a, add_o, add_v):
        mpg = MetaPathGenerator()
        mpg.read_data(join(self.processed_data_root, 'relations', mode, name))
        all_embs = []
        cp = set()
        for k in range(self.repeat_num):
            rw_path = join(self.processed_data_root, 'rw_path', mode)
            os.makedirs(rw_path, exist_ok=True)
            rw_file = join(rw_path, 'RW.txt')
            mpg.generate_WMRW(rw_file, self.num_walk, self.walk_len, add_a, add_o, add_v)
            sentences = word2vec.Text8Corpus(rw_file)
            model = word2vec.Word2Vec(sentences, size=self.rw_dim, negative=self.w2v_neg,
                                      min_count=1, window=self.w2v_window)
            embs = []
            for i, pid in enumerate(pubs):
                if pid in model:
                    embs.append(model[pid])
                else:
                    embs.append(np.zeros(100))
                    cp.add(i)
            all_embs.append(embs)
        all_embs = np.array(all_embs)

        sk_dis = np.zeros((len(pubs), len(pubs)))
        for k in range(self.repeat_num):
            sk_dis = sk_dis + pairwise_distances(all_embs[k], metric="cosine")
        sk_dis = sk_dis / self.repeat_num

        return sk_dis, cp