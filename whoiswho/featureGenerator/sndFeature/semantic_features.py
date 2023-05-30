from os.path import join
from sklearn.metrics.pairwise import pairwise_distances
from datetime import datetime
from whoiswho.config import version2path,snd_embs_path
from whoiswho.utils import load_json,load_pickle

class SemanticFeatures:
    def __init__(self, raw_data_root = None):
        self.raw_data_root = raw_data_root
        if not raw_data_root:
            self.raw_data_root ='./whoiswho'

    def cal_semantic_similarity(self, pubs, name):
        """Calculate semantic matrix of paper's by semantic feature.
        Args:
            Disambiguating name.
        Returns:
            Papers' similarity matrix (Numpy Array).
        """
        paper_embs = []
        ptext_emb = load_pickle(join(self.raw_data_root, snd_embs_path, name, 'ptext_emb.pkl'))
        tcp = load_pickle(join(self.raw_data_root, snd_embs_path, name, 'tcp.pkl'))
        for i, pid in enumerate(pubs):
            paper_embs.append(ptext_emb[pid])

        emb_dis = pairwise_distances(paper_embs, metric="cosine")
        return emb_dis, tcp