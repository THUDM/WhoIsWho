from os.path import join
from sklearn.metrics.pairwise import pairwise_distances
import sys
sys.path.append('../../../')
from whoiswho.config import version2path,snd_embs_path
from whoiswho.utils import load_json,load_pickle

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