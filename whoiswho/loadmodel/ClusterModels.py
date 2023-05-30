import numpy as np
from sklearn.cluster import DBSCAN


class DBSCANModel:
    def __init__(self,
                 db_eps = 0.2,
                 db_min = 4):
        self.db_eps = db_eps
        self.db_min = db_min

    def fit(self,feature):
        """Cluster papers by similarity matrix.
            Args:
                similarity matrix (Numpy Array).

            Returns:
                clustering labels (Numpy Array).
            """
        pred = DBSCAN(eps=self.db_eps, min_samples=self.db_min, metric='precomputed').fit_predict(feature)
        pred = np.array(pred)

        return pred