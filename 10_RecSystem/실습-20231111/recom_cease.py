# +
#
# Cornac version of CEASE (ver1.0)
#

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from cornac.models import Recommender
from cornac.exception import CornacException


# -

class CEASE(Recommender):    
    def __init__(
        self,
        name="CEASE",
        feature="item", #["user","item"]
        lamb=60,
        alpha=0.5,
        extend="additive", #["additive","collective"]
        trainable=True,
        verbose=False,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.feature = feature
        self.lamb = lamb
        self.alpha = alpha
        self.extend = extend
        self.verbose = verbose
        self.seed = seed
        if feature not in ["user","item"]:
            raise CornacException("Unsupported parameters: %s)" % self.feature)
        if extend not in ["additive","collective"]:    
            raise CornacException("Unsupported parameters: %s)" % self.extend)            
        
    def fit(self, train_set, val_set=None, user_features=None, item_features=None):       
        Recommender.fit(self, train_set, val_set)
        ratings = train_set.matrix.toarray()
        # data.Dataset.from_uir()이 FeatureModality를 지원하지 않아 .fit()의 파라미터로 item feature를 전달
        if self.feature == "user":
            ratings = ratings.T
            features = train_set.user_feature.features if user_features is None else user_features
        elif self.feature == "item":
            features = train_set.item_feature.features if item_features is None else item_features

        B_rating = self._compute_EASE(ratings, lamb=self.lamb)
        if self.extend == "additive":
            B_side = self._compute_EASE(features, lamb=self.lamb)
            B_ext = (1.0 - self.alpha) * B_rating + self.alpha * B_side
        elif self.extend == "collective":
            X_ext = np.vstack((ratings, features * self.alpha))
            B_ext = self._compute_EASE(X_ext, lamb=self.lamb)

        self.B_ext = B_ext   
        self.B_ext_csr = csr_matrix(B_ext)   
        self.U = train_set.matrix.toarray()
        self.U_csr = train_set.matrix

        return self

    def _compute_EASE(self, X, lamb):
        ''' Compute a closed-form OLS SLIM-like item-based model. (H. Steck @ WWW 2019) '''
        G = X.T @ X + lamb * np.identity((X.shape[1]))
        B = np.linalg.inv(G)
        B /= -np.diag(B)
        B[np.diag_indices(B.shape[0])] = .0
        
        return B
        
    def score(self, user_idx, item_idx=None):
        if self.feature == "user":
            pred = self.B_ext_csr.transpose()[user_idx, :].dot(self.U)
        else:    
            pred = (self.U_csr[user_idx, :].dot(self.B_ext))
        return pred[0]
