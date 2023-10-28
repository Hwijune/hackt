# <font color='tomato'><font color="#CC3D3D"><p>
# # EASE: Embarrassingly Shallow Autoencoders for Sparse Data

# $$ Loss = ||X - XW^T W||_F^2 + \lambda ||W||_F^2 $$  
# - 이 논문에서 제안하는 autoencoder는 전통적인 deep autoencoder와는 달리 깊이가 얕으나(latent층이 없음), sparse한 데이터에 대해 잘 동작함.
# - $λ$ 값이 크면 가중치 행렬 $W$에 대한 regularization이 강해져 모델이 더욱 단순해지며, $λ$ 값이 작으면 regularization의 영향이 줄어들어 모델이 데이터에 더욱 복잡하게 적합됨   
# (이 코드는 https://github.com/Darel13712/ease_rec을 참고하여 일부 수정한 것임)

# ##### Global Setting & Imports

DEFAULT_USER_COL = 'resume_seq'
DEFAULT_ITEM_COL = 'recruitment_seq'
DEFAULT_RATING_COL = 'rating'
DEFAULT_PREDICTION_COL = 'prediction'
TOP_K = 5     # 추천 아이템 수
LAMBDA = 300  # EASE의 유일한 Hyper-parameter

from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count


# ##### Model Building

class EASE:
    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def _get_users_and_items(self, df):
        users = self.user_enc.fit_transform(df.loc[:, DEFAULT_USER_COL])
        items = self.item_enc.fit_transform(df.loc[:, DEFAULT_ITEM_COL])
        return users, items

    def fit(self, df, reg_lambda = 0.5, implicit=True):
        """
        df: pandas.DataFrame with columns user_id, item_id and (rating)
        lambda_: l2-regularization term
        implicit: if True, ratings are ignored and taken as 1, else normalized ratings are used
        """
        users, items = self._get_users_and_items(df)
        values = (
            np.ones(df.shape[0])
            if implicit
            else df[DEFAULT_RATING_COL].to_numpy() / df[DEFAULT_RATING_COL].max()
        )

        X = csr_matrix((values, (users, items)))
        self.X = X

        G = X.T.dot(X).toarray()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += reg_lambda
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0

        self.B = B
        self.pred = X.dot(B)

    def predict(self, train, users, items, k):
        items = self.item_enc.transform(items)
        dd = train.loc[train[DEFAULT_USER_COL].isin(users)]
        dd['ci'] = self.item_enc.transform(dd[DEFAULT_ITEM_COL])
        dd['cu'] = self.user_enc.transform(dd[DEFAULT_USER_COL])
        g = dd.groupby('cu')
        with Pool(cpu_count()) as p:
            user_preds = p.starmap(
                self.predict_for_user,
                [(user, group, self.pred[user, :], items, k) for user, group in g],
            )
        df = pd.concat(user_preds)
        df[DEFAULT_ITEM_COL] = self.item_enc.inverse_transform(df[DEFAULT_ITEM_COL])
        df[DEFAULT_USER_COL] = self.user_enc.inverse_transform(df[DEFAULT_USER_COL])
        return df

    @staticmethod
    def predict_for_user(user, group, pred, items, k):
        watched = set(group['ci'])
        candidates = [item for item in items if item not in watched]
        pred = np.take(pred, candidates)
        res = np.argpartition(pred, -k)[-k:]
        r = pd.DataFrame(
            {
                DEFAULT_USER_COL: [user] * len(res),
                DEFAULT_ITEM_COL: np.take(candidates, res),
                DEFAULT_PREDICTION_COL: np.take(pred, res),
            }
        ).sort_values(DEFAULT_PREDICTION_COL, ascending=False)
        return r


# #### Data Loading

df = pd.read_csv('apply_train.csv'); df

# ##### Training & Recommendation

# +
# Fitting
model = EASE()
model.fit(df, reg_lambda=LAMBDA)

# Top-k item generation
top_k = model.predict(df, users=df[DEFAULT_USER_COL].unique(), items=df[DEFAULT_ITEM_COL].unique(), k=TOP_K)

# Make submissions
top_k = top_k.drop(DEFAULT_PREDICTION_COL, axis=1).sort_values(by=DEFAULT_USER_COL)
top_k.to_csv('submission_ease.csv', index=False)
# -

# <font color='tomato'><font color="#CC3D3D"><p>
# # End
