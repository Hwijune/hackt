# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from cornac.models import Recommender
import cornac
import tensorflow as tf
import numpy as np
import numbers
import os
from tqdm.auto import trange


# +
class CornacException(Exception):
    """Exception base class to extend from
    Original source code - "https://github.com/PreferredAI/cornac/blob/master/cornac/utils/common.py"
    """

    pass


class ScoreException(CornacException):
    """Exception raised in score function when facing unknowns

    """

    pass


def clip_by_bound(values, lower_bound, upper_bound):
    """enforce values to lie in a [lower_bound, upper_bound] range

    Args:
        values (np.array): values to be clipped.
        lower_bound (scalar): Lower bound.
        upper_bound (scalar): Upper bound.

    Returns:
        np.array: Clipped values in range [lower_bound, upper_bound]
    """
    values = np.where(values > upper_bound, upper_bound, values)
    values = np.where(values < lower_bound, lower_bound, values)

    return values


def get_rng(seed):
    """Return a RandomState of Numpy.
    Original source code - "https://github.com/PreferredAI/cornac/blob/master/cornac/utils/common.py"
    If seed is None, use RandomState singleton from numpy.
    If seed is an integer, create a RandomState from that seed.
    If seed is already a RandomState, just return it.
    """

    if seed is None:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('{} can not be used to create a numpy.random.RandomState'.format(seed))


def uniform(shape=None, low=0.0, high=1.0, random_state=None, dtype=np.float32):
    """Draw samples from a uniform distribution.
    Original source code - "https://github.com/PreferredAI/cornac/blob/master/cornac/utils/init_utils.py"
    Args:
        shape (int or tuple of ints): Output shape. If shape is ``None`` (default), a single value is returned.
        low (float or array_like of floats): Lower boundary of the output interval. The default value is 0.
        high (float or array_like of floats):  Upper boundary of the output interval. The default value is 1.0.
        random_state (int or np.random.RandomState): If an integer is given, it will be used as seed value for creating a RandomState.
        dtype (str or dtype): Returned data-type for the output array.

    Returns:
        ndarray or scalar: Drawn samples from the parameterized uniform distribution.
    """
    return get_rng(random_state).uniform(low, high, shape).astype(dtype)


def xavier_uniform(shape, random_state=None, dtype=np.float32):
    """Return a numpy array by performing 'Xavier' initializer also known as 'Glorot' initializer on Uniform distribution.
    Original source code - "https://github.com/PreferredAI/cornac/blob/master/cornac/utils/init_utils.py"
    Args:
        shape (int or tuple of ints): Output shape.
        random_state (int or np.random.RandomState): If an integer is given, it will be used as seed value for creating a RandomState.
        dtype (str or dtype): Returned data-type for the output array.

    Returns:
        ndarray: Output matrix.
    """

    assert len(shape) == 2  # only support matrix
    std = np.sqrt(2.0 / np.sum(shape))
    limit = np.sqrt(3.0) * std
    return uniform(shape, -limit, limit, random_state, dtype)


def weight_user_oriented(data, alpha):
    """Return weight vector based on user-oriented strategy (Pan, Rong, et al. One-class collaborative filtering. 2008)

    Args:
        data (pd.DataFrame): Implicit Feedback DataFrame which is binarized.
        alpha (scalar): Hyper-parameter that controls the strength of weights

    Returns:
        np.array: (n_users, ) size weight array
    """
    return data.iloc[:,0].value_counts().sort_index().to_numpy() * alpha


def weight_item_oriented(data, alpha):
    """Return weight vector based on item-oriented strategy (Pan, Rong, et al. One-class collaborative filtering. 2008)

    Args:
        data (pd.DataFrame): Implicit Feedback DataFrame which is binarized.
        alpha (scalar): Hyper-parameter that controls the strength of weights

    Returns:
        np.array: (n_items, ) size weight array
    """
    n_users = data.iloc[:,0].nunique()
    return alpha * (n_users - data.iloc[:,1].value_counts().sort_index().to_numpy())


def weight_item_popularity(data, alpha, c_0):
    """Return weight vector based on item-popularity strategy
        (He, Xiangnan, et al. Fast matrix factorization for online recommendation with implicit feedback. 2016)

    Args:
        data (pd.DataFrame): Implicit Feedback DataFrame which is binarized.
        alpha (scalar): Hyper-parameter that controls the significance level of popular items over unpopular ones.
            If alpha > 1,  the difference of weights between popular items and unpopular ones is strengthened.
            If 0 < alpha < 1 the difference is weakened and the weight of popular items is suppressed.
        c_0: Hyper-parameter that determines the overall weight of unobserved instances.

    Returns:
         np.array: (n_items, ) size weight array
    """
    u_j = data.iloc[:,1].value_counts().sort_index().to_numpy()
    f_vec = u_j / u_j.sum()
    f_alpha_vec = f_vec ** alpha
    return c_0 * (f_alpha_vec / f_alpha_vec.sum())



# -

class Model(tf.keras.Model):
    def __init__(self, n_users, n_items, k, lambda_u, lambda_v, lr, U, V):
        super(Model, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lr = lr  # learning rate
        self.k = k  # latent dimension
        self.U = tf.Variable(U, dtype=tf.float32, trainable=True)
        self.V = tf.Variable(V, dtype=tf.float32, trainable=True)
        self.optimizer = tf.optimizers.Adam(learning_rate=self.lr)

    def call(self, ratings, C, item_ids):
        V_batch = tf.gather(self.V, item_ids)

        predictions = tf.matmul(self.U, V_batch, transpose_b=True)
        squared_error = tf.square(ratings - predictions)
        loss_1 = tf.reduce_sum(tf.multiply(C, squared_error))
        loss_2 = self.lambda_u * tf.nn.l2_loss(self.U) + self.lambda_v * tf.nn.l2_loss(V_batch)

        return loss_1 + loss_2

    def train_step(self, ratings, C, item_ids):
        with tf.GradientTape() as tape:
            loss = self.call(ratings, C, item_ids)

        grads = tape.gradient(loss, [self.U, self.V])
        clipped_grads = [tf.clip_by_value(grad, -5., 5.) for grad in grads]

        self.optimizer.apply_gradients(zip(clipped_grads, [self.U, self.V]))
        return loss


# +
class WRMF(Recommender):
    """Weighted Matrix Factorization.
    original source code - "https://github.com/PreferredAI/cornac/blob/master/cornac/models/wmf/recom_wmf.py"

    Parameters
    ----------
    name: string, default: 'WMF'
        The name of the recommender model.

    weight_strategy: string, default: 'uniform_pos'
        Weighting strategy - 'uniform_pos', 'uniform_neg', 'user_oriented', 'item-oriented', 'item_popularity'

    alpha: scalar, default: 1
        Hyper-parameter that controls the strength of weights

    c_0: scalar, default: 1
        Hyper-parameter of item-popularity strategy that determines the overall weight of unobserved instances.

    k: int, optional, default: 200
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    learning_rate: float, optional, default: 0.001
        The learning rate for AdamOptimizer.

    lambda_u: float, optional, default: 0.01
        The regularization parameter for users.

    lambda_v: float, optional, default: 0.01
        The regularization parameter for items.

    a: float, optional, default: 1
        The confidence of observed ratings.

    b: float, optional, default: 0.01
        The confidence of unseen ratings.

    batch_size: int, optional, default: 128
        The batch size for SGD.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already
        pre-trained (U and V are not None).

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'U':U, 'V':V}

        U: ndarray, shape (n_users,k)
            The user latent factors, optional initialization via init_params.

        V: ndarray, shape (n_items,k)
            The item latent factors, optional initialization via init_params.

    seed: int, optional, default: None
        Random seed for weight initialization.

    References
    ----------
    * Hu, Y., Koren, Y., & Volinsky, C. (2008, December). Collaborative filtering for implicit feedback datasets. \
    In 2008 Eighth IEEE International Conference on Data Mining (pp. 263-272).

    * Pan, R., Zhou, Y., Cao, B., Liu, N. N., Lukose, R., Scholz, M., & Yang, Q. (2008, December). \
    One-class collaborative filtering. In 2008 Eighth IEEE International Conference on Data Mining (pp. 502-511).

    """

    def __init__(
            self,
            data,
            name="WRMF",
            weight_strategy="uniform_pos",
            alpha=1,
            c_0=1,
            k=200,
            lambda_u=0.01,
            lambda_v=0.01,
            learning_rate=0.001,
            batch_size=128,
            max_iter=100,
            trainable=True,
            verbose=True,
            init_params=None,
            seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.strategy = weight_strategy
        self.data = data
        self.alpha = alpha
        self.c_0 = c_0

        if self.strategy == "user_oriented":
            self.weights = weight_user_oriented(self.data, self.alpha)
        elif self.strategy == "item_oriented":
            self.weights = weight_item_oriented(self.data, self.alpha)
        elif self.strategy == "item_popularity":
            self.weights = weight_item_popularity(self.data, self.alpha, self.c_0)
        elif self.strategy == "uniform_pos" or self.strategy == "uniform_neg":
            self.weights = np.ones(shape=data.iloc[:,1].nunique()) * alpha
        else:
            print('wrong strategy')

        print('maximum of weights={}, minimum={}'.format(self.weights.max(), self.weights.min()))
        self.learning_rate = learning_rate
        self.name = name
        self.init_params = init_params
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.seed = seed

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.U = self.init_params.get("U", None)
        self.V = self.init_params.get("V", None)

    def _init(self):
        rng = get_rng(self.seed)
        n_users, n_items = self.train_set.num_users, self.train_set.num_items

        if self.U is None:
            self.U = xavier_uniform((n_users, self.k), rng)
        if self.V is None:
            self.V = xavier_uniform((n_items, self.k), rng)

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)

        self._init()

        if self.trainable:
            self._fit_cf()

        return self

    def _fit_cf(self):
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        R = self.train_set.csc_matrix  # csc for efficient slicing over items
        n_users, n_items = self.train_set.num_users, self.train_set.num_items

        # Initialize model
        model = Model(
            n_users=n_users,
            n_items=n_items,
            k=self.k,
            lambda_u=self.lambda_u,
            lambda_v=self.lambda_v,
            lr=self.learning_rate,
            U=self.U,
            V=self.V,
        )

        loop = trange(self.max_iter, disable=not self.verbose)
        for _ in loop:
            sum_loss = 0
            count = 0
            for i, batch_ids in enumerate(
                    self.train_set.item_iter(self.batch_size, shuffle=True)
            ):
                batch_R = R[:, batch_ids]

                if self.strategy == "uniform_pos":
                    batch_C = np.ones(batch_R.shape)
                    batch_C[batch_R.nonzero()] = self.alpha
                elif self.strategy == "uniform_neg":
                    batch_C = np.zeros(batch_R.shape) + self.alpha
                    batch_C[batch_R.nonzero()] = 1
                else:
                    if self.strategy == "user_oriented":
                        weight_vec = self.weights.reshape(batch_R.shape[0], -1)
                    else:
                        weight_vec = self.weights[batch_ids].reshape(-1, len(batch_ids))

                    batch_C = np.zeros(batch_R.shape) + weight_vec
                    batch_C[batch_R.nonzero()] = 1

#                feed_dict = {
#                    model.ratings: batch_R.A,
#                    model.C: batch_C,
#                    model.item_ids: batch_ids,
#                }

                loss = model.train_step(batch_R.A, batch_C, batch_ids)
                sum_loss += loss
                count += len(batch_ids)
                if i % 10 == 0:
                    loop.set_postfix(loss=(sum_loss / count))

            self.U, self.V = model.U.numpy(), model.V.numpy()

        if self.verbose:
            print("Learning completed!")
            
    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items
        """
        if item_idx is None:
            if self.train_set.is_unk_user(user_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )

            known_item_scores = self.V.dot(self.U[user_idx, :])
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                    item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )
            user_pred = self.V[item_idx, :].dot(self.U[user_idx, :])
            return user_pred
