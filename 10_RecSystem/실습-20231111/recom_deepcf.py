# +
#
# Cornac version of DeepCF (ver1.0)
#

from cornac.models import Recommender
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import initializers
from keras.models import Model
from keras.layers import Input, Dense, Flatten, concatenate, Lambda, multiply
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras import backend as K
from tqdm.auto import trange  # progress bar


# -

class DeepCF(Recommender):    
    def __init__(
        self,
        name="DeepCF",
        userlayers=[512,64],
        itemlayers=[1024,64],
        layers=[512,256,128,64],
        act_fn="relu",
        n_epochs=10,
        batch_size=256,
        num_neg=4,
        learning_rate=0.001,
        learner="adam",
        backend="tensorflow",
        early_stopping=None,  # Not yet implemented
        trainable=True,
        verbose=False,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.userlayers = userlayers
        self.itemlayers = itemlayers
        self.layers = layers
        self.act_fn = act_fn
        self.num_epochs = n_epochs
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.lr = learning_rate
        self.learner = learner
        self.backend = backend
        self.early_stopping = early_stopping  # Not yet implemented
        self.seed = seed
        
    def fit(self, train_set, val_set=None):       
        Recommender.fit(self, train_set, val_set)
        train = train_set.matrix.toarray()  # Cornac 데이터형식의 평점행렬을 numpy array 형식으로
        self.num_items = train_set.num_items # number of items 

        ##### Build a Model #####
        dmf_num_layer = len(self.userlayers) # number of layers in the DMF
        mlp_num_layer = len(self.layers)     # number of layers in the MLP
        user_matrix = K.constant(train)      # sparse matrix => dense matrix => TF tensor
        item_matrix = K.constant(train.T)    # sparse matrix => dense matrix => transposed matrix => TF tensor
        # Input variables
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')
        # Embedding layer
        user_rating = Lambda(lambda x: tf.gather(user_matrix, x))(user_input) # 위 셀의 tf.gather 기능 참조
        item_rating = Lambda(lambda x: tf.gather(item_matrix, x))(item_input)
        user_rating = Flatten()(user_rating) # 2D Tensor (1, num_items) => 1D Tensor (num_items,)
        item_rating = Flatten()(item_rating) # 2D Tensor (1, num_users) => 1D Tensor (num_users,)
        # DMF part
        dmf_user_latent = Dense(self.userlayers[0], activation="linear" , name='user_layer0')(user_rating)
        dmf_item_latent = Dense(self.itemlayers[0], activation="linear" , name='item_layer0')(item_rating)
        for idx in range(1, dmf_num_layer):
            dmf_user_latent = Dense(self.userlayers[idx], activation=self.act_fn, name='user_layer%d' % idx)(dmf_user_latent)
            dmf_item_latent = Dense(self.itemlayers[idx], activation=self.act_fn, name='item_layer%d' % idx)(dmf_item_latent)
        dmf_vector = multiply([dmf_user_latent, dmf_item_latent])  # element-wise product
        # MLP part 
        mlp_user_latent = Dense(self.layers[0]//2, activation="linear" , name='user_embedding')(user_rating)
        mlp_item_latent  = Dense(self.layers[0]//2, activation="linear" , name='item_embedding')(item_rating)
        mlp_vector = concatenate([mlp_user_latent, mlp_item_latent])
        for idx in range(1, mlp_num_layer):
            mlp_vector = Dense(self.layers[idx], activation=self.act_fn, name="layer%d" % idx)(mlp_vector)
        # Concatenate DMF and MLP parts
        predict_vector = concatenate([dmf_vector, mlp_vector])
        # Final prediction layer
        prediction = Dense(1, activation='sigmoid', kernel_initializer=initializers.lecun_normal(),
                           name="prediction")(predict_vector)
        # Full model
        model = Model(inputs=[user_input, item_input], outputs=prediction)
        
        ##### Set loss, Optimizer and Metrics
        if self.learner.lower() == "adagrad": 
            model.compile(optimizer=Adagrad(learning_rate=self.lr), loss='binary_crossentropy')
        elif self.learner.lower() == "rmsprop":
            model.compile(optimizer=RMSprop(learning_rate=self.lr), loss='binary_crossentropy')
        elif self.learner.lower() == "adam":
            model.compile(optimizer=Adam(learning_rate=self.lr), loss='binary_crossentropy')
        elif self.learner.lower() == "sgd":
            model.compile(optimizer=SGD(learning_rate=self.lr), loss='binary_crossentropy')
        else:
            model.compile(optimizer=self.learner, loss='binary_crossentropy')

        ##### Learning a Model #####
        loop = trange(self.num_epochs, disable=not self.verbose)
        for _ in loop:
            count = 0
            sum_loss = 0
            # Generate training samples
            for i, (batch_users, batch_items, batch_ratings) in enumerate(
                train_set.uir_iter(self.batch_size, shuffle=True, binary=True, num_zeros=self.num_neg)):
                # Training
                hist = model.fit([batch_users, batch_items], batch_ratings, batch_size=self.batch_size, epochs=1, verbose=0)
                count += len(batch_users)
                sum_loss += len(batch_users) * hist.history['loss'][0]
                if i % 10 == 0:
                    loop.set_postfix(loss=(sum_loss / count))
        loop.close()    
        self.model = model 

        return self
    
    def score(self, user_idx, item_idx=None):
        if item_idx is None:
            user_id = np.ones(self.num_items) * user_idx,
            item_id = np.arange(self.num_items)
        else:
            user_id = [user_idx]
            item_id = [item_idx]
        
        return self.model.predict([user_id, item_id], batch_size=self.batch_size*2, verbose=0).ravel()
