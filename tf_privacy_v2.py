import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sampler import Sampler as sampler
from tqdm import tqdm
import gflags
from utils import dataframe_to_dict, splitting, create_maps
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from tensorflow_privacy.privacy.analysis.gdp_accountant import compute_eps_poisson
from tensorflow_privacy.privacy.analysis.gdp_accountant import compute_mu_poisson
from tensorflow_privacy.privacy.optimizers import dp_optimizer
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras

#### FLAGS
FLAGS = gflags.FLAGS
gflags.DEFINE_boolean(
    'dpsgd', False, 'If True, train with DP-SGD. If False, '
                    'train with vanilla SGD.')
gflags.DEFINE_float('learning_rate', .01, 'Learning rate for training')
gflags.DEFINE_float('noise_multiplier', 0.55,
                    'Ratio of the standard deviation to the clipping norm')
gflags.DEFINE_float('l2_norm_clip', 5, 'Clipping norm')
gflags.DEFINE_integer('epochs', 25, 'Number of epochs')
gflags.DEFINE_integer('max_mu', 3, 'GDP upper limit')
gflags.DEFINE_string('model_dir', None, 'Model directory')


class NCF(keras.Model):
    def __init__(self, dataset, maps,
                 embed_mf_size, embed_mlp_size, mlp_hidden_size, dropout, learning_rate=0.01, **kwargs):

        super().__init__(name='NCF', **kwargs)

        self.ext2int_user_map, self.int2ext_user_map, self.ext2int_item_map, self.int2ext_item_map = maps
        self.transactions = len(dataset)
        self.batch_size = 512
        self.train_dict = dataframe_to_dict(dataset)

        self.i_train_dict = {self.ext2int_user_map[user]: {self.ext2int_item_map[i]: v for i, v in items.items()}
                             for user, items in self.train_dict.items()}

        self._sampler = sampler(self.i_train_dict, 2)

        num_users = len(self.ext2int_user_map)
        num_items = len(self.ext2int_item_map)

        self.num_users = num_users
        self.num_items = num_items
        self.embed_mf_size = embed_mf_size
        self.embed_mlp_size = embed_mlp_size
        self.mlp_hidden_size = mlp_hidden_size
        self.dropout = dropout

        self.initializer = tf.initializers.GlorotUniform()

        self.user_mf_embedding = keras.layers.Embedding(input_dim=self.num_users, output_dim=self.embed_mf_size,
                                                        embeddings_initializer=self.initializer, name='U_MF',
                                                        dtype=tf.float32)
        self.item_mf_embedding = keras.layers.Embedding(input_dim=self.num_items, output_dim=self.embed_mf_size,
                                                        embeddings_initializer=self.initializer, name='I_MF',
                                                        dtype=tf.float32)
        self.user_mlp_embedding = keras.layers.Embedding(input_dim=self.num_users, output_dim=self.embed_mlp_size,
                                                         embeddings_initializer=self.initializer, name='U_MLP',
                                                         dtype=tf.float32)
        self.item_mlp_embedding = keras.layers.Embedding(input_dim=self.num_items, output_dim=self.embed_mlp_size,
                                                         embeddings_initializer=self.initializer, name='I_MLP',
                                                         dtype=tf.float32)

        self.user_mf_embedding(0)
        self.user_mlp_embedding(0)
        self.item_mf_embedding(0)
        self.item_mlp_embedding(0)

        self.mlp_layers = keras.Sequential()

        for units in mlp_hidden_size:
            self.mlp_layers.add(keras.layers.Dropout(dropout))
            self.mlp_layers.add(keras.layers.Dense(units, activation='relu'))

        self.predict_layer = keras.layers.Dense(1, input_dim=self.embed_mf_size + self.mlp_hidden_size[-1])

        self.sigmoid = keras.activations.sigmoid
        self.loss = keras.losses.BinaryCrossentropy(reduction=tf.losses.Reduction.NONE)

        # self.optimizer = tf.optimizers.Adam(learning_rate)

        self.optimizer = dp_optimizer.DPAdamGaussianOptimizer(
                l2_norm_clip=5,
                noise_multiplier=0.55,
                num_microbatches=512,
                learning_rate=learning_rate)

        # self.optimizer = dp_optimizer_keras.DPKerasAdamOptimizer(
        #         l2_norm_clip=5,
        #         noise_multiplier=0.55,
        #         num_microbatches=512,
        #         learning_rate=learning_rate)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        user, item = inputs
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)
        mf_output = user_mf_e * item_mf_e  # [batch_size, embedding_size]
        mlp_output = self.mlp_layers(tf.concat([user_mlp_e, item_mlp_e], -1))  # [batch_size, layers[-1]]
        output = self.sigmoid(self.predict_layer(tf.concat([mf_output, mlp_output], -1)))
        return output

    def train(self, epochs):
        for e in range(epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self.transactions * 3 // self.batch_size)) as t:
                for batch in self._sampler.step(self.batch_size):
                    steps += 1
                    loss += self.train_step(batch).numpy()
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

    @tf.function
    def train_step(self, batch):

        user, pos, label = batch
        with tf.GradientTape() as tape:
            # Clean Inference
            output = self(inputs=(user, pos), training=True)
            loss = self.loss(tf.reshape(label, [len(label), 1]), output)

            def loss_fn():
                output = self(inputs=(user, pos), training=True)
                loss = self.loss(tf.reshape(label, [len(label), 1]), output)
                # loss = tf.reshape(loss, [1])
                return loss

            grads_and_vars = self.optimizer.compute_gradients(loss=loss, var_list=self.trainable_weights)
        # self.optimizer.minimize(loss=loss_fn, var_list=[self.weights])
        self.optimizer.apply_gradients(grads_and_vars)
        return tf.reduce_mean(input_tensor=loss)


if __name__ == "__main__":
    ratings = pd.read_csv('data/movielens/dataset.csv')

    # prepara dataframe per mf
    df = ratings.loc[:, ['userId', 'movieId', 'rating', 'timestamp']]
    train_set, test_set = splitting(df)
    maps = create_maps(train_set)

    b_m = 1
    b_u = 1
    eps_global_avg = 1
    eps_item_avg = 1
    eps_user_avg = 1
    clamping = 1

    f = 100
    lr = 0.001
    epochs = 30

    mf = NCF(train_set, maps, 4, 4, (4*4, 4*2, 4), 0, 0.001)
    # If we don't want to send privatized input:
    # mf = MF(train_set, f)
    mf.train(50)
    # NCF.evaluate(test_set)
    #mf.train_dp(lr, epochs, 10)