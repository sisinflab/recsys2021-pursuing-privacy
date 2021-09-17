import requests
import os
import io
import zipfile
import pandas as pd
import numpy as np
from tqdm import tqdm

url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
print(f"Getting Movielens Small from : {url} ..")
response = requests.get(url)

ml_ratings = []

print("Extracting ratings...")
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    for line in zip_ref.open("ml-latest-small/ratings.csv"):
        ml_ratings.append(str(line, "utf-8"))

print("Printing ratings to data/movielens/ ..")
os.makedirs("data/movielens", exist_ok=True)
with open("data/movielens/dataset.csv", "w") as f:
    f.writelines(ml_ratings)


b_m = 1
b_u = 1
e1 = 1
e2 = 1
b = 1

ratings = pd.read_csv('data/movielens/dataset.csv')

# TODO: Splitting train-test


min_rating = ratings['rating'].min()
max_rating = ratings['rating'].max()
delta_r = max_rating - min_rating

global_average_item = (ratings['rating'].sum() + np.random.laplace(scale=(delta_r / e1))) / len(ratings)

item_sets = ratings.groupby('movieId')['rating']
i_avg = (item_sets.sum() + b_m * global_average_item + np.random.laplace(scale=(delta_r / e2))) / (item_sets.count() + b_m)
i_avg = np.clip(i_avg, min_rating, max_rating)

merged = ratings.join(i_avg, on=['movieId'], lsuffix='_x', rsuffix='_y')

merged['rating'] = merged['rating_x'] - merged['rating_y']
merged = merged.drop(columns=['rating_x', 'rating_y'], axis=1)

global_average_user = (merged['rating'].sum() + np.random.laplace(scale=(delta_r / e1))) / len(merged)

user_sets = merged.groupby('userId')['rating']
u_avg = (user_sets.sum() + b_u * global_average_user + np.random.laplace(scale=(delta_r / e2))) / (user_sets.count() + b_u)
u_avg = np.clip(u_avg, -2, 2) # Valore di progetto

new_ratings = merged.join(u_avg, on=['userId'], lsuffix='_x', rsuffix='_y')

new_ratings['rating'] = new_ratings['rating_x'] - new_ratings['rating_y']
new_ratings = new_ratings.drop(columns=['rating_x', 'rating_y'], axis=1)
new_ratings['rating'] = np.clip(new_ratings['rating'], -b, b)

# Questi rating possono essere usati in un algoritmo di factorization non privato per derivare P e Q


# INPUT PERTURBATION

# range of the ratings is now [-b, b]
# global sensitivity is 2 * b

delta_r = 2 * b
e = 10


ratings = new_ratings
ratings['rating'] = new_ratings['rating'] + np.random.laplace(scale=(delta_r / e))
ratings['rating'] = np.clip(ratings['rating'], -b, b)

f = 100
lr = 0.001
epochs = 10


class MF:
    def __init__(self, dataset, n_factors):
        """
        :param dataset: interaction dataset should be a Pandas dataframe with three columns for user, item, and rating
        :param n_factors:
        """
        self.dataset = dict()
        self.ext2int_user_map, self.int2ext_user_map, self.ext2int_item_map, self.int2ext_item_map = self.create_maps(
            dataset)
        n_users = dataset.iloc[:, 0].nunique()
        n_items = dataset.iloc[:, 1].nunique()
        self.n_interactions = len(dataset)
        self.delta_ratings = dataset.iloc[:, 2].max() - dataset.iloc[:, 2].min()
        self.p = np.random.randn(n_users, n_factors)
        self.q = np.random.randn(n_items, n_factors)

    def create_maps(self, dataset):
        ext2int_user_map = {v: k for k, v in enumerate(dataset.iloc[:, 0].unique())}
        int2ext_user_map = {k: v for k, v in enumerate(dataset.iloc[:, 0].unique())}
        ext2int_item_map = {v: k for k, v in enumerate(dataset.iloc[:, 1].unique())}
        int2ext_item_map = {k: v for k, v in enumerate(dataset.iloc[:, 1].unique())}
        self.dataset['userId'] = dataset.iloc[:, 0].map(ext2int_user_map).to_dict()
        self.dataset['itemId'] = dataset.iloc[:, 1].map(ext2int_item_map).to_dict()
        self.dataset['rating'] = dataset.iloc[:, 2].to_dict()
        return ext2int_user_map, int2ext_user_map, ext2int_item_map, int2ext_item_map

    def train(self, lr, epochs):
        for e in range(epochs):
            print(f"*** Epoch {e + 1}/{epochs} ***")
            for i in tqdm(range(self.n_interactions)):
                p_u = self.p[self.dataset['userId'][i]]
                q_i = self.q[self.dataset['itemId'][i]]
                pred = p_u.dot(q_i)
                err = self.dataset['rating'][i] - pred
                self.p[self.dataset['userId'][i]] = p_u + lr * (err * q_i)
                self.q[self.dataset['itemId'][i]] = q_i + lr * (err * p_u)

    def train_dp(self, lr, epochs, eps, err_max=None):
        for e in range(epochs):
            print(f"*** Epoch {e + 1}/{epochs} ***")
            for i in tqdm(range(self.n_interactions)):
                p_u = self.p[self.dataset['userId'][i]]
                q_i = self.q[self.dataset['itemId'][i]]
                pred = p_u.dot(q_i)
                err = self.dataset['rating'][i] - pred + np.random.laplace(scale=(epochs * self.delta_ratings / eps))
                if err_max:
                    err = np.clip(err, -err_max, err_max)
                self.p[self.dataset['userId'][i]] = p_u + lr * (err * q_i)
                self.q[self.dataset['itemId'][i]] = q_i + lr * (err * p_u)



# prepara dataframe per mf
df = ratings.loc[:, ['userId', 'movieId', 'rating']]
mf = MF(df, f)
# mf.train(lr, epochs)
mf.train_dp(lr, epochs, 10)