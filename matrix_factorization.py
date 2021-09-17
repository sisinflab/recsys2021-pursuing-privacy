import requests
import os
import io
import zipfile
import pandas as pd
import numpy as np
from tqdm import tqdm
import math

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

def splitting(dataset, ratio=0.8):
    user_size = dataset.groupby(['userId'], as_index=True).size()
    user_threshold = user_size.apply(lambda x: math.floor(x * (1 - ratio)))
    if len(dataset.columns) == 4:
        dataset['rank_first'] = dataset.groupby(['userId'])['timestamp'].rank(method='first', ascending=True, axis=1)
        dataset["test_flag"] = dataset.apply(
            lambda x: x["rank_first"] > user_threshold.loc[x["userId"]], axis=1)
    test = dataset[dataset["test_flag"] == True].drop(columns=["rank_first", "test_flag"]).reset_index(drop=True)
    train = dataset[dataset["test_flag"] == False].drop(columns=["rank_first", "test_flag"]).reset_index(drop=True)

    return train, test

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
# Given i_avg and u_avg, the prediction outcoming from the model should be summed up to i_avg(i) and u_avg(u)


# INPUT PERTURBATION

# range of the ratings is now [-b, b]
# global sensitivity is 2 * b

delta_r = 2 * b
e = 10


# CREATE INPUT PERTURBATION FOR BOUNDED DP

ratings = new_ratings
ratings['rating'] = new_ratings['rating'] + np.random.laplace(scale=(delta_r / e))
ratings['rating'] = np.clip(ratings['rating'], -b, b)


# CREATE INPUT PERTURBATION FOR UNBOUNDED DP
# To mask the existence of ratings, we should add noise also to non existing interactions (i.e. add fake ratings)
# The noise should be proportional to the maximum difference btw any rating and the 0 rating, i.e. b --> lap(b / e)
# (Io non lo realizzerei, perche passeremmo a una URM densa, e non converrebbe piu usare Pandas)



f = 100
lr = 0.001
epochs = 2


class MF:
    def __init__(self, dataset, n_factors, i_avg=None, u_avg=None):
        """
        :param dataset: interaction dataset should be a Pandas dataframe with three columns for user, item, and rating
        :param n_factors:
        """
        self.dataset = dict()
        self.ext2int_user_map, self.int2ext_user_map, self.ext2int_item_map, self.int2ext_item_map = self.create_maps(
            dataset, 0.5)
        n_users = dataset.iloc[:, 0].nunique()
        n_items = dataset.iloc[:, 1].nunique()
        self.n_interactions = len(dataset)
        self.delta_ratings = dataset.iloc[:, 2].max() - dataset.iloc[:, 2].min()
        self.p = np.random.randn(n_users, n_factors)
        self.q = np.random.randn(n_items, n_factors)
        if i_avg and u_avg:
            self.i_avg = i_avg.to_numpy()
            self.u_avg = u_avg.to_numpy()

    def create_maps(self, dataset, relevance):
        ext2int_user_map = {v: k for k, v in enumerate(dataset.iloc[:, 0].unique())}
        int2ext_user_map = {k: v for k, v in enumerate(dataset.iloc[:, 0].unique())}
        ext2int_item_map = {v: k for k, v in enumerate(dataset.iloc[:, 1].unique())}
        int2ext_item_map = {k: v for k, v in enumerate(dataset.iloc[:, 1].unique())}
        self.dataset['userId'] = dataset.iloc[:, 0].map(ext2int_user_map).to_dict()
        self.dataset['itemId'] = dataset.iloc[:, 1].map(ext2int_item_map).to_dict()
        self.dataset['rating'] = dataset.iloc[:, 2].to_dict()
        self.relevant_items = {
            ext2int_user_map[u]: dataset[(dataset.iloc[:, 0] == u) & (dataset.iloc[:, 2] >= relevance)].iloc[:,
                                 1].to_list() for u in ext2int_user_map}
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

    def train_laplace_dp(self, lr, epochs, eps, err_max=None):
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

    def train_gaussian_unbounded_dp(self, lr, epochs, eps, delta, err_max=None):
        for e in range(epochs):
            print(f"*** Epoch {e + 1}/{epochs} ***")
            for i in tqdm(range(self.n_interactions)):
                p_u = self.p[self.dataset['userId'][i]]
                q_i = self.q[self.dataset['itemId'][i]]
                pred = p_u.dot(q_i)
                err = np.clip(self.dataset['rating'][i] - pred, err_max)
                self.p[self.dataset['userId'][i]] = p_u + lr * (err * q_i)
                self.q[self.dataset['itemId'][i]] = q_i + lr * (err * p_u)
            for u in self.int2ext_user_map:
                2 * s_p * epochs * np.sqrt(2 * np.log(2 / delta)) / eps

                # TODO: Terminare


    def evaluate(self, test=None, cutoff=10, relevance=0.5):
        prediction = (np.dot(self.p, self.q.T).T + i_avg).T + u_avg
        precisions = []
        recalls = []
        for u in self.int2ext_user_map:
            # TODO: Mettere a -inf gli item nel training set?
            unordered_top_k = np.argpartition(prediction[u], -cutoff)[-cutoff:]
            top_k = unordered_top_k[np.argsort(prediction[u][unordered_top_k])][::-1]
            top_k_score = prediction[u][top_k]
            # TODO: aggiustare n_rel, mi servono i rilevanti nel test!?
            n_rel = len(self.relevant_items[u])
            n_rec_k = sum(top_k_score >= relevance)
            n_rel_and_rec_k = sum(i in test[u] and i >= relevance for i in top_k)
            precisions.append(n_rel_and_rec_k / n_rec_k)
            recalls.append(n_rel_and_rec_k / n_rel)
        precision = sum(precisions) / len(precisions)
        recall = sum(recalls) / len(recalls)



# prepara dataframe per mf
df = ratings.loc[:, ['userId', 'movieId', 'rating']]
train_set, test_set = splitting(df)
mf = MF(train_set, f, i_avg, u_avg)
# If we don't want to send privatized input:
# mf = MF(train_set, f)
mf.train(lr, epochs)
mf.evaluate(test_set)
#mf.train_dp(lr, epochs, 10)