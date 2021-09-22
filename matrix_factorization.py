import requests
import os
import io
import zipfile
import pandas as pd
import numpy as np
from tqdm import tqdm
import math


def create_maps(dataset):
    ext2int_user_map = {v: k for k, v in enumerate(dataset.iloc[:, 0].unique())}
    int2ext_user_map = {k: v for k, v in enumerate(dataset.iloc[:, 0].unique())}
    ext2int_item_map = {v: k for k, v in enumerate(dataset.iloc[:, 1].unique())}
    int2ext_item_map = {k: v for k, v in enumerate(dataset.iloc[:, 1].unique())}
    return ext2int_user_map, int2ext_user_map, ext2int_item_map, int2ext_item_map


def splitting(dataset, ratio=0.2):
    print("Performing splitting...")
    user_size = dataset.groupby(['userId'], as_index=True).size()
    user_threshold = user_size.apply(lambda x: math.floor(x * (1 - ratio)))
    dataset['rank_first'] = dataset.groupby(['userId'])['timestamp'].rank(method='first', ascending=True)
    dataset["test_flag"] = dataset.apply(
        lambda x: x["rank_first"] > user_threshold.loc[x["userId"]], axis=1)
    test = dataset[dataset["test_flag"] == True].drop(columns=["rank_first", "test_flag"]).reset_index(drop=True)
    train = dataset[dataset["test_flag"] == False].drop(columns=["rank_first", "test_flag"]).reset_index(drop=True)

    return train, test


def privatize_global_effects(ratings, b_m, b_u, eps_global_avg, eps_item_avg, eps_user_avg, clamping):
    min_rating = ratings['rating'].min()
    max_rating = ratings['rating'].max()
    delta_r = max_rating - min_rating

    global_average_item = (ratings['rating'].sum() + np.random.laplace(scale=(delta_r / eps_global_avg))) / len(ratings)

    item_sets = ratings.groupby('movieId')['rating']
    i_avg = (item_sets.sum() + b_m * global_average_item + np.random.laplace(scale=(delta_r / eps_item_avg))) / (
                item_sets.count() + b_m)
    i_avg = np.clip(i_avg, min_rating, max_rating)

    merged = ratings.join(i_avg, on=['movieId'], lsuffix='_x', rsuffix='_y')

    merged['rating'] = merged['rating_x'] - merged['rating_y']
    merged = merged.drop(columns=['rating_x', 'rating_y'], axis=1)

    global_average_user = (merged['rating'].sum() + np.random.laplace(scale=(delta_r / eps_global_avg))) / len(merged)

    user_sets = merged.groupby('userId')['rating']
    u_avg = (user_sets.sum() + b_u * global_average_user + np.random.laplace(scale=(delta_r / eps_user_avg))) / (
                user_sets.count() + b_u)
    u_avg = np.clip(u_avg, -2, 2)  # Valore dal paper

    preprocessed_ratings = merged.join(u_avg, on=['userId'], lsuffix='_x', rsuffix='_y')

    preprocessed_ratings['rating'] = preprocessed_ratings['rating_x'] - preprocessed_ratings['rating_y']
    preprocessed_ratings = preprocessed_ratings.drop(columns=['rating_x', 'rating_y'], axis=1)
    preprocessed_ratings['rating'] = np.clip(preprocessed_ratings['rating'], -clamping, clamping)

    return preprocessed_ratings, i_avg, u_avg

    # Questi rating possono essere usati in un algoritmo di factorization non privato per derivare P e Q
    # Given i_avg and u_avg, the prediction outcoming from the model should be summed up to i_avg(i) and u_avg(u)


def input_perturbation(ratings, clamping, eps):
    # range of the received ratings is [-clamping, clamping]
    # it's better to have the value of clamping, to know the exact value
    delta_r = 2 * clamping  # global sensitivity

    # bounded differential privacy
    ratings['rating'] = np.clip(ratings['rating'] + np.random.laplace(scale=(delta_r / eps)), -clamping, clamping)

    # CREATE INPUT PERTURBATION FOR UNBOUNDED DP
    # To mask the existence of ratings, we should add noise also to non existing interactions (i.e. add fake ratings)
    # The noise should be proportional to the maximum difference btw any rating and the 0 rating, i.e. b --> lap(b / e)
    # (Io non lo realizzerei, perche passeremmo a una URM densa, e non converrebbe piu usare Pandas)


class MF:
    def __init__(self, dataset, maps, n_factors, relevance=3.5, i_avg=None, u_avg=None):
        """
        :param dataset: interaction dataset should be a Pandas dataframe with three columns for user, item, and rating
        :param n_factors:
        """
        print("Building model...")
        self.ext2int_user_map, self.int2ext_user_map, self.ext2int_item_map, self.int2ext_item_map = maps
        self.dataset = self.format_dataset(dataset)
        self.rated_items = {
            self.ext2int_user_map[u]: dataset[(dataset.iloc[:, 0] == u) & (dataset.iloc[:, 2] >= relevance)].iloc[:,
                                      1].map(self.ext2int_item_map).astype(int).to_list() for u in
            self.ext2int_user_map}
        n_users = len(self.ext2int_user_map)
        n_items = len(self.ext2int_item_map)
        self.n_interactions = len(dataset)
        self.delta_ratings = dataset.iloc[:, 2].max() - dataset.iloc[:, 2].min()
        self.p = np.random.randn(n_users, n_factors)
        self.q = np.random.randn(n_items, n_factors)
        if i_avg is not None and u_avg is not None:
            self.i_avg = i_avg.to_numpy()
            self.u_avg = u_avg.to_numpy()

    def format_dataset(self, df):
        dataset = {}
        dataset['userId'] = df.iloc[:, 0].map(self.ext2int_user_map).to_dict()
        dataset['itemId'] = df.iloc[:, 1].map(self.ext2int_item_map).to_dict()
        dataset['rating'] = df.iloc[:, 2].to_dict()
        return dataset

    def train(self, lr, epochs):
        print("Starting training...")
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

    def evaluate(self, test=None, cutoff=10, relevance=3.5):
        print("Starting evaluation...")
        prediction = (np.dot(self.p, self.q.T).T + self.i_avg[:, None]).T + self.u_avg[:, None]
        precisions = []
        recalls = []
        print("Reading test set...")
        relevant_items_test = {
            self.ext2int_user_map[u]: set(test[(test.iloc[:, 0] == u) & (test.iloc[:, 2] >= relevance)].iloc[:, 1].map(
                self.ext2int_item_map).dropna().astype(int).to_list()) for u in self.ext2int_user_map}
        print("Computing metrics...")
        for u in self.int2ext_user_map:
            prediction[u, self.rated_items[u]] = - np.inf
            unordered_top_k = np.argpartition(prediction[u], -cutoff)[-cutoff:]
            top_k = unordered_top_k[np.argsort(prediction[u][unordered_top_k])][::-1]
            # top_k_score = prediction[u][top_k]
            n_rel_and_rec_k = sum(i in relevant_items_test[u] for i in top_k)
            precisions.append(n_rel_and_rec_k / cutoff)
            try:
                recalls.append(n_rel_and_rec_k / len(relevant_items_test[u]))
            except ZeroDivisionError:
                recalls.append(0)
        precision = sum(precisions) / len(precisions)
        recall = sum(recalls) / len(recalls)

        print(f"Precision@{cutoff}: {precision}")
        print(f"Recall@{cutoff}: {recall}")


url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
print(f"Getting Movielens Small from : {url} ...")
response = requests.get(url)

ml_ratings = []

print("Extracting ratings...")
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    for line in zip_ref.open("ml-latest-small/ratings.csv"):
        ml_ratings.append(str(line, "utf-8"))

print("Printing ratings to data/movielens/ ...")
os.makedirs("data/movielens", exist_ok=True)
with open("data/movielens/dataset.csv", "w") as f:
    f.writelines(ml_ratings)


df = pd.read_csv('data/movielens/dataset.csv')

# prepara dataframe per mf
train_set, test_set = splitting(df)
train_set = train_set.loc[:, ['userId', 'movieId', 'rating']]
test_set = test_set.loc[:, ['userId', 'movieId', 'rating']]
maps = create_maps(train_set)

b_m = 1
b_u = 1
eps_global_avg = 1
eps_item_avg = 1
eps_user_avg = 1
clamping = 1

preproc_train_set, i_avg, u_avg = privatize_global_effects(train_set, b_m, b_u, eps_global_avg, eps_item_avg,
                                                           eps_user_avg, clamping)

f = 50
lr = 0.01
epochs = 50

mf = MF(preproc_train_set, maps, f, relevance=0, i_avg=i_avg, u_avg=u_avg)
# If we don't want to send privatized input:
# mf = MF(train_set, f)
mf.train(lr, epochs)
mf.evaluate(test_set)
#mf.train_dp(lr, epochs, 10)