import math
import numpy as np

np.random.seed(42)


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


def splitting_leave_one_out(dataset):
    print("Performing splitting...")
    dataset['rank_first'] = dataset.groupby(['userId'])['timestamp'].rank(method='first', ascending=False, axis=1)
    dataset["test_flag"] = dataset.apply(
        lambda x: x["rank_first"] <= 1, axis=1)
    test = dataset[dataset["test_flag"] == True].drop(columns=["rank_first", "test_flag"]).reset_index(drop=True)
    train = dataset[dataset["test_flag"] == False].drop(columns=["rank_first", "test_flag"]).reset_index(drop=True)

    return train, test


def dataframe_to_dict(data):
    users = list(data['userId'].unique())

    "Conversion to Dictionary"
    ratings = {}
    for u in users:
        sel_ = data[data['userId'] == u]
        ratings[u] = dict(zip(sel_['movieId'], sel_['rating']))
    return ratings

def split_data(data, n_organizations):
    item_idxs = np.array_split(np.random.permutation(data.iloc[:, 1].unique()), n_organizations)
    dfs = [data[data.iloc[:, 1].isin(item_idxs[i])] for i in range(n_organizations)]
    return dfs


class MovieLensDataset(torch.utils.data.Dataset):
    """
    MovieLens 20M Dataset
    Data preparation
        treat samples with a rating less than 3 as negative samples
    :param dataset_path: MovieLens dataset path
    Reference:
        https://grouplens.org/datasets/movielens
    """

    def __init__(self, dataset, sep=',', engine='c', header='infer'):
        # data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header).to_numpy()[:, :3]
        data = dataset.to_numpy()[:, :3]
        self.items = data[:, :2].astype(np.int)  # -1 because ID begins from 1
        self.targets = self.__preprocess_target(data[:, 2]).astype(np.float32)
        self.field_dims = np.max(self.items, axis=0) + 1
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target