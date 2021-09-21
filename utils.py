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