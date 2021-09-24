import requests
import os
import io
import copy
import zipfile
import pandas as pd
import numpy as np
import syft as sy
import torch
from torch.utils.data import DataLoader
import tqdm
from torchfm.layer import FeaturesEmbedding, MultiLayerPerceptron
from collections import OrderedDict

# pip3 install "syft @ git+https://github.com/OpenMined/PySyft@sympc-dev#egg=syft&subdirectory=packages/syft"
# pip3 install -e "git+https://github.com/OpenMined/SyMPC#egg=sympc"

from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import MPCTensor

# The MPCTensor is the tensor that holds reference to the shares owned by the different parties.

from torch.utils.data import Dataset


class PandasDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        return self.dataframe.iloc[index]



class NeuralCollaborativeFiltering(torch.nn.Module):
    """
    A pytorch implementation of Neural Collaborative Filtering.
    Reference:
        X He, et al. Neural Collaborative Filtering, 2017.
    """

    def __init__(self, field_dims, user_field_idx, item_field_idx, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.user_field_idx = user_field_idx
        self.item_field_idx = item_field_idx
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.fc = torch.nn.Linear(mlp_dims[-1] + embed_dim, 1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_user_fields)``
        """
        x = self.embedding(x)
        user_x = x[:, self.user_field_idx].squeeze(1)
        item_x = x[:, self.item_field_idx].squeeze(1)
        x = self.mlp(x.view(-1, self.embed_output_dim))
        gmf = user_x * item_x
        x = torch.cat([gmf, x], dim=1)
        x = self.fc(x).squeeze(1)
        return torch.sigmoid(x)


def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        # fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)


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


def split_data(data, n_organizations):
    item_idxs = np.array_split(np.random.permutation(data.iloc[:, 1].unique()), n_organizations)
    dfs = [data[data.iloc[:, 1].isin(item_idxs[i])] for i in range(n_organizations)]
    return dfs


# # Let's download MovieLens Small
# url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
# print(f"Getting Movielens Small from : {url} ...")
# response = requests.get(url)
#
# ml_ratings = []
#
# print("Extracting ratings...")
# with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
#     for line in zip_ref.open("ml-latest-small/ratings.csv"):
#         ml_ratings.append(str(line, "utf-8"))
#
# print("Printing ratings to data/movielens/ ...")
# os.makedirs("data/movielens", exist_ok=True)
# with open("data/movielens/dataset.csv", "w") as f:
#     f.writelines(ml_ratings)


# Read and split data, then create private dataloaders

df = pd.read_csv('data/movielens/dataset.csv')
df['userId'] = df['userId'].map({v: k for k, v in enumerate(df['userId'].unique())})
df['movieId'] = df['movieId'].map({v: k for k, v in enumerate(df['movieId'].unique())})
data = split_data(df, 3)

batch_size = 64

train_data_loaders = []
valid_data_loaders = []
test_data_loaders = []
users = set()
items = set()
for i in range(len(data)):
    train_length = int(len(data[i]) * 0.8)
    valid_length = int(len(data[i]) * 0.1)
    test_length = len(data[i]) - train_length - valid_length
    local_dataset = MovieLensDataset(data[i])
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        local_dataset, (train_length, valid_length, test_length))
    train_data_loaders.append(DataLoader(train_dataset, batch_size=batch_size, num_workers=0))
    valid_data_loaders.append(DataLoader(valid_dataset, batch_size=batch_size, num_workers=0))
    test_data_loaders.append(DataLoader(test_dataset, batch_size=batch_size, num_workers=0))
    users = users.union(set(np.unique(train_dataset.dataset.items[:, 0])))
    items = items.union(set(np.unique(train_dataset.dataset.items[:, 1])))
n_users = len(users)
n_items = len(items)


# We have three organizations, let's create their virtual machines

store1_vm = sy.VirtualMachine(name="store1")
store2_vm = sy.VirtualMachine(name="store2")
store3_vm = sy.VirtualMachine(name="store3")

# Get clients from each VM
store1 = store1_vm.get_root_client()
store2 = store2_vm.get_root_client()
store3 = store3_vm.get_root_client()

parties = [store1, store2, store3]

session = Session(parties=parties)

# When we do not pass any protocol to session, SyMPC uses SPDZ protocol with semi-honest security type.
# SPDZ is used for multiplication and related operations (convolution,matmul,etc) and could extend to N parties.

# Falcon can also provide a malicious security guarantee for an honest majority at the cost of higher inference time.
# Malicious security ensures that all the parties compute according to the protocol
# and do not deviate from protocol or tamper with shares.
#

SessionManager.setup_mpc(session)


# Initialize the global model

ncf = NeuralCollaborativeFiltering(field_dims=(n_users, n_items), embed_dim=64, mlp_dims=(64, 32, 16), dropout=0.2,
                                   user_field_idx=[0],
                                   item_field_idx=[1])
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(params=ncf.parameters(), lr=0.01)

rounds = 10
local_epochs = 1

for round in range(rounds):
    print('***\nRound', round + 1)
    local_params = []
    for u in range(len(data)):
        # Local training of the model
        print('Organization #', u + 1)
        local_ncf = NeuralCollaborativeFiltering(field_dims=(n_users, n_items), embed_dim=64, mlp_dims=(64, 32, 16),
                                           dropout=0.2,
                                           user_field_idx=[0],
                                           item_field_idx=[1])
        local_ncf.load_state_dict(copy.deepcopy(ncf.state_dict()))

        for i in range(local_epochs):
            print('Local epoch', i + 1)
            # copy the model
            train(local_ncf, optimizer, train_data_loaders[u], criterion, 'cpu')
        # The parameters in local_params are the secrets of each client
        local_params.append({param: value for param, value in local_ncf.state_dict().items()})
    # Now, we should compute the mean of the local parameters with SMPC
    # 1. Each party "shares" its secret with the other parties in a secure fashion
    print("Creating the shares...")
    secret_local_params = []
    for u in local_params:
        secret_local_params.append(
            {param: value.share(session=session) if not param.endswith('num_batches_tracked') else value for
             param, value in u.items()})
    # 2. We can compute the secret mean
    print("Computing the secret mean...")
    secret_global_params = {name: sum([secret_local_params[i][name] for i in range(len(secret_local_params))]) / len(
        secret_local_params) if not name.endswith('num_batches_tracked') else sum(
        [secret_local_params[i][name] for i in range(len(secret_local_params))]) for name in secret_local_params[0]}

    # Finally, we can reconstruct the real value of the mean
    print("Reconstructing the mean...")
    global_params = {name: value.reconstruct() if not name.endswith('num_batches_tracked') else value for name, value in
                     secret_global_params.items()}

    # Set the global model with the new computed parameters
    with torch.no_grad():
        print("Updating the global model...")
        ncf.load_state_dict(OrderedDict(global_params))