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
from neural_collaborative_filtering import NeuralCollaborativeFiltering
from utils import MovieLensDataset, split_data


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