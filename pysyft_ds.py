import requests
import os
import io
import zipfile
import pandas as pd
import syft as sy
import torch

# pip3 install "syft @ git+https://github.com/OpenMined/PySyft@sympc-dev#egg=syft&subdirectory=packages/syft"
# pip3 install -e "git+https://github.com/OpenMined/SyMPC#egg=sympc"

from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import MPCTensor

# The MPCTensor is the tensor that holds reference to the shares owned by the different parties.

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
SessionManager.setup_mpc(session)

x_secret = torch.Tensor([[0.1, -1], [-4, 4]])
y_secret = torch.Tensor([[4.0, -2.5], [5, 2]])

print(x_secret)
print(y_secret)



# 2. Share the secret between all members of a session
x = x_secret.share(session=session)
y = y_secret.share(session=session)

print(x)
print(y)

print((x + y).reconstruct())

# duet_p1 = sy.join_duet(loopback=True)
# duet_p2 = sy.join_duet(loopback=True)

# FORSE NON CI SERVE

# We need to setup a session to send some config infomration only once between the parties
# e.g. the ring size, the precision and the base

# session = Session(parties=[duet_p1, duet_p2])
# print(session)
# SessionManager.setup_mpc(session)

# Let's download MovieLens Small
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

