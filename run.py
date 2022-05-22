import networkx as nx, pandas as pd
from train import EllipticDataset




features = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_features.csv', header=None)
classes = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_classes.csv', na_values='unknown')
edges = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')


print(features)
print(classes)
print(edges)

data = EllipticDataset(features.copy(), classes.copy(), edges.copy())