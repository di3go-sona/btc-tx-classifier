#%%
import networkx as nx, pandas as pd
import sklearn
from train import EllipticDataset, GCNModel, CombinedDataset
import torch


features = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_features.csv', header=None)
classes = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_classes.csv', na_values='unknown')
edges = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')


# %%
data = EllipticDataset(features.copy(), classes.copy(), edges.copy())
model = GCNModel.load_from_checkpoint('btc-xai/369sy289/checkpoints/epoch=14-step=17460.ckpt', data=data)


# %%
nodes = data.features.T[0].astype(int)
edges = data.edges.numpy().T
ts = data.features.T[1].astype(int)

# %%
graph = nx.DiGraph()
graph.add_nodes_from(nodes)
graph.add_edges_from(edges)

# %%
# timestep = 1
# subgraph_nodes = nodes[ts == timestep]
# subgraph = nx.subgraph(graph, subgraph_nodes)


import gradio as gr
from matplotlib import pyplot  as plt
from  matplotlib.colors import LinearSegmentedColormap

node_max = nodes.max()


gr_node_id = gr.Number( 4, label=f'Select node Id in range [0,{node_max}]', round=0)
gr_neighs_slider = gr.Slider( 1,10, label=f'Select neighborhood size ', step=1)

gr_graph = gr.Image( label ='Graph')
gr_edges = gr.Number( label='# Edges')
gr_nodes = gr.Number( label='# Nodes')

cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 
#%%
def predict(node_id, neighs):
    # Get subgraph
    subgraph_nodes = list( nx.dfs_preorder_nodes(graph, node_id, int(neighs)) )

    subgraph = nx.subgraph(graph, subgraph_nodes)
    # Get scores
    
    raw_scores, out = model(torch.tensor(list(subgraph_nodes), dtype=torch.long))
    scores = torch.sigmoid(raw_scores).detach().numpy()
    print(scores, raw_scores.shape, out.shape)

    #plot result
    plt.figure(figsize=(10,10))
    pos = nx.kamada_kawai_layout(subgraph)
    
    nx.draw(subgraph, pos, node_color = scores, cmap=cmap, vmin=0, vmax=1)
    plt.savefig("graph.png", format="PNG", )
    
    return 'graph.png', len(subgraph.edges), len(subgraph.nodes)

demo = gr.Interface(fn=predict, inputs=[gr_node_id, gr_neighs_slider], outputs=[gr_graph, gr_edges, gr_nodes])

#%%
demo.launch(share=True)

# %%
