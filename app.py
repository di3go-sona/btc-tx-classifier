#%%
import networkx as nx, pandas as pd
import numpy as np
from train import EllipticDataset, GCNModel
import torch
import os


# install pytorch-geometric stuff as it need pytorch to be already intalled before launching the command
os.system('pip3 install torch-scatter ')
os.system('pip3 install torch-sparse ')
os.system('pip3 install torch-cluster ')
os.system('pip3 install torch-spline-conv ')
os.system('pip3 install torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html')


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

print(np.argwhere(data.classes ==1 ))
cls_map = dict(data.classes.tolist())
shapes_map = {-1: 'o', 0: 's', 1: '^'}
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
    scores_map = dict( zip(subgraph_nodes, scores))
    pos = nx.spring_layout(subgraph)

    #plot result
    plt.figure(figsize=(15,15))
    
    #For each node class...
    for shape in set('so^'):
        #...filter and draw the subset of nodes with the same symbol in the positions that are now known through the use of the layout.
        nodelist =  [node for node in filter(lambda x: shapes_map[cls_map[x]] == shape,subgraph.nodes)]
        scorelist = [ 1-scores_map[n] for n in nodelist]
        if len(nodelist):
            nx.draw_networkx_nodes(subgraph,pos,node_shape = shape,  node_color = scorelist, nodelist =nodelist, cmap=cmap, vmin=0, vmax=1, edgecolors=['black'])
    
    # nx.draw_networkx_nodes(subgraph, pos, node_color = scores, cmap=cmap, vmin=0, vmax=1, node_shape=shapes, edgecolors=border_colors, linewidths=2)
    nx.draw_networkx_edges(subgraph, pos)
    
    plt.savefig("/tmp/graph.png", format="PNG", )
    
    return "/tmp/graph.png", len(subgraph.edges), len(subgraph.nodes)

demo = gr.Interface(fn=predict, inputs=[gr_node_id, gr_neighs_slider], outputs=[gr_graph, gr_edges, gr_nodes])

#%%
demo.launch(share=True)

# %%
