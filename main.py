
#%%
import pandas as pd, numpy as np
import torch, torch_geometric, pytorch_lightning


from pytorch_lightning.loggers import WandbLogger

features = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_features.csv', header=None)
classes = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_classes.csv', na_values='unknown')
edges = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')


print(features)
print(classes)
print(edges)


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, ids, y) -> None:
        self.ids = ids
        self.y = y

    def __getitem__(self, index):
        return self.ids[index], self.y[index]

    def __len__(self):
        return len(self.ids)
#%%
class EllipticDataset(pytorch_lightning.LightningDataModule):
    def __init__(self, features, classes, edges, local_features=True):
        super().__init__()
        self.id2txid = dict(enumerate(features[0].to_list()))
        self.txid2id = {v:k for k,v in self.id2txid.items()}
        self.classes_map = {
                0: -1,
                1: 1,
                2: 0
            }

        classes = classes.fillna(0)

        features[0] = features[0].map( lambda x: self.txid2id[x] )
        edges['txId1']    = edges['txId1'].map( lambda x: self.txid2id[x] )
        edges['txId2']    = edges['txId2'].map( lambda x: self.txid2id[x] )
        classes['txId']  = classes['txId'].map( lambda x: self.txid2id[x] ) 
        classes['class']  = classes['class'].map( lambda x: self.classes_map[x] ) 

        features = features.sort_values(0).to_numpy()
        classes = classes.sort_values('txId').to_numpy()
        edges = edges.to_numpy()

        self.n_features = 94 if local_features else 164

        self.ids = torch.arange(len(classes))
        self.X = torch.tensor( features[:,2:self.n_features+2] ).float()
        self.ts = torch.tensor( features[:,1].astype(int) )
        self.y = torch.tensor( classes[:,1].astype(int) )
        self.edges = torch.tensor( edges.astype(int) ).T

        self.valid_mask = self.y > -1
        self.valid_indices = torch.arange(len(self.valid_mask))[self.valid_mask]

        self.pos_weight =  len(self.valid_indices)  / ( self.y == 0 ).sum() 
        self.neg_weight =  len(self.valid_indices)  / ( self.y == 1 ).sum() 

        t = self.pos_weight + self.neg_weight

        self.pos_weight /= t 
        self.neg_weight /= t

        pos_mask = ( self.y == 0 ) * self.pos_weight
        neg_mask = ( self.y == 1 ) * self.neg_weight

        self.valid_weights = (pos_mask + neg_mask)[self.valid_indices]
        
        ds = CombinedDataset(self.ids[self.valid_indices], self.y[self.valid_indices])
        l = len(ds)
        train_l = int(l*0.8)

        train_ds, val_ds = torch.utils.data.random_split(ds, [train_l, l-train_l], generator=torch.Generator().manual_seed(42))
        train_w, _ = torch.utils.data.random_split(self.valid_weights, [train_l, l-train_l], generator=torch.Generator().manual_seed(42))

         
        self._train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=32, sampler= torch.utils.data.WeightedRandomSampler(train_w, len(train_ds), replacement=True))
        self._val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=32)

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader






data = EllipticDataset(features.copy(), classes.copy(), edges.copy())
# loader = torch.utils.data.DataLoader(data, batch_size=32, sampler = data.sampler)


# %%

class GCNModel(pytorch_lightning.LightningModule):
    def __init__(self, data):
        super().__init__()
        self.data = data

        self.gcn_layer1 = torch_geometric.nn.GCNConv(data.n_features, 32)
        self.gelu1 = torch.nn.GELU()
        self.gcn_layer2 = torch_geometric.nn.GCNConv(32, 32)
        self.gelu2 = torch.nn.GELU()
        self.linear_layer = torch.nn.Linear(32,1)

        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, x_ids):
        out = self.gcn_layer1(self.data.X, self.data.edges)
        out = self.gelu1(out)
        out = self.gcn_layer2(out, self.data.edges)
        batch_out = out[x_ids]
        batch_out = self.gelu2(batch_out)
        scores = self.linear_layer(batch_out).squeeze()
        return scores

    def training_step(self, batch, batch_idx) :
        x_ids, y = batch
        out =  self(x_ids)
        loss = self.loss(out, y.float() )

        out_pos = out[y == 1]
        out_neg = out[y == 0]

        prec_pos = (out_pos > 0).sum() / ( (y == 1).sum() + 1e-6)
        prec_neg = (out_neg <= 0).sum() / ( (y == 0).sum() + 1e-6)
        prec = ( (out_pos > 0).sum() + (out_neg <= 0).sum() ) / len(out)
        prec_bal = prec_pos / 2 + prec_neg / 2

        self.log('train/loss', loss)
        self.log('train/prec_pos', prec_pos)
        self.log('train/prec_neg', prec_neg)
        self.log('train/prec', prec)
        self.log('train/prec_bal', prec_bal)


        

        
        

        return loss

    def validation_step(self, batch, batch_idx) :
        x_ids, y = batch
        out =  self(x_ids)
        loss = self.loss(out, y.float() )

        out_pos = out[y == 1]
        out_neg = out[y == 0]

        prec_pos = (out_pos > 0).sum() / ( (y == 1).sum() + 1e-6)
        prec_neg = (out_neg <= 0).sum() / ( (y == 0).sum() + 1e-6)
        prec = ( (out_pos > 0).sum() + (out_neg <= 0).sum() ) / len(out)
        prec_bal = prec_pos / 2 + prec_neg / 2

        self.log('val/loss', loss)
        self.log('val/prec_pos', prec_pos)
        self.log('val/prec_neg', prec_neg)
        self.log('val/prec', prec)
        self.log('val/prec_bal', prec_bal)

        return loss


    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optim


model = GCNModel(data)

logger = WandbLogger(log_model=True, project='btc-xai')

trainer = pytorch_lightning.Trainer(
    log_every_n_steps=1,
    logger=logger,
    check_val_every_n_epoch=1
)

trainer.fit(model, data)

# %%
