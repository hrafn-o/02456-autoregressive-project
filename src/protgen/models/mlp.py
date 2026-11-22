import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,vocab_size=23,context=3, emb_dim=64,hidden=256):
        super().__init__()
        self.context=context
        self.emb=nn.Embedding(vocab_size,emb_dim,padding_idx=0)
        self.net=nn.Sequential(
            nn.Linear(context*emb_dim,hidden), nn.ReLU(),
            nn.Linear(hidden,hidden), nn.ReLU(),
            nn.Linear(hidden,vocab_size)
        )
    def forward(self,x):
        B,T=x.shape
        xp=F.pad(x,(2,0), value=0)
        a=xp[:,0:T]
        b=xp[:,1:T+1]
        c=xp[:,2:T+2]
        e=torch.cat([self.emb(a),self.emb(b),self.emb(c)],dim=-1)
        return self.net(e)