import torch.nn as nn
class GRU(nn.Module):
    def __init__(self,vocab_size=23,emb_dim=64,hidden=256, layers=2):
        super().__init__()
        self.emb=nn.Embedding(vocab_size,emb_dim,padding_idx=0)
        self.gru=nn.GRU(emb_dim,hidden,layers,batch_first=True)
        self.out=nn.Linear(hidden,vocab_size)
    def forward(self,x):
        e=self.emb(x)
        h,_=self.gru(e)
        return self.out(h)
    def features(self,x):
        e=self.emb(x)
        h,_=self.gru(e)
        return h
    def forward(self,x):
        return self.out(self.features(x))