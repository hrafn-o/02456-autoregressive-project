import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from protgen.data import ProteinDataset
from protgen.models.gru import GRU

N=32
MIN_LEN, MAX_LEN=68,127
TARGET=2
LAM=2.0
data=ProteinDataset('data/train.csv')
backbone=GRU()
backbone.load_state_dict(torch.load('checkpoints/gru.pt'))
backbone.eval()
head=nn.Sequential(nn.Linear(256,256), nn.ReLU(), nn.Linear(256,3))
head.load_state_dict(torch.load('checkpoints/gru_head.pt'))
head.eval()

cand=torch.arange(2,23)
K=len(cand)
tokens=torch.ones(N,1,dtype=torch.long)
for LAM in [0.0,2.0,8.0]:
    tokens=torch.ones(N,1,dtype=torch.long)
    with torch.no_grad():
        for step in range(MAX_LEN):
            if step%20==0:
                print(f'lam{LAM} step{step}', flush=True)
            logits=backbone(tokens)[:,-1,:]
            logits[:,0]=float('-inf')
            logits[:,1]=float('-inf')
            if step<MIN_LEN:
                logits[:,2]=float('-inf')
            logp=F.log_softmax(logits,dim=1)

            guide=torch.zeros(N,23)
            if LAM!=0:
                ext=torch.cat([tokens.repeat_interleave(K,dim=0),cand.repeat(N)[:,None]], dim=1)
                h=backbone.features(ext)[:, -1,:]
                g=F.log_softmax(head(h), dim=-1)[:,TARGET].view(N,K)
                guide[:,2:]=g
            scores=logp+LAM*guide
            probs=F.softmax(scores,dim=-1)
            new=torch.multinomial(probs,1)
            tokens=torch.cat([tokens,new],dim=1)

    seqs=[]
    for row in tokens.tolist():
        if 2 in row:
            row=row[:row.index(2)]
        seqs.append(data.decode(row))

    pd.DataFrame({'sequence':seqs}).to_csv(f'results/sequences/gru_HU_lam{LAM:g}.csv',index=False)

    x=torch.zeros(len(seqs),129,dtype=torch.long)
    for i,s in enumerate(seqs):
        ids=[1]+[data.vocab[c] for c in s]+[2]
        x[i,:len(ids)]=torch.tensor(ids)
    with torch.no_grad():
        pred=head(backbone.features(x)).argmax(-1)
    eos=torch.tensor([len(s)+1 for s in seqs])
    frac=(pred[torch.arange(len(seqs)),eos]==TARGET).float().mean()
    print(f'lam {LAM}: {frac:.2%} classified as HU')