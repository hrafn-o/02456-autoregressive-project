import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from protgen.data import ProteinDataset
from protgen.models.gru import GRU

val_loader=DataLoader(ProteinDataset('data/val.csv'),batch_size=128)

backbone=GRU()
backbone.load_state_dict(torch.load('checkpoints/gru.pt'))
backbone.eval()
head=nn.Sequential(nn.Linear(256,256), nn.ReLU(), nn.Linear(256,3))
head.load_state_dict(torch.load('checkpoints/gru_head.pt'))
head.eval()

T=129
correct=torch.zeros(T)
seen=torch.zeros(T)
with torch.no_grad():
    for x,x_shifted,y,length in val_loader:
        logits=head(backbone.features(x))
        pred=logits.argmax(-1)
        mask=x!=0
        hit=(pred==y[:,None]) & mask
        correct+=hit.sum(0)
        seen+=mask.sum(0)
acc=correct/seen.clamp(min=1)
for k in [1,5,10,20,40,60,80,100]:
    print(f'{k:3d} residues seen: accuracy {acc[k].item():.3f}')