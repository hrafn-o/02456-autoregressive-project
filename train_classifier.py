import torch 
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from protgen.data import ProteinDataset
from protgen.models.gru import GRU

train_loader=DataLoader(ProteinDataset("data/train.csv"), batch_size=128, shuffle=True)
backbone=GRU()
backbone.load_state_dict(torch.load("checkpoints/gru.pt"))
backbone.eval()
for p in backbone.parameters():
    p.requires_grad=False
head=nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256,3))
opt=torch.optim.Adam(head.parameters(), lr=1e-3)

def clf_loss(x,y):
    with torch.no_grad():
        h=backbone.features(x)
    logits=head(h)
    mask=x!=0
    labels=y[:,None].expand_as(x)
    return F.cross_entropy(logits[mask],labels[mask])

val_loader=DataLoader(ProteinDataset('data/val.csv'),batch_size=128)
best=float('inf')
for epoch in range(1,11):
    for x,x_shifted,y,length in train_loader:
        loss=clf_loss(x,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    total=0.0
    with torch.no_grad():
        for x,x_shifted,y,length in val_loader:
            loss=clf_loss(x,y)
            total+=loss.item()*x.size(0)
    val_loss=total/len(val_loader.dataset)
    print(f'epoch {epoch}|val loss {val_loss:.4f}')

    if val_loss<best:
        best=val_loss
        torch.save(head.state_dict(), "checkpoints/gru_head.pt")