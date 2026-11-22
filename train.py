import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from protgen.data import ProteinDataset, IGNORE_INDEX
from protgen.models.mlp import MLP

train_data=ProteinDataset("data/train.csv")
val_data=ProteinDataset("data/val.csv")
train_loader=DataLoader(train_data, batch_size=128, shuffle=True)
val_loader=DataLoader(val_data, batch_size=128)

model=MLP()
opt=torch.optim.Adam(model.parameters(), lr=1e-3)

def loss_fn(model, x, x_shifted):
    B,T=x.shape
    V=train_data.vocab_size
    logits=model(x)
    loss=F.cross_entropy(logits.view(-1,V), x_shifted.view(-1), ignore_index=IGNORE_INDEX)
    return loss

import os
os.makedirs("checkpoints", exist_ok=True)
best = float("inf")

for epoch in range(1,11):
    model.train()
    for x,x_shifted,y,length in train_loader:
        loss=loss_fn(model, x, x_shifted)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    total=0.0
    with torch.no_grad():
        for x,x_shifted,y,length in val_loader:
            loss=loss_fn(model, x, x_shifted)
            total+=loss.item()*x.size(0)
    val_loss=total/len(val_data)
    if val_loss < best:
        best = val_loss
        torch.save(model.state_dict(), "checkpoints/mlp.pt")
    print(f'epoch{epoch}|val loss{val_loss:.4f}|val perplexity:{math.exp(val_loss):.2f}')