import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from protgen.data import ProteinDataset, IGNORE_INDEX
from protgen.models.gru import GRU

train=ProteinDataset("data/train.csv")
loader=DataLoader(train, batch_size=2, shuffle=True)
x,x_shifted,y,length=next(iter(loader))

B,T=x.shape
V=train.vocab_size
model=GRU()
logits = model(x)
print(logits.shape)

loss=F.cross_entropy(logits.view(-1,V), x_shifted.view(-1), ignore_index=IGNORE_INDEX)  
print("loss:", loss.item())