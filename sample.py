import torch
import pandas as pd
from protgen.data import ProteinDataset
from protgen.models.mlp import MLP
import torch.nn.functional as F
from protgen.models.gru import GRU

N=256
MIN_LENGTH, MAX_LENGTH=68,127
data=ProteinDataset('data/train.csv')
MODEL='gru'
model=GRU() if MODEL=='gru' else MLP()
model.load_state_dict(torch.load(f'checkpoints/{MODEL}.pt'))
model.eval()
tokens=torch.ones(N,1,dtype=torch.long)

with torch.no_grad():
    for step in range(MAX_LENGTH):
        logits=model(tokens)[:,-1,:]
        logits[:,0]=float("-inf")
        logits[:,1]=float("-inf")
        if step<MIN_LENGTH:
            logits[:,2]=float("-inf")
        probs=F.softmax(logits,dim=-1)
        new=torch.multinomial(probs,1)
        tokens=torch.cat([tokens,new],dim=1)
seqs=[]
for row in tokens.tolist():
    if 2 in row:
        row=row[:row.index(2)]
    seqs.append(data.decode(row))

pd.DataFrame({"sequence":seqs}).to_csv(f"results/sequences/{MODEL}_unguided.csv",index=False)
print(seqs[0])