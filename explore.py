from protgen.data import ProteinDataset

train =ProteinDataset("data/train.csv")
print(train)
print(train.vocab)

x, x_shifted, y, length =train[0]
print("x        :", x[:10].tolist())
print("x_shifted:", x_shifted[:10].tolist())
print("family:", y.item(), "| length:", length)
print("last real positions of x        :", x[length - 1:length + 4].tolist())
print("last real positions of x_shifted:", x_shifted[length - 1:length + 4].tolist())