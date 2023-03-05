import torch
from torch_geometric.data import Data

# define a custom collate function
def my_collate(batch):
    batch = [dict(sample) for sample in batch]
    out = {}
    for key in batch[0].keys():
        out[key] = torch.stack([sample[key] for sample in batch], dim=0)
    return out

# create some example data
x1 = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.float)
edge_index1 = torch.tensor([[0, 1, 1, 2],
                            [1, 0, 2, 1]], dtype=torch.long)
data1 = Data(x=x1, edge_index=edge_index1)

print(edge_index1.shape)

x2 = torch.tensor([[6, 7], [8, 9], [10, 11]], dtype=torch.float)
edge_index2 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
data2 = Data(x=x2, edge_index=edge_index2)

# create a PyTorch DataLoader with the custom collate function
dataset = [data1, data2]
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=my_collate)

# iterate over the DataLoader
for batch in dataloader:
    print(batch)