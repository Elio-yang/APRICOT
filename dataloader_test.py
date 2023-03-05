# dataloader test

from torch.utils.data import Dataset
from torch.utils.data import random_split
from config import *

from torch_geometric.data import DataLoader

if __name__ == '__main__':
    conf = PGOConfig()

    setup_seed(conf.seed)

    dataset = PGOGraphDataset(conf.train_dataset_path)

    train_size = int(conf.train_percent * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    for data in train_dataloader:
        # Move the data to the GPU
        nodes, adj_matrix, labels = data.x,data.edge_index,data.y
        print(nodes.shape)
        print(adj_matrix.shape)
        print(labels.shape)
        pass
