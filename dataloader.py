import os
from torch.utils.data import Dataset, DataLoader
import torch
from torch_geometric.data import Data
import  numpy as np
type_map = {
    "LESS": 0,
    "MID": 1,
    "MORE": 2,
    "OUTSIDE": 3,
    "DIRECT": 4
}


class PGOGraphDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = os.listdir(data_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]

        # Load the .cites file
        cites_path = os.path.join(self.data_dir, sample_name, sample_name + ".link")
        with open(cites_path, 'r') as f:
            num_edges = int(f.readline().strip().split(',')[0])
            edge_index = []
            for i in range(num_edges):
                edge = f.readline().strip('\n').split('\t')
                edge_index.append([int(edge[0]), int(edge[1])])
            edge_index = torch.LongTensor(edge_index).t()
            edge_index = edge_index.reshape(2, -1)

        # Load the .content file
        content_path = os.path.join(self.data_dir, sample_name, sample_name + ".content")
        with open(content_path, 'r') as f:
            num_nodes = int(f.readline().strip().split(',')[0])
            nodes = []
            labels = []
            for i in range(num_nodes):
                node_info = f.readline().strip('\n').split(',')

                # data.x
                node_features = [float(x) for x in node_info[1:-1]]
                shape = 512 - len(node_features)

                # seed set in config
                rand_feature = np.random.normal(loc=0.5, scale=0.5, size=shape)
                rand_feature = np.clip(rand_feature, 0, 1)
                node_features.extend(rand_feature)
                # data.y
                node_label = type_map[node_info[-1]]
                nodes.append(node_features)
                labels.append(node_label)
            # features for all nodes
            nodes = torch.FloatTensor(nodes)
            # labels for cor, nodes
            labels = torch.LongTensor(labels)

        # max nodes per batch
        batch_size = 32
        batch_indices = torch.arange(0, num_nodes, batch_size)
        batch_indices = torch.repeat_interleave(batch_indices, torch.ceil(torch.tensor(num_nodes / batch_size)).long())
        batch_indices = batch_indices[:num_nodes]

        data = Data(x=nodes, edge_index=edge_index, y=labels, batch=batch_indices, name=sample_name)

        return data
