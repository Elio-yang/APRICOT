from torch import nn
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv, GlobalAttention, TransformerConv, GCNConv
from torch_geometric.nn.models import JumpingKnowledge
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.inits import reset


class MyGlobalAttention(torch.nn.Module):
    r"""Global soft attention layer from the `"Gated Graph Sequence Neural
    Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathrm{softmax} \left(
        h_{\mathrm{gate}} ( \mathbf{x}_n ) \right) \odot
        h_{\mathbf{\Theta}} ( \mathbf{x}_n ),

    where :math:`h_{\mathrm{gate}} \colon \mathbb{R}^F \to
    \mathbb{R}` and :math:`h_{\mathbf{\Theta}}` denote neural networks, *i.e.*
    MLPS.

    Args:
        gate_nn (torch.nn.Module): A neural network :math:`h_{\mathrm{gate}}`
            that computes attention scores by mapping node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, 1]`, *e.g.*,
            defined by :class:`torch.nn.Sequential`.
        nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, out_channels]`
            before combining them with the attention scores, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
    """

    def __init__(self, gate_nn, nn=None):
        super(MyGlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, x, batch, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out, gate

    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)


class Model(nn.Module):

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 num_heads: int,
                 num_layers: int,
                 num_classes: int,
                 dropout: float,
                 conv_type: str,
                 jkn_type: str,
                 activation_type: str,
                 glob_att: bool):
        super(Model, self).__init__()

        self.drop = dropout
        self.num_layers = num_layers
        self.glob_att = glob_att

        # conv layers
        if conv_type == "transformer":
            conv_class = TransformerConv
        elif conv_type == "gcn":
            conv_class = GCNConv
        elif conv_type == "gat":
            conv_class = GATConv
        else:
            raise NotImplementedError()

        # activations
        if activation_type == "relu":
            self.activation = F.relu
        else:
            self.activation = F.elu

        # backbones

        # output shape is [hidden_features * num_heads]
        self.conv1 = conv_class(
            in_features,
            hidden_features,
            heads=num_heads,
            dropout=dropout
        )

        # cause conv1 output size, so in_channels = hidden_features * num_heads here
        self.convs = nn.ModuleList([
            conv_class(
                hidden_features * num_heads,
                hidden_features,
                heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # output size is [out_features * num_heads]
        # not used actually
        # self.conv2 = conv_class(
        #     hidden_features * num_heads,
        #     out_features,
        #     heads=num_heads,
        #     dropout=dropout
        # )

        self.jkn = JumpingKnowledge(jkn_type, hidden_features, num_layers=2)

        if self.glob_att:
            self.gate_nn = Sequential(
                Linear(hidden_features * num_heads, hidden_features),
                ReLU(),
                Linear(hidden_features, 1)
            )
            self.glob = MyGlobalAttention(self.gate_nn, None)

        if self.glob_att:
            # prediction
            self.mlp = nn.Sequential(
                nn.Linear(hidden_features * num_heads + 1, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, num_classes)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_features * num_heads, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, num_classes)
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        """
        * `x` is the features vectors of shape `[n_nodes, in_features]`
        * `adj_mat` is the adjacency matrix of the form
         `[n_nodes, n_nodes, n_heads]` or `[n_nodes, n_nodes, 1]`
        """
        print(f"start shape {x.shape}")
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        print(f"after conv 1 shape {x.shape}")
        hlist = [x]
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.drop, training=self.training)
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = self.activation(x)
            print(x.shape)
            hlist.append(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        print(f"after conv layers shape {x.shape}")
        hlist.append(x)
        for j in hlist:
            x = self.jkn(hlist)
        print(f"after jkn shape {x.shape}")
        # TODO: FIX SHAPE CONFLICT HERE
        #   MAY NEED BETTER GPU
        if self.glob_att:
            y, node_att_scores = self.glob(x, batch)
            x = torch.cat([x, node_att_scores], dim=1)
            print(f"node_score shape {node_att_scores.shape}")

        print(f"after global att shape {x.shape}")
        x = self.mlp(x)
        print(f"final shape {x.shape}")
        return x
