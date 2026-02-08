from abc import abstractmethod, ABC
import torch
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, APPNP
from utils import node_based_neg_sampling
from torch.nn import Linear
import torch.nn.functional as F

class LinkPredictionModel(ABC):

    @abstractmethod
    def train(self, data):
        """
        Train the link prediction model on the provided data.
        
        :param data: Training data for the model.
        """
        pass

    @abstractmethod
    def test(self, data):
        """
        Test the link prediction model on the provided data.
        
        :param data: Testing data for the model.
        :return: Evaluation metrics or results of the test.
        """
        pass


class GAENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, conv_layer = 'GCNConv', p_dropout = 0.5):
        super().__init__()
        self.conv_layer = conv_layer
        self.dropout = p_dropout
        if conv_layer == 'GCNConv':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif conv_layer == 'SAGEConv':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        elif conv_layer == 'GraphConv':
            self.conv1 = GraphConv(in_channels, hidden_channels)
            self.conv2 = GraphConv(hidden_channels, out_channels)
        elif conv_layer == 'APPNP':
            self.conv1 = Linear(in_channels, hidden_channels)
            self.conv2 = Linear(hidden_channels, out_channels)
            self.prop = APPNP(10, 0.1)
        else:
            raise ValueError(f"Unsupported conv_layer: {conv_layer}")

    def encode(self, x, edge_index):
        if self.conv_layer == 'APPNP':
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv1(x).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x).relu()
            x = self.prop(x, edge_index)
            return x
        else:   
            x = self.conv1(x, edge_index).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return x

    def decode(self, z, edge_label_index):
        
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        #return the indices of a non-zero element
        return (prob_adj > 0.5).nonzero(as_tuple=False).t()

class GraphAutoencoderLP(LinkPredictionModel):
    """
    Graph Autoencoder for Link Prediction.
    
    This class implements a graph autoencoder model for link prediction tasks.
    It inherits from the LinkPredictionModel abstract base class.
    """

    def __init__(self, in_channels, hidden_channels = 128, out_channels = 64, optimizer=None, criterion=None, conv_layer = 'GCN_Conv', p_dropout = 0.5):
        """Initialize the Graph Autoencoder for link prediction.
        :param in_channels: Number of input features for each node.
        :param hidden_channels: Number of hidden channels in the GCN layers.
        :param out_channels: Number of output channels for the encoded representation.
        """
        super().__init__()
        # Initialize model parameters and architecture here
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GAENet(in_channels, hidden_channels, out_channels, conv_layer, p_dropout).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01) if optimizer is None else optimizer
        self.device = device
        self.criterion = torch.nn.BCEWithLogitsLoss() if criterion is None else criterion

    def train(self, train_data, use_sampler=True):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encode(train_data.x, train_data.edge_index)

        # We perform a new round of negative sampling for every training epoch:
        train_data_with_neg = node_based_neg_sampling(train_data, neg_sampling_ratio=1.0, sampling_scheme='random') if use_sampler else train_data

        out = self.model.decode(z, train_data_with_neg.edge_label_index).view(-1)
        loss = self.criterion(out, train_data_with_neg.edge_label)
        loss.backward()
        self.optimizer.step()
        return loss

    @torch.no_grad()
    def test(self, data):
        self.model.eval()
        z = self.model.encode(data.x, data.edge_index)
        out = self.model.decode(z, data.edge_label_index).view(-1).sigmoid()
        return out