import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Actor, WikipediaNetwork
import os.path as osp
from torch_geometric.utils import dropout_edge
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import to_undirected
from torch_sparse import coalesce


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('test2')

def node_based_neg_sampling(data, neg_sampling_ratio=1.0, sampling_scheme='random', query=None):
    if sampling_scheme == 'random':
        edge_label_index = data.edge_label_index
        all_edges = torch.cat([edge_label_index, data.edge_index], dim=-1)
        n_edges = data.edge_label_index.size(1)
        if query is not None:
            query_nodes = torch.tensor(query, dtype=torch.long, device=data.edge_index.device)
        else:
            assert hasattr(data, 'query_nodes'), "Query nodes must be provided or present in the data object."
            query_nodes = data.query_nodes
        n_queries = query_nodes.size(0)
        n_samples_per_node = int(n_edges * neg_sampling_ratio / n_queries)
        all_nodes = torch.unique(all_edges.flatten())
        new_data = data.clone()

        for node in query_nodes:
            edges_for_node = all_edges[:, (all_edges[0] == node) | (all_edges[1] == node)]
            forbidden_nodes = torch.unique(edges_for_node.flatten()) # I could probably do this more efficiently TODO
            avaliable_nodes = all_nodes[~torch.isin(all_nodes, forbidden_nodes)]
            neg_edges_for_node = torch.stack([
                node.repeat(n_samples_per_node),
                avaliable_nodes[torch.randperm(len(avaliable_nodes))[:n_samples_per_node]]
            ], dim=0)
            new_data.edge_label_index = torch.cat([new_data.edge_label_index, neg_edges_for_node], dim=-1)
            new_data.edge_label = torch.cat([new_data.edge_label, 
                                             new_data.edge_label.new_zeros(n_samples_per_node)], dim=0)
            all_edges = torch.cat([all_edges, neg_edges_for_node], dim=-1)
       
        transform = T.ToUndirected()
        new_data = transform(new_data)
    
    elif sampling_scheme == 'degree':
        new_data = data.clone()
    return new_data

def node_query_based_split(data, query, train_ratio=10, val_ratio=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if type(query) is not torch.Tensor:
        query = torch.tensor(query, dtype=torch.long, device=data.edge_index.device)
    import math
    n_queries = len(query)
    n_nodes = data.num_nodes
    train_size = math.ceil(n_queries * train_ratio )
    val_size = math.ceil(n_queries * val_ratio)
    #print(f"Train size: {train_size}, Val size: {val_size}, Queries: {n_queries}, Nodes: {n_nodes}")
    assert train_size + val_size + n_queries <= n_nodes, "Train, test and validation sizes exceed total nodes"

    edge_index = data.edge_index
    edge_index = edge_index[:, data.edge_index[0] < edge_index[1]]
    candidate_nodes = torch.unique(edge_index.flatten())
    candidate_nodes = candidate_nodes[~torch.isin(candidate_nodes, query)]
    k_sample = train_size + val_size
    sampled_nodes = candidate_nodes[torch.randperm(len(candidate_nodes))[:k_sample]]
    train_nodes = sampled_nodes[:train_size]
    val_nodes = sampled_nodes[train_size:train_size + val_size]

    test_data = data.clone()
    cond = torch.isin(edge_index[0], query) | torch.isin(edge_index[1], query)
    test_data.edge_label_index = edge_index[:, cond]
    remaining_edges = edge_index[:, ~cond]
    test_data.edge_index = remaining_edges
    test_data.edge_label = torch.ones(test_data.edge_label_index.size(1), dtype=torch.float, device=device)

    train_data = data.clone()
    cond_train = torch.isin(remaining_edges[0], train_nodes) | torch.isin(remaining_edges[1], train_nodes)
    train_data.edge_index = remaining_edges[:, ~cond_train]
    train_data.edge_label_index = remaining_edges[:, cond_train]
    train_data.edge_label = torch.ones(train_data.edge_label_index.size(1), dtype=torch.float, device=device)
    train_data.query_nodes = train_nodes

    val_data = data.clone()
    cond_val = torch.isin(remaining_edges[0], val_nodes) | torch.isin(remaining_edges[1], val_nodes)
    val_data.edge_index = remaining_edges[:, ~cond_val]
    val_data.edge_label_index = remaining_edges[:, cond_val]
    val_data.edge_label = torch.ones(val_data.edge_label_index.size(1), dtype=torch.float, device=device)
    val_data.query_nodes = val_nodes

    transform = T.ToUndirected()
    train_data = transform(train_data)
    val_data = transform(val_data)
    test_data = transform(test_data)
    test_data.query_nodes = query

    return train_data, val_data, test_data




class WebKB(InMemoryDataset):
    r"""The WebKB datasets used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features are the bag-of-words representation of web pages.
    The task is to classify the nodes into one of the five categories, student,
    project, course, staff, and faculty.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cornell"`,
            :obj:`"Texas"` :obj:`"Washington"`, :obj:`"Wisconsin"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
           'master/new_data')

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']

        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


def load_planetoid_data(name):
    root_path = '../'
    path = osp.join(root_path, 'data', name)
    dataset = Planetoid(path, name, transform=T.NormalizeFeatures())

    return dataset, dataset[0]

def load_actor():
    root_path = '../'
    path = osp.join(root_path, 'data', 'actor')
    dataset = Actor(path, transform=T.NormalizeFeatures())

    return dataset, dataset[0]

def load_cornell():
    dataset = WebKB(root='../data/', name='cornell', transform=T.NormalizeFeatures())
    return dataset, dataset[0]

def load_texas():
    dataset = WebKB(root='../data/', name='texas', transform=T.NormalizeFeatures())
    return dataset, dataset[0]

def load_squirrel():
    preProcDs = WikipediaNetwork(
            root='../data/', name='squirrel', geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
    dataset = WikipediaNetwork(
            root='../data/', name='squirrel', geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
    data = dataset[0]
    data.edge_index = preProcDs[0].edge_index
    return dataset, data

def load_chameleon():
    preProcDs = WikipediaNetwork(
            root='../data/', name='chameleon', geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
    dataset = WikipediaNetwork(
            root='../data/', name='chameleon', geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
    data = dataset[0]
    data.edge_index = preProcDs[0].edge_index
    return dataset, data

def perturb_edges(data, probs=[0, 0.1, 0.2, 0.4]): #TODO
    perturbations = []
    for p in probs:
        edge_index, _ = dropout_edge(data.edge_index, p, force_undirected=True)
        perturbations.append(edge_index)

        
    return probs, perturbations

def calc_density(data):
    n = float(data.num_nodes)
    if data.is_directed():
        e = data.num_edges
        possible = n*n
    else:
        e = data.num_edges/2
        possible = n*(n-1)*0.5
    density = float(e) / possible
    return density

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag == 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(
            rest_index[val_lb:], size=data.num_nodes)
    else:
        val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data