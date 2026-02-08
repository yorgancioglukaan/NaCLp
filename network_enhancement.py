import torch
from utils import node_based_neg_sampling
from link_prediction import GraphAutoencoderLP
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling
import numpy as np
from utils import node_query_based_split

class NetworkEnhancer:
    def __init__(self, data, lp_module = None):
        self.data = data
        if lp_module is None:
            lp_module = GraphAutoencoderLP(in_channels=data.num_features, conv_layer='GCNConv', p_dropout=0.9)
        self.lp_module = lp_module
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, train_data = None, sampling_scheme = 'random', epochs=50, lr=0.01, train_ratio = 0.1, neg_ratio=1, query = None):
        if train_data is None:
            if sampling_scheme == 'random':
                train_data = self.data.clone()
                # sample edges from edge_index to create edge_label_index
                edge_index = train_data.edge_index
                n_edges = edge_index.size(1)
                perm = torch.randperm(n_edges)
                n_train_edges = int(n_edges * train_ratio)
                train_edges = edge_index[:, perm[:n_train_edges]]
                train_data.edge_label_index = train_edges
                train_data.edge_label = torch.ones(train_edges.size(1), dtype=torch.float, device=self.device)
                #train_data.edge_index = edge_index[:, perm[n_train_edges:]]
                transform = T.ToUndirected()
                train_data = transform(train_data)
            if sampling_scheme == 'node_based':
                #raise NotImplementedError("Node-based sampling not implemented for automatic training data generation.")
                train_data, _, _ = node_query_based_split(self.data, 
                                                             query=query,
                                                             train_ratio=train_ratio,
                                                             val_ratio=train_ratio)
                
        
        for epoch in range(1,epochs+1):            
            train_data_with_negs = train_data.clone()
            if sampling_scheme == 'random':
                neg_edge_index = negative_sampling(
                        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
                        num_neg_samples=int(train_data.edge_label_index.size(1)*neg_ratio))

                edge_label_index = torch.cat(
                    [train_data.edge_label_index, neg_edge_index],
                    dim=-1,
                )
                edge_label = torch.cat([
                    train_data.edge_label,
                    train_data.edge_label.new_zeros(neg_edge_index.size(1))
                ], dim=0)
                train_data_with_negs.edge_label_index = edge_label_index
                train_data_with_negs.edge_label = edge_label
            elif sampling_scheme == 'node_based':
                #raise NotImplementedError("Node-based sampling not implemented for automatic training data generation.")
                train_data_with_negs = node_based_neg_sampling(train_data, neg_sampling_ratio=neg_ratio, sampling_scheme='random')

            self.lp_module.train(train_data_with_negs, use_sampler=False)

    def enhance(self, target_nodes, enhance_threshold=0.5, enhance_k=None):
        #print("DIAG: Enhancing the network...")
        data = self.data
        enhance_dud = data.clone()
        lp_module = self.lp_module
        # create an edge label index for edges between query nodes and all other nodes
        first = True
        for q in target_nodes:
            from_nodes = torch.full((data.num_nodes,), q, dtype=torch.long, device=data.edge_index.device)
            to_nodes = torch.arange(data.num_nodes, dtype=torch.long, device=data.edge_index.device)
            temp_edge_label_index = torch.stack([from_nodes, to_nodes], dim=0)
            # append to growing list of edge label indices
            if not first:
                enhance_dud.edge_label_index = torch.cat([enhance_dud.edge_label_index, temp_edge_label_index], dim=1)
            else:
                enhance_dud.edge_label_index = temp_edge_label_index
                first = False
        
        result_data = data.clone()
        # get the scores for these edges
        edge_scores = lp_module.test(enhance_dud).cpu().numpy()
        # get the edges with scores above the threshold
        if enhance_k is None:
            high_score_edge_indices = np.where(edge_scores >= enhance_threshold)[0]
        else:
            high_score_edge_indices = np.argsort(edge_scores)[-enhance_k:]
        # get the corresponding edge label indices
        high_score_edge_label_index = enhance_dud.edge_label_index[:, high_score_edge_indices]

        # calculate ratio of edges added
        ratio_of_edges_added = high_score_edge_label_index.size(1) / enhance_dud.edge_label_index.size(1)
        #print(f"DIAG: Ratio of edges added: {ratio_of_edges_added:.4f}")
        #print(f"DIAG: Number of edges added: {high_score_edge_label_index.size(1)}")

        # add these edges to the edge index of test_data if they are not already present
        for idx in range(high_score_edge_label_index.size(1)): 
            edge = high_score_edge_label_index[:, idx].view(2, 1)
            # check if the edge is already present in data.edge_index
            if not ((data.edge_index[0] == edge[0]) & (data.edge_index[1] == edge[1])).any():
                result_data.edge_index = torch.cat([result_data.edge_index, edge, torch.flipud(edge)], dim=1) # add reverse edge as well

        return result_data
        