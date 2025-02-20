import os
import sys
sys.path.insert(1, os.getcwd())
from model.ggnn import GatedGraphNeuralNetwork, AdjacencyList
import torch

def main():
    gnn = GatedGraphNeuralNetwork(hidden_size=256, num_edge_types=4,
                                  layer_timesteps=[8, 8, 8, 8, 8], residual_connections={})

    adj_list_type1 = AdjacencyList(node_num=4, adj_list=[(0, 0), (1, 1), (2, 3), (3, 3)], device=gnn.device)
    adj_list_type2 = AdjacencyList(node_num=4, adj_list=[(0, 0), (0, 1)], device=gnn.device)

    node_representations = gnn.compute_node_representations(initial_node_representation=torch.randn(10, 256),
                                                            adjacency_lists=[adj_list_type1, adj_list_type2])

    assert(node_representations.shape == torch.Size([10, 256]))

if __name__ == '__main__':
    main()