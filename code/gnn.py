import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GIN, GraphNorm
from torch_geometric.utils import dropout_edge 

from util import initialize_weights, global_mean_pool
from torch.nn.utils import spectral_norm

class Action_GNN(torch.nn.Module):
    def __init__(
        self, 
        num_nodes: int = 15,
        input_length:int = 1920, 
        out_channels:int = 256, 
        kernel_size:int = 5,
        add_self_loops:bool=True,
        device: str='cuda',
    ):

        super(Action_GNN, self).__init__()
        self.num_nodes = num_nodes
        self.input_length = input_length
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.add_self_loops = add_self_loops
        self.length = self.input_length - self.kernel_size + 1
        self.length_2 = self.input_length - 2*self.kernel_size + 2
        self.length_3 = self.input_length - 3*self.kernel_size + 3
        self.Conv1D_1 = spectral_norm(nn.Conv1d(in_channels=num_nodes, out_channels=16*num_nodes, kernel_size=kernel_size, groups=num_nodes).to(device))
        self.Conv1D_2 = spectral_norm(nn.Conv1d(in_channels=16*num_nodes, out_channels=32*num_nodes, kernel_size=kernel_size, groups=num_nodes).to(device))
        self.Conv1D_3 = spectral_norm(nn.Conv1d(in_channels=32*num_nodes, out_channels=num_nodes, kernel_size=kernel_size, groups=num_nodes).to(device))

        self.GCN = GCNConv(in_channels=self.length_3, out_channels=out_channels, add_self_loops=add_self_loops).to(device)
        self.lin1 = torch.nn.Linear(self.input_length, self.out_channels)
        self.graph_norm = GraphNorm(out_channels).to(device)

        self.output_layer = torch.nn.Linear(self.out_channels, 16)
        self.device = device

        initialize_weights(self)

    def forward(
        self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight : torch.FloatTensor = None, 
        drop_edge : bool = False, p : float = 0.1
    ) -> torch.FloatTensor:
        
        ################### Conv1D 1st layer ##################
        conv_out1 = self.Conv1D_1(X) # [seq, channel, conv1d_output]
        conv_out1 = F.leaky_relu(conv_out1)

        ################### Conv1D 2nd layer #################
        conv_out2 = self.Conv1D_2(conv_out1) # [seq_length, channel, conv1d_output]
        conv_out2 = F.leaky_relu(conv_out2)

        ############### Conv1D 3rd layer ################
        conv_out3 = self.Conv1D_3(conv_out2) # [seq_length, channel, conv1d_output]
        conv_out3 = F.leaky_relu(conv_out3)
        ################## GCN Conv ########################
        T = torch.zeros(conv_out3.size(0), conv_out3.size(1), self.out_channels).to(self.device)

        for t in range(conv_out3.size(0)): # sequence-wise GCN
            if drop_edge:
                edge_index, _ = dropout_edge(edge_index = edge_index, p = 0.1, force_undirected=True, training=True)
            T[t] = self.GCN(conv_out3[t], edge_index, edge_weight)

        T = self.graph_norm(T)
        T = F.elu(T+self.lin1(X))

        output = global_mean_pool(T, batch=None)
        class_output = self.output_layer(output)
        return output, class_output


class Body_GNN(torch.nn.Module):
    def __init__(
        self, 
        num_nodes: int = 4,
        input_length:int = 240, 
        out_channels:int = 256, 
        kernel_size:int = 3,
        add_self_loops:bool=True,
        device: str='cuda',
    ):

        super(Body_GNN, self).__init__()
        self.num_nodes = num_nodes
        self.input_length = input_length
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.add_self_loops = add_self_loops
        self.length = self.input_length - self.kernel_size + 1
        self.length_2 = self.input_length - 2*self.kernel_size + 2

        self.Conv1D_1 = spectral_norm(nn.Conv1d(in_channels=num_nodes, out_channels=16*num_nodes, kernel_size=kernel_size, groups=num_nodes).to(device))
        self.Conv1D_2 = spectral_norm(nn.Conv1d(in_channels=16*num_nodes, out_channels=num_nodes, kernel_size=kernel_size, groups=num_nodes).to(device))
        
        self.GCN = GCNConv(in_channels=self.length_2, out_channels=out_channels, add_self_loops=add_self_loops).to(device)
        self.lin1 = torch.nn.Linear(self.input_length, self.out_channels)
        self.graph_norm = GraphNorm(out_channels).to(device)
        self.output_layer = torch.nn.Linear(self.out_channels, 16)
        initialize_weights(self)

    def forward(
        self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight : torch.FloatTensor = None, 
        drop_edge : bool = False, p : float = 0.1
    ) -> torch.FloatTensor:
            
        ################### Conv1D 1st layer ##################
        conv_out1 = self.Conv1D_1(X).to(X.device) # [seq, channel, conv1d_output]
        conv_out1 = F.leaky_relu(conv_out1)

        ################### Conv1D 2nd layer #################
        conv_out2 = self.Conv1D_2(conv_out1).to(conv_out1.device) # [seq_length, channel, conv1d_output]
        conv_out2 = F.leaky_relu(conv_out2)

        ################## GCN Conv ########################
        T = torch.zeros(conv_out2.size(0), conv_out2.size(1), self.out_channels).to(conv_out2.device)

        for t in range(conv_out2.size(0)): # sequence-wise GCN
            if drop_edge:
                edge_index, _ = dropout_edge(edge_index = edge_index, p = 0.1, force_undirected=True, training=True)
            T[t] = self.GCN(conv_out2[t], edge_index, edge_weight).to(X.device)

        T = self.graph_norm(T)
        T = F.elu(T+self.lin1(X))

        output = global_mean_pool(T, batch=None)
        class_output = self.output_layer(output)
        return output, class_output

