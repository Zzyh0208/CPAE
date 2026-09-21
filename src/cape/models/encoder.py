import torch
from torch import nn

class TrafficEncoder(nn.Module):
    def __init__(self, features, hidden, heads):
        super().__init__()
        self.input = nn.Linear(features, hidden)
        self.temporal = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden, hidden * 2, heads, batch_first=True, activation='gelu'), 2)
        self.graph = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])

    def forward(self, history, adjacency):
        batch, steps, nodes, features = history.shape
        values = self.input(history.permute(0, 2, 1, 3).reshape(batch * nodes, steps, features))
        nodes = self.temporal(values)[:, -1].reshape(batch, nodes, -1)
        for layer in self.graph: nodes = torch.nn.functional.gelu(layer(adjacency @ nodes))
        return nodes, nodes.mean(1)
