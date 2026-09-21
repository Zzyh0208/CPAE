import torch

def tensor(value, device, dtype=torch.float32): return torch.as_tensor(value, dtype=dtype, device=device)

def encode_observation(agent, observation, device):
    history = tensor(observation.history, device).unsqueeze(0)
    adjacency = tensor(observation.adjacency, device).unsqueeze(0)
    return agent.representations(history, adjacency)
