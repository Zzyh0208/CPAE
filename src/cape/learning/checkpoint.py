from pathlib import Path
import torch

def save(path, agent, planner, executor, encoder_optimizer, effect_optimizer, step, configuration):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'agent': agent.state_dict(), 'planner': planner.optimizer.state_dict(), 'executor': executor.optimizer.state_dict(), 'encoder': encoder_optimizer.state_dict(), 'effect': effect_optimizer.state_dict(), 'step': step, 'configuration': configuration.values}, path)

def load(path, agent, planner, executor, encoder_optimizer, effect_optimizer, device):
    value = torch.load(path, map_location=device); agent.load_state_dict(value['agent']); planner.optimizer.load_state_dict(value['planner']); executor.optimizer.load_state_dict(value['executor']); encoder_optimizer.load_state_dict(value['encoder']); effect_optimizer.load_state_dict(value['effect']); return value['step']
