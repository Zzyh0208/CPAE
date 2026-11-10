import argparse
import copy
import json
from pathlib import Path
import torch
from cape.models.agent import CAPE, Config, load_checkpoint, save_checkpoint
from cape.planning.controller import load_observation, candidates, legal_edits

def planner_terms(model, obs):
    (nodes, global_state) = model.encode(obs['history'], obs['adjacency'])
    pool = candidates(model, nodes, global_state, obs, obs['active'])
    edits = legal_edits(obs['active'], pool, model.config.max_nodes)
    features = model.edit_features(nodes, global_state, obs['active'], obs['previous'], pool, edits)
    return (edits, features)

def planner_loss(model, target, obs, nxt, record, alpha):
    (edits, features) = planner_terms(model, obs)
    action = tuple(record['edit'])
    if action not in edits:
        raise ValueError('Recorded edit is not legal in this observation/candidate pool.')
    commit = bool(record['commit'])
    (reward, discount) = (float(record['reward']), float(record['discount']))
    if commit != (action[0] == 3):
        raise ValueError('Only STOP commits the outer-period reward.')
    if not commit and (reward != 0 or discount != 1):
        raise ValueError('Internal edits require reward=0 and discount=1 in this contract.')
    index = edits.index(action)
    (q1, q2) = (model.planner_q1(features).squeeze(-1), model.planner_q2(features).squeeze(-1))
    with torch.no_grad():
        next_nodes, next_global = model.encode(nxt['history'], nxt['adjacency'])
        next_pool = candidates(model, next_nodes, next_global, nxt, nxt['active'])
        next_edits = legal_edits(nxt['active'], next_pool, model.config.max_nodes)
        next_features = model.edit_features(next_nodes, next_global, nxt['active'],
                                           nxt['previous'], next_pool, next_edits)
        target_nodes, target_global = target.encode(nxt['history'], nxt['adjacency'])
        target_features = target.edit_features(target_nodes, target_global, nxt['active'],
                                              nxt['previous'], next_pool, next_edits)
        probabilities = model.planner_actor(next_features).squeeze(-1).softmax(-1)
        logp = probabilities.clamp_min(1e-12).log()
        next_q = torch.minimum(target.planner_q1(target_features), target.planner_q2(target_features)).squeeze(-1)
        future = (probabilities * (next_q - alpha * logp)).sum()
        expected = reward + discount * (1 - float(record['terminal'])) * future
    critic = (q1[index] - expected).square() + (q2[index] - expected).square()
    logits = model.planner_actor(features).squeeze(-1)
    logp = logits.log_softmax(-1)
    policy = (logp.exp() * (alpha * logp - torch.minimum(q1, q2).detach())).sum()
    return critic + policy

def executor_loss(model, target, obs, nxt, record, alpha):
    (nodes, global_state) = model.encode(obs['history'], obs['adjacency'])
    support = obs['active']
    code = model.code(nodes, global_state, support)
    action = torch.as_tensor(record['action'], dtype=nodes.dtype, device=nodes.device)
    if action.shape != obs['dim_mask'].shape or (action.abs() > 1).any() or (not torch.isfinite(action).all()):
        raise ValueError('Executor replay actions must be finite normalized [N,M,W,D].')
    inputs = model.q_input(nodes, global_state, support, code, action, obs['dim_mask'])
    (q1, q2) = (model.executor_q1(inputs).squeeze(), model.executor_q2(inputs).squeeze())
    with torch.no_grad():
        (nnodes, nglobal) = model.encode(nxt['history'], nxt['adjacency'])
        nsupport = nxt['active']
        ncode = model.code(nnodes, nglobal, nsupport)
        (naction, nlogp) = model.sample(nnodes, nglobal, nsupport, ncode, nxt['dim_mask'])
        target_nodes, target_global = target.encode(nxt['history'], nxt['adjacency'])
        ninputs = target.q_input(target_nodes, target_global, nsupport, ncode, naction, nxt['dim_mask'])
        future = torch.minimum(target.executor_q1(ninputs), target.executor_q2(ninputs)).squeeze() - alpha * nlogp
        expected = float(record['reward']) + float(record['discount']) * (1 - float(record['terminal'])) * future
    critic = (q1 - expected).square() + (q2 - expected).square()
    with torch.no_grad():
        value_target = model.policy_q_expectation(nodes, global_state, support, code, obs['dim_mask'])
    value_loss = (model.execution_value(nodes, global_state, support, code, obs['dim_mask']) - value_target).square()
    q_parameters = list(model.executor_q1.parameters()) + list(model.executor_q2.parameters())
    for p in q_parameters:
        p.requires_grad_(False)
    (sampled, logp) = model.sample(nodes, global_state, support, code, obs['dim_mask'])
    sampled_inputs = model.q_input(nodes, global_state, support, code, sampled, obs['dim_mask'])
    execution_q = torch.minimum(model.executor_q1(sampled_inputs), model.executor_q2(sampled_inputs)).squeeze()
    actor = alpha * logp - execution_q
    for p in q_parameters:
        p.requires_grad_(True)
    return critic + actor + value_loss

def effect_loss(model, obs, record):
    (nodes, global_state) = model.encode(obs['history'], obs['adjacency'])
    (edits, features) = planner_terms(model, obs)
    index = edits.index(tuple(record['edit']))
    q = torch.minimum(model.planner_q1(features), model.planner_q2(features)).squeeze(-1)
    ap = q[index] - q[0]
    from cape.planning.controller import apply_edit
    (support, edited) = (obs['active'], apply_edit(obs['active'], edits[index]))
    (z0, z1) = (model.code(nodes, global_state, support), model.code(nodes, global_state, edited))
    ae = model.execution_value(nodes, global_state, edited, z1, obs['dim_mask']) - model.execution_value(nodes, global_state, support, z0, obs['dim_mask'])
    y = ap.new_tensor(record['paired_effect'])
    loss = torch.nn.functional.huber_loss(ap, y) + torch.nn.functional.huber_loss(ae, y) + model.config.consistency_weight * (ap - ae).square()
    if 'adapted_code' in record:
        adapted = torch.as_tensor(record['adapted_code'], dtype=z1.dtype, device=z1.device)
        if adapted.shape != z1.shape:
            raise ValueError('adapted_code shape mismatch')
        loss = model.config.effect_weight * loss + model.config.distill_weight * (z1 - adapted.detach()).square().sum()
    else:
        loss = model.config.effect_weight * loss
    return loss

def update(model, target, optimizer, obs, nxt, record, alpha=0.2):
    kind = record['kind']
    if kind in ['planner', 'executor']:
        if not 0 <= float(record['discount']) <= 1 or record['terminal'] not in [0, 1, False, True]:
            raise ValueError('Invalid discount or terminal flag.')
        if kind == 'planner':
            loss = planner_loss(model, target, obs, nxt, record, alpha)
        else:
            loss = executor_loss(model, target, obs, nxt, record, alpha)
    elif kind == 'effect':
        loss = effect_loss(model, obs, record)
    else:
        raise ValueError('Unknown replay record kind.')
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite loss; inspect replay data and normalization.')
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    with torch.no_grad():
        for (dst, src) in zip(target.parameters(), model.parameters()):
            dst.lerp_(src, 0.005)
    return float(loss.detach())

def next_observation_path(root, record):
    if record['kind'] in ['planner', 'executor'] and not record['terminal']:
        if 'next_observation' not in record:
            raise ValueError('Nonterminal replay transitions require next_observation.')
    return root / record.get('next_observation', record['observation'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', required=True, help='JSON array of replay records')
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', help='JSON object of Config fields')
    parser.add_argument('--initialize-from', help='Checkpoint used to initialize model weights')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    root = Path(args.index).resolve().parent
    records = json.loads(Path(args.index).read_text(encoding='utf-8'))
    if not records or args.epochs < 1:
        raise ValueError('Need nonempty replay and positive epochs.')
    config = Config(**json.loads(Path(args.config).read_text(encoding='utf-8'))) if args.config else Config()
    model = load_checkpoint(args.initialize_from, args.device)[0] if args.initialize_from else CAPE(config).to(args.device)
    model.train()
    target = copy.deepcopy(model).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    losses = []
    for _ in range(args.epochs):
        for index in torch.randperm(len(records)).tolist():
            record = records[index]
            obs = load_observation(root / record['observation'], model.config, args.device)
            nxt = load_observation(next_observation_path(root, record), model.config, args.device)
            losses.append(update(model, target, optimizer, obs, nxt, record))
    save_checkpoint(args.output, model, {'dataset_index': str(Path(args.index).resolve()), 'updates': len(losses), 'seed': args.seed}, optimizer)
    print(json.dumps({'updates': len(losses), 'last_loss': losses[-1]}))
if __name__ == '__main__':
    main()
