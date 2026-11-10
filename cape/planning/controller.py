import argparse
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import time
import copy
import numpy as np
import torch
from cape.models.agent import load_checkpoint

def load_observation(path, config, device='cpu'):
    with np.load(path, allow_pickle=False) as data:
        required = ['history', 'adjacency', 'legal', 'active', 'previous', 'lower', 'upper', 'dim_mask']
        if not set(required) <= set(data.files):
            raise ValueError('Missing fields: ' + str(set(required) - set(data.files)))
        values = {key: np.array(data[key]) for key in required}
    history = values['history']
    if history.ndim != 3 or history.shape[0] != config.history or history.shape[2] != config.features:
        raise ValueError('history must be [configured history, nodes, features].')
    n = history.shape[1]
    shapes = {'adjacency': (n, n), 'legal': (n, config.measures), 'active': (n, config.measures), 'previous': (n, config.measures)}
    full = (n, config.measures, config.waves, config.action_dim)
    shapes.update({key: full for key in ['lower', 'upper', 'dim_mask']})
    for (key, shape) in shapes.items():
        if values[key].shape != shape:
            raise ValueError(f'{key}: expected {shape}, got {values[key].shape}')
    for (key, value) in values.items():
        if not np.isfinite(value).all():
            raise ValueError(f'{key} contains nonfinite values.')
    for key in ['legal', 'active', 'previous', 'dim_mask']:
        if not np.isin(values[key], [0, 1]).all():
            raise ValueError(f'{key} must be boolean.')
    if np.any(values['active'].astype(bool) & ~values['legal'].astype(bool)):
        raise ValueError('Active pairs must remain legal in the supplied state.')
    if np.any(values['upper'] < values['lower']):
        raise ValueError('upper < lower')
    result = {}
    for (key, value) in values.items():
        dtype = torch.bool if key in ['legal', 'active', 'previous', 'dim_mask'] else torch.float32
        result[key] = torch.as_tensor(value, dtype=dtype, device=device)
    return result

def candidates(model, nodes, global_state, obs, support):
    if 'candidate_pool' in obs:
        return obs['candidate_pool'].tolist()
    active = support.flatten().nonzero().flatten().tolist()
    limit = model.config.max_candidates
    if len(active) > limit:
        raise ValueError('Active pair count exceeds candidate cap; cannot retain all active pairs.')
    scores = model.pair_scores(nodes, global_state, support, obs['previous'])
    order = scores.masked_fill(~obs['legal'].flatten(), -torch.inf).argsort(descending=True)
    chosen = list(active)
    for index in order.tolist():
        if len(chosen) == limit:
            break
        if obs['legal'].flatten()[index] and index not in chosen:
            chosen.append(index)
    return chosen

def apply_edit(support, edit):
    changed = support.clone().flatten()
    if edit[1] >= 0:
        changed[edit[1]] = False
    if edit[2] >= 0:
        changed[edit[2]] = True
    return changed.reshape_as(support)

def legal_edits(support, pool, max_nodes):
    selected = support.flatten().nonzero().flatten().tolist()
    inactive = [index for index in pool if index not in selected]
    edits = [(3, -1, -1)]
    edits.extend(((1, index, -1) for index in selected))
    edits.extend(((0, -1, index) for index in inactive))
    edits.extend(((2, old, new) for old in selected for new in inactive))
    return [edit for edit in edits if apply_edit(support, edit).any(-1).sum().item() <= max_nodes
            and apply_edit(support, edit).sum().item() <= len(pool)]

@contextmanager
def frozen(model):
    parameters = list(model.parameters())
    flags = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for (p, flag) in zip(parameters, flags):
            p.requires_grad_(flag)

def synchronized_time(model):
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize(next(model.parameters()).device)
    return time.perf_counter()

def calibrate(model, nodes, global_state, reference, edited, obs, probe_provider, deadline=None):
    nodes, global_state = nodes.detach(), global_state.detach()
    with torch.no_grad():
        initial = model.code(nodes, global_state, edited)
        reference_code = model.code(nodes, global_state, reference)
        baseline_value = model.execution_value(nodes, global_state, reference, reference_code, obs['dim_mask'])
    start = synchronized_time(model)
    with frozen(model), torch.no_grad():
        batch = probe_provider(model=model, observation=obs, reference=reference.clone(), edited=edited.clone(), code=initial.detach().clone(), pairs=model.config.probe_pairs)
    probe_seconds = synchronized_time(model) - start
    records = batch['records']
    if len(records) != model.config.probe_pairs:
        raise ValueError('Unexpected number of matched probe pairs.')
    effects = initial.new_tensor([r['edited_return'] - r['reference_return'] for r in records])
    actions = torch.stack([torch.as_tensor(r['edited_action'], device=initial.device, dtype=initial.dtype) for r in records])
    if actions.shape != (model.config.probe_pairs,) + tuple(obs['dim_mask'].shape) or not torch.isfinite(actions).all():
        raise ValueError('Invalid sampled probe actions.')
    if not torch.isfinite(effects).all() or (actions.abs() > 1).any():
        raise ValueError('Invalid effects or normalized actions.')
    calls = batch['additional_calls']
    if not isinstance(calls, int) or calls < 0:
        raise ValueError('additional_calls must be an observed nonnegative integer.')
    code = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([code], lr=model.config.calibration_lr)
    start = synchronized_time(model)
    with frozen(model):
        for _ in range(model.config.calibration_steps):
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError('Code adaptation deadline exceeded.')
            logps = torch.stack([model.log_prob(nodes, global_state, edited, code, action, obs['dim_mask']) for action in actions])
            policy_loss = -(effects.detach() * logps).mean()
            (mean, _) = model.sample(nodes, global_state, edited, code, obs['dim_mask'], deterministic=True)
            mask = edited[..., None, None] & obs['dim_mask']
            difference = (mean - actions[effects.argmax()]).square()
            action_loss = difference[mask].sum() if mask.any() else difference.sum() * 0
            advantage = model.execution_value(nodes, global_state, edited, code, obs['dim_mask']) - baseline_value
            value_loss = torch.nn.functional.huber_loss(advantage.expand_as(effects), effects)
            loss = policy_loss + model.config.calibration_action_weight * action_loss + model.config.calibration_value_weight * value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    adaptation_seconds = synchronized_time(model) - start
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError('Distillation deadline exceeded.')
    start = synchronized_time(model)
    generator_state = copy.deepcopy(model.generator.state_dict())
    generator_optimizer = torch.optim.Adam(model.generator.parameters(), lr=0.0003)
    distill = (model.code(nodes, global_state, edited) - code.detach()).square().sum()
    generator_optimizer.zero_grad()
    distill.backward()
    generator_optimizer.step()
    if deadline is not None and synchronized_time(model) >= deadline:
        model.generator.load_state_dict(generator_state)
        raise TimeoutError('Distillation exceeded decision deadline; update discarded.')
    with torch.no_grad():
        advantage = model.execution_value(nodes, global_state, edited, code.detach(), obs['dim_mask']) - baseline_value
    return (code.detach(), float(advantage), {'probe_seconds': probe_seconds, 'adaptation_seconds': adaptation_seconds, 'distill_seconds': synchronized_time(model) - start, 'additional_calls': calls, 'paired_effect': float(effects.mean()), 'calibration_records': len(records), 'code_updates': model.config.calibration_steps})

def decide(model, obs, probe_provider=None, projector=None, stochastic=False, deadline=None, capture=False,
           coverage_counts=None, operation_counts=None, collect_pairs=False, action_preparer=None):
    model.eval()
    start = synchronized_time(model)
    with torch.no_grad():
        (nodes, global_state) = model.encode(obs['history'], obs['adjacency'])
    support = obs['active'].clone()
    if support.any(-1).sum() > model.config.max_nodes:
        raise ValueError('Inherited support exceeds active-node limit.')
    (trace, timing, calibrated) = ([], {}, False)
    transitions = []
    effect_records = []
    timed_out = False
    calibration_attempted = False
    calibration_seconds = 0.
    (override_support, override_code) = (None, None)
    corrections = {}
    for iteration in range(model.config.max_edits):
        if deadline is not None and time.perf_counter() >= deadline:
            timed_out = True
            break
        with torch.no_grad():
            pool = candidates(model, nodes, global_state, obs, support)
            if coverage_counts is not None:
                active_indices = support.flatten().nonzero().flatten().tolist()
                legal_indices = obs['legal'].flatten().nonzero().flatten().tolist()
                unseen = sorted((i for i in legal_indices if i not in active_indices), key=lambda i: coverage_counts.flat[i])
                pool = active_indices + unseen[:max(0, model.config.max_candidates - len(active_indices))]
            edits = legal_edits(support, pool, model.config.max_nodes)
            features = model.edit_features(nodes, global_state, support, obs['previous'], pool, edits)
            q = torch.minimum(model.planner_q1(features), model.planner_q2(features)).squeeze(-1)
            advantages = q - q[0]
            support_key = tuple(support.flatten().nonzero().flatten().tolist())
            for index, edit in enumerate(edits):
                saved = corrections.get((support_key, edit))
                if saved is not None:
                    advantages[index] = min(float(advantages[index]), saved) if saved > 0 else -torch.inf
            if operation_counts is not None:
                operations = sorted(set(edit[0] for edit in edits))
                operation = min(operations, key=lambda value: operation_counts[value])
                available = [i for i, edit in enumerate(edits) if edit[0] == operation and torch.isfinite(advantages[i])]
                if not available:
                    selected = 0
                elif coverage_counts is not None:
                    selected = min(available, key=lambda i: coverage_counts.flat[edits[i][2] if edits[i][2] >= 0 else max(edits[i][1], 0)])
                else:
                    selected = available[int(torch.randint(len(available), ()).item())]
                operation_counts[operation] += 1
            elif stochastic:
                logits = model.planner_actor(features).squeeze(-1)
                logits = logits.masked_fill(~torch.isfinite(advantages), -torch.inf)
                selected = int(torch.distributions.Categorical(logits=logits).sample())
            else:
                selected = int(advantages.argmax())
        event = {'iteration': iteration, 'proposed': edits[selected], 'A_P': float(advantages[selected])}
        if selected == 0:
            trace.append(dict(event, committed=(3, -1, -1)))
            break
        edited = apply_edit(support, edits[selected])
        with torch.no_grad():
            z0 = model.code(nodes, global_state, support)
            z1 = model.code(nodes, global_state, edited)
            ae = model.execution_value(nodes, global_state, edited, z1, obs['dim_mask']) - model.execution_value(nodes, global_state, support, z0, obs['dim_mask'])
            ap = advantages[selected]
            disagreement = float((ap - ae).abs() / (ap.abs() + ae.abs() + 1e-06))
        event.update(A_E=float(ae), R=disagreement)
        if probe_provider is not None and (disagreement > model.config.threshold or collect_pairs) and not calibration_attempted:
            calibration_attempted = True
            calibration_start = synchronized_time(model)
            try:
                (override_code, ae_cal, timing) = calibrate(model, nodes, global_state, support, edited, obs, probe_provider, deadline)
            except TimeoutError:
                timed_out = True
                event.update(timeout=True, committed=edits[selected], calibration_skipped=True)
                trace.append(event)
                if capture:
                    transitions.append((support.clone(), edits[selected], edited.clone(), pool))
                support = edited
                break
            finally:
                calibration_seconds += synchronized_time(model) - calibration_start
            override_support = edited.clone()
            calibrated = True
            corrected = min(float(ap), ae_cal)
            corrections[(support_key, edits[selected])] = corrected
            event.update(A_E_cal=ae_cal, corrected_advantage=corrected, calibrated=True)
            if capture:
                effect_records.append({'support': support.detach().cpu(), 'edit': edits[selected],
                    'pool': list(pool), 'paired_effect': timing['paired_effect'],
                    'adapted_code': override_code.detach().cpu().tolist()})
            advantages = advantages.clone()
            advantages[selected] = corrected if corrected > 0 else -torch.inf
            selected = int(advantages.argmax())
        event['committed'] = edits[selected]
        trace.append(event)
        if selected == 0:
            break
        if capture:
            transitions.append((support.clone(), edits[selected], apply_edit(support, edits[selected]), pool))
        support = apply_edit(support, edits[selected])
        if coverage_counts is not None:
            coverage_counts += support.cpu().numpy()
    with torch.no_grad():
        use_override = override_support is not None and torch.equal(support, override_support)
        code = override_code if use_override else model.code(nodes, global_state, support)
        (normalized, _) = model.sample(nodes, global_state, support, code, obs['dim_mask'], deterministic=not stochastic)
        bounded = obs['lower'] + (normalized + 1) * 0.5 * (obs['upper'] - obs['lower'])
        bounded = bounded * (support[..., None, None] & obs['dim_mask'])
    if projector is not None:
        bounded = projector(observation=obs, support=support, bounded_actions=bounded)
        if not isinstance(bounded, torch.Tensor) or bounded.shape != obs['lower'].shape:
            raise ValueError('Projector must return an action tensor with the original shape.')
        mask = support[..., None, None] & obs['dim_mask']
        if not torch.isfinite(bounded).all() or ((bounded < obs['lower']) & mask).any() or ((bounded > obs['upper']) & mask).any():
            raise ValueError('Projected actions are nonfinite or outside supplied bounds.')
    prepared = action_preparer(support.cpu().numpy(), normalized.cpu().numpy()) if action_preparer else None
    total = synchronized_time(model) - start
    timed_out = timed_out or (deadline is not None and time.perf_counter() >= deadline)
    timings = dict(timing, end_to_end_seconds=total, policy_inference_seconds=total - calibration_seconds,
                   calibration_seconds=calibration_seconds, includes_safety_projection=prepared is not None or projector is not None)
    result = {'support': support.cpu().tolist(), 'actions': bounded.detach().cpu().tolist(), 'normalized_actions': normalized.cpu().tolist(), 'trace': trace, 'timing': timings, 'calibrated': calibrated, 'timeout': timed_out, 'calibration_enabled': probe_provider is not None, 'state_dependent_projection_applied': projector is not None, 'output_kind': 'projected_action' if projector else 'bounded_proposal'}
    if capture:
        result['transitions'] = transitions
        result['effect_records'] = effect_records
    if prepared is not None:
        result['prepared_action'] = prepared
        result['state_dependent_projection_applied'] = True
        result['output_kind'] = 'projected_action'
    return result

def plugin(value):
    if value is None:
        return None
    (module, name) = value.split(':', 1)
    return getattr(importlib.import_module(module), name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--observation', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--probe-provider', help='module:function implementing matched remaining-horizon probes')
    parser.add_argument('--projector', help='module:function implementing network-specific safety projection')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    (model, metadata) = load_checkpoint(args.checkpoint, args.device)
    if metadata.get('test_only'):
        raise ValueError('Test-only checkpoint is not accepted by the inference command.')
    observation = load_observation(args.observation, model.config, args.device)
    result = decide(model, observation, plugin(args.probe_provider), plugin(args.projector))
    result['checkpoint_metadata'] = metadata
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
if __name__ == '__main__':
    main()
