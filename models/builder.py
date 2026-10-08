"""Model construction shared by the CLI entry points.

`inference.py` and `evaluate_full.py` both have to turn "which architecture, with which
constructor flags" into a network, and a checkpoint only loads into the exact flags it was
trained with. That logic lived twice, in two slightly different copies. It lives here now,
so a new architecture knob is added once and the mismatch error reads the same everywhere.

A **model spec** is a plain dict, which is what lets the same builder serve argparse and a
JSON file without either side knowing about the other:

    {"arch": "stencil",
     "enc_channels": [64, 192, 320, 448],
     "bottleneck_dim": 256,
     "v2": {"neighbors": "stencil", "stencil": 2, "diff": true, "directional": true}}

`v1` keys (`context_mode`, `width_mult`, `mid_transformer`, `sampler`) sit at the top level
next to `arch`; the v2-only knobs are grouped under `"v2"` so the two architectures cannot
quietly read each other's settings.
"""

import json

import torch
import torch.nn as nn

from . import get_arch, resolve_arch

# Constructor defaults, kept next to the builder so a spec may be partial. These mirror the
# argparse defaults of the entry points; a checkpoint trained with anything else must say so
# in its spec.
V1_DEFAULTS = {
	'context_mode': 'input',
	'width_mult': 1.0,
	'mid_transformer': False,
	'sampler': 'fps',
	'enc_channels': None,
	'bottleneck_dim': 256,
}

V2_DEFAULTS = {
	'knn': 16,
	'curves': 0,
	'neighbors': 'knn',
	'stencil': 1,
	'diff': False,
	'base_grid': 0.04,
	'pool_grids': (0.08, 0.16, 0.32),
	'directional': False,
	'stencil_z': 0,
}


def _int_tuple(value):
	"""Accept 64,192,320,448 | [64,192,320,448] | (64,...) -> tuple of int, or None."""
	if value is None or value == '':
		return None
	if isinstance(value, str):
		value = value.split(',')
	return tuple(int(v) for v in value)


def _float_tuple(value):
	"""Same, for grid sizes."""
	if value is None or value == '':
		return None
	if isinstance(value, str):
		value = value.split(',')
	return tuple(float(v) for v in value)


def spec_from_args(args):
	"""Build a model spec from the argparse namespace shared by the entry points."""
	return {
		'arch': args.arch,
		'enc_channels': _int_tuple(getattr(args, 'enc_channels', None)),
		'bottleneck_dim': getattr(args, 'bottleneck_dim', V1_DEFAULTS['bottleneck_dim']),
		'context_mode': getattr(args, 'context_mode', V1_DEFAULTS['context_mode']),
		'width_mult': getattr(args, 'width_mult', V1_DEFAULTS['width_mult']),
		'mid_transformer': getattr(args, 'mid_transformer', V1_DEFAULTS['mid_transformer']),
		'sampler': getattr(args, 'sampler', V1_DEFAULTS['sampler']),
		'v2': {
			'knn': getattr(args, 'v2_knn', V2_DEFAULTS['knn']),
			'curves': getattr(args, 'v2_curves', V2_DEFAULTS['curves']),
			'neighbors': getattr(args, 'v2_neighbors', V2_DEFAULTS['neighbors']),
			'stencil': getattr(args, 'v2_stencil', V2_DEFAULTS['stencil']),
			'diff': getattr(args, 'v2_diff', V2_DEFAULTS['diff']),
			'base_grid': getattr(args, 'v2_base_grid', V2_DEFAULTS['base_grid']),
			'pool_grids': _float_tuple(getattr(args, 'v2_pool_grids', None)) or V2_DEFAULTS['pool_grids'],
			'directional': getattr(args, 'v2_directional', V2_DEFAULTS['directional']),
			'stencil_z': getattr(args, 'v2_stencil_z', V2_DEFAULTS['stencil_z']),
		},
	}


def merge_spec(base, override):
	"""Overlay `override` on `base`, descending one level into the 'v2' group.

	Only keys present in `override` win, so a per-model entry in an ensemble file can say
	just {"v2": {"directional": false}} and inherit everything else.
	"""
	merged = dict(base)
	for key, value in (override or {}).items():
		if key == 'v2':
			merged['v2'] = {**base.get('v2', {}), **(value or {})}
		else:
			merged[key] = value
	return merged


def describe_spec(spec):
	"""One-line human-readable summary, used in logs and in the result JSON."""
	arch = resolve_arch(spec.get('arch', 'edgeconv'))
	if arch == 'stencil':
		v2 = {**V2_DEFAULTS, **spec.get('v2', {})}
		bits = [f"neighbors={v2['neighbors']}", f"stencil={v2['stencil']}"]
		if v2['diff']:
			bits.append('diff')
		if v2['directional']:
			bits.append('directional')
		if v2['stencil_z']:
			bits.append(f"stencil_z={v2['stencil_z']}")
	else:
		bits = [f"context={spec.get('context_mode')}", f"width={spec.get('width_mult')}"]
	enc = _int_tuple(spec.get('enc_channels'))
	bits.append(f"enc={','.join(str(c) for c in enc)}" if enc else 'enc=default')
	bits.append(f"bottleneck={spec.get('bottleneck_dim', V1_DEFAULTS['bottleneck_dim'])}")
	return f"{arch} [{', '.join(bits)}]"


def build_model(spec, num_features, num_classes, feature_dims):
	"""Instantiate (on CPU) the architecture named by the spec. No weights are loaded."""
	arch = resolve_arch(spec.get('arch', 'edgeconv'))
	cls = get_arch(arch)
	enc = _int_tuple(spec.get('enc_channels'))
	bottleneck = spec.get('bottleneck_dim', V1_DEFAULTS['bottleneck_dim'])
	if arch == 'stencil':
		v2 = {**V2_DEFAULTS, **spec.get('v2', {})}
		return cls(num_features=num_features, num_classes=num_classes,
				   feature_dims=feature_dims, enc_channels=enc or (64, 192, 320, 448),
				   bottleneck_dim=bottleneck,
				   knn=v2['knn'], curves=v2['curves'],
				   neighbor_mode=v2['neighbors'], stencil_radius=v2['stencil'],
				   feature_diff=v2['diff'], base_grid=v2['base_grid'],
				   pool_grids=_float_tuple(v2['pool_grids']),
				   directional=v2['directional'],
				   stencil_z=v2['stencil_z'] or None)
	return cls(num_features=num_features, num_classes=num_classes, feature_dims=feature_dims,
			   context_mode=spec.get('context_mode', V1_DEFAULTS['context_mode']),
			   width_mult=spec.get('width_mult', V1_DEFAULTS['width_mult']),
			   mid_transformer=spec.get('mid_transformer', V1_DEFAULTS['mid_transformer']),
			   sampler=spec.get('sampler', V1_DEFAULTS['sampler']),
			   enc_channels=enc, bottleneck_dim=bottleneck)


def load_weights(model, weights_path, device, spec):
	"""Load a checkpoint into `model`, reporting a flag mismatch as such instead of a stack trace."""
	state = torch.load(weights_path, map_location=device, weights_only=False)
	if isinstance(state, dict) and 'model_state_dict' in state:
		state = state['model_state_dict']
	try:
		model.load_state_dict(state)
	except RuntimeError as e:
		arch = resolve_arch(spec.get('arch', 'edgeconv'))
		hint = ("check --arch (a v2 checkpoint cannot load into the v1 model and vice versa), "
				"the --v2_* flags and --enc_channels/--bottleneck_dim"
				if arch == 'stencil' else
				"check --arch, --context_mode (input|bottleneck), --width_mult and --mid_transformer")
		raise RuntimeError(
			f"Checkpoint/architecture mismatch for '{weights_path}'. The constructor flags must "
			f"match training: {hint}.\nSpec in use: {describe_spec(spec)}\nOriginal error: {e}") from e
	return model.to(device).eval()


def build_and_load(spec, weights_path, num_features, num_classes, feature_dims, device):
	"""build_model + load_weights, the combination every entry point actually wants."""
	model = build_model(spec, num_features, num_classes, feature_dims)
	return load_weights(model, weights_path, device, spec)


class EnsembleModel(nn.Module):
	"""Average the member softmaxes and return the result in log space.

	Callers apply their own `softmax` to a model's output (see `voxel_chunk.predict_room_chunks`),
	and `softmax(log p) == p` for a normalized p, so returning log-probabilities makes the
	ensemble a drop-in for a single network: no call site has to know how many models there are.
	Members may have different architectures -- only the class count has to agree.
	"""

	def __init__(self, members, weights=None):
		super().__init__()
		if not members:
			raise ValueError('an ensemble needs at least one member')
		self.members = nn.ModuleList(members)
		w = torch.tensor([1.0] * len(members) if weights is None else list(weights), dtype=torch.float32)
		if w.numel() != len(members):
			raise ValueError(f'{w.numel()} weights for {len(members)} members')
		if float(w.sum()) <= 0:
			raise ValueError('ensemble weights must sum to a positive number')
		self.register_buffer('weights', w / w.sum())

	def forward(self, data):
		acc = None
		for weight, member in zip(self.weights, self.members):
			probs = torch.softmax(member(data).float(), dim=-1) * weight
			acc = probs if acc is None else acc + probs
		return torch.log(acc.clamp_min(1e-12))


def load_ensemble_spec(path):
	"""Read an ensemble JSON file. Returns (defaults, entries) with entries validated.

	File shape -- `defaults` is optional and every member key is optional except `weights`:

		{"defaults": {"arch": "stencil", "bottleneck_dim": 256, "v2": {...}},
		 "models": [{"weights": "logs/<run>/final_model.pth", "weight": 1.0, "v2": {...}}]}
	"""
	with open(path, 'r', encoding='utf-8') as fh:
		cfg = json.load(fh)
	entries = cfg.get('models') or []
	if not entries:
		raise ValueError(f"{path}: 'models' is empty -- list at least one checkpoint")
	for i, entry in enumerate(entries):
		if not entry.get('weights'):
			raise ValueError(f"{path}: models[{i}] has no 'weights' path")
	return cfg.get('defaults') or {}, entries


def build_ensemble(path, base_spec, num_features, num_classes, feature_dims, device, verbose=True):
	"""Construct the ensemble described by `path`.

	`base_spec` (normally the CLI flags) is the outermost default, the file's `defaults` block
	overrides it, and each member's own keys override that. Returns (model, member_descriptions).
	"""
	file_defaults, entries = load_ensemble_spec(path)
	defaults = merge_spec(base_spec, file_defaults)
	members, weights, described = [], [], []
	for entry in entries:
		spec = merge_spec(defaults, {k: v for k, v in entry.items() if k not in ('weights', 'weight')})
		members.append(build_and_load(spec, entry['weights'], num_features, num_classes,
									  feature_dims, device))
		weights.append(float(entry.get('weight', 1.0)))
		described.append({'weights': entry['weights'], 'weight': weights[-1],
						  'spec': describe_spec(spec)})
		if verbose:
			print(f"  + {entry['weights']}  (w={weights[-1]:g})  {describe_spec(spec)}")
	return EnsembleModel(members, weights).to(device).eval(), described

def load_ensemble_members(base_spec, weights_path, ensemble_paths, ensemble_config,
						  num_features, num_classes, feature_dims, device, verbose=True):
	"""Resolve however the caller named the network(s) into one runnable model.

	There are three ways to ask for a model, and every entry point accepts all three so the
	same command shape works whether you are scoring a benchmark or segmenting a new scan:

	  weights_path only                -> that single checkpoint
	  weights_path + ensemble_paths    -> all members built from `base_spec` (the CLI flags),
	                                      i.e. checkpoints that share one architecture
	  ensemble_config (a JSON file)    -> members may each override the architecture, which is
	                                      what it takes to mix, say, a directional checkpoint
	                                      with an isotropic one

	Returns `(model, members)`. `members` is None for a single checkpoint and a list of
	descriptions otherwise. An ensemble comes back as an EnsembleModel, which behaves exactly
	like one network at the call site, so nothing downstream counts models.
	"""
	if ensemble_config and ensemble_paths:
		raise ValueError('use --ensemble (checkpoints sharing one architecture) or '
						 '--ensemble_config (per-member architectures), not both')
	if ensemble_config:
		return build_ensemble(ensemble_config, base_spec, num_features, num_classes,
							  feature_dims, device, verbose=verbose)
	if not weights_path:
		raise ValueError('no checkpoint given: pass --model_weights or --ensemble_config')

	paths = [weights_path, *(ensemble_paths or [])]
	if len(paths) == 1:
		return build_and_load(base_spec, weights_path, num_features, num_classes,
							  feature_dims, device), None

	members, described = [], []
	for path in paths:
		members.append(build_and_load(base_spec, path, num_features, num_classes,
									  feature_dims, device))
		described.append({'weights': path, 'weight': 1.0, 'spec': describe_spec(base_spec)})
		if verbose:
			print(f"  + {path}  (w=1)  {describe_spec(base_spec)}")
	return EnsembleModel(members).to(device).eval(), described


def add_ensemble_arguments(ap):
	"""Register the ensemble flags, identically, on every CLI that can run a model.

	Only these two are shared. `--model_weights` stays with each tool because their spellings
	differ by history (inference.py also answers to `-m` and has a default path), and changing
	that would break documented commands.
	"""
	ap.add_argument('--ensemble', nargs='*', default=None, metavar='WEIGHTS.pth',
					help='Extra checkpoints to softmax-average with --model_weights. Every member '
						 'is built from the architecture flags on this command line, so they must '
						 'share one architecture; use --ensemble_config to mix architectures.')
	ap.add_argument('--ensemble_config', default=None, metavar='SPEC.json',
					help='JSON listing ensemble members, each free to override the architecture '
						 'flags (see ensemble_example.json). Mutually exclusive with --ensemble.')
