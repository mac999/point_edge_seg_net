"""One file per application domain, holding everything that differs between domains.

The pipeline is domain-agnostic -- indoor rooms, bridges, tunnels and terrain all run the
same code -- but each domain needs its own data paths, block geometry, architecture flags
and training hyperparameters. Those used to live in shell scripts as one long CLI string,
which meant a domain's recipe was neither discoverable nor reviewable, and drifted between
whoever last ran it.

A domain file collects the whole recipe:

    {
      "name": "bridge",
      "config": "model_params_semanticbridge.json",
      "train_args": { "block_size": 20480, "arch": "stencil", ... }
    }

`train_args` keys are `train_model.py` option names (no leading `--`), so anything the
CLI accepts can be set here with no code change. An explicit command-line flag always
wins over the domain file, matching how `--config` already behaves, so a domain stays a
default set rather than a straitjacket.

Unknown keys are a hard error. A silently-ignored `oversample` that should have been
`oversample_rare` is exactly the kind of typo that costs a training run.
"""

import json
import os

DOMAIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'domains')


def resolve_path(name):
	"""Accept a bare domain name ('bridge'), or a path to a domain JSON."""
	if os.path.isfile(name):
		return name
	candidate = os.path.join(DOMAIN_DIR, f"{name}.json")
	if os.path.isfile(candidate):
		return candidate
	available = sorted(f[:-5] for f in os.listdir(DOMAIN_DIR)) if os.path.isdir(DOMAIN_DIR) else []
	raise FileNotFoundError(
		f"no domain '{name}' (looked for {name} and {candidate}). Available: {available or 'none'}")


def load_domain(name):
	path = resolve_path(name)
	with open(path, 'r', encoding='utf-8') as fh:
		domain = json.load(fh)
	domain['_path'] = path
	if 'train_args' not in domain:
		raise ValueError(f"{path}: missing required key 'train_args'")
	return domain


def apply_domain(args, domain, argv, parser=None):
	"""Fill `args` from the domain file, leaving anything given on the command line alone.

	Returns the list of (key, value) actually applied, for logging -- a run should be able
	to show which settings came from the domain rather than from its own command line.
	"""
	known = set(vars(args))
	unknown = sorted(set(domain['train_args']) - known)
	if unknown:
		raise ValueError(
			f"{domain['_path']}: train_args has no-such-option {unknown}. "
			f"Keys must be train_model.py option names without '--'.")

	# Whether an option holds a list or a scalar is argparse's business, not a guess: an
	# option declared with nargs wants a list, and one declared without wants a scalar even
	# when it is conceptually a sequence (--enc_channels is a comma-joined string). Reading
	# it off the parser keeps a domain file from having to know which is which.
	takes_list = {}
	if parser is not None:
		takes_list = {a.dest: (a.nargs in ('+', '*') or isinstance(a.nargs, int))
					  for a in parser._actions}

	explicit = {a.lstrip('-').replace('-', '_') for a in argv if a.startswith('--')}
	applied = []
	if domain.get('config') and 'config' not in explicit:
		args.config = domain['config']
		applied.append(('config', args.config))
	for key, value in domain['train_args'].items():
		if key in explicit:
			continue
		wants_list = takes_list.get(key, False)
		if isinstance(value, list) and not wants_list:
			value = ','.join(str(v) for v in value)
		elif wants_list and not isinstance(value, list):
			value = [value]
		setattr(args, key, value)
		applied.append((key, value))
	return applied


# --- scoring side ---------------------------------------------------------------------
#
# A checkpoint must be scored on the geometry it was trained on. Getting this wrong is not
# hypothetical: w6 and w12 were first scored at the default 2 m window and full resolution
# and came out 3.9 mIoU below their real figures, which produced three wrong conclusions
# before the mismatch was found. The fix is to read the geometry from the same domain file
# the run was trained from, rather than retyping it on the scoring command line.
#
# Most option names are shared between the two tools; these are the ones that differ.
EVAL_ARG_MAP = {
	'column_window': 'window',
	'column_stride': 'stride',
}

# Training-only settings. Listing them explicitly keeps the unknown-key check meaningful on
# the scoring side: anything NOT here and NOT an evaluate_full.py option is still an error.
EVAL_IGNORED = {
	'block_data_path', 'log_root', 'train_areas', 'num_epochs', 'learning_rate',
	'val_batch_size', 'focal_gamma', 'oversample_rare', 'aug_preset', 'cooldown_sec',
	'pad_blocks', 'cover_columns', 'block_mode', 'early_stop_patience', 'warmup_epochs',
	'lr_eta_min', 'accumulation_steps', 'gradient_clip', 'max_grad_norm',
	'early_stop_lr_tol', 'early_stop_require_anneal', 'allow_early_stop_before_anneal',
	'structure_loss', 'structure_weight', 'structure_presets', 'init_weights',
	'no_wandb', 'diagnose', 'resume',
}


def apply_domain_eval(args, domain, argv, parser=None):
	"""Fill scoring `args` from a training domain file; explicit flags still win.

	Scoring needs the block geometry, the voxel lattice and the architecture -- everything
	that has to agree with training -- and nothing about the optimizer. `batch_size` is
	taken from the domain too: the scoring default (18) assumes dense full-resolution
	blocks and runs out of memory on voxelised ones, while training's own width is known to
	fit and inference carries no backward pass.
	"""
	known = set(vars(args))
	train_args = domain['train_args']
	unknown = sorted(k for k in train_args
					 if EVAL_ARG_MAP.get(k, k) not in known and k not in EVAL_IGNORED)
	if unknown:
		raise ValueError(
			f"{domain['_path']}: train_args has keys that are neither an evaluate_full.py "
			f"option nor a known training-only setting: {unknown}")

	takes_list = {}
	if parser is not None:
		takes_list = {a.dest: (a.nargs in ('+', '*') or isinstance(a.nargs, int))
					  for a in parser._actions}

	explicit = {a.lstrip('-').replace('-', '_') for a in argv if a.startswith('--')}
	applied = []
	if domain.get('config') and 'config' not in explicit:
		args.config = domain['config']
		applied.append(('config', args.config))
	for key, value in train_args.items():
		if key in EVAL_IGNORED:
			continue
		dest = EVAL_ARG_MAP.get(key, key)
		if dest in explicit or key in explicit:
			continue
		wants_list = takes_list.get(dest, False)
		if isinstance(value, list) and not wants_list:
			value = ','.join(str(v) for v in value)
		elif wants_list and not isinstance(value, list):
			value = [value]
		setattr(args, dest, value)
		applied.append((dest, value))
	return applied


def describe(domain, applied):
	head = f"Domain '{domain.get('name', '?')}' ({domain['_path']})"
	note = domain.get('description')
	lines = [head] + ([f"  {note}"] if note else [])
	lines += [f"  {k} = {v}" for k, v in applied]
	return '\n'.join(lines)
