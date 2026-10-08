# structure_loss.py
# Structure-oriented loss (SOL): a training-time penalty encoding WHERE a component sits in
# a structure, after the structure-oriented concept of Lin et al. (CACAIE 2025).
#
# One prior is implemented: a soft ORDERING between two classes along the structure's main
# horizontal axis. Two components can be locally identical -- the same material, the same
# shape -- and be told apart only by their place in the whole, which a per-block network
# cannot see. An absolute position fed as an input channel does not transfer between scenes,
# because the same coordinate means different things in different structures; a relative
# order often does, so only the order is penalised here.
#
# The axis is derived per scene and is TRAINING-ONLY: no inference path reads it, and a
# checkpoint stays interchangeable with one trained without the term.
#
# Everything is read from structure_presets.json; with the default preset ('none') no term
# is added and training is bit-identical to a run from before this file existed.

import json
import os

import numpy as np
import torch
import torch.nn as nn

PRESETS_PATH = 'structure_presets.json'
AXIS_TABLE = 'structure_axis.json'


# --- preset loading ------------------------------------------------------------------

def load_preset(name, class_names, path=PRESETS_PATH):
	"""Resolve a named preset against the active config's class list.

	Returns None for 'none' (or an empty constraint list), otherwise a dict with class
	NAMES already resolved to indices. An unknown preset or an unknown class name raises,
	rather than silently training without the prior the caller asked for.
	"""
	if not name or name == 'none':
		return None
	if not os.path.exists(path):
		raise FileNotFoundError(f'structure preset file not found: {path}')
	with open(path) as f:
		presets = json.load(f)
	if name not in presets or name.startswith('_'):
		available = sorted(k for k in presets if not k.startswith('_'))
		raise ValueError(f'unknown structure preset {name!r}; available: {available}')
	spec = dict(presets[name])
	idx = {n: i for i, n in enumerate(class_names)}

	def resolve(cls):
		if cls not in idx:
			raise ValueError(f'structure preset {name!r} names class {cls!r}, which is not in '
							 f'the active config ({class_names})')
		return idx[cls]

	constraints = []
	for c in spec.get('constraints', []):
		if c.get('type') != 'ordering':
			raise ValueError(f'structure preset {name!r}: unsupported constraint type {c.get("type")!r}')
		if c.get('feature', 'abs_axis') != 'abs_axis':
			raise ValueError(f'structure preset {name!r}: unsupported feature {c.get("feature")!r}')
		constraints.append({'outer': resolve(c['outer']), 'inner': resolve(c['inner']),
							'outer_name': c['outer'], 'inner_name': c['inner'],
							'margin': float(c.get('margin', 0.15)),
							'min_mass': float(c.get('min_mass', 64.0))})
	if not constraints:
		return None
	axis = spec.get('axis', {})
	if axis.get('source', 'pca_xy') != 'pca_xy':
		raise ValueError(f'structure preset {name!r}: unsupported axis source {axis.get("source")!r}')
	spec['constraints'] = constraints
	spec['axis'] = {'source': 'pca_xy',
					'class_subset': [resolve(c) for c in axis.get('class_subset', [])],
					'class_subset_names': list(axis.get('class_subset', [])),
					'normalize': axis.get('normalize', 'max_abs')}
	spec['weight'] = float(spec.get('weight', 0.1))
	spec['name'] = name
	return spec


def describe(spec):
	if spec is None:
		return 'structure loss: off'
	parts = [f"{c['outer_name']} outboard of {c['inner_name']} (margin {c['margin']})"
			 for c in spec['constraints']]
	subset = spec['axis']['class_subset_names'] or ['<all points>']
	return (f"structure loss '{spec['name']}': weight {spec['weight']}, "
			f"axis = PCA-XY of {'+'.join(subset)}, " + '; '.join(parts))


# --- per-room axis table -------------------------------------------------------------

def _axis_for_room(points, labels, subset, rng_seed=0):
	"""Main horizontal axis of one room: PCA of the XY of the subset classes (the deck, so
	the axis follows the span rather than a riverbank). Falls back to all points when the
	subset is absent. Returns (center_xy, unit_axis_xy, scale)."""
	m = np.isin(labels, subset) if subset else np.ones(len(labels), dtype=bool)
	if m.sum() < 100:
		m = np.ones(len(labels), dtype=bool)
	xy = points[m, :2].astype(np.float64)
	center = xy.mean(axis=0)
	X = xy - center
	if len(X) > 20000:
		X = X[np.random.RandomState(rng_seed).choice(len(X), 20000, replace=False)]
	axis = np.linalg.svd(X, full_matrices=False)[2][0]
	t = (points[:, :2].astype(np.float64) - center) @ axis
	scale = float(np.abs(t).max()) or 1.0
	return center.astype(np.float32), axis.astype(np.float32), scale


def build_axis_table(processed_dirs, spec, sanitize):
	"""Axis parameters per room, keyed by the sanitized room token used in block filenames.

	`sanitize` is the block builder's own room-name sanitizer, passed in so the two can
	never drift apart.
	"""
	import glob
	table = {}
	subset = spec['axis']['class_subset']
	for d in processed_dirs:
		for pt in sorted(glob.glob(os.path.join(d, '*.pt'))):
			data = torch.load(pt, weights_only=False)
			points = data.pos.numpy()
			labels = data.y.numpy()
			center, axis, scale = _axis_for_room(points, labels, subset)
			key = sanitize(os.path.splitext(os.path.basename(pt))[0])
			table[key] = {'center': center.tolist(), 'axis': axis.tolist(), 'scale': scale}
	return table


def save_axis_table(table, path):
	with open(path, 'w') as f:
		json.dump(table, f, indent=2)


def load_axis_table(path):
	with open(path) as f:
		return json.load(f)


def axis_coords(points_xy, entry):
	"""Normalised |position| along the room's main axis, in [0, 1]."""
	c = np.asarray(entry['center'], dtype=np.float32)
	a = np.asarray(entry['axis'], dtype=np.float32)
	t = (points_xy.astype(np.float32) - c) @ a
	return np.abs(t) / np.float32(entry['scale'])


# --- the loss ------------------------------------------------------------------------

class StructureOrderingLoss(nn.Module):
	"""Soft ranking penalty on the predicted spatial order of two classes.

	For each constraint the probability-weighted mean axis position is taken over the whole
	batch for the outer and the inner class, and a hinge asks the outer mean to exceed the
	inner mean by `margin`:

	    mu_c = sum_i p_i,c * t_i / sum_i p_i,c        L = max(0, margin - (mu_outer - mu_inner))

	Using predicted probabilities rather than labels is what makes it a training signal:
	calling points at the middle of the span 'abutment' drags mu_outer down and is penalised,
	so the gradient pushes that mass towards the class whose position matches.

	A constraint is skipped when either class has less than `min_mass` of predicted
	probability in the batch, which keeps the mean from being defined by a handful of points.
	"""

	def __init__(self, spec):
		super().__init__()
		self.spec = spec
		self.weight = spec['weight']
		self.constraints = spec['constraints']
		self.last_terms = {}

	def forward(self, probs, axis):
		"""probs: [N, C] softmax over valid points; axis: [N] normalised |axis position|."""
		total = probs.new_zeros(())
		self.last_terms = {}
		for c in self.constraints:
			p_out = probs[:, c['outer']]
			p_in = probs[:, c['inner']]
			m_out, m_in = p_out.sum(), p_in.sum()
			if m_out.item() < c['min_mass'] or m_in.item() < c['min_mass']:
				continue
			mu_out = (p_out * axis).sum() / m_out
			mu_in = (p_in * axis).sum() / m_in
			term = torch.clamp(c['margin'] - (mu_out - mu_in), min=0.0)
			total = total + term
			self.last_terms[f"{c['outer_name']}>{c['inner_name']}"] = float(term.detach())
		return self.weight * total
