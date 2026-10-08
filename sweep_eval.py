"""Grid-search `evaluate_full.py` settings and rank the results.

Inference-side knobs (`--halo`, `--core_max`, `--tta_d4`, ...) have never been tuned -- the
released numbers use the values the training cache happened to be built with. Changing them
costs no retraining, so the cheapest available gain is to measure a few and keep the best.

This driver runs `evaluate_full.py` as a subprocess and reads the JSON it already writes, so
the evaluation logic stays in exactly one place and a sweep can never disagree with a
hand-run evaluation. Everything about *what* to sweep lives in a JSON config, not here.

Two properties the scoring pipeline forces on any honest sweep:

- **It is not deterministic.** Repeating an identical evaluation moves mIoU by a few
  hundredths (GPU reduction order, then argmax flipping on near-tied voxels). Set
  `"repeats"` on a stage and the report prints the spread, which is the threshold a
  difference has to clear before it means anything.
- **Cost varies across the grid.** Screen wide and cheap (`tta_d4: 1`), then confirm only
  the survivors at full cost (`tta_d4: 8`) via a stage with `"from_top"`.

Usage:
    python sweep_eval.py --config sweep_eval.json
    python sweep_eval.py --config sweep_eval.json --dry_run    # print the commands only
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time


# ---------------------------------------------------------------- config

def load_config(path):
	with open(path, 'r', encoding='utf-8') as fh:
		cfg = json.load(fh)
	for key in ('base', 'stages'):
		if key not in cfg:
			raise ValueError(f"{path}: missing required key '{key}'")
	if not cfg['stages']:
		raise ValueError(f"{path}: 'stages' is empty")
	return cfg


def expand_grid(grid):
	"""{'halo': [1.0, 1.5], 'core_max': [12288]} -> [{'halo':1.0,...}, {'halo':1.5,...}].

	Insertion order is preserved so the run order matches how the grid reads in the file.
	"""
	if not grid:
		return [{}]
	keys = list(grid)
	return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def combo_tag(combo):
	"""Stable, filename-safe identifier for one point of the grid."""
	if not combo:
		return 'base'
	parts = [f"{k}{str(v).replace('.', 'p').replace('-', 'm')}" for k, v in sorted(combo.items())]
	return '_'.join(parts)


# ---------------------------------------------------------------- running

def to_cli(settings):
	"""Turn {'mode': 'chunk', 'v2_diff': True, 'tta': 1} into CLI tokens.

	`True` becomes a bare flag and `False`/`None` is dropped, which is what argparse
	`store_true` options expect; everything else becomes `--key value`.
	"""
	argv = []
	for key, value in settings.items():
		if value is False or value is None:
			continue
		argv.append(f'--{key}')
		if value is not True:
			argv.append(str(value))
	return argv


def build_command(base, combo, out_path):
	# --overwrite because resume above already decided whether this point needs running;
	# evaluate_full refuses to replace a result file on its own, which is right for a
	# hand-run evaluation and wrong for a sweep that owns its own output directory.
	return [sys.executable, 'evaluate_full.py', *to_cli({**base, **combo}),
			'--out', out_path, '--overwrite']


def read_metrics(path):
	"""Pull the headline numbers out of an evaluate_full.py result file."""
	with open(path, 'r', encoding='utf-8') as fh:
		result = json.load(fh)
	m = result['overall_metrics']
	return {'mIoU': m['mIoU'] * 100, 'OA': m['accuracy'] * 100, 'mAcc': m['mAcc'] * 100,
			'points': m['total_points']}


class RunBudget:
	"""Stops the sweep after N fresh evaluations, so it can be driven in short sessions.

	Reused results never count against the budget, so re-invoking with the same budget walks
	steadily through the plan instead of redoing the beginning of it.
	"""

	def __init__(self, limit=None):
		self.limit = limit
		self.spent = 0

	def exhausted(self):
		return self.limit is not None and self.spent >= self.limit

	def charge(self):
		self.spent += 1


class BudgetExhausted(Exception):
	pass


def run_once(cmd, out_path, log_path, resume, dry_run, budget=None):
	"""Run one evaluation. Returns its metrics, or None in dry-run."""
	if dry_run:
		print('    ' + ' '.join(cmd))
		return None
	if resume and os.path.exists(out_path):
		try:
			metrics = read_metrics(out_path)
			print(f"    reused {os.path.basename(out_path)}")
			return metrics
		except (KeyError, ValueError, OSError):
			pass  # unreadable or truncated -> just run it again
	if budget is not None:
		if budget.exhausted():
			raise BudgetExhausted()
		budget.charge()
	started = time.time()
	with open(log_path, 'w', encoding='utf-8') as log:
		proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
	if proc.returncode != 0:
		raise RuntimeError(f"evaluate_full.py failed (exit {proc.returncode}); see {log_path}")
	metrics = read_metrics(out_path)
	metrics['minutes'] = (time.time() - started) / 60.0
	return metrics


def summarize(runs):
	"""Collapse repeats of one setting into mean and spread."""
	values = [r['mIoU'] for r in runs]
	return {
		'mIoU': sum(values) / len(values),
		'mIoU_spread': max(values) - min(values),
		'OA': sum(r['OA'] for r in runs) / len(runs),
		'mAcc': sum(r['mAcc'] for r in runs) / len(runs),
		'minutes': sum(r.get('minutes', 0.0) for r in runs),
		'repeats': len(runs),
	}


# ---------------------------------------------------------------- stages

def stage_combos(stage, previous):
	"""The settings this stage evaluates.

	A stage either expands its own `grid`, or takes the best `from_top` settings of earlier
	stages and re-runs them under this stage's `overrides` (the screen-then-confirm pattern).
	Either way the stage's `overrides` are applied on top.

	`from_stages` restricts which earlier stages are ranked. Without it a confirm stage would
	rank screening rows (cheap protocol) against an earlier confirm's rows (expensive
	protocol) in one list, and the expensive ones win on protocol rather than on setting.
	"""
	overrides = stage.get('overrides', {})
	if 'from_top' in stage:
		pool = previous
		if stage.get('from_stages'):
			wanted = set(stage['from_stages'])
			pool = [r for r in previous if r['stage'] in wanted]
			missing = wanted - {r['stage'] for r in previous}
			if missing:
				raise ValueError(f"stage '{stage['name']}': from_stages names no-such/not-yet-run "
								 f"stage(s) {sorted(missing)}")
		if not pool:
			raise ValueError(f"stage '{stage['name']}': 'from_top' needs a preceding stage")
		ranked = sorted(pool, key=lambda r: r['mIoU'], reverse=True)
		combos, seen = [], set()
		for row in ranked:                      # coordinate sweeps share their crossing point
			key = combo_tag(row['combo'])
			if key in seen:
				continue
			seen.add(key)
			combos.append(dict(row['combo']))
			if len(combos) >= int(stage['from_top']):
				break
	else:
		combos = expand_grid(stage.get('grid', {}))
	return [{**c, **overrides} for c in combos]


def run_stage(stage, base, out_dir, resume, dry_run, previous, budget=None):
	name = stage['name']
	repeats = int(stage.get('repeats', 1))
	combos = stage_combos(stage, previous)
	print(f"\n[{name}] {len(combos)} setting(s) x {repeats} repeat(s)")
	rows = []
	for combo in combos:
		tag = combo_tag(combo)
		runs = []
		for i in range(repeats):
			stem = os.path.join(out_dir, f"{name}__{tag}" + (f"__r{i + 1}" if repeats > 1 else ''))
			metrics = run_once(build_command(base, combo, stem + '.json'),
							   stem + '.json', stem + '.log', resume, dry_run, budget)
			if metrics is not None:
				runs.append(metrics)
		if not runs:
			# Dry-run: keep a placeholder so a later `from_top` stage still has something to
			# select and the whole plan can be printed without running anything.
			if dry_run:
				rows.append({'stage': name, 'combo': combo, 'tag': tag, 'mIoU': 0.0,
							 'mIoU_spread': 0.0, 'OA': 0.0, 'mAcc': 0.0, 'minutes': 0.0,
							 'repeats': 0})
			continue
		row = {'stage': name, 'combo': combo, 'tag': tag, **summarize(runs)}
		rows.append(row)
		spread = f" +-{row['mIoU_spread']:.2f}" if repeats > 1 else ''
		print(f"    {tag:<28} mIoU {row['mIoU']:.2f}{spread}  OA {row['OA']:.2f}  "
			  f"({row['minutes']:.1f} min)")
	return rows


def report(all_rows, out_dir, noise_hint):
	"""Print the ranking and persist it next to the per-run JSONs."""
	path = os.path.join(out_dir, 'sweep_results.json')
	with open(path, 'w', encoding='utf-8') as fh:
		json.dump({'rows': all_rows, 'noise_mIoU': noise_hint}, fh, indent=2)

	print('\n' + '=' * 78)
	print('RANKED (per stage, best mIoU first)')
	if noise_hint is not None:
		print(f"repeat spread observed: {noise_hint:.2f} mIoU -- treat smaller gaps as a tie")
	for stage in dict.fromkeys(r['stage'] for r in all_rows):
		print(f"\n[{stage}]")
		print(f"  {'setting':<30} {'mIoU':>7} {'OA':>7} {'mAcc':>7}")
		for row in sorted((r for r in all_rows if r['stage'] == stage),
						  key=lambda r: r['mIoU'], reverse=True):
			print(f"  {row['tag']:<30} {row['mIoU']:>7.2f} {row['OA']:>7.2f} {row['mAcc']:>7.2f}")
	print(f"\nSaved: {path}")


def main():
	ap = argparse.ArgumentParser(description=__doc__,
								 formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--config', default='sweep_eval.json', help='Sweep definition (JSON)')
	ap.add_argument('--out_dir', default=None, help="Override the config's out_dir")
	ap.add_argument('--no_resume', action='store_true',
					help='Re-run settings whose result JSON already exists')
	ap.add_argument('--dry_run', action='store_true', help='Print the commands and exit')
	ap.add_argument('--max_runs', type=int, default=None, metavar='N',
					help='Stop after N fresh evaluations (reused results are free). Re-invoke to '
						 'continue -- useful when a single session cannot hold the whole sweep.')
	args = ap.parse_args()

	cfg = load_config(args.config)
	out_dir = args.out_dir or cfg.get('out_dir', 'logs/sweep')
	if not args.dry_run:
		os.makedirs(out_dir, exist_ok=True)

	budget = RunBudget(args.max_runs)
	all_rows, complete = [], True
	for stage in cfg['stages']:
		try:
			all_rows.extend(run_stage(stage, cfg['base'], out_dir, not args.no_resume,
									  args.dry_run, all_rows, budget))
		except BudgetExhausted:
			complete = False
			print(f"\n  budget of {budget.limit} run(s) spent -- re-invoke to continue")
			break

	if args.dry_run or not all_rows:
		return
	spreads = [r['mIoU_spread'] for r in all_rows if r['repeats'] > 1]
	report(all_rows, out_dir, max(spreads) if spreads else None)
	if not complete:
		print('PARTIAL: the plan is not finished; run the same command again to continue.')


if __name__ == '__main__':
	main()
