"""Auditable post-review sensitivity analysis. Does not overwrite legacy results."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from analyze_groundtruth_sensitivity import exact_sign_p, holm_adjust
from class_matched_placebo import (
    FORMULAS, METRICS, add_result, empty_accumulator, metric_values,
    sample_class_matched, stable_seed, summarize_accumulator, tie_inspection_sizes,
)
from groundtruth_sensitivity_audit import normalized_location, mapped_line_indices, source_line_index
from spectrum import average_ranks_desc, load_gzoltar, suspiciousness
from gmar import gmar_score, random_hit_probability


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def annotation_sets(directory, bug_id):
    direct, markers, candidates, strict, relations = set(), set(), set(), set(), []
    path = directory / (bug_id + '.buggy.lines')
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if line.strip():
            assert '\ufffd' not in '#'.join(line.split('#')[:2]), ('Invalid location encoding', path)
            (markers if 'FAULT_OF_OMISSION' in line else direct).add(normalized_location(line))
    candidate_path = directory / (bug_id + '.candidates')
    if candidate_path.exists():
        for line in candidate_path.read_text(encoding='utf-8', errors='replace').splitlines():
            if line.strip():
                assert '\ufffd' not in line, ('Invalid candidate encoding', candidate_path)
                left, right = line.split(',', 1)
                anchor, target = normalized_location(left), normalized_location(right)
                candidates.add(target)
                relations.append((anchor, target))
                if anchor in markers:
                    strict.add(target)
    return direct, markers, candidates, strict, relations


def matched_pools(source_lines, additions, excluded, failed_lines, match_failure):
    def stratum(i):
        return source_lines[i][0] + ('|failed=' + str(int(failed_lines[i])) if match_failure else '')
    counts = Counter(stratum(int(i)) for i in additions)
    pools = {key: [] for key in counts}
    for i in range(len(source_lines)):
        key = stratum(i)
        if i not in excluded and key in pools:
            pools[key].append(i)
    pools = {k: np.asarray(v, dtype=int) for k, v in pools.items()}
    insufficient = any(len(pools[k]) < count for k, count in counts.items())
    return pools, dict(counts), insufficient


def paired_summary(rows, left, right, formula, tie, seed):
    lk, rk = f'{left}_{formula}_{tie}', f'{right}_{formula}_{tie}'
    a = [r for r in rows if r[lk] is not None]
    b = [r for r in rows if r[rk] is not None]
    paired = [r for r in rows if r[lk] is not None and r[rk] is not None]
    output = {'n': len(rows), 'left_evaluable': len(a), 'right_evaluable': len(b),
              'paired': len(paired), 'gained_evaluable': sum(r[lk] is None and r[rk] is not None for r in rows),
              'lost_evaluable': sum(r[lk] is not None and r[rk] is None for r in rows), 'budgets': {}}
    for k in (1, 5, 10, 20):
        gains = sum(r[lk] > k and r[rk] <= k for r in paired)
        losses = sum(r[lk] <= k and r[rk] > k for r in paired)
        output['budgets'][str(k)] = {'left_hits': sum(r[lk] <= k for r in a),
            'right_hits': sum(r[rk] <= k for r in b), 'gains': gains, 'losses': losses,
            'p_raw': exact_sign_p(gains, losses)}
    if paired:
        delta = np.asarray([1 / r[rk] - 1 / r[lk] for r in paired])
        rng = np.random.default_rng(seed)
        boot = np.concatenate([delta[rng.integers(0, len(delta), (1000, len(delta)))].mean(axis=1)
                               for _ in range(20)])
        output.update(delta_mrr=float(delta.mean()), delta_mrr_ci=np.quantile(boot, [.025, .975]).tolist())
        output['rank_gains_equal_losses'] = [sum(r[rk] < r[lk] for r in paired),
            sum(r[rk] == r[lk] for r in paired), sum(r[rk] > r[lk] for r in paired)]
    return output


def random_top10_reference(row):
    return random_hit_probability(inspected=min(10, row['N']), targets=row['broad_m'], population=row['N'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError('Use a new output directory to preserve previous runs')
    out.mkdir(parents=True)
    annotations = root / 'external/fault-localization-data/rjust-fault-localization-data-db9a70bfeaef/analysis/pipeline-scripts/buggy-lines'
    spectra = root / 'data/d4j-v12-all/gzoltars'
    legacy_path = root / 'results/full/groundtruth_sensitivity.json'
    legacy = json.loads(legacy_path.read_text(encoding='utf-8'))['rows']
    encoding_notes = []
    for old in legacy:
        annotation_sets(annotations, old['bug_id'])
        path = annotations / (old['bug_id'] + '.buggy.lines')
        replacements = path.read_text(encoding='utf-8', errors='replace').count('\ufffd')
        if replacements:
            encoding_notes.append({'bug_id': old['bug_id'], 'replacement_characters_in_source_comment': replacements})
    print(json.dumps({'annotation_preflight': len(legacy), 'encoding_notes': encoding_notes}), flush=True)
    seed, repetitions = 20260910, 5000
    started = time.time()
    manifest = {'legacy': sha256(legacy_path), 'analysis': sha256(Path(__file__)),
                'protocol': sha256(root / 'paper/chinese_manuscript/v1.4.0/REVISION_PROTOCOL.md'),
                'dependencies': {name: sha256(root/'src'/name) for name in (
                    'class_matched_placebo.py', 'groundtruth_sensitivity_audit.py', 'spectrum.py',
                    'analyze_groundtruth_sensitivity.py', 'gmar.py')}, 'inputs': {}}
    accumulators = {design: {f: empty_accumulator(repetitions) for f in FORMULAS}
                    for design in ('class_only_same_sample', 'class_and_failure')}
    rows, catalogue, eligible, skipped = [], [], [], []
    reproduced = 0
    for number, old in enumerate(legacy, 1):
        bug_id = old['bug_id']
        direct, markers, candidates, strict, relations = annotation_sets(annotations, bug_id)
        for suffix in ('.buggy.lines', '.candidates'):
            path = annotations / (bug_id + suffix)
            if path.exists():
                manifest['inputs'][str(path.relative_to(root))] = sha256(path)
        meta = {'bug_id': bug_id, 'project': old['project'], 'direct_locations': len(direct),
            'marker_locations': len(markers), 'candidate_locations': len(candidates),
            'strict_candidate_locations': len(strict), 'relations': len(relations),
            'strict_relations': sum(a in markers for a, b in relations),
            'mapped_direct': old['mapped_direct_lines'], 'mapped_broad': old['mapped_official_lines']}
        catalogue.append(meta)
        row = {'bug_id': bug_id, 'project': old['project'], 'has_marker': bool(markers),
               'has_candidates': bool(candidates), 'N': old['line_ranking_size'],
               'direct_m': old['mapped_direct_lines'], 'broad_m': old['mapped_official_lines']}
        for policy, old_policy in (('deleted', 'deleted_only'), ('direct', 'official_direct'), ('broad', 'official_omission_aware')):
            for f in FORMULAS:
                row[f'{policy}_{f}_average'] = old[f'line_{old_policy}_{f}_rank']
                row[f'{policy}_{f}_end'] = old[f'line_{old_policy}_{f}_tie_expanded_size']
        if candidates:
            directory = spectra / old['project'] / bug_id.rsplit('-', 1)[1]
            spectrum = load_gzoltar(directory)
            for name in ('matrix', 'spectra'):
                manifest['inputs'][str((directory/name).relative_to(root))] = sha256(directory/name)
            lines, mapping = source_line_index(spectrum.statements)
            assert len(lines) == row['N']
            direct_idx = mapped_line_indices(lines, direct)
            broad_idx = mapped_line_indices(lines, direct | candidates)
            strict_idx = mapped_line_indices(lines, direct | strict)
            row['strict_m'] = len(strict_idx)
            added = np.asarray(sorted(set(strict_idx) - set(direct_idx)), dtype=int)
            failed_lines = np.zeros(len(lines), dtype=bool)
            np.logical_or.at(failed_lines, mapping, spectrum.coverage[spectrum.failed].any(axis=0))
            samples = {}
            if not markers:
                skipped.append({'bug_id': bug_id, 'reason': 'no_omission_marker'})
            elif not len(added):
                skipped.append({'bug_id': bug_id, 'reason': 'no_strict_mapped_addition'})
            else:
                strong_pool, strong_counts, insufficient = matched_pools(lines, added, set(broad_idx), failed_lines, True)
                if insufficient:
                    skipped.append({'bug_id': bug_id, 'reason': 'insufficient_class_failure_pool'})
                else:
                    weak_pool, weak_counts, weak_bad = matched_pools(lines, added, set(broad_idx), failed_lines, False)
                    assert not weak_bad
                    for design, pool, counts in (('class_only_same_sample', weak_pool, weak_counts), ('class_and_failure', strong_pool, strong_counts)):
                        samples[design] = sample_class_matched(np.random.default_rng(stable_seed(seed, bug_id + design)), pool, counts, repetitions)
                        assert not set(samples[design].ravel()).intersection(set(broad_idx))
                    eligible.append({'bug_id': bug_id, 'project': old['project'], 'strict_targets': len(strict_idx),
                        'fixed_targets': len(direct_idx), 'randomized_targets': len(added),
                        'strata': strong_counts, 'pool_sizes': {k: len(v) for k, v in strong_pool.items()}})
            for f in FORMULAS:
                scores = np.full(len(lines), -np.inf)
                np.maximum.at(scores, mapping, suspiciousness(spectrum.coverage, spectrum.failed, f))
                ranks, ends = average_ranks_desc(scores), tie_inspection_sizes(scores)
                for policy, idx in (('broad', broad_idx), ('direct', direct_idx)):
                    for tie, values in (('average', ranks), ('end', ends)):
                        observed = float(values[idx].min()) if len(idx) else None
                        assert observed == row[f'{policy}_{f}_{tie}'], (bug_id, policy, f, tie)
                        reproduced += 1
                for tie, values in (('average', ranks), ('end', ends)):
                    row[f'strict_{f}_{tie}'] = float(values[strict_idx].min()) if len(strict_idx) else None
                row[f'strict_{f}_gmar_a'] = float(np.mean((len(lines) - ends[strict_idx]) / len(lines))) if len(strict_idx) else None
                for design, sampled in samples.items():
                    observed, placebo = metric_values(scores, strict_idx, direct_idx, sampled)
                    add_result(accumulators[design][f], observed, placebo, len(strict_idx))
        else:
            row['strict_m'] = row['direct_m']
            for f in FORMULAS:
                for tie in ('average', 'end'):
                    row[f'strict_{f}_{tie}'] = row[f'direct_{f}_{tie}']
            skipped.append({'bug_id': bug_id, 'reason': 'no_candidates'})
        meta['mapped_strict'] = row['strict_m']
        for f in FORMULAS:
            for tie in ('average', 'end'):
                if row[f'direct_{f}_{tie}'] is not None:
                    assert row[f'strict_{f}_{tie}'] <= row[f'direct_{f}_{tie}']
                if row[f'strict_{f}_{tie}'] is not None:
                    assert row[f'broad_{f}_{tie}'] <= row[f'strict_{f}_{tie}']
        rows.append(row)
        if number % 10 == 0 or number == len(legacy):
            print(json.dumps({'processed': number, 'total': len(legacy), 'control_eligible': len(eligible), 'elapsed_seconds': round(time.time()-started)}, ensure_ascii=False), flush=True)
    comparisons, all_budget_tests = {}, []
    configs = [('deleted_broad_all', rows, 'deleted', 'broad'), ('deleted_strict_all', rows, 'deleted', 'strict'),
               ('direct_strict_all', rows, 'direct', 'strict'),
               ('deleted_strict_marker', [r for r in rows if r['has_marker']], 'deleted', 'strict')]
    for label, subset, left, right in configs:
        comparisons[label] = {}
        for f in FORMULAS:
            comparisons[label][f] = {}
            for tie in ('average', 'end'):
                result = paired_summary(subset, left, right, f, tie, stable_seed(seed, label+f+tie))
                comparisons[label][f][tie] = result
                all_budget_tests.extend(result['budgets'].values())
    assert len(all_budget_tests) == 128
    for result, adjusted in zip(all_budget_tests, holm_adjust([t['p_raw'] for t in all_budget_tests])):
        result['p_holm_128'] = adjusted
    controls = {d: {f: summarize_accumulator(acc, repetitions) for f, acc in fs.items()} for d, fs in accumulators.items()}
    tests = [value for fs in controls.values() for result in fs.values() for value in result['metrics'].values()]
    assert len(tests) == 64
    for result, adjusted in zip(tests, holm_adjust([t['one_sided_randomization_p'] for t in tests])):
        result['p_holm_64'] = adjusted
    old_placebo = json.loads((root/'results/full/class_matched_placebo.json').read_text(encoding='utf-8'))['summaries']['all']
    old_tests = [m for f in old_placebo.values() for m in f['metrics'].values()]
    for result, adjusted in zip(old_tests, holm_adjust([t['one_sided_randomization_p'] for t in old_tests])):
        result['p_holm_32'] = adjusted
    groups = {}
    for label, condition in [('one', lambda m: m == 1), ('two_three', lambda m: 2 <= m <= 3), ('one_three', lambda m: 1 <= m <= 3), ('four_plus', lambda m: m >= 4)]:
        subset = [r for r in rows if r['has_candidates'] and condition(r['broad_m'])]
        groups[label] = {'n': len(subset), 'top10_average': float(np.mean([r['broad_ochiai_average'] <= 10 for r in subset])),
                         'random_top10': float(np.mean([random_top10_reference(r) for r in subset]))}
    scope = {'total': len(rows), 'marker_bugs': sum(bool(c['marker_locations']) for c in catalogue),
        'candidate_bugs': sum(bool(c['candidate_locations']) for c in catalogue),
        'marker_bugs_with_candidates': sum(bool(c['marker_locations']) and bool(c['candidate_locations']) for c in catalogue),
        'strict_anchor_candidate_bugs': sum(bool(c['strict_candidate_locations']) for c in catalogue),
        'marker_locations': sum(c['marker_locations'] for c in catalogue),
        'all_candidate_locations': sum(c['candidate_locations'] for c in catalogue),
        'strict_candidate_locations': sum(c['strict_candidate_locations'] for c in catalogue),
        'candidate_nonmarker_ids': [c['bug_id'] for c in catalogue if c['candidate_locations'] and not c['marker_locations']],
        'control_eligible': len(eligible), 'control_exclusions': dict(Counter(x['reason'] for x in skipped))}
    assert len(eligible) + len(skipped) == len(rows)
    result = {'design': {'seed': seed, 'replicates': repetitions, 'bootstrap': 20000,
        'status': 'post_review_sensitivity_not_preregistered', 'topk_primary': 'average_rank',
        'strict': 'direct union candidates whose source anchor is an omission marker'},
        'scope': scope, 'encoding_notes': encoding_notes, 'comparisons': comparisons, 'controls': controls,
        'historical_control_holm32': old_placebo, 'corrected_target_groups': groups,
        'catalogue': catalogue, 'rows': rows, 'eligible': eligible, 'excluded': skipped,
        'checks': {'legacy_rank_equalities': reproduced, 'all_invariants_passed': True},
        'elapsed_seconds': round(time.time()-started), 'manifest': manifest}
    (out/'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'finished': True, 'scope': scope, 'checks': result['checks']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
