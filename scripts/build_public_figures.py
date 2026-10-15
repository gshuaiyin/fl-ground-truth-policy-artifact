"""Replay the public figure values without manuscript, raw review, or upstream code."""
from __future__ import annotations
import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from collections import Counter
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / 'replay/mpl_cache'))
import plotting_helpers as c
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / 'replay/figures'
PATHS = dict(c.SOURCES,
 python=ROOT/'results/full/bugsinpy_external_validation.json',
 project=ROOT/'results/full/project_robustness.json',
 audit=ROOT/'results/manual_audit_summary/target_counts.json')
FORMULAS = ['Ochiai', 'DStar', 'Tarantula']

def load(key):
    return json.loads(PATHS[key].read_text(encoding='utf-8-sig'))


def csv_out(name, rows):
    with (FIGS/name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def diagram_figure(outputs, stem):
    """Replay the authored schematic by copying its approved final exports.

    These are explanatory diagrams, not fitted statistical results. The native
    draw.io source for Figure 1 and editable-text SVG for S1 are distributed.
    Numeric statistical panels below are regenerated from frozen JSON.
    """
    assert stem in {'figure1', 'figureS1'}
    for extension in ['pdf', 'svg', 'png']:
        source = ROOT/'figures/v3.5.0'/f'{stem}.{extension}'
        target = FIGS/source.name
        shutil.copyfile(source, target)
        outputs.append({'file': target.name, 'sha256': c.sha(target)})
    return {'type': 'authored schematic', 'replay': 'verified copy of final exports',
            'statistical_estimates_recomputed': False,
            'source': 'figure1.drawio' if stem == 'figure1' else 'figureS1.svg'}


def matched_figure(outputs):
    data = load('rq1')['controls']
    formulas = ['ochiai', 'dstar', 'tarantula', 'barinel']
    names = ['Ochiai', 'DStar', 'Tarantula', 'Barinel']
    conditions = [('class_only_same_sample', c.BLUE, '', 'C: class and target count'),
                  ('class_and_failure', c.ORANGE, '///', 'C+F: also failing-test coverage')]
    specs = [('mrr', 'MRR advantage', 1, .15), ('gmar_f', 'GMAR-F advantage', 1, .35),
             ('gmar_a', 'GMAR-A advantage', 1, .45), ('top10', 'Top-10 advantage', 100, 30)]
    fig, axes = plt.subplots(2, 2, figsize=(190/25.4, 125/25.4), layout='constrained')
    handles = [Rectangle((0, 0), 1, 1, fc=color, ec=c.GRAY, hatch=hatch, lw=.6) for _, color, hatch, _ in conditions]
    fig.legend(handles, [x[3] for x in conditions], loc='outside upper center', ncols=2, frameon=False, fontsize=8)
    records = []; x = np.arange(4); width = .32
    for index, (metric, title, scale, upper) in enumerate(specs):
        ax = axes.flat[index]
        for k, (key, color, hatch, _) in enumerate(conditions):
            pos = x+(k-.5)*width; vals = []; errors = [[], []]
            for formula in formulas:
                assert data[key][formula]['bugs'] == 206
                m = data[key][formula]['metrics'][metric]
                v = m['effect']*scale
                assert abs(m['effect']-(m['observed']-m['placebo_mean'])) < 1e-12
                lo, hi = [(m['observed']-z)*scale for z in reversed(m['placebo_95_randomization_interval'])]
                assert 0 <= lo <= v <= hi <= upper
                vals.append(v); errors[0].append(v-lo); errors[1].append(hi-v)
                records.append(dict(formula=formula, condition=key, metric=metric, effect=v,
                                    randomization_low=lo, randomization_high=hi,
                                    unit='percentage_points' if metric == 'top10' else 'score'))
            ax.bar(pos, vals, width=width, color=color, edgecolor=c.GRAY, hatch=hatch, linewidth=.55, zorder=2)
            ax.errorbar(pos, vals, yerr=errors, fmt='none', ecolor='#333333', capsize=2.1, lw=.85, zorder=3)
        ax.set(xticks=x, xticklabels=names, xlim=(-.65, 3.65), ylim=(0, upper),
               ylabel='Score difference' if scale == 1 else 'Difference (percentage points)')
        ax.set_title(f'({chr(97+index)}) {title}', loc='left', fontsize=8.5)
        ax.tick_params(axis='x', length=0); ax.grid(axis='y', color='.9', lw=.5)
        ax.set_yticks([[0, .05, .10, .15], [0, .10, .20, .30], [0, .15, .30, .45], [0, 10, 20, 30]][index])
    c.export(fig, 'figure5', outputs)
    csv_out('figure5_controls.csv', records)
    return {'records': 32, 'n': 206, 'bar_baseline': 0,
            'range': '95% randomization quantiles transformed as observed minus control; not a bootstrap CI',
            'top10_scale': 100, 'redundancy': 'C+F hatch and consistent within-formula order'}


def python_rows():
    d = load('python'); records = []; counts = []
    for name in FORMULAS:
        eligible = [r for r in d['rows'][name] if r['primary'] is not None and r['expanded'] is not None]
        assert len(eligible) == 120
        for r in eligible:
            records.append(dict(formula=name, bug_id=r['bug_id'], project=r['project'],
                                delta_rr=r['expanded']['mrr']-r['primary']['mrr'],
                                delta_gmar_a=r['expanded']['gmar_a']-r['primary']['gmar_a']))
        for metric, source in [('delta_rr', 'mrr'), ('delta_gmar_a', 'gmar_a')]:
            values = [r[metric] for r in records if r['formula'] == name]
            contrast = d['policy_summary'][name]['contrast']
            assert contrast['paired'] == 120
            assert abs(np.mean(values)-contrast['metrics'][source]['mean_effect']) < 1e-12
            item = dict(formula=name, metric=metric, paired_n=120,
                        increased=sum(x > 1e-12 for x in values), unchanged=sum(abs(x) <= 1e-12 for x in values),
                        decreased=sum(x < -1e-12 for x in values), min_change=min(values), max_change=max(values))
            assert (item['increased'], item['unchanged'], item['decreased']) == ((4, 116, 0) if source == 'mrr' else (7, 100, 13))
            counts.append(item)
    assert len(records) == 360
    csv_out('figure6_python_paired_rows.csv', records)
    csv_out('figure6_python_change_counts.csv', counts)
    return records, counts


def python_figure(outputs):
    d = load('python'); records, counts = python_rows()
    styles = [(c.BLUE, '-'), (c.ORANGE, '--'), (c.GRAY, ':')]
    fig, axes = plt.subplots(2, 2, figsize=(190/25.4, 143/25.4), layout='constrained')
    handles = [Line2D([], [], color=color, ls=ls, lw=1.3) for color, ls in styles]
    handles += [Line2D([], [], color=c.ORANGE, marker='o', lw=0, ms=4),
                Line2D([], [], color=c.GRAY, marker='_', lw=1.2, ms=6)]
    fig.legend(handles, FORMULAS+['Published targets', 'Random mean / 95% range'],
               loc='outside upper center', ncols=5, frameon=False, fontsize=7.3)
    summaries = []
    for row, (metric, source, label) in enumerate([('delta_rr', 'mrr', 'reciprocal-rank'), ('delta_gmar_a', 'gmar_a', 'GMAR-A')]):
        left, right = axes[row]
        limits = (-.06, 1.02) if row == 0 else (-.4, .5)
        left.set(xlim=limits, ylim=(0, 1.02), xlabel='Expanded minus primary', ylabel='Cumulative share of faults')
        left.set_title(f'({chr(97+2*row)}) Paired {label} changes\nn = 120; all common faults', loc='left', fontsize=8.3, pad=19)
        left.axvline(0, color='#999999', ls='--', lw=.6)
        left.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        left.set_yticks([0, .25, .5, .75, 1]); left.xaxis.set_major_locator(MaxNLocator(nbins=4))
        left.grid(color='.92', lw=.5)
        inset = left.inset_axes([.25, .21, .69, .40]) if row == 0 else None
        for j, (name, (color, linestyle)) in enumerate(zip(FORMULAS, styles)):
            values = np.sort([r[metric] for r in records if r['formula'] == name])
            y = np.arange(1, 121)/120
            xx = np.r_[limits[0], values, limits[1]]; yy = np.r_[0, y, 1]
            left.step(xx, yy, where='post', color=color, ls=linestyle, lw=1.3)
            if inset is not None:
                inset.step(xx, yy, where='post', color=color, ls=linestyle, lw=1.1)
            m = d['placebo_summary']['all'][name]['metrics'][source]
            assert d['placebo_summary']['all'][name]['bugs'] == 28
            lo, hi = m['placebo_95_randomization_interval']
            right.plot([j, j], [lo, hi], color=c.GRAY, lw=1.5)
            right.plot(j, m['placebo_mean'], marker='_', color=c.GRAY, ms=10, markeredgewidth=1.6)
            right.plot(j, m['observed'], marker='o', color=c.ORANGE, ms=4.4)
            # Per-formula p-values are positioned above the axes; no p-value color coding.
            right.text(j, 1.035, f'p = {m["holm_adjusted_p_across_24"]:.5f}',
                       transform=right.get_xaxis_transform(), ha='center', fontsize=7.2)
            summaries.append(dict(analysis='matched_control', formula=name, metric=source, n=28,
                                  observed=m['observed'], reference=m['placebo_mean'], low=lo, high=hi,
                                  p_holm=m['holm_adjusted_p_across_24'], interval='randomization_95_quantiles'))
        if inset is not None:
            inset.set(xlim=(-.03, 1), ylim=(.96, 1.003), xticks=[0, .5, 1], yticks=[.9666666666667, .9833333333333, 1])
            inset.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
            inset.tick_params(labelsize=7.2, length=2, pad=2)
            inset.set_title('Detail: upper 4% of the same ECDF', fontsize=7.3, loc='left')
            inset.grid(color='.92', lw=.4)
        count_note = '4 increase; 0 decrease\n116 unchanged (all formulas)' if row == 0 else '7 increase; 13 decrease\n100 unchanged (all formulas)'
        left.text(.97, .75 if row == 0 else .32, count_note,
                  transform=left.transAxes, ha='right', fontsize=7.2, linespacing=1.4)
        right.set(xticks=range(3), xticklabels=FORMULAS, xlim=(-.55, 2.55),
                  ylim=(.10, .18) if row == 0 else (.49, .72), ylabel='MRR' if row == 0 else 'GMAR-A')
        right.set_title(f'({chr(98+2*row)}) Matched control, n = 28\nHolm-24 p-values', loc='left', fontsize=8.3, pad=19)
        right.tick_params(axis='x', length=0); right.grid(axis='y', color='.9', lw=.5)
        right.yaxis.set_major_locator(MaxNLocator(nbins=4))
    c.export(fig, 'figure6', outputs)
    csv_out('figure6_python_control.csv', summaries)
    return {'policy_n': 120, 'control_n': 28, 'paired_rows': 360, 'counts': counts,
            'ecdf': 'all paired rows including zero; no smoothing, jitter, binning or filtering',
            'detail': 'RR inset magnifies only the vertical scale to 96%-100.3%; full ECDF remains visible',
            'count_tolerance': 'abs(change)<=1e-12; CSV/ECDF retain original precision',
            'uncertainty': 'ECDF descriptive. Control ranges are randomization quantiles. Policy CIs retained in Figure S5.'}


def python_policy_figure(outputs):
    d = load('python'); rows = []
    fig, axes = plt.subplots(1, 2, figsize=(190/25.4, 69/25.4), layout='constrained')
    for col, (metric, name) in enumerate([('mrr', 'MRR'), ('gmar_a', 'GMAR-A')]):
        ax = axes[col]; limits = [0]
        for y, formula in enumerate(FORMULAS):
            contrast = d['policy_summary'][formula]['contrast']; assert contrast['paired'] == 120
            m = contrast['metrics'][metric]; v = m['mean_effect']; lo, hi = m['bootstrap_95ci']
            ax.errorbar(v, y, xerr=[[v-lo], [hi-v]], fmt='o', color=c.BLUE, ms=4.2, capsize=2.2, lw=1)
            limits.extend([lo, hi])
            rows.append(dict(formula=formula, metric=metric, n=120, effect=v, ci_low=lo, ci_high=hi))
        margin = (max(limits)-min(limits))*.12
        ax.set(xlim=(min(limits)-margin, max(limits)+margin), yticks=range(3), yticklabels=FORMULAS,
               ylim=(2.5, -.5), xlabel='Expanded minus primary')
        ax.set_title(f'({chr(97+col)}) Paired {name} effect, n = 120', loc='left', fontsize=8.5)
        ax.axvline(0, color=c.GRAY, ls='--', lw=.7); ax.grid(axis='x', color='.9', lw=.5)
        ax.tick_params(axis='y', length=0); ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    c.export(fig, 'figureS5', outputs)
    csv_out('figureS5_python_policy_effects.csv', rows)
    return {'n': 120, 'records': 6, 'range': 'existing paired bootstrap 95% confidence intervals, unchanged'}

def project_figure(outputs):
    d=load('project'); formulas=['ochiai','dstar','tarantula','barinel']; names=['Ochiai','DStar','Tarantula','Barinel']; rows=[]
    fig,axes=plt.subplots(2,4,figsize=(190/25.4,165/25.4),layout='constrained')
    labels=['Full sample']+['Without '+p for p in d['projects']]
    for col,(formula,name) in enumerate(zip(formulas,names)):
        source=d['by_formula'][formula]
        items=[source['all_projects_pooled']]+[source['leave_one_project_out'][p] for p in d['projects']]
        for row,(key,label) in enumerate([('mrr_delta','MRR difference'),('exam_improvement','EXAM improvement')]):
            ax=axes[row,col]
            for y,item in enumerate(items):
                m=item[key]; v=m['all_paired_mean']; lo,hi=m['all_paired_bootstrap_95ci']
                ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',color=c.BLUE,ms=3.3,capsize=1.8,lw=.8)
                rows.append(dict(formula=formula,metric=key,excluded_project='none' if y==0 else d['projects'][y-1],paired_n=item['paired'],effect=v,ci_low=lo,ci_high=hi))
            ax.set(yticks=range(7),yticklabels=labels if col==0 else ['']*7,ylim=(6.5,-.5),xlabel=label)
            ax.set_title(f'({chr(97+row*4+col)}) {name}',loc='left',fontsize=8)
            ax.axvline(0,color=c.GRAY,ls='--',lw=.65); ax.grid(axis='x',color='.9',lw=.5); ax.tick_params(axis='y',length=0)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    for row in range(2):
        vals=[r for r in rows if r['metric']==['mrr_delta','exam_improvement'][row]]
        lo=min([r['ci_low'] for r in vals]+[0]); hi=max([r['ci_high'] for r in vals]+[0]); pad=(hi-lo)*.1
        for ax in axes[row]: ax.set_xlim(lo-pad,hi+pad)
    c.export(fig,'figureS2',outputs); csv_out('figureS2_project_robustness.csv',rows)
    contributions=[]
    for formula in formulas:
        for project in d['projects']:
            v=d['by_formula'][formula]['by_project'][project]; top=v['topk']['10']
            contributions.append(dict(formula=formula,project=project,paired_n=v['paired'],gains_top10=top['gains'],losses_top10=top['losses'],net_evaluable=v['evaluability_gain']))
    csv_out('project_contributions.csv',contributions)
    return {'records':56,'policy':'deleted to broad only','uncertainty':'frozen paired bootstrap 95% CI','project_contributions':contributions}


def audit_figure(outputs):
    d=load('audit'); rows=d['audit_changes']; assert len(rows)==36
    assert sum(r['raw_targets'] for r in rows)==187 and sum(r['audited_targets'] for r in rows)==130
    types=[('direct_only','Direct-only'),('mixed','Mixed'),('pure_omission','Omission-only')]
    fig,axes=plt.subplots(1,3,figsize=(190/25.4,113/25.4),layout='constrained')
    exported=[]
    for ax,(kind,name) in zip(axes,types):
        subset=sorted([r for r in rows if r['fault_type']==kind],key=lambda r:(r['project'],r['bug_id']))
        assert len(subset)==12
        for y,r in enumerate(subset):
            ax.plot([r['raw_targets'],r['audited_targets']],[y,y],color='#89969F',lw=.8)
            ax.plot(r['raw_targets'],y,'o',color=c.GRAY,ms=5,markerfacecolor='white',markeredgewidth=.8)
            ax.plot(r['audited_targets'],y,'s',color=c.BLUE,ms=3)
            exported.append({k:r[k] for k in ['bug_id','fault_type','raw_targets','audited_targets','target_reduction']})
        ax.set(yticks=range(12),yticklabels=[r['bug_id'] for r in subset],ylim=(11.6,-.6),xlim=(-.7,16),xlabel='Unique targets')
        ax.set_title(name+' (n = 12)',loc='left',fontsize=8.5); ax.tick_params(axis='y',length=0); ax.grid(axis='x',color='.9',lw=.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4,integer=True))
    max_count=max(r['raw_targets'] for r in rows)
    for ax in axes: ax.set_xlim(-.7,max_count+1)
    fig.legend([Line2D([],[],marker='o',color=c.GRAY,markerfacecolor='white',lw=0),Line2D([],[],marker='s',color=c.BLUE,lw=0)],
               ['Official targets (187)','Frozen AST targets (130)'],loc='outside upper center',ncols=2,frameon=False,fontsize=8)
    c.export(fig,'figureS3',outputs); csv_out('figureS3_audit_counts.csv',exported)
    return {'bugs':36,'original_targets':187,'ast_targets':130,'bugs_reduced':sum(r['target_reduction']>0 for r in rows),'limitation':'Stratified sample; not a population accuracy estimate'}


def joint_figure(outputs):
    d=load('rq5'); runs=d['seed_results']; records=[]; per_seed=[]
    fig,axes=plt.subplots(2,3,figsize=(190/25.4,132/25.4),layout='constrained')
    all_coordinates=[]
    for ax,run in zip(axes.flat,runs):
        eligible=[r for r in run['rows'] if r['direct']['rank'] is not None and r['strict']['rank'] is not None]; assert len(eligible)==240
        coordinates=[]; both=0
        for r in eligible:
            dx=1/r['strict']['rank']-1/r['direct']['rank']; dy=r['strict']['gmar_a']-r['direct']['gmar_a']
            both+=dx>1e-12 and dy < -1e-12
            coordinates.append((round(dx,12),round(dy,12))); all_coordinates.append((dx,dy))
            records.append(dict(seed=run['seed'],bug_id=r['bug_id'],delta_rr=dx,delta_gmar_a=dy))
        groups=Counter(coordinates)
        origin=groups.pop((0.0,0.0),0)
        xs,ys=zip(*groups); sizes=[6*groups[xy] for xy in groups]
        ax.scatter(xs,ys,s=sizes,color=c.BLUE,alpha=.8,edgecolors='white',linewidth=.25)
        ax.axvline(0,color=c.GRAY,ls='--',lw=.6); ax.axhline(0,color=c.GRAY,ls='--',lw=.6)
        ax.set_title(f'Seed {run["seed"]}: both changes {both}/240',loc='left',fontsize=8)
        ax.set(xlabel='Reciprocal-rank change',ylabel='GMAR-A change'); ax.grid(color='.93',lw=.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3)); ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.text(.95,.08,f'At (0, 0): {origin}/240',transform=ax.transAxes,ha='right',va='bottom',fontsize=7.4)
        per_seed.append(dict(seed=run['seed'],paired_n=240,both=both,origin=origin,
                             reciprocal_rank_improved=sum(x>1e-12 for x,y in coordinates)))
    x_max=max(x for x,y in all_coordinates); y_min=min(y for x,y in all_coordinates); y_max=max(y for x,y in all_coordinates)
    for ax in list(axes.flat)[:5]: ax.set_xlim(-.10*x_max,1.1*x_max); ax.set_ylim(y_min-(y_max-y_min)*.1,y_max+(y_max-y_min)*.1)
    key=axes.flat[5]; key.axis('off')
    for y,label in [(.94,'Same 240 faults per seed\nOrigin overlap is reported as a count.'),
                    (.70,'Marker area counts coincident faults.\nZero lines separate change directions.'),
                    (.43,'"Both changes": reciprocal rank improves\nand GMAR-A declines.\nSeeds do not increase independent n.')]:
        key.text(.02,y,label,transform=key.transAxes,va='top',fontsize=7.5,linespacing=1.4)
    for x,count in [(.18,1),(.45,2),(.72,5)]:
        key.scatter(x,.11,s=6*count,color=c.BLUE,alpha=.8,edgecolors='white',linewidth=.25,transform=key.transAxes)
        key.text(x,.04,str(count)+(' fault' if count==1 else ' faults'),transform=key.transAxes,ha='center',fontsize=7.2)
    c.export(fig,'figureS4',outputs); csv_out('figureS4_joint_changes.csv',records)
    return {'n_per_seed':240,'counts':per_seed,'grouping':'coordinates rounded to 12 decimals for overlap only; CSV retains full precision',
            'marker_area':'6 square points per coincident fault; origin count is printed instead of plotted to prevent occlusion'}


def main():
    global FIGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=FIGS)
    args = parser.parse_args()
    FIGS = args.output.resolve()
    if FIGS == (ROOT/'figures/v3.5.0').resolve():
        parser.error('Replay output must not overwrite the frozen original exports')
    FIGS.mkdir(parents=True, exist_ok=True)
    c.HERE = FIGS
    before = {p.relative_to(ROOT).as_posix(): c.sha(p) for p in PATHS.values()}
    outputs = []
    with plt.rc_context(c.STYLE):
        checks = {'figure1': diagram_figure(outputs, 'figure1'), 'figureS1': diagram_figure(outputs, 'figureS1'),
                  'figure5': matched_figure(outputs), 'figure6': python_figure(outputs),
                  'figureS2': project_figure(outputs), 'figureS3': audit_figure(outputs),
                  'figureS4': joint_figure(outputs), 'figureS5': python_policy_figure(outputs)}
        for method, source, target in [(c.rq1_candidate, 'candidate_rq1_coverage_and_paired_effects', 'figure3'),
                                        (c.rq2_candidate, 'candidate_rq2_target_count', 'figure4'),
                                        (c.rq5_candidate, 'candidate_rq5_seed_effects', 'figure7')]:
            checks[target] = method(outputs)
            for extension in ['svg', 'pdf', 'png']:
                old = FIGS/f'{source}.{extension}'
                new = FIGS/f'{target}.{extension}'
                old.replace(new)
                for item in outputs:
                    if item['file'] == old.name: item['file'] = new.name
    holm = []
    for comparison, formulas in load('rq1')['comparisons'].items():
        for formula, ties in formulas.items():
            for tie, item in ties.items():
                for budget, value in item['budgets'].items():
                    holm.append(dict(comparison=comparison, formula=formula, tie_rule=tie,
                                     paired_n=item['paired'], k=budget, paired_gains=value['gains'],
                                     paired_losses=value['losses'], p_raw=value['p_raw'],
                                     p_holm_128=value['p_holm_128']))
    assert len(holm) == 128
    csv_out('holm128_complete.csv', holm)
    assert all(c.sha(ROOT/name) == digest for name, digest in before.items())
    report = {'artifact_version': 'v3.5.0-public-20261015', 'source_sha256': before,
              'sources_unchanged': True, 'experimental_changes': 'none; frozen estimates and intervals reused',
              'omitted_figures': {'figure2': 'Third-party source excerpt not distributed'},
              'checks': checks, 'exports': outputs, 'dpi': 300,
              'font': 'DejaVu Sans; SVG text references a font rather than embedding font software',
              'scope': 'Frozen-statistic replay; not an independent full experiment rerun or journal certification'}
    (FIGS/'REPLAY_MANIFEST.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f'Replayed 11 figures and {len(holm)} Holm test records; source checksums unchanged.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
