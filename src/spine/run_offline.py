"""Run the offline spine observability analysis and write results/spine_offline_v1.

Usage: python -m spine.run_offline [--out results/spine_offline_v1]
(run with src/ on PYTHONPATH). Every number written here is a model-based
prediction from synthetic deadlift postures, not a measurement.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from .spine_information import (REL_TOL, WELL_DETERMINED_DEG, analyze_static,  # noqa: E402
                                n_well_determined, project_smooth, trajectory_information)
from .spine_kinematics import MODELS  # noqa: E402
from .spine_observation import CONFIG_LABELS, CONFIGS, NotRepresentable  # noqa: E402

LABEL = 'Model-based prediction (synthetic deadlift posture; not measured data)'
SURFACE, INK, INK2, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
SETTINGS = {'phase': 0.5, 'sigma_ori_deg': 2.0, 'sigma_bend_deg': 2.0,
            'trajectory_frames': 41, 'kappas': [0.1, 0.5, 2.0], 'smooth_modes': 6}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def finite(v):
    return None if v is None or not np.isfinite(v) else round(v, 3)


def static_table():
    rows = []
    for mname, build in MODELS.items():
        model = build()
        for config in CONFIGS:
            try:
                res = analyze_static(model, config, phase=SETTINGS['phase'],
                                     sigma_ori_deg=SETTINGS['sigma_ori_deg'],
                                     sigma_bend_deg=SETTINGS['sigma_bend_deg'])
            except NotRepresentable as exc:
                rows.append({'model': mname, 'config': config, 'representable': False,
                             'reason': str(exc)})
                continue
            row = {'model': mname, 'config': config, 'representable': True,
                   'joint_dof': res['meas']['dim']}
            for key in ('meas', 'post'):
                s = res[key]
                row[key] = {'rank': s['rank'], 'n_well_determined': s['n_well_determined'],
                            'logdet_plus': round(s['logdet_plus'], 3),
                            'lambda_min_plus': float(f"{s['lambda_min_plus']:.4g}"),
                            'cond_plus': float(f"{s['cond_plus']:.4g}"),
                            'eigenvalues': [float(f'{e:.4g}') for e in s['eigenvalues']]}
            row['qoi'] = {k: (None if v is None else {kk: finite(vv) for kk, vv in v.items()})
                          for k, v in res['qoi'].items()}
            rows.append(row)
    return rows


def trajectory_spectra():
    out = {}
    n = SETTINGS['trajectory_frames']
    model = MODELS['M3']()
    cases = [('C0', None), ('C1', None)] + [('C0', k) for k in SETTINGS['kappas']]
    for config, kappa in cases:
        info = trajectory_information(model, config, n_frames=n,
                                      sigma_ori_deg=SETTINGS['sigma_ori_deg'], kappa=kappa)
        smooth = project_smooth(info, n, 3 * model.n_joints, SETTINGS['smooth_modes'])
        eig = np.clip(np.linalg.eigvalsh(smooth)[::-1], 0, None)
        tol = REL_TOL * eig[0]
        key = f'{config}_orientation_only' if kappa is None else f'{config}_plus_acc_kappa_{kappa}'
        out[key] = {'model': 'M3', 'config': config, 'kappa': kappa, 'frames': n,
                    'smooth_modes': SETTINGS['smooth_modes'],
                    'smooth_dims': int(smooth.shape[0]), 'rank': int((eig > tol).sum()),
                    'n_well_determined': n_well_determined(eig),
                    'eigenvalues': [float(f'{e:.4g}') for e in eig]}
    return out


def fig_rank(rows, path):
    configs = list(CONFIGS)
    fig, ax = plt.subplots(figsize=(7.2, 3.4), facecolor=SURFACE)
    style(ax)
    width = 0.26
    for k, mname in enumerate(MODELS):
        xs, hs = [], []
        for i, c in enumerate(configs):
            r = next(r for r in rows if r['model'] == mname and r['config'] == c)
            if r['representable']:
                xs.append(i + (k - 1) * width)
                hs.append(r['meas']['rank'])
        bars = ax.bar(xs, hs, width - 0.03, color=SERIES[k], label=mname, zorder=2)
        for b, h in zip(bars, hs):
            ax.text(b.get_x() + b.get_width() / 2, h + 0.15, str(h), ha='center',
                    va='bottom', fontsize=7, color=INK2)
    for i, c in enumerate(configs):
        if c.startswith('C2'):
            ax.text(i - width, 0.3, 'n/a', ha='center', fontsize=7, color=INK2, rotation=90)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([CONFIG_LABELS[c].split('  ')[0] for c in configs], color=INK)
    ax.set_ylabel('rank of I_meas on joint angles', color=INK2, fontsize=8)
    ax.set_ylim(0, 8.5)
    ax.set_title('Sensor-only information rank (orientation + bend, one frame)',
                 color=INK, fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper left', labelcolor=INK)
    fig.text(0.01, 0.01, LABEL, fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_qoi(rows, path):
    configs = [c for c in CONFIGS]
    qois = [('lumbar_flexion_L1_S1', 'Lumbar flexion L1-S1'),
            ('lumbar_flexion_L1_L5', 'Lumbar flexion L1-L5'),
            ('lumbosacral_flexion_L5_S1', 'Lumbosacral flexion L5/S1')]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), facecolor=SURFACE, sharey=True)
    width = 0.38
    for ax, (q, title) in zip(axes, qois):
        style(ax)
        prior = None
        for k, mname in enumerate(['M2', 'M3']):
            xs, hs, unident = [], [], []
            for i, c in enumerate(configs):
                r = next(r for r in rows if r['model'] == mname and r['config'] == c)
                v = r.get('qoi', {}).get(q) if r['representable'] else None
                if v is None:
                    continue
                prior = v['prior_std_deg']
                xs.append(i + (k - 0.5) * width)
                hs.append(v['posterior_std_deg'])
                unident.append(v['meas_only_std_deg'] is None)
            bars = ax.bar(xs, hs, width - 0.04, color=SERIES[k], label=mname, zorder=2)
            for b, u in zip(bars, unident):
                if u:      # not identifiable from sensors alone: posterior is prior-driven
                    b.set_hatch('////')
                    b.set_edgecolor(SURFACE)
        if prior is not None:
            ax.axhline(prior, color=INK2, linestyle='--', linewidth=1)
            ax.text(-0.45, prior + 0.1, 'prior only', ha='left', va='bottom',
                    fontsize=7, color=INK2)
        ax.set_title(title, color=INK, fontsize=9, loc='left')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([CONFIG_LABELS[c].split('  ')[0] for c in configs],
                           fontsize=7, color=INK)
    axes[0].set_ylabel('posterior std (deg)', color=INK2, fontsize=8)
    from matplotlib.patches import Patch
    handles = [Patch(color=SERIES[0], label='M2'), Patch(color=SERIES[1], label='M3'),
               Patch(facecolor=AXIS, hatch='////', edgecolor=SURFACE,
                     label='not identifiable from sensors alone')]
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=3, loc='upper right',
               labelcolor=INK)
    fig.text(0.01, 0.01, LABEL + '.  Hatched bars: the posterior is set by the ROM prior.',
             fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_spectra(rows, traj, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURFACE)
    style(a1)
    for k, c in enumerate(['C0', 'C1', 'C1pp']):
        r = next(r for r in rows if r['model'] == 'M3' and r['config'] == c)
        em = np.array(r['meas']['eigenvalues'])
        ep = np.array(r['post']['eigenvalues'])
        floor = 1e-2
        idx = np.arange(1, em.size + 1)
        a1.semilogy(idx, np.maximum(em, floor), '-', color=SERIES[k], linewidth=2,
                    marker='o', markersize=4, label=f'{c} I_meas')
        a1.semilogy(idx, ep, '--', color=SERIES[k], linewidth=1.5, label=f'{c} H_post')
    a1.axhline(1e-2, color=AXIS, linewidth=0.8)
    a1.text(1, 1.4e-2, 'zero (null space)', fontsize=7, color=INK2)
    a1.set_title('M3 single frame: solid I_meas, dashed H_post', color=INK, fontsize=9,
                 loc='left')
    a1.set_xlabel('eigenvalue index', color=INK2, fontsize=8)
    a1.set_ylabel('information (rad^-2)', color=INK2, fontsize=8)
    a1.legend(frameon=False, fontsize=7, labelcolor=INK, ncol=2, loc='center right')
    a1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    style(a2)
    show = [('C0_orientation_only', 'C0 orientation only'),
            ('C0_plus_acc_kappa_0.5', 'C0 + acceleration (kappa 0.5)'),
            ('C1_orientation_only', 'C1 orientation only')]
    for k, (key, lab) in enumerate(show):
        e = np.array(traj[key]['eigenvalues'])
        e = e[e > 0]
        a2.semilogy(np.arange(1, e.size + 1), e, '-', color=SERIES[k], linewidth=2,
                    label=f"{lab}: {traj[key]['n_well_determined']} determined")
    thr = 1.0 / np.deg2rad(WELL_DETERMINED_DEG) ** 2
    a2.axhline(thr, color=INK2, linestyle=':', linewidth=1)
    a2.text(1, thr * 1.3, f'std = {WELL_DETERMINED_DEG:g} deg', fontsize=7, color=INK2)
    a2.set_title(f"M3 one repetition, slow modes (K={SETTINGS['smooth_modes']}): "
                 'joint-trajectory spectrum', color=INK, fontsize=9, loc='left')
    a2.set_xlabel('eigenvalue index', color=INK2, fontsize=8)
    a2.set_ylim(1e-6, 3e5)
    a2.set_xlim(0, 60)
    a2.legend(frameon=False, fontsize=7, labelcolor=INK, loc='lower left')
    fig.text(0.01, 0.01, LABEL + '.  kappa = sigma_a tau^2 / L (dimensionless; acceleration '
             'unit pending).', fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def write_markdown(rows, traj, path):
    lines = ['# spine_offline_v1 summary', '', f'> {LABEL}', '',
             f"Settings: posture phase {SETTINGS['phase']}, orientation noise "
             f"{SETTINGS['sigma_ori_deg']} deg/axis, bend noise {SETTINGS['sigma_bend_deg']} deg.",
             '', '## Sensor-only rank and posterior std of lumbar flexion', '',
             f'"determined" = eigen-direction with sensor-only std < {WELL_DETERMINED_DEG:g} deg.', '',
             '| model | config | joint DoF | rank I_meas | determined dirs | logdet+ I_meas | rank H_post | '
             'L1-S1 post std (deg) | L1-L5 post std | L5/S1 post std | identifiable from sensors alone |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        if not r['representable']:
            lines.append(f"| {r['model']} | {r['config']} | - | n/a | - | - | - | - | - | - | "
                         'span does not align with model joints |')
            continue
        q = r['qoi']

        def post(name):
            v = q.get(name)
            return '-' if v is None else f"{v['posterior_std_deg']:.2f}"
        ident = [lab for name, lab in (('lumbar_flexion_L1_S1', 'L1-S1'),
                                       ('lumbar_flexion_L1_L5', 'L1-L5'),
                                       ('lumbosacral_flexion_L5_S1', 'L5/S1'))
                 if q.get(name) and q[name]['meas_only_std_deg'] is not None]
        lines.append(f"| {r['model']} | {r['config']} | {r['joint_dof']} | {r['meas']['rank']} | "
                     f"{r['meas']['n_well_determined']} | "
                     f"{r['meas']['logdet_plus']:.2f} | {r['post']['rank']} | "
                     f"{post('lumbar_flexion_L1_S1')} | {post('lumbar_flexion_L1_L5')} | "
                     f"{post('lumbosacral_flexion_L5_S1')} | {', '.join(ident) or 'none'} |")
    lines += ['', '## Slow joint-trajectory information over one repetition (M3)', '',
              f"Joint trajectories restricted to the first {SETTINGS['smooth_modes']} cosine modes "
              '(slow posture changes); kappa = sigma_a tau^2 / L.', '',
              '| case | dims | numerical rank | determined dirs (std < 5 deg) | largest eig | smallest positive eig |',
              '|---|---:|---:|---:|---:|---:|']
    for key, v in traj.items():
        e = [x for x in v['eigenvalues'] if x > 0]
        lines.append(f"| {key} | {v['smooth_dims']} | {v['rank']} | {v['n_well_determined']} | {e[0]:.3g} | "
                     f"{e[v['rank'] - 1]:.3g} |")
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/spine_offline_v1')
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = static_table()
    traj = trajectory_spectra()
    (out / 'static_information.json').write_text(json.dumps(
        {'label': LABEL, 'settings': SETTINGS, 'rows': rows}, indent=1), encoding='utf-8')
    (out / 'trajectory_information.json').write_text(json.dumps(
        {'label': LABEL, 'settings': SETTINGS, 'cases': traj}, indent=1), encoding='utf-8')
    fig_rank(rows, out / 'fig_rank_by_config.png')
    fig_qoi(rows, out / 'fig_lumbar_flexion_std.png')
    fig_spectra(rows, traj, out / 'fig_spectra.png')
    write_markdown(rows, traj, out / 'summary.md')
    print((out / 'summary.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
