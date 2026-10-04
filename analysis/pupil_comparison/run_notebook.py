"""Run the restored notebook's core pipeline, preserving executed cells and diagnostics.

Run from repository root. Archived step-search and unrelated later explorations
are excluded explicitly; processing and phase tracking cover the entire sample.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np

OUT = Path('analysis/pupil_comparison/results')
OUT.mkdir(parents=True, exist_ok=True)
nb = nbformat.read('anisotropy_rotation_processing.ipynb', as_version=4)
env = {'__name__': '__main__'}
selected = [4, 6, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 35, 36, 37]
executed = nbformat.v4.new_notebook(metadata=nb.metadata)
executed.cells.append(nbformat.v4.new_markdown_cell(
    'Core pipeline execution on files3_first40s.tdms, fit 30–31 s, manual cycle [0,700]. '
    'Cells 31–34 are archived step searches. Cell 39 needs an unavailable separate fixed-rod TDMS file; '
    'later interactive and unrelated explorations are not run.'))
current = [0]
def save_figures(*args, **kwargs):
    for j, f in enumerate(plt.get_fignums()):
        plt.figure(f).savefig(OUT / f'cell{current[0]:02d}_{j}.png', dpi=130)
    plt.close('all')
plt.show = save_figures
for i in selected:
    current[0] = i
    print(f'Executing cell {i}', flush=True)
    cell = nbformat.v4.new_code_cell(nb.cells[i].source)
    start = time.time()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compile(cell.source, f'notebook_cell_{i}', 'exec'), env)
        cell.execution_count = len(executed.cells)
        cell.outputs = [nbformat.v4.new_output('stream', name='stdout', text=buf.getvalue())]
    except Exception:
        cell.outputs = [nbformat.v4.new_output('stream', name='stderr', text=buf.getvalue()+traceback.format_exc())]
        executed.cells.append(cell)
        nbformat.write(executed, OUT / 'executed_core.ipynb')
        print(buf.getvalue(), flush=True)
        raise
    executed.cells.append(cell)
    nbformat.write(executed, OUT / 'executed_core.ipynb')
    print(buf.getvalue(), flush=True)
    print(f'Cell {i} completed in {time.time()-start:.1f}s', flush=True)
    if i == 25:
        names = ['Theta_axis_fitted','Phi_axis_fitted','Lambda_fitted','dc_fitted','fa_fitted','fb_fitted']
        b = np.array([env[x] for x in ['b0_fitted','b90_fitted','b45_fitted','b135_fitted']])
        fit = np.array([env[x] for x in ['_c0_cyc','_c90_cyc','_c45_cyc','_c135_cyc']])
        result = dict(zip(names, map(float,env['shape_fit_res'].x)))
        result.update(cost=float(env['shape_fit_res'].fun), optimizer_success=bool(env['shape_fit_res'].success),
                      optimizer_message=str(env['shape_fit_res'].message), background=b.tolist(),
                      inverse_gains_stack_90_45_135_0=list(env['a']),
                      ABC=[float(env[k]) for k in ['A_cal','B_cal','C_cal']],
                      retained_fit_samples=int(np.all(fit-b[:,None]>0,axis=0).sum()),
                      fit_samples=fit.shape[1], unique_matched_samples=int(env['_n_unique']))
        (OUT/'baseline.json').write_text(json.dumps(result,indent=2)+'\n')
        np.savez_compressed(OUT/'fit_channels.npz', channels=fit, backgrounds=b,
                            raw_fit=np.array([env[x] for x in ['c0_fit','c90_fit','c45_fit','c135_fit']]),
                            matched_indices=env['_matched_idx'])
    if i == 27:
        np.savez_compressed(OUT/'phase_trace.npz',time=env['time_proc'][::env['PHASE_DECIMATION']],psi=env['psi'])
        # Fixed, time-ordered held-out points retain all channels, without ridge selection.
        sel=np.arange(0,len(env['time_s']),250)
        raw=np.array([env[x][sel] for x in ['c0_raw','c90_raw','c45_raw','c135_raw']])
        np.savez_compressed(OUT/'validation_channels.npz',time=env['time_s'][sel],raw=raw)
print('Core execution complete.',flush=True)
