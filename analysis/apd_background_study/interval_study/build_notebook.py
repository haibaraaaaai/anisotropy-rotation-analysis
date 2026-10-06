from pathlib import Path
import json,base64
import numpy as np
r=Path(__file__).resolve().parent
out=r.parent/'deliverables/trajectory_repeatability.ipynb'
s=(r/'study.py').read_text()
s=s[s.index('from pathlib import Path'):]
s=s.replace("ROOT=Path(__file__).resolve().parent\nraw=np.load(ROOT.parent/'prefit_audit/legacy.npz')['raw']\nN=128;DT=4e-6",'''ROOT=OUT
from nptdms import TdmsFile
with TdmsFile.open(TDMS_PATH) as f:
    channels_tdms=f.groups()[0].channels()
    DT=float(channels_tdms[0].properties['wf_increment'])
    offset=float(channels_tdms[0].properties.get('wf_start_offset',0))
    lo=round((WINDOW_START-offset)/DT);hi=round((WINDOW_END-offset)/DT)
    raw=np.array([ch[lo:hi] for ch in channels_tdms],float)[[3,0,1,2]]
N=N_PHASE_BINS''')
s=s.replace("n=len(ref);D=", "n=len(ref);D=")
s=s.replace("pp=g[a:b];ss=", "pp=g[a:b]\n  if len(pp)<N:raise ValueError('A detected interval has fewer samples than phase bins; review the gate.')\n  ss=")
s=s.replace('30+bounds', 'WINDOW_START+bounds')
start=s.index("nb=json.loads(");end=s.index("cal=ns['T_Icor_Matrix']()",start)
nb=json.load(open(r.parent/'prefit-pipeline/anisotropy_rotation_processing.ipynb'))
model=''.join(nb['cells'][6]['source']);a=model.index('def T_Icor_Matrix');b=model.index('# ── Gain fit',a)
s=s[:start]+model[a:b]+"\nns={'T_Icor_Matrix':T_Icor_Matrix}\na=np.asarray(GAIN_STACK_90_45_135_0)[[3,0,1,2]]\n"+s[end:]
start=s.index("old=np.load(");end=s.index("axs[2].plot(qo",start)
z=np.load(r.parent/'prefit-pipeline/analysis/pupil_comparison/report/fit_channels.npz')['raw_fit']
q=np.column_stack([(z[0]-z[1])/(z[0]+z[1]),(z[2]-z[3])/(z[2]+z[3])])
s=s[:start]+"# Archived 90-point reference, embedded only for the historical comparison.\nqo=np.array("+repr(q.tolist())+");"+s[end:]
s=s.replace("print(json.dumps(metrics,indent=2))", "print(json.dumps(metrics,indent=2))\ntry:\n    from IPython.display import display, Image\nexcept ImportError:\n    display = print\n    def Image(filename): return filename\nfor name in ['repeatability.png','interval_groups.png','sensitivity_and_calibration.png']:\n    display(Image(filename=str(ROOT/name)))")
# Auxiliary checks use the same already-defined functions and original raw window.
t=(r/'extra_checks.py').read_text();start=t.index("z=np.load(R/'averaged_channels.npz')")
t="R=OUT\n"+t[start:]
t=t.replace("raw=ns['raw'];p=guide(raw,41)", "p=guide(raw,41)")
t=t.replace("with TdmsFile.open(R.parent/'anisotropy-rotation-analysis/files3_first40s.tdms') as f:", "with TdmsFile.open(TDMS_PATH) as f:")
t=t.replace("a=round(start/4e-6);e=a+250000", "a=round((start-offset)/DT);e=a+round(1/DT)")
t+="\nfor name in ['chronological_vs_interleaved.png','separated_windows.png']:\n    display(Image(filename=str(OUT/name)))\n"
summary='''# Does the pre-fit trajectory repeat?

Local investigation, 5 October 2026. No repository changes or background/interference fits.

**Result:** the broad two-pass curve repeats, and averaging substantially reduces jitter.
The precise curve is not invariant across chronological intervals. Preserve that variability
instead of treating one smooth average as known ground truth.

- 30–31 s: 304 complete *detected traversals*, each with one clockwise outer-loop winding.
  A data-specific outer-loop crossing defines boundaries; this does not independently measure mechanical phase.
- Even/odd averages (152 cycles each) differ by about **0.008 geometric RMS** in raw X/Y.
- Eight consecutive 38-cycle averages differ from the full mean by **0.013–0.039**;
  eight interleaved groups of the same size differ by **0.009–0.017**. Chronological
  differences exceed those obtained by mixing cycles across the second. This is
  evidence of time-correlated variation, not a formal separation of detector noise,
  preprocessing error, background drift and rod motion.
- Constrained time alignment changes the overall mean by **0.026 geometric RMS**.
  It reduces registration error but must not be accepted just because curves agree better.
- Guide windows 21–81 samples change the unwarped mean by up to **0.015 geometric RMS**
  relative to the 41-sample default. Fine features smaller than this are not yet robust.
- Separated windows at 0, 10, 20, 30 and 39 s retain the broad two-pass structure.
  10–11 s is more consistent across its four subintervals than 30–31 s.

**Correction to the previous interpretation:** the selected first cycle reaching Y≈0.54
was not proof that the ensemble trajectory should reach that height. The 304-cycle
mean reaches Y≈0.39 without the old 201-sample periodic filter. That earlier single-cycle
comparison overstated the evidence for bias in the ensemble reference. The old filter
still demonstrably changes individual traces; neither old nor new reference is ground truth.

## What is averaged

The four original channel voltages are averaged within contiguous bins on each
observed cycle, then corresponding bins are averaged with **equal weight per cycle**.
X/Y are calculated afterwards. Cycles are not weighted by how long they take.
A light channel smoother sets bin boundaries only. Both repeated passes remain in
acquisition order. No cone or optical anisotropy template is imposed.

The optional alignment uses a reference from even cycles only, with monotone mappings,
fixed endpoints, a ±12-bin band and a step penalty. Some source bins are reused under
that mapping (about 22% of mapping steps repeat an index); those outputs are correlated.
It is a sensitivity diagnostic, not the default recommended final reference. The
unwarped and chronological-group comparisons are essential controls.

All comparisons use the same calibration. The archived gain estimates are held fixed;
T keeps the current numerical coefficients and /8 scale. Fresnel and pupil inversion
are not needed for this empirical repeatability test. No angle inference is performed.

## Interpretation and next step

Use a **time window containing several complete traversals**, review a single ordered
cycle as a branch-order reference, and check agreement across subwindows. This dataset
allows automatic boundaries using an isolated outer-loop crossing, avoiding hundreds
of manual cycle selections. The gate is dataset-specific and must be visually reviewed
on another sample. There is no universal one-track/two-track switch or assumed speed.

The old single-cycle binning is useful diagnostically but too noisy to promote directly
as the final fit reference. A candidate final fit should use multiple chronological
averages, retain between-window scatter, and demonstrate consistent conclusions across
registration choices. Do not infer that a good fit to the grand average fits every cycle.

## Run

Requires numpy, scipy, matplotlib, npTDMS and IPython. Put this notebook beside
`files3_first40s.tdms`, or edit `TDMS_PATH`. Saved outputs and plots are embedded below;
rerunning reads the raw data. The final cell also checks four other one-second windows.
The current numerical summary applies only to the supplied sample and settings.
'''
config='''from pathlib import Path
TDMS_PATH = Path("files3_first40s.tdms")
OUT = Path("trajectory_repeatability_outputs")
OUT.mkdir(exist_ok=True)
WINDOW_START, WINDOW_END = 30.0, 31.0
N_PHASE_BINS = 128
GAIN_STACK_90_45_135_0 = [1.00826325584303, 1.2803656485620056, 1.091743363827011, 1.0]
'''
def cell(kind,source,ident,outputs=None,count=None):
 c=dict(cell_type=kind,id=ident,metadata={},source=source.splitlines(keepends=True))
 if kind=='code':c.update(execution_count=count,outputs=outputs or [])
 return c
def image_output(name):
 return dict(output_type='display_data',metadata={},data={'image/png':base64.b64encode((r/name).read_bytes()).decode(),'text/plain':[name]})
outputs=[dict(output_type='stream',name='stdout',text=(r/'metrics.json').read_text().splitlines(keepends=True))]+[image_output(n) for n in ['repeatability.png','interval_groups.png','sensitivity_and_calibration.png']]
outputs2=[dict(output_type='stream',name='stdout',text=(r/'extra_metrics.json').read_text().splitlines(keepends=True))]+[image_output(n) for n in ['chronological_vs_interleaved.png','separated_windows.png']]
nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},cells=[cell('markdown',summary,'summary'),cell('code',config,'configuration',count=1),cell('code',s,'main-comparison',outputs,2),cell('markdown','## Chronological versus interleaved groups, and separated windows\n\nThe same group size separates time-local variation from variability after mixing cycles across the whole second.\n','controls'),cell('code',t,'independent-controls',outputs2,3)])
for c in nb['cells']:
 if c['cell_type']=='code':compile(''.join(c['source']),'<notebook>','exec')
out.write_text(json.dumps(nb,indent=1)+'\n')
(r/'README.md').write_text(summary)
print(out,out.stat().st_size)
