from pathlib import Path
import json,re,html
R=Path(__file__).resolve().parent
pages=json.load(open(R/'report_content.json'))
# Correct coefficient notation everywhere in the prose and display mathematics.
for page in pages:
 for b in page:
  if b[0]=='eq':b[1]=re.sub(r'\bA\b','A_F',b[1])
  if b[0]=='p':
   b[1]=b[1].replace('A=0.89080598','A_F=0.89080598').replace('A=0.890805977','A_F=0.890805977').replace('Subscripts F distinguish','The F subscripts distinguish').replace('A+B=1','A_F+B_F=1').replace('A,B,C','A_F,B_F,C_F').replace('Ar/(C−Br)','A_F r/(C_F−B_F r)').replace('C sin²θ/(A+B sin²θ)','C_F sin²θ/(A_F+B_F sin²θ)')
# Add concise fitted-quantity boxes; no optimization algorithm detail.
for page in pages:
 title=page[0][1]
 if title.startswith('2A.'):
  page.append(['fit','Common to the physical cone fits','Per interval: motor-axis tilt and azimuth, cone half-angle (3); starting mechanical phase and monotone phase increments (16); one constant brightness scale a₀ (1). Total: 20 local real parameters per interval. Counts describe the parameterization, not a guarantee that every parameter is identifiable. Optical coefficients, NA, Fresnel/BFP response and the current gains/matrix are held fixed.'])
 if title.startswith('2B.'):
  idx=next(i for i,b in enumerate(page) if b[0]=='p' and b[1].startswith('B, polarization'))
  page[idx:idx+1]=[['p','This is a <b>common-background assumption</b>, not a universal law for all offsets. One optical background observed through ideal linear analyzers has Stokes parameters S₀,S₁,S₂ with nonnegative coherence matrix. Setting B=2S₀ gives:'],['eq',r'B_j=\frac12\left[S_0+S_1\cos(2\psi_j)+S_2\sin(2\psi_j)\right]'],['eq',r'p=\frac{\sqrt{S_1^2+S_2^2}}{S_0}\leq1,\qquad B_0+B_{90}=B_{45}+B_{135}=B/2'],['p','B, p and α are constant. This permits polarized or unpolarized background while enforcing physical channel intensities and equal pair sums. “Positive” refers to the physical intensity/coherence matrix, not to requiring S₁ and S₂ individually to be positive. Circular Stokes S₃ is not measured by these linear analyzers. Electronic offsets, unequal detector acceptance, calibration errors or independent post-analyzer backgrounds need not satisfy this pattern; four independent nonnegative offsets would be a different model.'],['fit','Background only: fitted quantities','B, p, α (3 background parameters), plus the 20 local cone/phase/brightness parameters: 23 per interval. C=0. Fits are separate in the two windows.']]
  page.append(['fit','Richard effective model: fitted quantities','B, p, α and four constant c_j (7 background/interference parameters), plus the same 20 local parameters: 27 per interval. Fits are separate by window. The c_j are effective real overlaps, not separately measured phases.'])
 if title.startswith('2C.'):
  i=next(i for i,b in enumerate(page) if b[0]=='p' and b[1].startswith('<b>4.'))
  page.insert(i,['fit','Uniform field: fitted quantities','Two complex amplitudes b_x,b_y (4 real values) and noninterfering Stokes parameters U,p_U,α_U (3): 7 background parameters, or 27 including local parameters, per interval. Γ is fixed by the optics.'])
  page.append(['fit','Uniform + radial: fitted quantities','Add complex b_r (2 real values): 9 background parameters. Separate fits have 29 parameters per interval; the shared three-mode fit has 9 shared plus 20 local per window, totaling 49. κ is fixed by A_F.'])
 if title.startswith('2D.'):
  page.append(['fit','Fitted quantities for each stationary family','Five-mode: 5 complex b_m plus U,p_U,α_U = 13 shared background parameters; 53 including both windows. General stationary: 10 complex b_m plus U,p_U,α_U = 23 shared; 63 total. Edge: 6 complex b_m = 12 shared; 52 total, with U fixed to zero. Mode shapes, overlaps and edge width w are fixed for each case; background caps are imposed limits, not fitted parameters.'])
 if title.startswith('2E.'):
  i=next(i for i,b in enumerate(page) if b[0]=='p' and b[1].startswith('<b>Measured-mask'))
  page.insert(i,['fit','Axial motion: fitted quantities','The general field’s 23 shared background parameters, 20 local parameters per window, and R,δ per window: 67 total. Radius cap, wavelength and z_R are fixed scenario settings.'])
  i=next(i for i,b in enumerate(page) if b[0]=='p' and b[1].startswith('<b>Arbitrary-overlap'))
  page.insert(i,['fit','Measured mask: fitted quantities','The same three complex field amplitudes and noninterfering Stokes background as the separate three-mode fits: 29 parameters per window. Mask geometry/rotation and the resulting Q,H are fixed in each comparison.'])
  page.append(['fit','Flexible controls: what is adjustable','Free-overlap test: cone, phase and a₀ determine S; constant B_j are selected under scalar feasibility bounds (the 10% witness fixes equal B_j). Each ρ_j(χ) is then constructed point by point to match I_j, rather than predicted by one field. Earlier free-brightness fits similarly recover a(χ) from the measured total at every bin; this adds pointwise freedom even though those amplitudes are not independent optimizer coordinates.'])
# Split long model pages at family boundaries, retaining section identity.
out=[]
for page in pages:
 if page[0][1].startswith('2B.'):
  i=next(i for i,b in enumerate(page) if b[0]=='p' and b[1].startswith('<b>2. Richard'))
  out.extend([page[:i],[['h','2B (continued). Richard’s effective model']]+page[i:]])
 elif page[0][1].startswith('2E.'):
  i=next(i for i,b in enumerate(page) if b[0]=='p' and b[1].startswith('<b>Measured-mask'))
  out.extend([page[:i],[['h','2E (continued). Mask and flexible controls']]+page[i:]])
 else:out.append(page)
pages=out

# Editorial correction: distinguish the original notebook and the quoted four-C proposal.
for page in pages:
 title=page[0][1]
 if title.startswith('2B.'):
  page[0][1]='2B. Additive backgrounds: notebook versus Stokes'
  page[1:1]=[
   ['p','<b>The original notebook was less restrictive.</b> It fitted three background parameters, not four independent channel offsets:'],
   ['eq',r'(B_0,B_{90},B_{45},B_{135})=\frac{d_c}{2}(1+f_a,1-f_a,1+f_b,1-f_b)'],
   ['eq',r'd_c\geq0,\qquad -1\leq f_a\leq1,\qquad -1\leq f_b\leq1'],
   ['p','This guarantees nonnegative channel backgrounds and equal pair sums. The current Stokes version has the same algebra after the following substitution, but restricts the allowed region from a square to its inscribed disk:'],
   ['eq',r'B=2d_c,\quad f_a=p\cos(2\alpha),\quad f_b=p\sin(2\alpha),\quad f_a^2+f_b^2\leq1'],
   ['p','For example, f_a=f_b=1 is allowed by the notebook but excluded by a common physical Stokes background. Thus the two models agree inside the disk, but are not equivalent over their full allowed parameter ranges. Four fully independent offsets would be more general still because they need not have equal pair sums.'],
  ]
  # Save physical explanation for a continuation to keep this page readable.
  split=next(i for i,b in enumerate(page) if b[0]=='p' and b[1].startswith('This is a <b>common-background'))
  # Defer actual split until below.
  page.append(['p','The original solver fitted cone geometry (3) plus d_c,f_a,f_b (3) against anisotropy shape. The current comparison also constrains ordered channels and total power with one a₀ per interval. Consequently its residual/angle differences cannot be attributed solely to the disk constraint. Saved legacy square-versus-disk controls exist, but no new controlled refit was run for this revision.'])
 if title.startswith('2B (continued).'):
  page[:]=[
   ['h','2C. Richard’s four-coefficient approximation'],
   ['p','The quoted email proposes a correction whose value changes with the real signal, but whose coefficient is constant within each channel:'],
   ['eq',r'I_{m,j}=I_{r,j}+D_j(I_{r,j}),\qquad D_j(I_{r,j})=C_j\sqrt{I_{r,j}}'],
   ['eq',r'T(\theta,\phi)=S(\theta)+\sum_j C_j\sqrt{S_j(\theta,\phi)}'],
   ['fit','What Richard’s quoted model fits','Four real coefficients C_j, one per channel. With our common fixed-excitation cone framework this would mean 4 correction parameters plus 20 local parameters = 24 per separately fitted interval. This exact four-coefficient model has NOT been run in the saved comparison.'],
   ['p','The C_j can have either sign. They are constants in this approximation; D_j varies because the signal varies. Phase is not absent: phase and spatial overlap are absorbed into each C_j. For a fixed, perfectly matched single-mode background field of amplitude b_j, the full optical intensity is:'],
   ['eq',r'I_{m,j}=I_{r,j}+|b_j|^2+2|b_j|\cos\delta_j\sqrt{I_{r,j}}'],
   ['eq',r'C_j=2|b_j|\cos\delta_j\quad\hbox{(matched single mode)}'],
   ['p','Writing C=2 cos(phase) requires the background amplitude to have been normalized to one or absorbed into the intensity units. Otherwise C has units of square-root intensity. With spatially integrated fields, the effective coefficient includes overlap and need not stay constant as rod orientation changes.'],
   ['p','The quoted equation omits the background-only intensity |b_j|². It therefore describes an approximation where that term is neglected, or a signal from which it has already been removed. It is not obtained by setting the background power to zero in a full coherent model: that would also remove the cross term.'],
   ['p','For positive measured intensity, the nonnegative-amplitude solution is:'],
   ['eq',r'I_{r,j}=\left[\frac{-C_j+\sqrt{C_j^2+4I_{m,j}}}{2}\right]^2'],
   ['p','The email’s suggested x,y,S₁,S₂ notation is a possible way to express four quantities; without its precise definitions it should not be equated with the three-parameter physical Stokes background used in 2B. The direct C_j representation avoids that ambiguity.'],
  ]
 elif title.startswith('2C.'):
  page[0][1]=title.replace('2C.','2D.',1)
 elif title.startswith('2D.'):
  page[0][1]=title.replace('2D.','2E.',1)
 elif title.startswith('2E.'):
  page[0][1]=title.replace('2E.','2F.',1)
 elif title.startswith('2E (continued).'):
  page[0][1]=title.replace('2E (continued).','2F (continued).',1)
# Actual fitted extension gets its own page and an unambiguous label.
new=[]
for page in pages:
 if page[0][1].startswith('2B.'):
  split=next(i for i,b in enumerate(page) if b[0]=='p' and b[1].startswith('This is a <b>common-background'))
  new.extend([page[:split],[['h','2B (continued). Meaning of the Stokes restriction']]+page[split:]])
 else:new.append(page)
 if page[0][1].startswith('2C.'):
  new.append([
   ['h','2C (continued). The extension actually fitted'],
   ['p','The plots previously labeled “Richard effective” are from a different, seven-parameter background/interference model:'],
   ['eq',r'I_{m,j}=S_j+B_j+2c_j\sqrt{B_jS_j},\qquad |c_j|\leq1'],
   ['eq',r'T=S+B+2\sum_j c_j\sqrt{B_jS_j}'],
   ['eq',r'C_j^{\mathrm{eff}}=2c_j\sqrt{B_j},\qquad |C_j^{\mathrm{eff}}|\leq2\sqrt{B_j}'],
   ['fit','Fitted quantities in the saved extension','Three constant Stokes-background parameters B,p,α plus four constant real overlap coefficients c_j: 7, or 27 including the local cone/phase/brightness parameters, per interval. Coefficients and backgrounds were fitted separately in the two windows.'],
   ['p','This is additive background plus an effective interference term. Relative phase and spatial overlap are folded into c_j rather than fitted separately. At a fixed c_j its interference magnitude follows √S_j, and its channelwise sign is fixed; the total signed cross term may still change sign as channels vary.'],
   ['p','The added B_j term and its bound on C_j distinguish this extension from Richard’s quoted approximation. The extension reduces to the quoted mathematical form only after its additive B_j is removed from the measured intensity, with the remaining coefficient identified as C_j=2c_j√B_j. That identification still retains constraints absent from a free four-C fit.'],
   ['p','Neither four independent c_j nor the bound alone proves that one stationary vector field can realize them throughout a trajectory. In a coherent field model, the overlap may depend on orientation; treating it as constant is the approximation being tested.'],
   ['p','All displayed fit scores, fitted background powers and angle ranges under the revised label “Additive + sqrt extension” refer to this saved seven-parameter model. They cannot be used to claim that Richard’s exact four-coefficient proposal has succeeded or failed. No new fits were run for this editorial correction.'],
  ])
pages=new
# Relabel the scientific record without changing any fitted values.
def relabel(x):
 if isinstance(x,str):return x.replace('Richard effective intensity','Additive + sqrt extension').replace('Richard effective','Additive + sqrt extension').replace('Richard-effective','additive-plus-sqrt').replace('Richard’s model need not win','The additive-plus-sqrt extension need not win')
 if isinstance(x,list):return [relabel(a) for a in x]
 return x
for page in pages:
 if not page[0][1].startswith(('2B','2C')):page[:]=relabel(page)

# Text conversion: native TeX math for symbolic runs; native Unicode math font for displays.
sub={'₀':'0','₁':'1','₂':'2','₃':'3','₄':'4','₅':'5','₆':'6','₇':'7','₈':'8','₉':'9','ₘ':'m','ⱼ':'j'}
sup={'⁰':'0','¹':'1','²':'2','³':'3','⁴':'4','⁵':'5','⁶':'6','⁷':'7','⁸':'8','⁹':'9','⁻':'-'}
greek='θφχψακΓδρλΣΔπµϕ'
def txt(s):
 s=html.unescape(str(s));tokens=[]
 def token(v):tokens.append(v);return f'ZZPLACE{len(tokens)-1}ZZ'
 for pat,repl in [(r'<b>(.*?)</b>',lambda m:token(r'\textbf{'+txt(m[1])+'}')),(r'<sub>(.*?)</sub>',lambda m:token(r'\textsubscript{'+txt(m[1])+'}')),(r'<sup>(.*?)</sup>',lambda m:token(r'\textsuperscript{'+txt(m[1])+'}'))]:s=re.sub(pat,repl,s)
 s=s.replace('<br/>','\n')
 s=re.sub(r'https://[^ ;<]+',lambda m:token(r'\url{'+m[0]+'}'),s)
 s=re.sub(r'(?<![A-Za-z0-9])([A-Za-z])_([A-Za-z0-9]+)',lambda m:token(r'\('+m[1]+'_{'+m[2]+'}'+r'\)'),s)
 s=re.sub('['+''.join(sub)+']+',lambda m:token(r'\textsubscript{'+''.join(sub[c] for c in m[0])+'}'),s)
 s=re.sub('['+''.join(sup)+']+',lambda m:token(r'\textsuperscript{'+''.join(sup[c] for c in m[0])+'}'),s)
 for c in greek:s=s.replace(c,token(r'\('+ ('\mu' if c=='µ' else c)+r'\)'))
 for c,math in {'√':r'\surd','∑':r'\sum','∫':r'\int','≤':r'\leq','≥':r'\geq','≈':r'\approx','∞':r'\infty','×':r'\times','→':r'\rightarrow','−':'-','±':r'\pm','∝':r'\propto','∈':r'\in','∗':'*'}.items():s=s.replace(c,token(r'\('+math+r'\)'))
 s=re.sub(r'([%&#_$])',r'\\\1',s);s=s.replace('~',r'\textasciitilde{}').replace('°',r'\textdegree{}').replace('\n',r'\newline ')
 for i,t in enumerate(tokens):s=s.replace(f'ZZPLACE{i}ZZ',t)
 return s
header=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=17mm,headheight=14pt,footskip=10mm]{geometry}
\usepackage{fontspec,unicode-math}
\setmainfont{Latin Modern Roman}
\setsansfont{Latin Modern Sans}
\setmathfont{Latin Modern Math}
\usepackage{amsmath,graphicx,booktabs,array,tabularx,xcolor,colortbl,fancyhdr,enumitem}
\usepackage[most]{tcolorbox}
\usepackage[hidelinks]{hyperref}
\definecolor{ink}{HTML}{123D54}
\definecolor{pale}{HTML}{EFF4F6}
\pagestyle{fancy}\fancyhf{}
\fancyfoot[L]{\sffamily\scriptsize APD background / interference — discussion draft — 6 October 2026}
\fancyfoot[R]{\sffamily\scriptsize\thepage}
\renewcommand{\headrulewidth}{0pt}
\emergencystretch=3em
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\setlength{\abovedisplayskip}{7pt}\setlength{\belowdisplayskip}{7pt}
\renewcommand{\arraystretch}{1.18}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\newcommand{\heading}[1]{{\sffamily\Large\bfseries\color{ink}#1}\par\vspace{7pt}}
\newtcolorbox{fitbox}[1]{colback=pale,colframe=ink,boxrule=0.35pt,arc=0pt,left=6pt,right=6pt,top=4pt,bottom=4pt,title=#1,fonttitle=\sffamily\bfseries\small,fontupper=\small}
\begin{document}
'''
parts=[header]
for i,page in enumerate(pages):
 if i:parts.append(r'\clearpage')
 for b in page:
  if b[0]=='h':parts.append(r'\heading{'+txt(b[1])+'}')
  elif b[0]=='p':parts.append(txt(b[1])+r'\par')
  elif b[0]=='eq':parts.append(r'\begin{equation*}'+b[1]+r'\end{equation*}')
  elif b[0]=='fit':parts.append(r'\begin{fitbox}{'+txt(b[1])+'}'+txt(b[2])+r'\end{fitbox}')
  elif b[0]=='fig':
   name,w,h=b[1:];h=min(h or 6.5,7.7);parts.append(r'\begin{center}\includegraphics[width=\linewidth,height='+str(h)+r'in,keepaspectratio]{figures/'+name+r'.png}\end{center}')
  elif b[0]=='table':
   heads,rows,widths=b[1:];n=len(heads);widths=widths or [1]*n;tot=sum(widths);spec=''.join('>{\\hsize='+f'{n*w/tot:.3f}'+r'\hsize}Y' for w in widths)
   parts.append(r'{\footnotesize\setlength{\tabcolsep}{4pt}\begin{tabularx}{\linewidth}{'+spec+r'}\toprule\rowcolor{pale}')
   parts.append(' & '.join(r'\textbf{'+txt(h)+'}' for h in heads)+r'\\\midrule')
   parts.extend(' & '.join(txt(c) for c in row)+r'\\' for row in rows)
   parts.append(r'\bottomrule\end{tabularx}}\par\medskip')
parts.append(r'\end{document}')
(R/'APD_notes.tex').write_text('\n'.join(parts));(R/'latex_content.json').write_text(json.dumps(pages,indent=2))
print('Wrote',len(pages),'planned pages')
