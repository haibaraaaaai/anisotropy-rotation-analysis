"""Consistent vector/scalar notation for the current report (not legacy reports)."""
import re
KEY=r'''
\subsection*{Notation: scalars, vectors and spatial fields}
Bold symbols denote vectors or matrices; plain components and channel-indexed intensities are scalars.
\begin{center}\begin{tabular}{lp{112mm}}\toprule
Symbol & Type and meaning\\\midrule
$\mathbf d,\mathbf k,\mathbf e_1,\mathbf e_2$ & Real three-vectors: rod direction, cone axis and orthonormal cone basis.\\
$\boldsymbol\xi$ & Two-dimensional pupil position; not a polarization vector.\\
$\mathbf E_b(\boldsymbol\xi),\mathbf U_m(\boldsymbol\xi)$ & Two-component complex polarization fields, defined at every pupil position. The mode patterns are fixed.\\
$b_m$ & Complex scalar coefficient fitted for mode $m$: amplitude and phase.\\
$\mathbf a_i$ & Real two-component analyser vector.\\
$E_{b,i},E_{s,i}$ & Complex scalar spatial fields after projection onto analyser $i$.\\
$I_i,S_i,I_{\mathrm{int},i},B_i,N_i$ & Real scalar channel intensities after spatial integration; only the interference contribution can have either sign.\\
$\mathbf I,\mathbf B$ & Four-component channel lists, not directions in physical space.\\
$\mathbf e_j,\mathbf J_j$ & Two-component anisotropy residual and its $2\times2$ local angular Jacobian.\\\bottomrule
\end{tabular}\end{center}
The distinction before and after an analyser is
\[
\mathbf E_b(\boldsymbol\xi)=\sum_m b_m\mathbf U_m(\boldsymbol\xi),\qquad
E_{b,i}(\boldsymbol\xi)=\mathbf a_i^{\mathsf T}\mathbf E_b(\boldsymbol\xi).
\]
Pupil arguments are suppressed below when unambiguous. Components such as $d_x,d_y$ remain plain scalars. $I_{\mathrm{int},i}$ replaces the former symbol $L_i$ for the complete interference intensity, including brightness. $\mathcal{L}_i$ is a linear optical operator, not that intensity.

\textbf{Noninterfering does not mean unpolarized.} The extra $N_i$ in physical field models is a potentially polarized common optical background with a positive Stokes constraint. Unpolarized light is its equal-channel special case. Independent detector offsets or downstream path backgrounds need not obey this common-field restriction. The standalone additive model uses the looser equal-pair-sum square constraint, as requested.

\textbf{Why these models?} Richard proposed the constant four-$C_i$ effective approximation and discussed unknown phase and spatial variation. The uniform, dipole-mode and general stationary bases are our chosen approximations to those possibilities, not a catalogue of the only possible sources or models prescribed by Richard. The ten-mode basis frees scalar spatial patterns in each polarization; it is not strictly the five-mode basis plus five new modes, since the uniform patterns need not lie in its span.
\newpage
'''
def format_notation(tex):
    tex=tex.replace(r'\mathcal L_i',r'\mathcal{L}_i')
    tex=tex.replace('L_i',r'I_{\mathrm{int},i}').replace('L_j^{fit}',r'I_{\mathrm{int},j}^{fit}').replace('B-L_j',r'B-I_{\mathrm{int},j}')
    replacements=[(r'E_b',r'\mathbf{E}_b'),(r'u_{d_z}',r'\mathbf{U}_{d_z}'),(r'u_x',r'\mathbf{U}_x'),(r'u_y',r'\mathbf{U}_y'),(r'u_m',r'\mathbf{U}_m'),(r'v=\delta',r'\mathbf v=\delta'),(r'\|v\|',r'\|\mathbf v\|'),(r'd_{local}',r'\mathbf d_{local}'),(r'd_j',r'\mathbf d_j'),(r'J_j',r'\mathbf J_j'),(r'e_j',r'\mathbf e_j'),(r't_\perp',r'\mathbf t_\perp'),(r't_\parallel',r'\mathbf t_\parallel')]
    # Only change mathematical content; prose and section names stay untouched.
    def math(m):
        s=m.group(0)
        for old,new in replacements:s=s.replace(old,new)
        s=re.sub(r'(?<![A-Za-z\\_])e_([12])(?![A-Za-z])',r'\\mathbf e_\1',s)
        s=re.sub(r'(?<![A-Za-z\\_])([dkcv])(?![A-Za-z_\'])',lambda q:r'\mathbf '+q.group(1),s)
        return s
    tex=re.sub(r'\\\[[\s\S]*?\\\]|\$[^$]*\$',math,tex)
    # Remove redundant bold commands caused by component replacements.
    tex=tex.replace(r'\mathbf \mathbf ',r'\mathbf ')
    tex=tex.replace(r'\widehat S=I^{obs}-B-L(\mathbf d^{fit})=S^{fit}+\varepsilon.',r'\widehat{\mathbf S}=\mathbf I^{obs}-\mathbf B-\mathbf I_{\mathrm{int}}(\mathbf d^{fit})=\mathbf S^{fit}+\boldsymbol\varepsilon.')
    tex=tex.replace('$S$, $B$ and $L$',r'$\mathbf S$, $\mathbf B$ and $\mathbf I_{\mathrm{int}}$')
    tex=tex.replace(r'XY(I^{heldout}_j-B-I_{\mathrm{int},j}^{fit})-XY(S_j^{fit})',r'XY(\mathbf I^{heldout}_j-\mathbf B-\mathbf I_{\mathrm{int},j}^{fit})-XY(\mathbf S_j^{fit})')
    tex=tex.replace(r'\mathbf e_j,\mathbf J_j',r'\mathbf e_j,\mathbf J_j')
    tex=tex.replace(r'\section{Signal and field composition}',r'\section{Signal and field composition}'+KEY)
    tex=tex.replace('This is a draft for revision; no commit or push.','This report and its reproducible scripts and saved results are included in the project wrap-up commit. No new fits were performed for this notation revision.')
    return tex
