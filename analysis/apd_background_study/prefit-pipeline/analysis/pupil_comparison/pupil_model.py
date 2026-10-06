"""Integrated dipole response on a Cartesian NA/BFP grid.

Order: 0, 90, 45, 135. A real unit dipole d gives I_i=d.T Q_i d.
Water-side propagating far field, flat water/oil interface, ideal aplanatic
objective and linear analysers. No near-field/interface-induced dipole changes,
supercritical collection, spatial detector clipping or background are included.
The excitation amplitude cancels in instantaneous anisotropies.
"""
import json
import numpy as np
from PIL import Image

def fourkas_abc(na_in=0.38, na_out=1.3, n=1.33):
    def f(na):
        c=np.sqrt(1-(na/n)**2)
        return np.array([1/6-c/4+c**3/12,c/8-c**3/8,7/48-c/16-c*c/16-c**3/48])
    return f(na_out)-f(na_in)

def analytic_q(abc):
    A,B,C=abc
    q=np.zeros((4,3,3))
    q[0]=np.diag([A+B+C,A+B-C,A])
    q[1]=np.diag([A+B-C,A+B+C,A])
    q[2]=np.diag([A+B,A+B,A]);q[2,0,1]=q[2,1,0]=C
    q[3]=np.diag([A+B,A+B,A]);q[3,0,1]=q[3,1,0]=-C
    return q

def basis(radius=300,nw=1.33,no=1.51,na_out=1.3):
    # Pixel centres; integrate only valid collected propagating rays.
    v=(np.arange(2*radius)+0.5-radius)*na_out/radius
    x,y=np.meshgrid(v,v,indexing='xy')
    keep=x*x+y*y<na_out**2;x=x[keep];y=y[keep]
    rho=np.hypot(x,y);c=x/rho;s=y/rho
    sw=rho/nw;cw=np.sqrt(1-sw**2);co=np.sqrt(1-(rho/no)**2)
    er=np.stack([cw*c,cw*s,-sw],axis=1)
    es=np.stack([-s,c,np.zeros_like(s)],axis=1)
    tp=2*nw*cw/(nw*co+no*cw)
    ts=2*nw*cw/(nw*cw+no*co)
    # dP_t ∝ |t E_w|² no cos(o)/(nw cos(w)) dΩ_w.
    # dA_BFP ∝ nw² cos(w) dΩ_w, so amplitude ∝ sqrt(cos(o))/cos(w).
    def fields(fresnel=True,legacy=False):
        p=er*(tp[:,None] if fresnel else 1)
        z=es*(ts[:,None] if fresnel else 1)
        ex=c[:,None]*p-s[:,None]*z
        ey=s[:,None]*p+c[:,None]*z
        w=1/co if legacy else np.sqrt(co)/cw
        return np.stack([ex,ey,(ex+ey)/np.sqrt(2),(ex-ey)/np.sqrt(2)])*w[None,:,None]
    return x,y,fields,(na_out/radius)**2

def load_mask(path):
    im=Image.open(path);meta=json.loads(im.tag_v2[270])
    return np.asarray(im),meta

def mask_blocked(x,y,path,rotation_deg=0):
    m,meta=load_mask(path);angle=np.deg2rad(rotation_deg)
    u=np.cos(angle)*x+np.sin(angle)*y
    v=-np.sin(angle)*x+np.cos(angle)*y
    scale=meta['outer_radius_mm']/1.3/meta['mask_scale_mm_per_px']
    # TIFF explicitly stores axes as x,y, rather than ordinary image row=y.
    ix=np.rint(meta['mask_center_xy'][0]+u*scale).astype(int)
    iy=np.rint(meta['mask_center_xy'][1]+v*scale).astype(int)
    valid=(ix>=0)&(iy>=0)&(ix<m.shape[0])&(iy<m.shape[1])
    b=np.zeros(len(x),dtype=bool);b[valid]=m[ix[valid],iy[valid]]==meta['blocked_value']
    return b

def response(fields,keep,area):
    f=fields[:,keep,:]
    return np.einsum('ipj,ipk->ijk',f,f)*area

def directions(theta,phi):
    t,p=np.deg2rad(theta),np.deg2rad(phi)
    return np.stack([np.sin(t)*np.cos(p),np.sin(t)*np.sin(p),np.cos(t)],axis=-1)

def intensities(q,d):
    return np.einsum('...j,ijk,...k->...i',d,q,d)

def xy(q,d):
    i=intensities(q,d)
    return np.stack([(i[...,0]-i[...,1])/(i[...,0]+i[...,1]),
                     (i[...,2]-i[...,3])/(i[...,2]+i[...,3])],axis=-1)

def abc_from_q(q):
    A=q[0,2,2];B=(q[0,0,0]+q[0,1,1])/2-A;C=(q[0,0,0]-q[0,1,1])/2
    return np.array([A,B,C])

def invert_radial(points,abc):
    A,B,C=abc;r=np.linalg.norm(points,axis=-1);u=A*r/(C-B*r)
    theta=np.full_like(u,np.nan);valid=(u>=0)&(u<=1)
    theta[valid]=np.rad2deg(np.arcsin(np.sqrt(u[valid])))
    phi=np.rad2deg(np.arctan2(points[...,1],points[...,0]))/2
    return theta,phi,u

def cone(theta_axis,phi_axis,lam,n=720):
    k=directions(theta_axis,phi_axis)
    e1=np.array([0.,0.,1.])-k[2]*k
    if np.linalg.norm(e1)<1e-9:e1=np.array([1.,0.,0.])
    else:e1/=np.linalg.norm(e1)
    e2=np.cross(k,e1);chi=np.arange(n)*2*np.pi/n
    l=np.deg2rad(lam)
    return np.cos(l)*k+np.sin(l)*(np.cos(chi)[:,None]*e1+np.sin(chi)[:,None]*e2)
