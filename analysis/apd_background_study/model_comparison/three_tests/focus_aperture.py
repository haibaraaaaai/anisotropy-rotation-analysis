import run as r
import numpy as np,json
from scipy.fft import fft2,fftshift,ifftshift
from models import ABC
R=r.R;radius=110;side=2*radius;pad=660;na=1.3;wave=.633;n=1.33
v=(np.arange(side)+.5-radius)*na/radius;xx,yy=np.meshgrid(v,v);rho=np.hypot(xx,yy);valid=(rho<na)&(rho>=.38)
x,y,get,area=r.prev.pm.basis(radius=radius);disk=np.hypot(xx,yy)<na;ff=get();basis=np.zeros((2,side,side,3));basis[:,disk,:]=ff[:2];basis[:,~valid,:]=0
coord=(np.arange(pad)-pad//2)*wave/(pad*(na/radius));ix,iy=np.meshgrid(coord,coord);rad=np.hypot(ix,iy);apertures=[.3,.6,1.2,3.0]
masks=[rad<a for a in apertures];masks.append(np.ones_like(rad,bool));zs=np.linspace(-.3,.3,13);records=[];infinite_error=0
for theta in [20,45,75]:
 for phi in [0,22.5,45]:
  t,p=np.deg2rad([theta,phi]);d=np.array([np.sin(t)*np.cos(p),np.sin(t)*np.sin(p),np.cos(t)]);field=np.einsum('ipqj,j->ipq',basis,d)
  outputs=[]
  for z in zs:
   phase=np.exp(1j*2*np.pi/wave*z*np.sqrt(np.maximum(n*n-rho*rho,0)));a=field*phase[None,:,:];b=np.pad(a,((0,0),((pad-side)//2,)*2,((pad-side)//2,)*2));img=fftshift(fft2(ifftshift(b,axes=(-2,-1)),axes=(-2,-1),norm='ortho'),axes=(-2,-1))
   channels=np.einsum('ji,ipq->jpq',r.ANALYZERS,img);power=abs(channels)**2
   intens=np.array([power[:,mask].sum(1) for mask in masks]);outputs.append(intens)
  outputs=np.array(outputs);ref=outputs[6];infinite_error=max(infinite_error,float(np.max(abs(outputs[:,-1]/ref[-1]-1))))
  for j,ap in enumerate(apertures+[None]):
   vals=outputs[:,j];xy=r.xy(vals.T);refxy=r.xy(ref[j,:,None])[0];an=np.linalg.norm(xy,axis=1);s2=ABC[0]*an/(ABC[2]-ABC[1]*an);angles=np.where((s2>=0)&(s2<=1),np.rad2deg(np.arcsin(np.sqrt(np.clip(s2,0,1)))),np.nan)
   records.append({'theta':theta,'phi':phi,'aperture_radius_um':ap,'fraction_collected_at_focus':float(ref[j].sum()/ref[-1].sum()),'max_abs_total_change_fraction':float(np.max(abs(vals.sum(1)/ref[j].sum()-1))),'max_xy_change':float(np.max(np.linalg.norm(xy-refxy,axis=1))),'theta_change_deg':float(np.nanmax(abs(angles-angles[6]))),'invalid_angle_points':int(np.isnan(angles).sum()),'z_um':zs.tolist(),'intensities':vals.tolist()})
summary={}
for ap in apertures+[None]:
 group=[v for v in records if v['aperture_radius_um']==ap];summary[str(ap)]={k:max(v[k] for v in group) for k in ['max_abs_total_change_fraction','max_xy_change','theta_change_deg']};summary[str(ap)]['min_fraction_collected_at_focus']=min(v['fraction_collected_at_focus'] for v in group)
result={'assumptions':'Pure phase defocus, centred circular object-space acceptance apertures equal in four channels, fixed orientations, no BG, no excitation modulation. Not an APD calibration. Aperture sizes and ±300nm are sensitivity scenarios, not measured bounds.','infinite_collection_error':infinite_error,'summary':summary,'records':records}
(R/'focus_aperture.json').write_text(json.dumps(result,indent=2));print(json.dumps({'infinite_collection_error':infinite_error,'summary':summary},indent=2))
