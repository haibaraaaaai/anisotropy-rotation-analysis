"""Ordered discrete branch choice; never modify measured radii."""
import numpy as np

def recover(cor, reference, u):
    xy=np.column_stack([(cor[0]-cor[1])/(cor[0]+cor[1]),(cor[2]-cor[3])/(cor[2]+cor[3])])
    r=np.linalg.norm(xy,axis=1)
    s2=.8908059777413628*r/(.9277805723143716-.10919402225863714*r)
    valid=(cor>=0).all(0)&np.isfinite(s2)&(s2>=0)&(s2<=1)
    ids=np.flatnonzero(valid);obs=np.full((len(r),3),np.nan)
    if not len(ids):return obs,valid,dict(valid=0)
    phi=.5*np.arctan2(xy[ids,1],xy[ids,0])
    base=np.column_stack([np.sqrt(s2[ids])*np.cos(phi),np.sqrt(s2[ids])*np.sin(phi),np.sqrt(1-s2[ids])])
    cand=base[:,None,:]*np.array([[1,1,1],[-1,-1,1],[1,1,-1],[-1,-1,-1]])[None,:,:]
    n=len(ids);du=(np.roll(u[ids],-1)-u[ids])%1
    du=np.maximum(du,1e-12)
    step=np.arccos(np.clip(np.einsum('nki,nli->nkl',cand,np.roll(cand,-1,axis=0)),-1,1))**2/du[:,None,None]
    # Exhaust all four starting branches. Primary objective: angular continuity,
    # without imposing closure. Cone proximity only breaks numerical ties.
    unary=np.arccos(np.clip(np.einsum('nki,ni->nk',cand,reference[ids]),-1,1))**2
    paths=[]
    for start in range(4):
        cost=np.full(4,np.inf);cost[start]=0;tie=np.full(4,np.inf);tie[start]=unary[0,start];back=[]
        for j in range(1,n):
            nc=np.empty(4);nt=np.empty(4);prev=np.empty(4,int)
            for k in range(4):
                v=cost+step[j-1,:,k];mn=v.min();pool=np.flatnonzero(np.isclose(v,mn,rtol=1e-11,atol=1e-11));p=pool[np.argmin(tie[pool])]
                nc[k]=v[p];nt[k]=tie[p]+unary[j,k];prev[k]=p
            cost,tie=nc,nt;back.append(prev)
        v=cost;mn=v.min();pool=np.flatnonzero(np.isclose(v,mn,rtol=1e-11,atol=1e-11));end=pool[np.argmin(tie[pool])]
        path=[int(end)]
        for prev in back[::-1]:path.append(int(prev[path[-1]]))
        paths.append((float(v[end]),float(tie[end]),path[::-1]))
    best=min(x[0] for x in paths);ties=[x for x in paths if np.isclose(x[0],best,rtol=1e-11,atol=1e-11)];path=np.array(min(ties,key=lambda x:x[1])[2]);obs[ids]=cand[np.arange(n),path]
    old=cand[np.arange(n),np.argmax(np.einsum('nki,ni->nk',cand,reference[ids]),axis=1)]
    def steps(p):return np.degrees(np.arccos(np.clip(np.sum(p*np.roll(p,-1,axis=0),axis=1),-1,1)))
    st=steps(obs[ids]);oldst=steps(old)
    # Re-forward the chosen branches: anisotropy must be unchanged.
    ss=obs[ids,0]**2+obs[ids,1]**2
    radius=.9277805723143716*ss/(.8908059777413628+.10919402225863714*ss)
    az=np.arctan2(obs[ids,1],obs[ids,0]);xyback=radius[:,None]*np.column_stack([np.cos(2*az),np.sin(2*az)])
    err=float(np.max(abs(xyback-xy[ids])));assert err<1e-12
    endpoint=cand[0,np.argmax(cand[0]@obs[ids[-1]])]
    closure=float(np.degrees(np.arccos(np.clip(endpoint@obs[ids[0]],-1,1))))
    z=xy[:,0]+1j*xy[:,1]
    winding=float(np.angle(np.roll(z,-1)*np.conj(z)).sum()/(2*np.pi)) if valid.all() else None
    return obs,valid,dict(valid=n,max_step_deg=float(st[:-1].max()) if n>1 else 0.0,closure_mismatch_deg=closure,winding=winding,closure_assessed=bool(valid.all()),median_step_deg=float(np.median(st)),seam_step_deg=float(st[-1]),old_max_step_deg=float(oldst.max()),changed_points=int(np.sum(np.linalg.norm(old-obs[ids],axis=1)>1e-8)),inversion_roundtrip_max_error=err,steps_across_invalid_gaps=int(np.sum((np.roll(ids,-1)-ids)%len(r)>1)))
