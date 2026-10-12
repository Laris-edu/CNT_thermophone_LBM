"""Closed rectangle obtained from the three sides of a symmetric upper strip."""
import numpy as np


def upper_rectangle(half_width_m,height_m,spacing_m):
    if not all(np.isfinite(v) and v>0 for v in (half_width_m,height_m,spacing_m)):
        raise ValueError('positive finite rectangle dimensions required')
    nx=int(np.ceil(2*half_width_m/spacing_m)); ny=int(np.ceil(height_m/spacing_m))
    dx=2*half_width_m/nx; dy=height_m/ny
    x=-half_width_m+(np.arange(nx)+.5)*dx
    y=(np.arange(ny)+.5)*dy
    return dict(x_m=np.r_[x,np.full(ny,-half_width_m),np.full(ny,half_width_m)],
        y_m=np.r_[np.full(nx,height_m),y,y],
        normal_x=np.r_[np.zeros(nx),-np.ones(ny),np.ones(ny)],
        normal_y=np.r_[np.ones(nx),np.zeros(2*ny)],ds_m=np.r_[np.full(nx,dx),np.full(2*ny,dy)])


def mirror_strip_surface(upper,p_hat_Pa,ux_hat_m_s,uy_hat_m_s):
    """p/ux even and uy odd; no extra factor two on either phasor channel."""
    p,ux,uy=(np.asarray(v,dtype=complex) for v in (p_hat_Pa,ux_hat_m_s,uy_hat_m_s))
    shape=np.asarray(upper['x_m']).shape
    if any(v.shape!=shape for v in (p,ux,uy)):
        raise ValueError('surface and phasor arrays must share shape')
    if not all(np.all(np.isfinite(v)) for v in (p,ux,uy)):
        raise ValueError('nonfinite surface phasors')
    result={k:np.r_[v,-v if k in ('y_m','normal_y') else v] for k,v in upper.items()}
    result['p_hat_Pa']=np.r_[p,p]
    ux_full,uy_full=np.r_[ux,ux],np.r_[uy,-uy]
    result['un_hat_m_s']=ux_full*result['normal_x']+uy_full*result['normal_y']
    return result
