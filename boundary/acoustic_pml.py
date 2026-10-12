"""Experimental acoustic CFS-PML coupled through equilibrium differences.

Four convolution memories replace each Euler acoustic derivative D_j by
D_j+psi_j, with psi' = -(sigma+alpha)psi - sigma D_j q.  The native D2Q37
step supplies the original derivative; this module supplies only the added
source. It does not claim a discrete perfectly matched *thermal* LBM layer.
Viscous and entropy modes retain the native collision. There is no relaxation
of temperature to a prescribed ambient value and no fitted gain or phase.

All calculations below use the common mapping's LU, including time in steps.
The source is RK4-integrated in a Strang half-step; its exact population mass
and energy changes are returned separately for independent accounting.
"""
from __future__ import annotations

import numpy as np

from core.equilibrium import equilibrium_fg
from core.macroscopic import recover_macro


def derivative(field, axis):
    """Fourth-order centred unit-grid derivative, second-order at edge pairs."""
    v=np.moveaxis(np.asarray(field),axis,-1)
    out=np.empty_like(v)
    out[...,2:-2]=(v[...,:-4]-8*v[...,1:-3]+8*v[...,3:-1]-v[...,4:])/12
    out[...,1]=(v[...,2]-v[...,0])/2
    out[...,-2]=(v[...,-1]-v[...,-3])/2
    out[...,0]=(-3*v[...,0]+4*v[...,1]-v[...,2])/2
    out[...,-1]=(3*v[...,-1]-4*v[...,-2]+v[...,-3])/2
    return np.moveaxis(out,-1,axis)


class AcousticPML:
    """Top and both side acoustic layers, with no bottom absorption."""

    def __init__(self,mapping,lattice,ny,nx,*,cells,target_return=.001,alpha_Hz=1000.,backend='numpy'):
        if type(cells) is not int or not 3<=cells<min(ny,nx//2):
            raise ValueError('PML requires 3 <= cells < min(ny,nx//2)')
        if not np.isfinite(target_return) or not 0<target_return<1:
            raise ValueError('PML target_return must lie in (0,1)')
        if not np.isfinite(alpha_Hz) or alpha_Hz<0:
            raise ValueError('PML alpha_Hz must be finite and nonnegative')
        self.mapping,self.lattice=mapping,lattice
        self.compiled_rhs=self.compiled_increment=None
        if backend=='numpy':
            self.recover,self.equilibrium=recover_macro,equilibrium_fg
        elif backend=='numba':
            from core.collision_numba import (recover_macro_compiled,equilibrium_compiled,
                                              apply_equilibrium_increment)
            from boundary.acoustic_pml_numba import acoustic_rhs
            self.recover,self.equilibrium=recover_macro_compiled,equilibrium_compiled
            self.compiled_rhs,self.compiled_increment=acoustic_rhs,apply_equilibrium_increment
        else:
            raise ValueError('PML backend must be numpy or numba')
        self.rho0=mapping.lattice.rho_ref_lu
        self.p0=self.rho0*mapping.theta_ref_lu
        self.c2=(1+2/(mapping.lattice.D+mapping.lattice.S))*mapping.theta_ref_lu
        # Design attenuation, not a measured or guaranteed returned error.
        sigma_max=-4*np.sqrt(self.c2)*np.log(target_return)/(2*cells)
        x,y=np.arange(nx)+.5,np.arange(ny)+.5
        sx=sigma_max*np.maximum(0,1-np.minimum(x,nx-x)/cells)**3
        sy=sigma_max*np.maximum(0,1-(ny-y)/cells)**3
        self.sigma_x=np.broadcast_to(sx,(ny,nx)).copy()
        self.sigma_y=np.broadcast_to(sy[:,None],(ny,nx)).copy()
        self.active=(self.sigma_x>0)|(self.sigma_y>0)
        alpha=2*np.pi*alpha_Hz*mapping.lattice.dt_s
        self.decay_x=self.sigma_x+alpha*(self.sigma_x>0)
        self.decay_y=self.sigma_y+alpha*(self.sigma_y>0)
        self.memory=np.zeros((4,ny,nx))
        self.kinetic=.5*np.sum(lattice.c*lattice.c,axis=1)

    def _rhs(self,v,rho,c2):
        if self.compiled_rhs is not None:
            return self.compiled_rhs(v,rho,c2,self.sigma_x,self.sigma_y,self.decay_x,self.decay_y)
        p,ux,uy,px,py,vx,vy=v
        out=np.empty_like(v)
        out[0]=-rho*c2*(vx+vy)
        out[1]=-px/rho
        out[2]=-py/rho
        out[3]=-self.decay_x*px-self.sigma_x*derivative(p,1)
        out[4]=-self.decay_y*py-self.sigma_y*derivative(p,0)
        out[5]=-self.decay_x*vx-self.sigma_x*derivative(ux,1)
        out[6]=-self.decay_y*vy-self.sigma_y*derivative(uy,0)
        return out

    def advance(self,f,g,memory,duration_lu):
        """Return corrected copies and trial memories; never commit input state."""
        if duration_lu<=0 or not np.isfinite(duration_lu):
            raise ValueError('positive finite PML duration required')
        m=self.mapping
        state=self.recover(f,g,D=m.lattice.D,S=m.lattice.S,lattice=self.lattice)
        # Linearize the acoustic source at the current local thermodynamic
        # state, held during this half-step. A fixed 300 K impedance would be
        # inconsistent with the native Euler part after mean thermal drift.
        # No fitted or time-filtered background is used. This remains a small-
        # amplitude acoustic source, not a nonlinear viscothermal PML proof.
        local_c2=state.gamma*state.theta
        v=np.concatenate((np.stack((state.p-self.p0,state.u[...,0],state.u[...,1])),memory))
        h=duration_lu
        a=self._rhs(v,state.rho,local_c2); b=self._rhs(v+.5*h*a,state.rho,local_c2)
        c=self._rhs(v+.5*h*b,state.rho,local_c2); d=self._rhs(v+h*c,state.rho,local_c2)
        new=v+h*(a+2*b+2*c+d)/6
        active=self.active
        dp=new[0]-v[0]
        rho=state.rho+dp/local_c2
        u=np.moveaxis(new[1:3],0,-1)
        theta=(state.p+dp)/rho
        if self.compiled_increment is not None:
            fn,gn,change=self.compiled_increment(f,g,state.rho,state.u,state.theta,
                rho,u,theta,active,self.lattice,m.lattice.S)
        else:
            before=self.equilibrium(state.rho[active],state.u[active],state.theta[active],m.lattice.S,self.lattice)
            after=self.equilibrium(rho[active],u[active],theta[active],m.lattice.S,self.lattice)
            df,dg=(a-b for a,b in zip(after,before))
            fn,gn=f.copy(),g.copy()
            fn[active]+=df; gn[active]+=dg
            # Sum actual population deltas, including roundoff in their write.
            actual_f,actual_g=fn[active]-f[active],gn[active]-g[active]
            change=np.array([np.sum(actual_f),np.sum(actual_f*self.kinetic+actual_g)])
        if not np.isfinite(new).all():
            raise FloatingPointError('nonfinite acoustic PML memory; state not committed')
        return fn,gn,new[3:].copy(),change
