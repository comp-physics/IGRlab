#!/usr/bin/env python3
"""Bounded CPU hp/alpha and sample-estimated front-coordinate research screen.

Uses the existing full-energy Mach-2 stationary IGR profile. No PDE advancement,
GPU timing, or changed production model. Profile coordinates are fitted from
noisy nodal observations or one conserved density moment, never supplied the
true front location. The known end states and template shape are explicit prior
information; a distorted-profile case tests this modeling assumption.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import scipy
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar, least_squares
from scipy.special import roots_legendre
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/perf"))
from igr_shock_representation_probe import Profile, integration_rule, legendre_matrix, fluxes


class ExtendedProfile(Profile):
    """Same integrating factor, extended to 1e-12 of each far-field density.

    Beyond that bound use the limiting constant state. This changes density by
    at most 1e-12 times the branch jump; the cutoff is reported and refined.
    """
    def __init__(self, panels=2048, quadrature=32, tail_exponent=28.0):
        self.panels, self.quadrature = panels, quadrature
        self.g, self.w = roots_legendre(quadrature)
        self.branches, self.primitives = {}, {}
        pg, pw = roots_legendre(12)
        v0 = np.linspace(0, .9**(1/6), panels+1)
        vt = (-np.expm1(-np.linspace(-math.log(.1), tail_exponent, panels+1)))**(1/6)
        v = np.unique(np.r_[v0, vt[1:]])
        for sign, endpoint in ((-1, self.rho_lo), (1, self.rho_hi)):
            delta = abs(endpoint-self.rho_s)
            increments=[]
            for start in range(0,len(v)-1,128):
                end=min(start+128,len(v)-1)
                left,right=v[start:end],v[start+1:end+1]
                vp=(left[:,None]+right[:,None])/2+(right-left)[:,None]*pg/2
                dx=6*delta*vp**5/np.sqrt(self.slope_squared(delta*vp**6,sign))
                increments.extend((dx@pw)*(right-left)/2)
            xx=np.r_[0,np.cumsum(increments)]
            tt=xx**self.beta
            spline=CubicSpline(tt,delta*v**6,extrapolate=False)
            self.branches[sign]=(spline,float(xx[-1]))
            left,right=tt[:-1],tt[1:]
            tq=(left[:,None]+right[:,None])/2+(right-left)[:,None]*pg/2
            inc=(spline(tq)*tq**(1/self.beta-1)/self.beta@pw)*(right-left)/2
            self.primitives[sign]=(tt,np.r_[0,np.cumsum(inc)])
        self.pg,self.pw=pg,pw
        self.tail_exponent=tail_exponent

    def density_distance(self,x):
        x=np.asarray(x,dtype=float)
        result=np.zeros_like(x)
        for sign,endpoint in ((-1,self.rho_lo),(1,self.rho_hi)):
            mask=x*sign>0
            spline,xmax=self.branches[sign]
            if np.any(mask):
                y=np.abs(x[mask]); result[mask]=spline(np.minimum(y,xmax)**self.beta)
                result[np.logical_and(mask,np.abs(x)>xmax)]=abs(endpoint-self.rho_s)
        return result

    def density_derivative(self,x):
        x=np.asarray(x,dtype=float)
        if np.any(x==0): raise ValueError("Cusp derivative must be integrated weakly")
        result=np.zeros_like(x)
        distance=self.density_distance(x)
        for sign in (-1,1):
            mask=(x*sign>0)&(abs(x)<self.branches[sign][1])
            result[mask]=np.sqrt(self.slope_squared(distance[mask],sign))
        return result

    def primitive(self,x):
        """Integral rho from 0 to scalar x, using the profile's own spline."""
        if x==0:return 0.
        sign=1 if x>0 else -1
        spline,xmax=self.branches[sign]; grid,acc=self.primitives[sign]
        z=min(abs(x),xmax); t=z**self.beta
        j=min(len(grid)-2,max(0,int(np.searchsorted(grid,t,side="right")-1)))
        left=grid[j]; tq=(left+t)/2+(t-left)*self.pg/2
        extra=float((spline(tq)*tq**(1/self.beta-1)/self.beta)@self.pw)*(t-left)/2
        # Integral from 0 to x: sign*rho_s*|x| + integral_0^|x| d.
        value=sign*self.rho_s*z+acc[j]+extra
        endpoint=self.rho_hi if sign>0 else self.rho_lo
        return value+sign*endpoint*(abs(x)-z)


def fields_from_rho(rho,profile):
    energy=profile.e0*rho-profile.momentum_flux+profile.mass_flux**2/rho
    return np.stack((rho,np.full_like(rho,profile.mass_flux),energy,profile.sigma(rho)),axis=-1)


def target(profile,x,front,ell,distortion=0.,derivative=False):
    y=(np.asarray(x)-front)/ell
    rho=profile.fields(y)[:,0]
    perturb=distortion*(profile.rho_hi-profile.rho_lo)*np.exp(-y*y/2)*np.sin(2*y)
    rho=rho+perturb
    if not derivative:return fields_from_rho(rho,profile)
    dr=profile.density_derivative(y)+distortion*(profile.rho_hi-profile.rho_lo)*np.exp(-y*y/2)*(2*np.cos(2*y)-y*np.sin(2*y))
    dr=dr/ell
    return np.stack((dr,np.zeros_like(dr),(profile.e0-profile.mass_flux**2/rho**2)*dr,profile.sigma_r(rho)*dr),axis=-1)


def lgl(p):
    if p==1:return np.array([-1.,1.])
    return np.r_[-1.,np.polynomial.legendre.Legendre.basis(p).deriv().roots(),1.]


def quadrature(front,ell,order=16,extra=()):
    breaks=list(np.arange(-8.,10.))
    breaks+=list(front+ell*np.array([-28,-16,-8,-4,-2,-1,-.25,0,.25,1,2,4,8,16,28]))
    breaks=sorted(set(float(x) for x in breaks if -8<=x<=9))
    for point in extra:
        if -8<point<9 and min(abs(point-b) for b in breaks)>64*np.finfo(float).eps*max(1.,abs(point)):
            breaks.append(float(point))
    x,w=integration_rule(sorted(breaks),order)
    # An almost coincident fitted cusp can round onto a quadrature node. Move
    # only that zero-measure coincidence by one ULP; never evaluate its infinite
    # point derivative. Order refinement checks the surrounding integral.
    singularities=(front,)+tuple(extra)
    for _ in range(len(singularities)+1):
        for point in singularities:
            x[x==point]=np.nextafter(point,math.inf)
    return x,w


class Polynomial:
    def __init__(self,p,x,w,values,mode,profile,front,ell):
        self.p=p;self.coefficients=[]
        for cell in range(-8,9):
            if mode=="l2":
                mask=(x>=cell)&(x<cell+1)
                basis=legendre_matrix(2*(x[mask]-cell)-1,p+1)
                self.coefficients.append(((basis.T*w[mask])@values[mask])*(2*np.arange(p+1)+1)[:,None])
            else:
                z=lgl(p); nodes=cell+(z+1)/2
                self.coefficients.append(np.linalg.solve(legendre_matrix(z,p+1),target(profile,nodes,front,ell)))
    def evaluate(self,x,derivative=False):
        x=np.asarray(x);result=np.zeros((len(x),4))
        for index,cell in enumerate(range(-8,9)):
            mask=(x>=cell)&((x<cell+1) if cell<8 else (x<=cell+1))
            result[mask]=legendre_matrix(2*(x[mask]-cell)-1,self.p+1,derivative)@self.coefficients[index]*(2 if derivative else 1)
        return result


class H1TraceReconstruction(Polynomial):
    """Conservative H1-weighted recovery from available polynomial moments.

    (M+alpha*K)(c_new-c) = alpha*(phi'_R*delta_trace_R
                                      - phi'_L*delta_trace_L).
    Traces are averaged from the two available neighboring polynomials. No
    profile value, derivative, or front location enters this reconstruction.
    Matching traces imply exact identity, an intentional consistency test.
    """
    def __init__(self,base,alpha):
        self.p=base.p
        n=np.arange(self.p+1)
        g,w=roots_legendre(self.p+2)
        derivative=legendre_matrix(g,self.p+1,True)
        matrix=np.diag(1/(2*n+1))+2*alpha*(derivative.T*w)@derivative
        derivative_right=n*(n+1)
        derivative_left=(-1.)**(n+1)*n*(n+1)
        # These two columns can be constructed once for all equal h,p,alpha.
        response=np.linalg.solve(matrix,alpha*np.stack((derivative_right,-derivative_left),axis=1))
        coefficients=np.asarray(base.coefficients)
        left=np.einsum('n,cnf->cf',(-1.)**n,coefficients)
        right=np.sum(coefficients,axis=1)
        shared=(right[:-1]+left[1:])/2
        delta_right=np.zeros_like(right);delta_left=np.zeros_like(left)
        delta_right[:-1]=shared-right[:-1];delta_left[1:]=shared-left[1:]
        correction=np.einsum('nk,ckf->cnf',response,np.stack((delta_right,delta_left),axis=1))
        self.coefficients=coefficients+correction
        self.maximum_cell_mean_change=float(np.max(abs(self.coefficients[:,0]-coefficients[:,0])))
        self.maximum_coefficient_change=float(np.max(abs(correction)))
        self.matrix_condition=float(np.linalg.cond(matrix))


def metrics(x,w,truth,truth_derivative,fit,fit_derivative,alpha,ell,front,profile):
    source=lambda f,d:2*alpha*((d[:,1]*f[:,0]-f[:,1]*d[:,0])/f[:,0]**2)**2
    reference_source=source(truth,truth_derivative); fit_source=source(fit,fit_derivative)
    z=(x-front)/ell
    abs_mom=np.array([w@(reference_source*abs(z)**j) for j in range(3)])
    reference_mom=np.array([w@(reference_source*z**j) for j in range(3)])
    moments=np.array([w@(fit_source*z**j) for j in range(3)])
    pressure=(profile.gamma-1)*(fit[:,2]-fit[:,1]**2/(2*fit[:,0]))
    flux,_=fluxes(fit,profile.gamma)
    expected_flux=np.array([profile.mass_flux,profile.momentum_flux,profile.energy_flux])
    finite=bool(np.isfinite(fit).all() and np.isfinite(fit_derivative).all() and np.isfinite(moments).all())
    jump=profile.rho_hi-profile.rho_lo
    return {"density_l2_over_jump_sqrt_ell":float(np.sqrt(w@((fit[:,0]-truth[:,0])**2))/(jump*np.sqrt(ell))),
            "sigma_relative_l2":float(np.sqrt(w@((fit[:,3]-truth[:,3])**2)/(w@(truth[:,3]**2)))),
            "source_moment_error_relative_absolute_reference":((moments-reference_mom)/abs_mom).tolist(),
            "max_source_moment_relative_error":float(np.max(abs((moments-reference_mom)/abs_mom))),
            "source_moments":moments.tolist(),"reference_source_moments":reference_mom.tolist(),
            "minimum_density_on_quadrature":float(np.min(fit[:,0])),"minimum_pressure_on_quadrature":float(np.min(pressure)),
            "rms_mass_momentum_energy_flux_defect_over_exact":(np.sqrt(w@((flux-expected_flux)**2)/np.sum(w))/np.abs(expected_flux)).tolist(),
            "finite_and_positive_on_quadrature":finite and bool(np.min(fit[:,0])>0 and np.min(pressure)>0)}


def mass(profile,front,ell,a=-8.,b=9.):
    return ell*(profile.primitive((b-front)/ell)-profile.primitive((a-front)/ell))


def fit_nodal(profile,nodes,observations,ell):
    calls=0
    def objective(front):
        nonlocal calls
        calls+=1
        return float(np.sum((profile.fields((nodes-front)/ell)[:,0]-observations)**2))
    grid=np.linspace(-.25,1.25,257)
    scores=np.array([objective(s) for s in grid]);j=int(np.argmin(scores))
    result=minimize_scalar(objective,bounds=(grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]),method="bounded",options={"xatol":2e-13})
    return float(result.x),calls,float(result.fun)


def fit_mass(profile,observed_mass,ell):
    calls=0
    def residual(front):
        nonlocal calls
        calls+=1
        return mass(profile,front,ell,a=-1.,b=2.)-observed_mass
    answer=brentq(residual,-.25,1.25,xtol=2e-13,rtol=2e-14)
    return float(answer),calls


def fit_stretch(profile,nodes,observations,ell):
    """Fit location plus positive width, retaining fixed physical alpha.

    This is a richer representation of the given state, not permission to
    replace alpha in its closure. Source metrics still use the original alpha.
    Saturated samples can make both parameters unidentifiable.
    """
    front,calls,_=fit_nodal(profile,nodes,observations,ell)
    count=0
    def residual(parameters):
        nonlocal count
        count+=1
        width=ell*math.exp(parameters[1])
        return (profile.fields((nodes-parameters[0])/width)[:,0]-observations)/(profile.rho_hi-profile.rho_lo)
    fit=least_squares(residual,[front,0.],bounds=([-.25,math.log(.5)],[1.25,math.log(2.)]),
                      xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=100)
    singular=np.linalg.svd(fit.jac,compute_uv=False)
    # Columns have different units: inspect d/d(front/ell), d/d(log width).
    normalized_jac=fit.jac@np.diag([ell,1.])
    normalized_singular=np.linalg.svd(normalized_jac,compute_uv=False)
    return float(fit.x[0]),ell*math.exp(fit.x[1]),calls+count,{
        "fit_converged":bool(fit.success),"normalized_parameter_jacobian_singular_values":normalized_singular.tolist(),
        "raw_parameter_jacobian_singular_values":singular.tolist(),
        "normalized_nodal_residual_l2":float(np.linalg.norm(fit.fun))}


def self_test():
    p=ExtendedProfile(512,32)
    x,w=quadrature(.37,.125,24)
    numerical=float(w@target(p,x,.37,.125)[:,0]);analytical=mass(p,.37,.125)
    assert abs(numerical-analytical)<2e-9,(numerical,analytical)
    for porder in (1,3,7):
        nodes=np.unique(np.concatenate([i+(lgl(porder)+1)/2 for i in range(-8,9)]))
        for front,ell in ((.0,.25),(.37,.125),(.95,.02)):
            rho=target(p,nodes,front,ell)[:,0]
            fitted,_,_=fit_nodal(p,nodes,rho,ell)
            # No guarantee from nodal data if every observation is in a tail.
            if np.any((rho-p.rho_lo>.05)&(p.rho_hi-rho>.05)):
                assert abs(fitted-front)/ell<2e-5,(porder,front,ell,fitted)
            fitted,_=fit_mass(p,mass(p,front,ell,a=-1.,b=2.),ell)
            assert abs(fitted-front)<1e-10
    x,w=quadrature(.37,.125,24)
    truth=target(p,x,.37,.125)
    nodal=Polynomial(7,x,w,truth,"lgl",p,.37,.125)
    lifted=H1TraceReconstruction(nodal,.125**2)
    assert lifted.maximum_coefficient_change<1e-12
    projected=Polynomial(7,x,w,truth,"l2",p,.37,.125)
    lifted=H1TraceReconstruction(projected,.125**2)
    assert lifted.maximum_cell_mean_change<1e-14
    assert lifted.maximum_coefficient_change>1e-6
    print(json.dumps({"self_test":"pass","mass_quadrature_change":abs(numerical-analytical)}))


def run(args):
    started=time.perf_counter()
    profile=ExtendedProfile(args.panels,args.profile_quadrature)
    coarse=ExtendedProfile(args.panels//2,args.profile_quadrature//2)
    test_x=np.r_[-np.geomspace(1e-9,28,400),np.geomspace(1e-9,28,400)]
    reference_change=float(np.max(abs(profile.fields(test_x)-coarse.fields(test_x))))
    check_x,check_w=quadrature(0.,1.,max(32,args.quadrature))
    check_fields=target(profile,check_x,0.,1.)
    check_derivative=target(profile,check_x,0.,1.,derivative=True)
    check_source=2*(profile.mass_flux*check_derivative[:,0]/check_fields[:,0]**2)**2
    boundary=np.array([-8.,9.]);bf=target(profile,boundary,0.,1.)
    bd=target(profile,boundary,0.,1.,derivative=True)
    weak=[]
    for j in range(3):
        value=check_w@((check_fields[:,3]/check_fields[:,0]-check_source)*check_x**j)
        if j:value+=check_w@(check_derivative[:,3]/check_fields[:,0]*j*check_x**(j-1))
        value-=bd[1,3]/bf[1,0]*boundary[1]**j-bd[0,3]/bf[0,0]*boundary[0]**j
        weak.append(float(value)/(check_w@(check_source*abs(check_x)**j)))
    primitive_error=abs(float(check_w@check_fields[:,0])-mass(profile,0.,1.))
    records=[]
    for p in args.orders:
        nodes=np.unique(np.concatenate([i+(lgl(p)+1)/2 for i in range(-1,2)]))
        for s in args.exponents:
            ell=math.sqrt(args.c_alpha)/(p+1)**(s/2)
            for front in args.fronts:
                x,w=quadrature(front,ell,args.quadrature)
                truth=target(profile,x,front,ell)
                derivative=target(profile,x,front,ell,derivative=True)
                modes=("l2","lgl","l2_h1_shared_trace","lgl_h1_shared_trace") if args.h1_controls else ("l2","lgl")
                for mode in modes:
                    begin=time.perf_counter(); poly=Polynomial(p,x,w,truth,mode.split("_")[0],profile,front,ell)
                    reconstruction={}
                    if "h1" in mode:
                        poly=H1TraceReconstruction(poly,ell*ell)
                        reconstruction={"maximum_cell_mean_change":poly.maximum_cell_mean_change,
                            "maximum_coefficient_change":poly.maximum_coefficient_change,
                            "local_matrix_condition":poly.matrix_condition,
                            "inputs":"Existing polynomial moments and average of adjacent polynomial face values; no oracle profile or derivative"}
                    result=metrics(x,w,truth,derivative,poly.evaluate(x),poly.evaluate(x,True),ell*ell,ell,front,profile)
                    fine_x,fine_w=quadrature(front,ell,2*args.quadrature)
                    fine=metrics(fine_x,fine_w,target(profile,fine_x,front,ell),target(profile,fine_x,front,ell,derivative=True),poly.evaluate(fine_x),poly.evaluate(fine_x,True),ell*ell,ell,front,profile)
                    records.append({"kind":"polynomial_"+mode,"p":p,"exponent":s,"alpha":ell*ell,"ell_over_h":ell,"front_over_h":front,
                        "coefficients_per_field":17*(p+1),"cell_count":17,"h":1,"c_alpha":args.c_alpha,
                        "reconstruction":reconstruction,
                        "setup_and_measure_cpu_seconds":time.perf_counter()-begin,"result":fine,
                        "source_quadrature_refinement_change":float(np.max(abs(np.array(result["source_moment_error_relative_absolute_reference"])-fine["source_moment_error_relative_absolute_reference"]))),
                        "within_cell_mean_resolution_indicator":(p+1)*ell,
                        "lgl_endpoint_resolution_indicator":(p+1)**2*ell})
                for distortion in args.distortions:
                    observations=target(profile,nodes,front,ell,distortion)[:,0]
                    true_mass=mass(profile,front,ell,a=-1.,b=2.)
                    if distortion:
                        local=(x>=-1.)&(x<=2.)
                        true_mass=float(w[local]@target(profile,x[local],front,ell,distortion)[:,0])
                    for noise in args.noise:
                        # Same deterministic bounded perturbation; no averaging
                        # across favorable random seeds. Mass error has units
                        # delta_rho*h and is explicitly charged separately.
                        noisy=observations+noise*(profile.rho_hi-profile.rho_lo)*np.sin(1.7*np.arange(len(nodes))+0.4)
                        for fitting in ("nodal","conserved_mass","nodal_stretch"):
                            begin=time.perf_counter()
                            fit_ell=ell;diagnostics={}
                            if fitting=="nodal": fitted,calls,objective=fit_nodal(profile,nodes,noisy,ell)
                            elif fitting=="conserved_mass":
                                fitted,calls=fit_mass(profile,true_mass+noise*(profile.rho_hi-profile.rho_lo),ell)
                                objective=0.
                            else:
                                fitted,fit_ell,calls,diagnostics=fit_stretch(profile,nodes,noisy,ell)
                                objective=0.
                            fit_x,fit_w=quadrature(front,ell,args.quadrature,extra=(fitted,))
                            result=metrics(fit_x,fit_w,target(profile,fit_x,front,ell,distortion),target(profile,fit_x,front,ell,distortion,True),target(profile,fit_x,fitted,fit_ell),target(profile,fit_x,fitted,fit_ell,derivative=True),ell*ell,ell,front,profile)
                            records.append({"kind":"profile_"+fitting,"p":p,"exponent":s,"alpha":ell*ell,"ell_over_h":ell,"front_over_h":front,
                                "distortion_fraction_of_density_jump":distortion,"input_error_fraction_of_jump_or_jump_h":noise,
                                "estimated_front":fitted,"front_error_over_ell":(fitted-front)/ell,"fitted_parameters":2 if fitting=="nodal_stretch" else 1,
                                "fitted_width_over_true_ell":fit_ell/ell,"fit_diagnostics":diagnostics,
                                "known_parameters":["Mach-2 end states","full-energy stationary template","alpha"],
                                "observed_nodal_count":len(nodes) if fitting.startswith("nodal") else 0,
                                "observed_density_moments":1 if fitting=="conserved_mass" else 0,
                                "density_moment_patch":[-1.,2.] if fitting=="conserved_mass" else None,
                                "fit_function_evaluations":calls,"nodal_objective":objective,
                                "setup_and_measure_cpu_seconds":time.perf_counter()-begin,"result":result})
    path=Path(__file__).resolve(); helper=ROOT/"scripts/perf/igr_shock_representation_probe.py"
    return {"schema":"igr-hp-front-v1","source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "script_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"profile_helper_sha256":hashlib.sha256(helper.read_bytes()).hexdigest(),
        "numpy":np.__version__,"scipy":scipy.__version__,"configuration":vars(args),
        "scope":"Approximation, source-moment and parameter-identifiability probes only. No DG evolution, solver residual, GPU time, dynamic stability, front tracking or arbitrary-flow closure claim.",
        "reference_max_field_refinement_change":reference_change,"reference_density_tail_cutoff_relative_branch_jump":math.exp(-profile.tail_exponent),
        "reference_weak_closure_moment_residuals_relative_absolute_source":weak,
        "reference_density_primitive_vs_independent_quadrature_error":primitive_error,
        "reference_checks_pass":bool(reference_change<1e-8 and max(abs(v) for v in weak)<1e-7 and primitive_error<1e-8),
        "alpha_interpretation":"s=0 fixes physical alpha=C*h² while p changes; s=2,4 change the regularization family. All comparisons use their corresponding high-resolution profile. No changed production continuum model is adopted.",
        "accounting":"Nodal and conserved-mass fits have different observations and are not equivalent algorithms: the latter requires an evolved conservative mean, not a quadrature estimate from underresolved point samples. End states, shape and alpha are known prior information. The fit cannot identify arbitrary shock shapes from one number. Distorted data are prescribed positive full-energy states, not exact stationary IGR solutions. Polynomial source uses broken velocity derivatives; jumps are not squared as distributions.",
        "records":records,"cpu_seconds":time.perf_counter()-started}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--h1-controls",action="store_true")
    parser.add_argument("--orders",nargs="+",type=int,default=[1,3,7,15])
    parser.add_argument("--exponents",nargs="+",type=int,default=[0,2,4])
    parser.add_argument("--fronts",nargs="+",type=float,default=[0.,.37,.5,.95])
    parser.add_argument("--noise",nargs="+",type=float,default=[0.,1e-4,1e-3])
    parser.add_argument("--distortions",nargs="+",type=float,default=[0.,.02])
    parser.add_argument("--c-alpha",type=float,default=1.)
    parser.add_argument("--panels",type=int,default=2048)
    parser.add_argument("--profile-quadrature",type=int,default=32)
    parser.add_argument("--quadrature",type=int,default=24)
    args=parser.parse_args()
    if args.self_test:self_test();return
    if not args.output:parser.error("--output is required")
    if any(p<1 or p>31 for p in args.orders) or args.panels<512 or args.profile_quadrature<16 or args.quadrature<8 or args.c_alpha<=0:
        parser.error("Require 1<=p<=31, panels>=512, profile quadrature>=16, quadrature>=8, C>0")
    result=run(args);out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(out.resolve()),"records":len(result["records"]),"reference_change":result["reference_max_field_refinement_change"],"cpu_seconds":result["cpu_seconds"]}))
    if not result["reference_checks_pass"]:raise SystemExit(2)
if __name__=="__main__":main()
