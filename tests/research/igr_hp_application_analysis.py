#!/usr/bin/env python3
"""Analyze the bounded hp-alpha planar application sweep using existing VTK readers.

Input is the explicit application-commands.json produced for the experiment;
its entries name runs and declare p, nx, c_alpha, physical alpha, and dt. Same-
alpha comparisons use the finest available mesh, not an assumed exact solution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/perf'))
import compare_dgsem_physical_fields as physical


def diagnostics(mesh, alpha, source_gauss=48):
    p = mesh['order']
    if len(mesh['y']) != 2:
        raise ValueError('This centerline diagnostic requires exactly one element in y')
    xedges = mesh['x']
    # Interior samples include both one-sided face limits; this is a sampled
    # diagnostic, not a proof of positivity throughout a polynomial element.
    local = np.linspace(1e-10, 1-1e-10, 257)
    x = (xedges[:-1,None] + np.diff(xedges)[:,None]*local).ravel()
    y = np.full_like(x, (mesh['y'][0]+mesh['y'][-1])/2)
    u = physical.evaluate(mesh, x, y)
    pressure = .4*(u[:,3]-(u[:,1]**2+u[:,2]**2)/(2*u[:,0]))
    velocity = u[:,1]/u[:,0]
    jump = 8/3-1
    crossing = {}
    for fraction in [.1,.5,.9]:
        level = 8/3-fraction*jump
        ids = np.flatnonzero((u[:-1,0]-level)*(u[1:,0]-level) <= 0)
        crossing[str(fraction)] = [float(x[i]+(x[i+1]-x[i])*(level-u[i,0])/(u[i+1,0]-u[i,0]))
                                  for i in ids if u[i+1,0] != u[i,0]]
    nodes, weights = np.polynomial.legendre.leggauss(source_gauss)
    localg = (nodes+1)/2
    qx = (xedges[:-1,None]+np.diff(xedges)[:,None]*localg).ravel()
    qu = physical.evaluate(mesh,qx,np.full_like(qx,y[0])).reshape(-1,len(nodes),5)
    deriv = np.zeros((len(nodes),p+1))
    for j in range(p+1):
        poly = np.polynomial.Polynomial.fromroots(np.delete(np.linspace(0,1,p+1),j))
        poly /= poly(j/p)
        deriv[:,j] = poly.deriv()(localg)
    ly = physical.lagrange_basis(np.array([.5]),p)[0]
    # The case is one cell in y, with a y-independent solution. Retain its
    # polynomial restriction to the centerline instead of averaging VTK nodes.
    coeff = np.einsum('j,xjic->xic',ly,mesh['values'][0])
    du = np.einsum('qi,xic->xqc',deriv,coeff)/np.diff(xedges)[:,None,None]
    ux = (qu[:,:,0]*du[:,:,1]-qu[:,:,1]*du[:,:,0])/qu[:,:,0]**2
    source = float(np.sum(2*alpha*ux**2 * np.diff(xedges)[:,None]*weights/2))
    return {'centerline_density_min':float(u[:,0].min()),'centerline_density_max':float(u[:,0].max()),
            'centerline_pressure_min':float(pressure.min()),'centerline_pressure_max':float(pressure.max()),
            'centerline_pressure_min_x':float(x[np.argmin(pressure)]),
            'centerline_sigma_min':float(u[:,4].min()),'centerline_sigma_max':float(u[:,4].max()),
            'density_total_variation_sampled':float(np.abs(np.diff(u[:,0])).sum()),
            'velocity_face_jump_l1':float(np.abs(velocity.reshape(-1,257)[1:,0]-velocity.reshape(-1,257)[:-1,-1]).sum()),
            'density_crossings':crossing,
            'broken_source_centerline_integral':source,
            'source_gauss_points_per_element':source_gauss,
            'piece_sha256':mesh['piece_sha256']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    entries=json.loads((args.directory/'application-commands.json').read_text())
    result=[]; meshes={}
    for entry in entries:
        folder=args.directory/entry['name']
        if not (folder/'run.json').exists(): continue
        run=json.loads((folder/'run.json').read_text())
        row=dict(entry,run=run)
        for line in (folder/'stdout.txt').read_text().splitlines():
            if line.startswith('[explicit-conservation]'):
                row['conservation'] = dict(v.split('=',1) for v in line.split() if '=' in v)
        result.append(row)
        if not run['completed_and_certified']:continue
        pvds=list((folder/'vtk').rglob('*.pvd'))
        if len(pvds)!=1: raise ValueError('Expected exactly one case PVD')
        frame=physical.select_frame(pvds[0],6.,1e-12)
        mesh=physical.load_mesh(Path(frame['file']),entry['p'])
        actual_alpha=entry['c_alpha']*min(np.diff(mesh['x'])[0],np.diff(mesh['y'])[0])**2
        if not np.isclose(actual_alpha,entry['alpha'],rtol=1e-12,atol=0):raise ValueError('Alpha metadata mismatch')
        meshes[entry['name']]=mesh
        row['diagnostics']=diagnostics(mesh,actual_alpha)
        refined_source=diagnostics(mesh,actual_alpha,96)['broken_source_centerline_integral']
        source=row['diagnostics']['broken_source_centerline_integral']
        row['diagnostics']['source_quadrature_48_to_96_relative_change']=abs(refined_source-source)/max(abs(refined_source),1e-300)
    for row in result:
        if row['name'] not in meshes:continue
        candidates=[v for v in result if v['name'] in meshes and
                    np.isclose(v['alpha'],row['alpha'],rtol=1e-12,atol=0) and
                    v.get('shock_x',0)==row.get('shock_x',0)]
        reference=max(candidates,key=lambda v:(v['nx'],v['p'],-v['dt']))
        row['reference_name']=reference['name']
        row['reference_is_self']=reference['name']==row['name']
        row['same_alpha_completed_cases']=len(candidates)
        row['same_alpha_error']=physical.compare(meshes[reference['name']],meshes[row['name']],24,6)
    report={'schema':'igr-hp-application-comparison-v1',
            'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'reference_limitation':'Finest available discretization at identical alpha and initial width; not an exact solution.',
            'runs':result}
    (args.directory/'application-analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for row in result:
        if 'same_alpha_error' in row:
            err=row['same_alpha_error']['fields']
            print(row['name'],f"wall={row['run']['wall_seconds']:.3f}",f"rho={err['density']['l2_relative']:.6g}",
                  f"sigma={err['sigma']['l2_relative']:.6g}",f"source={row['diagnostics']['broken_source_centerline_integral']:.6g}")
if __name__=='__main__':main()
