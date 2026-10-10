#!/usr/bin/env python3
"""Prepare one existing VTK state for the bounded frozen-operator test.

Uses the existing strict VTK/physics reader. The C++ probe independently checks
each source point against its mesh element before interpolating the polynomial.
No application capture or restart behavior is added.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pvd',type=Path)
    parser.add_argument('--time',type=float,required=True)
    parser.add_argument('--output-order',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 1<=args.output_order<=8: parser.error('output order must be 1..8')
    root=Path(__file__).resolve().parents[2]
    reader=root/'scripts/perf/shock_qoi.py'
    spec=importlib.util.spec_from_file_location('igr_existing_qoi',reader)
    qoi=importlib.util.module_from_spec(spec);sys.modules[spec.name]=qoi;spec.loader.exec_module(qoi)
    limits=qoi.Limits(64<<20,2000000,8,8)
    frame=qoi.select_frame(args.pvd,args.time,args.pvd.stem,limits,1e-12,1e-12)
    fields=qoi.load_frame(frame,args.pvd.parent.resolve(),limits)
    if any(len(x)>2 and abs(x[2])>1e-12 for x in fields.points) or any(
            len(m)>2 and abs(m[2])>1e-12 for m in fields.momentum):
        raise ValueError('This research export requires 2D geometry and zero third momentum')
    pressure=qoi.reconstruct_pressure(fields,1.4)
    if min(fields.density)<=0 or min(pressure)<=0:raise ValueError('Invalid saved samples')
    q=(args.output_order+1)**2
    if len(fields.points)%q:raise ValueError('Point count does not match tensor output order')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w') as out:
        out.write(f'{len(fields.points)//q} {q}\n')
        for x,rho,m,e in zip(fields.points,fields.density,fields.momentum,fields.energy):
            out.write(' '.join(format(v,'.17g') for v in (x[0],x[1],rho,m[0],m[1],e))+'\n')
    record={'source_pvd':str(args.pvd.resolve()),'time':args.time,'output_order':args.output_order,
        'cells':len(fields.points)//q,'points':len(fields.points),'density_min':min(fields.density),
        'pressure_min':min(pressure),'nodal_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'reader_sha256':hashlib.sha256(reader.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'interpretation':'VTK polynomial samples in piece order. Exact state reconstruction requires sampling degree at least the known original state degree; per-element physical coordinates are independently validated by the C++ probe.'}
    args.output.with_suffix(args.output.suffix+'.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
if __name__=='__main__':main()
