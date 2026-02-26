import subprocess
import shlex

CASES = [
    "./dg_igr_2d_cases/sedov_taylor_blast/sedov_taylor_blast_p_3_ref_0_is_H1_ig_H1_ca_5/sedov_taylor_blast_p_3_ref_0_is_H1_ig_H1_ca_5.pvd",
    "./dg_igr_2d_cases/sedov_taylor_blast/sedov_taylor_blast_p_3_ref_1_is_H1_ig_H1_ca_5/sedov_taylor_blast_p_3_ref_1_is_H1_ig_H1_ca_5.pvd",
    "./dg_igr_2d_cases/sedov_taylor_blast/sedov_taylor_blast_p_3_ref_2_is_H1_ig_H1_ca_5/sedov_taylor_blast_p_3_ref_2_is_H1_ig_H1_ca_5.pvd",
    "./dg_igr_2d_cases/sedov_taylor_blast/sedov_taylor_blast_p_3_ref_3_is_H1_ig_H1_ca_5/sedov_taylor_blast_p_3_ref_3_is_H1_ig_H1_ca_5.pvd",
]

FIELDS = ["density", "energy", "pressure", "sigma"]

# Choose either indices:
TIMES = ["0", "15", "25", "35", "45", "55"]   # timestep indices
USE_TIME_VALUES = False         # set True if TIMES are actual floating times

OUTDIR = "renders_scaled"
W, H = 4200, 2300

PV_PYTHON = "pvpython"  # change if needed (full path)

cmd = [
    PV_PYTHON,
    "export_paraview_fields_scaled.py",
    "--inputs", *CASES,
    "--fields", *FIELDS,
    "--outdir", OUTDIR,
    "--w", str(W),
    "--h", str(H),
]

if TIMES:
    cmd += ["--times", *TIMES]

if USE_TIME_VALUES:
    cmd += ["--use-time-values"]

print("Running:\n", " ".join(shlex.quote(c) for c in cmd))
subprocess.check_call(cmd)
