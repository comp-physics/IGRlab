conda activate paraview-env  

./dg_igr_2d_cases/sedov_taylor_blast/sedov_taylor_blast_p_3_ref_0_is_H1_ig_H1_ca_5/sedov_taylor_blast_p_3_ref_0_is_H1_ig_H1_ca_5.pvd
./dg_igr_2d_cases/sedov_taylor_blast/sedov_taylor_blast_p_3_ref_1_is_H1_ig_H1_ca_5/sedov_taylor_blast_p_3_ref_1_is_H1_ig_H1_ca_5.pvd
./dg_igr_2d_cases/sedov_taylor_blast/sedov_taylor_blast_p_3_ref_2_is_H1_ig_H1_ca_5/sedov_taylor_blast_p_3_ref_2_is_H1_ig_H1_ca_5.pvd
./dg_igr_2d_cases/sedov_taylor_blast/sedov_taylor_blast_p_3_ref_3_is_H1_ig_H1_ca_5/sedov_taylor_blast_p_3_ref_3_is_H1_ig_H1_ca_5.pvd

pvpython export_fields.py \
  --input /path/to/your/output.pvd \
  --fields density energy pressure sigma \
  --times 0 15 25 35 45 55 \
  --outdir renders \
  --w 4200 --h 2300
