spack env activate .
spack load openmpi mfem
MFEM_PREFIX=/home/jarias9/projects/dg_igr_miniapp/tpl/mfem/build

# Configure (out-of-source) and build
cmake -S . -B build -DMFEM_DIR="$MFEM_PREFIX"
cmake --build build -j4

# Install (adjust prefix as needed)
cmake --install build --prefix "$PWD/install"

# Run an example
mpirun -np 2 ./build/sedov_taylor_blast \
  -rs 0 -rp 1 -o 3 -c 0.25 -vs 10 \
  -ca 5.0 \
  -is H1 \
  -ig H1 \
  -dt_save 0.1
