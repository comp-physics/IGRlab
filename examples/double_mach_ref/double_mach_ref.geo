// Double Mach reflection mesh: [0,4] x [0,1], 126 x 32 quads

SetFactory("OpenCASCADE");
Mesh.MshFileVersion = 2.2;

// --- Parameters ---
nx = 126;  // elements in x (streamwise)
ny = 32;   // elements in y (normal)
Lx = 4.0;
Ly = 1.0;

// --- Corner points ---
Point(1) = {0.0, 0.0, 0.0, Lx/nx};
Point(2) = {Lx,  0.0, 0.0, Lx/nx};
Point(3) = {Lx,  Ly,  0.0, Lx/nx};
Point(4) = {0.0, Ly,  0.0, Lx/nx};

// --- Boundary lines ---
// 1: bottom (y = 0)  -> slip wall
// 2: right  (x = 4)  -> outlet
// 3: top    (y = 1)  -> top BC (problem-dependent)
// 4: left   (x = 0)  -> inflow
Line(1) = {1, 2};  // bottom
Line(2) = {2, 3};  // right
Line(3) = {3, 4};  // top
Line(4) = {4, 1};  // left

// --- Surface ---
Line Loop(1)     = {1, 2, 3, 4};
Plane Surface(1) = {1};

// --- Structured transfinite mesh ---
Transfinite Curve{1, 3} = nx + 1 Using Progression 1; // along x (bottom/top)
Transfinite Curve{2, 4} = ny + 1 Using Progression 1; // along y (right/left)

Transfinite Surface{1} = {1, 2, 3, 4};
Recombine Surface{1};           // quads instead of triangles
Mesh.RecombineAll = 1;

// --- Physical groups ---
// Boundary attributes for MFEM (or similar):
Physical Curve("bottom", 1) = {1};
Physical Curve("right",  2) = {2};
Physical Curve("top",    3) = {3};
Physical Curve("left",   4) = {4};

// Domain region (cell attribute)
Physical Surface("domain", 1) = {1};
