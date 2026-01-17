// shock_vortex_IH.geo
// Inoue–Hattori canonical domain: X ∈ [-30,10], Y ∈ [-20,20]

SetFactory("OpenCASCADE");
Mesh.MshFileVersion = 2.2;

// Parameters
xL = -30.0;
xR =  10.0;
yB = -20.0;
yT =  20.0;

// Number of cells in each direction
NX = 32;
NY = 32;

// Characteristic length
lc = 1.0;

// Corner points
Point(1) = {xL, yB, 0.0, lc};
Point(2) = {xR, yB, 0.0, lc};
Point(3) = {xR, yT, 0.0, lc};
Point(4) = {xL, yT, 0.0, lc};

// Boundary lines
Line(1) = {1, 2}; // bottom
Line(2) = {2, 3}; // right
Line(3) = {3, 4}; // top
Line(4) = {4, 1}; // left

Line Loop(10) = {1, 2, 3, 4};
Plane Surface(11) = {10};

// Structured mesh: 32×32
Transfinite Line{1,3} = NX + 1;
Transfinite Line{2,4} = NY + 1;
Transfinite Surface{11};
Recombine Surface{11}; // quads

// Physical groups
Physical Surface("fluid", 1) = {11};

Physical Line("bottom", 1) = {1};
Physical Line("right",  2) = {2};
Physical Line("top",    3) = {3};
Physical Line("left",   4) = {4};