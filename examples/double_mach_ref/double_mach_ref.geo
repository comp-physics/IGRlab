// Double Mach reflection mesh: [0,4] x [0,1], 126 x 32 quads
SetFactory("OpenCASCADE");
Mesh.MshFileVersion = 2.2;

// --- Parameters ---
nx = 126;  // elements in x (streamwise)
ny = 32;   // elements in y (normal)
Lx = 4.0;
Ly = 1.0;

// Shock foot location (for bottom split)
x0 = 4.0/6.0; // ≈ 0.666666...

// Derive element counts for uniform Δx
nx_left  = Round(nx * x0 / Lx);    // ≈ 21
nx_right = nx - nx_left;           // ≈ 105

// --- Corner points ---
lc = Lx/nx; // uniform characteristic length
Point(1) = {0.0, 0.0, 0.0, lc}; // bottom-left
Point(2) = {Lx,  0.0, 0.0, lc}; // bottom-right
Point(3) = {Lx,  Ly,  0.0, lc}; // top-right
Point(4) = {0.0, Ly,  0.0, lc}; // top-left

// Points where we split bottom & top at x = x0
Point(5) = {x0, 0.0, 0.0, lc}; // bottom split point
Point(6) = {x0, Ly,  0.0, lc}; // top split point

// --- Boundary lines ---
// Bottom is now two segments:
// 10: bottom_left  (0   -> x0)
// 11: bottom_right (x0  -> 4)
Line(10) = {1, 5};  // bottom_left
Line(11) = {5, 2};  // bottom_right

// Vertical sides
Line(2)  = {2, 3};  // right  (x = 4)
Line(4)  = {4, 1};  // left   (x = 0)

// Top is also split so transfinite mapping stays structured
Line(30) = {3, 6};  // top_right  (4 -> x0)
Line(31) = {6, 4};  // top_left   (x0 -> 0)

// --- Surface ---
Line Loop(1) = {10, 11, 2, 30, 31, 4};
Plane Surface(1) = {1};

// --- Structured transfinite mesh ---
// Left part in x-direction: 0 -> x0 (nx_left elements)
Transfinite Curve{10, 31} = nx_left + 1 Using Progression 1;

// Right part in x-direction: x0 -> 4 (nx_right elements)
Transfinite Curve{11, 30} = nx_right + 1 Using Progression 1;

// Vertical sides (y-direction): ny elements
Transfinite Curve{2, 4} = ny + 1 Using Progression 1;

// Logical corners still 1,2,3,4
Transfinite Surface{1} = {1, 2, 3, 4};
Recombine Surface{1};
Mesh.RecombineAll = 1;

// --- Physical groups ---
// Boundary attributes for MFEM (or similar):
Physical Curve("bottom_left",  1) = {10}; // inflow-ish / pre-shock region
Physical Curve("bottom_right", 5) = {11}; // slip wall segment

Physical Curve("right",  2) = {2};
Physical Curve("top",    3) = {30, 31};
Physical Curve("left",   4) = {4};

// Domain region (cell attribute)
Physical Surface("domain", 1) = {1};
