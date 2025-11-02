// Forward-Facing Step — Structured Quads w/ base dx (OpenCASCADE, Gmsh 2.2)
SetFactory("OpenCASCADE");

// -------------------------------
// Geometry
// -------------------------------
Lx   = 3.0;
ymin = 0;
ymax =  1.0;

step_x = 0.6;          // step location
step_h = 0.2;          // step height

yb  = ymin;
yb2 = ymin + step_h;

// -------------------------------
// Uniform spacing control
// -------------------------------
dx = 0.049999;             // <-- base target cell size (set me)

// Integer cell counts per block edge (>=1), derived from dx
// (Use Ceil to avoid cells larger than dx)
nx_left  = Max(1, Ceil( step_x        / dx ));
nx_right = Max(1, Ceil((Lx - step_x)  / dx ));
ny_lower = Max(1, Ceil( step_h        / dx ));
ny_upper = Max(1, Ceil((ymax - yb2)   / dx ));

// -------------------------------
// Points
// -------------------------------
Point(1) = {0.0,    yb,   0, 1.0};
Point(2) = {step_x, yb,   0, 1.0};
Point(3) = {step_x, yb2,  0, 1.0};
Point(4) = {Lx,     yb2,  0, 1.0};
Point(5) = {Lx,     ymax, 0, 1.0};
Point(6) = {0.0,    ymax, 0, 1.0};
Point(7) = {step_x, ymax, 0, 1.0};
Point(8) = {0.0,    yb2,  0, 1.0};

// -------------------------------
// Lines (boundary & internal)
// -------------------------------
Line(11) = {1, 2};   // bottom upstream
Line(12) = {2, 3};   // step face
Line(13) = {3, 4};   // bottom downstream (raised floor)
Line(14) = {4, 5};   // outlet (right)
Line(15) = {5, 7};   // top-right
Line(16) = {7, 6};   // top-left
Line(17) = {6, 8};   // inlet upper
Line(18) = {8, 1};   // inlet lower
Line(19) = {8, 3};   // internal: y = yb2 (upper-left bottom)
Line(20) = {3, 7};   // internal: x = step_x (upper vertical)

// -------------------------------
// Surfaces (auto-tagged to avoid collisions)
// -------------------------------
ll1 = newll; Line Loop(ll1) = {11, 12, -19, 18};  s1 = news; Plane Surface(s1) = {ll1}; // lower-left
ll2 = newll; Line Loop(ll2) = {16, -20, -19, -17}; s2 = news; Plane Surface(s2) = {ll2}; // upper-left
ll3 = newll; Line Loop(ll3) = {13, 14, 15, -20};  s3 = news; Plane Surface(s3) = {ll3}; // upper-right

// -------------------------------
// Transfinite (structured) setup — uniform spacing (Progression=1)
// -------------------------------
// Horizontal (x)
Transfinite Curve {11, 19, 16} = nx_left  Using Progression 1;   // width = step_x
Transfinite Curve {13, 15}     = nx_right Using Progression 1;   // width = Lx - step_x
// Vertical (y)
Transfinite Curve {12, 18}     = ny_lower Using Progression 1;   // height = step_h
Transfinite Curve {17, 20, 14} = ny_upper Using Progression 1;   // height = ymax - yb2

Transfinite Surface {s1, s2, s3};
Recombine Surface   {s1, s2, s3};

// -------------------------------
// Physical groups
// -------------------------------
Physical Curve("inlet_upper")       = {17};
Physical Curve("inlet_lower")       = {18};
Physical Curve("inlet")             = {17, 18};
Physical Curve("outlet")            = {14};
Physical Curve("top")               = {15, 16};
Physical Curve("bottom_upstream")   = {11};
Physical Curve("step_face")         = {12};
Physical Curve("bottom_downstream") = {13};
Physical Curve("walls")             = {15, 16, 11, 12, 13};

Physical Surface("domain") = {s1, s2, s3};

// -------------------------------
// Mesh controls (P1 rectilinear quads)
// -------------------------------
Mesh.ElementOrder = 1;
Mesh.HighOrderOptimize = 0;
Mesh.RecombineAll = 1;   // quads
Mesh.Algorithm = 8;      // harmless with transfinite

Mesh.MshFileVersion = 2.2;
Mesh.Binary = 0;

Mesh 2;
