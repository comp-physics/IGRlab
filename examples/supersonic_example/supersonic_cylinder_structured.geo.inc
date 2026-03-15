// Upper-half source geometry for the supersonic-cylinder mesh.
// The final full mesh is produced by mirroring this half mesh in
// `build_supersonic_cylinder_mesh.py`.
//
// Policy:
// - explicit structured concentric annulus around the cylinder,
// - generic quads outside the annulus,
// - mild wall-layer meshing only on the top wall.

SetFactory("OpenCASCADE");

xc = 0.6;
yc = 0.0;
r_cyl = 0.25;
r_ring = DefineNumber[0.42, Name "Parameters/ring_outer_radius"];

x_min = 0.0;
x_max = 4.0;
y_mid = 0.0;
y_max = 1.0;

h_far = DefineNumber[0.24, Name "Parameters/background_dx"];
h_mid = DefineNumber[0.18, Name "Parameters/midfield_dx"];
h_wall = DefineNumber[0.064, Name "Parameters/cylinder_layer_dx"];
ring_layers = DefineNumber[2, Name "Parameters/ring_layers"];
ring_ratio = DefineNumber[1.2, Name "Parameters/ring_ratio"];
top_wall_dx = DefineNumber[0.14, Name "Parameters/top_wall_dx"];
top_wall_layers = DefineNumber[1, Name "Parameters/top_wall_layers"];
top_wall_ratio = DefineNumber[1.00, Name "Parameters/top_wall_ratio"];
top_wall_thickness = DefineNumber[0.16, Name "Parameters/top_wall_thickness"];
arc_cells_per_sector = DefineNumber[3, Name "Parameters/arc_cells_per_sector"];
order = DefineNumber[2, Name "Parameters/order"];

a45 = Pi / 4.0;

Point(1) = {x_min, y_max, 0, h_far};
Point(2) = {x_max, y_max, 0, h_far};
Point(3) = {x_max, y_mid, 0, h_far};
Point(4) = {xc + r_ring, y_mid, 0, h_mid};
Point(5) = {xc + r_cyl, y_mid, 0, h_wall};
Point(6) = {xc + r_cyl * Cos(a45), yc + r_cyl * Sin(a45), 0, h_wall};
Point(7) = {xc, yc + r_cyl, 0, h_wall};
Point(8) = {xc - r_cyl * Cos(a45), yc + r_cyl * Sin(a45), 0, h_wall};
Point(9) = {xc - r_cyl, y_mid, 0, h_wall};
Point(10) = {xc + r_ring * Cos(a45), yc + r_ring * Sin(a45), 0, h_mid};
Point(11) = {xc, yc + r_ring, 0, h_mid};
Point(12) = {xc - r_ring * Cos(a45), yc + r_ring * Sin(a45), 0, h_mid};
Point(13) = {xc - r_ring, y_mid, 0, h_mid};
Point(14) = {x_min, y_mid, 0, h_far};
Point(100) = {xc, yc, 0, h_wall};

// Outer generic region boundary, oriented CCW.
Line(1) = {1, 14};
Line(2) = {14, 13};
Circle(3) = {13, 100, 12};
Circle(4) = {12, 100, 11};
Circle(5) = {11, 100, 10};
Circle(6) = {10, 100, 4};
Line(7) = {4, 3};
Line(8) = {3, 2};
Line(9) = {2, 1};

Curve Loop(101) = {1, 2, 3, 4, 5, 6, 7, 8, 9};
Plane Surface(101) = {101};

// Structured annulus around the cylinder.
Circle(21) = {9, 100, 8};
Circle(22) = {8, 100, 7};
Circle(23) = {7, 100, 6};
Circle(24) = {6, 100, 5};

Line(31) = {9, 13};
Line(32) = {8, 12};
Line(33) = {7, 11};
Line(34) = {6, 10};
Line(35) = {5, 4};

Curve Loop(201) = {31, 3, -32, -21};
Plane Surface(201) = {201};
Curve Loop(202) = {32, 4, -33, -22};
Plane Surface(202) = {202};
Curve Loop(203) = {33, 5, -34, -23};
Plane Surface(203) = {203};
Curve Loop(204) = {34, 6, -35, -24};
Plane Surface(204) = {204};

Physical Surface("domain", 6) = {101, 201, 202, 203, 204};

// Outer-region size control based on distance from the structured annulus.
Field[1] = Distance;
Field[1].CurvesList = {3, 4, 5, 6};
Field[1].Sampling = 128;

Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = h_mid;
Field[2].LcMax = h_far;
Field[2].DistMin = 0.0;
Field[2].DistMax = 1.6;
Field[2].Sigmoid = 1;

// Mild top-wall layer. The mirrored bottom wall gets the same layer.
Field[3] = BoundaryLayer;
Field[3].CurvesList = {9};
Field[3].NbLayers = top_wall_layers;
Field[3].Size = top_wall_dx;
Field[3].Thickness = top_wall_thickness;
Field[3].Ratio = top_wall_ratio;
Field[3].Quads = 1;
Field[3].IntersectMetrics = 1;
Field[3].FanPointsList = {1, 2};
Field[3].FanPointsSizesList = {6, 6};
BoundaryLayer Field = 3;

Background Field = 2;

Transfinite Curve {21, 22, 23, 24, 3, 4, 5, 6} = arc_cells_per_sector + 1;
Transfinite Curve {31, 32, 33, 34, 35} = ring_layers + 1 Using Progression ring_ratio;
Transfinite Surface {201, 202, 203, 204};
Recombine Surface {201, 202, 203, 204};
Recombine Surface {101};

Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.Algorithm = 11;
Mesh.QuadqsSizemapMethod = 4;
Mesh.RecombineAll = 0;
Mesh.RecombinationAlgorithm = 2;
Mesh.Smoothing = 60;
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;
Mesh.HighOrderOptimize = 2;
Mesh.ElementOrder = order;
Mesh.MinimumCirclePoints = 96;
Mesh.MshFileVersion = 2.2;
Mesh.Binary = 0;
