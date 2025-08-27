// Gmsh geometry file for high-order mesh generation
// Cleaned and optimized for Gmsh 2.2 format

//+
SetFactory("OpenCASCADE");

// Mesh parameters
mesh_size = 0.5;        // Global mesh size
circle_mesh_size = 0.2; // Finer mesh around circle

// Define corner points of the rectangular domain
Point(1) = {0, -1, 0, mesh_size};
Point(2) = {4, -1, 0, mesh_size};
Point(3) = {4, 1, 0, mesh_size};
Point(4) = {0, 1, 0, mesh_size};

// Define lines forming the rectangular boundary
Line(1) = {1, 2};  // Bottom edge
Line(2) = {2, 3};  // Right edge  
Line(3) = {3, 4};  // Top edge
Line(4) = {4, 1};  // Left edge

// Define circular hole
Circle(5) = {0.6, 0, 0, 0.25, 0, 2*Pi};

// Create curve loops
Curve Loop(1) = {1, 2, 3, 4};  // Outer rectangle
Curve Loop(2) = {5};           // Inner circle (hole)

// Create surface with hole
Plane Surface(1) = {1, 2};

// Define physical entities for boundary conditions
Physical Curve("bottom") = {1};    // Bottom boundary
Physical Curve("right") = {2};     // Right boundary  
Physical Curve("top") = {3};       // Top boundary
Physical Curve("left") = {4};      // Left boundary
Physical Curve("circle") = {5};    // Circular hole boundary
Physical Surface("domain") = {1};  // Domain surface

// Mesh generation settings for high-order elements
Mesh.ElementOrder = 3;              // High-order elements (3rd order)
Mesh.HighOrderOptimize = 1;         // Optimize high-order mesh
Mesh.Algorithm = 6;                 // Frontal-Delaunay algorithm
Mesh.Algorithm3D = 1;               // Delaunay 3D algorithm
Mesh.OptimizeNetgen = 1;            // Optimize with Netgen
Mesh.Smoothing = 10;                // Smoothing iterations

// Mesh size field for refinement around circle
Field[1] = Distance;
Field[1].EdgesList = {5};           // Distance from circle
Field[1].NNodesByEdge = 100;

Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = circle_mesh_size;  // Fine mesh size near circle
Field[2].LcMax = mesh_size;         // Coarse mesh size away from circle
Field[2].DistMin = 0.1;             // Start transition distance
Field[2].DistMax = 0.5;             // End transition distance

Background Field = 2;

// Force Gmsh to use version 2.2 MSH format
Mesh.MshFileVersion = 2.2;
Mesh.Binary = 0;                    // ASCII format for compatibility

// Quality settings
Mesh.MinimumCirclePoints = 12;      // Minimum points on circle
Mesh.RecombineAll = 0;              // Keep triangular elements
Mesh.SubdivisionAlgorithm = 1;      // Subdivision for high-order

// Generate 2D mesh
Mesh 2;