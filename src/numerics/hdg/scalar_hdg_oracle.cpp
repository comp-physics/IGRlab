#include "numerics/hdg/scalar_hdg_oracle.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace dg::hdg
{
namespace
{

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double kappa = 1.0;
constexpr double beta = 1.3;
constexpr double solve_tolerance = 2.0e-12;
constexpr double comparison_tolerance = 2.0e-10;
constexpr double recovery_tolerance = 5.0e-10;
constexpr double jacobian_tolerance = 2.0e-7;

struct LocalSystem
{
   mfem::DenseMatrix A;
   mfem::DenseMatrix B;
   mfem::DenseMatrix C;
   mfem::DenseMatrix D;
   mfem::Vector rhs;
   std::array<int, 2> faces{};
   int scalar_dofs = 0;
};

struct DiscreteSystem
{
   ScalarProblem problem = ScalarProblem::poisson;
   MeshGeometry geometry = MeshGeometry::affine;
   int order = 1;
   int elements = 0;
   int scalar_dofs = 0;
   int interior_dofs = 0;
   std::vector<LocalSystem> local;
   std::vector<int> face_to_free;
   std::vector<double> essential_value;
   std::vector<double> face_coordinate;
   double h_max = 0.0;
};

struct Solution
{
   mfem::Vector trace;
   std::vector<mfem::Vector> interior;
};

struct CondensedSystem
{
   mfem::DenseMatrix matrix;
   mfem::Vector rhs;
};

double Exact(const ScalarProblem problem, const double x)
{
   if (problem == ScalarProblem::poisson)
   {
      return 1.0 + std::sin(pi * x) + 0.25 * std::sin(2.0 * pi * x);
   }
   return 1.0 + 0.2 * std::sin(2.0 * pi * x) + 0.1 * std::cos(pi * x);
}

double ExactDerivative(const ScalarProblem problem, const double x)
{
   if (problem == ScalarProblem::poisson)
   {
      return pi * std::cos(pi * x) + 0.5 * pi * std::cos(2.0 * pi * x);
   }
   return 0.4 * pi * std::cos(2.0 * pi * x) - 0.1 * pi * std::sin(pi * x);
}

double Source(const ScalarProblem problem, const double x)
{
   if (problem == ScalarProblem::poisson)
   {
      return pi * pi * (std::sin(pi * x) + std::sin(2.0 * pi * x));
   }
   return beta * ExactDerivative(problem, x);
}

double PoissonSourceCoefficient(const mfem::Vector &point)
{
   return Source(ScalarProblem::poisson, point[0]);
}

double AdvectionSourceCoefficient(const mfem::Vector &point)
{
   return Source(ScalarProblem::advection, point[0]);
}


double WarpedCoordinate(const double point)
{
   constexpr double amplitude = 0.18;
   return point + amplitude * std::sin(2.0 * pi * point) / (2.0 * pi);
}

mfem::Mesh MakeMesh(const int elements, const MeshGeometry geometry,
                    const int order)
{
   mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(elements, 1.0);
   if (geometry == MeshGeometry::curved)
   {
      mesh.SetCurvature(std::max(3, order + 1));
      mfem::GridFunction &nodes = *mesh.GetNodes();
      const mfem::Vector original(nodes);
      for (int i = 0; i < nodes.Size(); ++i)
      {
         nodes[i] = WarpedCoordinate(original[i]);
      }
   }
   return mesh;
}

void AddComparison(std::vector<Comparison> &checks, const std::string &name,
                   const double value, const double tolerance)
{
   checks.push_back({name, value, tolerance,
                     std::isfinite(value) && value <= tolerance});
}

double MaxAbs(const mfem::Vector &v)
{
   double result = 0.0;
   for (int i = 0; i < v.Size(); ++i)
   {
      result = std::max(result, std::abs(v[i]));
   }
   return result;
}

double MaxDifference(const mfem::Vector &a, const mfem::Vector &b)
{
   if (a.Size() != b.Size()) { return std::numeric_limits<double>::infinity(); }
   double result = 0.0;
   for (int i = 0; i < a.Size(); ++i)
   {
      result = std::max(result, std::abs(a[i] - b[i]));
   }
   return result;
}

double MaxDifference(const mfem::DenseMatrix &a, const mfem::DenseMatrix &b)
{
   if (a.Height() != b.Height() || a.Width() != b.Width())
   {
      return std::numeric_limits<double>::infinity();
   }
   double result = 0.0;
   for (int i = 0; i < a.Height(); ++i)
   {
      for (int j = 0; j < a.Width(); ++j)
      {
         result = std::max(result, std::abs(a(i, j) - b(i, j)));
      }
   }
   return result;
}

mfem::Vector Solve(const mfem::DenseMatrix &matrix, const mfem::Vector &rhs)
{
   if (matrix.Height() == 0) { return mfem::Vector(0); }
   mfem::DenseMatrixInverse inverse(matrix);
   mfem::Vector solution(rhs.Size());
   inverse.Mult(rhs, solution);
   mfem::Vector residual(rhs.Size());
   matrix.Mult(solution, residual);
   residual -= rhs;
   if (MaxAbs(residual) > solve_tolerance * (1.0 + MaxAbs(rhs)))
   {
      throw std::runtime_error("MFEM dense direct solve exceeded frozen tolerance");
   }
   return solution;
}

mfem::DenseMatrix ApplyInverse(const mfem::DenseMatrix &matrix,
                               const mfem::DenseMatrix &rhs)
{
   mfem::DenseMatrix result(rhs.Height(), rhs.Width());
   mfem::DenseMatrixInverse inverse(matrix);
   mfem::Vector column(rhs.Height());
   mfem::Vector solved(rhs.Height());
   for (int j = 0; j < rhs.Width(); ++j)
   {
      for (int i = 0; i < rhs.Height(); ++i) { column[i] = rhs(i, j); }
      inverse.Mult(column, solved);
      for (int i = 0; i < rhs.Height(); ++i) { result(i, j) = solved[i]; }
   }
   return result;
}

double FaceTau(const ScalarProblem problem, const int order,
               const double h, const double normal)
{
   if (problem == ScalarProblem::poisson)
   {
      const double degree = static_cast<double>(order + 1);
      return 4.0 * degree * degree / h;
   }
   return std::max(beta * normal, 0.0);
}

LocalSystem AssembleLocal(const ScalarProblem problem, const int order,
                          mfem::Mesh &mesh, mfem::FiniteElementSpace &space,
                          const int element, const std::array<int, 2> faces)
{
   const mfem::FiniteElement &fe = *space.GetFE(element);
   const int dofs = fe.GetDof();
   const int interior = problem == ScalarProblem::poisson ? 2 * dofs : dofs;
   LocalSystem system;
   system.A.SetSize(interior, interior);
   system.B.SetSize(interior, 2);
   system.C.SetSize(2, interior);
   system.D.SetSize(2, 2);
   system.rhs.SetSize(interior);
   system.A = 0.0;
   system.B = 0.0;
   system.C = 0.0;
   system.D = 0.0;
   system.rhs = 0.0;
   system.faces = faces;
   system.scalar_dofs = dofs;

   mfem::ElementTransformation &transformation =
      *mesh.GetElementTransformation(element);
   const mfem::IntegrationRule &rule =
      mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2 * order + 8);
   mfem::Vector shape(dofs);
   mfem::DenseMatrix derivative(dofs, 1);
   mfem::Vector physical(1);
   for (int q = 0; q < rule.GetNPoints(); ++q)
   {
      const mfem::IntegrationPoint &ip = rule.IntPoint(q);
      transformation.SetIntPoint(&ip);
      transformation.Transform(ip, physical);
      const double jacobian = transformation.Weight();
      fe.CalcShape(ip, shape);
      fe.CalcDShape(ip, derivative);
      const double weight = ip.weight;
      if (problem == ScalarProblem::poisson)
      {
         for (int i = 0; i < dofs; ++i)
         {
            system.rhs[dofs + i] +=
               weight * jacobian * Source(problem, physical[0]) * shape[i];
            for (int j = 0; j < dofs; ++j)
            {
               system.A(i, j) +=
                  weight * jacobian * shape[i] * shape[j] / kappa;
               system.A(i, dofs + j) -=
                  weight * derivative(i, 0) * shape[j];
               system.A(dofs + i, j) -=
                  weight * derivative(i, 0) * shape[j];
            }
         }
      }
      else
      {
         for (int i = 0; i < dofs; ++i)
         {
            system.rhs[i] +=
               weight * jacobian * Source(problem, physical[0]) * shape[i];
            for (int j = 0; j < dofs; ++j)
            {
               system.A(i, j) -=
                  weight * beta * derivative(i, 0) * shape[j];
            }
         }
      }
   }

   std::array<mfem::Vector, 2> face_shape = {mfem::Vector(dofs),
                                              mfem::Vector(dofs)};
   mfem::IntegrationPoint left;
   mfem::IntegrationPoint right;
   left.x = 0.0;
   right.x = 1.0;
   fe.CalcShape(left, face_shape[0]);
   fe.CalcShape(right, face_shape[1]);
   transformation.SetIntPoint(&left);
   mfem::Vector left_x(1);
   transformation.Transform(left, left_x);
   transformation.SetIntPoint(&right);
   mfem::Vector right_x(1);
   transformation.Transform(right, right_x);
   const double h = std::abs(right_x[0] - left_x[0]);

   for (int side = 0; side < 2; ++side)
   {
      const double normal = side == 0 ? -1.0 : 1.0;
      const double tau = FaceTau(problem, order, h, normal);
      const mfem::Vector &value = face_shape[side];
      if (problem == ScalarProblem::poisson)
      {
         for (int i = 0; i < dofs; ++i)
         {
            system.B(i, side) += normal * value[i];
            system.B(dofs + i, side) -= tau * value[i];
            system.C(side, i) += normal * value[i];
            system.C(side, dofs + i) += tau * value[i];
            for (int j = 0; j < dofs; ++j)
            {
               system.A(dofs + i, j) +=
                  normal * value[i] * value[j];
               system.A(dofs + i, dofs + j) +=
                  tau * value[i] * value[j];
            }
         }
         system.D(side, side) -= tau;
      }
      else
      {
         for (int i = 0; i < dofs; ++i)
         {
            system.B(i, side) += (beta * normal - tau) * value[i];
            system.C(side, i) += tau * value[i];
            for (int j = 0; j < dofs; ++j)
            {
               system.A(i, j) += tau * value[i] * value[j];
            }
         }
         system.D(side, side) += beta * normal - tau;
      }
   }
   return system;
}

DiscreteSystem AssembleSystem(const ScalarProblem problem,
                              const MeshGeometry geometry, const int order,
                              const int elements)
{
   mfem::Mesh mesh = MakeMesh(elements, geometry, order);
   mfem::L2_FECollection interior_collection(
      order, 1, mfem::BasisType::GaussLegendre, mfem::FiniteElement::VALUE);
   mfem::FiniteElementSpace interior_space(&mesh, &interior_collection);
   mfem::H1_Trace_FECollection trace_collection(1, 1);
   mfem::FiniteElementSpace trace_space(&mesh, &trace_collection);

   DiscreteSystem system;
   system.problem = problem;
   system.geometry = geometry;
   system.order = order;
   system.elements = elements;
   system.scalar_dofs = order + 1;
   system.interior_dofs =
      problem == ScalarProblem::poisson ? 2 * (order + 1) : order + 1;
   system.face_to_free.assign(mesh.GetNV(), -1);
   system.essential_value.assign(mesh.GetNV(), 0.0);
   system.face_coordinate.assign(mesh.GetNV(), 0.0);

   if (interior_space.GetNDofs() != elements * (order + 1) ||
       trace_space.GetNDofs() != mesh.GetNV())
   {
      throw std::runtime_error("MFEM scalar interior/trace space size changed");
   }
   for (int vertex = 0; vertex < mesh.GetNV(); ++vertex)
   {
      system.face_coordinate[vertex] = mesh.GetVertex(vertex)[0];
   }

   int left_boundary = 0;
   int right_boundary = 0;
   for (int vertex = 1; vertex < mesh.GetNV(); ++vertex)
   {
      if (system.face_coordinate[vertex] <
          system.face_coordinate[left_boundary])
      {
         left_boundary = vertex;
      }
      if (system.face_coordinate[vertex] >
          system.face_coordinate[right_boundary])
      {
         right_boundary = vertex;
      }
   }
   int free_index = 0;
   for (int vertex = 0; vertex < mesh.GetNV(); ++vertex)
   {
      const bool poisson_essential =
         problem == ScalarProblem::poisson &&
         (vertex == left_boundary || vertex == right_boundary);
      const bool advection_inflow =
         problem == ScalarProblem::advection && vertex == left_boundary;
      const bool advection_outflow =
         problem == ScalarProblem::advection && vertex == right_boundary;
      if (poisson_essential || advection_inflow)
      {
         system.essential_value[vertex] =
            Exact(problem, system.face_coordinate[vertex]);
      }
      else if (!advection_outflow)
      {
         system.face_to_free[vertex] = free_index++;
      }
   }

   mfem::Array<int> vertices;
   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      mesh.GetElementVertices(element, vertices);
      if (vertices.Size() != 2)
      {
         throw std::runtime_error("HDG oracle requires segment elements");
      }
      mfem::ElementTransformation &transformation =
         *mesh.GetElementTransformation(element);
      mfem::IntegrationPoint left;
      mfem::IntegrationPoint right;
      left.x = 0.0;
      right.x = 1.0;
      mfem::Vector left_x(1);
      mfem::Vector right_x(1);
      transformation.Transform(left, left_x);
      transformation.Transform(right, right_x);
      system.h_max =
         std::max(system.h_max, std::abs(right_x[0] - left_x[0]));
      system.local.push_back(AssembleLocal(
         problem, order, mesh, interior_space, element,
         {vertices[0], vertices[1]}));
   }
   return system;
}

mfem::Vector LocalTrace(const DiscreteSystem &system,
                        const LocalSystem &local,
                        const mfem::Vector &free_trace)
{
   mfem::Vector trace(2);
   for (int side = 0; side < 2; ++side)
   {
      const int face = local.faces[side];
      const int free = system.face_to_free[face];
      trace[side] = free >= 0 ? free_trace[free]
                                : system.essential_value[face];
   }
   return trace;
}

CondensedSystem Condense(const DiscreteSystem &system,
                         const int first_element = 0,
                         int last_element = -1)
{
   const int free_dofs = *std::max_element(system.face_to_free.begin(),
                                           system.face_to_free.end()) + 1;
   CondensedSystem condensed;
   condensed.matrix.SetSize(free_dofs, free_dofs);
   condensed.rhs.SetSize(free_dofs);
   condensed.matrix = 0.0;
   condensed.rhs = 0.0;
   if (last_element < 0) { last_element = system.elements; }

   for (int element = first_element; element < last_element; ++element)
   {
      const LocalSystem &local = system.local[element];
      mfem::Vector effective_rhs(local.rhs);
      const mfem::Vector particular = Solve(local.A, effective_rhs);
      const mfem::DenseMatrix inverse_b = ApplyInverse(local.A, local.B);
      for (int row_side = 0; row_side < 2; ++row_side)
      {
         const int row_face = local.faces[row_side];
         const int global_row = system.face_to_free[row_face];
         if (global_row < 0) { continue; }
         double rhs_value = 0.0;
         for (int i = 0; i < local.C.Width(); ++i)
         {
            rhs_value -= local.C(row_side, i) * particular[i];
         }
         for (int column_side = 0; column_side < 2; ++column_side)
         {
            const int column_face = local.faces[column_side];
            const int global_column = system.face_to_free[column_face];
            double schur_value = local.D(row_side, column_side);
            for (int i = 0; i < local.C.Width(); ++i)
            {
               schur_value -=
                  local.C(row_side, i) * inverse_b(i, column_side);
            }
            if (global_column >= 0)
            {
               condensed.matrix(global_row, global_column) += schur_value;
            }
            else
            {
               rhs_value -= schur_value * system.essential_value[column_face];
            }
         }
         condensed.rhs[global_row] += rhs_value;
      }
   }
   return condensed;
}

Solution Recover(const DiscreteSystem &system, const mfem::Vector &free_trace)
{
   Solution solution;
   solution.trace.SetSize(system.face_to_free.size());
   solution.trace = 0.0;
   for (int face = 0; face < solution.trace.Size(); ++face)
   {
      const int free = system.face_to_free[face];
      solution.trace[face] = free >= 0 ? free_trace[free]
                                         : system.essential_value[face];
   }
   solution.interior.reserve(system.local.size());
   for (const LocalSystem &local : system.local)
   {
      const mfem::Vector trace = LocalTrace(system, local, free_trace);
      mfem::Vector rhs(local.rhs);
      mfem::Vector product(local.A.Height());
      local.B.Mult(trace, product);
      rhs -= product;
      solution.interior.push_back(Solve(local.A, rhs));
   }
   if (system.problem == ScalarProblem::advection)
   {
      const LocalSystem &last = system.local.back();
      const int outflow_face = last.faces[1];
      mfem::Mesh mesh = MakeMesh(system.elements, system.geometry, system.order);
      mfem::L2_FECollection collection(system.order, 1,
                                       mfem::BasisType::GaussLegendre);
      mfem::FiniteElementSpace space(&mesh, &collection);
      mfem::IntegrationPoint endpoint;
      endpoint.x = 1.0;
      mfem::Vector shape(system.scalar_dofs);
      space.GetFE(system.elements - 1)->CalcShape(endpoint, shape);
      double value = 0.0;
      for (int i = 0; i < system.scalar_dofs; ++i)
      {
         value += shape[i] * solution.interior.back()[i];
      }
      solution.trace[outflow_face] = value;
   }
   return solution;
}

Solution SolveCondensed(const DiscreteSystem &system)
{
   const CondensedSystem condensed = Condense(system);
   return Recover(system, Solve(condensed.matrix, condensed.rhs));
}

Solution SolveFull(const DiscreteSystem &system)
{
   const int free_dofs = *std::max_element(system.face_to_free.begin(),
                                           system.face_to_free.end()) + 1;
   const int local_dofs = system.interior_dofs;
   const int total_interior = local_dofs * system.elements;
   mfem::DenseMatrix matrix(total_interior + free_dofs);
   mfem::Vector rhs(total_interior + free_dofs);
   matrix = 0.0;
   rhs = 0.0;
   for (int element = 0; element < system.elements; ++element)
   {
      const LocalSystem &local = system.local[element];
      const int offset = element * local_dofs;
      for (int i = 0; i < local_dofs; ++i)
      {
         rhs[offset + i] = local.rhs[i];
         for (int j = 0; j < local_dofs; ++j)
         {
            matrix(offset + i, offset + j) += local.A(i, j);
         }
         for (int side = 0; side < 2; ++side)
         {
            const int face = local.faces[side];
            const int free = system.face_to_free[face];
            if (free >= 0)
            {
               matrix(offset + i, total_interior + free) += local.B(i, side);
            }
            else
            {
               rhs[offset + i] -=
                  local.B(i, side) * system.essential_value[face];
            }
         }
      }
      for (int row_side = 0; row_side < 2; ++row_side)
      {
         const int row_face = local.faces[row_side];
         const int free_row = system.face_to_free[row_face];
         if (free_row < 0) { continue; }
         const int row = total_interior + free_row;
         for (int j = 0; j < local_dofs; ++j)
         {
            matrix(row, offset + j) += local.C(row_side, j);
         }
         for (int column_side = 0; column_side < 2; ++column_side)
         {
            const int column_face = local.faces[column_side];
            const int free_column = system.face_to_free[column_face];
            if (free_column >= 0)
            {
               matrix(row, total_interior + free_column) +=
                  local.D(row_side, column_side);
            }
            else
            {
               rhs[row] -= local.D(row_side, column_side) *
                           system.essential_value[column_face];
            }
         }
      }
   }
   const mfem::Vector monolithic = Solve(matrix, rhs);
   mfem::Vector free_trace(free_dofs);
   for (int i = 0; i < free_dofs; ++i)
   {
      free_trace[i] = monolithic[total_interior + i];
   }
   Solution solution = Recover(system, free_trace);
   for (int element = 0; element < system.elements; ++element)
   {
      for (int i = 0; i < local_dofs; ++i)
      {
         solution.interior[element][i] =
            monolithic[element * local_dofs + i];
      }
   }
   return solution;
}

double SolutionDifference(const DiscreteSystem &system, const Solution &a,
                          const Solution &b)
{
   double difference = MaxDifference(a.trace, b.trace);
   for (int element = 0; element < system.elements; ++element)
   {
      difference = std::max(
         difference,
         MaxDifference(a.interior[element], b.interior[element]));
   }
   return difference;
}

double L2Error(const DiscreteSystem &system, const Solution &solution)
{
   mfem::Mesh mesh = MakeMesh(system.elements, system.geometry, system.order);
   mfem::L2_FECollection collection(system.order, 1,
                                    mfem::BasisType::GaussLegendre);
   mfem::FiniteElementSpace space(&mesh, &collection);
   const mfem::IntegrationRule &rule =
      mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2 * system.order + 12);
   double error = 0.0;
   for (int element = 0; element < system.elements; ++element)
   {
      const mfem::FiniteElement &fe = *space.GetFE(element);
      mfem::ElementTransformation &transformation =
         *mesh.GetElementTransformation(element);
      mfem::Vector shape(system.scalar_dofs);
      mfem::Vector physical(1);
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
         const mfem::IntegrationPoint &ip = rule.IntPoint(q);
         transformation.SetIntPoint(&ip);
         transformation.Transform(ip, physical);
         fe.CalcShape(ip, shape);
         double numerical = 0.0;
         const int offset = system.problem == ScalarProblem::poisson
                               ? system.scalar_dofs
                               : 0;
         for (int i = 0; i < system.scalar_dofs; ++i)
         {
            numerical += shape[i] * solution.interior[element][offset + i];
         }
         const double delta = numerical - Exact(system.problem, physical[0]);
         error += ip.weight * transformation.Weight() * delta * delta;
      }
   }
   return std::sqrt(error);
}

double ConservationError(const DiscreteSystem &system,
                         const Solution &solution)
{
   double error = 0.0;
   std::vector<double> face_balance(system.face_to_free.size(), 0.0);
   for (int element = 0; element < system.elements; ++element)
   {
      const LocalSystem &local = system.local[element];
      mfem::Vector flux(2);
      local.C.Mult(solution.interior[element], flux);
      mfem::Vector local_trace(2);
      for (int side = 0; side < 2; ++side)
      {
         local_trace[side] = solution.trace[local.faces[side]];
      }
      mfem::Vector dtrace(2);
      local.D.Mult(local_trace, dtrace);
      flux += dtrace;
      for (int side = 0; side < 2; ++side)
      {
         face_balance[local.faces[side]] += flux[side];
      }

      const int constant_row =
         system.problem == ScalarProblem::poisson ? system.scalar_dofs : 0;
      mfem::Vector residual(local.A.Height());
      local.A.Mult(solution.interior[element], residual);
      mfem::Vector btrace(local.A.Height());
      local.B.Mult(local_trace, btrace);
      residual += btrace;
      residual -= local.rhs;
      error = std::max(error, std::abs(residual[constant_row]));
   }
   for (int face = 0; face < static_cast<int>(face_balance.size()); ++face)
   {
      if (system.face_to_free[face] >= 0)
      {
         error = std::max(error, std::abs(face_balance[face]));
      }
   }
   return error;
}

double EssentialBoundaryError(const DiscreteSystem &system,
                              const Solution &solution)
{
   double error = 0.0;
   for (int face = 0; face < static_cast<int>(system.face_to_free.size()); ++face)
   {
      if (system.face_to_free[face] < 0)
      {
         const bool advection_outflow =
            system.problem == ScalarProblem::advection &&
            system.face_coordinate[face] > 0.999999;
         if (!advection_outflow)
         {
            error = std::max(
               error,
               std::abs(solution.trace[face] - system.essential_value[face]));
         }
      }
   }
   return error;
}

double ConstantStateError(const DiscreteSystem &system)
{
   constexpr double state_value = 1.25;
   const LocalSystem &local = system.local.front();
   mfem::Vector interior(local.A.Height());
   mfem::Vector trace(2);
   interior = 0.0;
   trace = state_value;
   const int offset = system.problem == ScalarProblem::poisson
                         ? system.scalar_dofs
                         : 0;
   for (int i = 0; i < system.scalar_dofs; ++i)
   {
      interior[offset + i] = state_value;
   }
   mfem::Vector residual(local.A.Height());
   local.A.Mult(interior, residual);
   mfem::Vector btrace(local.A.Height());
   local.B.Mult(trace, btrace);
   residual += btrace;
   mfem::Vector flux(2);
   local.C.Mult(interior, flux);
   mfem::Vector dtrace(2);
   local.D.Mult(trace, dtrace);
   flux += dtrace;
   const double flux_balance = std::abs(flux[0] + flux[1]);
   return std::max(MaxAbs(residual), flux_balance);
}

double ResidualJacobianError(const DiscreteSystem &system)
{
   const LocalSystem &local = system.local.front();
   const int rows = local.A.Height();
   const int columns = rows + 2;
   mfem::Vector state(columns);
   for (int i = 0; i < columns; ++i)
   {
      state[i] = 0.125 + 0.03125 * static_cast<double>(i + 1);
   }
   auto residual = [&](const mfem::Vector &input)
   {
      mfem::Vector interior(input.GetData(), rows);
      mfem::Vector trace(input.GetData() + rows, 2);
      mfem::Vector value(rows);
      local.A.Mult(interior, value);
      mfem::Vector contribution(rows);
      local.B.Mult(trace, contribution);
      value += contribution;
      value -= local.rhs;
      return value;
   };
   mfem::DenseMatrix analytic(rows, columns);
   for (int i = 0; i < rows; ++i)
   {
      for (int j = 0; j < rows; ++j) { analytic(i, j) = local.A(i, j); }
      for (int j = 0; j < 2; ++j)
      {
         analytic(i, rows + j) = local.B(i, j);
      }
   }
   mfem::DenseMatrix finite_difference(rows, columns);
   const double step = 1.0e-7;
   for (int column = 0; column < columns; ++column)
   {
      mfem::Vector plus(state);
      mfem::Vector minus(state);
      plus[column] += step;
      minus[column] -= step;
      const mfem::Vector r_plus = residual(plus);
      const mfem::Vector r_minus = residual(minus);
      for (int row = 0; row < rows; ++row)
      {
         finite_difference(row, column) =
            (r_plus[row] - r_minus[row]) / (2.0 * step);
      }
   }
   return MaxDifference(analytic, finite_difference);
}

mfem::Vector EvaluateSkeletonResidual(const DiscreteSystem &system,
                                      const mfem::Vector &free_trace)
{
   mfem::Vector residual(free_trace.Size());
   residual = 0.0;
   for (const LocalSystem &local : system.local)
   {
      const mfem::Vector trace = LocalTrace(system, local, free_trace);
      mfem::Vector interior_rhs(local.rhs);
      mfem::Vector btrace(local.A.Height());
      local.B.Mult(trace, btrace);
      interior_rhs -= btrace;
      const mfem::Vector interior = Solve(local.A, interior_rhs);
      mfem::Vector local_residual(2);
      local.C.Mult(interior, local_residual);
      mfem::Vector dtrace(2);
      local.D.Mult(trace, dtrace);
      local_residual += dtrace;
      for (int side = 0; side < 2; ++side)
      {
         const int free = system.face_to_free[local.faces[side]];
         if (free >= 0) { residual[free] += local_residual[side]; }
      }
   }
   return residual;
}

double SkeletonResidualJacobianError(const DiscreteSystem &system)
{
   const CondensedSystem condensed = Condense(system);
   const int size = condensed.rhs.Size();
   mfem::Vector trace(size);
   for (int i = 0; i < size; ++i)
   {
      trace[i] = 0.35 + 0.017 * static_cast<double>(i + 1);
   }
   mfem::DenseMatrix finite_difference(size);
   const double step = 1.0e-5;
   for (int column = 0; column < size; ++column)
   {
      mfem::Vector plus(trace);
      mfem::Vector minus(trace);
      plus[column] += step;
      minus[column] -= step;
      const mfem::Vector residual_plus =
         EvaluateSkeletonResidual(system, plus);
      const mfem::Vector residual_minus =
         EvaluateSkeletonResidual(system, minus);
      for (int row = 0; row < size; ++row)
      {
         finite_difference(row, column) =
            (residual_plus[row] - residual_minus[row]) / (2.0 * step);
      }
   }

   mfem::Vector direct = EvaluateSkeletonResidual(system, trace);
   mfem::Vector assembled(size);
   condensed.matrix.Mult(trace, assembled);
   assembled -= condensed.rhs;
   return std::max(MaxDifference(condensed.matrix, finite_difference),
                   MaxDifference(direct, assembled));
}

double SchurActionError(const DiscreteSystem &system)
{
   const CondensedSystem condensed = Condense(system);
   mfem::Vector input(condensed.rhs.Size());
   for (int i = 0; i < input.Size(); ++i)
   {
      input[i] = 0.2 + 0.05 * static_cast<double>(i);
   }
   mfem::Vector assembled(input.Size());
   condensed.matrix.Mult(input, assembled);
   mfem::Vector action(input.Size());
   action = 0.0;
   for (const LocalSystem &local : system.local)
   {
      mfem::Vector local_trace(2);
      for (int side = 0; side < 2; ++side)
      {
         const int free = system.face_to_free[local.faces[side]];
         local_trace[side] = free >= 0 ? input[free] : 0.0;
      }
      mfem::Vector btrace(local.A.Height());
      local.B.Mult(local_trace, btrace);
      mfem::Vector response = Solve(local.A, btrace);
      response *= -1.0;
      mfem::Vector local_action(2);
      local.C.Mult(response, local_action);
      mfem::Vector dtrace(2);
      local.D.Mult(local_trace, dtrace);
      local_action += dtrace;
      for (int side = 0; side < 2; ++side)
      {
         const int free = system.face_to_free[local.faces[side]];
         if (free >= 0) { action[free] += local_action[side]; }
      }
   }
   return MaxDifference(assembled, action);
}

double IndependentMfemOperatorError(const ScalarProblem problem)
{
   constexpr int order = 3;
   const DiscreteSystem system =
      AssembleSystem(problem, MeshGeometry::curved, order, 2);
   mfem::Mesh mesh = MakeMesh(2, MeshGeometry::curved, order);
   mfem::L2_FECollection collection(order, 1,
                                    mfem::BasisType::GaussLegendre);
   mfem::FiniteElementSpace space(&mesh, &collection);
   const mfem::FiniteElement &fe = *space.GetFE(0);
   mfem::ElementTransformation &transformation =
      *mesh.GetElementTransformation(0);
   const mfem::IntegrationRule &rule =
      mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2 * order + 8);
   const LocalSystem &local = system.local.front();
   double error = 0.0;

   mfem::FunctionCoefficient source_coefficient(
      problem == ScalarProblem::poisson ? PoissonSourceCoefficient
                                        : AdvectionSourceCoefficient);
   mfem::DomainLFIntegrator source_integrator(source_coefficient, &rule);
   mfem::Vector mfem_rhs;
   source_integrator.AssembleRHSElementVect(fe, transformation, mfem_rhs);
   const int rhs_offset =
      problem == ScalarProblem::poisson ? local.scalar_dofs : 0;
   for (int i = 0; i < local.scalar_dofs; ++i)
   {
      error = std::max(
         error, std::abs(local.rhs[rhs_offset + i] - mfem_rhs[i]));
   }

   if (problem == ScalarProblem::poisson)
   {
      mfem::MassIntegrator mass_integrator(&rule);
      mfem::DenseMatrix mfem_mass;
      mass_integrator.AssembleElementMatrix(fe, transformation, mfem_mass);
      for (int i = 0; i < local.scalar_dofs; ++i)
      {
         for (int j = 0; j < local.scalar_dofs; ++j)
         {
            error = std::max(
               error, std::abs(local.A(i, j) - mfem_mass(i, j)));
         }
      }
      mfem::MixedScalarWeakDerivativeIntegrator weak_derivative;
      weak_derivative.SetIntRule(&rule);
      mfem::DenseMatrix mfem_weak_derivative;
      weak_derivative.AssembleElementMatrix2(
         fe, fe, transformation, mfem_weak_derivative);
      for (int i = 0; i < local.scalar_dofs; ++i)
      {
         for (int j = 0; j < local.scalar_dofs; ++j)
         {
            error = std::max(
               error,
               std::abs(local.A(i, local.scalar_dofs + j) -
                        mfem_weak_derivative(i, j)));
         }
      }
   }
   else
   {
      mfem::Vector velocity(1);
      velocity[0] = beta;
      mfem::VectorConstantCoefficient velocity_coefficient(velocity);
      mfem::ConvectionIntegrator convection_integrator(velocity_coefficient);
      mfem::DenseMatrix mfem_convection;
      convection_integrator.AssembleElementMatrix(
         fe, transformation, mfem_convection);

      mfem::DenseMatrix manual_volume(local.A);
      std::array<mfem::Vector, 2> face_shape = {
         mfem::Vector(local.scalar_dofs), mfem::Vector(local.scalar_dofs)};
      mfem::IntegrationPoint endpoint;
      endpoint.x = 0.0;
      fe.CalcShape(endpoint, face_shape[0]);
      endpoint.x = 1.0;
      fe.CalcShape(endpoint, face_shape[1]);
      const double h =
         std::abs(system.face_coordinate[local.faces[1]] -
                  system.face_coordinate[local.faces[0]]);
      for (int side = 0; side < 2; ++side)
      {
         const double normal = side == 0 ? -1.0 : 1.0;
         const double tau = FaceTau(problem, order, h, normal);
         for (int i = 0; i < local.scalar_dofs; ++i)
         {
            for (int j = 0; j < local.scalar_dofs; ++j)
            {
               manual_volume(i, j) -=
                  tau * face_shape[side][i] * face_shape[side][j];
            }
         }
      }
      for (int i = 0; i < local.scalar_dofs; ++i)
      {
         for (int j = 0; j < local.scalar_dofs; ++j)
         {
            error = std::max(
               error,
               std::abs(manual_volume(i, j) + mfem_convection(j, i)));
         }
      }
   }
   return error;
}

int AbsoluteDof(const int dof)
{
   return dof < 0 ? -1 - dof : dof;
}

double DofSign(const int dof)
{
   return dof < 0 ? -1.0 : 1.0;
}

double OrientedSkeletonCondensationError(const int order)
{
   mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
      2, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false);
   mfem::L2_FECollection interior_collection(
      order, 2, mfem::BasisType::GaussLegendre, mfem::FiniteElement::VALUE);
   mfem::FiniteElementSpace interior_space(&mesh, &interior_collection);
   mfem::DG_Interface_FECollection trace_collection(
      order, 2, mfem::FiniteElement::VALUE, mfem::BasisType::GaussLegendre);
   mfem::FiniteElementSpace trace_space(&mesh, &trace_collection);
   mfem::DenseMatrix assembled(trace_space.GetNDofs());
   mfem::DenseMatrix assembled_reversed(trace_space.GetNDofs());
   assembled = 0.0;
   assembled_reversed = 0.0;

   int shared_face = -1;
   for (int face = 0; face < mesh.GetNumFaces(); ++face)
   {
      if (mesh.FaceIsInterior(face))
      {
         if (shared_face >= 0)
         {
            return std::numeric_limits<double>::infinity();
         }
         shared_face = face;
      }
   }
   if (shared_face < 0) { return std::numeric_limits<double>::infinity(); }

   std::array<int, 2> shared_orientation = {0, 0};
   int shared_orientation_count = 0;
   const mfem::IntegrationRule &face_rule =
      mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2 * order + 8);
   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      const mfem::FiniteElement &interior_fe = *interior_space.GetFE(element);
      mfem::ElementTransformation &element_transformation =
         *mesh.GetElementTransformation(element);
      mfem::MassIntegrator mass_integrator;
      mfem::DiffusionIntegrator diffusion_integrator;
      mfem::DenseMatrix interior_matrix;
      mfem::DenseMatrix diffusion_matrix;
      mass_integrator.AssembleElementMatrix(
         interior_fe, element_transformation, interior_matrix);
      diffusion_integrator.AssembleElementMatrix(
         interior_fe, element_transformation, diffusion_matrix);
      interior_matrix += diffusion_matrix;

      mfem::Array<int> faces;
      mfem::Array<int> orientations;
      mesh.GetElementEdges(element, faces, orientations);
      const int face_dofs = order + 1;
      const int local_trace_size = faces.Size() * face_dofs;
      mfem::DenseMatrix coupling(interior_fe.GetDof(), local_trace_size);
      mfem::DenseMatrix coupling_reversed(
         interior_fe.GetDof(), local_trace_size);
      mfem::DenseMatrix trace_matrix(local_trace_size);
      mfem::DenseMatrix trace_matrix_reversed(local_trace_size);
      coupling = 0.0;
      coupling_reversed = 0.0;
      trace_matrix = 0.0;
      trace_matrix_reversed = 0.0;
      mfem::Array<int> global_trace_dofs(local_trace_size);

      for (int local_face = 0; local_face < faces.Size(); ++local_face)
      {
         const int face = faces[local_face];
         mfem::Array<int> face_global_dofs;
         trace_space.GetFaceDofs(face, face_global_dofs);
         if (face_global_dofs.Size() != face_dofs)
         {
            return std::numeric_limits<double>::infinity();
         }
         for (int i = 0; i < face_dofs; ++i)
         {
            global_trace_dofs[local_face * face_dofs + i] =
               face_global_dofs[i];
         }
         if (face == shared_face)
         {
            shared_orientation[shared_orientation_count++] =
               orientations[local_face];
         }

         mfem::FaceElementTransformations *face_transformation =
            mesh.GetFaceElementTransformations(face);
         if (!face_transformation)
         {
            return std::numeric_limits<double>::infinity();
         }
         const bool first = face_transformation->Elem1No == element;
         if (!first && face_transformation->Elem2No != element)
         {
            return std::numeric_limits<double>::infinity();
         }
         const mfem::FiniteElement &trace_fe =
            *trace_space.GetFaceElement(face);
         mfem::Vector interior_shape(interior_fe.GetDof());
         mfem::Vector trace_shape(face_dofs);
         mfem::Vector reversed_shape(face_dofs);
         for (int q = 0; q < face_rule.GetNPoints(); ++q)
         {
            const mfem::IntegrationPoint &face_point =
               face_rule.IntPoint(q);
            mfem::IntegrationPoint element_point;
            if (first)
            {
               face_transformation->Loc1.Transform(
                  face_point, element_point);
            }
            else
            {
               face_transformation->Loc2.Transform(
                  face_point, element_point);
            }
            interior_fe.CalcShape(element_point, interior_shape);
            trace_fe.CalcShape(face_point, trace_shape);
            if (face == shared_face)
            {
               mfem::IntegrationPoint reversed_point(face_point);
               reversed_point.x = 1.0 - face_point.x;
               trace_fe.CalcShape(reversed_point, reversed_shape);
            }
            else
            {
               reversed_shape = trace_shape;
            }
            face_transformation->Face->SetIntPoint(&face_point);
            const double weight =
               face_point.weight * face_transformation->Face->Weight();
            const int offset = local_face * face_dofs;
            for (int i = 0; i < interior_fe.GetDof(); ++i)
            {
               for (int j = 0; j < face_dofs; ++j)
               {
                  coupling(i, offset + j) -=
                     weight * interior_shape[i] * trace_shape[j];
                  coupling_reversed(i, offset + j) -=
                     weight * interior_shape[i] * reversed_shape[j];
               }
            }
            for (int i = 0; i < face_dofs; ++i)
            {
               for (int j = 0; j < face_dofs; ++j)
               {
                  trace_matrix(offset + i, offset + j) +=
                     2.0 * weight * trace_shape[i] * trace_shape[j];
                  trace_matrix_reversed(offset + i, offset + j) +=
                     2.0 * weight * reversed_shape[i] * reversed_shape[j];
               }
            }
         }
      }

      const mfem::DenseMatrix inverse_coupling =
         ApplyInverse(interior_matrix, coupling);
      const mfem::DenseMatrix inverse_coupling_reversed =
         ApplyInverse(interior_matrix, coupling_reversed);
      mfem::DenseMatrix schur(trace_matrix);
      mfem::DenseMatrix schur_reversed(trace_matrix_reversed);
      for (int i = 0; i < local_trace_size; ++i)
      {
         for (int j = 0; j < local_trace_size; ++j)
         {
            for (int k = 0; k < interior_fe.GetDof(); ++k)
            {
               schur(i, j) -=
                  coupling(k, i) * inverse_coupling(k, j);
               schur_reversed(i, j) -=
                  coupling_reversed(k, i) *
                  inverse_coupling_reversed(k, j);
            }
            const int global_i = AbsoluteDof(global_trace_dofs[i]);
            const int global_j = AbsoluteDof(global_trace_dofs[j]);
            const double sign =
               DofSign(global_trace_dofs[i]) * DofSign(global_trace_dofs[j]);
            assembled(global_i, global_j) += sign * schur(i, j);
            assembled_reversed(global_i, global_j) +=
               sign * schur_reversed(i, j);
         }
      }
   }

   if (shared_orientation_count != 2 ||
       shared_orientation[0] == shared_orientation[1])
   {
      return std::numeric_limits<double>::infinity();
   }
   mfem::Array<int> shared_dofs;
   trace_space.GetFaceDofs(shared_face, shared_dofs);
   const int *permutation =
      trace_collection.DofOrderForOrientation(
         mfem::Geometry::SEGMENT, -1);
   std::vector<int> global_permutation(trace_space.GetNDofs());
   std::iota(global_permutation.begin(), global_permutation.end(), 0);
   for (int i = 0; i < shared_dofs.Size(); ++i)
   {
      const int mapped =
         permutation[i] < 0 ? -1 - permutation[i] : permutation[i];
      global_permutation[AbsoluteDof(shared_dofs[i])] =
         AbsoluteDof(shared_dofs[mapped]);
   }
   double error = 0.0;
   for (int i = 0; i < assembled.Height(); ++i)
   {
      for (int j = 0; j < assembled.Width(); ++j)
      {
         error = std::max(
            error,
            std::abs(assembled(i, j) -
                     assembled_reversed(global_permutation[i],
                                        global_permutation[j])));
      }
   }
   return error;
}

double OrientationShapeError(const int order)
{
   mfem::DG_Interface_FECollection collection(
      order, 2, mfem::FiniteElement::VALUE, mfem::BasisType::GaussLegendre);
   const mfem::FiniteElement &fe =
      *collection.FiniteElementForGeometry(mfem::Geometry::SEGMENT);
   const int *permutation =
      collection.DofOrderForOrientation(mfem::Geometry::SEGMENT, -1);
   double error = 0.0;
   for (double point : {0.07, 0.23, 0.61, 0.91})
   {
      mfem::IntegrationPoint forward_point;
      mfem::IntegrationPoint reverse_point;
      forward_point.x = point;
      reverse_point.x = 1.0 - point;
      mfem::Vector forward(fe.GetDof());
      mfem::Vector reverse(fe.GetDof());
      fe.CalcShape(forward_point, forward);
      fe.CalcShape(reverse_point, reverse);
      for (int i = 0; i < fe.GetDof(); ++i)
      {
         const int mapped = permutation[i] < 0 ? -1 - permutation[i]
                                                : permutation[i];
         const double sign = permutation[i] < 0 ? -1.0 : 1.0;
         error = std::max(error,
                          std::abs(forward[i] - sign * reverse[mapped]));
      }
   }
   return error;
}

double OrientationSchurError(const DiscreteSystem &system)
{
   const LocalSystem &local = system.local.front();
   const int dofs = local.scalar_dofs;
   mfem::L2_FECollection collection(system.order, 1,
                                    mfem::BasisType::GaussLegendre);
   const int *ordering =
      collection.DofOrderForOrientation(mfem::Geometry::SEGMENT, -1);
   std::vector<int> permutation(local.A.Height());
   std::vector<double> sign(local.A.Height(), 1.0);
   const int blocks = system.problem == ScalarProblem::poisson ? 2 : 1;
   for (int block = 0; block < blocks; ++block)
   {
      for (int i = 0; i < dofs; ++i)
      {
         const int entry = ordering[i];
         permutation[block * dofs + i] =
            block * dofs + (entry < 0 ? -1 - entry : entry);
         sign[block * dofs + i] = entry < 0 ? -1.0 : 1.0;
      }
   }
   mfem::DenseMatrix reversed_a(local.A.Height());
   mfem::DenseMatrix reversed_b(local.A.Height(), 2);
   mfem::DenseMatrix reversed_c(2, local.A.Width());
   mfem::DenseMatrix reversed_d(2);
   for (int i = 0; i < local.A.Height(); ++i)
   {
      for (int j = 0; j < local.A.Width(); ++j)
      {
         reversed_a(i, j) = sign[i] * sign[j] *
                            local.A(permutation[i], permutation[j]);
      }
      for (int side = 0; side < 2; ++side)
      {
         reversed_b(i, side) =
            sign[i] * local.B(permutation[i], 1 - side);
         reversed_c(side, i) =
            sign[i] * local.C(1 - side, permutation[i]);
      }
   }
   for (int i = 0; i < 2; ++i)
   {
      for (int j = 0; j < 2; ++j)
      {
         reversed_d(i, j) = local.D(1 - i, 1 - j);
      }
   }
   const mfem::DenseMatrix inverse_b = ApplyInverse(local.A, local.B);
   const mfem::DenseMatrix reversed_inverse_b =
      ApplyInverse(reversed_a, reversed_b);
   mfem::DenseMatrix schur(2);
   mfem::DenseMatrix reversed_schur(2);
   for (int i = 0; i < 2; ++i)
   {
      for (int j = 0; j < 2; ++j)
      {
         schur(i, j) = local.D(i, j);
         reversed_schur(i, j) = reversed_d(i, j);
         for (int k = 0; k < local.A.Height(); ++k)
         {
            schur(i, j) -= local.C(i, k) * inverse_b(k, j);
            reversed_schur(i, j) -=
               reversed_c(i, k) * reversed_inverse_b(k, j);
         }
      }
   }
   double error = 0.0;
   for (int i = 0; i < 2; ++i)
   {
      for (int j = 0; j < 2; ++j)
      {
         error = std::max(
            error,
            std::abs(schur(i, j) - reversed_schur(1 - i, 1 - j)));
      }
   }
   return error;
}

std::vector<ConvergenceRow> RunConvergence(
   std::vector<Comparison> &checks,
   std::vector<ConvergenceGate> &gates)
{
   std::vector<ConvergenceRow> rows;
   constexpr std::array<int, 4> levels = {4, 8, 16, 32};
   for (ScalarProblem problem : {ScalarProblem::poisson,
                                 ScalarProblem::advection})
   {
      for (MeshGeometry geometry : {MeshGeometry::affine,
                                    MeshGeometry::curved})
      {
         for (int order = 1; order <= 4; ++order)
         {
            double previous_error = 0.0;
            double previous_h = 0.0;
            double final_rate = std::numeric_limits<double>::quiet_NaN();
            bool all_values_finite = true;
            bool errors_strictly_decrease = true;
            for (int elements : levels)
            {
               const DiscreteSystem system =
                  AssembleSystem(problem, geometry, order, elements);
               const Solution condensed = SolveCondensed(system);
               const Solution full = SolveFull(system);
               const double error = L2Error(system, condensed);
               const double recovery =
                  SolutionDifference(system, condensed, full);
               const double conservation =
                  ConservationError(system, condensed);
               double rate = 0.0;
               if (previous_error > 0.0)
               {
                  rate = std::log(previous_error / error) /
                         std::log(previous_h / system.h_max);
                  final_rate = rate;
                  errors_strictly_decrease =
                     errors_strictly_decrease && error < previous_error;
               }
               all_values_finite =
                  all_values_finite && std::isfinite(system.h_max) &&
                  std::isfinite(error) && std::isfinite(rate) &&
                  std::isfinite(recovery) && std::isfinite(conservation);
               rows.push_back({problem, geometry, order, elements,
                               system.h_max, error, rate, recovery,
                               conservation});
               previous_error = error;
               previous_h = system.h_max;
            }
            const double required_rate = static_cast<double>(order) + 0.65;
            const double margin = final_rate - required_rate;
            const bool pass = all_values_finite &&
                              errors_strictly_decrease &&
                              std::isfinite(final_rate) && margin >= 0.0;
            gates.push_back({problem, geometry, order, final_rate,
                             required_rate, margin, all_values_finite,
                             errors_strictly_decrease, pass});
            AddComparison(checks,
                          std::string("convergence/") + ToString(problem) +
                             "/" + ToString(geometry) + "/p" +
                             std::to_string(order),
                          pass ? 0.0 : std::numeric_limits<double>::infinity(),
                          0.0);
         }
      }
   }
   return rows;
}

bool AllPass(const std::vector<Comparison> &checks)
{
   return std::all_of(checks.begin(), checks.end(),
                      [](const Comparison &check) { return check.pass; });
}

}  // namespace

const char *ToString(const ScalarProblem problem)
{
   return problem == ScalarProblem::poisson ? "poisson" : "advection";
}

const char *ToString(const MeshGeometry geometry)
{
   return geometry == MeshGeometry::affine ? "affine" : "curved";
}

SerialEvidence RunSerialOracle()
{
   SerialEvidence evidence;
   for (ScalarProblem problem : {ScalarProblem::poisson,
                                 ScalarProblem::advection})
   {
      const DiscreteSystem fixture =
         AssembleSystem(problem, MeshGeometry::curved, 3, 5);
      const CondensedSystem condensed = Condense(fixture);
      const Solution condensed_solution = SolveCondensed(fixture);
      const Solution full_solution = SolveFull(fixture);
      AddComparison(evidence.comparisons,
                    std::string("residual_jacobian/") + ToString(problem),
                    ResidualJacobianError(fixture), jacobian_tolerance);
      AddComparison(evidence.comparisons,
                    std::string("skeleton_residual_jacobian/") +
                       ToString(problem),
                    SkeletonResidualJacobianError(fixture),
                    jacobian_tolerance);
      AddComparison(evidence.comparisons,
                    std::string("mfem_element_operator/") +
                       ToString(problem),
                    IndependentMfemOperatorError(problem),
                    comparison_tolerance);
      AddComparison(evidence.comparisons,
                    std::string("constant_state/") + ToString(problem),
                    ConstantStateError(fixture), comparison_tolerance);
      AddComparison(evidence.comparisons,
                    std::string("full_assembly_vs_condensation/") + ToString(problem),
                    SolutionDifference(fixture, condensed_solution,
                                       full_solution),
                    comparison_tolerance);
      AddComparison(evidence.comparisons,
                    std::string("trace_solve/") + ToString(problem),
                    MaxDifference(condensed_solution.trace,
                                  full_solution.trace),
                    comparison_tolerance);
      AddComparison(evidence.comparisons,
                    std::string("back_substitution_recovery/") +
                       ToString(problem),
                    [&]() {
                       double error = 0.0;
                       for (int element = 0; element < fixture.elements; ++element)
                       {
                          error = std::max(
                             error, MaxDifference(
                                       condensed_solution.interior[element],
                                       full_solution.interior[element]));
                       }
                       return error;
                    }(),
                    comparison_tolerance);
      AddComparison(evidence.comparisons,
                    std::string("schur_action/") + ToString(problem),
                    SchurActionError(fixture), comparison_tolerance);
      AddComparison(evidence.comparisons,
                    std::string("reduced_rhs_residual/") + ToString(problem),
                    MaxAbs([&]() {
                       const int free_dofs = condensed.rhs.Size();
                       mfem::Vector free_trace(free_dofs);
                       for (int face = 0; face <
                                          static_cast<int>(fixture.face_to_free.size());
                            ++face)
                       {
                          const int free = fixture.face_to_free[face];
                          if (free >= 0)
                          {
                             free_trace[free] = condensed_solution.trace[face];
                          }
                       }
                       mfem::Vector residual(free_dofs);
                       condensed.matrix.Mult(free_trace, residual);
                       residual -= condensed.rhs;
                       return residual;
                    }()),
                    solve_tolerance);
      AddComparison(evidence.comparisons,
                    std::string("conservation/") + ToString(problem),
                    ConservationError(fixture, condensed_solution),
                    comparison_tolerance);
      AddComparison(evidence.comparisons,
                    std::string("essential_boundary/") + ToString(problem),
                    EssentialBoundaryError(fixture, condensed_solution), 0.0);
      AddComparison(evidence.comparisons,
                    std::string("orientation_schur/") + ToString(problem),
                    OrientationSchurError(fixture), comparison_tolerance);
   }
   for (int order = 1; order <= 4; ++order)
   {
      AddComparison(evidence.comparisons,
                    "mfem_trace_orientation/p" + std::to_string(order),
                    OrientationShapeError(order), comparison_tolerance);
      AddComparison(evidence.comparisons,
                    "oriented_skeleton_condensation/p" +
                       std::to_string(order),
                    OrientedSkeletonCondensationError(order),
                    comparison_tolerance);
   }
   evidence.convergence = RunConvergence(evidence.comparisons,
                                         evidence.convergence_gates);
   double maximum_recovery = 0.0;
   double maximum_conservation = 0.0;
   for (const ConvergenceRow &row : evidence.convergence)
   {
      maximum_recovery = std::max(maximum_recovery, row.recovery_error);
      maximum_conservation =
         std::max(maximum_conservation, row.conservation_error);
   }
   AddComparison(evidence.comparisons, "p1_p4_exact_local_recovery",
                 maximum_recovery, recovery_tolerance);
   AddComparison(evidence.comparisons, "p1_p4_conservation",
                 maximum_conservation, comparison_tolerance);
   evidence.pass = AllPass(evidence.comparisons) &&
                   std::all_of(evidence.convergence_gates.begin(),
                               evidence.convergence_gates.end(),
                               [](const ConvergenceGate &gate)
                               { return gate.pass; });
   return evidence;
}

MpiEvidence RunMpiOracle(MPI_Comm comm)
{
   MpiEvidence evidence;
   int rank = 0;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &evidence.ranks);
   evidence.global_elements = 8;
   if (evidence.ranks != 1 && evidence.ranks != 2)
   {
      throw std::runtime_error("MPI oracle is frozen for one or two ranks");
   }

   mfem::Mesh serial_mesh =
      MakeMesh(evidence.global_elements, MeshGeometry::curved, 3);
   std::vector<int> partition(evidence.global_elements);
   for (int element = 0; element < evidence.global_elements; ++element)
   {
      partition[element] =
         std::min(evidence.ranks - 1,
                  element * evidence.ranks / evidence.global_elements);
   }
   mfem::ParMesh parallel_mesh(comm, serial_mesh, partition.data());
   const int local_elements = parallel_mesh.GetNE();
   MPI_Allreduce(&local_elements, &evidence.mfem_partition_elements, 1,
                 MPI_INT, MPI_SUM, comm);

   mfem::H1_Trace_FECollection trace_collection(1, 1);
   mfem::ParFiniteElementSpace trace_space(&parallel_mesh, &trace_collection);
   const int local_trace_dofs = trace_space.GetVSize();
   const int local_true_dofs = trace_space.GetTrueVSize();
   std::vector<int> true_dof_counts(evidence.ranks);
   MPI_Allgather(&local_true_dofs, 1, MPI_INT, true_dof_counts.data(), 1,
                 MPI_INT, comm);
   std::vector<long long> true_dof_offsets(evidence.ranks + 1, 0);
   for (int process = 0; process < evidence.ranks; ++process)
   {
      true_dof_offsets[process + 1] =
         true_dof_offsets[process] + true_dof_counts[process];
   }
   const long long global_true_dofs = true_dof_offsets.back();

   std::vector<int> local_map_rank(local_trace_dofs, rank);
   std::vector<int> local_map_dof(local_trace_dofs);
   std::vector<long long> local_map_global(local_trace_dofs);
   std::vector<int> local_map_owner(local_trace_dofs);
   std::vector<int> local_map_owned(local_trace_dofs);
   int local_owned = 0;
   int local_ghost = 0;
   int local_map_errors = 0;
   for (int local_dof = 0; local_dof < local_trace_dofs; ++local_dof)
   {
      local_map_dof[local_dof] = local_dof;
      const long long global_dof =
         static_cast<long long>(trace_space.GetGlobalTDofNumber(local_dof));
      local_map_global[local_dof] = global_dof;
      int owner = 0;
      while (owner + 1 < evidence.ranks &&
             global_dof >= true_dof_offsets[owner + 1])
      {
         ++owner;
      }
      local_map_owner[local_dof] = owner;
      const int local_true = trace_space.GetLocalTDofNumber(local_dof);
      const bool owned = local_true >= 0;
      local_map_owned[local_dof] = owned ? 1 : 0;
      local_owned += owned ? 1 : 0;
      local_ghost += owned ? 0 : 1;
      if (global_dof < 0 || global_dof >= global_true_dofs ||
          owned != (owner == rank) ||
          (owned && global_dof != true_dof_offsets[rank] + local_true))
      {
         ++local_map_errors;
      }
   }
   MPI_Allreduce(&local_owned, &evidence.owned_trace_dofs, 1, MPI_INT,
                 MPI_SUM, comm);
   MPI_Allreduce(&local_ghost, &evidence.ghost_trace_dofs, 1, MPI_INT,
                 MPI_SUM, comm);
   int map_errors = 0;
   MPI_Allreduce(&local_map_errors, &map_errors, 1, MPI_INT, MPI_SUM, comm);

   const int local_shared_faces = parallel_mesh.GetNSharedFaces();
   int shared_face_records = 0;
   MPI_Allreduce(&local_shared_faces, &shared_face_records, 1, MPI_INT,
                 MPI_SUM, comm);
   evidence.shared_faces = shared_face_records / 2;

   std::vector<int> map_counts(evidence.ranks);
   MPI_Gather(&local_trace_dofs, 1, MPI_INT, map_counts.data(), 1, MPI_INT, 0,
              comm);
   std::vector<int> map_displacements(evidence.ranks, 0);
   int total_map_entries = 0;
   if (rank == 0)
   {
      for (int process = 0; process < evidence.ranks; ++process)
      {
         map_displacements[process] = total_map_entries;
         total_map_entries += map_counts[process];
      }
      evidence.map_rank.resize(total_map_entries);
      evidence.map_local_dof.resize(total_map_entries);
      evidence.map_global_true_dof.resize(total_map_entries);
      evidence.map_owner_rank.resize(total_map_entries);
      evidence.map_is_owned.resize(total_map_entries);
   }
   MPI_Gatherv(local_map_rank.data(), local_trace_dofs, MPI_INT,
               rank == 0 ? evidence.map_rank.data() : nullptr,
               map_counts.data(), map_displacements.data(), MPI_INT, 0, comm);
   MPI_Gatherv(local_map_dof.data(), local_trace_dofs, MPI_INT,
               rank == 0 ? evidence.map_local_dof.data() : nullptr,
               map_counts.data(), map_displacements.data(), MPI_INT, 0, comm);
   MPI_Gatherv(local_map_global.data(), local_trace_dofs, MPI_LONG_LONG_INT,
               rank == 0 ? evidence.map_global_true_dof.data() : nullptr,
               map_counts.data(), map_displacements.data(), MPI_LONG_LONG_INT,
               0, comm);
   MPI_Gatherv(local_map_owner.data(), local_trace_dofs, MPI_INT,
               rank == 0 ? evidence.map_owner_rank.data() : nullptr,
               map_counts.data(), map_displacements.data(), MPI_INT, 0, comm);
   MPI_Gatherv(local_map_owned.data(), local_trace_dofs, MPI_INT,
               rank == 0 ? evidence.map_is_owned.data() : nullptr,
               map_counts.data(), map_displacements.data(), MPI_INT, 0, comm);

   mfem::ParGridFunction expected_trace(&trace_space);
   for (int local_dof = 0; local_dof < local_trace_dofs; ++local_dof)
   {
      expected_trace[local_dof] =
         2.0 + 0.125 * static_cast<double>(local_map_global[local_dof]);
   }
   mfem::ParGridFunction owner_seed(&trace_space);
   owner_seed = -12345.0;
   for (int local_dof = 0; local_dof < local_trace_dofs; ++local_dof)
   {
      if (trace_space.GetLocalTDofNumber(local_dof) >= 0)
      {
         owner_seed[local_dof] = expected_trace[local_dof];
      }
   }
   std::unique_ptr<mfem::HypreParVector> true_trace(
      owner_seed.ParallelProject());
   mfem::ParGridFunction exchanged_trace(&trace_space);
   exchanged_trace = -6789.0;
   exchanged_trace.Distribute(*true_trace);
   double local_exchange_error = 0.0;
   for (int local_dof = 0; local_dof < local_trace_dofs; ++local_dof)
   {
      local_exchange_error =
         std::max(local_exchange_error,
                  std::abs(exchanged_trace[local_dof] -
                           expected_trace[local_dof]));
   }
   double exchange_error = 0.0;
   MPI_Allreduce(&local_exchange_error, &exchange_error, 1, MPI_DOUBLE,
                 MPI_MAX, comm);

   mfem::L2_FECollection interior_collection(
      3, 1, mfem::BasisType::GaussLegendre);
   mfem::ParFiniteElementSpace interior_space(&parallel_mesh,
                                               &interior_collection);
   mfem::SparseMatrix local_schur(local_trace_dofs);
   mfem::Vector local_reduced_rhs(local_trace_dofs);
   local_reduced_rhs = 0.0;
   for (int element = 0; element < local_elements; ++element)
   {
      mfem::Array<int> trace_dofs;
      trace_space.GetElementDofs(element, trace_dofs);
      if (trace_dofs.Size() != 2)
      {
         throw std::runtime_error(
            "one-dimensional trace element must have two endpoint dofs");
      }
      const LocalSystem local = AssembleLocal(
         ScalarProblem::poisson, 3, parallel_mesh, interior_space, element,
         {AbsoluteDof(trace_dofs[0]), AbsoluteDof(trace_dofs[1])});
      const mfem::DenseMatrix inverse_b = ApplyInverse(local.A, local.B);
      const mfem::Vector particular = Solve(local.A, local.rhs);
      mfem::DenseMatrix schur(2);
      mfem::Vector reduced_rhs(2);
      for (int row = 0; row < 2; ++row)
      {
         reduced_rhs[row] = 0.0;
         for (int interior = 0; interior < local.C.Width(); ++interior)
         {
            reduced_rhs[row] -= local.C(row, interior) * particular[interior];
         }
         for (int column = 0; column < 2; ++column)
         {
            schur(row, column) = local.D(row, column);
            for (int interior = 0; interior < local.C.Width(); ++interior)
            {
               schur(row, column) -=
                  local.C(row, interior) * inverse_b(interior, column);
            }
         }
         local_reduced_rhs[AbsoluteDof(trace_dofs[row])] +=
            DofSign(trace_dofs[row]) * reduced_rhs[row];
      }
      local_schur.AddSubMatrix(trace_dofs, trace_dofs, schur);
   }
   local_schur.Finalize();

   mfem::ParBilinearForm schur_form(&trace_space);
   std::unique_ptr<mfem::HypreParMatrix> true_schur(
      schur_form.ParallelAssemble(&local_schur));
   mfem::Vector true_reduced_rhs(local_true_dofs);
   trace_space.Dof_TrueDof_Matrix()->MultTranspose(local_reduced_rhs,
                                                   true_reduced_rhs);
   mfem::Vector operator_residual(local_true_dofs);
   true_schur->Mult(*true_trace, operator_residual);
   operator_residual -= true_reduced_rhs;

   mfem::Vector local_direct_residual(local_trace_dofs);
   local_direct_residual = 0.0;
   for (int element = 0; element < local_elements; ++element)
   {
      mfem::Array<int> trace_dofs;
      trace_space.GetElementDofs(element, trace_dofs);
      const LocalSystem local = AssembleLocal(
         ScalarProblem::poisson, 3, parallel_mesh, interior_space, element,
         {AbsoluteDof(trace_dofs[0]), AbsoluteDof(trace_dofs[1])});
      mfem::Vector local_trace(2);
      for (int side = 0; side < 2; ++side)
      {
         local_trace[side] = DofSign(trace_dofs[side]) *
                             exchanged_trace[AbsoluteDof(trace_dofs[side])];
      }
      mfem::Vector interior_rhs(local.rhs);
      mfem::Vector trace_product(local.A.Height());
      local.B.Mult(local_trace, trace_product);
      interior_rhs -= trace_product;
      const mfem::Vector interior_solution = Solve(local.A, interior_rhs);
      mfem::Vector skeleton_residual(2);
      mfem::Vector trace_residual(2);
      local.C.Mult(interior_solution, skeleton_residual);
      local.D.Mult(local_trace, trace_residual);
      skeleton_residual += trace_residual;
      for (int side = 0; side < 2; ++side)
      {
         local_direct_residual[AbsoluteDof(trace_dofs[side])] +=
            DofSign(trace_dofs[side]) * skeleton_residual[side];
      }
   }
   mfem::Vector true_direct_residual(local_true_dofs);
   trace_space.Dof_TrueDof_Matrix()->MultTranspose(local_direct_residual,
                                                   true_direct_residual);
   const double local_operator_error =
      MaxDifference(operator_residual, true_direct_residual);
   double operator_error = 0.0;
   MPI_Allreduce(&local_operator_error, &operator_error, 1, MPI_DOUBLE,
                 MPI_MAX, comm);

   AddComparison(evidence.comparisons, "mpi_partition_element_count",
                 std::abs(evidence.mfem_partition_elements -
                          evidence.global_elements),
                 0.0);
   AddComparison(evidence.comparisons, "mfem_parallel_trace_true_dofs",
                 std::abs(static_cast<double>(global_true_dofs -
                                              (evidence.global_elements + 1))),
                 0.0);
   AddComparison(evidence.comparisons, "mfem_true_dof_owner_map",
                 static_cast<double>(map_errors), 0.0);
   AddComparison(evidence.comparisons, "mfem_owned_true_dof_count",
                 std::abs(static_cast<double>(evidence.owned_trace_dofs) -
                          static_cast<double>(global_true_dofs)),
                 0.0);
   AddComparison(evidence.comparisons, "mfem_owner_ghost_presence",
                 evidence.ranks == 2 && evidence.ghost_trace_dofs == 0 ? 1.0
                                                                       : 0.0,
                 0.0);
   AddComparison(evidence.comparisons, "mfem_shared_trace_face_count",
                 std::abs(evidence.shared_faces - (evidence.ranks - 1)), 0.0);
   AddComparison(evidence.comparisons, "mfem_owner_ghost_exchange",
                 exchange_error, comparison_tolerance);
   AddComparison(evidence.comparisons, "mfem_parallel_schur_residual",
                 operator_error, comparison_tolerance);
   evidence.pass = AllPass(evidence.comparisons);
   return evidence;
}

}  // namespace dg::hdg
