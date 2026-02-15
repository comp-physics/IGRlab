#pragma once

#include "mfem.hpp"

/**
 * @file physics/euler/characteristics.hpp
 * @brief Euler characteristic matrices and helpers (2D).
 * @ingroup physics_euler
 */

namespace mfem
{

/**
 * @brief Euler right eigenvectors (2D) for x/y directions at mean state.
 */
void EulerRightEigen2D(const mfem::Vector& Ubar,
                       double c,
                       double H,
                       mfem::DenseMatrix& R_x,
                       mfem::DenseMatrix& R_y);

/**
 * @brief Euler left eigenvectors (2D) for x/y directions at mean state.
 */
void EulerLeftEigen2D(const mfem::Vector& Ubar,
                      double c,
                      double H,
                      mfem::DenseMatrix& L_x,
                      mfem::DenseMatrix& L_y);

/**
 * @brief Directional left/right eigenvectors using a linear combination of x/y blocks.
 *
 * Builds L_n ~= n_x L_x + n_y L_y and R_n ~= n_x R_x + n_y R_y.
 */
void EulerLeftRightInDirection(const mfem::Vector& Ubar,
                               double gamma,
                               double nx,
                               double ny,
                               mfem::DenseMatrix& L_n,
                               mfem::DenseMatrix& R_n);

/**
 * @brief Euler characteristic matrices along a normal for a rotated state.
 *
 * Assumes Ubar_rot is rotated so that x is the normal direction.
 */
void EulerCharNormalFromRotatedUbar(const mfem::Vector& Ubar_rot,
                                    double gamma,
                                    mfem::DenseMatrix& L_n,
                                    mfem::DenseMatrix& R_n);

} // namespace mfem
