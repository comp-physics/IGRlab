#pragma once
#include "mfem.hpp"

namespace mfem { namespace dgsem { namespace overintegration_internal {
// Private data-only implementation interface. The test uses these same maps
// and device applications, with an independent host FluxWithIGR weak oracle.
// interpolation(q,j)=phi_j(xi_q); weak_derivative(i,q+Q*d)=w_q*d_d phi_i.
// The projected flux must already be contravariant; no mass or face term is
// hidden in this contraction.
void BuildVolumeQuadratureMaps2D(const FiniteElement& fe,
                                 const IntegrationRule& rule,
                                 DenseMatrix& interpolation,
                                 DenseMatrix& weak_derivative);
void InterpolateVolumeState2DDevice(int ne, int dof,
                                    const DenseMatrix& interpolation,
                                    const DenseMatrix& states,
                                    const Vector& sigma_nodes,
                                    DenseMatrix& quadrature_states,
                                    Vector& quadrature_sigma);
void ApplyWeakVolumeProjection2DDevice(int ne, int dof,
                                       const DenseMatrix& weak_derivative,
                                       const DenseMatrix& quadrature_flux,
                                       DenseMatrix& weak_volume);
} } }
