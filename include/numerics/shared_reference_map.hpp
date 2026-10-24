#pragma once

#include "mfem.hpp"

#include <cstdint>
#include <string>

namespace igrlab
{
namespace shared_reference
{

// Experimental spatial-map precision, independent of IGR solver precision.
enum class Precision { Off, FP64, FP32, TF32 };
Precision RequestedPrecision();
const char* PrecisionName(Precision precision);

// A fixed column-major M x K map applied to genuinely distinct columns.
// Owners pack/unpack in FP64; this object retains all workspaces between calls.
// Reduced precision changes the spatial residual and requires PDE qualification.
class Map
{
public:
  Map() = default;
  Map(const Map&) = delete;
  Map& operator=(const Map&) = delete;
  ~Map();
  void Configure(const mfem::DenseMatrix& matrix, int columns,
                 Precision precision, const char* method);
  // Conventional DG only: factor an admitted affine Gauss-Legendre weak
  // divergence into its one-dimensional derivative/mass tensor lines.
  // Input/output keep the dense map's original local-node ordering.
  void ConfigureDGTensorLines(const mfem::DenseMatrix& matrix, int columns,
                            Precision precision, const mfem::FiniteElement& fe,
                            const mfem::DenseMatrix& adjugate,
                            bool conservative_fusion = false);
  void Apply();
  // DG's opt-in p5/FP32 conservative tail consumes raw FP64 flux and adds the
  // centered, moment-corrected result directly to the existing face residual.
  // This numerical apply owns no full-sized centered-input/output scratch.
  void ApplyDGConservativeFusion(const mfem::Vector& flux,
                                const mfem::Vector& column_sums,
                                const mfem::DenseMatrix& row_sums,
                                const mfem::Vector& mass_fractions,
                                const mfem::Array<int>& state_map,
                                mfem::Vector& residual);
  bool Ready() const { return precision_ != Precision::Off; }
  Precision GetPrecision() const { return precision_; }
  bool UsesDGTensorLines() const { return tensor_n1d_ != 0; }
  bool UsesDGConservativeFusion() const { return conservative_fusion_; }
  mfem::DenseMatrix& Input()
  {
    MFEM_VERIFY(!conservative_fusion_, "Fused DG map has no buffered input.");
    return input_;
  }
  const mfem::DenseMatrix& Input() const
  {
    MFEM_VERIFY(!conservative_fusion_, "Fused DG map has no buffered input.");
    return input_;
  }
  const mfem::DenseMatrix& Output() const
  {
    MFEM_VERIFY(!conservative_fusion_, "Fused DG map has no buffered output.");
    return output_;
  }
  const mfem::DenseMatrix& Matrix() const { return matrix_; }

private:
  void ConfigureStorage(const mfem::DenseMatrix& matrix, int columns,
                        Precision precision, const char* method,
                        bool buffered = true);
  void LogConfiguration() const;
  Precision precision_ = Precision::Off;
  std::string method_;
  mfem::DenseMatrix matrix_, input_, output_;
  int columns_ = 0;
  int tensor_n1d_ = 0;
  bool conservative_fusion_ = false;
  mfem::Array<int> tensor_lex_to_local_;
  mfem::Vector tensor_derivative_, tensor_weights_, tensor_adjugate_;
  std::uint64_t applications_ = 0;
};

} // namespace shared_reference
} // namespace igrlab
