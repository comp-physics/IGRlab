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
                            const mfem::DenseMatrix& adjugate);
  void Apply();
  bool Ready() const { return precision_ != Precision::Off; }
  Precision GetPrecision() const { return precision_; }
  bool UsesDGTensorLines() const { return tensor_n1d_ != 0; }
  mfem::DenseMatrix& Input() { return input_; }
  const mfem::DenseMatrix& Input() const { return input_; }
  const mfem::DenseMatrix& Output() const { return output_; }
  const mfem::DenseMatrix& Matrix() const { return matrix_; }

private:
  void ConfigureStorage(const mfem::DenseMatrix& matrix, int columns,
                        Precision precision, const char* method);
  void LogConfiguration() const;
  Precision precision_ = Precision::Off;
  std::string method_;
  mfem::DenseMatrix matrix_, input_, output_;
  int tensor_n1d_ = 0;
  mfem::Array<int> tensor_lex_to_local_;
  mfem::Vector tensor_derivative_, tensor_weights_, tensor_adjugate_;
  std::uint64_t applications_ = 0;
};

} // namespace shared_reference
} // namespace igrlab
