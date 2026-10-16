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
  void Apply();
  bool Ready() const { return precision_ != Precision::Off; }
  Precision GetPrecision() const { return precision_; }
  mfem::DenseMatrix& Input() { return input_; }
  const mfem::DenseMatrix& Input() const { return input_; }
  const mfem::DenseMatrix& Output() const { return output_; }
  const mfem::DenseMatrix& Matrix() const { return matrix_; }

private:
  Precision precision_ = Precision::Off;
  std::string method_;
  mfem::DenseMatrix matrix_, input_, output_;
  std::uint64_t applications_ = 0;
};

} // namespace shared_reference
} // namespace igrlab
