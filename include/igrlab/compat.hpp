#pragma once

// IGRlab owns the application namespaces. The legacy alias keeps source clients
// compatible during the branding migration; serialized formats are unchanged.
namespace igrlab {}
namespace dg_igr = igrlab;

// MFEM extension helpers historically lived in mfem::dg_igr as well.
namespace mfem
{
namespace igrlab {}
namespace dg_igr = igrlab;
}
