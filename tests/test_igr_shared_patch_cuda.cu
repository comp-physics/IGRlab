#include "igrlab/compat.hpp"
#include "../src/physics/igr/IGR_shared_patch_cuda.cuh"
#include "../src/physics/igr/IGR_device_krylov.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

namespace {
class ScopedEnvironment
{
public:
  ScopedEnvironment(const char* name, const char* value) : name_(name)
  {
    const char* old=std::getenv(name); existed_=old!=nullptr; if(old) old_=old;
    MFEM_VERIFY(setenv(name,value,1)==0,"Cannot configure shared patch fixture.");
  }
  ~ScopedEnvironment() { if(existed_) setenv(name_.c_str(),old_.c_str(),1); else unsetenv(name_.c_str()); }
private:
  std::string name_,old_; bool existed_;
};
double Relative(const mfem::Vector& actual,const mfem::Vector& expected)
{
  const double* a=actual.HostRead(); const double* b=expected.HostRead();
  long double error=0.0L,scale=0.0L;
  for(int i=0;i<actual.Size();++i)
  {
    MFEM_VERIFY(std::isfinite(a[i]) && std::isfinite(b[i]),"Nonfinite shared patch result.");
    const long double delta=static_cast<long double>(a[i])-b[i];
    error+=delta*delta; scale+=static_cast<long double>(b[i])*b[i];
  }
  return std::sqrt(static_cast<double>(scale>0.0L ? error/scale : error));
}
void CheckRealBatch()
{
  constexpr int m=36,k=180,ne=37,groups=2,n=m*ne;
  std::vector<int> offsets{0,18,37},inputs(k*ne),outputs(n);
  std::vector<double> matrices(groups*m*k),scales(ne);
  for(int group=0;group<groups;++group) for(int col=0;col<k;++col) for(int row=0;row<m;++row)
    matrices[(group*k+col)*m+row]=(row==col?1.3+group*.2:0.01*std::cos(.11*(col+3*row+group)));
  for(int col=0;col<ne;++col)
  {
    const int e=(7*col)%ne; scales[col]=.8+.01*col;
    for(int row=0;row<m;++row) outputs[col*m+row]=n-1-(e*m+row);
    for(int j=0;j<k;++j) inputs[col*k+j]=n-1-(((e+j/m)%ne)*m+j%m);
  }
  igrlab::igr_internal::SharedPatchBatch control,mixed;
  control.Configure(m,k,n,matrices,offsets,inputs,outputs,scales,false);
  mixed.Configure(m,k,n,matrices,offsets,inputs,outputs,scales,true);
  mfem::Vector x(n),coarse(n),expected(n),fp64(n),tf32(n);
  for(int exponent:{0,700,-700})
  {
    double* values=x.HostWrite(); double* c=coarse.HostWrite(); double* reference=expected.HostWrite();
    for(int i=0;i<n;++i)
    {
      values[i]=std::scalbn(.7+.13*std::sin(.13*i)+.03*std::cos(.71*i),exponent);
      c[i]=std::scalbn(.01*std::sin(.31*i),exponent);
    }
    for(int col=0;col<ne;++col)
    {
      const int group=col<18?0:1;
      for(int row=0;row<m;++row)
      {
        long double sum=0.0L;
        for(int j=0;j<k;++j) sum+=static_cast<long double>(matrices[(group*k+j)*m+row])*values[inputs[col*k+j]];
        const int output=outputs[col*m+row]; reference[output]=c[output]+scales[col]*static_cast<double>(sum);
      }
    }
    if(exponent==-700)
    {
      mfem::Vector zero_reference(n); zero_reference=0.0;
      MFEM_VERIFY(std::abs(Relative(zero_reference,expected)-1.0)<1e-15,
                  "Small-vector relative error must reject an all-zero result.");
    }
    control.Mult(x,coarse,fp64); mixed.Mult(x,coarse,tf32);
    const double exact_error=Relative(fp64,expected), mixed_error=Relative(tf32,expected);
    MFEM_VERIFY(exact_error<5e-14 && mixed_error<3e-3,
                "Shared matrix real-column oracle failed: exponent="<<exponent<<" fp64="<<exact_error<<" tf32="<<mixed_error);
    std::cout<<"shared_patch_batch_test real_columns="<<ne<<" groups="<<groups
             <<" exponent="<<exponent<<" fp64_error="<<exact_error<<" tf32_error="<<mixed_error<<" passed\n";
  }
  x=0.0; coarse=0.0; mixed.Mult(x,coarse,tf32);
  const double* zero=tf32.HostRead();
  for(int i=0;i<n;++i) MFEM_VERIFY(zero[i]==0.0,"Shared patch zero RHS failed.");
  x.HostReadWrite()[0]=std::numeric_limits<double>::quiet_NaN();
  mixed.Mult(x,coarse,tf32);
  const double* bad=tf32.HostRead(); bool rejected=false;
  for(int i=0;i<n;++i) rejected=rejected||!std::isfinite(bad[i]);
  MFEM_VERIFY(rejected,"Shared patch hid nonfinite input from outer residual authority.");
}
// Dense independent composition oracle with nonidentity L2/column maps and
// nonsymmetric neighbor blocks. Changing coarse vectors must change r-A0*zc;
// source coefficients are mutated after configuration to verify the snapshot.
void CheckComposition()
{
  constexpr int dof=3,ne=7,k=5*dof,n=ne*dof,groups=2;
  mfem::Array<int> element_map(n);
  for(int i=0;i<n;++i) element_map[i]=n-1-i;
  std::vector<std::array<int,5>> patch(ne);
  std::vector<double> stencil(ne*k*dof),dense(n*n,0.0),matrices(groups*dof*k),scales(ne);
  std::vector<int> offsets{0,3,7},inputs(ne*k),outputs(n);
  for(int e=0;e<ne;++e)
  {
    for(int slot=0;slot<5;++slot)
    {
      patch[e][slot]=(e+slot)%ne;
      for(int col=0;col<dof;++col) for(int row=0;row<dof;++row)
      {
        const double value=slot==0&&row==col?2.0+.1*e:
            .03*std::sin(.17*(1+row+3*col+7*slot+11*e));
        stencil[(e*k+slot*dof+col)*dof+row]=value;
        dense[element_map[e*dof+row]*n+element_map[patch[e][slot]*dof+col]]=value;
      }
    }
  }
  for(int group=0;group<groups;++group) for(int col=0;col<k;++col) for(int row=0;row<dof;++row)
    matrices[(group*k+col)*dof+row]=(row==col?.7+.2*group:.04*std::cos(.23*(row+2*col+group)));
  for(int column=0;column<ne;++column)
  {
    const int e=(3*column)%ne; scales[column]=.8+.05*column;
    for(int row=0;row<dof;++row) outputs[column*dof+row]=element_map[e*dof+row];
    for(int slot=0;slot<5;++slot) for(int col=0;col<dof;++col)
      inputs[column*k+slot*dof+col]=element_map[patch[e][slot]*dof+col];
  }
  igrlab::igr_internal::SharedPatchBatch additive64,additive32,multiplicative64,multiplicative32;
  additive64.Configure(dof,k,n,matrices,offsets,inputs,outputs,scales,false);
  additive32.Configure(dof,k,n,matrices,offsets,inputs,outputs,scales,true);
  multiplicative64.Configure(dof,k,n,matrices,offsets,inputs,outputs,scales,false,"multiplicative");
  multiplicative32.Configure(dof,k,n,matrices,offsets,inputs,outputs,scales,true,"multiplicative");
  multiplicative64.ConfigureFrozenOperator(dof,patch,element_map,stencil);
  multiplicative32.ConfigureFrozenOperator(dof,patch,element_map,stencil);
  stencil[0]+=100.0; // The configured frozen operator must own its snapshot.
  mfem::Vector rhs(n),coarse(n),expected_add(n),expected_mult(n),a64(n),a32(n),m64(n),m32(n);
  for(double coarse_scale:{0.0,.2,-.3})
  {
    double* r=rhs.HostWrite(); double* c=coarse.HostWrite();
    for(int i=0;i<n;++i){r[i]=.4+std::sin(.19*(i+1));c[i]=coarse_scale*std::cos(.31*(i+1));}
    std::vector<long double> residual(n);
    for(int row=0;row<n;++row)
    {
      long double action=0.0L;
      for(int col=0;col<n;++col) action+=static_cast<long double>(dense[row*n+col])*c[col];
      residual[row]=static_cast<long double>(r[row])-action;
    }
    double* expected_a=expected_add.HostWrite();double* expected_m=expected_mult.HostWrite();
    for(int column=0;column<ne;++column) for(int row=0;row<dof;++row)
    {
      const int group=column<3?0:1,target=outputs[column*dof+row];
      long double add=0.0L,mult=0.0L;
      for(int col=0;col<k;++col)
      {
        const long double coefficient=matrices[(group*k+col)*dof+row];
        add+=coefficient*r[inputs[column*k+col]];
        mult+=coefficient*residual[inputs[column*k+col]];
      }
      expected_a[target]=c[target]+scales[column]*static_cast<double>(add);
      expected_m[target]=c[target]+scales[column]*static_cast<double>(mult);
    }
    additive64.Mult(rhs,coarse,a64);additive32.Mult(rhs,coarse,a32);
    multiplicative64.Mult(rhs,coarse,m64);multiplicative32.Mult(rhs,coarse,m32);
    const double exact_error=Relative(m64,expected_mult),mixed_error=Relative(m32,expected_mult);
    MFEM_VERIFY(Relative(a64,expected_add)<5e-14 && Relative(a32,expected_add)<3e-3 &&
                exact_error<5e-14 && mixed_error<3e-3,"Shared-patch composition dense oracle failed.");
    if(coarse_scale==0.0)
    {
      MFEM_VERIFY(Relative(m64,a64)<5e-14,"Zero coarse correction must preserve additive result.");
    }
    else
    {
      MFEM_VERIFY(Relative(expected_mult,expected_add)>1e-2,"Composition fixture must distinguish the two formulas.");
    }
    std::cout<<"shared_patch_composition_test coarse_scale="<<coarse_scale
             <<" fp64_error="<<exact_error<<" tf32_error="<<mixed_error<<" dense_reference=true passed\n";
  }
  coarse.HostReadWrite()[0]=std::numeric_limits<double>::infinity();
  multiplicative32.Mult(rhs,coarse,m32);
  const double* result=m32.HostRead();bool nonfinite=false;
  for(int i=0;i<n;++i)nonfinite=nonfinite||!std::isfinite(result[i]);
  MFEM_VERIFY(nonfinite,"Frozen stencil must propagate nonfinite coarse output to original-A authority.");
}
void Assemble(mfem::BilinearForm& form,mfem::Coefficient& reaction,mfem::Coefficient& diffusion)
{
  form.AddDomainIntegrator(new mfem::MassIntegrator(reaction));
  form.AddDomainIntegrator(new mfem::DiffusionIntegrator(diffusion));
  form.AddInteriorFaceIntegrator(new mfem::DGDiffusionIntegrator(diffusion,-1.0,36.0));
  form.Assemble(); form.Finalize();
}
void CheckCurrentOperator()
{
  mfem::Mesh box=mfem::Mesh::MakeCartesian2D(4,4,mfem::Element::QUADRILATERAL,true,1.0,1.0);
  mfem::Vector dx(2),dy(2); dx=0.0; dy=0.0; dx(0)=1.0; dy(1)=1.0;
  std::vector<mfem::Vector> shifts{dx,dy};
  mfem::Mesh mesh=mfem::Mesh::MakePeriodic(box,box.CreatePeriodicVertexMapping(shifts));
  mfem::ParMesh pmesh(MPI_COMM_WORLD,mesh);
  mfem::L2_FECollection collection(5,2,mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace sfes(&pmesh,&collection);
  mfem::FunctionCoefficient reaction([](const mfem::Vector& x){return 1.0+.1*std::cos(6.283185307179586*x(0));});
  mfem::FunctionCoefficient diffusion([](const mfem::Vector& x){return .01*(1.0+.1*std::sin(6.283185307179586*x(1)));});
  mfem::FunctionCoefficient changed_reaction([](const mfem::Vector& x){return 1.1+.05*std::sin(6.283185307179586*x(1));});
  mfem::FunctionCoefficient changed_diffusion([](const mfem::Vector& x){return .012*(1.0+.1*std::cos(6.283185307179586*x(0)));});
  mfem::BilinearForm frozen(&sfes),current(&sfes);
  Assemble(frozen,reaction,diffusion); Assemble(current,changed_reaction,changed_diffusion);
  const int n=sfes.GetVSize(); mfem::Vector exact(n),b(n),x(n),action(n),fp64_solution(n);
  for(int i=0;i<n;++i) exact(i)=.7+std::sin(.13*i)+.1*std::cos(.71*i);
  current.SpMat().Mult(exact,b);
  ScopedEnvironment precision("IGRLAB_IGR_TWO_LEVEL_BLOCK_PRECISION","fp64");
  ScopedEnvironment coarse_backend("IGRLAB_IGR_COARSE_CG_BACKEND","cuda-fp64");
  ScopedEnvironment coarse_execution("IGRLAB_IGR_COARSE_CG_EXECUTION","cooperative");
  for(const char* composition:{"additive","multiplicative"})
  for(const char* mode:{"shared-patch-fp64","shared-patch-tf32"})
  {
    ScopedEnvironment composition_selection("IGRLAB_IGR_TWO_LEVEL_PATCH_COMPOSITION",composition);
    ScopedEnvironment selection("IGRLAB_IGR_TWO_LEVEL_LOCAL",mode);
    auto prec=igrlab::igr_internal::MakeTwoLevelInversePreconditioner(frozen.SpMat(),sfes,"cg",1);
    auto solver=igrlab::igr_internal::MakeDeviceFGMRESSolver(MPI_COMM_WORLD);
    solver->SetOperator(current.SpMat()); solver->SetPreconditioner(*prec);
    solver->SetRelTol(1e-11); solver->SetAbsTol(0.0); solver->SetMaxIter(400); solver->SetPrintLevel(0);
    solver->iterative_mode=false; x=0.0; solver->Mult(b,x);
    current.SpMat().Mult(x,action);
    const double residual=Relative(action,b),error=Relative(x,exact);
    MFEM_VERIFY(solver->GetConverged() && residual<=1e-11 && error<1e-7,
                "Shared patch changed-A FP64 certificate failed: "<<mode<<" iterations="<<solver->GetNumIterations()
                <<" residual="<<residual<<" error="<<error);
    if(std::string(mode)=="shared-patch-fp64")
    {
      fp64_solution=x;
    }
    else
    {
      MFEM_VERIFY(Relative(x,fp64_solution)<1e-7,"Shared patch FP64/TF32 solutions disagree.");
    }
    std::cout<<"shared_patch_current_operator_test mode="<<mode<<" composition="<<composition<<" true_residual="<<residual
             <<" solution_error="<<error<<" iterations="<<solver->GetNumIterations()<<" passed\n";
  }
}
}
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv); mfem::Device device("cuda");
  int ordinal=0; cudaDeviceProp properties{};
  igrlab::igr_internal::shared_patch_internal::CUDA(cudaGetDevice(&ordinal));
  igrlab::igr_internal::shared_patch_internal::CUDA(cudaGetDeviceProperties(&properties,ordinal));
  if(properties.major<8) { std::cout<<"shared_patch_test skipped: TF32 requiresSM80+\n"; return 77; }
  CheckRealBatch(); CheckComposition(); CheckCurrentOperator();
  std::cout<<"shared_patch_test passed real_columns=true power_of_two_scaling=true multiplicative_dense_reference=true original_A_certificate=true nonfinite_rejected=true\n";
  return 0;
}
