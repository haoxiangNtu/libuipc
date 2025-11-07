#pragma once
#include <finite_element/finite_element_constitution.h>

namespace uipc::backend::cuda
{
class Codim0DConstitution : public FiniteElementConstitution
{
  public:
    using FiniteElementConstitution::FiniteElementConstitution;

    class BuildInfo
    {
      public:
    };

  protected:
    virtual void do_build(BuildInfo& info) = 0;

  private:
    friend class FiniteElementMethod;

    virtual void do_report_extent(ReportExtentInfo& info) override final;
    virtual void do_build(FiniteElementConstitution::BuildInfo& info) override final;
    virtual IndexT get_dim() const noexcept override final;
    virtual void do_compute_energy(FiniteElementConstitution::ComputeEnergyInfo& info) override final;
    virtual void do_compute_gradient_hessian(
        FiniteElementConstitution::ComputeGradientHessianInfo& info) override final;
    virtual void do_compute_gradient_hessian_by_vertex(FiniteElementConstitution::ComputeGradientHessianInfo& info,
                                                       IndexT vertexId) override final;
    // add do compute gradient hessian by color
    virtual void do_compute_gradient_hessian_by_color(
        FiniteElementConstitution::ComputeGradientHessianInfo& info,
        muda::CBufferView<IndexT> color_vertices) override final;
};
}  // namespace uipc::backend::cuda
