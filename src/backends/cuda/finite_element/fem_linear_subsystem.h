#pragma once
#include <algorithm/matrix_converter.h>
#include <linear_system/diag_linear_subsystem.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_animator.h>
#include <finite_element/fem_dytopo_effect_receiver.h>
#include <finite_element/finite_element_vertex_reporter.h>

namespace uipc::backend::cuda
{
class FEMLinearSubsystem final : public DiagLinearSubsystem
{
  public:
    using DiagLinearSubsystem::DiagLinearSubsystem;

    class ComputeGradientHessianInfo
    {
      public:
        ComputeGradientHessianInfo(Float                             dt,
                                   muda::DoubletVectorView<Float, 3> gradients,
                                   muda::TripletMatrixView<Float, 3> hessians)
            : m_dt(dt)
            , m_gradients(gradients)
            , m_hessians(hessians)
        {
        }

        auto gradients() const noexcept { return m_gradients; }
        auto hessians() const noexcept { return m_hessians; }
        auto dt() const noexcept { return m_dt; }

      private:
        friend class FiniteElementEnergyProducer;
        muda::DoubletVectorView<Float, 3> m_gradients;
        muda::TripletMatrixView<Float, 3> m_hessians;
        Float                             m_dt = 0.0;
    };

    class Impl
    {
      public:
        void report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info);
        void receive_init_dof_info(WorldVisitor& w, GlobalLinearSystem::InitDofInfo& info);

        void report_extent(GlobalLinearSystem::DiagExtentInfo& info);
        void assemble(GlobalLinearSystem::DiagInfo& info);

        void solve_system_vertex_test(GlobalLinearSystem::DiagInfo& info);
        void solve_system_vertex_cpu(GlobalLinearSystem::DiagInfo& info);
        void solve_system_vertex_gpu(GlobalLinearSystem::DiagInfo& info);
        void update_info(GlobalLinearSystem::DiagInfo& info);
        void _assemble_producers(GlobalLinearSystem::DiagInfo& info);
        void _assemble_producers_by_vertex(GlobalLinearSystem::DiagInfo& info, IndexT vertexId);
        //add by vertex coloring
        void _assemble_producers_by_color(GlobalLinearSystem::DiagInfo& info,
                                          muda::CBufferView<IndexT> color_vertices);
        void vertices_Coloring();
        void _assemble_dytopo_effect(GlobalLinearSystem::DiagInfo& info);

        void _assemble_animation(GlobalLinearSystem::DiagInfo& info);
        void accuracy_check(GlobalLinearSystem::AccuracyInfo& info);
        void retrieve_solution(GlobalLinearSystem::SolutionInfo& info);

        SimEngine* sim_engine = nullptr;

        SimSystemSlot<FiniteElementMethod> finite_element_method;
        FiniteElementMethod::Impl&         fem() noexcept
        {
            return finite_element_method->m_impl;
        }
        SimSystemSlot<FEMDyTopoEffectReceiver> dytopo_effect_receiver;
        SimSystemSlot<FiniteElementVertexReporter> finite_element_vertex_reporter;
        SimSystemSlot<FiniteElementAnimator> finite_element_animator;
        FiniteElementAnimator::Impl&         animator() noexcept
        {
            return finite_element_animator->m_impl;
        }
        SizeT energy_producer_hessian_offset = 0;
        SizeT energy_producer_hessian_count  = 0;

        SizeT dytopo_effect_hessian_offset = 0;
        SizeT dytopo_effect_hessian_count  = 0;

        SizeT animator_hessian_offset = 0;
        SizeT animator_hessian_count  = 0;

        Float dt = 0.0;

        Float reserve_ratio = 1.5;

        MatrixConverter<Float, 3>           converter;
        muda::DeviceTripletMatrix<Float, 3> triplet_A;
        muda::DeviceBCOOMatrix<Float, 3>    bcoo_A;

        std::vector<int> vertex_colors;
        std::vector<std::vector<int>> vertex_group;
    };

  private:
    virtual void do_build(DiagLinearSubsystem::BuildInfo& info) override;
    virtual void do_init(DiagLinearSubsystem::InitInfo& info) override;
    virtual void do_report_extent(GlobalLinearSystem::DiagExtentInfo& info) override;
    virtual void do_assemble(GlobalLinearSystem::DiagInfo& info) override;
    virtual void do_solve_system_vertex(GlobalLinearSystem::DiagInfo& info) override;
    virtual void do_update_info(GlobalLinearSystem::DiagInfo& info) override;
    virtual void do_accuracy_check(GlobalLinearSystem::AccuracyInfo& info) override;
    virtual void do_retrieve_solution(GlobalLinearSystem::SolutionInfo& info) override;
    virtual void do_report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info) override;
    virtual void do_receive_init_dof_info(GlobalLinearSystem::InitDofInfo& info) override;

    Impl m_impl;
};
}  // namespace uipc::backend::cuda
