#include <finite_element/fem_line_search_reporter.h>
#include <finite_element/finite_element_constitution.h>
#include <finite_element/finite_element_extra_constitution.h>
#include <muda/cub/device/device_reduce.h>
#include <kernel_cout.h>
#include <muda/ext/eigen/log_proxy.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(FEMLineSearchReporter);

void FEMLineSearchReporter::do_init(InitInfo& info) {}

void FEMLineSearchReporter::do_build(LineSearchReporter::BuildInfo& info)
{
    m_impl.finite_element_method = require<FiniteElementMethod>();

    auto fea = find<FiniteElementAnimator>();
    if(fea)
        m_impl.finite_element_animator = *fea;
}

void FEMLineSearchReporter::do_record_start_point(LineSearcher::RecordInfo& info)
{
    m_impl.record_start_point(info);
}

void FEMLineSearchReporter::do_step_forward(LineSearcher::StepInfo& info)
{
    m_impl.step_forward(info);
}

void FEMLineSearchReporter::do_step_forward_by_vertex(LineSearcher::StepInfo& info)
{
    m_impl.step_forward_by_vertex(info);
}

void FEMLineSearchReporter::do_compute_energy(LineSearcher::EnergyInfo& info)
{
    m_impl.compute_energy(info);
}

void FEMLineSearchReporter::Impl::record_start_point(LineSearcher::RecordInfo& info)
{
    using namespace muda;

    fem().x_temps = fem().xs;
}

void FEMLineSearchReporter::Impl::step_forward(LineSearcher::StepInfo& info)
{
    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(fem().xs.size(),
               [is_fixed = fem().is_fixed.cviewer().name("is_fixed"),
                x_temps  = fem().x_temps.cviewer().name("x_temps"),
                xs       = fem().xs.viewer().name("xs"),
                dxs      = fem().dxs.cviewer().name("dxs"),
                alpha    = info.alpha] __device__(int i) mutable
               { xs(i) = x_temps(i) + alpha * dxs(i); });
}

void FEMLineSearchReporter::Impl::step_forward_by_vertex(LineSearcher::StepInfo& info)
{
    //using namespace muda;
    //muda::DeviceBuffer<Float> alpha_by_vertex_d(info.alpha_by_vertex);
    //ParallelFor()
    //    .file_line(__FILE__, __LINE__)
    //    .apply(fem().xs.size(),
    //           [is_fixed = fem().is_fixed.cviewer().name("is_fixed"),
    //            x_temps  = fem().x_temps.cviewer().name("x_temps"),
    //            xs       = fem().xs.viewer().name("xs"),
    //            dxs      = fem().dxs.cviewer().name("dxs"),
    //            alpha = alpha_by_vertex_d.cviewer().name("alpha_by_vertex")] __device__(int i) mutable
    //           { xs(i) = x_temps(i) + alpha(i) * dxs(i); });

    using namespace muda;

    // alpha_by_vertex_d: max allowed displacement length per vertex
    DeviceBuffer<Float> alpha_by_vertex_d(info.alpha_by_vertex);
    // Per-vertex clamped flags; 1 if clamped, 0 otherwise
    DeviceBuffer<IndexT> clamped_flags(fem().xs.size());
    clamped_flags.fill(0);

    //ParallelFor()
    //    .file_line(__FILE__, __LINE__)
    //    .apply(fem().xs.size(),
    //           [is_fixed = fem().is_fixed.cviewer().name("is_fixed"),
    //            x_temps  = fem().x_temps.cviewer().name("x_temps"),
    //            xs       = fem().xs.viewer().name("xs"),
    //            dxs      = fem().dxs.cviewer().name("dxs"),
    //            alpha    = alpha_by_vertex_d.cviewer().name("alpha_by_vertex"),
    //            flags = clamped_flags.viewer().name("clamped_flags")] __device__(int i) mutable
    //           {
    //               auto   d = dxs(i);  // update direction (full displacement)
    //               Float  limit   = alpha(i);  // max allowed magnitude
    //               IndexT clamped = 0;

    //               // Only clamp when displacement exceeds the bound
    //               if(limit > Float(0))
    //               {
    //                   const Float len2   = d.squaredNorm();
    //                   const Float limit2 = limit * limit;
    //                   if(len2 > limit2)
    //                   {
    //                       const Float len = sqrt(len2);
    //                       d *= (limit / len);
    //                       clamped = 1;
    //                   }
    //               }
    //               else
    //               {
    //                   // limit <= 0: disallow movement if there is any
    //                   if(d.squaredNorm() > Float(0))
    //                   {
    //                       d.setZero();
    //                       clamped = 1;
    //                   }
    //               }

    //               xs(i)    = x_temps(i) + d;
    //               flags(i) = clamped;
    //           }).wait();

        ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(fem().xs.size(),
               [is_fixed = fem().is_fixed.cviewer().name("is_fixed"),
                x_temps  = fem().x_temps.cviewer().name("x_temps"),
                xs       = fem().xs.viewer().name("xs"),
                dxs      = fem().dxs.cviewer().name("dxs"),
                alpha    = alpha_by_vertex_d.cviewer().name("alpha_by_vertex"),
                flags    = clamped_flags.viewer().name("clamped_flags"),
                cout = KernelCout::viewer()] __device__(int i) mutable
               {
                   auto   d     = dxs(i);    // full displacement direction
                   Float  limit = alpha(i);  // max allowed magnitude
                   Float  len0  = sqrt(d.squaredNorm());  // original magnitude
                   IndexT clamped = 0;

                   if(limit > Float(0))
                   {
                       const Float limit2 = limit * limit;
                       const Float len2   = len0 * len0;
                       if(len2 > limit2)
                       {
                           // clamp to 'limit'
                           d *= (limit / len0);
                           clamped = 1;
                       }
                   }
                   else
                   {
                       if(len0 > Float(0))
                       {
                           d.setZero();
                           clamped = 1;
                       }
                   }

                   xs(i)    = x_temps(i) + d;
                   flags(i) = clamped;

                   // Debug: print only clamped vertices (throttled)
                   //if(i < 128)
                   //{
                   //    cout << "FEM clamp: i=" << i << ", alpha=" << limit
                   //         << ", |dx|=" << len0 << "\n";
                   //}
                   
               })
        .wait();

    // Reduce flags to get total clamped count
    muda::DeviceReduce().Sum(
        clamped_flags.data(), clamp_count.data(), clamped_flags.size());

}


void FEMLineSearchReporter::Impl::compute_energy(LineSearcher::EnergyInfo& info)
{
    using namespace muda;

    // Kinetic/Elastic/Contact ...
    for(auto* producer : fem().energy_producers)
        producer->compute_energy(info);

    DeviceReduce().Sum(fem().energy_producer_energies.data(),
                       fem().energy_producer_energy.data(),
                       fem().energy_producer_energies.size());

    // copy back to host
    Float E = fem().energy_producer_energy;

    // Animation
    Float anim_E = 0.0;
    if(finite_element_animator)
        anim_E = finite_element_animator->compute_energy(info);

    Float total_E = E + anim_E;

    info.energy(total_E);
}
}  // namespace uipc::backend::cuda
