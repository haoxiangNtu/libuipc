#include <finite_element/finite_element_kinetic.h>
#include <finite_element/finite_element_diff_dof_reporter.h>

namespace uipc::backend::cuda
{
void FiniteElementKinetic::do_build(FiniteElementEnergyProducer::BuildInfo& info)
{
    m_impl.finite_element_method = require<FiniteElementMethod>();

    BuildInfo this_info;
    do_build(this_info);

    m_impl.finite_element_method->add_kinetic(this);
}

void FiniteElementKinetic::do_report_extent(ReportExtentInfo& info)
{
    auto vert_count = m_impl.finite_element_method->xs().size();
    info.energy_count(vert_count);
    info.stencil_dim(1);
}

void FiniteElementKinetic::do_compute_energy(FiniteElementEnergyProducer::ComputeEnergyInfo& info)
{
    ComputeEnergyInfo this_info{&m_impl, &info};
    do_compute_energy(this_info);
}

void FiniteElementKinetic::do_compute_gradient_hessian(FiniteElementEnergyProducer::ComputeGradientHessianInfo& info)
{
    ComputeGradientHessianInfo this_info{&m_impl, &info};
    do_compute_gradient_hessian(this_info);
}

// 2. Protected version: Handles logic for derived class specific info (modifies parameter type)
void FiniteElementKinetic::do_compute_gradient_hessian_by_vertex(
    FiniteElementKinetic::ComputeGradientHessianInfo& info, IndexT /*vertexId*/  // Matches IndexT in the header file
)
{
    // Default implementation (or add your vertex computation logic)
}

// 3. Private version: Base class interface bridge (supplement to ensure forwarding)
void FiniteElementKinetic::do_compute_gradient_hessian_by_vertex(
    FiniteElementEnergyProducer::ComputeGradientHessianInfo& info, IndexT vertexId)
{
    FiniteElementKinetic::ComputeGradientHessianInfo this_info{&m_impl, &info};
    do_compute_gradient_hessian_by_vertex(this_info, vertexId);  // Forwards to the protected version
}

// Added by color, which includes a set of vertices
void FiniteElementKinetic::do_compute_gradient_hessian_by_color(
    FiniteElementKinetic::ComputeGradientHessianInfo& info,
    muda::CBufferView<IndexT> color_vertices /*vertexId*/  // Matches IndexT in the header file
)
{
    // Default implementation (or add your vertex computation logic)
}

// 3. Private version: Base class interface bridge (supplement to ensure forwarding)
void FiniteElementKinetic::do_compute_gradient_hessian_by_color(
    FiniteElementEnergyProducer::ComputeGradientHessianInfo& info,
    muda::CBufferView<IndexT>                                color_vertices)
{
    FiniteElementKinetic::ComputeGradientHessianInfo this_info{&m_impl, &info};
    do_compute_gradient_hessian_by_color(this_info, color_vertices);  // Forwards to the protected version
}

}  // namespace uipc::backend::cuda
