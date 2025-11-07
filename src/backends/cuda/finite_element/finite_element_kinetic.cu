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

// 2. protected版本：处理派生类具体info的逻辑（修改参数类型）
void FiniteElementKinetic::do_compute_gradient_hessian_by_vertex(
    FiniteElementKinetic::ComputeGradientHessianInfo& info, IndexT /*vertexId*/  // 与头文件的IndexT匹配
)
{
    // 默认实现（或添加你的顶点计算逻辑）
}

// 3. private版本：基类接口桥接（补充，确保转发）
void FiniteElementKinetic::do_compute_gradient_hessian_by_vertex(
    FiniteElementEnergyProducer::ComputeGradientHessianInfo& info, IndexT vertexId)
{
    FiniteElementKinetic::ComputeGradientHessianInfo this_info{&m_impl, &info};
    do_compute_gradient_hessian_by_vertex(this_info, vertexId);  // 转发给protected版本
}

// add by color that include a set of vertices
void FiniteElementKinetic::do_compute_gradient_hessian_by_color(
    FiniteElementKinetic::ComputeGradientHessianInfo& info,
    muda::CBufferView<IndexT> color_vertices /*vertexId*/  // 与头文件的IndexT匹配
)
{
    // 默认实现（或添加你的顶点计算逻辑）
}

// 3. private版本：基类接口桥接（补充，确保转发）
void FiniteElementKinetic::do_compute_gradient_hessian_by_color(
    FiniteElementEnergyProducer::ComputeGradientHessianInfo& info,
    muda::CBufferView<IndexT> color_vertices)
{
    FiniteElementKinetic::ComputeGradientHessianInfo this_info{&m_impl, &info};
    do_compute_gradient_hessian_by_color(this_info, color_vertices);  // 转发给protected版本
}

}  // namespace uipc::backend::cuda
