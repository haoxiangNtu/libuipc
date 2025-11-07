#include <finite_element/codim_1d_constitution.h>

namespace uipc::backend::cuda
{
void Codim1DConstitution::do_build(FiniteElementConstitution::BuildInfo& info)
{
    Codim1DConstitution::BuildInfo this_info;
    do_build(this_info);
}

void Codim1DConstitution::do_compute_energy(FiniteElementEnergyProducer::ComputeEnergyInfo& info)
{
    Codim1DConstitution::ComputeEnergyInfo this_info{
        this, m_index_in_dim, info.dt(), info.energies()};
    do_compute_energy(this_info);
}

void Codim1DConstitution::do_compute_gradient_hessian(FiniteElementEnergyProducer::ComputeGradientHessianInfo& info)
{
    ComputeGradientHessianInfo this_info{
        this, m_index_in_dim, info.dt(), info.gradients(), info.hessians()};
    do_compute_gradient_hessian(this_info);
}

// add by vertex (wrapper: generic -> class-specific info)
void Codim1DConstitution::do_compute_gradient_hessian_by_vertex(
    FiniteElementEnergyProducer::ComputeGradientHessianInfo& info, IndexT vertexId)
{
    ComputeGradientHessianInfo this_info{
        this, m_index_in_dim, info.dt(), info.gradients(), info.hessians()};
    do_compute_gradient_hessian_by_vertex(this_info, static_cast<int>(vertexId));
}

// define the protected virtual expected by the linker (default no-op)
void Codim1DConstitution::do_compute_gradient_hessian_by_vertex(ComputeGradientHessianInfo& /*info*/,
                                                                int /*vertexId*/)
{
    // Default: no per-vertex accumulation. Derived constitutions may override.
}
//now instead of add by vertex, we add by color that include a set of vertices
void Codim1DConstitution::do_compute_gradient_hessian_by_color(
    FiniteElementEnergyProducer::ComputeGradientHessianInfo& info, muda::CBufferView<IndexT> color_vertices)
{
    ComputeGradientHessianInfo this_info{
        this, m_index_in_dim, info.dt(), info.gradients(), info.hessians()};
    do_compute_gradient_hessian_by_color(this_info, color_vertices);
}
// define the protected virtual expected by the linker
void Codim1DConstitution::do_compute_gradient_hessian_by_color(
    ComputeGradientHessianInfo& /*info*/, muda::CBufferView<IndexT> /*color_vertices*/)
{
    // Default: no per-vertex accumulation. Derived constitutions may override.
}

IndexT Codim1DConstitution::get_dim() const noexcept
{
    return 1;
}

muda::CBufferView<Vector3> Codim1DConstitution::BaseInfo::xs() const noexcept
{
    return m_impl->fem().xs.view();  // must return full buffer, because the indices index into the full buffer
}

muda::CBufferView<Vector3> Codim1DConstitution::BaseInfo::x_bars() const noexcept
{
    return m_impl->fem().x_bars.view();  // must return full buffer, because the indices index into the full buffer
}

muda::CBufferView<Float> Codim1DConstitution::BaseInfo::rest_lengths() const noexcept
{
    auto& info = constitution_info();
    return m_impl->fem().rest_lengths.view(info.primitive_offset, info.primitive_count);
}

muda::CBufferView<Float> Codim1DConstitution::BaseInfo::thicknesses() const noexcept
{
    return m_impl->fem().thicknesses.view();
}

muda::CBufferView<Vector2i> Codim1DConstitution::BaseInfo::indices() const noexcept
{
    auto& info = constitution_info();
    return m_impl->fem().codim_1ds.view(info.primitive_offset, info.primitive_count);
}
muda::CBufferView<IndexT> Codim1DConstitution::BaseInfo::is_fixed() const noexcept
{
    return m_impl->fem().is_fixed.view();  // must return full buffer, because the indices index into the full buffer
}

const FiniteElementMethod::ConstitutionInfo& Codim1DConstitution::BaseInfo::constitution_info() const noexcept
{
    return m_impl->fem().codim_1d_constitution_infos[m_index_in_dim];
}

Float Codim1DConstitution::BaseInfo::dt() const noexcept
{
    return m_dt;
}
}  // namespace uipc::backend::cuda
