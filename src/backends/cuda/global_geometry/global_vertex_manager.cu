#include <global_geometry/global_vertex_manager.h>
#include <uipc/common/enumerate.h>
#include <uipc/common/range.h>
#include <muda/cub/device/device_reduce.h>
#include <global_geometry/vertex_reporter.h>
#include <finite_element/finite_element_vertex_reporter.h> 
/*************************************************************************************************
* Core Implementation
*************************************************************************************************/
namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(GlobalVertexManager);

void GlobalVertexManager::do_build()
{
    auto d_hat = world().scene().config().find<Float>("contact/d_hat");
    m_impl.default_d_hat = d_hat->view()[0];
}

void GlobalVertexManager::Impl::init()
{
    auto vertex_reporter_view = vertex_reporters.view();

    // 1) Setup index for each vertex reporter
    for(auto&& [i, R] : enumerate(vertex_reporter_view))
        R->m_index = i;

    // 2) Count the number of vertices reported by each reporter
    auto N = vertex_reporter_view.size();
    reporter_vertex_offsets_counts.resize(N);

    span<IndexT> reporter_vertex_counts = reporter_vertex_offsets_counts.counts();

    for(auto&& [i, R] : enumerate(vertex_reporter_view))
    {
        VertexCountInfo info;
        R->report_count(info);
        // get count back
        reporter_vertex_counts[i] = info.m_count;
    }
    reporter_vertex_offsets_counts.scan();
    SizeT total_count = reporter_vertex_offsets_counts.total_count();

    // 3) Initialize buffers for vertex attributes
    coindices.resize(total_count);
    positions.resize(total_count);
    rest_positions.resize(total_count);
    safe_positions.resize(total_count);
    contact_element_ids.resize(total_count, 0);
    contact_subscene_element_ids.resize(total_count, 0);
    thicknesses.resize(total_count, 0.0);
    dimensions.resize(total_count, 3);  // default 3D
    displacements.resize(total_count, Vector3::Zero());
    displacement_norms.resize(total_count, 0.0);
    body_ids.resize(total_count, -1);  // -1 means no care about body id
    d_hats.resize(total_count, default_d_hat);  // use default d_hat if not specified

    // 4) Create the subviews for each attribute_reporter,
    //    so that each reporter can write to its own subview
    for(auto&& [i, R] : enumerate(vertex_reporter_view))
    {
        VertexAttributeInfo attributes{this, i};
        R->report_attributes(attributes);
    }

    // 5) Initialize previous positions and safe positions
    prev_positions = positions;
    safe_positions = positions;

    // 6) Other initializations
    axis_max_disp = 0.0;
}

void GlobalVertexManager::Impl::rebuild()
{
    UIPC_ASSERT(false, "Not implemented yet");
}

void GlobalVertexManager::add_reporter(VertexReporter* reporter)
{
    check_state(SimEngineState::BuildSystems, "add_reporter()");
    m_impl.vertex_reporters.register_subsystem(*reporter);
}

void GlobalVertexManager::Impl::step_forward(Float alpha)
{
    using namespace muda;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(positions.size(),
               [pos      = positions.viewer().name("pos"),
                safe_pos = safe_positions.viewer().name("safe_pos"),
                disp     = displacements.viewer().name("disp"),
                alpha    = alpha] __device__(int i) mutable
               { pos(i) = safe_pos(i) + alpha * disp(i); });
}

void GlobalVertexManager::Impl::step_forward_by_vertex(std::vector<Float> alpha)
{
    using namespace muda;
    ///////////这里的公式要改一下，alpha 不是步长，而是最大允许移动的大小，如果超过这个大小，要进行缩放
    //类似于x_v = (x_v - xprev_v) / ||x_v - xprev_v|| * b_v + xprev_v;
    muda::DeviceBuffer<Float> alpha_by_vertex(alpha);
    //ParallelFor()
    //    .file_line(__FILE__, __LINE__)
    //    .apply(positions.size(),
    //           [pos      = positions.viewer().name("pos"),
    //            safe_pos = safe_positions.viewer().name("safe_pos"),
    //            disp     = displacements.viewer().name("disp"),
    //            alpha = alpha_by_vertex.viewer().name("alpha_by_vertex")] __device__(int i) mutable
    //           { pos(i) = safe_pos(i) + alpha(i) * disp(i); });


        ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(positions.size(),
               [pos      = positions.viewer().name("pos"),
                safe_pos = safe_positions.viewer().name("safe_pos"),
                disp     = displacements.viewer().name("disp"),
                alpha = alpha_by_vertex.viewer().name("alpha_by_vertex")] __device__(int i) mutable
               {
                   auto  d     = disp(i);   // Vector3 displacement
                   Float limit = alpha(i);  // max allowed magnitude

                   if(limit > Float(0))
                   {
                       const Float len2   = d.squaredNorm();
                       const Float limit2 = limit * limit;
                       if(len2 > limit2)
                       {
                           const Float len = sqrt(len2);
                           d *= (limit / len);  // clamp
                       }
                   }
                   // If limit <= 0, disallow movement:
                   else
                   {
                       d.setZero();
                   }

                   pos(i) = safe_pos(i) + d;
               });
}

void GlobalVertexManager::Impl::collect_vertex_displacements()
{
    for(auto&& [i, R] : enumerate(vertex_reporters.view()))
    {
        VertexDisplacementInfo vd{this, i};
        R->report_displacements(vd);
    }
}

void GlobalVertexManager::Impl::record_prev_positions()
{
    using namespace muda;
    BufferLaunch().copy<Vector3>(prev_positions.view(), std::as_const(positions).view());
}

void GlobalVertexManager::Impl::record_start_point()
{
    using namespace muda;
    BufferLaunch().copy<Vector3>(safe_positions.view(), std::as_const(positions).view());
}

Float GlobalVertexManager::Impl::compute_axis_max_displacement()
{
    muda::DeviceReduce().Reduce((Float*)displacements.data(),
                                axis_max_disp.data(),
                                displacements.size() * 3,
                                [] CUB_RUNTIME_FUNCTION(const Float& L, const Float& R)
                                {
                                    auto absL = std::abs(L);
                                    auto absR = std::abs(R);
                                    return absL > absR ? absL : absR;
                                },
                                0.0);
    return axis_max_disp;
}

AABB GlobalVertexManager::Impl::compute_vertex_bounding_box()
{
    Float max_float = std::numeric_limits<Float>::max();
    muda::DeviceReduce()
        .Reduce(
            positions.data(),
            min_pos.data(),
            positions.size(),
            [] CUB_RUNTIME_FUNCTION(const Vector3& L, const Vector3& R) -> Vector3
            { return L.cwiseMin(R); },
            Vector3{max_float, max_float, max_float})
        .Reduce(
            positions.data(),
            max_pos.data(),
            positions.size(),
            [] CUB_RUNTIME_FUNCTION(const Vector3& L, const Vector3& R) -> Vector3
            { return L.cwiseMax(R); },
            Vector3{-max_float, -max_float, -max_float});

    Vector3 min_pos_host, max_pos_host;
    min_pos_host = min_pos;
    max_pos_host = max_pos;

    vertex_bounding_box = AABB{min_pos_host, max_pos_host};
    return vertex_bounding_box;
}
}  // namespace uipc::backend::cuda

// Dump & Recover:
namespace uipc::backend::cuda
{
bool GlobalVertexManager::Impl::dump(DumpInfo& info)
{
    auto path  = info.dump_path(__FILE__);
    auto frame = info.frame();

    return dump_positions.dump(fmt::format("{}positions.{}", path, frame), positions)  //
           && dump_prev_positions.dump(fmt::format("{}prev_positions.{}", path, frame),
                                       prev_positions);
}

bool GlobalVertexManager::Impl::try_recover(RecoverInfo& info)
{
    auto path = info.dump_path(__FILE__);
    return dump_positions.load(fmt::format("{}positions.{}", path, info.frame()))  //
           && dump_prev_positions.load(
               fmt::format("{}prev_positions.{}", path, info.frame()));
}

void GlobalVertexManager::Impl::apply_recover(RecoverInfo& info)
{
    dump_positions.apply_to(positions);
    dump_prev_positions.apply_to(prev_positions);
}

void GlobalVertexManager::Impl::clear_recover(RecoverInfo& info)
{
    dump_positions.clean_up();
    dump_prev_positions.clean_up();
}
}  // namespace uipc::backend::cuda


/*************************************************************************************************
* API Implementation
*************************************************************************************************/
namespace uipc::backend::cuda
{
void GlobalVertexManager::VertexCountInfo::count(SizeT count) noexcept
{
    m_count = count;
}

void GlobalVertexManager::VertexCountInfo::changeable(bool is_changable) noexcept
{
    m_changable = is_changable;
}

GlobalVertexManager::VertexAttributeInfo::VertexAttributeInfo(Impl* impl, SizeT index) noexcept
    : m_impl(impl)
    , m_index(index)
{
}

muda::BufferView<Vector3> GlobalVertexManager::VertexAttributeInfo::rest_positions() const noexcept
{
    return m_impl->subview(m_impl->rest_positions, m_index);
}

muda::BufferView<Float> GlobalVertexManager::VertexAttributeInfo::thicknesses() const noexcept
{
    return m_impl->subview(m_impl->thicknesses, m_index);
}

muda::BufferView<IndexT> GlobalVertexManager::VertexAttributeInfo::coindices() const noexcept
{
    return m_impl->subview(m_impl->coindices, m_index);
}

muda::BufferView<IndexT> GlobalVertexManager::VertexAttributeInfo::dimensions() const noexcept
{
    return m_impl->subview(m_impl->dimensions, m_index);
}

muda::BufferView<Vector3> GlobalVertexManager::VertexAttributeInfo::positions() const noexcept
{
    return m_impl->subview(m_impl->positions, m_index);
}

muda::BufferView<IndexT> GlobalVertexManager::VertexAttributeInfo::contact_element_ids() const noexcept
{
    return m_impl->subview(m_impl->contact_element_ids, m_index);
}

muda::BufferView<IndexT> GlobalVertexManager::VertexAttributeInfo::contact_subscene_element_ids() const noexcept
{
    return m_impl->subview(m_impl->contact_subscene_element_ids, m_index);
}

muda::BufferView<IndexT> GlobalVertexManager::VertexAttributeInfo::body_ids() const noexcept
{
    return m_impl->subview(m_impl->body_ids, m_index);
}

muda::BufferView<Float> GlobalVertexManager::VertexAttributeInfo::d_hats() const noexcept
{
    return m_impl->subview(m_impl->d_hats, m_index);  // Assuming d_hats are stored in thicknesses
}

GlobalVertexManager::VertexDisplacementInfo::VertexDisplacementInfo(Impl* impl, SizeT index) noexcept
    : m_impl(impl)
    , m_index(index)
{
}

muda::BufferView<Vector3> GlobalVertexManager::VertexDisplacementInfo::displacements() const noexcept
{
    return m_impl->subview(m_impl->displacements, m_index);
}

muda::CBufferView<IndexT> GlobalVertexManager::VertexDisplacementInfo::coindices() const noexcept
{
    return m_impl->subview(m_impl->coindices, m_index);
}

bool GlobalVertexManager::do_dump(DumpInfo& info)
{
    return m_impl.dump(info);
}

bool GlobalVertexManager::do_try_recover(RecoverInfo& info)
{
    return m_impl.try_recover(info);
}

void GlobalVertexManager::do_apply_recover(RecoverInfo& info)
{
    m_impl.apply_recover(info);
}

void GlobalVertexManager::do_clear_recover(RecoverInfo& info)
{
    m_impl.clear_recover(info);
}

void GlobalVertexManager::init()
{
    m_impl.init();
}

void GlobalVertexManager::rebuild()
{
    m_impl.rebuild();
}

void GlobalVertexManager::record_prev_positions()
{
    m_impl.record_prev_positions();
}

void GlobalVertexManager::collect_vertex_displacements()
{
    m_impl.collect_vertex_displacements();
}

muda::CBufferView<IndexT> GlobalVertexManager::coindices() const noexcept
{
    return m_impl.coindices;
}

muda::CBufferView<IndexT> GlobalVertexManager::body_ids() const noexcept
{
    return m_impl.body_ids;
}

muda::CBufferView<Float> GlobalVertexManager::d_hats() const noexcept
{
    return m_impl.d_hats;
}

muda::CBufferView<Vector3> GlobalVertexManager::positions() const noexcept
{
    return m_impl.positions;
}

muda::CBufferView<Vector3> GlobalVertexManager::prev_positions() const noexcept
{
    return m_impl.prev_positions;
}

muda::CBufferView<Vector3> GlobalVertexManager::rest_positions() const noexcept
{
    return m_impl.rest_positions;
}

muda::CBufferView<Vector3> GlobalVertexManager::safe_positions() const noexcept
{
    return m_impl.safe_positions;
}

muda::CBufferView<IndexT> GlobalVertexManager::contact_element_ids() const noexcept
{
    return m_impl.contact_element_ids;
}

muda::CBufferView<IndexT> GlobalVertexManager::subscene_element_ids() const noexcept
{
    return m_impl.contact_subscene_element_ids;
}

muda::CBufferView<Vector3> GlobalVertexManager::displacements() const noexcept
{
    return m_impl.displacements;
}

muda::CBufferView<Float> GlobalVertexManager::thicknesses() const noexcept
{
    return m_impl.thicknesses;
}

Float GlobalVertexManager::compute_axis_max_displacement()
{
    return m_impl.compute_axis_max_displacement();
}

AABB GlobalVertexManager::compute_vertex_bounding_box()
{
    return m_impl.compute_vertex_bounding_box();
}

void GlobalVertexManager::step_forward(Float alpha)
{
    m_impl.step_forward(alpha);
}

void GlobalVertexManager::step_forward_by_vertex(Float alpha, std::vector<Float> alpha_vec)
{

        // 1) Build alpha_all = [alpha,...,alpha] by default
    const SizeT        total = m_impl.positions.size();
    std::vector<Float> alpha_all(total, alpha);

    // 2) Locate FEM reporter and its [offset, count]
    SizeT fem_offset = 0;
    SizeT fem_count  = 0;
    bool  has_fem    = false;

    auto reporter_view = m_impl.vertex_reporters.view();
    auto offsets       = m_impl.reporter_vertex_offsets_counts.offsets();
    auto counts        = m_impl.reporter_vertex_offsets_counts.counts();

    for(auto&& [i, R] : enumerate(reporter_view))
    {
        // Only the FEM reporter should get per-vertex alpha
        if(dynamic_cast<FiniteElementVertexReporter*>(R) != nullptr)
        {
            has_fem    = true;
            fem_offset = static_cast<SizeT>(offsets[i]);
            fem_count  = static_cast<SizeT>(counts[i]);
            break;
        }
    }

    // 3) Overwrite FEM segment with alpha_vec
    if(has_fem)
    {
        UIPC_ASSERT(alpha_vec.size() == fem_count,
                    "alpha_vec size mismatch: expected {}, got {}",
                    fem_count,
                    alpha_vec.size());
        std::copy(alpha_vec.begin(), alpha_vec.end(), alpha_all.begin() + fem_offset);
    }

    // 4) Dispatch to device
    m_impl.step_forward_by_vertex(std::move(alpha_all));
}

void GlobalVertexManager::record_start_point()
{
    m_impl.record_start_point();
}

muda::CBufferView<IndexT> GlobalVertexManager::dimensions() const noexcept
{
    return m_impl.dimensions;
}

AABB GlobalVertexManager::vertex_bounding_box() const noexcept
{
    return m_impl.vertex_bounding_box;
}
}  // namespace uipc::backend::cuda