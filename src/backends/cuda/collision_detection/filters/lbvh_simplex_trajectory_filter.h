#pragma once
#include <sim_system.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <contact_system/global_contact_manager.h>
#include <collision_detection/atomic_counting_lbvh.h>
#include <collision_detection/simplex_trajectory_filter.h>

namespace uipc::backend::cuda
{
class LBVHSimplexTrajectoryFilter final : public SimplexTrajectoryFilter
{
  public:
    using SimplexTrajectoryFilter::SimplexTrajectoryFilter;

    class Impl
    {
      public:
        void detect(DetectInfo& info);
        void filter_active(FilterActiveInfo& info);
        void filter_active_dcd_distance(FilterActiveInfo& info);
        void filter_toi(FilterTOIInfo& info);
        void filter_toi_distance(FilterTOIInfo& info);

        void init_ogc_data(DetectInfo& info);
        void build_edge_face_adjacency(const muda::CBufferView<Vector2i>& edges,
                                       const muda::CBufferView<Vector3i>& faces);
        void build_vertex_adjacency(const muda::CBufferView<Vector2i>& edges,
                                    const muda::CBufferView<Vector3i>& faces,
                                    int num_vertices);
        void preinitialize_contact_data(int num_vertices, int num_facets, int num_edges, float r_q);
        //void preprocess_adjacency(FilterActiveInfo& info);

        // New: Dedicated OGC contact detection functions (integrating Algorithm 1 + Algorithm 2)
        void detect_ogc_contact(DetectInfo& info);
        void phase1_vertex_facet_contact(DetectInfo& info);
        void phase2_edge_edge_contact(DetectInfo& info);

        /**
        * Compute conservative bound bv for all vertices (Equation 21 in the paper)
        * @param num_vertices: total number of global vertices
        * @param gamma_p: relaxation factor (default 0.45)
        */
        void compute_conservative_bounds(int num_vertices, float gamma_p = 0.45f);

        /****************************************************
        *                   Added: OGC-related member variables
        ****************************************************/

        // 1. OGC parameters (can be configured through file or API)
        float m_ogc_r = 0.01f;  // OGC contact radius (default 2mm, corresponds to r in the document)
        float m_ogc_rq = 0.02f;  // OGC query radius (r_q ≥ r, default 4mm, buffering inertial motion), ??????? seems unused ????????
        // 3. Minimum distance buffers (Algorithm 1: d_min,v, d_min,t; Algorithm 2: d_min,e)
        muda::DeviceBuffer<float> m_d_min_v;  // m_d_min_v[v]: minimum distance from vertex v to non-adjacent facets
        muda::DeviceBuffer<float> m_d_min_t;  // m_d_min_t[t]: minimum distance from facet t to non-adjacent vertices
        muda::DeviceBuffer<float> m_d_min_e;  // m_d_min_e[e]: minimum distance from edge e to non-adjacent edges

        //// Vertex => Neighbor edges (CSR structure)
        muda::DeviceBuffer<IndexT> m_v_edge_offsets;  // Per-vertex edge list start offsets
        muda::DeviceBuffer<IndexT> m_v_edge_indices;  // Neighbor edge indices (global edge IDs)

        //// Vertex => Neighbor faces (CSR structure)
        muda::DeviceBuffer<IndexT> m_v_face_offsets;  // Per-vertex face list start offsets
        muda::DeviceBuffer<IndexT> m_v_face_indices;  // Neighbor face indices (global face IDs)

        //// Vertex => Neighbor vertices
        muda::DeviceBuffer<IndexT> m_v_vertex_offsets;
        muda::DeviceBuffer<IndexT> m_v_vertex_indices;

        // build_edge_face_adjacency
        muda::DeviceBuffer<IndexT> m_edge_face_counts;
        muda::DeviceBuffer<IndexT> m_edge_face_vertices;

        //// Vertex-pair => EdgeID mapping (device)
        struct PairHash
        {
            size_t operator()(const std::pair<int, int>& p) const noexcept
            {
                return (static_cast<size_t>(p.first) << 32)
                       ^ static_cast<size_t>(p.second);
            }
        };
        //muda::DeviceBuffer<std::unordered_map<std::pair<int, int>, int, PairHash>> m_edge_id_map;

        // Conservative bound buffer (one bv value per vertex)
        muda::DeviceBuffer<Float> m_bv;

        // OGC parameter: relaxation factor γₚ (paper recommends 0.45 to avoid FP issues)
        float m_gamma_p = 0.45f;

        // 1. Contact sets (device): lists of contacting primitives per vertex / face / edge
        // FOGC: For each vertex v, the set of contacting sub-facets (index + type: 0=vertex/1=edge/2=face)
        //muda::BufferView<muda::vector<ContactFace>> d_FOGC;
        //muda::BufferView<muda::vector<int>> d_VOGC;
        //muda::BufferView<muda::vector<ContactFace>> d_EOGC;

        ////////////////////// Other information:
        // Adjacency lists on device
        //muda::DeviceBuffer<IndexT> m_vertex_adjacent_vertices;
        //muda::DeviceBuffer<IndexT> m_edge_adjacent_face_vertices;
        //muda::DeviceBuffer<Vector3> x_bars;  // Rest positions
        //muda::DeviceBuffer<Vector3> xs;      // Positions
        //////////////////////////////////////================================

        /****************************************************
        *                   Broad Phase
        ****************************************************/

        muda::DeviceBuffer<AABB> codim_point_aabbs;
        muda::DeviceBuffer<AABB> point_aabbs;
        muda::DeviceBuffer<AABB> edge_aabbs;
        muda::DeviceBuffer<AABB> triangle_aabbs;

        // CodimP count is always ≤ AllP count
        AtomicCountingLBVH              lbvh_CodimP;
        AtomicCountingLBVH::QueryBuffer candidate_AllP_CodimP_pairs;

        // Used to detect CodimP-AllE, and AllE-AllE pairs
        AtomicCountingLBVH              lbvh_E;
        AtomicCountingLBVH::QueryBuffer candidate_CodimP_AllE_pairs;
        AtomicCountingLBVH::QueryBuffer candidate_AllE_AllE_pairs;

        // Used to detect AllP-AllT pairs
        AtomicCountingLBVH              lbvh_T;
        AtomicCountingLBVH::QueryBuffer candidate_AllP_AllT_pairs;

        /****************************************************
        *                   Narrow Phase
        ****************************************************/

        muda::DeviceVar<IndexT> selected_PT_count;
        muda::DeviceVar<IndexT> selected_EE_count;
        muda::DeviceVar<IndexT> selected_PE_count;
        muda::DeviceVar<IndexT> selected_PP_count;

        muda::DeviceBuffer<Vector4i> temp_PTs;
        muda::DeviceBuffer<Vector4i> temp_EEs;
        muda::DeviceBuffer<Vector3i> temp_PEs;
        muda::DeviceBuffer<Vector2i> temp_PPs;

        muda::DeviceBuffer<Vector4i> PTs;
        muda::DeviceBuffer<Vector4i> EEs;
        muda::DeviceBuffer<Vector3i> PEs;
        muda::DeviceBuffer<Vector2i> PPs;


        /****************************************************
        *                   CCD TOI
        ****************************************************/

        muda::DeviceBuffer<Float> tois;               // PP, PE, PT, EE
        muda::DeviceBuffer<Float> penetration_depth;  // PP, PE, PT, EE
    };

  private:
    Impl m_impl;

    virtual void do_build(BuildInfo& info) override final;
    virtual void do_detect(DetectInfo& info) override final;
    virtual void do_detect_ogc(DetectInfo& info) override final;
    virtual void do_filter_active(FilterActiveInfo& info) override final;
    virtual void do_filter_active_ogc(FilterActiveInfo& info) override final;
    virtual void do_filter_toi(FilterTOIInfo& info) override final;
    virtual void do_filter_d_v(FilterActiveInfo& info, std::vector<Float>& d_bv) override final;
};
}  // namespace uipc::backend::cuda
