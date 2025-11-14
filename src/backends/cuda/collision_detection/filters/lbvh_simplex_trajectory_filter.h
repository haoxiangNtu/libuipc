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

        // 新增：OGC 接触检测专用成员函数（整合 Algorithm 1 + Algorithm 2）
        void detect_ogc_contact(DetectInfo& info);
        void phase1_vertex_facet_contact(DetectInfo& info);
        void phase2_edge_edge_contact(DetectInfo& info);
        /**
        * 计算所有顶点的保守边界bv（论文公式21）
        * @param num_vertices：全局顶点总数
        * @param gamma_p：松弛因子（默认0.45）
        */
        void compute_conservative_bounds(int num_vertices, float gamma_p = 0.45f);
        /****************************************************
        *                   新增：OGC 依赖成员变量
        ****************************************************/
        // 1. OGC 参数（可通过配置文件或函数接口设置）
        float m_ogc_r = 0.01f;  // OGC 接触半径（默认 2mm，对应文档 r）
        float m_ogc_rq = 0.02f;  // OGC 查询半径（r_q ≥ r，默认 4mm，预留惯性位移空间）??????? 好像没有使用到？？？？？？？？

        // 2. 邻接关系数据（在 preprocess_adjacency 中预处理）
        //muda::DeviceBuffer<IndexT> m_vadj_offsets;  // 顶点邻接列表偏移（CSR 结构）：m_vadj_offsets[v] 是顶点v的邻接列表起始索引
        //muda::DeviceBuffer<IndexT> m_vadj_indices;  // 顶点邻接列表索引：m_vadj_indices[p] 是顶点v的第p个相邻顶点ID
        //muda::DeviceBuffer<Vector2i> m_v_edge_indices;  // 边-only流形顶点所属边：m_v_edge_indices[v] = (e1, e2)（顶点v在边-only流形中的两条边）

        // 3. 最小距离存储（对应文档 Algorithm 1 的 d_min,v、d_min,t；Algorithm 2 的 d_min,e）
        muda::DeviceBuffer<float> m_d_min_v;  // m_d_min_v[v]：顶点v到非相邻面的最小距离
        muda::DeviceBuffer<float> m_d_min_t;  // m_d_min_t[t]：面t到非相邻顶点的最小距离
        muda::DeviceBuffer<float> m_d_min_e;  // m_d_min_e[e]：边e到非相邻边的最小距离

        //        // 2. 最小距离存储（device端，每个顶点/面/边的最小距离）
        //muda::DeviceBuffer<Float> d_d_min_v;  // d_min[v] = 顶点v到其他面的最小距离
        //muda::DeviceBuffer<Float> d_d_min_t;  // d_min[t] = 面t到其他顶点的最小距离
        //muda::DeviceBuffer<Float> d_d_min_e;  // d_min[e] = 边e到其他边的最小距离

        //// 顶点-邻居边：CSR结构（顶点v的邻居边索引存储在 m_v_edge_indices[m_v_edge_offsets[v] ... m_v_edge_offsets[v+1]-1]）
        muda::DeviceBuffer<IndexT> m_v_edge_offsets;  // 每个顶点的邻居边列表起始偏移
        muda::DeviceBuffer<IndexT> m_v_edge_indices;  // 每个顶点的邻居边索引（全局边ID）

        //// 顶点-邻居面：CSR结构（顶点v的邻居面索引存储在 m_v_face_indices[m_v_face_offsets[v] ... m_v_face_offsets[v+1]-1]）
        muda::DeviceBuffer<IndexT> m_v_face_offsets;  // 每个顶点的邻居面列表起始偏移
        muda::DeviceBuffer<IndexT> m_v_face_indices;  // 每个顶点的邻居面索引（全局面ID）
        //// 顶点-邻居点
        muda::DeviceBuffer<IndexT> m_v_vertex_offsets; 
        muda::DeviceBuffer<IndexT> m_v_vertex_indices; 

        //build_edge_face_adjacency
        muda::DeviceBuffer<IndexT> m_edge_face_counts;
        muda::DeviceBuffer<IndexT> m_edge_face_vertices;

        //// 顶点对→边ID映射（设备端）
        struct PairHash
        {
            size_t operator()(const std::pair<int, int>& p) const noexcept
            {
                return (static_cast<size_t>(p.first) << 32)
                       ^ static_cast<size_t>(p.second);
            }
        };
        //muda::DeviceBuffer<std::unordered_map<std::pair<int, int>, int, PairHash>> m_edge_id_map;  // 顶点对→边ID映射（设备端）




        // 保守边界缓冲区（每个顶点对应一个bv值）
        muda::DeviceBuffer<Float> m_bv;

        // OGC参数：松弛因子γₚ（论文建议0.45，避免浮点误差）
        float m_gamma_p = 0.45f;

        // 1. 接触集合存储（device端，每个顶点/面/边的接触对象列表）
        //FOGC: 每个顶点v接触的子面集合（子面a的索引，类型标记：高31位是索引，最低位0=顶点/1=边/2=面）
        //muda::BufferView <muda::vector<ContactFace>> d_FOGC;  // FOGC[v] = 顶点v的接触子面列表
        //muda::BufferView<muda::vector<int>> d_VOGC;  // VOGC[t] = 面t的接触顶点列表
        //muda::BufferView<muda::vector<ContactFace>> d_EOGC;  // EOGC[e] = 边e的接触子面列表
        
        //// 2. 最小距离存储（device端，每个顶点/面/边的最小距离）
        //muda::DeviceBuffer<Float> d_d_min_v;  // d_min[v] = 顶点v到其他面的最小距离
        //muda::DeviceBuffer<Float> d_d_min_t;  // d_min[t] = 面t到其他顶点的最小距离
        //muda::DeviceBuffer<Float> d_d_min_e;  // d_min[e] = 边e到其他边的最小距离
        //std::vector<Float> d_d_min_v;  // d_min[v] = 顶点v到其他面的最小距离
        //std::vector<Float> d_d_min_t;  // d_min[t] = 面t到其他顶点的最小距离
        //std::vector<Float> d_d_min_e;  // d_min[e] = 边e到其他边的最小距离
        //////////////////////other information:
        // 类成员变量（device端邻接关系）
        //muda::DeviceBuffer<IndexT> m_vertex_adjacent_vertices;  // 每个顶点的相邻顶点列表
        //muda::DeviceBuffer<IndexT> m_edge_adjacent_face_vertices;  // 每个边的相邻面顶点（最多2个）
        //muda::DeviceBuffer<Vector3> x_bars;  // Rest Positions
        //muda::DeviceBuffer<Vector3> xs;      // Positions
        //////////////////////////////////////================================

        /****************************************************
        *                   Broad Phase
        ****************************************************/

        muda::DeviceBuffer<AABB> codim_point_aabbs;
        muda::DeviceBuffer<AABB> point_aabbs;
        muda::DeviceBuffer<AABB> edge_aabbs;
        muda::DeviceBuffer<AABB> triangle_aabbs;

        // CodimP count always less or equal to AllP count.
        AtomicCountingLBVH              lbvh_CodimP;
        AtomicCountingLBVH::QueryBuffer candidate_AllP_CodimP_pairs;

        // Used to detect CodimP-AllE, and AllE-AllE pairs.
        AtomicCountingLBVH              lbvh_E;
        AtomicCountingLBVH::QueryBuffer candidate_CodimP_AllE_pairs;
        AtomicCountingLBVH::QueryBuffer candidate_AllE_AllE_pairs;

        // Used to detect AllP-AllT pairs.
        AtomicCountingLBVH              lbvh_T;
        AtomicCountingLBVH::QueryBuffer candidate_AllP_AllT_pairs;


        //AtomicCountingLBVH         lbvh_PP;
        //muda::BufferView<Vector2i> candidate_PP_pairs;

        //AtomicCountingLBVH lbvh_PE;
        //// codimP-allE pairs
        //muda::BufferView<Vector2i> candidate_PE_pairs;
        //AtomicCountingLBVH         lbvh_PT;
        //// allP-allT pairs
        //muda::BufferView<Vector2i> candidate_PT_pairs;
        //AtomicCountingLBVH         lbvh_EE;
        //// allE-allE pairs
        //muda::BufferView<Vector2i> candidate_EE_pairs;


        /****************************************************
        *                   Narrow Phase
        ****************************************************/

        //muda::DeviceBuffer<IndexT> PT_active_flags;
        //muda::DeviceBuffer<IndexT> PT_active_offsets;
        //muda::DeviceBuffer<IndexT> EE_active_flags;
        //muda::DeviceBuffer<IndexT> EE_active_offsets;
        //muda::DeviceBuffer<IndexT> PE_active_flags;
        //muda::DeviceBuffer<IndexT> PE_active_offsets;
        //muda::DeviceBuffer<IndexT> PP_active_flags;
        //muda::DeviceBuffer<IndexT> PP_active_offsets;

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

        muda::DeviceBuffer<Float> tois;  // PP, PE, PT, EE
        muda::DeviceBuffer<Float> penetration_depth;  // PP, PE, PT, EE
    };

  private:
    Impl m_impl;

    virtual void do_build(BuildInfo& info) override final;
    virtual void do_detect(DetectInfo& info) override final;
    virtual void do_filter_active(FilterActiveInfo& info) override final;
    virtual void do_filter_toi(FilterTOIInfo& info) override final;
    virtual void do_filter_d_v(FilterActiveInfo& info, std::vector<Float>& d_bv) override final;
};
}  // namespace uipc::backend::cuda