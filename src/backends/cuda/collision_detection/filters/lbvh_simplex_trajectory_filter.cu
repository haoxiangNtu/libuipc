#include <collision_detection/filters/lbvh_simplex_trajectory_filter.h>
#include <muda/cub/device/device_select.h>
#include <muda/ext/eigen/log_proxy.h>
#include <sim_engine.h>
#include <kernel_cout.h>
#include <utils/distance/distance_flagged.h>
#include <utils/distance.h>
#include <utils/codim_thickness.h>
#include <utils/simplex_contact_mask_utils.h>
#include <uipc/common/zip.h>
#include <utils/primitive_d_hat.h>

namespace uipc::backend::cuda
{
constexpr bool PrintDebugInfo = false;

REGISTER_SIM_SYSTEM(LBVHSimplexTrajectoryFilter);

void LBVHSimplexTrajectoryFilter::do_build(BuildInfo& info)
{
    auto& config = world().scene().config();
    auto  method = config.find<std::string>("collision_detection/method");
    if(method->view()[0] != "linear_bvh")
    {
        throw SimSystemException("Linear BVH unused");
    }
    // 新增：预处理几何体邻接关系（顶点相邻顶点、边相邻面顶点）
    auto m_ogc_contact_radius = 0.002;
    //preprocess_adjacency(info);  // 后续实现
    //Impl.detect_ogc_contact(info);
}

//void LBVHSimplexTrajectoryFilter::Impl::preprocess_adjacency(FilterActiveInfo& info)
//{
//    auto Vs = info.surf_vertices();   // 表面顶点列表（全局顶点ID）
//    auto Es = info.surf_edges();      // 表面边列表（e → (v0, v1)）
//    auto Fs = info.surf_triangles();  // 表面面列表（t → (v0, v1, v2)）
//    int   num_vertices = Vs.size();
//    int   num_edges    = Es.size();
//
//    /****************************************************
//    * 1. 预处理顶点邻接关系（m_vadj_offsets + m_vadj_indices，CSR 结构）
//    * 用于 checkVertexFeasibleRegion：验证顶点偏移块的方向约束
//    ****************************************************/
//    // CPU 端构建邻接列表
//    std::vector<IndexT> host_vadj_offsets(num_vertices + 1, 0);
//    std::vector<IndexT> host_vadj_indices;
//    std::vector<Vector3i> Fs_host(Fs.size());
//    Fs.copy_to(Fs_host.data());
//
//    for(const auto& f : Fs_host)
//    {
//        int v0 = f[0], v1 = f[1], v2 = f[2];
//        // 注意：每个三角形对每个顶点有2个邻居，计数应+2而不是+1
//        host_vadj_offsets[v0] += 2;
//        host_vadj_offsets[v1] += 2;
//        host_vadj_offsets[v2] += 2;
//    }
//    // 统计每个顶点的相邻顶点数量
//    for(auto& f : Fs_host)
//    {
//        int v0 = f[0], v1 = f[1], v2 = f[2];
//        host_vadj_offsets[v0]++;
//        host_vadj_offsets[v1]++;
//        host_vadj_offsets[v2]++;
//    }
//    // 计算偏移（CSR 格式）
//    for(int v = 1; v <= num_vertices; v++)
//        host_vadj_offsets[v] += host_vadj_offsets[v - 1];
//    // 填充相邻顶点索引（去重，避免重复存储）
//    std::vector<int> temp_count(num_vertices, 0);
//    host_vadj_indices.resize(host_vadj_offsets[num_vertices]);
//    for(auto& f : Fs_host)
//    {
//        int v0 = f[0], v1 = f[1], v2 = f[2];
//        // 为 v0 添加 v1、v2
//        if(std::find(host_vadj_indices.begin() + host_vadj_offsets[v0],
//                     host_vadj_indices.begin() + host_vadj_offsets[v0 + 1],
//                     v1)
//           == host_vadj_indices.end())
//            host_vadj_indices[host_vadj_offsets[v0] + temp_count[v0]++] = v1;
//        if(std::find(host_vadj_indices.begin() + host_vadj_offsets[v0],
//                     host_vadj_indices.begin() + host_vadj_offsets[v0 + 1],
//                     v2)
//           == host_vadj_indices.end())
//            host_vadj_indices[host_vadj_offsets[v0] + temp_count[v0]++] = v2;
//        // 同理为 v1 添加 v0、v2；为 v2 添加 v0、v1（代码省略，逻辑同上）
//        // ...
//    }
//    // 拷贝到 device 端
//    m_vadj_offsets.resize(num_vertices + 1);
//    m_vadj_indices.resize(host_vadj_indices.size());
//    m_vadj_offsets.copy_from(host_vadj_offsets);
//    m_vadj_indices.copy_from(host_vadj_indices);
//
//    /****************************************************
//    * 2. 预处理边-only流形顶点所属边（m_v_edge_indices）
//    * 用于 checkVertexFeasibleRegionEdgeOffset：验证边-边接触的顶点偏移块
//    ****************************************************/
//    std::vector<Vector2i> host_v_edge_indices(num_vertices, {-1, -1});
//    std::unordered_map<int, std::vector<int>> vertex_to_edges;  // 顶点 → 包含该顶点的边列表
//    for(int e = 0; e < num_edges; e++)
//    {
//        int v0 = Es[e][0], v1 = Es[e][1];
//        vertex_to_edges[v0].push_back(e);
//        vertex_to_edges[v1].push_back(e);
//    }
//    // 每个顶点在边-only流形中最多两条边（1维流形特性）
//    for(int v = 0; v < num_vertices; v++)
//    {
//        auto& edges = vertex_to_edges[v];
//        if(edges.size() >= 1)
//            host_v_edge_indices[v][0] = edges[0];
//        if(edges.size() >= 2)
//            host_v_edge_indices[v][1] = edges[1];
//    }
//    // 拷贝到 device 端
//    m_v_edge_indices.resize(num_vertices);
//    //m_v_edge_indices.copy_from(host_v_edge_indices);
//
//    /****************************************************
//    * 3. 初始化最小距离存储（m_d_min_v/m_d_min_t/m_d_min_e）
//    * 对应 Algorithm 1 第1行、Algorithm 2 第1行：d_min = r_q
//    ****************************************************/
//    m_d_min_v.resize(num_vertices);
//    m_d_min_t.resize(Fs.size());
//    m_d_min_e.resize(num_edges);
//    muda::ParallelFor()
//        .file_line(__FILE__, __LINE__)
//        .apply(num_vertices,
//               [m_d_min_v = m_d_min_v.viewer(), 
//            rq = m_ogc_rq] __device__(int v)
//               { m_d_min_v(v) = rq; })
//        .apply(Fs.size(),
//               [m_d_min_t = m_d_min_t.viewer(), rq = m_ogc_rq] __device__(int t)
//               { m_d_min_t(t) = rq; })
//        .apply(num_edges,
//               [m_d_min_e = m_d_min_e.viewer(), rq = m_ogc_rq] __device__(int e)
//               { m_d_min_e(e) = rq; });
//}


// 假设在SimSystem的初始化函数中
void LBVHSimplexTrajectoryFilter::Impl::init_ogc_data(DetectInfo& info)
{
    int  num_vertices = info.surf_vertices().size();
    auto edges        = info.surf_edges();
    auto faces        = info.surf_triangles();

    // 构建边-面邻接（你已实现）
    build_edge_face_adjacency(edges, faces);
    // 构建顶点-邻居边/面邻接（新增）
    build_vertex_adjacency(edges, faces, num_vertices);
    // 初始化最小距离缓冲区（你已实现preinitialize_contact_data，需调用）
    preinitialize_contact_data(num_vertices, faces.size(), edges.size(), m_ogc_rq);



    //// ===== DEBUG PRINT: per-vertex distances (vertex -> nearest face distance, its incident faces' min distances, its incident edges' min distances)
    //// NOTE:
    //// 1) Do NOT re-resize m_d_min_v/m_d_min_t/m_d_min_e here (that would erase computed values).
    //// 2) Printing everything can be huge; we limit to first N vertices.
    //{
    //    const int DebugPrintVertexCount = 32;  // adjust as needed
    //    // Host copies
    //    std::vector<float>  h_d_min_v;
    //    std::vector<float>  h_d_min_t;
    //    std::vector<float>  h_d_min_e;
    //    std::vector<IndexT> h_v_face_offsets;
    //    std::vector<IndexT> h_v_face_indices;
    //    std::vector<IndexT> h_v_edge_offsets;
    //    std::vector<IndexT> h_v_edge_indices;

    //    m_d_min_v.copy_to(h_d_min_v);
    //    m_d_min_t.copy_to(h_d_min_t);
    //    m_d_min_e.copy_to(h_d_min_e);

    //    h_v_face_offsets.resize(m_v_face_offsets.size());
    //    m_v_face_offsets.copy_to(h_v_face_offsets);
    //    h_v_face_indices.resize(m_v_face_indices.size());
    //    m_v_face_indices.copy_to(h_v_face_indices);

    //    h_v_edge_offsets.resize(m_v_edge_offsets.size());
    //    m_v_edge_offsets.copy_to(h_v_edge_offsets);
    //    h_v_edge_indices.resize(m_v_edge_indices.size());
    //    m_v_edge_indices.copy_to(h_v_edge_indices);

    //    int VPrint = std::min<int>((int)h_d_min_v.size(), DebugPrintVertexCount);
    //    std::cout << "=== d_min debug (first " << VPrint << " vertices) ===\n";
    //    for(int v = 0; v < VPrint; ++v)
    //    {
    //        float dv = h_d_min_v[v];
    //        std::cout << "Vertex " << v << " d_min_v=" << dv << "\n";

    //        // Faces incident to v
    //        if(v + 1 < (int)h_v_face_offsets.size())
    //        {
    //            int f_start = (int)h_v_face_offsets[v];
    //            int f_end   = (int)h_v_face_offsets[v + 1];
    //            std::cout << "  Faces (" << (f_end - f_start) << "): ";
    //            for(int p = f_start; p < f_end; ++p)
    //            {
    //                int f = (int)h_v_face_indices[p];
    //                if(f >= 0 && f < (int)h_d_min_t.size())
    //                    std::cout << "{f=" << f << ", d_min_t=" << h_d_min_t[f] << "} ";
    //            }
    //            std::cout << "\n";
    //        }

    //        // Edges incident to v
    //        if(v + 1 < (int)h_v_edge_offsets.size())
    //        {
    //            int e_start = (int)h_v_edge_offsets[v];
    //            int e_end   = (int)h_v_edge_offsets[v + 1];
    //            std::cout << "  Edges (" << (e_end - e_start) << "): ";
    //            for(int p = e_start; p < e_end; ++p)
    //            {
    //                int e = (int)h_v_edge_indices[p];
    //                if(e >= 0 && e < (int)h_d_min_e.size())
    //                    std::cout << "{e=" << e << ", d_min_e=" << h_d_min_e[e] << "} ";
    //            }
    //            std::cout << "\n";
    //        }
    //    }
    //    std::cout << "=== end d_min debug ===\n";
    //    int tempstop = 0;
    //}
}

void LBVHSimplexTrajectoryFilter::Impl::build_edge_face_adjacency(
    const muda::CBufferView<Vector2i>& edges, const muda::CBufferView<Vector3i>& faces)
{
    int E = static_cast<int>(edges.size());
    int F = static_cast<int>(faces.size());

    std::vector<int> edge_face_counts(E, 0);
    std::vector<int> edge_face_vertices(E * 2, -1);

    // 建立 (min,max) -> edge index 映射
    std::unordered_map<std::pair<int, int>, int, PairHash> edge_map;
    edge_map.reserve(E * 2);

    std::vector<Vector2i> edges_h(E);
    edges.copy_to(edges_h.data());
    for(int ei = 0; ei < E; ++ei)
    {
        int a = edges_h[ei][0];
        int b = edges_h[ei][1];
        if(a > b)
            std::swap(a, b);
        edge_map[{a, b}] = ei;
    }

    std::vector<Vector3i> faces_h(F);
    faces.copy_to(faces_h.data());
    for(const auto& tri : faces_h)
    {
        int  v0 = tri[0], v1 = tri[1], v2 = tri[2];
        auto handle_edge = [&](int a, int b, int opposite)
        {
            if(a > b)
                std::swap(a, b);
            auto it = edge_map.find({a, b});
            if(it == edge_map.end())
                return;
            int  ei  = it->second;
            int& cnt = edge_face_counts[ei];
            if(cnt < 2)
            {
                edge_face_vertices[ei * 2 + cnt] = opposite;
                ++cnt;
            }
        };
        handle_edge(v0, v1, v2);
        handle_edge(v1, v2, v0);
        handle_edge(v2, v0, v1);
    }

    m_edge_face_counts.resize(edge_face_counts.size());
    m_edge_face_vertices.resize(edge_face_vertices.size());
    m_edge_face_counts.view().copy_from(edge_face_counts.data());
    m_edge_face_vertices.view().copy_from(edge_face_vertices.data());

    // 【新增：复制临时edge_map到设备端成员变量m_edge_id_map】
    //m_edge_id_map.resize(1);  // 若用DeviceBuffer存储，大小为1（存整个map）
    //m_edge_id_map.view()[0] = edge_map;  // 复制CPU端edge_map到GPU端
}

/**
 * 构建顶点-邻居边、顶点-邻居面的CSR邻接表
 * @param edges：全局边列表（e → (v0, v1)）
 * @param faces：全局面列表（t → (v0, v1, v2)）
 * @param num_vertices：全局顶点总数
 */
void LBVHSimplexTrajectoryFilter::Impl::build_vertex_adjacency(
    const muda::CBufferView<Vector2i>& edges, const muda::CBufferView<Vector3i>& faces, int num_vertices)
{
    // -------------------------- 步骤1：构建顶点-邻居边 --------------------------
    std::vector<IndexT> v_edge_counts(num_vertices, 0);  // 每个顶点的邻居边数量
    std::vector<IndexT> v_edge_indices_temp;  // 临时存储邻居边ID（全局）

    // 先统计每个顶点的邻居边数量
    std::vector<Vector2i> edges_h(edges.size());
    edges.copy_to(edges_h.data());
    for(int e = 0; e < edges_h.size(); ++e)
    {
        int v0 = edges_h[e][0], v1 = edges_h[e][1];
        v_edge_counts[v0]++;
        v_edge_counts[v1]++;
    }

    // 构建CSR偏移数组 m_v_edge_offsets
    m_v_edge_offsets.resize(num_vertices + 1);  // 偏移数组长度=顶点数+1（最后一个元素是总长度）
    std::vector<int> v_edge_offsets_h(num_vertices + 1, 0);
    for(int v = 0; v < num_vertices; ++v)
    {
        v_edge_offsets_h[v + 1] = v_edge_offsets_h[v] + v_edge_counts[v];
    }
    m_v_edge_offsets.view().copy_from(v_edge_offsets_h.data());

    // 填充邻居边索引（确保每个边被两个顶点记录）
    v_edge_indices_temp.resize(v_edge_offsets_h.back(), -1);
    std::vector<int> v_edge_cursor(num_vertices, 0);  // 每个顶点的当前填充位置
    for(int e = 0; e < edges_h.size(); ++e)
    {
        int v0 = edges_h[e][0], v1 = edges_h[e][1];
        // 顶点v0的邻居边
        int pos0                  = v_edge_offsets_h[v0] + v_edge_cursor[v0]++;
        v_edge_indices_temp[pos0] = e;
        // 顶点v1的邻居边
        int pos1                  = v_edge_offsets_h[v1] + v_edge_cursor[v1]++;
        v_edge_indices_temp[pos1] = e;
    }
    m_v_edge_indices.resize(v_edge_indices_temp.size());
    m_v_edge_indices.view().copy_from(v_edge_indices_temp.data());

    // -------------------------- 步骤2：构建顶点-邻居面 --------------------------
    std::vector<int> v_face_counts(num_vertices, 0);  // 每个顶点的邻居面数量
    std::vector<int> v_face_indices_temp;  // 临时存储邻居面ID（全局）

    // 统计每个顶点的邻居面数量
    std::vector<Vector3i> faces_h(faces.size());
    faces.copy_to(faces_h.data());
    for(int t = 0; t < faces_h.size(); ++t)
    {
        int v0 = faces_h[t][0], v1 = faces_h[t][1], v2 = faces_h[t][2];
        v_face_counts[v0]++;
        v_face_counts[v1]++;
        v_face_counts[v2]++;
    }

    // 构建CSR偏移数组 m_v_face_offsets
    m_v_face_offsets.resize(num_vertices + 1);
    std::vector<int> v_face_offsets_h(num_vertices + 1, 0);
    for(int v = 0; v < num_vertices; ++v)
    {
        v_face_offsets_h[v + 1] = v_face_offsets_h[v] + v_face_counts[v];
    }
    m_v_face_offsets.view().copy_from(v_face_offsets_h.data());

    // 填充邻居面索引（每个面被三个顶点记录）
    v_face_indices_temp.resize(v_face_offsets_h.back(), -1);
    std::vector<int> v_face_cursor(num_vertices, 0);  // 每个顶点的当前填充位置
    for(int t = 0; t < faces_h.size(); ++t)
    {
        int v0 = faces_h[t][0], v1 = faces_h[t][1], v2 = faces_h[t][2];
        // 顶点v0的邻居面
        int pos0                  = v_face_offsets_h[v0] + v_face_cursor[v0]++;
        v_face_indices_temp[pos0] = t;
        // 顶点v1的邻居面
        int pos1                  = v_face_offsets_h[v1] + v_face_cursor[v1]++;
        v_face_indices_temp[pos1] = t;
        // 顶点v2的邻居面
        int pos2                  = v_face_offsets_h[v2] + v_face_cursor[v2]++;
        v_face_indices_temp[pos2] = t;
    }
    m_v_face_indices.resize(v_face_indices_temp.size());
    m_v_face_indices.view().copy_from(v_face_indices_temp.data());

    // -------------------------- 步骤3：构建顶点-邻居顶点（直接存储，去重） --------------------------
    std::vector<int> v_vertex_counts(num_vertices, 0);  // 每个顶点的邻居顶点数量（去重后）
    std::vector<int> v_vertex_indices_temp;             // 临时存储邻居顶点索引

    // 第一步：统计每个顶点的邻居顶点数量（去重）
    for(int v = 0; v < num_vertices; ++v)
    {
        // 获取顶点v的所有关联边
        int                     start = v_edge_offsets_h[v];
        int                     end   = v_edge_offsets_h[v + 1];
        std::unordered_set<int> neighbors;  // 用集合去重

        for(int i = start; i < end; ++i)
        {
            int   e_id = v_edge_indices_temp[i];         // 边的全局ID
            auto& edge = edges_h[e_id];                  // 边的两个端点
            int u = (edge[0] == v) ? edge[1] : edge[0];  // 另一个端点（邻居顶点）
            neighbors.insert(u);
        }

        v_vertex_counts[v] = neighbors.size();  // 去重后的邻居数量
    }

    // 第二步：构建CSR偏移数组 m_v_vertex_offsets
    m_v_vertex_offsets.resize(num_vertices + 1);
    std::vector<int> v_vertex_offsets_h(num_vertices + 1, 0);
    for(int v = 0; v < num_vertices; ++v)
    {
        v_vertex_offsets_h[v + 1] = v_vertex_offsets_h[v] + v_vertex_counts[v];
    }
    m_v_vertex_offsets.view().copy_from(v_vertex_offsets_h.data());

    // 第三步：填充邻居顶点索引（去重后）
    v_vertex_indices_temp.resize(v_vertex_offsets_h.back());
    std::vector<int> v_vertex_cursor(num_vertices, 0);  // 每个顶点的当前填充位置

    for(int v = 0; v < num_vertices; ++v)
    {
        int                     start = v_edge_offsets_h[v];
        int                     end   = v_edge_offsets_h[v + 1];
        std::unordered_set<int> neighbors;

        // 先收集去重的邻居顶点
        for(int i = start; i < end; ++i)
        {
            int   e_id = v_edge_indices_temp[i];
            auto& edge = edges_h[e_id];
            int   u    = (edge[0] == v) ? edge[1] : edge[0];
            neighbors.insert(u);
        }

        // 填充到临时数组
        int pos = v_vertex_offsets_h[v];
        for(int u : neighbors)
        {
            v_vertex_indices_temp[pos + v_vertex_cursor[v]++] = u;
        }
    }

    m_v_vertex_indices.resize(v_vertex_indices_temp.size());
    m_v_vertex_indices.view().copy_from(v_vertex_indices_temp.data());
}

//// 3. 预初始化（在每次接触检测前调用，如filter_active的开头）
void LBVHSimplexTrajectoryFilter::Impl::preinitialize_contact_data(int num_vertices,
                                                                   int num_facets,
                                                                   int num_edges,
                                                                   float r_q)
{
    // 初始化最小距离为查询半径r_q（文档Algorithm 1第1行、Algorithm 2第1行）
    // 这里应该改成类似于m_d_min_v
    
    m_d_min_v.resize(num_vertices);
    m_d_min_t.resize(num_facets);
    m_d_min_e.resize(num_edges);
    muda::ParallelFor()
        .apply(num_vertices,
               [d_min_v = m_d_min_v.viewer(), r_q] __device__(int v)
               { d_min_v(v) = r_q; })
        .apply(num_facets,
               [d_min_t = m_d_min_t.viewer(), r_q] __device__(int t)
               { d_min_t(t) = r_q; })
        .apply(num_edges,
               [d_min_e = m_d_min_e.viewer(), r_q] __device__(int e)
               { d_min_e(e) = r_q; });

    // 清空上一帧的接触集合
    //d_FOGC.resize(num_vertices);
    //d_VOGC.resize(num_facets);
    //d_EOGC.resize(num_edges);
    //muda::ParallelFor()
    //    .apply(num_vertices,
    //           [d_FOGC = d_FOGC.viewer()] __device__(int v)
    //           { d_FOGC[v].clear(); })
    //    .apply(num_facets,
    //           [d_VOGC = d_VOGC.viewer()] __device__(int t)
    //           { d_VOGC[t].clear(); })
    //    .apply(num_edges,
    //           [d_EOGC = d_EOGC.viewer()] __device__(int e)
    //           { d_EOGC[e].clear(); });
}


void LBVHSimplexTrajectoryFilter::Impl::detect_ogc_contact(DetectInfo& info)
{
    using namespace muda;
    auto Vs = info.surf_vertices();   // 表面顶点列表（全局顶点ID）
    auto Es = info.surf_edges();      // 表面边列表（e → (v0, v1)）
    auto Fs = info.surf_triangles();  // 表面面列表（t → (v0, v1, v2)）
    auto positions = info.positions();  // 顶点位置缓冲区（positions[v] = 3D坐标）

    //// 前置检查：确保邻接关系和BVH已初始化（由 preprocess_adjacency 和 detect 函数提前处理）
    //if(m_vadj_offsets.empty() || candidate_AllP_AllT_pairs.empty())
    //    throw SimSystemException("OGC contact detect: adjacency/BVH not initialized");

    /****************************************************
    * 阶段1：顶点-面接触检测（对应文档 Algorithm 1）
    * 输入：candidate_AllP_AllT_pairs（顶点-面候选对，由 lbvh_T 查询生成）
    * 输出：temp_PTs/PEs/PPs（接触集合）、m_d_min_v/m_d_min_t（最小距离）
    ****************************************************/
    phase1_vertex_facet_contact(info);

    /***********************************
    * 阶段2：边-边接触检测（对应文档 Algorithm 2）
    * 输入：candidate_AllE_AllE_pairs（边-边候选对，由 lbvh_E 查询生成）
    * 输出：temp_EEs/PEs/PPs（接触集合）、m_d_min_e（最小距离）
    ****************************************************/
    phase2_edge_edge_contact(info);

    /****************************************************
    * 新增阶段3：计算所有顶点的保守边界bv
    ****************************************************/
    int num_vertices = Vs.size();
    compute_conservative_bounds(num_vertices, m_gamma_p);

    /****************************************************
    * 后续：筛选有效接触对（复用原有filter_active逻辑）
    ****************************************************/
    // filter_active(info);
}



/**
 * @brief OGC核心函数：验证待检测点是否落在某顶点的偏移块（Uₐ）内，对应文档公式(8)
 * @details 顶点偏移块Uₐ是"半径为r的球体，被所有相邻顶点的垂直平面切割"的区域，需满足两个核心约束：
 *          1. 待检测点到顶点a的距离 ≤ 接触半径r（距离约束）
 *          2. 待检测点与顶点a的连线，与顶点a到所有相邻顶点的连线夹角 ≥ 90°（方向约束）
 *          函数返回true表示点在偏移块内，可作为OGC的有效接触对；返回false则为无效接触
 * 
 * @param x              待验证的点（如顶点-面接触中的顶点V、边-边接触的最近点）
 * @param a              顶点偏移块对应的顶点ID（即公式中的v，待验证的顶点子面）
 * @param positions      所有顶点的当前位置缓冲区（device端，存储每个顶点的3D坐标）
 * @param vadj_offsets   顶点邻接关系的偏移缓冲区（CSR-like结构，存储每个顶点邻接列表的起始索引）
 *                       结构说明：vadj_offsets[a]是顶点a的邻接列表在vadj_indices中的起始位置；
 *                                vadj_offsets[a+1]是顶点a的邻接列表的结束位置（下一个顶点的起始）
 * @param vadj_indices   顶点邻接关系的索引缓冲区（CSR-like结构，存储具体的邻接顶点ID）
 *                       结构说明：从vadj_offsets[a]到vadj_offsets[a+1]-1的元素，均为顶点a的相邻顶点ID
 * @param r              OGC接触半径（与文档中的偏移半径一致，控制顶点偏移块的大小）
 * @return bool          true：点在顶点a的偏移块Uₐ内；false：点不在偏移块内
 */
//
//__device__ bool checkVertexFeasibleRegionEdgeOffset(const Vector3& x_c,
//                                                    int            v_idx,
//                                                    const muda::CDense1D<Vector3>& positions,
//                                                    const muda::CDense1D<Vector2i>& surf_edges,
//                                                    const muda::CDense1D<Vector2i>& v_edge_indices,
//                                                    float ogc_r)

// 【OGC核心函数】判断点x是否在边e的偏移块Uₑ内（对应文档3.2节Equation 9，Algorithm 1第15行checkEdgeFeasibleRegion）
// 边偏移块Uₑ定义：半径r的圆柱（以e为轴）被4个平面切割后的区域，仅允许点在“边内部+相邻面片的有效侧”
// 返回值：true=点x在边e的偏移块内（可产生正交接触力），false=不在（排除接触）
//__device__ bool checkEdgeFeasibleRegion(
//    const Vector3& x,  // 待判断的点（通常是算法1中的顶点v的位置x(v)）
//    int            e,  // 当前待检查的边e（索引，对应网格边集E中的元素）
//    const muda::CDense1D<Vector2i>& surf_edges,  // 网格边的顶点对：surf_edges(e) = (v1, v2)，存储边e的两个端点索引
//    const muda::CDense1D<Vector3>& positions,  // 所有顶点的位置数据：positions(v) = 顶点v的3D坐标
//    const muda::CDense1D<int>& edge_face_counts,  // 边e关联的面片数量：edge_face_counts(e) = k（流形网格中k≤2，边最多属于2个相邻面片）
//    const muda::CDense1D<int>& edge_face_vertices,  // 边e关联面片的“对顶点”：edge_face_vertices(e*2 + k) = opp（面片v1v2opp中，除边e外的第三个顶点）
//    float r  // OGC接触半径（边偏移块的圆柱半径，对应文档中的r）
//)
//{
//    // -------------------------- 步骤1：获取边e的基础几何信息 --------------------------
//    // 从边集获取边e的两个端点v1、v2（文档3.2节：边e由顶点v_{e,1}和v_{e,2}构成）
//    Vector2i e_vs = surf_edges(e);   // e_vs：边e的顶点索引对 (v1, v2)
//    int v1 = e_vs[0], v2 = e_vs[1];  // v1 = 边e的起点，v2 = 边e的终点（Vector2i内部元素仍用[]访问，符合向量成员访问习惯）
//    const Vector3& x1    = positions(v1);  // x1：顶点v1的当前位置
//    const Vector3& x2    = positions(v2);  // x2：顶点v2的当前位置
//    Vector3        vec_e = x2 - x1;        // vec_e：边e的方向向量（从v1指向v2）
//    float len_e_sq = vec_e.squaredNorm();  // len_e_sq：边e的长度平方（避免开方，提升GPU效率）
//
//    // 退化边检查：若边长度接近0（浮点误差范围内），边偏移块无意义，直接返回false
//    //if(len_e_sq < 1e-12f)
//    //    return false;
//
//
//    // -------------------------- 步骤2：判断点x是否在边e的“圆柱范围内” --------------------------
//    // 文档3.2节Equation 9条件1：dis(x, e) ≤ r（点x到边e的最短距离≤接触半径r，即x在以e为轴、r为半径的圆柱内）
//    Vector3 vec_x1x = x - x1;  // vec_x1x：从v1指向x的向量
//    float t = vec_x1x.dot(vec_e) / len_e_sq;  // t：x在边e上的投影参数（t∈[0,1]表示投影在边内部，t<0在v1外侧，t>1在v2外侧）
//
//    // 【可选】强制投影范围到边内部（注释中保留，因后续dot1/dot2判断已等效实现此逻辑）
//    // if(t < 0.f || t > 1.f) return false;
//
//    Vector3 closest = x1 + t * vec_e;  // closest：点x到边e的最短距离点（投影点）
//    // 检查x到投影点的距离是否≤r（用平方距离避免开方，减少GPU计算开销，加1e-6浮点误差容差）
//    if((x - closest).squaredNorm() > r * r)
//        return false;
//
//
//    // -------------------------- 步骤3：判断点x是否在边e的“内部圆柱段” --------------------------
//    // 文档3.2节Equation 9条件2-3：排除x在边e的端点延长线上（仅保留边内部的圆柱段）
//    // 条件2：(x - x_{e,1}) · (x_{e,2} - x_{e,1}) > 0 → x在v1的“远离v2”侧（非v1外侧）
//    float dot1 = vec_x1x.dot(vec_e);
//    // 条件3：(x - x_{e,2}) · (x_{e,1} - x_{e,2}) > 0 → x在v2的“远离v1”侧（非v2外侧）
//    float dot2 = (x - x2).dot(-vec_e);  // -vec_e：从v2指向v1的方向向量
//
//    // 若x在v1或v2的外侧（dot≤0，浮点误差容差1e-6），则不在边偏移块内，返回false
//    if(dot1 <= 0 || dot2 <= 0)
//        return false;
//
//
//    // -------------------------- 步骤4：判断点x是否在“相邻面片的有效侧” --------------------------
//    // 文档3.2节Equation 9条件4：排除x在边e相邻面片的“禁止侧”（边偏移块是圆柱被相邻面片的垂直平面切割后的区域）
//    // 边e最多关联2个面片（流形网格），循环处理每个关联面片
//    int cnt = edge_face_counts(e);  // cnt：边e关联的面片数量（k=0/1/2）
//    for(int k = 0; k < cnt; ++k)
//    {
//        // 获取边e在第k个关联面片中的“对顶点”opp（面片由v1、v2、opp构成，opp是除边e外的第三个顶点）
//        int opp = edge_face_vertices(e * 2 + k);
//        if(opp < 0)  // 无效顶点索引（如边界边仅关联1个面片，另一个位置存-1），跳过
//            continue;
//
//        // 计算对顶点opp在边e上的“垂足p”（文档3.2节中的p(x1,x2,x3)：x3在x1x2线上的垂直投影）
//        Vector3 vec_v1opp = positions(opp) - x1;  // vec_v1opp：从v1指向opp的向量
//        float t_p = vec_v1opp.dot(vec_e) / len_e_sq;  // t_p：opp在边e上的投影参数
//        Vector3 p = x1 + t_p * vec_e;                 // p：opp在边e上的垂足
//
//        // 文档3.2节条件4：(x - p) · (p - opp) ≥ 0 → x在面片v1v2opp的“外侧”（符合边偏移块的切割平面约束）
//        Vector3 vec_xp = x - p;  // vec_xp：从p指向x的向量
//        Vector3 vec_p_opp = p - positions(opp);  // vec_p_opp：从opp指向p的向量（面片的内侧方向）
//        float dot_adj = vec_xp.dot(vec_p_opp);  // 点积判断x是否在切割平面的有效侧
//
//        // 若x在面片的“禁止侧”（dot_adj < -1e-6，浮点误差容差），返回false
//        if(dot_adj < 0)
//            return false;
//    }
//
//
//    // -------------------------- 步骤5：所有条件满足，x在边e的偏移块Uₑ内 --------------------------
//    return true;
//}


/**
 * 验证点x是否在边的偏移块Uₑ内（直接传入边的两个顶点位置，无需边索引→顶点的映射）
 * 核心修改：
 * 1. 移除`surf_edges`和通过边索引`e`获取顶点的逻辑，直接接收边的两个顶点位置x1、x2
 * 2. 保留边索引`e`（因`edge_face_counts`和`edge_face_vertices`仍需通过e访问关联面片）
 * 3. 核心约束逻辑（圆柱范围、端点限制、相邻面片切割）完全不变
 * 
 * @param x                     待判断的点（如顶点v的位置x(v)）
 * @param x1                    边的第一个顶点位置（直接传入，无需通过边索引获取）
 * @param x2                    边的第二个顶点位置（直接传入，无需通过边索引获取）
 * @param e                     边的全局索引（用于访问关联面片信息，不可省略）
 * @param positions             所有顶点的位置数据（用于获取关联面片的对顶点位置）
 * @param edge_face_counts      边e关联的面片数量（流形网格中k≤2）
 * @param edge_face_vertices    边e关联面片的“对顶点”（面片v1v2opp中除边外的第三个顶点）
 * @param r                     OGC接触半径（边偏移块的圆柱半径）
 * @return bool                 true=x在边的偏移块内；false=不在
 */
__device__ bool checkEdgeFeasibleRegion(const Vector3& x,  // 待判断的点
                                        const Vector3& x1,  // 边的第一个顶点位置（直接传入）
                                        const Vector3& x2,  // 边的第二个顶点位置（直接传入）
                                        int e,  // 边的全局索引（关联面片依赖）
                                        const muda::CDense1D<Vector3>& positions,  // 所有顶点位置（获取对顶点opp的位置）
                                        const muda::CDense1D<int>& edge_face_counts,  // 边e关联的面片数量
                                        const muda::CDense1D<int>& edge_face_vertices,  // 边e关联面片的对顶点
                                        float r  // 接触半径
)
{
    // -------------------------- 步骤1：获取边的基础几何信息（直接基于x1、x2计算） --------------------------
    Vector3 vec_e = x2 - x1;               // 边的方向向量（从x1指向x2）
    float len_e_sq = vec_e.squaredNorm();  // 边的长度平方（避免开方，提升GPU效率）

    // 退化边检查：长度接近0的边无偏移块意义（浮点误差容差1e-12）
    if(len_e_sq < 1e-12f)
        return false;


    // -------------------------- 步骤2：判断点x是否在边的“圆柱范围内” --------------------------
    // 条件：点x到边的最短距离 ≤ r（即x在以边为轴、r为半径的圆柱内）
    Vector3 vec_x1x = x - x1;                 // 从x1指向x的向量
    float t = vec_x1x.dot(vec_e) / len_e_sq;  // x在边上的投影参数（t∈[0,1]为边内部）

    Vector3 closest = x1 + t * vec_e;  // x到边的最短距离点（投影点）
    // 平方距离 > r² + 1e-12 → 不在圆柱内（容忍浮点误差）
    if((x - closest).squaredNorm() > r * r + 1e-12f)
        return false;


    // -------------------------- 步骤3：判断点x是否在边的“内部圆柱段”（排除端点外侧） --------------------------
    // 条件1：x不在x1的外侧（(x - x1) · (x2 - x1) > 0）
    float dot1 = vec_x1x.dot(vec_e);
    // 条件2：x不在x2的外侧（(x - x2) · (x1 - x2) > 0 → 等价于(x - x2) · (-vec_e) > 0）
    float dot2 = (x - x2).dot(-vec_e);

    // 若x在x1或x2的外侧（dot≤0），不在边偏移块内
    if(dot1 <= 1e-6f || dot2 <= 1e-6f)  // 放宽至1e-6容差，兼容GPU浮点精度
        return false;


    // -------------------------- 步骤4：判断点x是否在“相邻面片的有效侧” --------------------------
    // 边偏移块需被相邻面片的垂直平面切割，仅保留外侧区域（文档3.2节Equation 9条件4）
    int cnt = edge_face_counts(e);  // 边e关联的面片数量（0/1/2）
    for(int k = 0; k < cnt; ++k)
    {
        // 获取第k个关联面片的“对顶点”opp（面片由x1、x2、opp构成）
        int opp = edge_face_vertices(e * 2 + k);
        if(opp < 0)  // 无效对顶点（如边界边仅关联1个面片，另一个为-1），跳过
            continue;

        // 计算opp在边上的垂足p（用于确定切割平面）
        const Vector3& x_opp     = positions(opp);      // 对顶点opp的位置
        Vector3        vec_v1opp = x_opp - x1;          // 从x1指向opp的向量
        float   t_p = vec_v1opp.dot(vec_e) / len_e_sq;  // opp在边上的投影参数
        Vector3 p   = x1 + t_p * vec_e;                 // 垂足p的位置

        // 条件：(x - p) · (p - x_opp) ≥ 0 → x在切割平面的有效侧（外侧）
        Vector3 vec_xp    = x - p;      // 从p指向x的向量
        Vector3 vec_p_opp = p - x_opp;  // 从opp指向p的向量（面片内侧方向）
        float   dot_adj   = vec_xp.dot(vec_p_opp);

        // 若x在面片的“禁止侧”（dot_adj < -1e-6），返回false
        if(dot_adj < 0)
            return false;
    }


    // -------------------------- 所有条件满足，x在边的偏移块内 --------------------------
    return true;
}


// -------------------------- 附录：GPU并行调用示例（对应OGC的并行接触检测逻辑） --------------------------
// 文档4.1节：OGC接触检测是“局部并行操作”，无全局同步，适合GPU大规模并行
// 此处示例使用muda的ParallelFor，对所有边E并行调用checkEdgeFeasibleRegion
// template <typename EsView, typename PsView, typename CountView, typename VertView>
// void parallelCheckEdgeFeasibleRegion(
//     const EsView& Es,          // 边集视图
//     const PsView& Ps,          // 顶点位置视图
//     const CountView& face_counts, // 边-面片数量视图
//     const VertView& face_verts, // 边-面片对顶点视图
//     float ogc_r,               // OGC接触半径r
//     const Vector3& x_to_check  // 待判断的点x（需根据实际场景传入，如算法1中的顶点v的位置）
// )
// {
//     ParallelFor().apply(
//         Es.size(),  // 并行维度：边的总数（每个线程处理一条边e）
//         [surf_edges  = Es.cviewer().name("edges"),    // 边集（设备端视图）
//          positions   = Ps.cviewer().name("positions"),// 顶点位置（设备端视图）
//          face_counts = face_counts.cviewer().name("edge_face_counts"), // 边-面片数量
//          face_verts  = face_verts.cviewer().name("edge_face_vertices"),// 边-面片对顶点
//          r           = ogc_r,                         // OGC接触半径
//          x           = x_to_check]                    // 待判断的点x（如算法1中的顶点v）
//         __device__(int e) mutable                     // GPU线程：处理第e条边
//         {
//             // 核心调用：判断点x是否在边e的偏移块内
//             bool is_in_edge_block = checkEdgeFeasibleRegion(
//                 x, e, surf_edges, positions, face_counts, face_verts, r
//             );

//             // 后续逻辑：若is_in_edge_block为true，将边e加入顶点x的接触面集FOGC(v)（算法1第16行）
//             // （需结合原子操作避免线程竞争，如算法1第16-17行的FOGC/EOGC更新）
//         }
//     );
// }

__device__ bool checkVertexFeasibleRegion(
    const Vector3& x,  // test oint：待检测的点（如碰撞检测中的候选点）
    int            a,  // vertex id：顶点偏移块对应的顶点ID（公式中的v）
    const muda::CDense1D<Vector3>& positions,  // 所有顶点的3D位置（device端，index为顶点ID）
    const muda::CDense1D<IndexT>& vadj_offsets,  // 顶点邻接列表的偏移缓冲区（CSR起始索引）
    const muda::CDense1D<IndexT>& vadj_indices,  // 顶点邻接列表的ID缓冲区（CSR具体ID）
    float r)  // OGC接触半径（公式中的r，偏移块的球体半径）
{
    // 1. 提取顶点a的位置，计算待检测点x到顶点a的向量（对应公式中的x - x_v）
    const Vector3& x_a = positions(a);  // x_a：顶点a的当前位置（公式中的x_v）
    Vector3 vec_xa = x - x_a;  // vec_xa：待检测点x到顶点a的向量（公式中的x - x_v）

    // 2. 约束1：距离约束（公式8的||x - x_v|| ≤ r）
    // 优化点：用平方模（squaredNorm）替代模长（norm），避免开方运算，提升GPU计算效率
    // 容忍微小浮点误差：若平方距离略大于r²（如因精度导致），仍判定为超出范围
    if(vec_xa.squaredNorm() > r * r)
        return false;  // 距离超出接触半径，点不在偏移块内

    // 3. 约束2：方向约束（公式8的(x - x_v)·(x_v - x_{v'}) ≥ 0，对所有相邻顶点v'）
    // 3.1 解析顶点a的邻接列表范围（基于CSR-like结构）
    const int start = (int)vadj_offsets(a);  // 顶点a的邻接列表在vadj_indices中的起始索引
    const int end = (int)vadj_offsets(a + 1);  // 顶点a的邻接列表的结束索引（下一个顶点的起始）
    // 遍历顶点a的所有相邻顶点v'（v_prime）
    for(int p = start; p < end; ++p)
    {
        // 3.2 提取当前相邻顶点v'的ID和位置
        const int v_prime = (int)vadj_indices(p);  // v_prime：顶点a的一个相邻顶点（公式中的v'）
        const Vector3& x_vp = positions(v_prime);  // x_vp：相邻顶点v'的位置（公式中的x_{v'}）

        // 3.3 计算方向约束的向量（公式中的x_v - x_{v'}）
        Vector3 vec_av = x_a - x_vp;  // vec_av：顶点a指向相邻顶点v'的向量（a到v'的反方向）

        // 3.4 计算点积，验证方向约束（公式中的(x - x_v)·(x_v - x_{v'}) ≥ 0）
        float dot = vec_xa.dot(vec_av);  // 点积结果：若≥0，说明夹角≤90°，满足方向约束
        // 浮点误差处理：允许-1e-6的偏差（避免GPU浮点精度误差导致的误判，如理论0变为-1e-15）
        //if(dot < -1e-6f)
        //    return false;  // 方向约束不满足，点不在偏移块内
        if(dot < 0)
            return false;  // 方向约束不满足，点不在偏移块内
    }

    // 4. 所有约束均满足：待检测点x在顶点a的偏移块Uₐ内，返回有效
    return true;

    // ------------------------------ 并行调用示例（注释部分说明）------------------------------
    // 以下是函数在GPU并行环境中的典型调用方式（基于muda的ParallelFor）
    // ParallelFor().apply(num_queries,  // num_queries：并行查询的数量（如待检测点的总数）
    //                     [positions = info.positions().cviewer().name("positions"),  // 顶点位置（const视图）
    //                      vadj_offsets = m_vadj_offsets.cviewer().name("vadj_offsets"),  // 邻接偏移（const视图）
    //                      vadj_indices = m_vadj_indices.cviewer().name("vadj_indices"),  // 邻接ID（const视图）
    //                      r = ogc_radius] __device__(int i) mutable  // i：并行线程索引
    //                     {
    //                         // 假设x是第i个待检测点的位置，v是待验证的顶点ID
    //                         // bool ok = checkVertexFeasibleRegion(x, v, positions, vadj_offsets, vadj_indices, r);
    //                         // 根据ok的结果，标记接触对是否有效（如加入FOGC接触集合）
    //                     });
}

/**
 * 【重构版】验证边-边接触的最近点x_c是否在顶点v_idx的偏移块Uₐ内（支持多相邻顶点）
 * 核心修改：
 * 1. 用CSR结构（vadj_offsets/vadj_indices）替代v_edge_indices/surf_edges，遍历所有相邻顶点
 * 2. 方向约束验证覆盖顶点v_idx的所有邻居，不再局限于2个边邻居
 * 3. 适配多相邻顶点场景（如非边-only流形的分叉顶点）
 * 
 * @param x_c           边-边接触的最近点（待验证的点）
 * @param v_idx         顶点偏移块对应的顶点ID（待验证的顶点）
 * @param positions     所有顶点的3D位置（device端，index=顶点ID）
 * @param vadj_offsets  顶点邻接表的CSR偏移缓冲区（vadj_offsets[v] = 邻居列表起始索引）
 * @param vadj_indices  顶点邻接表的CSR索引缓冲区（vadj_indices[p] = 相邻顶点ID）
 * @param ogc_r         OGC接触半径（顶点偏移块的球体半径）
 * @return bool         true=x_c在顶点v_idx的偏移块内；false=不在
 */
__device__ bool checkVertexFeasibleRegionEdgeOffset(
    const Vector3&                 x_c,
    int                            v_idx,
    const muda::CDense1D<Vector3>& positions,
    const muda::CDense1D<IndexT>& vadj_offsets,  // 新增：CSR偏移（替代v_edge_indices）
    const muda::CDense1D<IndexT>& vadj_indices,  // 新增：CSR索引（替代surf_edges）
    float ogc_r)
{
    // -------------------------- 步骤1：基础校验与顶点位置提取 --------------------------
    // 1.2 提取顶点v_idx的当前位置（偏移块的球心）
    const Vector3& x_v = positions(v_idx);

    // -------------------------- 步骤2：距离约束验证（||x_c - x_v|| ≤ ogc_r） --------------------------
    Vector3 vec_cv = x_c - x_v;             // x_c到x_v的向量
    float dist_sq  = vec_cv.squaredNorm();  // 平方距离（避免开方，提升GPU效率）
    float r_sq     = ogc_r * ogc_r;         // 接触半径的平方

    // 容忍微小浮点误差（如GPU精度导致的略超范围），平方距离>r²+1e-12视为超出
    if(dist_sq > r_sq + 1e-12f)
        return false;

    // -------------------------- 步骤3：遍历所有相邻顶点，验证方向约束 --------------------------
    // 3.1 从CSR结构中解析顶点v_idx的邻居范围（start=起始索引，end=结束索引）
    const int start = static_cast<int>(vadj_offsets(v_idx));
    const int end = static_cast<int>(vadj_offsets(v_idx + 1));  // CSR偏移的“下一个顶点起始”即当前结束

    // 3.2 若无相邻顶点（孤立顶点），仅需满足距离约束（特殊场景，如单个顶点碰撞）
    if(start >= end)
        return true;

    // 3.3 对每个相邻顶点v_prime，验证方向约束：(x_c - x_v) · (x_v - x_v_prime) ≥ 0
    // （含义：x_c在“x_v指向v_prime”的反方向侧，避免进入邻居的偏移块重叠区）
    for(int p = start; p < end; ++p)
    {
        // 3.3.1 提取相邻顶点ID（注意IndexT到int的转换，适配muda容器）
        int v_prime = static_cast<int>(vadj_indices(p));
        // 跳过自身（理论上CSR邻接表不含自身，此处防异常）
        if(v_prime == v_idx)
            continue;

        // 3.3.2 提取相邻顶点v_prime的位置
        const Vector3& x_vp = positions(v_prime);

        // 3.3.3 计算方向约束的向量与点积
        Vector3 vec_v_vp = x_v - x_vp;  // x_v到v_prime的向量（反方向）
        float   dot      = vec_cv.dot(vec_v_vp);  // 点积判断方向

        // 浮点误差容忍：点积<0视为不满足（避免GPU精度导致的误判）
        if(dot < 0)  // 放宽1e-6容差，兼容微小浮点波动
            return false;
    }

    // -------------------------- 步骤4：退化情况补充校验（多邻居重合） --------------------------
    // 若存在多个相邻顶点位置重合（如退化边），视为无效偏移块
    Vector3 first_neighbor_pos = positions(static_cast<int>(vadj_indices(start)));
    for(int p = start + 1; p < end; ++p)
    {
        int     v_prime           = static_cast<int>(vadj_indices(p));
        Vector3 curr_neighbor_pos = positions(v_prime);
        // 相邻顶点位置重合（平方距离<1e-24，视为完全重合）
        if((curr_neighbor_pos - first_neighbor_pos).squaredNorm() < 1e-24f)
            return false;
    }

    // -------------------------- 所有约束满足 --------------------------
    return true;
}


//__device__ bool checkVertexFeasibleRegionEdgeOffset(const Vector3& x_c,
//                                                    int            v_idx,
//                                                    const muda::CDense1D<Vector3>& positions,
//                                                    const muda::CDense1D<Vector2i>& surf_edges,
//                                                    const muda::CDense1D<Vector2i>& v_edge_indices,
//                                                    float ogc_r)
//{
//    // -------------------------- 步骤1：获取顶点v的位置和相邻顶点 --------------------------
//    const Vector3& x_v = positions(v_idx);  // 容器索引改用()
//
//    // 提取顶点v在边-only流形中所属的两条边
//    Vector2i v_edges = v_edge_indices(v_idx);           // 容器索引改用()
//    int      e1_idx = v_edges[0], e2_idx = v_edges[1];  // Vector2i内部仍用[]
//
//    // 从边e1中提取相邻顶点v1（排除v自身）
//    Vector2i e1_vs = surf_edges(e1_idx);                 // 容器索引改用()
//    int v1 = (e1_vs[0] == v_idx) ? e1_vs[1] : e1_vs[0];  // Vector2i内部仍用[]
//
//    // 从边e2中提取相邻顶点v2（排除v自身）
//    Vector2i e2_vs = surf_edges(e2_idx);                 // 容器索引改用()
//    int v2 = (e2_vs[0] == v_idx) ? e2_vs[1] : e2_vs[0];  // Vector2i内部仍用[]
//
//    // 退化情况：两个相邻顶点重合（边为点，无意义）
//    if(v1 == v2 || (positions(v1) - positions(v2)).squaredNorm() < 1e-12f)  // 容器索引改用()
//        return false;
//
//    // -------------------------- 步骤2：验证距离约束（||x_c - x_v|| ≤ r） --------------------------
//    Vector3 vec_cv  = x_c - x_v;
//    float   dist_sq = vec_cv.squaredNorm();
//    if(dist_sq > ogc_r * ogc_r + 1e-12f)  // 平方距离 > r²，不满足
//        return false;
//
//    // -------------------------- 步骤3：验证方向约束（两个相邻顶点） --------------------------
//    // 3.1 与v1的方向约束：(x_c - x_v) · (x_v - x_v1) ≥ 0
//    Vector3 vec_vv1 = x_v - positions(v1);  // 容器索引改用()
//    float   dot1    = vec_cv.dot(vec_vv1);
//
//    // 3.2 与v2的方向约束：(x_c - x_v) · (x_v - x_v2) ≥ 0
//    Vector3 vec_vv2 = x_v - positions(v2);  // 容器索引改用()
//    float   dot2    = vec_cv.dot(vec_vv2);
//
//    // 浮点误差容忍：点积 < 0 视为不满足（避免因精度误判）
//    if(dot1 < 0 || dot2 < 0)
//        return false;
//
//    // -------------------------- 所有约束满足 --------------------------
//    return true;
//}


//__device__ bool checkVertexFeasibleRegionEdgeOffset(
//    const Vector3&                 x_c,
//    int                            v_idx,
//    const muda::CDense1D<Vector3>& positions,
//    const muda::CDense1D<IndexT>& vadj_offsets,  // 新增：CSR偏移（替代v_edge_indices）
//    const muda::CDense1D<IndexT>& vadj_indices,  // 新增：CSR索引（替代surf_edges）
//    float ogc_r)
    
__device__ int find_edge_id(int             a,
                            int             b,
                            const muda::CDense1D<IndexT>&   v_edge_offsets,
                            const muda::CDense1D<IndexT>&   v_edge_indices,
                            const muda::CDense1D<Vector2i>& edges)
{
    if(a == b)
        return -1;
    int va = a;
    int vb = b;
    // 遍历 a 的邻接边
    for(IndexT k = v_edge_offsets(va); k < v_edge_offsets(va + 1); ++k)
    {
        IndexT e  = v_edge_indices(k);
        auto   e2 = edges(e);  // (v0,v1)
        int    v0 = e2[0];
        int    v1 = e2[1];
        if((v0 == va && v1 == vb) || (v0 == vb && v1 == va))
            return static_cast<int>(e);
    }
    return -1;
}

void LBVHSimplexTrajectoryFilter::Impl::phase1_vertex_facet_contact(DetectInfo& info)
{
    using namespace muda;
    auto Vs                 = info.surf_vertices();
    auto Fs                 = info.surf_triangles();
    auto Es                 = info.surf_edges();
    auto positions          = info.positions();
    ///////////this is not added yet
    // ///////
    //auto edge_face_counts   = info.edge_face_counts();    // 原有边-面关联数量
    //auto edge_face_vertices = info.edge_face_vertices();  // 原有边-面关联顶点

    // 候选对数量：每个候选对是 (v_idx_in_Vs, t_idx)（Vs 是表面顶点列表，v_idx_in_Vs 是 Vs 的索引）
    SizeT num_candidates = candidate_AllP_AllT_pairs.size();
    if(num_candidates == 0)
        return;

    // 初始化临时接触集合（清空上一帧数据）
    temp_PTs.resize(num_candidates);
    temp_PEs.resize(num_candidates);
    temp_PPs.resize(num_candidates);

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(num_candidates,
               [  // 输入参数（OGC 配置）
                   ogc_r  = m_ogc_r,
                   ogc_rq = m_ogc_rq,
                   // 输入参数（几何数据）
                   Vs        = Vs.cviewer().name("Vs"),
                   Fs        = Fs.cviewer().name("Fs"),
                   Es        = Es.cviewer().name("Es"),  // 新增
                   edge_face_counts = m_edge_face_counts.cviewer().name("edge_face_counts"),
                   edge_face_vertices = m_edge_face_vertices.cviewer().name("edge_face_vertices"),
                   //m_edge_id_map = m_edge_id_map.cviewer().name("edge_id_map"),
                   positions = positions.cviewer().name("positions"),
                   // 顶点→邻接边 CSR（新增）
                   v_edge_offsets = m_v_edge_offsets.cviewer().name("v_edge_offsets"),
                   v_edge_indices = m_v_edge_indices.cviewer().name("v_edge_indices"),
                   // 输入参数（邻接关系）
   /*                vadj_offsets = m_vadj_offsets.cviewer().name("vadj_offsets"),
                   vadj_indices = m_vadj_indices.cviewer().name("vadj_indices"),*/
                   //vadj_offsets = m_v_edge_offsets.cviewer().name("vadj_offsets"),
                   //vadj_indices = m_v_edge_indices.cviewer().name("vadj_indices"),

                   vertex_offsets = m_v_vertex_offsets.cviewer().name("vadj_offsets"),
                   vertex_indices = m_v_vertex_indices.cviewer().name("vadj_indices"),

                   // 输入参数（候选对）
                   candidates = candidate_AllP_AllT_pairs.viewer().name("candidates"),
                   // 输出参数（接触集合）
                   temp_PTs = temp_PTs.viewer().name("temp_PTs"),
                   temp_PEs = temp_PEs.viewer().name("temp_PEs"),
                   temp_PPs = temp_PPs.viewer().name("temp_PPs"),
                   // 输出参数（最小距离）
                   d_min_v = m_d_min_v.viewer().name("d_min_v"),
                   d_min_t = m_d_min_t.viewer().name("d_min_t")] __device__(int idx) mutable
               {
                   // 1. 解析候选对：(v_idx_in_Vs, t_idx) → 全局顶点ID v，面ID t
                   Vector2i cand        = candidates(idx);
                   int      v_idx_in_Vs = cand[0];
                   int      t_idx       = cand[1];
                   int v = Vs(v_idx_in_Vs);    // 全局顶点ID（positions 的索引）
                   Vector3i t_vs = Fs(t_idx);  // 面t的三个顶点（v0, v1, v2）

                   // 2. Algorithm 1 第3行：跳过 v 所属的面（v ⊂ t）
                   if(t_vs[0] == v || t_vs[1] == v || t_vs[2] == v)
                   {
                       temp_PTs(idx).setConstant(-1);
                       temp_PEs(idx).setConstant(-1);
                       temp_PPs(idx).setConstant(-1);
                       return;
                   }

                   // 3. Algorithm 1 第4行：计算顶点v到面t的距离 d
                   const Vector3& v_pos    = positions(v);
                   const Vector3& t_v0_pos = positions(t_vs[0]);
                   const Vector3& t_v1_pos = positions(t_vs[1]);
                   const Vector3& t_v2_pos = positions(t_vs[2]);

                   //distance::point_triangle_distance_flag
                   Vector4i flag = distance::point_triangle_distance_flag(
                       v_pos, t_v0_pos, t_v1_pos, t_v2_pos);

                   /////////edge edge 传入的永远是顶点位置，里面好像不涉及到id？？？？？？？
                   Float d_square;
                   distance::point_triangle_distance2(
                       flag, v_pos, t_v0_pos, t_v1_pos, t_v2_pos, d_square);
                   Float d_update = sqrtf(d_square);

                   //////////////////////please note that we use d_square instead of d!!!!!!!!!!!!!!!!!!!!!!!
                   //// 4. Algorithm 1 第5行：更新 d_min_v（顶点v到面的最小距离）
                   //atomic_min(&d_min_v(v), d_update);
                   //// 5. Algorithm 1 第6行：原子更新 d_min_t（面t到顶点的最小距离，避免多线程竞争）
                   //atomic_min(&d_min_t(t_idx), d_update);

                   // 6. Algorithm 1 第7行：距离 ≥ r，不构成接触，跳过
                   if(d_square >= ogc_r * ogc_r)  // 容忍浮点误差
                   {
                       temp_PTs(idx).setConstant(-1);
                       temp_PEs(idx).setConstant(-1);
                       temp_PPs(idx).setConstant(-1);
                       return;
                   }

                   // 7. Algorithm 1 第8行：找到面t上离v最近的子面 a（顶点/边/面内部\
                   ///// now flag has successfully reflected by the position relation between point and triangle
                   //ContactFace a = find_closest_subface(v_pos, t_idx, Fs, positions);
                   // Now we do not need this corner case, will be tested later
                   //if(a == -1)  // 退化子面（如无效边）
                   //{
                   //    temp_PTs(idx).setConstant(-1);
                   //    temp_PEs(idx).setConstant(-1);
                   //    temp_PPs(idx).setConstant(-1);
                   //    return;
                   //}

                   // 8. Algorithm 1 第9行：过滤重复接触（简化：利用候选对唯一性，避免多线程遍历）
                   // （注：若需严格去重，可新增标记缓冲区，此处省略以简化逻辑）

                   // 计算活跃点数量（dim决定退化类型）
                   int dim = distance::detail::active_count(flag);


                   // 9. Algorithm 1 第10-21行：按子面类型调用 check 函数，验证偏移块
                   bool is_valid = false;
                   Vector4i pt_pair = {-1, -1, -1, -1};  // PT：(v, t_v0, t_v1, t_v2)
                   Vector3i pe_pair = {-1, -1, -1};  // PE：(v, e_v0, e_v1)
                   Vector2i pp_pair = {-1, -1};      // PP：(v, a_vertex)

                   if(dim == 2)  // 退化类型：点到点（子面是三角形的某个顶点）
                   {
                       //// 从flag获取参与计算的两个点的索引（P数组：[v, t0, t1, t2]）
                       Vector2i offsets = distance::detail::pp_from_pt(flag);
                       // 其中offsets[0]应为v（索引0），offsets[1]为三角形的某个顶点（索引1/2/3）
                       int tri_vertex_idx_in_P = offsets[1];  // 三角形顶点在P数组中的索引
                       int a_vertex = t_vs[tri_vertex_idx_in_P - 1];  // 映射到三角形的顶点ID（t_vs是[0:t0,1:t1,2:t2]）

                       is_valid = checkVertexFeasibleRegion(
                           v_pos, a_vertex, positions, vertex_offsets, vertex_indices, ogc_r);
                       if(is_valid)
                       {
                           pp_pair       = {v, a_vertex};  // 记录点-点接触对
                           temp_PPs(idx) = pp_pair;
                       }
                   }
                   else if(dim == 3)  // 退化类型：点到边（子面是三角形的某条边）
                   {
                       // 从flag获取参与计算的三个点的索引（P数组：[v, t0, t1, t2]）
                       Vector3i offsets = distance::detail::pe_from_pt(flag);
                       // 其中offsets[0]应为v（索引0），offsets[1]和offsets[2]为三角形的两个顶点（构成边）
                       int tri_v0_idx_in_P = offsets[1];  // 边的第一个顶点在P数组中的索引
                       int tri_v1_idx_in_P = offsets[2];  // 边的第二个顶点在P数组中的索引
                       Vector2i e_vs = {t_vs[tri_v0_idx_in_P - 1],  // 映射到三角形的顶点ID
                                        t_vs[tri_v1_idx_in_P - 1]};

                       // ... 前面的逻辑 ...
                       int v0 = e_vs[0], v1 = e_vs[1];
                       int edge_id =
                           find_edge_id(v0, v1, v_edge_offsets, v_edge_indices, Es);

                       // 调用checkEdgeFeasibleRegion时传入：
                       is_valid = checkEdgeFeasibleRegion(v_pos,
                                                          positions(v0),
                                                          positions(v1),
                                                          edge_id,
                                                          positions,
                                                          edge_face_counts,  // 直接用设备端成员变量
                                                          edge_face_vertices,
                                                          ogc_r);
                       if(is_valid)
                       {
                           pe_pair       = {v, v0, v1};  // 记录点-边接触对
                           temp_PEs(idx) = pe_pair;
                       }

                       ///////////////////////////////////=====================================
                       //// 验证点-边接触的可行性（假设check函数可直接用顶点对，或传入边ID）
                       //is_valid = checkEdgeFeasibleRegion(
                       //    v_pos, e_vs, Es, positions, edge_face_counts, edge_face_vertices, ogc_r);
                       ////////////////////////////////=========================================
                   }
                   else if(dim == 4)  // 非退化：点到三角形内部（子面是面本身）
                   {
                       is_valid = true;  // 面内部接触无需额外验证
                       pt_pair = {v, t_vs[0], t_vs[1], t_vs[2]};  // 记录点-面接触对
                       temp_PTs(idx) = pt_pair;
                   }
                   else  // 无效flag（理论上不会触发，原距离计算函数已做校验）
                   {
                       temp_PTs(idx).setConstant(-1);
                       temp_PEs(idx).setConstant(-1);
                       temp_PPs(idx).setConstant(-1);
                       return;
                   }
                   //// 10. 写入临时接触集合
                   //temp_PTs(idx) = pt_pair;
                   //temp_PEs(idx) = pe_pair;
                   //temp_PPs(idx) = pp_pair;
               });
}
//
//// 辅助函数：根据两个顶点找边ID（复用 preprocess_adjacency 中构建的映射）
//__device__ int Impl::get_edge_id(int v0, int v1)
//{
//    // 注：需提前构建顶点对→边ID的device映射（如 m_edge_id_map），此处简化
//    if(v0 > v1)
//        std::swap(v0, v1);
//    auto it = m_edge_id_map.find({v0, v1});
//    return it != m_edge_id_map.end() ? it->second : -1;
//}


// 典型的 clamp 函数定义（项目全局）
template <typename T>
__device__ T clamp(T val, T min_val, T max_val)
{
    return val < min_val ? min_val : (val > max_val ? max_val : val);
}
// 辅助函数：计算边-边最近点（对应 Algorithm 2 第8行 C(e,e')）
__device__ Vector3 edge_edge_closest_point(Vector2i e_vs,
                                           Vector2i ep_vs,
                                           const muda::CDense1D<Vector3>& positions)
{
    const Vector3 &p0 = positions(e_vs[0]), p1 = positions(e_vs[1]);
    const Vector3 &q0 = positions(ep_vs[0]), q1 = positions(ep_vs[1]);
    // 标准边-边最近点计算（参数化求解）
    Vector3 d1 = p1 - p0, d2 = q1 - q0, d3 = p0 - q0;
    float a = d1.dot(d1), b = d1.dot(d2), c = d2.dot(d2), e = d1.dot(d3), f = d2.dot(d3);
    float denom = a * c - b * b;
    float s     = 0.5f;  // 默认中点（退化边处理）
    if(denom > 1e-12f)
    {
        s = clamp((b * f - c * e) / denom, 0.0f, 1.0f);
    }
    return p0 + s * d1;
}

void LBVHSimplexTrajectoryFilter::Impl::phase2_edge_edge_contact(DetectInfo& info)
{
    using namespace muda;
    auto Es        = info.surf_edges();
    auto positions = info.positions();

    // 候选对数量：每个候选对是 (e_idx, e_prime_idx)
    SizeT num_candidates = candidate_AllE_AllE_pairs.size();
    if(num_candidates == 0)
        return;

    // 初始化临时边-边接触集合
    temp_EEs.resize(num_candidates);
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(num_candidates,
               [  // 输入参数（OGC 配置）
                   ogc_r  = m_ogc_r,
                   ogc_rq = m_ogc_rq,
                   // 输入参数（几何数据）
                   Es        = Es.viewer().name("Es"),
                   positions = positions.viewer().name("positions"),
                   //v_edge_indices = m_v_edge_indices.viewer().name("v_edge_indices"),
                   //vadj_offsets = m_v_edge_offsets.cviewer().name("vadj_offsets"),
                   //vadj_indices = m_v_edge_indices.cviewer().name("vadj_indices"),

                   vertex_offsets = m_v_vertex_offsets.cviewer().name("vadj_offsets"),
                   vertex_indices = m_v_vertex_indices.cviewer().name("vadj_indices"),
                   // 输入参数（候选对）
                   candidates = candidate_AllE_AllE_pairs.viewer().name("candidates"),
                   // 输出参数（接触集合）
                   temp_EEs = temp_EEs.viewer().name("temp_EEs"),
                   temp_PEs = temp_PEs.viewer().name("temp_PEs"),
                   temp_PPs = temp_PPs.viewer().name("temp_PPs"),
                   // 输出参数（最小距离）
                   d_min_e = m_d_min_e.viewer().name("d_min_e")] __device__(int idx) mutable
               {
                   // 1. 解析候选对：(e_idx, e_prime_idx)
                   Vector2i cand        = candidates(idx);
                   int      e_idx       = cand[0];
                   int      e_prime_idx = cand[1];
                   Vector2i e_vs        = Es(e_idx);  // 边e的顶点（v0, v1）
                   Vector2i ep_vs = Es(e_prime_idx);  // 边e'的顶点（v0', v1'）
                   int      e_v0 = e_vs[0], e_v1 = e_vs[1];
                   int      ep_v0 = ep_vs[0], ep_v1 = ep_vs[1];

                   // 2. Algorithm 2 第5行：跳过相邻边（e 和 e' 有公共顶点）
                   if(e_v0 == ep_v0 || e_v0 == ep_v1 || e_v1 == ep_v0 || e_v1 == ep_v1)
                   {
                       temp_EEs(idx).setConstant(-1);
                       return;
                   }

                   // 3. Algorithm 2 第6行：计算边e到边e'的距离 d
                   const Vector3 &e_p0 = positions(e_v0), e_p1 = positions(e_v1);
                   const Vector3 &ep_p0 = positions(ep_v0), ep_p1 = positions(ep_v1);

                   Vector4i flag = distance::edge_edge_distance_flag(e_p0, e_p1, ep_p0, ep_p1);
                   Float d_square;
                   distance::edge_edge_distance2(
                       flag, e_p0, e_p1, ep_p0, ep_p1, d_square);

                   Float d_update = sqrtf(d_square);
                   // 4. Algorithm 2 第7行：更新 d_min_e（边e到边的最小距离）
                   //atomic_min(&d_min_e(e_idx), d_update);
                   //atomic_min(&d_min_e(e_prime_idx), d_update);  // e' 的最小距离也更新

                   //// 5. Algorithm 1 第6行：原子更新 d_min_t（面t到顶点的最小距离，避免多线程竞争）


                   // 5. Algorithm 2 第8行：距离 ≥ r，不构成接触，跳过
                   if(d_square >= ogc_r * ogc_r)
                   {
                       temp_EEs(idx).setConstant(-1);
                       return;
                   }

                   // 6. Algorithm 2 第8行：计算边-边最近点 x_c（边e上的最近点）
                   Vector3 x_c = edge_edge_closest_point(e_vs, ep_vs, positions);


                   // 计算活跃点数量（dim决定退化类型）
                   int dim = distance::detail::active_count(flag);

                   // 7. 按退化类型（dim）处理接触对，替代原find_edge_closest_subface
                   bool is_valid = false;
                   Vector4i ee_pair = {-1, -1, -1, -1};  // EE：(e_v0, e_v1, ep_v0, ep_v1)
                   Vector2i pp_pair = {-1, -1};  // PP：(a_vertex, ep_vertex)

                   if(dim == 2)  // 完全退化：点-点接触（两条边均退化为点）
                   {
                       // 从flag获取2个有效点的索引（P数组：[ea0,ea1,eb0,eb1]）
                       Vector2i offsets = distance::detail::pp_from_ee(flag);
                       // 映射到原始顶点ID（offsets[0]对应e边的点，offsets[1]对应ep边的点）
                       int a_vertex = (offsets[0] == 0) ? e_v0 : e_v1;  // e边退化后的点
                       int ep_vertex = (offsets[1] == 2) ? ep_v0 : ep_v1;  // ep边退化后的点

                       // 验证顶点可行性（复用原check逻辑）
                       is_valid = checkVertexFeasibleRegionEdgeOffset(
                           x_c, a_vertex, positions, vertex_offsets, vertex_indices, ogc_r);

                       if(is_valid)
                           pp_pair = {a_vertex, ep_vertex};
                   }
                   else if(dim == 3)  // 部分退化：点-边接触（一条边退化，一条边正常）
                   {
                       // 从flag获取3个有效点的索引（1个退化点 + 2个边端点）
                       Vector3i offsets = distance::detail::pe_from_ee(flag);
                       // 判断哪个是退化点（仅出现1次的索引）：0/1属于e边，2/3属于ep边
                       int point_idx = -1, edge_v0_idx = -1, edge_v1_idx = -1;
                       if(offsets[0] == offsets[1])  // 前两个索引相同→e边退化
                       {
                           point_idx   = offsets[0];
                           edge_v0_idx = offsets[1];
                           edge_v1_idx = offsets[2];
                       }
                       else  // 后两个索引相同→ep边退化
                       {
                           point_idx   = offsets[2];
                           edge_v0_idx = offsets[0];
                           edge_v1_idx = offsets[1];
                       }

                       // 映射到原始顶点ID
                       int a_vertex = (point_idx == 0) ? e_v0 :
                                      (point_idx == 1) ? e_v1 :
                                      (point_idx == 2) ? ep_v0 :
                                                         ep_v1;

                       // 验证顶点可行性（复用原check逻辑）
                       //is_valid = checkVertexFeasibleRegionEdgeOffset(
                       //    x_c, a_vertex, positions, Es, v_edge_indices, ogc_r);
                       //is_valid = checkVertexFeasibleRegionEdgeOffset(
                       //    x_c, a_vertex, positions, vadj_offsets, vadj_indices, ogc_r);
                       is_valid = checkVertexFeasibleRegionEdgeOffset(
                           x_c, a_vertex, positions, vertex_offsets, vertex_indices, ogc_r);
                       
                       if(is_valid)
                       {
                           // 点-边接触仍记录为点-点对（匹配原逻辑）
                           int ep_closest_v = (edge_v0_idx == 2) ? ep_v0 :
                                              (edge_v0_idx == 3) ? ep_v1 :
                                              (edge_v0_idx == 0) ? e_v0 :
                                                                   e_v1;
                           pp_pair          = {a_vertex, ep_closest_v};
                       }
                   }
                   else if(dim == 4)  // 非退化：边-边接触（两条边均正常）
                   {
                       is_valid = true;  // 边内部接触无需验证，直接有效
                       ee_pair  = {e_v0, e_v1, ep_v0, ep_v1};
                   }
                   else  // 无效flag（原距离计算函数已校验，理论不触发）
                   {
                       temp_EEs(idx).setConstant(-1);
                       return;
                   }

                   // 8. 写入临时接触集合
                   temp_EEs(idx) = ee_pair;
                   if(is_valid && pp_pair[0] != -1)
                       temp_PPs(idx) = pp_pair;  // 顶点-顶点接触写入PP集合
               });
}

void LBVHSimplexTrajectoryFilter::Impl::compute_conservative_bounds(int num_vertices, float gamma_p)
{
    // 初始化bv缓冲区（长度=顶点数）
    m_bv.resize(num_vertices);

    // 并行计算每个顶点的bv（每个线程处理一个顶点v）
    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(num_vertices,
               [  // 输入：CSR邻接表（顶点-邻居边/面）
                   v_edge_offsets = m_v_edge_offsets.cviewer().name("v_edge_offsets"),
                   v_edge_indices = m_v_edge_indices.cviewer().name("v_edge_indices"),
                   v_face_offsets = m_v_face_offsets.cviewer().name("v_face_offsets"),
                   v_face_indices = m_v_face_indices.cviewer().name("v_face_indices"),
                   // 输入：碰撞检测更新的最小距离
                   d_min_v = m_d_min_v.viewer().name("d_min_v"),
                   d_min_e = m_d_min_e.viewer().name("d_min_e"),
                   d_min_t = m_d_min_t.viewer().name("d_min_t"),
                   // 输入：参数
                   gamma = gamma_p,
                   // 输出：保守边界bv
                   bv = m_bv.viewer().name("bv"),
                   cout = KernelCout::viewer()] __device__(int v) mutable
               {
                   // -------------------------- 步骤1：获取d_min_v[v]（顶点到其他面的最小距离） --------------------------
                   float min_v = d_min_v(v);  // phase1中更新的d_min_v

                   // -------------------------- 步骤2：计算d_min_v^E（顶点邻居边到其他边的最小距离的最小值） --------------------------
                   float min_v_E = FLT_MAX;  // 初始化为最大浮点数
                   int   e_start = v_edge_offsets(v);
                   int   e_end   = v_edge_offsets(v + 1);
                   for(int p = e_start; p < e_end; ++p)
                   {
                       int e = v_edge_indices(p);  // 顶点v的一个邻居边ID
                       if(d_min_e(e) < min_v_E)
                       {  // 取邻居边d_min_e的最小值
                           min_v_E = d_min_e(e);
                       }
                   }
                   // 边界处理：若没有邻居边（理论不发生），用min_v替代
                   if(min_v_E == FLT_MAX)
                       min_v_E = min_v;

                   // -------------------------- 步骤3：计算d_min_v^T（顶点邻居面到其他顶点的最小距离的最小值） --------------------------
                   float min_v_T = FLT_MAX;  // 初始化为最大浮点数
                   int   t_start = v_face_offsets(v);
                   int   t_end   = v_face_offsets(v + 1);
                   for(int p = t_start; p < t_end; ++p)
                   {
                       int t = v_face_indices(p);  // 顶点v的一个邻居面ID
                       if(d_min_t(t) < min_v_T)
                       {  // 取邻居面d_min_t的最小值
                           min_v_T = d_min_t(t);
                       }
                   }
                   // 边界处理：若没有邻居面（理论不发生），用min_v替代
                   if(min_v_T == FLT_MAX)
                       min_v_T = min_v;

                   // -------------------------- 步骤4：计算bv = γₚ * min(三个最小距离) --------------------------
                   // 取三个值的最小值，避免负距离（浮点误差）
                   //这里再gpu中输出min_v, min_v_E 和min_v_T三个值
                   float min_all = std::min(std::min(min_v, min_v_E), min_v_T);
                   min_all = std::max(min_all, 1e-12f);  // 防止min_all为0或负数（避免bv=0）
                   bv(v) = gamma * min_all;
                   // Throttled debug output from device
                   //if(v < 64)
                   //{
                   //    cout << "bv dbg v=" << v << ", min_v=" << min_v
                   //         << ", min_v_E=" << min_v_E << ", min_v_T=" << min_v_T
                   //         << ", bv010=" << bv(v) << "\n";
                   //}
               });

    //auto gamma0 = gamma_p;
    ////////we need to obtain d_bv from global_vertex_manager, Directly print all entries (may be large)
    //std::vector<Float> h_temp;
    //m_bv.copy_to(h_temp);
    //for(size_t i = 0; i < m_bv.size(); ++i)
    //{
    //    std::cout << "m_bv111: " << h_temp[i] << std::endl;
    //}
}

void LBVHSimplexTrajectoryFilter::do_detect(DetectInfo& info)
{
    //m_impl.init_ogc_data(info);
    m_impl.detect(info);
}

void LBVHSimplexTrajectoryFilter::do_filter_active(FilterActiveInfo& info)
{
    //=============m_impl.detect_ogc_contact(info);
    //m_impl.filter_active_dcd_distance(info);
    //m_impl.compute_conservative_bounds(info.surf_vertices().size(), m_impl.m_gamma_p);
    //=============original filter active does not compute bv
    //这里导致最后的计算结果会有一点不太一样，会是潜在的问题吗??????????
    m_impl.filter_active(info);
}

void LBVHSimplexTrajectoryFilter::do_filter_toi(FilterTOIInfo& info)
{
    m_impl.filter_toi(info);
}


void LBVHSimplexTrajectoryFilter::do_filter_d_v(FilterActiveInfo& info, std::vector<Float>& d_bv)
{
    m_impl.compute_conservative_bounds(info.surf_vertices().size(), m_impl.m_gamma_p);
    d_bv.resize(m_impl.m_bv.size());
    m_impl.m_bv.copy_to(d_bv);
    //m_impl.filter_toi(info);
}

void LBVHSimplexTrajectoryFilter::Impl::detect(DetectInfo& info)
{
    using namespace muda;

    auto alpha   = info.alpha();
    auto Ps      = info.positions();
    auto dxs     = info.displacements();
    auto codimVs = info.codim_vertices();
    auto Vs      = info.surf_vertices();
    auto Es      = info.surf_edges();
    auto Fs      = info.surf_triangles();

    point_aabbs.resize(Vs.size());
    triangle_aabbs.resize(Fs.size());
    edge_aabbs.resize(Es.size());

    // build AABBs for codim vertices
    if(codimVs.size() > 0)
    {
        codim_point_aabbs.resize(codimVs.size());

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(codimVs.size(),
                   [codimVs = codimVs.viewer().name("codimVs"),
                    Ps      = Ps.viewer().name("Ps"),
                    dxs     = dxs.viewer().name("dxs"),
                    aabbs   = codim_point_aabbs.viewer().name("aabbs"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    d_hats = info.d_hats().viewer().name("d_hats"),
                    alpha  = alpha] __device__(int i) mutable
                   {
                       auto vI = codimVs(i);

                       Float thickness       = thicknesses(vI);
                       Float d_hat_expansion = point_dcd_expansion(d_hats(vI));

                       const auto& pos   = Ps(vI);
                       Vector3     pos_t = pos + dxs(vI) * alpha;

                       AABB aabb;
                       aabb.extend(pos).extend(pos_t);

                       Float expand = d_hat_expansion + thickness;

                       aabb.min().array() -= expand;
                       aabb.max().array() += expand;
                       aabbs(i) = aabb;
                   });
    }

    // build AABBs for surf vertices (including codim vertices)
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(Vs.size(),
               [Vs          = Vs.viewer().name("V"),
                dxs         = dxs.viewer().name("dx"),
                Ps          = Ps.viewer().name("Ps"),
                aabbs       = point_aabbs.viewer().name("aabbs"),
                thicknesses = info.thicknesses().viewer().name("thicknesses"),
                d_hats      = info.d_hats().viewer().name("d_hats"),
                alpha       = alpha] __device__(int i) mutable
               {
                   auto vI = Vs(i);

                   Float thickness       = thicknesses(vI);
                   Float d_hat_expansion = point_dcd_expansion(d_hats(vI));

                   const auto& pos   = Ps(vI);
                   Vector3     pos_t = pos + dxs(vI) * alpha;

                   AABB aabb;
                   aabb.extend(pos).extend(pos_t);

                   Float expand = d_hat_expansion + thickness;

                   aabb.min().array() -= expand;
                   aabb.max().array() += expand;
                   aabbs(i) = aabb;
               });

    // build AABBs for edges
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(Es.size(),
               [Es          = Es.viewer().name("E"),
                Ps          = Ps.viewer().name("Ps"),
                aabbs       = edge_aabbs.viewer().name("aabbs"),
                dxs         = dxs.viewer().name("dx"),
                thicknesses = info.thicknesses().viewer().name("thicknesses"),
                d_hats      = info.d_hats().viewer().name("d_hats"),
                alpha       = alpha] __device__(int i) mutable
               {
                   auto eI = Es(i);

                   Float thickness =
                       edge_thickness(thicknesses(eI[0]), thicknesses(eI[1]));
                   Float d_hat_expansion =
                       edge_dcd_expansion(d_hats(eI[0]), d_hats(eI[1]));

                   const auto& pos0   = Ps(eI[0]);
                   const auto& pos1   = Ps(eI[1]);
                   Vector3     pos0_t = pos0 + dxs(eI[0]) * alpha;
                   Vector3     pos1_t = pos1 + dxs(eI[1]) * alpha;

                   Vector3 max = pos0_t;
                   Vector3 min = pos0_t;

                   AABB aabb;

                   aabb.extend(pos0).extend(pos1).extend(pos0_t).extend(pos1_t);

                   Float expand = d_hat_expansion + thickness;

                   aabb.min().array() -= expand;
                   aabb.max().array() += expand;
                   aabbs(i) = aabb;
               });

    // build AABBs for triangles
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(Fs.size(),
               [Fs          = Fs.viewer().name("F"),
                Ps          = Ps.viewer().name("Ps"),
                aabbs       = triangle_aabbs.viewer().name("aabbs"),
                dxs         = dxs.viewer().name("dx"),
                thicknesses = info.thicknesses().viewer().name("thicknesses"),
                d_hats      = info.d_hats().viewer().name("d_hats"),
                alpha       = alpha] __device__(int i) mutable
               {
                   auto fI = Fs(i);

                   Float thickness = triangle_thickness(thicknesses(fI[0]),
                                                        thicknesses(fI[1]),
                                                        thicknesses(fI[2]));
                   Float d_hat_expansion = triangle_dcd_expansion(
                       d_hats(fI[0]), d_hats(fI[1]), d_hats(fI[2]));

                   const auto& pos0   = Ps(fI[0]);
                   const auto& pos1   = Ps(fI[1]);
                   const auto& pos2   = Ps(fI[2]);
                   Vector3     pos0_t = pos0 + dxs(fI[0]) * alpha;
                   Vector3     pos1_t = pos1 + dxs(fI[1]) * alpha;
                   Vector3     pos2_t = pos2 + dxs(fI[2]) * alpha;

                   AABB aabb;

                   aabb.extend(pos0)
                       .extend(pos1)
                       .extend(pos2)
                       .extend(pos0_t)
                       .extend(pos1_t)
                       .extend(pos2_t);

                   Float expand = d_hat_expansion + thickness;

                   aabb.min().array() -= expand;
                   aabb.max().array() += expand;
                   aabbs(i) = aabb;
               });

    lbvh_E.build(edge_aabbs);
    lbvh_T.build(triangle_aabbs);

    if(codimVs.size() > 0)
    {
        // Use AllP to query CodimP
        {
            lbvh_CodimP.build(codim_point_aabbs);

            muda::KernelLabel label{__FUNCTION__, __FILE__, __LINE__};
            lbvh_CodimP.query(
                point_aabbs,                                  // AllP
                [Vs      = Vs.viewer().name("Vs"),            // AllP
                 codimVs = codimVs.viewer().name("codimVs"),  // CodimP

                 Ps          = Ps.viewer().name("Ps"),
                 dxs         = dxs.viewer().name("dxs"),
                 thicknesses = info.thicknesses().viewer().name("thicknesses"),
                 dimensions  = info.dimensions().viewer().name("dimensions"),
                 contact_element_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                 contact_mask_tabular = info.contact_mask_tabular().viewer().name("contact_mask_tabular"),
                 subscene_element_ids = info.subscene_element_ids().viewer().name("subscene_element_ids"),
                 subscene_mask_tabular = info.subscene_mask_tabular().viewer().name("subscene_mask_tabular"),
                 v2b = info.v2b().viewer().name("v2b"),
                 body_self_collision = info.body_self_collision().viewer().name("body_self_collision"),
                 d_hats = info.d_hats().viewer().name("d_hats"),
                 alpha  = alpha] __device__(IndexT i, IndexT j)
                {
                    const auto& V      = Vs(i);
                    const auto& codimV = codimVs(j);

                    Vector2i cids = {contact_element_ids(V), contact_element_ids(codimV)};
                    Vector2i scids = {subscene_element_ids(V), subscene_element_ids(codimV)};

                    // discard if the contact is disabled
                    if(!allow_PP_contact(subscene_mask_tabular, scids))
                        return false;
                    if(!allow_PP_contact(contact_mask_tabular, cids))
                        return false;

                    bool V_is_codim = dimensions(V) <= 2;  // codim 0D vert and vert from codim 1D edge

                    if(V_is_codim && V >= codimV)  // avoid duplicate CodimP-CodimP pairs
                        return false;

                    auto body_i = v2b(V);
                    auto body_j = v2b(codimV);
                    // skip self-collision for the same body if self collision off
                    if(body_i == body_j && !body_self_collision(body_i))
                        return false;


                    Vector3 P0  = Ps(V);
                    Vector3 dP0 = alpha * dxs(V);

                    Vector3 P1  = Ps(codimV);
                    Vector3 dP1 = alpha * dxs(codimV);

                    Float thickness = PP_thickness(thicknesses(V), thicknesses(codimV));
                    Float d_hat = PP_d_hat(d_hats(V), d_hats(codimV));

                    Float expand = d_hat + thickness;

                    if(!distance::point_point_ccd_broadphase(P0, P1, dP0, dP1, expand))
                        return false;

                    return true;
                },
                candidate_AllP_CodimP_pairs);
        }

        // Use CodimP to query AllE
        {
            muda::KernelLabel label{__FUNCTION__, __FILE__, __LINE__};
            lbvh_E.query(
                codim_point_aabbs,
                [codimVs     = codimVs.viewer().name("Vs"),
                 Es          = Es.viewer().name("Es"),
                 Ps          = Ps.viewer().name("Ps"),
                 dxs         = dxs.viewer().name("dxs"),
                 thicknesses = info.thicknesses().viewer().name("thicknesses"),
                 contact_element_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                 contact_mask_tabular = info.contact_mask_tabular().viewer().name("contact_mask_tabular"),
                 subscene_element_ids = info.subscene_element_ids().viewer().name("subscene_element_ids"),
                 subscene_mask_tabular = info.subscene_mask_tabular().viewer().name("subscene_mask_tabular"),
                 v2b = info.v2b().viewer().name("v2b"),
                 body_self_collision = info.body_self_collision().viewer().name("body_self_collision"),
                 d_hats = info.d_hats().viewer().name("d_hats"),
                 alpha  = alpha] __device__(IndexT i, IndexT j)
                {
                    const auto& codimV = codimVs(i);
                    const auto& E      = Es(j);

                    Vector3i cids = {contact_element_ids(codimV),
                                     contact_element_ids(E[0]),
                                     contact_element_ids(E[1])};

                    Vector3i scids = {subscene_element_ids(codimV),
                                      subscene_element_ids(E[0]),
                                      subscene_element_ids(E[1])};

                    // discard if the contact is disabled
                    if(!allow_PE_contact(subscene_mask_tabular, scids))
                        return false;
                    if(!allow_PE_contact(contact_mask_tabular, cids))
                        return false;

                    // discard if the vertex is on the edge
                    if(E[0] == codimV || E[1] == codimV)
                        return false;

                    auto body_i = v2b(codimV);
                    auto body_j = v2b(E[0]);
                    // skip self-collision for the same body if self collision off
                    if(body_i == body_j && !body_self_collision(body_i))
                        return false;

                    Vector3 E0  = Ps(E[0]);
                    Vector3 E1  = Ps(E[1]);
                    Vector3 dE0 = alpha * dxs(E[0]);
                    Vector3 dE1 = alpha * dxs(E[1]);

                    Vector3 P  = Ps(codimV);
                    Vector3 dP = alpha * dxs(codimV);

                    Float thickness = PE_thickness(thicknesses(codimV),
                                                   thicknesses(E[0]),
                                                   thicknesses(E[1]));
                    Float d_hat = PE_d_hat(d_hats(codimV), d_hats(E[0]), d_hats(E[1]));

                    Float expand = d_hat + thickness;

                    if(!distance::point_edge_ccd_broadphase(P, E0, E1, dP, dE0, dE1, expand))
                        return false;

                    return true;
                },
                candidate_CodimP_AllE_pairs);
        }
    }

    // Use AllE to query AllE
    {
        muda::KernelLabel label{__FUNCTION__, __FILE__, __LINE__};
        lbvh_E.detect(
            [Es          = Es.viewer().name("Es"),
             Ps          = Ps.viewer().name("Ps"),
             dxs         = dxs.viewer().name("dxs"),
             thicknesses = info.thicknesses().viewer().name("thicknesses"),
             contact_element_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
             contact_mask_tabular = info.contact_mask_tabular().viewer().name("contact_mask_tabular"),
             subscene_element_ids = info.subscene_element_ids().viewer().name("subscene_element_ids"),
             subscene_mask_tabular = info.subscene_mask_tabular().viewer().name("subscene_mask_tabular"),
             v2b = info.v2b().viewer().name("v2b"),
             body_self_collision = info.body_self_collision().viewer().name("body_self_collision"),
             d_hats = info.d_hats().viewer().name("d_hats"),
             alpha  = alpha] __device__(IndexT i, IndexT j)
            {
                const auto& E0 = Es(i);
                const auto& E1 = Es(j);

                Vector4i cids = {contact_element_ids(E0[0]),
                                 contact_element_ids(E0[1]),
                                 contact_element_ids(E1[0]),
                                 contact_element_ids(E1[1])};

                Vector4i scids = {subscene_element_ids(E0[0]),
                                  subscene_element_ids(E0[1]),
                                  subscene_element_ids(E1[0]),
                                  subscene_element_ids(E1[1])};

                // discard if the contact is disabled
                if(!allow_EE_contact(subscene_mask_tabular, scids))
                    return false;
                if(!allow_EE_contact(contact_mask_tabular, cids))
                    return false;

                // discard if the edges share same vertex
                if(E0[0] == E1[0] || E0[0] == E1[1] || E0[1] == E1[0] || E0[1] == E1[1])
                    return false;

                auto body_i = v2b(E0[0]);
                auto body_j = v2b(E1[0]);
                if(body_i == body_j && !body_self_collision(body_i))
                    return false;  // skip self-collision for the same body


                Vector3 E0_0  = Ps(E0[0]);
                Vector3 E0_1  = Ps(E0[1]);
                Vector3 dE0_0 = alpha * dxs(E0[0]);
                Vector3 dE0_1 = alpha * dxs(E0[1]);

                Vector3 E1_0  = Ps(E1[0]);
                Vector3 E1_1  = Ps(E1[1]);
                Vector3 dE1_0 = alpha * dxs(E1[0]);
                Vector3 dE1_1 = alpha * dxs(E1[1]);

                Float thickness = EE_thickness(thicknesses(E0[0]),
                                               thicknesses(E0[1]),
                                               thicknesses(E1[0]),
                                               thicknesses(E1[1]));

                Float d_hat =
                    EE_d_hat(d_hats(E0[0]), d_hats(E0[1]), d_hats(E1[0]), d_hats(E1[1]));

                Float expand = d_hat + thickness;

                if(!distance::edge_edge_ccd_broadphase(
                       E0_0, E0_1, E1_0, E1_1, dE0_0, dE0_1, dE1_0, dE1_1, expand))
                    return false;

                return true;
            },
            candidate_AllE_AllE_pairs);
    }

    // Use AllP to query AllT
    {
        muda::KernelLabel label{__FUNCTION__, __FILE__, __LINE__};
        lbvh_T.query(
            point_aabbs,
            [Vs          = Vs.viewer().name("Vs"),
             Fs          = Fs.viewer().name("Fs"),
             Ps          = Ps.viewer().name("Ps"),
             dxs         = dxs.viewer().name("dxs"),
             thicknesses = info.thicknesses().viewer().name("thicknesses"),
             contact_element_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
             contact_mask_tabular = info.contact_mask_tabular().viewer().name("contact_mask_tabular"),
             subscene_element_ids = info.subscene_element_ids().viewer().name("subscene_element_ids"),
             subscene_mask_tabular = info.subscene_mask_tabular().viewer().name("subscene_mask_tabular"),
             v2b = info.v2b().viewer().name("v2b"),
             body_self_collision = info.body_self_collision().viewer().name("body_self_collision"),
             d_hats = info.d_hats().viewer().name("d_hats"),
             alpha  = alpha] __device__(IndexT i, IndexT j)
            {
                auto V = Vs(i);
                auto F = Fs(j);

                Vector4i cids = {contact_element_ids(V),
                                 contact_element_ids(F[0]),
                                 contact_element_ids(F[1]),
                                 contact_element_ids(F[2])};

                Vector4i scids = {subscene_element_ids(V),
                                  subscene_element_ids(F[0]),
                                  subscene_element_ids(F[1]),
                                  subscene_element_ids(F[2])};

                // discard if the contact is disabled
                if(!allow_PT_contact(subscene_mask_tabular, scids))
                    return false;
                if(!allow_PT_contact(contact_mask_tabular, cids))
                    return false;

                // discard if the point is on the triangle
                if(F[0] == V || F[1] == V || F[2] == V)
                    return false;

                auto body_i = v2b(V);
                auto body_j = v2b(F[0]);
                // skip self-collision for the same body if self collision off
                if(body_i == body_j && !body_self_collision(body_i))
                    return false;


                Vector3 P  = Ps(V);
                Vector3 dP = alpha * dxs(V);

                Vector3 F0 = Ps(F[0]);
                Vector3 F1 = Ps(F[1]);
                Vector3 F2 = Ps(F[2]);

                Vector3 dF0 = alpha * dxs(F[0]);
                Vector3 dF1 = alpha * dxs(F[1]);
                Vector3 dF2 = alpha * dxs(F[2]);

                Float thickness = PT_thickness(thicknesses(V),
                                               thicknesses(F[0]),
                                               thicknesses(F[1]),
                                               thicknesses(F[2]));

                Float d_hat =
                    PT_d_hat(d_hats(V), d_hats(F[0]), d_hats(F[1]), d_hats(F[2]));

                Float expand = d_hat + thickness;

                if(!distance::point_triangle_ccd_broadphase(P, F0, F1, F2, dP, dF0, dF1, dF2, expand))
                    return false;

                return true;
            },
            candidate_AllP_AllT_pairs);
    }
}

void LBVHSimplexTrajectoryFilter::Impl::filter_active(FilterActiveInfo& info)
{
    using namespace muda;

    // we will filter-out the active pairs
    auto positions = info.positions();

    SizeT N_PCoimP  = candidate_AllP_CodimP_pairs.size();
    SizeT N_CodimPE = candidate_CodimP_AllE_pairs.size();
    SizeT N_PTs     = candidate_AllP_AllT_pairs.size();
    SizeT N_EEs     = candidate_AllE_AllE_pairs.size();

    // PT, EE, PT, PP can degenerate to PP
    temp_PPs.resize(N_PCoimP + N_CodimPE + N_PTs + N_EEs);
    // PT, EE, PT can degenerate to PE
    temp_PEs.resize(N_CodimPE + N_PTs + N_EEs);

    temp_PTs.resize(N_PTs);
    temp_EEs.resize(N_EEs);

    SizeT temp_PP_offset = 0;
    SizeT temp_PE_offset = 0;

    // AllP and CodimP
    if(N_PCoimP > 0)
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_PCoimP);

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(candidate_AllP_CodimP_pairs.size(),
                   [positions = positions.viewer().name("positions"),
                    PCodimP_pairs = candidate_AllP_CodimP_pairs.viewer().name("PP_pairs"),
                    surf_vertices = info.surf_vertices().viewer().name("surf_vertices"),
                    codim_vertices = info.codim_vertices().viewer().name("codim_vertices"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    temp_PPs = PP_view.viewer().name("temp_PPs"),
                    d_hats = info.d_hats().viewer().name("d_hats")] __device__(int i) mutable
                   {
                       // default invalid
                       auto& PP = temp_PPs(i);
                       PP.setConstant(-1);

                       Vector2i indices = PCodimP_pairs(i);

                       IndexT P0 = surf_vertices(indices(0));
                       IndexT P1 = codim_vertices(indices(1));


                       const auto& V0 = positions(P0);
                       const auto& V1 = positions(P1);

                       Float thickness = PP_thickness(thicknesses(P0), thicknesses(P1));
                       Float d_hat = PP_d_hat(d_hats(P0), d_hats(P1));

                       Vector2 range = D_range(thickness, d_hat);

                       Float D;
                       distance::point_point_distance2(V0, V1, D);


                       if(!is_active_D(range, D))
                           return;  // early return

                       PP = {P0, P1};
                   });

        temp_PP_offset += N_PCoimP;
    }
    // CodimP and AllE
    if(N_CodimPE > 0)
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_CodimPE);
        auto PE_view = temp_PEs.view(temp_PE_offset, N_CodimPE);

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(
                candidate_CodimP_AllE_pairs.size(),
                [positions = positions.viewer().name("positions"),
                 CodimP_AllE_pairs = candidate_CodimP_AllE_pairs.viewer().name("PE_pairs"),
                 codim_veritces = info.codim_vertices().viewer().name("codim_vertices"),
                 surf_edges  = info.surf_edges().viewer().name("surf_edges"),
                 thicknesses = info.thicknesses().viewer().name("thicknesses"),
                 temp_PPs    = PP_view.viewer().name("temp_PPs"),
                 temp_PEs    = PE_view.viewer().name("temp_PEs"),
                 d_hats = info.d_hats().viewer().name("d_hats")] __device__(int i) mutable
                {
                    auto& PP = temp_PPs(i);
                    PP.setConstant(-1);
                    auto& PE = temp_PEs(i);
                    PE.setConstant(-1);

                    Vector2i indices = CodimP_AllE_pairs(i);
                    IndexT   V       = codim_veritces(indices(0));
                    Vector2i E       = surf_edges(indices(1));

                    Vector3i vIs = {V, E(0), E(1)};
                    Vector3 Ps[] = {positions(vIs(0)), positions(vIs(1)), positions(vIs(2))};

                    Float thickness = PE_thickness(
                        thicknesses(V), thicknesses(E(0)), thicknesses(E(1)));

                    Float d_hat = PE_d_hat(d_hats(V), d_hats(E(0)), d_hats(E(1)));


                    Vector3i flag =
                        distance::point_edge_distance_flag(Ps[0], Ps[1], Ps[2]);

                    Vector2 range = D_range(thickness, d_hat);

                    Float D;
                    distance::point_edge_distance2(flag, Ps[0], Ps[1], Ps[2], D);

                    if(!is_active_D(range, D))
                        return;  // early return

                    Vector3i offsets;
                    auto dim = distance::degenerate_point_edge(flag, offsets);

                    switch(dim)
                    {
                        case 2:  // PP
                        {
                            IndexT V0 = vIs(offsets(0));
                            IndexT V1 = vIs(offsets(1));
                            PP        = {V0, V1};
                        }
                        break;
                        case 3:  // PE
                        {
                            PE = vIs;
                        }
                        break;
                        default: {
                            MUDA_ERROR_WITH_LOCATION("unexpected degenerate case dim=%d", dim);
                        }
                        break;
                    }
                });

        temp_PP_offset += N_CodimPE;
        temp_PE_offset += N_CodimPE;
    }

    // AllP and AllT
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_PTs);
        auto PE_view = temp_PEs.view(temp_PE_offset, N_PTs);

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(
                candidate_AllP_AllT_pairs.size(),
                [positions = positions.viewer().name("Ps"),
                 PT_pairs = candidate_AllP_AllT_pairs.viewer().name("PT_pairs"),
                 surf_vertices = info.surf_vertices().viewer().name("surf_vertices"),
                 surf_triangles = info.surf_triangles().viewer().name("surf_triangles"),
                 thicknesses = info.thicknesses().viewer().name("thicknesses"),
                 temp_PPs    = PP_view.viewer().name("temp_PPs"),
                 temp_PEs    = PE_view.viewer().name("temp_PEs"),
                 temp_PTs    = temp_PTs.viewer().name("temp_PTs"),
                 d_hats = info.d_hats().viewer().name("d_hats")] __device__(int i) mutable
                {
                    auto& PP = temp_PPs(i);
                    PP.setConstant(-1);
                    auto& PE = temp_PEs(i);
                    PE.setConstant(-1);
                    auto& PT = temp_PTs(i);
                    PT.setConstant(-1);

                    Vector2i indices = PT_pairs(i);
                    IndexT   V       = surf_vertices(indices(0));
                    Vector3i F       = surf_triangles(indices(1));

                    Vector4i vIs  = {V, F(0), F(1), F(2)};
                    Vector3  Ps[] = {positions(vIs(0)),
                                     positions(vIs(1)),
                                     positions(vIs(2)),
                                     positions(vIs(3))};

                    Float thickness = PT_thickness(thicknesses(V),
                                                   thicknesses(F(0)),
                                                   thicknesses(F(1)),
                                                   thicknesses(F(2)));

                    Float d_hat =
                        PT_d_hat(d_hats(V), d_hats(F(0)), d_hats(F(1)), d_hats(F(2)));

                    Vector4i flag =
                        distance::point_triangle_distance_flag(Ps[0], Ps[1], Ps[2], Ps[3]);

                    Vector2 range = D_range(thickness, d_hat);

                    Float D;
                    distance::point_triangle_distance2(flag, Ps[0], Ps[1], Ps[2], Ps[3], D);

                    MUDA_ASSERT(
                        D > 0.0, "D=%f, V F = (%d,%d,%d,%d)", D, vIs(0), vIs(1), vIs(2), vIs(3));

                    if(!is_active_D(range, D))
                        return;  // early return

                    Vector4i offsets;
                    auto dim = distance::degenerate_point_triangle(flag, offsets);

                    switch(dim)
                    {
                        case 2:  // PP
                        {
                            IndexT V0 = vIs(offsets(0));
                            IndexT V1 = vIs(offsets(1));
                            PP        = {V0, V1};
                        }
                        break;
                        case 3:  // PE
                        {
                            IndexT V0 = vIs(offsets(0));
                            IndexT V1 = vIs(offsets(1));
                            IndexT V2 = vIs(offsets(2));
                            PE        = {V0, V1, V2};
                        }
                        break;
                        case 4:  // PT
                        {
                            PT = vIs;
                        }
                        break;
                        default: {
                            MUDA_ERROR_WITH_LOCATION("unexpected degenerate case dim=%d", dim);
                        }
                        break;
                    }
                });

        temp_PP_offset += N_PTs;
        temp_PE_offset += N_PTs;
    }
    // AllE and AllE
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_EEs);
        auto PE_view = temp_PEs.view(temp_PE_offset, N_EEs);

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(
                candidate_AllE_AllE_pairs.size(),
                [positions = positions.viewer().name("Ps"),
                 rest_positions = info.rest_positions().viewer().name("rest_positions"),
                 EE_pairs = candidate_AllE_AllE_pairs.viewer().name("EE_pairs"),
                 surf_edges  = info.surf_edges().viewer().name("surf_edges"),
                 thicknesses = info.thicknesses().viewer().name("thicknesses"),
                 temp_PPs    = PP_view.viewer().name("temp_PPs"),
                 temp_PEs    = PE_view.viewer().name("temp_PEs"),
                 temp_EEs    = temp_EEs.viewer().name("temp_EEs"),
                 d_hats = info.d_hats().viewer().name("d_hats")] __device__(int i) mutable
                {
                    auto& PP = temp_PPs(i);
                    PP.setConstant(-1);
                    auto& PE = temp_PEs(i);
                    PE.setConstant(-1);
                    auto& EE = temp_EEs(i);
                    EE.setConstant(-1);

                    Vector2i indices = EE_pairs(i);
                    Vector2i E0      = surf_edges(indices(0));
                    Vector2i E1      = surf_edges(indices(1));

                    Vector4i vIs  = {E0(0), E0(1), E1(0), E1(1)};
                    Vector3  Ps[] = {positions(vIs(0)),
                                     positions(vIs(1)),
                                     positions(vIs(2)),
                                     positions(vIs(3))};

                    Float thickness = EE_thickness(thicknesses(E0(0)),
                                                   thicknesses(E0(1)),
                                                   thicknesses(E1(0)),
                                                   thicknesses(E1(1)));

                    Float d_hat = EE_d_hat(
                        d_hats(E0(0)), d_hats(E0(1)), d_hats(E1(0)), d_hats(E1(1)));

                    Vector2 range = D_range(thickness, d_hat);

                    Vector4i flag =
                        distance::edge_edge_distance_flag(Ps[0], Ps[1], Ps[2], Ps[3]);

                    Float D;
                    distance::edge_edge_distance2(flag, Ps[0], Ps[1], Ps[2], Ps[3], D);

                    if(!is_active_D(range, D))
                        return;  // early return

                    Float eps_x;
                    distance::edge_edge_mollifier_threshold(rest_positions(vIs(0)),
                                                            rest_positions(vIs(1)),
                                                            rest_positions(vIs(2)),
                                                            rest_positions(vIs(3)),
                                                            eps_x);

                    if(distance::need_mollify(Ps[0], Ps[1], Ps[2], Ps[3], eps_x))
                    {
                        EE = vIs;
                        return;
                    }
                    else  // classify to EE/PE/PP
                    {
                        Vector4i offsets;
                        auto dim = distance::degenerate_edge_edge(flag, offsets);

                        switch(dim)
                        {
                            case 2:  // PP
                            {
                                IndexT V0 = vIs(offsets(0));
                                IndexT V1 = vIs(offsets(1));
                                PP        = {V0, V1};
                            }
                            break;
                            case 3:  // PE
                            {
                                IndexT V0 = vIs(offsets(0));
                                IndexT V1 = vIs(offsets(1));
                                IndexT V2 = vIs(offsets(2));
                                PE        = {V0, V1, V2};
                            }
                            break;
                            case 4:  // EE
                            {
                                EE = vIs;
                            }
                            break;
                            default: {
                                MUDA_ERROR_WITH_LOCATION("unexpected degenerate case dim=%d", dim);
                            }
                            break;
                        }
                    }
                })
            .wait();

        temp_PP_offset += N_EEs;
        temp_PE_offset += N_EEs;
    }

    UIPC_ASSERT(temp_PP_offset == temp_PPs.size(), "size mismatch");
    UIPC_ASSERT(temp_PE_offset == temp_PEs.size(), "size mismatch");

    {  // select the valid ones
        PPs.resize(temp_PPs.size());
        PEs.resize(temp_PEs.size());
        PTs.resize(temp_PTs.size());
        EEs.resize(temp_EEs.size());

        DeviceSelect().If(temp_PPs.data(),
                          PPs.data(),
                          selected_PP_count.data(),
                          temp_PPs.size(),
                          [] CUB_RUNTIME_FUNCTION(const Vector2i& PP)
                          { return PP(0) != -1; });

        DeviceSelect().If(temp_PEs.data(),
                          PEs.data(),
                          selected_PE_count.data(),
                          temp_PEs.size(),
                          [] CUB_RUNTIME_FUNCTION(const Vector3i& PE)
                          { return PE(0) != -1; });

        DeviceSelect().If(temp_PTs.data(),
                          PTs.data(),
                          selected_PT_count.data(),
                          temp_PTs.size(),
                          [] CUB_RUNTIME_FUNCTION(const Vector4i& PT)
                          { return PT(0) != -1; });

        DeviceSelect().If(temp_EEs.data(),
                          EEs.data(),
                          selected_EE_count.data(),
                          temp_EEs.size(),
                          [] CUB_RUNTIME_FUNCTION(const Vector4i& EE)
                          { return EE(0) != -1; });

        IndexT PP_count = selected_PP_count;
        IndexT PE_count = selected_PE_count;
        IndexT PT_count = selected_PT_count;
        IndexT EE_count = selected_EE_count;

        PPs.resize(PP_count);
        PEs.resize(PE_count);
        PTs.resize(PT_count);
        EEs.resize(EE_count);
    }

    info.PPs(PPs);
    info.PEs(PEs);
    info.PTs(PTs);
    info.EEs(EEs);

    if constexpr(PrintDebugInfo)
    {
        std::vector<Vector2i> PPs_host;
        std::vector<Float>    PP_thicknesses_host;

        std::vector<Vector3i> PEs_host;
        std::vector<Float>    PE_thicknesses_host;

        std::vector<Vector4i> PTs_host;
        std::vector<Float>    PT_thicknesses_host;

        std::vector<Vector4i> EEs_host;
        std::vector<Float>    EE_thicknesses_host;

        PPs.copy_to(PPs_host);
        PEs.copy_to(PEs_host);
        PTs.copy_to(PTs_host);
        EEs.copy_to(EEs_host);

        std::cout << "filter result:" << std::endl;

        for(auto&& [PP, thickness] : zip(PPs_host, PP_thicknesses_host))
        {
            std::cout << "PP: " << PP.transpose() << " thickness: " << thickness << "\n";
        }

        for(auto&& [PE, thickness] : zip(PEs_host, PE_thicknesses_host))
        {
            std::cout << "PE: " << PE.transpose() << " thickness: " << thickness << "\n";
        }

        for(auto&& [PT, thickness] : zip(PTs_host, PT_thicknesses_host))
        {
            std::cout << "PT: " << PT.transpose() << " thickness: " << thickness << "\n";
        }

        for(auto&& [EE, thickness] : zip(EEs_host, EE_thicknesses_host))
        {
            std::cout << "EE: " << EE.transpose() << " thickness: " << thickness << "\n";
        }

        std::cout << std::flush;
    }
}

// Fallback for float atomicMin on architectures that don't provide it.
//__device__ inline float atomicMinFloat(float* addr, float value)
//{
//    // Loop with CAS using float comparison; CAS uses bitwise equality on ints.
//    int*  addr_as_i = reinterpret_cast<int*>(addr);
//    float old       = __int_as_float(*addr_as_i);
//    while(value < old)
//    {
//        float assumed = old;
//        int old_i = atomicCAS(addr_as_i, __float_as_int(assumed), __float_as_int(value));
//        if(old_i == __float_as_int(assumed))
//            break;                    // success
//        old = __int_as_float(old_i);  // another thread updated; retry
//    }
//    return old;
//}
//
//__global__ void kernel(float* data, float* result)
//{
//    int   i   = threadIdx.x + blockIdx.x * blockDim.x;
//    float val = data[i];
//
//    // Atomic add is fine on all recent architectures
//    atomicAdd(result, val);
//
//#if __CUDA_ARCH__ >= 700
//    atomicMin(result, val);
//#else
//    atomicMinFloat(result, val);
//#endif
//}

__device__ float atomicMinFloat(float* addr, float value)
{
    if(isnan(value))
        return *addr;  // 忽略 NaN

    int*  addr_i = reinterpret_cast<int*>(addr);
    int   old_i  = *addr_i, assumed_i;
    float old_f;

    do
    {
        assumed_i = old_i;
        old_f     = __int_as_float(assumed_i);
        if(old_f <= value)
            break;
        old_i = atomicCAS(addr_i, assumed_i, __float_as_int(value));
    } while(assumed_i != old_i);

    return old_f;
}


void LBVHSimplexTrajectoryFilter::Impl::filter_active_dcd_distance(FilterActiveInfo& info)
{
    using namespace muda;

    // we will filter-out the active pairs
    auto positions = info.positions();

    SizeT N_PCoimP  = candidate_AllP_CodimP_pairs.size();
    SizeT N_CodimPE = candidate_CodimP_AllE_pairs.size();
    SizeT N_PTs     = candidate_AllP_AllT_pairs.size();
    SizeT N_EEs     = candidate_AllE_AllE_pairs.size();

    // PT, EE, PT, PP can degenerate to PP
    temp_PPs.resize(N_PCoimP + N_CodimPE + N_PTs + N_EEs);
    // PT, EE, PT can degenerate to PE
    temp_PEs.resize(N_CodimPE + N_PTs + N_EEs);

    temp_PTs.resize(N_PTs);
    temp_EEs.resize(N_EEs);

    SizeT temp_PP_offset = 0;
    SizeT temp_PE_offset = 0;

    // AllP and CodimP
    if(N_PCoimP > 0)
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_PCoimP);

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(candidate_AllP_CodimP_pairs.size(),
                   [positions = positions.viewer().name("positions"),
                    PCodimP_pairs = candidate_AllP_CodimP_pairs.viewer().name("PP_pairs"),
                    surf_vertices = info.surf_vertices().viewer().name("surf_vertices"),
                    codim_vertices = info.codim_vertices().viewer().name("codim_vertices"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    temp_PPs = PP_view.viewer().name("temp_PPs"),
                    d_hats = info.d_hats().viewer().name("d_hats"),
                    // 传入OGC依赖的邻接表（顶点-邻居顶点CSR）
                    vertex_offsets = m_v_vertex_offsets.cviewer().name("vertex_offsets"),
                    vertex_indices = m_v_vertex_indices.cviewer().name("vertex_indices"),
                    ogc_r = m_ogc_r,
                    d_min_v = m_d_min_v.viewer().name("d_min_v"),
                    d_min_t = m_d_min_t.viewer().name("d_min_t"),
                    d_min_e = m_d_min_e.viewer().name("d_min_e")
                   ] __device__(int i) mutable
                   {
                       // default invalid
                       auto& PP = temp_PPs(i);
                       PP.setConstant(-1);

                       Vector2i indices = PCodimP_pairs(i);

                       IndexT P0 = surf_vertices(indices(0));
                       IndexT P1 = codim_vertices(indices(1));


                       const auto& V0 = positions(P0);
                       const auto& V1 = positions(P1);

                       Float thickness = PP_thickness(thicknesses(P0), thicknesses(P1));
                       Float d_hat = PP_d_hat(d_hats(P0), d_hats(P1));

                       Vector2 range = D_range(thickness, d_hat);

                       Float D;
                       distance::point_point_distance2(V0, V1, D);


                       if(!is_active_D(range, D))
                           return;  // early return

                       // 【新增OGC逻辑】验证V0是否在P1的偏移块内（点-点偏移块验证）
                       bool is_ogc_valid =
                           checkVertexFeasibleRegion(V0,  // 待检测点（表面顶点P0的位置）
                                                     P1,  // 偏移块顶点（余维顶点P1）
                                                     positions,  // 所有顶点位置
                                                     vertex_offsets,  // 顶点邻接表CSR偏移
                                                     vertex_indices,  // 顶点邻接表CSR索引
                                                     ogc_r  // OGC接触半径
                           );

                       // OGC验证有效才标记为有效接触对
                       if(is_ogc_valid){
                           PP = {P0, P1};
                           // 在“distance::point_point_distance2(V0, V1, D);”之后添加：
                           Float d_update = sqrtf(D);  // 注意：原D是距离平方，需开方
                           // 原子更新：顶点P0到其他点的最小距离
                           //atomicMin(&d_min_v(P0), d_update);
                           atomicMinFloat(&d_min_v(P0), d_update);
                           // after (CUDA atomic)

                           //atomic_min(d_min_v.data() + P0, d_update);
                           //atomicMin(d_min_v.data() + P0, d_update);

                           //atomic_min
                           //atomic_min(&d_min_v(P0), d_update);
                           // 原子更新：顶点P1到其他点的最小距离（如果是余维顶点也需要记录）
                           //atomic_min(&d_min_v(P1), d_update);
                           atomicMinFloat(&d_min_v(P1), d_update);
                       }
                       //PP = {P0, P1};
                   });

        temp_PP_offset += N_PCoimP;
    }
    // CodimP and AllE
    if(N_CodimPE > 0)
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_CodimPE);
        auto PE_view = temp_PEs.view(temp_PE_offset, N_CodimPE);

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(
                candidate_CodimP_AllE_pairs.size(),
                [positions = positions.viewer().name("positions"),
                 CodimP_AllE_pairs = candidate_CodimP_AllE_pairs.viewer().name("PE_pairs"),
                 codim_veritces = info.codim_vertices().viewer().name("codim_vertices"),
                 surf_edges  = info.surf_edges().viewer().name("surf_edges"),
                 thicknesses = info.thicknesses().viewer().name("thicknesses"),
                 temp_PPs    = PP_view.viewer().name("temp_PPs"),
                 temp_PEs    = PE_view.viewer().name("temp_PEs"),
                 d_hats = info.d_hats().viewer().name("d_hats"),
                 // 传入OGC依赖的边-面关联、顶点-邻接边CSR
                 edge_face_counts = m_edge_face_counts.cviewer().name("edge_face_counts"),
                 edge_face_vertices = m_edge_face_vertices.cviewer().name("edge_face_vertices"),
                 v_edge_offsets = m_v_edge_offsets.cviewer().name("v_edge_offsets"),
                 v_edge_indices = m_v_edge_indices.cviewer().name("v_edge_indices"),
                 v_vertex_offsets = m_v_vertex_offsets.cviewer().name("v_vertex_offsets"),
                 v_vertex_indices = m_v_vertex_indices.cviewer().name("v_vertex_indices"),
                 ogc_r = m_ogc_r,
                 d_min_v = m_d_min_v.viewer().name("d_min_v"),
                 d_min_t = m_d_min_t.viewer().name("d_min_t"),
                 d_min_e = m_d_min_e.viewer().name("d_min_e")
                ] __device__(int i) mutable
                {
                    auto& PP = temp_PPs(i);
                    PP.setConstant(-1);
                    auto& PE = temp_PEs(i);
                    PE.setConstant(-1);

                    Vector2i indices = CodimP_AllE_pairs(i);
                    IndexT   V       = codim_veritces(indices(0));
                    Vector2i E       = surf_edges(indices(1));

                    Vector3i vIs = {V, E(0), E(1)};
                    Vector3 Ps[] = {positions(vIs(0)), positions(vIs(1)), positions(vIs(2))};

                    Float thickness = PE_thickness(
                        thicknesses(V), thicknesses(E(0)), thicknesses(E(1)));

                    Float d_hat = PE_d_hat(d_hats(V), d_hats(E(0)), d_hats(E(1)));


                    Vector3i flag =
                        distance::point_edge_distance_flag(Ps[0], Ps[1], Ps[2]);

                    Vector2 range = D_range(thickness, d_hat);

                    Float D;
                    distance::point_edge_distance2(flag, Ps[0], Ps[1], Ps[2], D);

                    if(!is_active_D(range, D))
                        return;  // early return



                    // 【OGC逻辑】退化分类+偏移块验证
                    Vector3i offsets;
                    auto dim = distance::degenerate_point_edge(flag, offsets);
                    bool is_ogc_valid = false;

                    switch(dim)
                    {
                        case 2:  // 退化PP（点-点）
                        {
                            IndexT V0 = vIs(offsets(0));  // 余维顶点
                            IndexT V1 = vIs(offsets(1));  // 边顶点
                            // OGC：验证V0是否在V1的偏移块内
                            is_ogc_valid = checkVertexFeasibleRegion(
                                positions(V0), V1, positions, v_vertex_offsets, v_vertex_indices, ogc_r);
                            if(is_ogc_valid)
                            {
                                PP = {V0, V1};
                                // 在“distance::point_edge_distance2(flag, Ps[0], Ps[1], Ps[2], D);”之后添加：
                                Float d_update = sqrtf(D);
                                // 原子更新：余维顶点V到边的最小距离
                                //atomic_min(&d_min_v(V), d_update);
                                atomicMinFloat(&d_min_v(V), d_update);
                                // 原子更新：边E的最小距离（E是surf_edges(indices(1))，需先获取边ID）
                                int e_idx = indices(1);  // 边在surf_edges中的索引即边ID
                                //atomic_min(&d_min_e(e_idx), d_update);
                                atomicMinFloat(&d_min_e(e_idx), d_update);
                            }
                        }
                        break;
                        case 3:  // 非退化PE（点-边）
                        {
                            int v0 = E(0), v1 = E(1);
                            // OGC：通过顶点对找边ID（复用你写的find_edge_id）
                            int edge_id =
                                find_edge_id(v0, v1, v_edge_offsets, v_edge_indices, surf_edges);
                            if(edge_id == -1)
                                break;
                            // OGC：验证余维顶点V是否在边的偏移块内
                            is_ogc_valid =
                                checkEdgeFeasibleRegion(positions(V),  // 待检测点（余维顶点V）
                                                        positions(v0),  // 边顶点v0位置
                                                        positions(v1),  // 边顶点v1位置
                                                        edge_id,  // 边ID
                                                        positions,
                                                        edge_face_counts,
                                                        edge_face_vertices,
                                                        ogc_r);
                            if(is_ogc_valid)
                            {
                                PE = vIs;
                                // 在“distance::point_edge_distance2(flag, Ps[0], Ps[1], Ps[2], D);”之后添加：
                                Float d_update = sqrtf(D);
                                // 原子更新：余维顶点V到边的最小距离
                                atomicMinFloat(&d_min_v(V), d_update);
                                // 原子更新：边E的最小距离（E是surf_edges(indices(1))，需先获取边ID）
                                int e_idx = indices(1);  // 边在surf_edges中的索引即边ID
                                atomicMinFloat(&d_min_e(e_idx), d_update);
                            }
                        }
                        break;
                        default:
                            MUDA_ERROR_WITH_LOCATION("unexpected degenerate case dim=%d", dim);
                            break;
                    }
                    //Vector3i offsets;
                    //auto dim = distance::degenerate_point_edge(flag, offsets);

                    //switch(dim)
                    //{
                    //    case 2:  // PP
                    //    {
                    //        IndexT V0 = vIs(offsets(0));
                    //        IndexT V1 = vIs(offsets(1));
                    //        PP        = {V0, V1};
                    //    }
                    //    break;
                    //    case 3:  // PE
                    //    {
                    //        PE = vIs;
                    //    }
                    //    break;
                    //    default: {
                    //        MUDA_ERROR_WITH_LOCATION("unexpected degenerate case dim=%d", dim);
                    //    }
                    //    break;
                    //}
                });

        temp_PP_offset += N_CodimPE;
        temp_PE_offset += N_CodimPE;
    }

    // AllP and AllT
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_PTs);
        auto PE_view = temp_PEs.view(temp_PE_offset, N_PTs);

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(
                candidate_AllP_AllT_pairs.size(),
                [positions = positions.viewer().name("Ps"),
                 PT_pairs = candidate_AllP_AllT_pairs.viewer().name("PT_pairs"),
                 surf_vertices = info.surf_vertices().viewer().name("surf_vertices"),
                 surf_triangles = info.surf_triangles().viewer().name("surf_triangles"),
                 thicknesses = info.thicknesses().viewer().name("thicknesses"),
                 temp_PPs    = PP_view.viewer().name("temp_PPs"),
                 temp_PEs    = PE_view.viewer().name("temp_PEs"),
                 temp_PTs    = temp_PTs.viewer().name("temp_PTs"),
                 d_hats = info.d_hats().viewer().name("d_hats"),
                 // 传入OGC依赖的所有参数（和phase1一致）
                 Es = info.surf_edges().viewer().name("Es"),
                 edge_face_counts = m_edge_face_counts.cviewer().name("edge_face_counts"),
                 edge_face_vertices = m_edge_face_vertices.cviewer().name("edge_face_vertices"),
                 v_edge_offsets = m_v_edge_offsets.cviewer().name("v_edge_offsets"),
                 v_edge_indices = m_v_edge_indices.cviewer().name("v_edge_indices"),
                 vertex_offsets = m_v_vertex_offsets.cviewer().name("vertex_offsets"),
                 vertex_indices = m_v_vertex_indices.cviewer().name("vertex_indices"),
                 ogc_r = m_ogc_r,
                 d_min_v = m_d_min_v.viewer().name("d_min_v"),
                 d_min_t = m_d_min_t.viewer().name("d_min_t"),
                 d_min_e = m_d_min_e.viewer().name("d_min_e")
                ] __device__(int i) mutable
                {
                    auto& PP = temp_PPs(i);
                    PP.setConstant(-1);
                    auto& PE = temp_PEs(i);
                    PE.setConstant(-1);
                    auto& PT = temp_PTs(i);
                    PT.setConstant(-1);

                    Vector2i indices = PT_pairs(i);
                    IndexT   V       = surf_vertices(indices(0));
                    Vector3i F       = surf_triangles(indices(1));

                    Vector4i vIs  = {V, F(0), F(1), F(2)};
                    Vector3  Ps[] = {positions(vIs(0)),
                                     positions(vIs(1)),
                                     positions(vIs(2)),
                                     positions(vIs(3))};

                    Float thickness = PT_thickness(thicknesses(V),
                                                   thicknesses(F(0)),
                                                   thicknesses(F(1)),
                                                   thicknesses(F(2)));

                    Float d_hat =
                        PT_d_hat(d_hats(V), d_hats(F(0)), d_hats(F(1)), d_hats(F(2)));

                    Vector4i flag =
                        distance::point_triangle_distance_flag(Ps[0], Ps[1], Ps[2], Ps[3]);

                    Vector2 range = D_range(thickness, d_hat);

                    Float D;
                    distance::point_triangle_distance2(flag, Ps[0], Ps[1], Ps[2], Ps[3], D);

                    MUDA_ASSERT(
                        D > 0.0, "D=%f, V F = (%d,%d,%d,%d)", D, vIs(0), vIs(1), vIs(2), vIs(3));

                    if(!is_active_D(range, D))
                        return;  // early return

                    // 【嵌入phase1的OGC逻辑】退化分类+偏移块验证
                    Vector4i offsets;
                    auto dim = distance::degenerate_point_triangle(flag, offsets);
                    bool is_ogc_valid = false;

                    switch(dim)
                    {
                        case 2:  // 退化PP（点-点）
                        {
                            IndexT V0 = vIs(offsets(0));  // 表面顶点V
                            IndexT V1 = vIs(offsets(1));  // 面的顶点（t0/t1/t2）
                            // OGC：验证V0是否在V1的偏移块内（复用phase1逻辑）
                            is_ogc_valid = checkVertexFeasibleRegion(
                                positions(V0), V1, positions, vertex_offsets, vertex_indices, ogc_r);
                            if(is_ogc_valid){
                                PP = {V0, V1};
                                // 在“distance::point_triangle_distance2(flag, Ps[0], Ps[1], Ps[2], Ps[3], D);”之后添加：
                                Float d_update = sqrtf(D);
                                // 原子更新：表面顶点V到面的最小距离
                                //atomic_min(&d_min_v(V), d_update);
                                atomicMinFloat(&d_min_v(V), d_update);
                                // 原子更新：面t_idx到顶点的最小距离
                                int t_idx = indices(1);  // 面在surf_triangles中的索引即面ID
                                // atomic_min(&d_min_t(t_idx), d_update);
                                atomicMinFloat(&d_min_t(t_idx), d_update);
                            }
                        }
                        break;
                        case 3:  // 退化PE（点-边）
                        {
                            IndexT   V0   = vIs(offsets(0));  // 表面顶点V
                            IndexT   V1   = vIs(offsets(1));  // 边顶点1
                            IndexT   V2   = vIs(offsets(2));  // 边顶点2
                            Vector2i e_vs = {V1, V2};
                            int      v0 = e_vs[0], v1 = e_vs[1];
                            // OGC：找边ID（复用phase1的find_edge_id）
                            int edge_id =
                                find_edge_id(v0, v1, v_edge_offsets, v_edge_indices, Es);
                            if(edge_id == -1)
                                break;
                            // OGC：验证V0是否在边的偏移块内（复用phase1逻辑）
                            is_ogc_valid = checkEdgeFeasibleRegion(positions(V0),
                                                                   positions(v0),
                                                                   positions(v1),
                                                                   edge_id,
                                                                   positions,
                                                                   edge_face_counts,
                                                                   edge_face_vertices,
                                                                   ogc_r);
                            if(is_ogc_valid){
                                PE = {V0, V1, V2};
                                // 在“distance::point_triangle_distance2(flag, Ps[0], Ps[1], Ps[2], Ps[3], D);”之后添加：
                                Float d_update = sqrtf(D);
                                // 原子更新：表面顶点V到面的最小距离
                                //atomic_min(&d_min_v(V), d_update);
                                atomicMinFloat(&d_min_v(V), d_update);
                                // 原子更新：面t_idx到顶点的最小距离
                                int t_idx = indices(1);  // 面在surf_triangles中的索引即面ID
                                // atomic_min(&d_min_t(t_idx), d_update);
                                atomicMinFloat(&d_min_t(t_idx), d_update);
                            }
                        }
                        break;
                        case 4:  // 非退化PT（点-面）
                        {
                            // OGC：面内部接触无需额外验证（复用phase1逻辑）
                            is_ogc_valid = true;
                            if(is_ogc_valid){
                                PT = vIs;
                                // 在“distance::point_triangle_distance2(flag, Ps[0], Ps[1], Ps[2], Ps[3], D);”之后添加：
                                Float d_update = sqrtf(D);
                                // 原子更新：表面顶点V到面的最小距离
                                //atomic_min(&d_min_v(V), d_update);
                                atomicMinFloat(&d_min_v(V), d_update);
                                // 原子更新：面t_idx到顶点的最小距离
                                int t_idx = indices(1);  // 面在surf_triangles中的索引即面ID
                                // atomic_min(&d_min_t(t_idx), d_update);
                                atomicMinFloat(&d_min_t(t_idx), d_update);
                            }
                        }
                        break;
                        default:
                            MUDA_ERROR_WITH_LOCATION("unexpected degenerate case dim=%d", dim);
                            break;
                    }
                    /////原本的ipc逻辑/////
                    //Vector4i offsets;
                    //auto dim = distance::degenerate_point_triangle(flag, offsets);

                    //switch(dim)
                    //{
                    //    case 2:  // PP
                    //    {
                    //        IndexT V0 = vIs(offsets(0));
                    //        IndexT V1 = vIs(offsets(1));
                    //        PP        = {V0, V1};
                    //    }
                    //    break;
                    //    case 3:  // PE
                    //    {
                    //        IndexT V0 = vIs(offsets(0));
                    //        IndexT V1 = vIs(offsets(1));
                    //        IndexT V2 = vIs(offsets(2));
                    //        PE        = {V0, V1, V2};
                    //    }
                    //    break;
                    //    case 4:  // PT
                    //    {
                    //        PT = vIs;
                    //    }
                    //    break;
                    //    default: {
                    //        MUDA_ERROR_WITH_LOCATION("unexpected degenerate case dim=%d", dim);
                    //    }
                    //    break;
                    //}
                });

        temp_PP_offset += N_PTs;
        temp_PE_offset += N_PTs;
    }
    // AllE and AllE
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_EEs);
        auto PE_view = temp_PEs.view(temp_PE_offset, N_EEs);

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(
                candidate_AllE_AllE_pairs.size(),
                [positions = positions.viewer().name("Ps"),
                 rest_positions = info.rest_positions().viewer().name("rest_positions"),
                 EE_pairs = candidate_AllE_AllE_pairs.viewer().name("EE_pairs"),
                 surf_edges  = info.surf_edges().viewer().name("surf_edges"),
                 thicknesses = info.thicknesses().viewer().name("thicknesses"),
                 temp_PPs    = PP_view.viewer().name("temp_PPs"),
                 temp_PEs    = PE_view.viewer().name("temp_PEs"),
                 temp_EEs    = temp_EEs.viewer().name("temp_EEs"),
                 d_hats = info.d_hats().viewer().name("d_hats"),
                 // 传入OGC依赖的参数（和phase2一致）
                 vertex_offsets = m_v_vertex_offsets.cviewer().name("vertex_offsets"),
                 vertex_indices = m_v_vertex_indices.cviewer().name("vertex_indices"),
                 ogc_r = m_ogc_r,
                 d_min_v = m_d_min_v.viewer().name("d_min_v"),
                 d_min_t = m_d_min_t.viewer().name("d_min_t"),
                 d_min_e = m_d_min_e.viewer().name("d_min_e")
                ] __device__(int i) mutable
                {
                    auto& PP = temp_PPs(i);
                    PP.setConstant(-1);
                    auto& PE = temp_PEs(i);
                    PE.setConstant(-1);
                    auto& EE = temp_EEs(i);
                    EE.setConstant(-1);

                    Vector2i indices = EE_pairs(i);
                    Vector2i E0      = surf_edges(indices(0));
                    Vector2i E1      = surf_edges(indices(1));

                    Vector4i vIs  = {E0(0), E0(1), E1(0), E1(1)};
                    Vector3  Ps[] = {positions(vIs(0)),
                                     positions(vIs(1)),
                                     positions(vIs(2)),
                                     positions(vIs(3))};

                    Float thickness = EE_thickness(thicknesses(E0(0)),
                                                   thicknesses(E0(1)),
                                                   thicknesses(E1(0)),
                                                   thicknesses(E1(1)));

                    Float d_hat = EE_d_hat(
                        d_hats(E0(0)), d_hats(E0(1)), d_hats(E1(0)), d_hats(E1(1)));

                    Vector2 range = D_range(thickness, d_hat);

                    Vector4i flag =
                        distance::edge_edge_distance_flag(Ps[0], Ps[1], Ps[2], Ps[3]);

                    Float D;
                    distance::edge_edge_distance2(flag, Ps[0], Ps[1], Ps[2], Ps[3], D);

                    if(!is_active_D(range, D))
                        return;  // early return


                                        // 【原IPC逻辑】软化判断
                    Float eps_x;
                    distance::edge_edge_mollifier_threshold(rest_positions(vIs(0)),
                                                            rest_positions(vIs(1)),
                                                            rest_positions(vIs(2)),
                                                            rest_positions(vIs(3)),
                                                            eps_x);
                    if(distance::need_mollify(Ps[0], Ps[1], Ps[2], Ps[3], eps_x))
                    {
                        EE = vIs;
                        return;
                    }
                    else  // 【嵌入phase2的OGC逻辑】退化分类+偏移块验证
                    {
                        Vector4i offsets;
                        auto dim = distance::degenerate_edge_edge(flag, offsets);
                        bool is_ogc_valid = false;

                        switch(dim)
                        {
                            case 2:  // 完全退化PP（点-点）
                            {
                                // 复用phase2逻辑：从flag获取退化点
                                int a_vertex = (offsets[0] == 0) ? E0(0) : E0(1);
                                int ep_vertex = (offsets[1] == 2) ? E1(0) : E1(1);
                                // OGC：验证边-边最近点是否在偏移块内（复用phase2的check函数）
                                Vector3 x_c = edge_edge_closest_point(E0, E1, positions);
                                is_ogc_valid = checkVertexFeasibleRegionEdgeOffset(
                                    x_c, a_vertex, positions, vertex_offsets, vertex_indices, ogc_r);
                                if(is_ogc_valid){
                                    PP = {a_vertex, ep_vertex};
                                    // 在“distance::edge_edge_distance2(flag, Ps[0], Ps[1], Ps[2], Ps[3], D);”之后添加：
                                    Float d_update = sqrtf(D);
                                    // 原子更新：边e_idx的最小距离
                                    int e_idx = indices(0);
                                    //atomic_min(&d_min_e(e_idx), d_update);
                                    atomicMinFloat(&d_min_e(e_idx), d_update);
                                    // 原子更新：边e_prime_idx的最小距离
                                    int e_prime_idx = indices(1);
                                    //atomic_min(&d_min_e(e_prime_idx), d_update);
                                    atomicMinFloat(&d_min_e(e_prime_idx), d_update);
                                }
                            }
                            break;
                            case 3:  // 部分退化PE（点-边）
                            {
                                // 复用phase2逻辑：判断退化点和边
                                int point_idx = -1, edge_v0_idx = -1, edge_v1_idx = -1;
                                if(offsets[0] == offsets[1])
                                {
                                    point_idx   = offsets[0];
                                    edge_v0_idx = offsets[1];
                                    edge_v1_idx = offsets[2];
                                }
                                else
                                {
                                    point_idx   = offsets[2];
                                    edge_v0_idx = offsets[0];
                                    edge_v1_idx = offsets[1];
                                }
                                int a_vertex = (point_idx == 0) ? E0(0) :
                                               (point_idx == 1) ? E0(1) :
                                               (point_idx == 2) ? E1(0) :
                                                                  E1(1);
                                // OGC：验证最近点是否在偏移块内
                                Vector3 x_c = edge_edge_closest_point(E0, E1, positions);
                                is_ogc_valid = checkVertexFeasibleRegionEdgeOffset(
                                    x_c, a_vertex, positions, vertex_offsets, vertex_indices, ogc_r);
                                if(is_ogc_valid)
                                {
                                    int ep_closest_v = (edge_v0_idx == 2) ? E1(0) :
                                                       (edge_v0_idx == 3) ? E1(1) :
                                                       (edge_v0_idx == 0) ? E0(0) :
                                                                            E0(1);
                                    PP = {a_vertex, ep_closest_v};
                                    // 在“distance::edge_edge_distance2(flag, Ps[0], Ps[1], Ps[2], Ps[3], D);”之后添加：
                                    Float d_update = sqrtf(D);
                                    // 原子更新：边e_idx的最小距离
                                    int e_idx = indices(0);
                                    //atomic_min(&d_min_e(e_idx), d_update);
                                    //// 原子更新：边e_prime_idx的最小距离
                                    int e_prime_idx = indices(1);
                                    //atomic_min(&d_min_e(e_prime_idx), d_update);
                                    atomicMinFloat(&d_min_e(e_prime_idx), d_update);
                                }
                            }
                            break;
                            case 4:  // 非退化EE（边-边）
                            {
                                // OGC：边-边接触无需额外验证（复用phase2逻辑）
                                is_ogc_valid = true;
                                if(is_ogc_valid)
                                {
                                    EE = vIs;
                                    // 在“distance::edge_edge_distance2(flag, Ps[0], Ps[1], Ps[2], Ps[3], D);”之后添加：
                                    Float d_update = sqrtf(D);
                                    // 原子更新：边e_idx的最小距离
                                    int e_idx = indices(0);
                                    //atomic_min(&d_min_e(e_idx), d_update);
                                    atomicMinFloat(&d_min_e(e_idx), d_update);
                                    // 原子更新：边e_prime_idx的最小距离
                                    int e_prime_idx = indices(1);
                                    //atomic_min(&d_min_e(e_prime_idx), d_update);
                                    atomicMinFloat(&d_min_e(e_prime_idx), d_update);
                                }
                            }
                            break;
                            default:
                                MUDA_ERROR_WITH_LOCATION("unexpected degenerate case dim=%d", dim);
                                break;
                        }
                    }

                    ////////////原本ipc的逻辑////////////
                    //Float eps_x;
                    //distance::edge_edge_mollifier_threshold(rest_positions(vIs(0)),
                    //                                        rest_positions(vIs(1)),
                    //                                        rest_positions(vIs(2)),
                    //                                        rest_positions(vIs(3)),
                    //                                        eps_x);

                    //if(distance::need_mollify(Ps[0], Ps[1], Ps[2], Ps[3], eps_x))
                    //{
                    //    EE = vIs;
                    //    return;
                    //}
                    //else  // classify to EE/PE/PP
                    //{
                    //    Vector4i offsets;
                    //    auto dim = distance::degenerate_edge_edge(flag, offsets);

                    //    switch(dim)
                    //    {
                    //        case 2:  // PP
                    //        {
                    //            IndexT V0 = vIs(offsets(0));
                    //            IndexT V1 = vIs(offsets(1));
                    //            PP        = {V0, V1};
                    //        }
                    //        break;
                    //        case 3:  // PE
                    //        {
                    //            IndexT V0 = vIs(offsets(0));
                    //            IndexT V1 = vIs(offsets(1));
                    //            IndexT V2 = vIs(offsets(2));
                    //            PE        = {V0, V1, V2};
                    //        }
                    //        break;
                    //        case 4:  // EE
                    //        {
                    //            EE = vIs;
                    //        }
                    //        break;
                    //        default: {
                    //            MUDA_ERROR_WITH_LOCATION("unexpected degenerate case dim=%d", dim);
                    //        }
                    //        break;
                    //    }
                    //}
                })
            .wait();

        temp_PP_offset += N_EEs;
        temp_PE_offset += N_EEs;
    }

    
    // ===== DEBUG PRINT: per-vertex distances (vertex -> nearest face distance, its incident faces' min distances, its incident edges' min distances)
    // NOTE:
    // 1) Do NOT re-resize m_d_min_v/m_d_min_t/m_d_min_e here (that would erase computed values).
    // 2) Printing everything can be huge; we limit to first N vertices.
    //{
    //    const int DebugPrintVertexCount = 32;  // adjust as needed
    //    // Host copies
    //    std::vector<float>  h_d_min_v;
    //    std::vector<float>  h_d_min_t;
    //    std::vector<float>  h_d_min_e;
    //    std::vector<IndexT> h_v_face_offsets;
    //    std::vector<IndexT> h_v_face_indices;
    //    std::vector<IndexT> h_v_edge_offsets;
    //    std::vector<IndexT> h_v_edge_indices;

    //    m_d_min_v.copy_to(h_d_min_v);
    //    m_d_min_t.copy_to(h_d_min_t);
    //    m_d_min_e.copy_to(h_d_min_e);

    //    h_v_face_offsets.resize(m_v_face_offsets.size());
    //    m_v_face_offsets.copy_to(h_v_face_offsets);
    //    h_v_face_indices.resize(m_v_face_indices.size());
    //    m_v_face_indices.copy_to(h_v_face_indices);

    //    h_v_edge_offsets.resize(m_v_edge_offsets.size());
    //    m_v_edge_offsets.copy_to(h_v_edge_offsets);
    //    h_v_edge_indices.resize(m_v_edge_indices.size());
    //    m_v_edge_indices.copy_to(h_v_edge_indices);
    //    const float DebugThreshold = 0.0018f;
    //    int VPrint = std::min<int>((int)h_d_min_v.size(), DebugPrintVertexCount);
    //    std::cout << "=== d_min debug (first " << VPrint << " vertices) ===\n";
    //    for(int v = 0; v < VPrint; ++v)
    //    {
    //        float dv = h_d_min_v[v];
    //        std::cout << "Vertex " << v << " d_min_v=" << dv << "\n";

    //        // Faces incident to v
    //        if(v + 1 < (int)h_v_face_offsets.size())
    //        {
    //            int f_start = (int)h_v_face_offsets[v];
    //            int f_end   = (int)h_v_face_offsets[v + 1];
    //            std::cout << "  Faces (" << (f_end - f_start) << "): ";
    //            for(int p = f_start; p < f_end; ++p)
    //            {
    //                int f = (int)h_v_face_indices[p];
    //                if(f >= 0 && f < (int)h_d_min_t.size())
    //                    std::cout << "{f=" << f << ", d_min_t=" << h_d_min_t[f] << "} ";
    //            }
    //            std::cout << "\n";
    //        }

    //        // Edges incident to v
    //        if(v + 1 < (int)h_v_edge_offsets.size())
    //        {
    //            int e_start = (int)h_v_edge_offsets[v];
    //            int e_end   = (int)h_v_edge_offsets[v + 1];
    //            std::cout << "  Edges (" << (e_end - e_start) << "): ";
    //            for(int p = e_start; p < e_end; ++p)
    //            {
    //                int e = (int)h_v_edge_indices[p];
    //                if(e >= 0 && e < (int)h_d_min_e.size())
    //                    std::cout << "{e=" << e << ", d_min_e=" << h_d_min_e[e] << "} ";
    //            }
    //            std::cout << "\n";
    //        }

    //        if(dv < DebugThreshold)
    //        {
    //            std::cout << "*** DEBUG BREAK: Vertex " << v << " d_min_v=" << dv
    //                      << " < " << DebugThreshold << "\n";
    //            int tempstop = 0;
    //        }

    //    }
    //    std::cout << "=== end d_min debug ===\n";
    //    int tempstop = 0;
    //}

    UIPC_ASSERT(temp_PP_offset == temp_PPs.size(), "size mismatch");
    UIPC_ASSERT(temp_PE_offset == temp_PEs.size(), "size mismatch");

    {  // select the valid ones
        PPs.resize(temp_PPs.size());
        PEs.resize(temp_PEs.size());
        PTs.resize(temp_PTs.size());
        EEs.resize(temp_EEs.size());

        DeviceSelect().If(temp_PPs.data(),
                          PPs.data(),
                          selected_PP_count.data(),
                          temp_PPs.size(),
                          [] CUB_RUNTIME_FUNCTION(const Vector2i& PP)
                          { return PP(0) != -1; });

        DeviceSelect().If(temp_PEs.data(),
                          PEs.data(),
                          selected_PE_count.data(),
                          temp_PEs.size(),
                          [] CUB_RUNTIME_FUNCTION(const Vector3i& PE)
                          { return PE(0) != -1; });

        DeviceSelect().If(temp_PTs.data(),
                          PTs.data(),
                          selected_PT_count.data(),
                          temp_PTs.size(),
                          [] CUB_RUNTIME_FUNCTION(const Vector4i& PT)
                          { return PT(0) != -1; });

        DeviceSelect().If(temp_EEs.data(),
                          EEs.data(),
                          selected_EE_count.data(),
                          temp_EEs.size(),
                          [] CUB_RUNTIME_FUNCTION(const Vector4i& EE)
                          { return EE(0) != -1; });

        IndexT PP_count = selected_PP_count;
        IndexT PE_count = selected_PE_count;
        IndexT PT_count = selected_PT_count;
        IndexT EE_count = selected_EE_count;

        PPs.resize(PP_count);
        PEs.resize(PE_count);
        PTs.resize(PT_count);
        EEs.resize(EE_count);
    }

    info.PPs(PPs);
    info.PEs(PEs);
    info.PTs(PTs);
    info.EEs(EEs);

    if constexpr(PrintDebugInfo)
    {
        std::vector<Vector2i> PPs_host;
        std::vector<Float>    PP_thicknesses_host;

        std::vector<Vector3i> PEs_host;
        std::vector<Float>    PE_thicknesses_host;

        std::vector<Vector4i> PTs_host;
        std::vector<Float>    PT_thicknesses_host;

        std::vector<Vector4i> EEs_host;
        std::vector<Float>    EE_thicknesses_host;

        PPs.copy_to(PPs_host);
        PEs.copy_to(PEs_host);
        PTs.copy_to(PTs_host);
        EEs.copy_to(EEs_host);

        std::cout << "filter result:" << std::endl;

        for(auto&& [PP, thickness] : zip(PPs_host, PP_thicknesses_host))
        {
            std::cout << "PP: " << PP.transpose() << " thickness: " << thickness << "\n";
        }

        for(auto&& [PE, thickness] : zip(PEs_host, PE_thicknesses_host))
        {
            std::cout << "PE: " << PE.transpose() << " thickness: " << thickness << "\n";
        }

        for(auto&& [PT, thickness] : zip(PTs_host, PT_thicknesses_host))
        {
            std::cout << "PT: " << PT.transpose() << " thickness: " << thickness << "\n";
        }

        for(auto&& [EE, thickness] : zip(EEs_host, EE_thicknesses_host))
        {
            std::cout << "EE: " << EE.transpose() << " thickness: " << thickness << "\n";
        }

        std::cout << std::flush;
    }
}


void LBVHSimplexTrajectoryFilter::Impl::filter_toi(FilterTOIInfo& info)
{
    using namespace muda;

    auto toi_size =
        candidate_AllP_CodimP_pairs.size() + candidate_CodimP_AllE_pairs.size()
        + candidate_AllP_AllT_pairs.size() + candidate_AllE_AllE_pairs.size();

    tois.resize(toi_size);

    auto offset  = 0;
    auto PP_tois = tois.view(offset, candidate_AllP_CodimP_pairs.size());
    offset += candidate_AllP_CodimP_pairs.size();
    auto PE_tois = tois.view(offset, candidate_CodimP_AllE_pairs.size());
    offset += candidate_CodimP_AllE_pairs.size();
    auto PT_tois = tois.view(offset, candidate_AllP_AllT_pairs.size());
    offset += candidate_AllP_AllT_pairs.size();
    auto EE_tois = tois.view(offset, candidate_AllE_AllE_pairs.size());
    offset += candidate_AllE_AllE_pairs.size();

    UIPC_ASSERT(offset == toi_size, "size mismatch");


    // TODO: Now hard code the minimum separation coefficient
    // gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
    constexpr Float eta = 0.1;

    // TODO: Now hard code the maximum iteration
    constexpr SizeT max_iter = 1000;

    // large enough toi (>1)
    constexpr Float large_enough_toi = 1.1;
    auto            alpha            = info.alpha();
    // AllP and CodimP
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(candidate_AllP_CodimP_pairs.size(),
                   [PP_tois = PP_tois.viewer().name("PP_tois"),
                    PCodimP_pairs = candidate_AllP_CodimP_pairs.viewer().name("PP_pairs"),
                    codim_vertices = info.codim_vertices().viewer().name("codim_vertices"),
                    surf_vertices = info.surf_vertices().viewer().name("surf_vertices"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    positions = info.positions().viewer().name("Ps"),
                    dxs       = info.displacements().viewer().name("dxs"),
                    d_hats    = info.d_hats().viewer().name("d_hats"),
                    alpha     = info.alpha(),

                    eta,
                    max_iter,
                    large_enough_toi] __device__(int i) mutable
                   {
                       auto   indices = PCodimP_pairs(i);
                       IndexT V0      = surf_vertices(indices(0));
                       IndexT V1      = codim_vertices(indices(1));

                       Float thickness = PP_thickness(thicknesses(V0), thicknesses(V1));
                       Float d_hat = PP_d_hat(d_hats(V0), d_hats(V1));

                       Vector3 VP0  = positions(V0);
                       Vector3 VP1  = positions(V1);
                       Vector3 dVP0 = alpha * dxs(V0);
                       Vector3 dVP1 = alpha * dxs(V1);

                       Float toi = large_enough_toi;

                       bool faraway = !distance::point_point_ccd_broadphase(
                           VP0, VP1, dVP0, dVP1, d_hat + thickness);

                       if(faraway)
                       {
                           PP_tois(i) = toi;
                           return;
                       }

                       bool hit = distance::point_point_ccd(
                           VP0, VP1, dVP0, dVP1, eta, thickness, max_iter, toi);

                       if(!hit)
                           toi = large_enough_toi;

                       PP_tois(i) = toi;
                   });
    }

    // CodimP and AllE
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(candidate_CodimP_AllE_pairs.size(),
                   [PE_tois = PE_tois.viewer().name("PE_tois"),
                    CodimP_AllE_pairs = candidate_CodimP_AllE_pairs.viewer().name("PE_pairs"),
                    codim_vertices = info.codim_vertices().viewer().name("codim_vertices"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    surf_edges = info.surf_edges().viewer().name("surf_edges"),
                    Ps         = info.positions().viewer().name("Ps"),
                    dxs        = info.displacements().viewer().name("dxs"),
                    d_hats     = info.d_hats().viewer().name("d_hats"),
                    alpha      = info.alpha(),
                    eta,
                    max_iter,
                    large_enough_toi] __device__(int i) mutable
                   {
                       auto     indices = CodimP_AllE_pairs(i);
                       IndexT   V       = codim_vertices(indices(0));
                       Vector2i E       = surf_edges(indices(1));

                       Float thickness = PE_thickness(
                           thicknesses(V), thicknesses(E(0)), thicknesses(E(1)));
                       Float d_hat = PE_d_hat(d_hats(V), d_hats(E(0)), d_hats(E(1)));

                       Vector3 VP  = Ps(V);
                       Vector3 dVP = alpha * dxs(V);

                       Vector3 EP0  = Ps(E[0]);
                       Vector3 EP1  = Ps(E[1]);
                       Vector3 dEP0 = alpha * dxs(E[0]);
                       Vector3 dEP1 = alpha * dxs(E[1]);

                       Float toi = large_enough_toi;

                       bool faraway = !distance::point_edge_ccd_broadphase(
                           VP, EP0, EP1, dVP, dEP0, dEP1, d_hat + thickness);

                       if(faraway)
                       {
                           PE_tois(i) = toi;
                           return;
                       }

                       bool hit = distance::point_edge_ccd(
                           VP, EP0, EP1, dVP, dEP0, dEP1, eta, thickness, max_iter, toi);

                       if(!hit)
                           toi = large_enough_toi;

                       PE_tois(i) = toi;
                   });
    }

    // AllP and AllT
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(candidate_AllP_AllT_pairs.size(),
                   [PT_tois = PT_tois.viewer().name("PT_tois"),
                    PT_pairs = candidate_AllP_AllT_pairs.viewer().name("PT_pairs"),
                    surf_vertices = info.surf_vertices().viewer().name("surf_vertices"),
                    surf_triangles = info.surf_triangles().viewer().name("surf_triangles"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    Ps     = info.positions().viewer().name("Ps"),
                    dxs    = info.displacements().viewer().name("dxs"),
                    d_hats = info.d_hats().viewer().name("d_hats"),
                    alpha  = info.alpha(),
                    eta,
                    max_iter,
                    large_enough_toi] __device__(int i) mutable
                   {
                       auto     indices = PT_pairs(i);
                       IndexT   V       = surf_vertices(indices(0));
                       Vector3i F       = surf_triangles(indices(1));

                       Float thickness = PT_thickness(thicknesses(V),
                                                      thicknesses(F(0)),
                                                      thicknesses(F(1)),
                                                      thicknesses(F(2)));
                       Float d_hat =
                           PT_d_hat(d_hats(V), d_hats(F(0)), d_hats(F(1)), d_hats(F(2)));

                       Vector3 VP  = Ps(V);
                       Vector3 dVP = alpha * dxs(V);

                       Vector3 FP0 = Ps(F[0]);
                       Vector3 FP1 = Ps(F[1]);
                       Vector3 FP2 = Ps(F[2]);

                       Vector3 dFP0 = alpha * dxs(F[0]);
                       Vector3 dFP1 = alpha * dxs(F[1]);
                       Vector3 dFP2 = alpha * dxs(F[2]);

                       Float toi = large_enough_toi;


                       bool faraway = !distance::point_triangle_ccd_broadphase(
                           VP, FP0, FP1, FP2, dVP, dFP0, dFP1, dFP2, d_hat + thickness);

                       if(faraway)
                       {
                           PT_tois(i) = toi;
                           return;
                       }

                       bool hit = distance::point_triangle_ccd(
                           VP, FP0, FP1, FP2, dVP, dFP0, dFP1, dFP2, eta, thickness, max_iter, toi);

                       if(!hit)
                           toi = large_enough_toi;

                       PT_tois(i) = toi;
                   });
    }

    // AllE and AllE
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(candidate_AllE_AllE_pairs.size(),
                   [EE_tois = EE_tois.viewer().name("EE_tois"),
                    EE_pairs = candidate_AllE_AllE_pairs.viewer().name("EE_pairs"),
                    surf_edges = info.surf_edges().viewer().name("surf_edges"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    Ps     = info.positions().viewer().name("Ps"),
                    dxs    = info.displacements().viewer().name("dxs"),
                    d_hats = info.d_hats().viewer().name("d_hats"),
                    alpha  = info.alpha(),
                    eta,
                    max_iter,
                    large_enough_toi] __device__(int i) mutable
                   {
                       auto     indices = EE_pairs(i);
                       Vector2i E0      = surf_edges(indices(0));
                       Vector2i E1      = surf_edges(indices(1));

                       Float thickness = EE_thickness(thicknesses(E0(0)),
                                                      thicknesses(E0(1)),
                                                      thicknesses(E1(0)),
                                                      thicknesses(E1(1)));

                       Float d_hat = EE_d_hat(
                           d_hats(E0(0)), d_hats(E0(1)), d_hats(E1(0)), d_hats(E1(1)));


                       Vector3 EP0  = Ps(E0[0]);
                       Vector3 EP1  = Ps(E0[1]);
                       Vector3 dEP0 = alpha * dxs(E0[0]);
                       Vector3 dEP1 = alpha * dxs(E0[1]);

                       Vector3 EP2  = Ps(E1[0]);
                       Vector3 EP3  = Ps(E1[1]);
                       Vector3 dEP2 = alpha * dxs(E1[0]);
                       Vector3 dEP3 = alpha * dxs(E1[1]);

                       Float toi = large_enough_toi;

                       bool faraway = !distance::edge_edge_ccd_broadphase(
                           // position
                           EP0,
                           EP1,
                           EP2,
                           EP3,
                           // displacement
                           dEP0,
                           dEP1,
                           dEP2,
                           dEP3,
                           d_hat + thickness);

                       if(faraway)
                       {
                           EE_tois(i) = toi;
                           return;
                       }

                       bool hit = distance::edge_edge_ccd(
                           // position
                           EP0,
                           EP1,
                           EP2,
                           EP3,
                           // displacement
                           dEP0,
                           dEP1,
                           dEP2,
                           dEP3,
                           eta,
                           thickness,
                           max_iter,
                           toi);

                       if(!hit)
                           toi = large_enough_toi;

                       EE_tois(i) = toi;
                   });
    }

    if(tois.size())
    {
        // get min toi
        DeviceReduce().Min(tois.data(), info.toi().data(), tois.size());
    }
    else
    {
        info.toi().fill(large_enough_toi);
    }
}


void LBVHSimplexTrajectoryFilter::Impl::filter_toi_distance(FilterTOIInfo& info)
{
    using namespace muda;

    auto toi_size =
        candidate_AllP_CodimP_pairs.size() + candidate_CodimP_AllE_pairs.size()
        + candidate_AllP_AllT_pairs.size() + candidate_AllE_AllE_pairs.size();

    tois.resize(toi_size);
    penetration_depth.resize(toi_size);

    auto offset  = 0;
    auto PP_tois = tois.view(offset, candidate_AllP_CodimP_pairs.size());
    auto PP_dis = penetration_depth.view(offset, candidate_AllP_CodimP_pairs.size());
    offset += candidate_AllP_CodimP_pairs.size();
    auto PE_tois = tois.view(offset, candidate_CodimP_AllE_pairs.size());
    auto PE_dis = penetration_depth.view(offset, candidate_CodimP_AllE_pairs.size());
    offset += candidate_CodimP_AllE_pairs.size();
    auto PT_tois = tois.view(offset, candidate_AllP_AllT_pairs.size());
    auto PT_dis = penetration_depth.view(offset, candidate_AllP_AllT_pairs.size());
    offset += candidate_AllP_AllT_pairs.size();
    auto EE_tois = tois.view(offset, candidate_AllE_AllE_pairs.size());
    auto EE_dis = penetration_depth.view(offset, candidate_AllE_AllE_pairs.size());
    offset += candidate_AllE_AllE_pairs.size();

    UIPC_ASSERT(offset == toi_size, "size mismatch");


    // TODO: Now hard code the minimum separation coefficient
    // gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
    constexpr Float eta = 0.1;

    // TODO: Now hard code the maximum iteration
    constexpr SizeT max_iter = 1000;

    // large enough toi (>1)
    constexpr Float large_enough_toi = 1.1;

    // no enough toi (>1)
    constexpr Float no_penetrate = 0.0;

    // AllP and CodimP
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(candidate_AllP_CodimP_pairs.size(),
                   [PP_tois = PP_tois.viewer().name("PP_tois"),
                    PP_dis  = PP_dis.viewer().name("PP_dis"),
                    PCodimP_pairs = candidate_AllP_CodimP_pairs.viewer().name("PP_pairs"),
                    codim_vertices = info.codim_vertices().viewer().name("codim_vertices"),
                    surf_vertices = info.surf_vertices().viewer().name("surf_vertices"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    positions = info.positions().viewer().name("Ps"),
                    dxs       = info.displacements().viewer().name("dxs"),
                    d_hats    = info.d_hats().viewer().name("d_hats"),
                    alpha     = info.alpha(),

                    eta,
                    max_iter,
                    large_enough_toi,
                    no_penetrate] __device__(int i) mutable
                   {
                       auto   indices = PCodimP_pairs(i);
                       IndexT V0      = surf_vertices(indices(0));
                       IndexT V1      = codim_vertices(indices(1));

                       Float thickness = PP_thickness(thicknesses(V0), thicknesses(V1));
                       Float d_hat = PP_d_hat(d_hats(V0), d_hats(V1));

                       Vector3 VP0  = positions(V0);
                       Vector3 VP1  = positions(V1);
                       Vector3 dVP0 = alpha * dxs(V0);
                       Vector3 dVP1 = alpha * dxs(V1);

                       Float toi = large_enough_toi;
                       Float dis = no_penetrate;
                       bool faraway = !distance::point_point_ccd_broadphase(
                           VP0, VP1, dVP0, dVP1, d_hat + thickness);

                       if(faraway)
                       {
                           PP_tois(i) = toi;
                           PP_dis(i)  = dis;
                           return;
                       }

                       //bool hit = distance::point_point_ccd(
                       //    VP0, VP1, dVP0, dVP1, eta, thickness, max_iter, toi);

                       bool hit = distance::point_point_ccd_compute_penetration_depth(
                           VP0, VP1, dVP0, dVP1, eta, thickness, max_iter, toi, dis);

                       if(!hit)
                       {
                           toi = large_enough_toi;
                           dis = no_penetrate;
                       }

                       PP_tois(i) = toi;
                       PP_dis(i)  = toi;
                   });
    }

    // CodimP and AllE
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(candidate_CodimP_AllE_pairs.size(),
                   [PE_tois = PE_tois.viewer().name("PE_tois"),
                    PE_dis  = PE_dis.viewer().name("PE_dis"),
                    CodimP_AllE_pairs = candidate_CodimP_AllE_pairs.viewer().name("PE_pairs"),
                    codim_vertices = info.codim_vertices().viewer().name("codim_vertices"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    surf_edges = info.surf_edges().viewer().name("surf_edges"),
                    Ps         = info.positions().viewer().name("Ps"),
                    dxs        = info.displacements().viewer().name("dxs"),
                    d_hats     = info.d_hats().viewer().name("d_hats"),
                    alpha      = info.alpha(),
                    eta,
                    max_iter,
                    large_enough_toi] __device__(int i) mutable
                   {
                       auto     indices = CodimP_AllE_pairs(i);
                       IndexT   V       = codim_vertices(indices(0));
                       Vector2i E       = surf_edges(indices(1));

                       Float thickness = PE_thickness(
                           thicknesses(V), thicknesses(E(0)), thicknesses(E(1)));
                       Float d_hat = PE_d_hat(d_hats(V), d_hats(E(0)), d_hats(E(1)));

                       Vector3 VP  = Ps(V);
                       Vector3 dVP = alpha * dxs(V);

                       Vector3 EP0  = Ps(E[0]);
                       Vector3 EP1  = Ps(E[1]);
                       Vector3 dEP0 = alpha * dxs(E[0]);
                       Vector3 dEP1 = alpha * dxs(E[1]);

                       Float toi = large_enough_toi;
                       Float dis     = no_penetrate;
                       bool faraway = !distance::point_edge_ccd_broadphase(
                           VP, EP0, EP1, dVP, dEP0, dEP1, d_hat + thickness);

                       if(faraway)
                       {
                           PE_tois(i) = toi;
                           PE_dis(i)  = dis;
                           return;
                       }

                       //bool hit = distance::point_edge_ccd(
                       //    VP, EP0, EP1, dVP, dEP0, dEP1, eta, thickness, max_iter, toi);

                       bool hit = distance::point_edge_ccd_compute_penetration_depth(
                           VP, EP0, EP1, dVP, dEP0, dEP1, eta, thickness, max_iter, toi, dis);

                       if(!hit)
                       {
                           toi = large_enough_toi;
                           dis = no_penetrate;
                       }

                       PE_tois(i) = toi;
                       PE_dis(i) = toi;
                   });
    }

    // AllP and AllT
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(candidate_AllP_AllT_pairs.size(),
                   [PT_tois = PT_tois.viewer().name("PT_tois"),
                    PT_dis  = PT_dis.viewer().name("PT_dis"),
                    PT_pairs = candidate_AllP_AllT_pairs.viewer().name("PT_pairs"),
                    surf_vertices = info.surf_vertices().viewer().name("surf_vertices"),
                    surf_triangles = info.surf_triangles().viewer().name("surf_triangles"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    Ps     = info.positions().viewer().name("Ps"),
                    dxs    = info.displacements().viewer().name("dxs"),
                    d_hats = info.d_hats().viewer().name("d_hats"),
                    alpha  = info.alpha(),
                    eta,
                    max_iter,
                    large_enough_toi] __device__(int i) mutable
                   {
                       auto     indices = PT_pairs(i);
                       IndexT   V       = surf_vertices(indices(0));
                       Vector3i F       = surf_triangles(indices(1));

                       Float thickness = PT_thickness(thicknesses(V),
                                                      thicknesses(F(0)),
                                                      thicknesses(F(1)),
                                                      thicknesses(F(2)));
                       Float d_hat =
                           PT_d_hat(d_hats(V), d_hats(F(0)), d_hats(F(1)), d_hats(F(2)));

                       Vector3 VP  = Ps(V);
                       Vector3 dVP = alpha * dxs(V);

                       Vector3 FP0 = Ps(F[0]);
                       Vector3 FP1 = Ps(F[1]);
                       Vector3 FP2 = Ps(F[2]);

                       Vector3 dFP0 = alpha * dxs(F[0]);
                       Vector3 dFP1 = alpha * dxs(F[1]);
                       Vector3 dFP2 = alpha * dxs(F[2]);

                       Float toi = large_enough_toi;
                       Float dis = no_penetrate;

                       bool faraway = !distance::point_triangle_ccd_broadphase(
                           VP, FP0, FP1, FP2, dVP, dFP0, dFP1, dFP2, d_hat + thickness);

                       if(faraway)
                       {
                           PT_tois(i) = toi;
                           PT_dis(i)  = dis;
                           return;
                       }

                       //bool hit = distance::point_triangle_ccd(
                       //    VP, FP0, FP1, FP2, dVP, dFP0, dFP1, dFP2, eta, thickness, max_iter, toi);

                       bool hit = distance::point_triangle_ccd_compute_penetration_depth(
                           VP, FP0, FP1, FP2, dVP, dFP0, dFP1, dFP2, eta, thickness, max_iter, toi, dis);

                       if(!hit)
                       {
                           toi = large_enough_toi;
                           dis = no_penetrate;
                       }

                       PT_tois(i) = toi;
                       PT_dis(i)  = dis;
                   });
    }

    // AllE and AllE
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(candidate_AllE_AllE_pairs.size(),
                   [EE_tois = EE_tois.viewer().name("EE_tois"),
                    EE_dis  = EE_dis.viewer().name("EE_dis"),
                    EE_pairs = candidate_AllE_AllE_pairs.viewer().name("EE_pairs"),
                    surf_edges = info.surf_edges().viewer().name("surf_edges"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    Ps     = info.positions().viewer().name("Ps"),
                    dxs    = info.displacements().viewer().name("dxs"),
                    d_hats = info.d_hats().viewer().name("d_hats"),
                    alpha  = info.alpha(),
                    eta,
                    max_iter,
                    large_enough_toi] __device__(int i) mutable
                   {
                       auto     indices = EE_pairs(i);
                       Vector2i E0      = surf_edges(indices(0));
                       Vector2i E1      = surf_edges(indices(1));

                       Float thickness = EE_thickness(thicknesses(E0(0)),
                                                      thicknesses(E0(1)),
                                                      thicknesses(E1(0)),
                                                      thicknesses(E1(1)));

                       Float d_hat = EE_d_hat(
                           d_hats(E0(0)), d_hats(E0(1)), d_hats(E1(0)), d_hats(E1(1)));


                       Vector3 EP0  = Ps(E0[0]);
                       Vector3 EP1  = Ps(E0[1]);
                       Vector3 dEP0 = alpha * dxs(E0[0]);
                       Vector3 dEP1 = alpha * dxs(E0[1]);

                       Vector3 EP2  = Ps(E1[0]);
                       Vector3 EP3  = Ps(E1[1]);
                       Vector3 dEP2 = alpha * dxs(E1[0]);
                       Vector3 dEP3 = alpha * dxs(E1[1]);

                       Float toi = large_enough_toi;
                       Float dis     = no_penetrate;
                       bool faraway = !distance::edge_edge_ccd_broadphase(
                           // position
                           EP0,
                           EP1,
                           EP2,
                           EP3,
                           // displacement
                           dEP0,
                           dEP1,
                           dEP2,
                           dEP3,
                           d_hat + thickness);

                       if(faraway)
                       {
                           EE_tois(i) = toi;
                           EE_dis(i)  = dis;
                           return;
                       }

                       //bool hit = distance::edge_edge_ccd(
                       //    // position
                       //    EP0,
                       //    EP1,
                       //    EP2,
                       //    EP3,
                       //    // displacement
                       //    dEP0,
                       //    dEP1,
                       //    dEP2,
                       //    dEP3,
                       //    eta,
                       //    thickness,
                       //    max_iter,
                       //    toi);

                       bool hit = distance::edge_edge_ccd_compute_penetration_depth(
                           // position
                           EP0,
                           EP1,
                           EP2,
                           EP3,
                           // displacement
                           dEP0,
                           dEP1,
                           dEP2,
                           dEP3,
                           eta,
                           thickness,
                           max_iter,
                           toi,
                           dis);

                       if(!hit)
                       {
                           toi = large_enough_toi;
                           dis = no_penetrate;
                       }

                       EE_tois(i) = toi;
                       EE_dis(i)  = dis;
                   });
    }

    if(tois.size())
    {
        // get min toi
        DeviceReduce().Min(tois.data(), info.toi().data(), tois.size());
        // copy the penetration depth this penetration_depth.data() to info.penetration_depth().data()
        info.penetration_depth().copy_from(penetration_depth.data());
    }
    else
    {
        info.toi().fill(large_enough_toi);
        info.penetration_depth().fill(no_penetrate);
    }
}

}  // namespace uipc::backend::cuda
