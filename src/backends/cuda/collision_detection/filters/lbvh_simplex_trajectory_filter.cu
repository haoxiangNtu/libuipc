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
    // New: Preprocess geometry adjacency relationships (vertex-adjacent vertices, edge-adjacent face vertices)
    auto m_ogc_contact_radius = 0.002;
    //Impl.detect_ogc_contact(info);
}

// Assume in the initialization function of SimSystem
void LBVHSimplexTrajectoryFilter::Impl::init_ogc_data(DetectInfo& info)
{
    int  num_vertices = info.surf_vertices().size();
    auto edges        = info.surf_edges();
    auto faces        = info.surf_triangles();

    // Build edge-face adjacency (already implemented by you)
    build_edge_face_adjacency(edges, faces);
    // Build vertex-neighbor edge/face adjacency (newly added)
    build_vertex_adjacency(edges, faces, num_vertices);
    // Initialize minimum distance buffer (preinitialize_contact_data is already implemented by you, need to call it)
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

    // Establish (min,max) -> edge index mapping
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

    // [New: Copy temporary edge_map to device-side member variable m_edge_id_map]
    //m_edge_id_map.resize(1);  // If stored in DeviceBuffer, size is 1 (stores the entire map)
    //m_edge_id_map.view()[0] = edge_map;  // Copy CPU-side edge_map to GPU-side
}

/**
 * Build CSR adjacency lists for vertex-neighbor edges and vertex-neighbor faces
 * @param edges：Global edge list (e => (v0, v1))
 * @param faces：Global face list (t => (v0, v1, v2))
 * @param num_vertices：Total number of global vertices
 */
void LBVHSimplexTrajectoryFilter::Impl::build_vertex_adjacency(
    const muda::CBufferView<Vector2i>& edges, const muda::CBufferView<Vector3i>& faces, int num_vertices)
{
    // -------------------------- Step 1: Build vertex-neighbor edges --------------------------
    std::vector<IndexT> v_edge_counts(num_vertices, 0);  // Number of neighbor edges for each vertex
    std::vector<IndexT> v_edge_indices_temp;  // Temporarily store neighbor edge IDs (global)

    // First count the number of neighbor edges for each vertex
    std::vector<Vector2i> edges_h(edges.size());
    edges.copy_to(edges_h.data());
    for(int e = 0; e < edges_h.size(); ++e)
    {
        int v0 = edges_h[e][0], v1 = edges_h[e][1];
        v_edge_counts[v0]++;
        v_edge_counts[v1]++;
    }

    // Build CSR offset array m_v_edge_offsets
    m_v_edge_offsets.resize(num_vertices + 1);  // Offset array length = number of vertices + 1 (last element is total length)
    std::vector<int> v_edge_offsets_h(num_vertices + 1, 0);
    for(int v = 0; v < num_vertices; ++v)
    {
        v_edge_offsets_h[v + 1] = v_edge_offsets_h[v] + v_edge_counts[v];
    }
    m_v_edge_offsets.view().copy_from(v_edge_offsets_h.data());

    // Fill neighbor edge indices (ensure each edge is recorded by two vertices)
    v_edge_indices_temp.resize(v_edge_offsets_h.back(), -1);
    std::vector<int> v_edge_cursor(num_vertices, 0);  // Current filling position for each vertex
    for(int e = 0; e < edges_h.size(); ++e)
    {
        int v0 = edges_h[e][0], v1 = edges_h[e][1];
        // Neighbor edge of vertex v0
        int pos0                  = v_edge_offsets_h[v0] + v_edge_cursor[v0]++;
        v_edge_indices_temp[pos0] = e;
        // Neighbor edge of vertex v1
        int pos1                  = v_edge_offsets_h[v1] + v_edge_cursor[v1]++;
        v_edge_indices_temp[pos1] = e;
    }
    m_v_edge_indices.resize(v_edge_indices_temp.size());
    m_v_edge_indices.view().copy_from(v_edge_indices_temp.data());

    // -------------------------- Step 2: Build vertex-neighbor faces --------------------------
    std::vector<int> v_face_counts(num_vertices, 0);  // Number of neighbor faces for each vertex
    std::vector<int> v_face_indices_temp;  // Temporarily store neighbor face IDs (global)

    // Count the number of neighbor faces for each vertex
    std::vector<Vector3i> faces_h(faces.size());
    faces.copy_to(faces_h.data());
    for(int t = 0; t < faces_h.size(); ++t)
    {
        int v0 = faces_h[t][0], v1 = faces_h[t][1], v2 = faces_h[t][2];
        v_face_counts[v0]++;
        v_face_counts[v1]++;
        v_face_counts[v2]++;
    }

    // Build CSR offset array m_v_face_offsets
    m_v_face_offsets.resize(num_vertices + 1);
    std::vector<int> v_face_offsets_h(num_vertices + 1, 0);
    for(int v = 0; v < num_vertices; ++v)
    {
        v_face_offsets_h[v + 1] = v_face_offsets_h[v] + v_face_counts[v];
    }
    m_v_face_offsets.view().copy_from(v_face_offsets_h.data());

    // Fill neighbor face indices (each face is recorded by three vertices)
    v_face_indices_temp.resize(v_face_offsets_h.back(), -1);
    std::vector<int> v_face_cursor(num_vertices, 0);  // Current filling position for each vertex
    for(int t = 0; t < faces_h.size(); ++t)
    {
        int v0 = faces_h[t][0], v1 = faces_h[t][1], v2 = faces_h[t][2];
        // Neighbor face of vertex v0
        int pos0                  = v_face_offsets_h[v0] + v_face_cursor[v0]++;
        v_face_indices_temp[pos0] = t;
        // Neighbor face of vertex v1
        int pos1                  = v_face_offsets_h[v1] + v_face_cursor[v1]++;
        v_face_indices_temp[pos1] = t;
        // Neighbor face of vertex v2
        int pos2                  = v_face_offsets_h[v2] + v_face_cursor[v2]++;
        v_face_indices_temp[pos2] = t;
    }
    m_v_face_indices.resize(v_face_indices_temp.size());
    m_v_face_indices.view().copy_from(v_face_indices_temp.data());

    // -------------------------- Step 3: Build vertex-neighbor vertices (directly stored, deduplicated) --------------------------
    std::vector<int> v_vertex_counts(num_vertices, 0);  // Number of neighbor vertices for each vertex (after deduplication)
    std::vector<int> v_vertex_indices_temp;             // Temporarily store neighbor vertex indices

    // Step 1: Count the number of neighbor vertices for each vertex (after deduplication)
    for(int v = 0; v < num_vertices; ++v)
    {
        // Get all associated edges of vertex v
        int                     start = v_edge_offsets_h[v];
        int                     end   = v_edge_offsets_h[v + 1];
        std::unordered_set<int> neighbors;  // Use set for deduplication

        for(int i = start; i < end; ++i)
        {
            int   e_id = v_edge_indices_temp[i];         // Global ID of the edge
            auto& edge = edges_h[e_id];                  // Two endpoints of the edge
            int u = (edge[0] == v) ? edge[1] : edge[0];  // The other endpoint (neighbor vertex)
            neighbors.insert(u);
        }

        v_vertex_counts[v] = neighbors.size();  // Number of neighbors after deduplication
    }

    // Step 2: Build CSR offset array m_v_vertex_offsets
    m_v_vertex_offsets.resize(num_vertices + 1);
    std::vector<int> v_vertex_offsets_h(num_vertices + 1, 0);
    for(int v = 0; v < num_vertices; ++v)
    {
        v_vertex_offsets_h[v + 1] = v_vertex_offsets_h[v] + v_vertex_counts[v];
    }
    m_v_vertex_offsets.view().copy_from(v_vertex_offsets_h.data());

    // Step 3: Fill neighbor vertex indices (after deduplication)
    v_vertex_indices_temp.resize(v_vertex_offsets_h.back());
    std::vector<int> v_vertex_cursor(num_vertices, 0);  // Current filling position for each vertex

    for(int v = 0; v < num_vertices; ++v)
    {
        int                     start = v_edge_offsets_h[v];
        int                     end   = v_edge_offsets_h[v + 1];
        std::unordered_set<int> neighbors;

        // First collect deduplicated neighbor vertices
        for(int i = start; i < end; ++i)
        {
            int   e_id = v_edge_indices_temp[i];
            auto& edge = edges_h[e_id];
            int   u    = (edge[0] == v) ? edge[1] : edge[0];
            neighbors.insert(u);
        }

        // Fill into temporary array
        int pos = v_vertex_offsets_h[v];
        for(int u : neighbors)
        {
            v_vertex_indices_temp[pos + v_vertex_cursor[v]++] = u;
        }
    }

    m_v_vertex_indices.resize(v_vertex_indices_temp.size());
    m_v_vertex_indices.view().copy_from(v_vertex_indices_temp.data());
}

//// 3. Pre-initialization (called before each contact detection, e.g., at the beginning of filter_active)
void LBVHSimplexTrajectoryFilter::Impl::preinitialize_contact_data(int num_vertices,
                                                                   int num_facets,
                                                                   int num_edges,
                                                                   float r_q)
{
    // Initialize minimum distance to query radius r_q (Document Algorithm 1 Line 1, Algorithm 2 Line 1)
    // This should be modified to be similar to m_d_min_v
    
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

    // Clear the contact set of the previous frame
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
    auto Vs = info.surf_vertices();   // Surface vertex list (global vertex ID)
    auto Es = info.surf_edges();      // Surface edge list (e => (v0, v1))
    auto Fs = info.surf_triangles();  // Surface face list (t => (v0, v1, v2))
    auto positions = info.positions();  // Vertex position buffer (positions[v] = 3D coordinate)

    //// Pre-check: Ensure adjacency relationships and BVH are initialized (handled in advance by preprocess_adjacency and detect functions)
    //if(m_vadj_offsets.empty() || candidate_AllP_AllT_pairs.empty())
    //    throw SimSystemException("OGC contact detect: adjacency/BVH not initialized");

    /****************************************************
    * Phase 1: Vertex-facet contact detection (corresponds to Document Algorithm 1)
    * Input: candidate_AllP_AllT_pairs (vertex-face candidate pairs, generated by lbvh_T query)
    * Output: temp_PTs/PEs/PPs (contact sets), m_d_min_v/m_d_min_t (minimum distances)
    ****************************************************/
    phase1_vertex_facet_contact(info);

    /***********************************
    * Phase 2: Edge-edge contact detection (corresponds to Document Algorithm 2)
    * Input: candidate_AllE_AllE_pairs (edge-edge candidate pairs, generated by lbvh_E query)
    * Output: temp_EEs/PEs/PPs (contact sets), m_d_min_e (minimum distance)
    ****************************************************/
    phase2_edge_edge_contact(info);

    /****************************************************
    * New Phase 3: Calculate conservative bounds bv for all vertices
    ****************************************************/
    int num_vertices = Vs.size();
    compute_conservative_bounds(num_vertices, m_gamma_p);

    /****************************************************
    * Subsequent: Filter valid contact pairs (reuse existing filter_active logic)
    ****************************************************/
    // filter_active(info);
}



/**
 * @brief OGC core function: Verify whether the point to be detected falls within the offset block (Uₐ) of a vertex, corresponding to Document Equation (8)
 * @details The vertex offset block Uₐ is a region of "a sphere with radius r, cut by the vertical planes of all adjacent vertices", which needs to satisfy two core constraints:
 *          1. The distance from the point to be detected to vertex a <= contact radius r (distance constraint)
 *          2. The angle between the line connecting the point to be detected and vertex a, and the line connecting vertex a to all adjacent vertices ≥ 90° (direction constraint)
 *          The function returns true if the point is within the offset block (can be used as a valid OGC contact pair); returns false if it is invalid
 * 
 * @param x              Point to be verified (e.g., vertex V in vertex-face contact, closest point in edge-edge contact)
 * @param a              Vertex ID corresponding to the vertex offset block (i.e., v in the equation, the vertex sub-face to be verified)
 * @param positions      Current position buffer of all vertices (device-side, stores 3D coordinates of each vertex)
 * @param vadj_offsets   Offset buffer of vertex adjacency relationships (CSR-like structure, stores the start index of each vertex's adjacency list)
 *                       Structure description: vadj_offsets[a] is the start position of vertex a's adjacency list in vadj_indices;
 *                                vadj_offsets[a+1] is the end position of vertex a's adjacency list (start of the next vertex)
 * @param vadj_indices   Index buffer of vertex adjacency relationships (CSR-like structure, stores specific adjacent vertex IDs)
 *                       Structure description: Elements from vadj_offsets[a] to vadj_offsets[a+1]-1 are all adjacent vertex IDs of vertex a
 * @param r              OGC contact radius (consistent with the offset radius in the document, controlling the size of the vertex offset block)
 * @return bool          true: The point is within the offset block Uₐ of vertex a; false: The point is not within the offset block
 */
//
//__device__ bool checkVertexFeasibleRegionEdgeOffset(const Vector3& x_c,
//                                                    int            v_idx,
//                                                    const muda::CDense1D<Vector3>& positions,
//                                                    const muda::CDense1D<Vector2i>& surf_edges,
//                                                    const muda::CDense1D<Vector2i>& v_edge_indices,
//                                                    float ogc_r)

// [OGC core function] Determine whether point x is within the offset block Uₑ of edge e (corresponds to Document Section 3.2 Equation 9, Algorithm 1 Line 15 checkEdgeFeasibleRegion)
// Definition of edge offset block Uₑ: A cylinder with radius r (axis along e) cut by 4 planes, only allowing points in "inside the edge + valid side of adjacent faces"
// Return value: true = point x is within the offset block of edge e (can generate orthogonal contact force), false = not within (exclude contact)
//__device__ bool checkEdgeFeasibleRegion(
//    const Vector3& x,  // Point to be judged (usually the position x(v) of vertex v in Algorithm 1)
//    int            e,  // Current edge e to be checked (index, corresponding to element in mesh edge set E)
//    const muda::CDense1D<Vector2i>& surf_edges,  // Vertex pairs of mesh edges: surf_edges(e) = (v1, v2), stores the two endpoint indices of edge e
//    const muda::CDense1D<Vector3>& positions,  // Position data of all vertices: positions(v) = 3D coordinate of vertex v
//    const muda::CDense1D<int>& edge_face_counts,  // Number of faces associated with edge e: edge_face_counts(e) = k (k <= 2 in manifold meshes, an edge belongs to at most 2 adjacent faces)
//    const muda::CDense1D<int>& edge_face_vertices,  // "Opposite vertex" of the face associated with edge e: edge_face_vertices(e*2 + k) = opp (the third vertex except edge e in face v1v2opp)
//    float r  // OGC contact radius (cylinder radius of the edge offset block, corresponding to r in the document)
//)
//{
//    // -------------------------- Step 1: Get basic geometric information of edge e --------------------------
//    // Get the two endpoints v1 and v2 of edge e from the edge set (Document Section 3.2: Edge e is composed of vertices v_{e,1} and v_{e,2})
//    Vector2i e_vs = surf_edges(e);   // e_vs: Vertex index pair of edge e (v1, v2)
//    int v1 = e_vs[0], v2 = e_vs[1];  // v1 = start point of edge e, v2 = end point of edge e (elements inside Vector2i are still accessed with [])
//    const Vector3& x1    = positions(v1);  // x1: Current position of vertex v1
//    const Vector3& x2    = positions(v2);  // x2: Current position of vertex v2
//    Vector3        vec_e = x2 - x1;        // vec_e: Direction vector of edge e (from v1 to v2)
//    float len_e_sq = vec_e.squaredNorm();  // len_e_sq: Squared length of edge e (avoids square root, improves GPU efficiency)
//
//    // Degenerate edge check: If the edge length is close to 0 (within floating-point error range), the edge offset block is meaningless, return false directly
//    //if(len_e_sq < 1e-12f)
//    //    return false;
//
//
//    // -------------------------- Step 2: Determine whether point x is within the "cylinder range" of edge e --------------------------
//    // Document Section 3.2 Equation 9 Condition 1: dis(x, e) <= r (the shortest distance from point x to edge e <= contact radius r, i.e., x is inside the cylinder with e as axis and r as radius)
//    Vector3 vec_x1x = x - x1;  // vec_x1x: Vector from v1 to x
//    float t = vec_x1x.dot(vec_e) / len_e_sq;  // t: Projection parameter of x on edge e (t (belong to) [0,1] means projection is inside the edge, t < 0 is outside v1, t > 1 is outside v2)
//
//    // [Optional] Force projection range to inside the edge (retained in comments because the subsequent dot1/dot2 judgment has equivalently implemented this logic)
//    // if(t < 0.f || t > 1.f) return false;
//
//    Vector3 closest = x1 + t * vec_e;  // closest: The shortest distance point (projection point) from x to edge e
//    // Check if the distance from x to the projection point is ≤ r (use squared distance to avoid square root, reduce GPU computation overhead, add 1e-6 floating-point error tolerance)
//    if((x - closest).squaredNorm() > r * r)
//        return false;
//
//
//    // -------------------------- Step 3: Determine whether point x is within the "internal cylinder segment" of edge e --------------------------
//    // Document Section 3.2 Equation 9 Conditions 2-3: Exclude x on the extension line of the endpoints of edge e (only retain the internal cylinder segment of the edge)
//    // Condition 2: (x - x_{e,1}) (dot) (x_{e,2} - x_{e,1}) > 0 => x is on the "side away from v2" of v1 (not outside v1)
//    float dot1 = vec_x1x.dot(vec_e);
//    // Condition 3: (x - x_{e,2}) (dot) (x_{e,1} - x_{e,2}) > 0 => x is on the "side away from v1" of v2 (not outside v2)
//    float dot2 = (x - x2).dot(-vec_e);  // -vec_e: Direction vector from v2 to v1
//
//    // If x is outside v1 or v2 (dot ≤ 0, floating-point error tolerance 1e-6), then it is not within the edge offset block, return false
//    if(dot1 <= 0 || dot2 <= 0)
//        return false;
//
//
//    // -------------------------- Step 4: Determine whether point x is on the "valid side of adjacent faces" --------------------------
//    // Document Section 3.2 Equation 9 Condition 4: Exclude x on the "forbidden side" of the faces adjacent to edge e (the edge offset block is the region of the cylinder cut by the vertical planes of the adjacent faces)
//    // An edge e is associated with at most 2 faces (manifold mesh), loop through each associated face
//    int cnt = edge_face_counts(e);  // cnt: Number of faces associated with edge e (k=0/1/2)
//    for(int k = 0; k < cnt; ++k)
//    {
//        // Get the "opposite vertex" opp of edge e in the k-th associated face (the face is composed of v1, v2, opp, opp is the third vertex except edge e)
//        int opp = edge_face_vertices(e * 2 + k);
//        if(opp < 0)  // Invalid vertex index (e.g., boundary edges are associated with only 1 face, the other position stores -1), skip
//            continue;
//
//        // Calculate the "foot of perpendicular p" of the opposite vertex opp on edge e (p(x1,x2,x3) in Document Section 3.2: the vertical projection of x3 on the line x1x2)
//        Vector3 vec_v1opp = positions(opp) - x1;  // vec_v1opp: Vector from v1 to opp
//        float t_p = vec_v1opp.dot(vec_e) / len_e_sq;  // t_p: Projection parameter of opp on edge e
//        Vector3 p = x1 + t_p * vec_e;                 // p: Foot of perpendicular of opp on edge e
//
//        // Document Section 3.2 Condition 4: (x - p) (dot) (p - opp) ≥ 0 => x is on the "outside" of face v1v2opp (meets the cutting plane constraint of the edge offset block)
//        Vector3 vec_xp = x - p;  // vec_xp: Vector from p to x
//        Vector3 vec_p_opp = p - positions(opp);  // vec_p_opp: Vector from opp to p (inner direction of the face)
//        float dot_adj = vec_xp.dot(vec_p_opp);  // Dot product to judge whether x is on the valid side of the cutting plane
//
//        // If x is on the "forbidden side" of the face (dot_adj < -1e-6, floating-point error tolerance), return false
//        if(dot_adj < 0)
//            return false;
//    }
//
//
//    // -------------------------- Step 5: All conditions are satisfied, x is within the offset block Uₑ of edge e --------------------------
//    return true;
//}


/**
 * Verify whether point x is within the offset block Uₑ of the edge (directly pass the positions of the two vertices of the edge, no need for edge index => vertex mapping)
 * Core modifications:
 * 1. Remove the logic of `surf_edges` and obtaining vertices through edge index `e`, directly receive the positions x1 and x2 of the two vertices of the edge
 * 2. Retain edge index `e` (because `edge_face_counts` and `edge_face_vertices` still need to access associated faces through e)
 * 3. The core constraint logic (cylinder range, endpoint restriction, adjacent face cutting) remains completely unchanged
 * 
 * @param x                     Point to be judged (e.g., position x(v) of vertex v)
 * @param x1                    Position of the first vertex of the edge (directly passed in)
 * @param x2                    Position of the second vertex of the edge (directly passed in)
 * @param e                     Global index of the edge (depends on associated face information)
 * @param positions             Position data of all vertices (used to get the position of the opposite vertex opp of the associated face)
 * @param edge_face_counts      Number of faces associated with edge e (k ≤ 2 in manifold meshes)
 * @param edge_face_vertices    "Opposite vertex" of the face associated with edge e (the third vertex except the edge in face v1v2opp)
 * @param r                     OGC contact radius (cylinder radius of the edge offset block)
 * @return bool                 true = x is within the offset block of the edge; false = not within
 */
__device__ bool checkEdgeFeasibleRegion(const Vector3& x,  // Point to be judged
                                        const Vector3& x1,  // Position of the first vertex of the edge (directly passed in)
                                        const Vector3& x2,  // Position of the second vertex of the edge (directly passed in)
                                        int e,  // Global index of the edge (depends on associated faces)
                                        const muda::CDense1D<Vector3>& positions,  // Position of all vertices (get position of opposite vertex opp)
                                        const muda::CDense1D<int>& edge_face_counts,  // Number of faces associated with edge e
                                        const muda::CDense1D<int>& edge_face_vertices,  // Opposite vertex of the face associated with edge e
                                        float r  // Contact radius
)
{
    // -------------------------- Step 1: Get basic geometric information of the edge (calculated directly based on x1 and x2) --------------------------
    Vector3 vec_e = x2 - x1;               // Direction vector of the edge (from x1 to x2)
    float len_e_sq = vec_e.squaredNorm();  // Squared length of the edge (avoids square root, improves GPU efficiency)

    // Degenerate edge check: Edges with length close to 0 have no meaning for offset blocks (floating-point error tolerance 1e-12)
    if(len_e_sq < 1e-12f)
        return false;


    // -------------------------- Step 2: Determine whether point x is within the "cylinder range" of the edge --------------------------
    // Condition: The shortest distance from point x to the edge ≤ r (i.e., x is inside the cylinder with the edge as axis and r as radius)
    Vector3 vec_x1x = x - x1;                 // Vector from x1 to x
    float t = vec_x1x.dot(vec_e) / len_e_sq;  // Projection parameter of x on the edge (t (belong to) [0,1] is inside the edge)

    Vector3 closest = x1 + t * vec_e;  // Shortest distance point (projection point) from x to the edge
    // Squared distance > r square + 1e-12 => not inside the cylinder (tolerate floating-point errors)
    if((x - closest).squaredNorm() > r * r + 1e-12f)
        return false;


    // -------------------------- Step 3: Determine whether point x is within the "internal cylinder segment" of the edge (exclude outside endpoints) --------------------------
    // Condition 1: x is not outside x1 ((x - x1) (dot) (x2 - x1) > 0)
    float dot1 = vec_x1x.dot(vec_e);
    // Condition 2: x is not outside x2 ((x - x2) (dot) (x1 - x2) > 0 => equivalent to (x - x2) (dot) (-vec_e) > 0)
    float dot2 = (x - x2).dot(-vec_e);

    // If x is outside x1 or x2 (dot ≤ 0), it is not within the edge offset block
    if(dot1 <= 1e-6f || dot2 <= 1e-6f)  // Relax to 1e-6 tolerance to be compatible with GPU floating-point precision
        return false;


    // -------------------------- Step 4: Determine whether point x is on the "valid side of adjacent faces" --------------------------
    // The edge offset block needs to be cut by the vertical planes of adjacent faces, only retaining the outer region (Document Section 3.2 Equation 9 Condition 4)
    int cnt = edge_face_counts(e);  // Number of faces associated with edge e (0/1/2)
    for(int k = 0; k < cnt; ++k)
    {
        // Get the "opposite vertex" opp of the k-th associated face (the face is composed of x1, x2, opp)
        int opp = edge_face_vertices(e * 2 + k);
        if(opp < 0)  // Invalid opposite vertex (e.g., boundary edges are associated with only 1 face, the other is -1), skip
            continue;

        // Calculate the foot of perpendicular p of opp on the edge (used to determine the cutting plane)
        const Vector3& x_opp     = positions(opp);      // Position of opposite vertex opp
        Vector3        vec_v1opp = x_opp - x1;          // Vector from x1 to opp
        float   t_p = vec_v1opp.dot(vec_e) / len_e_sq;  // Projection parameter of opp on the edge
        Vector3 p   = x1 + t_p * vec_e;                 // Position of foot of perpendicular p

        // Condition: (x - p) (dot) (p - x_opp) ≥ 0 => x is on the valid side (outside) of the cutting plane
        Vector3 vec_xp    = x - p;      // Vector from p to x
        Vector3 vec_p_opp = p - x_opp;  // Vector from opp to p (inner direction of the face)
        float   dot_adj   = vec_xp.dot(vec_p_opp);

        // If x is on the "forbidden side" of the face (dot_adj < -1e-6), return false
        if(dot_adj < 0)
            return false;
    }


    // -------------------------- All conditions are satisfied, x is within the offset block of the edge --------------------------
    return true;
}

__device__ bool checkVertexFeasibleRegion(
    const Vector3& x,  // test point：Point to be detected (e.g., candidate point in collision detection)
    int            a,  // vertex id：Vertex ID corresponding to the vertex offset block (v in the equation)
    const muda::CDense1D<Vector3>& positions,  // 3D positions of all vertices (device-side, index is vertex ID)
    const muda::CDense1D<IndexT>& vadj_offsets,  // Offset buffer of vertex adjacency list (CSR start index)
    const muda::CDense1D<IndexT>& vadj_indices,  // ID buffer of vertex adjacency list (CSR specific ID)
    float r)  // OGC contact radius (r in the equation, sphere radius of the offset block)
{
    // 1. Extract the position of vertex a and calculate the vector from the point x to be detected to vertex a (corresponds to x - x_v in the equation)
    const Vector3& x_a = positions(a);  // x_a：Current position of vertex a (x_v in the equation)
    Vector3 vec_xa = x - x_a;  // vec_xa：Vector from the point x to be detected to vertex a (x - x_v in the equation)

    // 2. Constraint 1: Distance constraint (||x - x_v|| <= r in Equation 8)
    // Optimization point: Use squaredNorm instead of norm to avoid square root operation and improve GPU computation efficiency
    // Tolerate small floating-point errors: If the squared distance is slightly larger than r square (e.g., caused by precision), it is still judged as out of range
    if(vec_xa.squaredNorm() > r * r)
        return false;  // Distance exceeds contact radius, point is not within the offset block

    // 3. Constraint 2: Direction constraint ((x - x_v)(dot)(x_v - x_{v'}) ≥ 0 in Equation 8, for all adjacent vertices v')
    // 3.1 Parse the range of the adjacency list of vertex a (based on CSR-like structure)
    const int start = (int)vadj_offsets(a);  // Start index of the adjacency list of vertex a in vadj_indices
    const int end = (int)vadj_offsets(a + 1);  // End index of the adjacency list of vertex a (start of the next vertex)
    // Traverse all adjacent vertices v' (v_prime) of vertex a
    for(int p = start; p < end; ++p)
    {
        // 3.2 Extract the ID and position of the current adjacent vertex v'
        const int v_prime = (int)vadj_indices(p);  // v_prime：An adjacent vertex of vertex a (v' in the equation)
        const Vector3& x_vp = positions(v_prime);  // x_vp：Position of adjacent vertex v' (x_{v'} in the equation)

        // 3.3 Calculate the vector for the direction constraint (x_v - x_{v'} in the equation)
        Vector3 vec_av = x_a - x_vp;  // vec_av：Vector from vertex a to adjacent vertex v' (reverse direction from a to v')

        // 3.4 Calculate the dot product to verify the direction constraint ((x - x_v)(dot)(x_v - x_{v'}) ≥ 0 in the equation)
        float dot = vec_xa.dot(vec_av);  // Dot product result: If ≥ 0, the angle <= 90°, satisfying the direction constraint
        // Floating-point error handling: Allow a deviation of -1e-6 (avoid misjudgment caused by GPU floating-point precision errors, e.g., theoretical 0 becomes -1e-15)
        //if(dot < -1e-6f)
        //    return false;  // Direction constraint not satisfied, point is not within the offset block
        if(dot < 0)
            return false;  // Direction constraint not satisfied, point is not within the offset block
    }

    // 4. All constraints are satisfied: The point x to be detected is within the offset block Uₐ of vertex a, return valid
    return true;

    // ------------------------------ Parallel call example (explanation in comments) ------------------------------
    // The following is a typical calling method of the function in a GPU parallel environment (based on muda's ParallelFor)
    // ParallelFor().apply(num_queries,  // num_queries：Number of parallel queries (e.g., total number of points to be detected)
    //                     [positions = info.positions().cviewer().name("positions"),  // Vertex positions (const view)
    //                      vadj_offsets = m_vadj_offsets.cviewer().name("vadj_offsets"),  // Adjacency offsets (const view)
    //                      vadj_indices = m_vadj_indices.cviewer().name("vadj_indices"),  // Adjacency IDs (const view)
    //                      r = ogc_radius] __device__(int i) mutable  // i：Parallel thread index
    //                     {
    //                         // Assume x is the position of the i-th point to be detected, v is the vertex ID to be verified
    //                         // bool ok = checkVertexFeasibleRegion(x, v, positions, vadj_offsets, vadj_indices, r);
    //                         // Mark whether the contact pair is valid according to the result of ok (e.g., add to FOGC contact set)
    //                     });
}

/**
 * [Refactored version] Verify whether the closest point x_c of edge-edge contact is within the offset block Uₐ of vertex v_idx (supports multiple adjacent vertices)
 * Core modifications:
 * 1. Use CSR structure (vadj_offsets/vadj_indices) instead of v_edge_indices/surf_edges to traverse all adjacent vertices
 * 2. Direction constraint verification covers all neighbors of vertex v_idx, no longer limited to 2 edge neighbors
 * 3. Adapt to scenarios with multiple adjacent vertices (e.g., bifurcated vertices in non-edge-only manifolds)
 * 
 * @param x_c           Closest point of edge-edge contact (point to be verified)
 * @param v_idx         Vertex ID corresponding to the vertex offset block (vertex to be verified)
 * @param positions     3D positions of all vertices (device-side, index=vertex ID)
 * @param vadj_offsets  CSR offset buffer of vertex adjacency list (vadj_offsets[v] = start index of neighbor list)
 * @param vadj_indices  CSR index buffer of vertex adjacency list (vadj_indices[p] = adjacent vertex ID)
 * @param ogc_r         OGC contact radius (sphere radius of the vertex offset block)
 * @return bool         true = x_c is within the offset block of vertex v_idx; false = not within
 */
__device__ bool checkVertexFeasibleRegionEdgeOffset(
    const Vector3&                 x_c,
    int                            v_idx,
    const muda::CDense1D<Vector3>& positions,
    const muda::CDense1D<IndexT>& vadj_offsets,  // New: CSR offsets (replace v_edge_indices)
    const muda::CDense1D<IndexT>& vadj_indices,  // New: CSR indices (replace surf_edges)
    float ogc_r)
{
    // -------------------------- Step 1: Basic verification and vertex position extraction --------------------------
    // 1.2 Extract the current position of vertex v_idx (center of the offset block)
    const Vector3& x_v = positions(v_idx);

    // -------------------------- Step 2: Distance constraint verification (||x_c - x_v|| <= ogc_r) --------------------------
    Vector3 vec_cv = x_c - x_v;             // Vector from x_c to x_v
    float dist_sq  = vec_cv.squaredNorm();  // Squared distance (avoids square root, improves GPU efficiency)
    float r_sq     = ogc_r * ogc_r;         // Squared contact radius

    // Tolerate small floating-point errors (e.g., slight out-of-range caused by GPU precision), squared distance > r_sq + 1e-12 is considered out of range
    if(dist_sq > r_sq + 1e-12f)
        return false;

    // -------------------------- Step 3: Traverse all adjacent vertices and verify direction constraints --------------------------
    // 3.1 Parse the neighbor range of vertex v_idx from the CSR structure (start=start index, end=end index)
    const int start = static_cast<int>(vadj_offsets(v_idx));
    const int end = static_cast<int>(vadj_offsets(v_idx + 1));  // "Start of next vertex" in CSR offsets is the end of current

    // 3.2 If there are no adjacent vertices (isolated vertex), only need to satisfy the distance constraint (special scenario)
    if(start >= end)
        return true;

    // 3.3 For each adjacent vertex v_prime, verify the direction constraint: (x_c - x_v) (dot) (x_v - x_v_prime) ≥ 0
    // (Meaning: x_c is on the opposite side of x_v pointing to v_prime, avoiding entering the overlapping area of neighbors' offset blocks)
    for(int p = start; p < end; ++p)
    {
        // 3.3.1 Extract adjacent vertex ID (note conversion from IndexT to int to adapt to muda containers)
        int v_prime = static_cast<int>(vadj_indices(p));
        // Skip self (CSR adjacency list theoretically does not contain self, here to prevent exceptions)
        if(v_prime == v_idx)
            continue;

        // 3.3.2 Extract the position of adjacent vertex v_prime
        const Vector3& x_vp = positions(v_prime);

        // 3.3.3 Calculate the vector and dot product for the direction constraint
        Vector3 vec_v_vp = x_v - x_vp;  // Vector from x_v to v_prime (reverse direction)
        float   dot      = vec_cv.dot(vec_v_vp);  // Dot product to judge direction

        // Tolerate floating-point errors: dot < 0 is considered unsatisfied (avoid misjudgment caused by GPU precision)
        if(dot < 0)  // Relax to 1e-6 tolerance to be compatible with small floating-point fluctuations
            return false;
    }

    // -------------------------- Step 4: Supplementary verification for degenerate cases (multiple overlapping neighbors) --------------------------
    // If multiple adjacent vertices have overlapping positions (e.g., degenerate edges), the offset block is considered invalid
    Vector3 first_neighbor_pos = positions(static_cast<int>(vadj_indices(start)));
    for(int p = start + 1; p < end; ++p)
    {
        int     v_prime           = static_cast<int>(vadj_indices(p));
        Vector3 curr_neighbor_pos = positions(v_prime);
        // Adjacent vertices have overlapping positions (squared distance < 1e-24, considered completely overlapping)
        if((curr_neighbor_pos - first_neighbor_pos).squaredNorm() < 1e-24f)
            return false;
    }

    // -------------------------- All constraints are satisfied --------------------------
    return true;
}


//__device__ bool checkVertexFeasibleRegionEdgeOffset(const Vector3& x_c,
//                                                    int            v_idx,
//                                                    const muda::CDense1D<Vector3>& positions,
//                                                    const muda::CDense1D<Vector2i>& surf_edges,
//                                                    const muda::CDense1D<Vector2i>& v_edge_indices,
//                                                    float ogc_r)
//{
//    // -------------------------- Step 1: Get the position of vertex v and adjacent vertices --------------------------
//    const Vector3& x_v = positions(v_idx);  // Container index uses () instead
//
//    // Extract the two edges that vertex v belongs to in the edge-only manifold
//    Vector2i v_edges = v_edge_indices(v_idx);           // Container index uses () instead
//    int      e1_idx = v_edges[0], e2_idx = v_edges[1];  // Elements inside Vector2i still use []
//
//    // Extract adjacent vertex v1 from edge e1 (exclude v itself)
//    Vector2i e1_vs = surf_edges(e1_idx);                 // Container index uses () instead
//    int v1 = (e1_vs[0] == v_idx) ? e1_vs[1] : e1_vs[0];  // Elements inside Vector2i still use []
//
//    // Extract adjacent vertex v2 from edge e2 (exclude v itself)
//    Vector2i e2_vs = surf_edges(e2_idx);                 // Container index uses () instead
//    int v2 = (e2_vs[0] == v_idx) ? e2_vs[1] : e2_vs[0];  // Elements inside Vector2i still use []
//
//    // Degenerate case: Two adjacent vertices are coincident (edge is a point, meaningless)
//    if(v1 == v2 || (positions(v1) - positions(v2)).squaredNorm() < 1e-12f)  // Container index uses () instead
//        return false;
//
//    // -------------------------- Step 2: Verify distance constraint (||x_c - x_v|| <= r) --------------------------
//    Vector3 vec_cv  = x_c - x_v;
//    float   dist_sq = vec_cv.squaredNorm();
//    if(dist_sq > ogc_r * ogc_r + 1e-12f)  // Squared distance > r square, not satisfied
//        return false;
//
//    // -------------------------- Step 3: Verify direction constraints (for two adjacent vertices) --------------------------
//    // 3.1 Direction constraint with v1: (x_c - x_v) (dot) (x_v - x_v1) ≥ 0
//    Vector3 vec_vv1 = x_v - positions(v1);  // Container index uses () instead
//    float   dot1    = vec_cv.dot(vec_vv1);
//
//    // 3.2 Direction constraint with v2: (x_c - x_v) (dot) (x_v - x_v2) ≥ 0
//    Vector3 vec_vv2 = x_v - positions(v2);  // Container index uses () instead
//    float   dot2    = vec_cv.dot(vec_vv2);
//
//    // Tolerate floating-point errors: dot < 0 is considered unsatisfied (avoid misjudgment due to precision)
//    if(dot1 < 0 || dot2 < 0)
//        return false;
//
//    // -------------------------- All constraints are satisfied --------------------------
//    return true;
//}


//__device__ bool checkVertexFeasibleRegionEdgeOffset(
//    const Vector3&                 x_c,
//    int                            v_idx,
//    const muda::CDense1D<Vector3>& positions,
//    const muda::CDense1D<IndexT>& vadj_offsets,  // New: CSR offsets (replace v_edge_indices)
//    const muda::CDense1D<IndexT>& vadj_indices,  // New: CSR indices (replace surf_edges)
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
    // Traverse the adjacent edges of a
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
    auto Vs        = info.surf_vertices();
    auto Fs        = info.surf_triangles();
    auto Es        = info.surf_edges();
    auto positions = info.positions();
    ///////////this is not added yet
    // ///////
    //auto edge_face_counts   = info.edge_face_counts();    // Original edge-face association count
    //auto edge_face_vertices = info.edge_face_vertices();  // Original edge-face association vertices

    // Number of candidate pairs: each candidate pair is (v_idx_in_Vs, t_idx) (Vs is the list of surface vertices, v_idx_in_Vs is the index in Vs)
    SizeT num_candidates = candidate_AllP_AllT_pairs.size();
    if(num_candidates == 0)
        return;

    // Initialize temporary contact sets (clear data from previous frame)
    temp_PTs.resize(num_candidates);
    temp_PEs.resize(num_candidates);
    temp_PPs.resize(num_candidates);

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(
            num_candidates,
            [  // Input parameters (OGC configuration)
                ogc_r  = m_ogc_r,
                ogc_rq = m_ogc_rq,
                // Input parameters (geometric data)
                Vs = Vs.cviewer().name("Vs"),
                Fs = Fs.cviewer().name("Fs"),
                Es = Es.cviewer().name("Es"),  // Added
                edge_face_counts = m_edge_face_counts.cviewer().name("edge_face_counts"),
                edge_face_vertices = m_edge_face_vertices.cviewer().name("edge_face_vertices"),
                //m_edge_id_map = m_edge_id_map.cviewer().name("m_edge_id_map"),
                positions = positions.cviewer().name("positions"),
                // Vertex=>adjacent edge CSR (added)
                v_edge_offsets = m_v_edge_offsets.cviewer().name("v_edge_offsets"),
                v_edge_indices = m_v_edge_indices.cviewer().name("v_edge_indices"),
                // Input parameters (adjacency relationship)
                /*                vadj_offsets = m_vadj_offsets.cviewer().name("vadj_offsets"),
                   vadj_indices = m_vadj_indices.cviewer().name("vadj_indices"),*/
                //vadj_offsets = m_v_edge_offsets.cviewer().name("vadj_offsets"),
                //vadj_indices = m_v_edge_indices.cviewer().name("vadj_indices"),

                vertex_offsets = m_v_vertex_offsets.cviewer().name("vadj_offsets"),
                vertex_indices = m_v_vertex_indices.cviewer().name("vadj_indices"),

                // Input parameters (candidate pairs)
                candidates = candidate_AllP_AllT_pairs.viewer().name("candidates"),
                // Output parameters (contact sets)
                temp_PTs = temp_PTs.viewer().name("temp_PTs"),
                temp_PEs = temp_PEs.viewer().name("temp_PEs"),
                temp_PPs = temp_PPs.viewer().name("temp_PPs"),
                // Output parameters (minimum distance)
                d_min_v = m_d_min_v.viewer().name("d_min_v"),
                d_min_t = m_d_min_t.viewer().name("d_min_t")] __device__(int idx) mutable
            {
                // 1. Parse candidate pair: (v_idx_in_Vs, t_idx) => global vertex ID v, face ID t
                Vector2i cand        = candidates(idx);
                int      v_idx_in_Vs = cand[0];
                int      t_idx       = cand[1];
                int v = Vs(v_idx_in_Vs);  // Global vertex ID (index of positions)
                Vector3i t_vs = Fs(t_idx);  // Three vertices of face t (v0, v1, v2)

                // 2. Algorithm 1 Line 3: Skip the face that v belongs to (v ⊂ t)
                if(t_vs[0] == v || t_vs[1] == v || t_vs[2] == v)
                {
                    temp_PTs(idx).setConstant(-1);
                    temp_PEs(idx).setConstant(-1);
                    temp_PPs(idx).setConstant(-1);
                    return;
                }

                // 3. Algorithm 1 Line 4: Calculate distance d from vertex v to face t
                const Vector3& v_pos    = positions(v);
                const Vector3& t_v0_pos = positions(t_vs[0]);
                const Vector3& t_v1_pos = positions(t_vs[1]);
                const Vector3& t_v2_pos = positions(t_vs[2]);

                //distance::point_triangle_distance_flag
                Vector4i flag =
                    distance::point_triangle_distance_flag(v_pos, t_v0_pos, t_v1_pos, t_v2_pos);

                /////////For edge-edge, vertex positions are always passed in; it seems there's no involvement of IDs??????
                Float d_square;
                distance::point_triangle_distance2(
                    flag, v_pos, t_v0_pos, t_v1_pos, t_v2_pos, d_square);
                Float d_update = sqrtf(d_square);

                //////////////////////please note that we use d_square instead of d!!!!!!!!!!!!!!!!!!!!!!!
                //// 4. Algorithm 1 Line 5: Update d_min_v (minimum distance from vertex v to a face)
                //atomic_min(&d_min_v(v), d_update);
                //// 5. Algorithm 1 Line 6: Atomically update d_min_t (minimum distance from face t to a vertex, avoiding multi-thread competition)
                //atomic_min(&d_min_t(t_idx), d_update);

                // 6. Algorithm 1 Line 7: Distance ≥ r, no contact formed, skip
                if(d_square >= ogc_r * ogc_r)  // Tolerate floating-point errors
                {
                    temp_PTs(idx).setConstant(-1);
                    temp_PEs(idx).setConstant(-1);
                    temp_PPs(idx).setConstant(-1);
                    return;
                }

                // 7. Algorithm 1 Line 8: Find the closest subface a on face t to v (vertex/edge/interior of the face)
                ///// now flag has successfully reflected by the position relation between point and triangle
                //ContactFace a = find_closest_subface(v_pos, t_idx, Fs, positions);
                // Now we do not need this corner case, will be tested later
                //if(a == -1)  // Degenerate subface (e.g., invalid edge)
                //{
                //    temp_PTs(idx).setConstant(-1);
                //    temp_PEs(idx).setConstant(-1);
                //    temp_PPs(idx).setConstant(-1);
                //    return;
                //}

                // 8. Algorithm 1 Line 9: Filter duplicate contacts (simplification: utilize uniqueness of candidate pairs to avoid multi-thread traversal)
                // (Note: To strictly remove duplicates, an additional flag buffer can be added; omitted here to simplify logic)

                // Calculate the number of active points (dim determines the degeneracy type)
                int dim = distance::detail::active_count(flag);


                // 9. Algorithm 1 Lines 10-21: Call check function according to subface type to verify offset block
                bool is_valid = false;
                Vector4i pt_pair = {-1, -1, -1, -1};  // PT: (v, t_v0, t_v1, t_v2)
                Vector3i pe_pair = {-1, -1, -1};      // PE: (v, e_v0, e_v1)
                Vector2i pp_pair = {-1, -1};          // PP: (v, a_vertex)

                if(dim == 2)  // Degeneracy type: point-to-point (subface is a vertex of the triangle)
                {
                    //// Get indices of the two points involved in calculation from flag (P array: [v, t0, t1, t2])
                    Vector2i offsets = distance::detail::pp_from_pt(flag);
                    // Among them, offsets[0] should be v (index 0), offsets[1] is a vertex of the triangle (index 1/2/3)
                    int tri_vertex_idx_in_P = offsets[1];  // Index of the triangle vertex in the P array
                    int a_vertex = t_vs[tri_vertex_idx_in_P - 1];  // Map to the vertex ID of the triangle (t_vs is [0:t0,1:t1,2:t2])

                    is_valid = checkVertexFeasibleRegion(
                        v_pos, a_vertex, positions, vertex_offsets, vertex_indices, ogc_r);
                    if(is_valid)
                    {
                        pp_pair = {v, a_vertex};  // Record point-point contact pair
                        temp_PPs(idx) = pp_pair;
                    }
                }
                else if(dim == 3)  // Degeneracy type: point-to-edge (subface is an edge of the triangle)
                {
                    // Get indices of the three points involved in calculation from flag (P array: [v, t0, t1, t2])
                    Vector3i offsets = distance::detail::pe_from_pt(flag);
                    // Among them, offsets[0] should be v (index 0), offsets[1] and offsets[2] are two vertices of the triangle (forming an edge)
                    int tri_v0_idx_in_P = offsets[1];  // Index of the first vertex of the edge in the P array
                    int tri_v1_idx_in_P = offsets[2];  // Index of the second vertex of the edge in the P array
                    Vector2i e_vs = {t_vs[tri_v0_idx_in_P - 1],  // Map to the vertex ID of the triangle
                                     t_vs[tri_v1_idx_in_P - 1]};

                    // ... Previous logic ...
                    int v0 = e_vs[0], v1 = e_vs[1];
                    int edge_id =
                        find_edge_id(v0, v1, v_edge_offsets, v_edge_indices, Es);

                    // Pass when calling checkEdgeFeasibleRegion:
                    is_valid = checkEdgeFeasibleRegion(v_pos,
                                                       positions(v0),
                                                       positions(v1),
                                                       edge_id,
                                                       positions,
                                                       edge_face_counts,  // Directly use device-side member variable
                                                       edge_face_vertices,
                                                       ogc_r);
                    if(is_valid)
                    {
                        pe_pair = {v, v0, v1};  // Record point-edge contact pair
                        temp_PEs(idx) = pe_pair;
                    }

                    ///////////////////////////////////=====================================
                    //// Verify the feasibility of point-edge contact (assuming the check function can directly use vertex pairs or pass edge IDs)
                    //is_valid = checkEdgeFeasibleRegion(
                    //    v_pos, e_vs, Es, positions, edge_face_counts, edge_face_vertices, ogc_r);
                    ////////////////////////////////=========================================
                }
                else if(dim == 4)  // Non-degenerate: point to the interior of the triangle (subface is the face itself)
                {
                    is_valid = true;  // No additional verification needed for interior face contact
                    pt_pair = {v, t_vs[0], t_vs[1], t_vs[2]};  // Record point-face contact pair
                    temp_PTs(idx) = pt_pair;
                }
                else  // Invalid flag (theoretically will not be triggered, the original distance calculation function has performed verification)
                {
                    temp_PTs(idx).setConstant(-1);
                    temp_PEs(idx).setConstant(-1);
                    temp_PPs(idx).setConstant(-1);
                    return;
                }
                //// 10. Write to temporary contact set
                //temp_PTs(idx) = pt_pair;
                //temp_PEs(idx) = pe_pair;
                //temp_PPs(idx) = pp_pair;
            });
}

// Typical clamp function definition (global in the project)
template <typename T>
__device__ T clamp(T val, T min_val, T max_val)
{
    return val < min_val ? min_val : (val > max_val ? max_val : val);
}
// Auxiliary function: Calculate edge-edge closest point (corresponds to Algorithm 2 Line 8 C(e,e'))
__device__ Vector3 edge_edge_closest_point(Vector2i e_vs,
                                           Vector2i ep_vs,
                                           const muda::CDense1D<Vector3>& positions)
{
    const Vector3 &p0 = positions(e_vs[0]), p1 = positions(e_vs[1]);
    const Vector3 &q0 = positions(ep_vs[0]), q1 = positions(ep_vs[1]);
    // Standard edge-edge closest point calculation (parametric solution)
    Vector3 d1 = p1 - p0, d2 = q1 - q0, d3 = p0 - q0;
    float a = d1.dot(d1), b = d1.dot(d2), c = d2.dot(d2), e = d1.dot(d3), f = d2.dot(d3);
    float denom = a * c - b * b;
    float s     = 0.5f;  // Default midpoint (degenerate edge handling)
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

    // Number of candidate pairs: each candidate pair is (e_idx, e_prime_idx)
    SizeT num_candidates = candidate_AllE_AllE_pairs.size();
    if(num_candidates == 0)
        return;

    // Initialize temporary edge-edge contact set
    temp_EEs.resize(num_candidates);
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(
            num_candidates,
            [  // Input parameters (OGC configuration)
                ogc_r  = m_ogc_r,
                ogc_rq = m_ogc_rq,
                // Input parameters (geometric data)
                Es        = Es.viewer().name("Es"),
                positions = positions.viewer().name("positions"),
                //v_edge_indices = m_v_edge_indices.viewer().name("v_edge_indices"),
                //vadj_offsets = m_v_edge_offsets.cviewer().name("vadj_offsets"),
                //vadj_indices = m_v_edge_indices.cviewer().name("vadj_indices"),

                vertex_offsets = m_v_vertex_offsets.cviewer().name("vadj_offsets"),
                vertex_indices = m_v_vertex_indices.cviewer().name("vadj_indices"),
                // Input parameters (candidate pairs)
                candidates = candidate_AllE_AllE_pairs.viewer().name("candidates"),
                // Output parameters (contact sets)
                temp_EEs = temp_EEs.viewer().name("temp_EEs"),
                temp_PEs = temp_PEs.viewer().name("temp_PEs"),
                temp_PPs = temp_PPs.viewer().name("temp_PPs"),
                // Output parameters (minimum distance)
                d_min_e = m_d_min_e.viewer().name("d_min_e")] __device__(int idx) mutable
            {
                // 1. Parse candidate pair: (e_idx, e_prime_idx)
                Vector2i cand        = candidates(idx);
                int      e_idx       = cand[0];
                int      e_prime_idx = cand[1];
                Vector2i e_vs = Es(e_idx);  // Vertices of edge e (v0, v1)
                Vector2i ep_vs = Es(e_prime_idx);  // Vertices of edge e' (v0', v1')
                int e_v0 = e_vs[0], e_v1 = e_vs[1];
                int ep_v0 = ep_vs[0], ep_v1 = ep_vs[1];

                // 2. Algorithm 2 Line 5: Skip adjacent edges (e and e' share a common vertex)
                if(e_v0 == ep_v0 || e_v0 == ep_v1 || e_v1 == ep_v0 || e_v1 == ep_v1)
                {
                    temp_EEs(idx).setConstant(-1);
                    return;
                }

                // 3. Algorithm 2 Line 6: Calculate distance d from edge e to edge e'
                const Vector3 &e_p0 = positions(e_v0), e_p1 = positions(e_v1);
                const Vector3 &ep_p0 = positions(ep_v0), ep_p1 = positions(ep_v1);

                Vector4i flag =
                    distance::edge_edge_distance_flag(e_p0, e_p1, ep_p0, ep_p1);
                Float d_square;
                distance::edge_edge_distance2(flag, e_p0, e_p1, ep_p0, ep_p1, d_square);

                Float d_update = sqrtf(d_square);
                // 4. Algorithm 2 Line 7: Update d_min_e (minimum distance from edge e to another edge)
                //atomic_min(&d_min_e(e_idx), d_update);
                //atomic_min(&d_min_e(e_prime_idx), d_update);  // Also update the minimum distance of e'

                //// 5. Algorithm 1 Line 6: Atomically update d_min_t (minimum distance from face t to a vertex, avoiding multi-thread competition)


                // 5. Algorithm 2 Line 8: Distance ≥ r, no contact formed, skip
                if(d_square >= ogc_r * ogc_r)
                {
                    temp_EEs(idx).setConstant(-1);
                    return;
                }

                // 6. Algorithm 2 Line 8: Calculate edge-edge closest point x_c (closest point on edge e)
                Vector3 x_c = edge_edge_closest_point(e_vs, ep_vs, positions);


                // Calculate the number of active points (dim determines the degeneracy type)
                int dim = distance::detail::active_count(flag);

                // 7. Process contact pairs according to degeneracy type (dim), replacing the original find_edge_closest_subface
                bool is_valid = false;
                Vector4i ee_pair = {-1, -1, -1, -1};  // EE: (e_v0, e_v1, ep_v0, ep_v1)
                Vector2i pp_pair = {-1, -1};  // PP: (a_vertex, ep_vertex)

                if(dim == 2)  // Fully degenerate: point-point contact (both edges degenerate to points)
                {
                    // Get indices of 2 valid points from flag (P array: [ea0,ea1,eb0,eb1])
                    Vector2i offsets = distance::detail::pp_from_ee(flag);
                    // Map to original vertex IDs (offsets[0] corresponds to a point on edge e, offsets[1] corresponds to a point on edge ep)
                    int a_vertex = (offsets[0] == 0) ? e_v0 : e_v1;  // Degenerated point on edge e
                    int ep_vertex = (offsets[1] == 2) ? ep_v0 : ep_v1;  // Degenerated point on edge ep

                    // Verify vertex feasibility (reuse original check logic)
                    is_valid = checkVertexFeasibleRegionEdgeOffset(
                        x_c, a_vertex, positions, vertex_offsets, vertex_indices, ogc_r);

                    if(is_valid)
                        pp_pair = {a_vertex, ep_vertex};
                }
                else if(dim == 3)  // Partially degenerate: point-edge contact (one edge degenerates, one edge is normal)
                {
                    // Get indices of 3 valid points from flag (1 degenerate point + 2 edge endpoints)
                    Vector3i offsets = distance::detail::pe_from_ee(flag);
                    // Determine which is the degenerate point (index appearing only once): 0/1 belong to edge e, 2/3 belong to edge ep
                    int point_idx = -1, edge_v0_idx = -1, edge_v1_idx = -1;
                    if(offsets[0] == offsets[1])  // First two indices are the same => edge e degenerates
                    {
                        point_idx   = offsets[0];
                        edge_v0_idx = offsets[1];
                        edge_v1_idx = offsets[2];
                    }
                    else  // Last two indices are the same => edge ep degenerates
                    {
                        point_idx   = offsets[2];
                        edge_v0_idx = offsets[0];
                        edge_v1_idx = offsets[1];
                    }

                    // Map to original vertex ID
                    int a_vertex = (point_idx == 0) ? e_v0 :
                                   (point_idx == 1) ? e_v1 :
                                   (point_idx == 2) ? ep_v0 :
                                                      ep_v1;

                    // Verify vertex feasibility (reuse original check logic)
                    //is_valid = checkVertexFeasibleRegionEdgeOffset(
                    //    x_c, a_vertex, positions, Es, v_edge_indices, ogc_r);
                    //is_valid = checkVertexFeasibleRegionEdgeOffset(
                    //    x_c, a_vertex, positions, vadj_offsets, vadj_indices, ogc_r);
                    is_valid = checkVertexFeasibleRegionEdgeOffset(
                        x_c, a_vertex, positions, vertex_offsets, vertex_indices, ogc_r);

                    if(is_valid)
                    {
                        // Point-edge contact is still recorded as a point-point pair (matching original logic)
                        int ep_closest_v = (edge_v0_idx == 2) ? ep_v0 :
                                           (edge_v0_idx == 3) ? ep_v1 :
                                           (edge_v0_idx == 0) ? e_v0 :
                                                                e_v1;
                        pp_pair          = {a_vertex, ep_closest_v};
                    }
                }
                else if(dim == 4)  // Non-degenerate: edge-edge contact (both edges are normal)
                {
                    is_valid = true;  // No verification needed for interior edge contact, directly valid
                    ee_pair = {e_v0, e_v1, ep_v0, ep_v1};
                }
                else  // Invalid flag (verified by original distance calculation function, theoretically not triggered)
                {
                    temp_EEs(idx).setConstant(-1);
                    return;
                }

                // 8. Write to temporary contact set
                temp_EEs(idx) = ee_pair;
                if(is_valid && pp_pair[0] != -1)
                    temp_PPs(idx) = pp_pair;  // Vertex-vertex contact is written to PP set
            });
}

void LBVHSimplexTrajectoryFilter::Impl::compute_conservative_bounds(int num_vertices, float gamma_p)
{
    // Initialize bv buffer (length = number of vertices)
    m_bv.resize(num_vertices);

    // Parallelly compute bv for each vertex (each thread processes one vertex v)
    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(num_vertices,
               [  // Input: CSR adjacency list (vertex-neighbor edges/faces)
                   v_edge_offsets = m_v_edge_offsets.cviewer().name("v_edge_offsets"),
                   v_edge_indices = m_v_edge_indices.cviewer().name("v_edge_indices"),
                   v_face_offsets = m_v_face_offsets.cviewer().name("v_face_offsets"),
                   v_face_indices = m_v_face_indices.cviewer().name("v_face_indices"),
                   // Input: Minimum distances updated by collision detection
                   d_min_v = m_d_min_v.viewer().name("d_min_v"),
                   d_min_e = m_d_min_e.viewer().name("d_min_e"),
                   d_min_t = m_d_min_t.viewer().name("d_min_t"),
                   // Input: Parameters
                   gamma = gamma_p,
                   // Output: Conservative boundary bv
                   bv   = m_bv.viewer().name("bv"),
                   cout = KernelCout::viewer()] __device__(int v) mutable
               {
                   // -------------------------- Step 1: Get d_min_v[v] (minimum distance from vertex to other faces) --------------------------
                   float min_v = d_min_v(v);  // d_min_v updated in phase1

                   // -------------------------- Step 2: Calculate d_min_v^E (minimum of minimum distances from vertex's neighboring edges to other edges) --------------------------
                   float min_v_E = FLT_MAX;  // Initialize to maximum float value
                   int e_start = v_edge_offsets(v);
                   int e_end   = v_edge_offsets(v + 1);
                   for(int p = e_start; p < e_end; ++p)
                   {
                       int e = v_edge_indices(p);  // A neighboring edge ID of vertex v
                       if(d_min_e(e) < min_v_E)
                       {  // Take the minimum value of neighboring edges' d_min_e
                           min_v_E = d_min_e(e);
                       }
                   }
                   // Boundary handling: If there are no neighboring edges (theoretically not happening), replace with min_v
                   if(min_v_E == FLT_MAX)
                       min_v_E = min_v;

                   // -------------------------- Step 3: Calculate d_min_v^T (minimum of minimum distances from vertex's neighboring faces to other vertices) --------------------------
                   float min_v_T = FLT_MAX;  // Initialize to maximum float value
                   int t_start = v_face_offsets(v);
                   int t_end   = v_face_offsets(v + 1);
                   for(int p = t_start; p < t_end; ++p)
                   {
                       int t = v_face_indices(p);  // A neighboring face ID of vertex v
                       if(d_min_t(t) < min_v_T)
                       {  // Take the minimum value of neighboring faces' d_min_t
                           min_v_T = d_min_t(t);
                       }
                   }
                   // Boundary handling: If there are no neighboring faces (theoretically not happening), replace with min_v
                   if(min_v_T == FLT_MAX)
                       min_v_T = min_v;

                   // -------------------------- Step 4: Calculate bv = γₚ * min(three minimum distances) --------------------------
                   // Take the minimum of the three values to avoid negative distances (floating-point errors)
                   // Output min_v, min_v_E, and min_v_T values in GPU here
                   float min_all = std::min(std::min(min_v, min_v_E), min_v_T);
                   min_all = std::max(min_all, 1e-12f);  // Prevent min_all from being 0 or negative (to avoid bv=0)
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

void LBVHSimplexTrajectoryFilter::do_detect_ogc(DetectInfo& info)
{
    m_impl.init_ogc_data(info);
    m_impl.detect(info);
}

void LBVHSimplexTrajectoryFilter::do_filter_active(FilterActiveInfo& info)
{
    m_impl.filter_active(info);
}

void LBVHSimplexTrajectoryFilter::do_filter_active_ogc(FilterActiveInfo& info)
{
    //=============m_impl.detect_ogc_contact(info);
    //m_impl.detect(info);

    m_impl.filter_active_dcd_distance(info);
    m_impl.compute_conservative_bounds(info.surf_vertices().size(), m_impl.m_gamma_p);
    //=============Original filter active does not compute bv
    //This causes the final calculation result to be slightly different; could this be a potential issue??????????
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
        return *addr;  // Ignore NaN

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

    // We will filter out the active pairs
    auto positions = info.positions();

    SizeT N_PCoimP  = candidate_AllP_CodimP_pairs.size();
    SizeT N_CodimPE = candidate_CodimP_AllE_pairs.size();
    SizeT N_PTs     = candidate_AllP_AllT_pairs.size();
    SizeT N_EEs     = candidate_AllE_AllE_pairs.size();

    // PT, EE, PT, and PP can degenerate to PP
    temp_PPs.resize(N_PCoimP + N_CodimPE + N_PTs + N_EEs);
    // PT, EE, and PT can degenerate to PE
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
                    d_hats   = info.d_hats().viewer().name("d_hats"),
                    // Pass OGC-dependent adjacency list (vertex-neighbor vertex CSR)
                    vertex_offsets = m_v_vertex_offsets.cviewer().name("vertex_offsets"),
                    vertex_indices = m_v_vertex_indices.cviewer().name("vertex_indices"),
                    ogc_r   = m_ogc_r,
                    d_min_v = m_d_min_v.viewer().name("d_min_v"),
                    d_min_t = m_d_min_t.viewer().name("d_min_t"),
                    d_min_e = m_d_min_e.viewer().name("d_min_e")] __device__(int i) mutable
                   {
                       // Default to invalid
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
                           return;  // Early return

                       // [Added OGC Logic]Verify if V0 is within the offset block of P1 (point-point offset block verification)
                       bool is_ogc_valid = checkVertexFeasibleRegion(
                           V0,  // Point to be detected (position of surface vertex P0)
                           P1,  // Offset block vertex (codimension vertex P1)
                           positions,       // All vertex positions
                           vertex_offsets,  // Vertex adjacency list CSR offsets
                           vertex_indices,  // Vertex adjacency list CSR indices
                           ogc_r            // OGC contact radius
                       );

                       // Mark as valid contact pair only if OGC verification passes
                       if(is_ogc_valid)
                       {
                           PP = {P0, P1};
                           // Add after "distance::point_point_distance2(V0, V1, D);":
                           Float d_update = sqrtf(D);  // Note: Original D is squared distance, need to take square root
                           // Atomic update: Minimum distance from vertex P0 to other points
                           //atomicMin(&d_min_v(P0), d_update);
                           atomicMinFloat(&d_min_v(P0), d_update);
                           // After (CUDA atomic)

                           //atomic_min(d_min_v.data() + P0, d_update);
                           //atomicMin(d_min_v.data() + P0, d_update);

                           //atomic_min
                           //atomic_min(&d_min_v(P0), d_update);
                           // Atomic update: Minimum distance from vertex P1 to other points (also need to record if it's a codimension vertex)
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
                 d_hats      = info.d_hats().viewer().name("d_hats"),
                 // Pass OGC-dependent edge-face associations and vertex-adjacent edge CSR
                 edge_face_counts = m_edge_face_counts.cviewer().name("edge_face_counts"),
                 edge_face_vertices = m_edge_face_vertices.cviewer().name("edge_face_vertices"),
                 v_edge_offsets = m_v_edge_offsets.cviewer().name("v_edge_offsets"),
                 v_edge_indices = m_v_edge_indices.cviewer().name("v_edge_indices"),
                 v_vertex_offsets = m_v_vertex_offsets.cviewer().name("v_vertex_offsets"),
                 v_vertex_indices = m_v_vertex_indices.cviewer().name("v_vertex_indices"),
                 ogc_r   = m_ogc_r,
                 d_min_v = m_d_min_v.viewer().name("d_min_v"),
                 d_min_t = m_d_min_t.viewer().name("d_min_t"),
                 d_min_e = m_d_min_e.viewer().name("d_min_e")] __device__(int i) mutable
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
                        return;  // Early return


                    // [OGC Logic]Degeneracy classification + offset block verification
                    Vector3i offsets;
                    auto dim = distance::degenerate_point_edge(flag, offsets);
                    bool is_ogc_valid = false;

                    switch(dim)
                    {
                        case 2:  // Degenerate PP (point-point)
                        {
                            IndexT V0 = vIs(offsets(0));  // Codimension vertex
                            IndexT V1 = vIs(offsets(1));  // Edge vertex
                            // OGC: Verify if V0 is within the offset block of V1
                            is_ogc_valid = checkVertexFeasibleRegion(
                                positions(V0), V1, positions, v_vertex_offsets, v_vertex_indices, ogc_r);
                            if(is_ogc_valid)
                            {
                                PP = {V0, V1};
                                // Add after "distance::point_edge_distance2(flag, Ps[0], Ps[1], Ps[2], D);":
                                Float d_update = sqrtf(D);
                                // Atomic update: Minimum distance from codimension vertex V to the edge
                                //atomic_min(&d_min_v(V), d_update);
                                atomicMinFloat(&d_min_v(V), d_update);
                                // Atomic update: Minimum distance of edge E (E is surf_edges(indices(1)), need to get edge ID first)
                                int e_idx = indices(1);  // The index of the edge in surf_edges is the edge ID
                                //atomic_min(&d_min_e(e_idx), d_update);
                                atomicMinFloat(&d_min_e(e_idx), d_update);
                            }
                        }
                        break;
                        case 3:  // Non-degenerate PE (point-edge)
                        {
                            int v0 = E(0), v1 = E(1);
                            // OGC: Find edge ID by vertex pair (reuse your find_edge_id)
                            int edge_id =
                                find_edge_id(v0, v1, v_edge_offsets, v_edge_indices, surf_edges);
                            if(edge_id == -1)
                                break;
                            // OGC: Verify if codimension vertex V is within the edge's offset block
                            is_ogc_valid = checkEdgeFeasibleRegion(
                                positions(V),  // Point to be detected (codimension vertex V)
                                positions(v0),  // Position of edge vertex v0
                                positions(v1),  // Position of edge vertex v1
                                edge_id,        // Edge ID
                                positions,
                                edge_face_counts,
                                edge_face_vertices,
                                ogc_r);
                            if(is_ogc_valid)
                            {
                                PE = vIs;
                                // Add after "distance::point_edge_distance2(flag, Ps[0], Ps[1], Ps[2], D);":
                                Float d_update = sqrtf(D);
                                // Atomic update: Minimum distance from codimension vertex V to the edge
                                atomicMinFloat(&d_min_v(V), d_update);
                                // Atomic update: Minimum distance of edge E (E is surf_edges(indices(1)), need to get edge ID first)
                                int e_idx = indices(1);  // The index of the edge in surf_edges is the edge ID
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
                 d_hats      = info.d_hats().viewer().name("d_hats"),
                 // Pass all OGC-dependent parameters (consistent with phase1)
                 Es = info.surf_edges().viewer().name("Es"),
                 edge_face_counts = m_edge_face_counts.cviewer().name("edge_face_counts"),
                 edge_face_vertices = m_edge_face_vertices.cviewer().name("edge_face_vertices"),
                 v_edge_offsets = m_v_edge_offsets.cviewer().name("v_edge_offsets"),
                 v_edge_indices = m_v_edge_indices.cviewer().name("v_edge_indices"),
                 vertex_offsets = m_v_vertex_offsets.cviewer().name("vertex_offsets"),
                 vertex_indices = m_v_vertex_indices.cviewer().name("vertex_indices"),
                 ogc_r   = m_ogc_r,
                 d_min_v = m_d_min_v.viewer().name("d_min_v"),
                 d_min_t = m_d_min_t.viewer().name("d_min_t"),
                 d_min_e = m_d_min_e.viewer().name("d_min_e")] __device__(int i) mutable
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
                        return;  // Early return

                    // [Embed OGC logic from phase1]Degeneracy classification + offset block verification
                    Vector4i offsets;
                    auto dim = distance::degenerate_point_triangle(flag, offsets);
                    bool is_ogc_valid = false;

                    switch(dim)
                    {
                        case 2:  // Degenerate PP (point-point)
                        {
                            IndexT V0 = vIs(offsets(0));  // Surface vertex V
                            IndexT V1 = vIs(offsets(1));  // Vertex of the face (t0/t1/t2)
                            // OGC: Verify if V0 is within the offset block of V1 (reuse phase1 logic)
                            is_ogc_valid = checkVertexFeasibleRegion(
                                positions(V0), V1, positions, vertex_offsets, vertex_indices, ogc_r);
                            if(is_ogc_valid)
                            {
                                PP = {V0, V1};
                                // Add after "distance::point_triangle_distance2(flag, Ps[0], Ps[1], Ps[2], Ps[3], D);":
                                Float d_update = sqrtf(D);
                                // Atomic update: Minimum distance from surface vertex V to the face
                                //atomic_min(&d_min_v(V), d_update);
                                atomicMinFloat(&d_min_v(V), d_update);
                                // Atomic update: Minimum distance from face t_idx to the vertex
                                int t_idx = indices(1);  // The index of the face in surf_triangles is the face ID
                                // atomic_min(&d_min_t(t_idx), d_update);
                                atomicMinFloat(&d_min_t(t_idx), d_update);
                            }
                        }
                        break;
                        case 3:  // Degenerate PE (point-edge)
                        {
                            IndexT   V0 = vIs(offsets(0));  // Surface vertex V
                            IndexT   V1 = vIs(offsets(1));  // Edge vertex 1
                            IndexT   V2 = vIs(offsets(2));  // Edge vertex 2
                            Vector2i e_vs = {V1, V2};
                            int      v0 = e_vs[0], v1 = e_vs[1];
                            // OGC: Find edge ID (reuse find_edge_id from phase1)
                            int edge_id =
                                find_edge_id(v0, v1, v_edge_offsets, v_edge_indices, Es);
                            if(edge_id == -1)
                                break;
                            // OGC: Verify if V0 is within the edge's offset block (reuse phase1 logic)
                            is_ogc_valid = checkEdgeFeasibleRegion(positions(V0),
                                                                   positions(v0),
                                                                   positions(v1),
                                                                   edge_id,
                                                                   positions,
                                                                   edge_face_counts,
                                                                   edge_face_vertices,
                                                                   ogc_r);
                            if(is_ogc_valid)
                            {
                                PE = {V0, V1, V2};
                                // Add after "distance::point_triangle_distance2(flag, Ps[0], Ps[1], Ps[2], Ps[3], D);":
                                Float d_update = sqrtf(D);
                                // Atomic update: Minimum distance from surface vertex V to the face
                                //atomic_min(&d_min_v(V), d_update);
                                atomicMinFloat(&d_min_v(V), d_update);
                                // Atomic update: Minimum distance from face t_idx to the vertex
                                int t_idx = indices(1);  // The index of the face in surf_triangles is the face ID
                                // atomic_min(&d_min_t(t_idx), d_update);
                                atomicMinFloat(&d_min_t(t_idx), d_update);
                            }
                        }
                        break;
                        case 4:  // Non-degenerate PT (point-face)
                        {
                            // OGC: No additional verification needed for face interior contact (reuse phase1 logic)
                            is_ogc_valid = true;
                            if(is_ogc_valid)
                            {
                                PT = vIs;
                                // Add after "distance::point_triangle_distance2(flag, Ps[0], Ps[1], Ps[2], Ps[3], D);":
                                Float d_update = sqrtf(D);
                                // Atomic update: Minimum distance from surface vertex V to the face
                                //atomic_min(&d_min_v(V), d_update);
                                atomicMinFloat(&d_min_v(V), d_update);
                                // Atomic update: Minimum distance from face t_idx to the vertex
                                int t_idx = indices(1);  // The index of the face in surf_triangles is the face ID
                                // atomic_min(&d_min_t(t_idx), d_update);
                                atomicMinFloat(&d_min_t(t_idx), d_update);
                            }
                        }
                        break;
                        default:
                            MUDA_ERROR_WITH_LOCATION("unexpected degenerate case dim=%d", dim);
                            break;
                    }
                    /////Original IPC logic/////
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
                 d_hats      = info.d_hats().viewer().name("d_hats"),
                 // Pass OGC-dependent parameters (consistent with phase2)
                 vertex_offsets = m_v_vertex_offsets.cviewer().name("vertex_offsets"),
                 vertex_indices = m_v_vertex_indices.cviewer().name("vertex_indices"),
                 ogc_r   = m_ogc_r,
                 d_min_v = m_d_min_v.viewer().name("d_min_v"),
                 d_min_t = m_d_min_t.viewer().name("d_min_t"),
                 d_min_e = m_d_min_e.viewer().name("d_min_e")] __device__(int i) mutable
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
                        return;  // Early return


                    // [Original IPC Logic]Softening judgment
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
                    else  // [Embed OGC logic from phase2]Degeneracy classification + offset block verification
                    {
                        Vector4i offsets;
                        auto dim = distance::degenerate_edge_edge(flag, offsets);
                        bool is_ogc_valid = false;

                        switch(dim)
                        {
                            case 2:  // Fully degenerate PP (point-point)
                            {
                                // Reuse phase2 logic: Get degenerate points from flag
                                int a_vertex = (offsets[0] == 0) ? E0(0) : E0(1);
                                int ep_vertex = (offsets[1] == 2) ? E1(0) : E1(1);
                                // OGC: Verify if the edge-edge closest point is within the offset block (reuse check function from phase2)
                                Vector3 x_c = edge_edge_closest_point(E0, E1, positions);
                                is_ogc_valid = checkVertexFeasibleRegionEdgeOffset(
                                    x_c, a_vertex, positions, vertex_offsets, vertex_indices, ogc_r);
                                if(is_ogc_valid)
                                {
                                    PP = {a_vertex, ep_vertex};
                                    // Add after "distance::edge_edge_distance2(flag, Ps[0], Ps[1], Ps[2], Ps[3], D);":
                                    Float d_update = sqrtf(D);
                                    // Atomic update: Minimum distance of edge e_idx
                                    int e_idx = indices(0);
                                    //atomic_min(&d_min_e(e_idx), d_update);
                                    atomicMinFloat(&d_min_e(e_idx), d_update);
                                    // Atomic update: Minimum distance of edge e_prime_idx
                                    int e_prime_idx = indices(1);
                                    //atomic_min(&d_min_e(e_prime_idx), d_update);
                                    atomicMinFloat(&d_min_e(e_prime_idx), d_update);
                                }
                            }
                            break;
                            case 3:  // Partially degenerate PE (point-edge)
                            {
                                // Reuse phase2 logic: Determine degenerate point and edge
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
                                // OGC: Verify if the closest point is within the offset block
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
                                    // Add after "distance::edge_edge_distance2(flag, Ps[0], Ps[1], Ps[2], Ps[3], D);":
                                    Float d_update = sqrtf(D);
                                    // Atomic update: Minimum distance of edge e_idx
                                    int e_idx = indices(0);
                                    //atomic_min(&d_min_e(e_idx), d_update);
                                    //// Atomic update: Minimum distance of edge e_prime_idx
                                    int e_prime_idx = indices(1);
                                    //atomic_min(&d_min_e(e_prime_idx), d_update);
                                    atomicMinFloat(&d_min_e(e_prime_idx), d_update);
                                }
                            }
                            break;
                            case 4:  // Non-degenerate EE (edge-edge)
                            {
                                // OGC: No additional verification needed for edge-edge contact (reuse phase2 logic)
                                is_ogc_valid = true;
                                if(is_ogc_valid)
                                {
                                    EE = vIs;
                                    // Add after "distance::edge_edge_distance2(flag, Ps[0], Ps[1], Ps[2], Ps[3], D);":
                                    Float d_update = sqrtf(D);
                                    // Atomic update: Minimum distance of edge e_idx
                                    int e_idx = indices(0);
                                    //atomic_min(&d_min_e(e_idx), d_update);
                                    atomicMinFloat(&d_min_e(e_idx), d_update);
                                    // Atomic update: Minimum distance of edge e_prime_idx
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

                    ////////////Original IPC logic////////////
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
                    //else  // Classify to EE/PE/PP
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
