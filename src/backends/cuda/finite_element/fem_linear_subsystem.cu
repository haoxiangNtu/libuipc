#include <finite_element/fem_linear_subsystem.h>
#include <sim_engine.h>
#include <kernel_cout.h>
#include <muda/ext/eigen.h>
#include <muda/ext/eigen/evd.h>
#include <muda/ext/eigen/atomic.h>
#include <finite_element/finite_element_constitution.h>
#include <finite_element/finite_element_extra_constitution.h>
#include <sim_engine.h>
#include <uipc/builtin/attribute_name.h>
#include <queue>
#include <random>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(FEMLinearSubsystem);

// 读取.t文件并构建邻接表
std::vector<std::vector<int>> readTetFileAndBuildAdjacency(const std::string& filename)
{
    // 数据结构定义
    std::unordered_map<int, int> external_to_internal;  // 外部索引→内部ID（0,1,2...）
    std::vector<int> internal_to_external;  // 内部ID→外部索引（可选，用于调试）
    std::vector<std::vector<int>> adjacency;  // 顶点邻接表（内部ID）
    int num_vertices = 0;                     // 顶点总数（内部ID从0开始）

    // 打开文件
    std::ifstream file(filename);
    if(!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filename);  // 中文替换为英文
    }

    std::string line;
    while(std::getline(file, line))
    {
        // 忽略空行
        if(line.empty())
            continue;

        // 解析行首标识（Vertex或Tet）
        std::istringstream iss(line);
        std::string        token;
        iss >> token;

        // 1. 解析顶点行（Vertex）
        if(token == "Vertex")
        {
            int   external_idx;
            float x, y, z;
            iss >> external_idx >> x >> y >> z;  // 提取外部索引和坐标（坐标着色时暂不用）

            // 若外部索引未映射，分配内部ID
            if(external_to_internal.find(external_idx) == external_to_internal.end())
            {
                external_to_internal[external_idx] = num_vertices;
                internal_to_external.push_back(external_idx);
                adjacency.emplace_back();  // 为新顶点创建空邻接列表
                num_vertices++;
            }
        }
        // 2. 解析四面体行（Tet）
        else if(token == "Tet")
        {
            int tet_idx, v0_ext, v1_ext, v2_ext, v3_ext;
            iss >> tet_idx >> v0_ext >> v1_ext >> v2_ext >> v3_ext;  // 提取四面体索引和4个顶点外部索引

            // 检查4个顶点是否已在映射表中（避免无效顶点）
            auto check_and_get = [&](int ext) -> int
            {
                if(external_to_internal.find(ext) == external_to_internal.end())
                {
                    // 中文替换为英文
                    throw std::runtime_error("Tetrahedron " + std::to_string(tet_idx) + " contains undefined vertex index: "
                                             + std::to_string(ext));
                }
                return external_to_internal[ext];
            };

            // 转换为内部ID
            int v0 = check_and_get(v0_ext);
            int v1 = check_and_get(v1_ext);
            int v2 = check_and_get(v2_ext);
            int v3 = check_and_get(v3_ext);

            // 四面体的4个顶点两两相邻，添加邻接关系（去重）
            std::vector<std::pair<int, int>> pairs = {
                {v0, v1}, {v0, v2}, {v0, v3}, {v1, v2}, {v1, v3}, {v2, v3}};
            for(auto [u, v] : pairs)
            {
                // 向u的邻接表添加v（去重）
                if(std::find(adjacency[u].begin(), adjacency[u].end(), v)
                   == adjacency[u].end())
                {
                    adjacency[u].push_back(v);
                }
                // 向v的邻接表添加u（去重）
                if(std::find(adjacency[v].begin(), adjacency[v].end(), u)
                   == adjacency[v].end())
                {
                    adjacency[v].push_back(u);
                }
            }
        }
        // 忽略其他无关行（如注释）
        else
        {
            continue;
        }
    }

    file.close();
    // 中文替换为英文
    std::cout << "File read completed: " << filename << std::endl;
    std::cout << "Total vertices: " << num_vertices
              << ", adjacency list built successfully" << std::endl;
    return adjacency;
}


class McsGraphColoring
{
  public:
    bool  balance_enabled_    = true;  // 默认不启用
    float goal_max_min_ratio_ = 1.05f;  // 默认目标比例
    // 构造函数：接收邻接表
    McsGraphColoring(const std::vector<std::vector<int>>& adjacency)
        : graph(adjacency)
        , num_vertices(adjacency.size())
    {
        graph_colors.resize(num_vertices, -1);  // 初始化颜色数组
        verbose = false;                       // 默认不打印进度
    }
    const std::vector<std::vector<int>>& get_categories() const
    {
        return categories;
    }

    // 执行着色，返回颜色数组
    std::vector<int>& color()
    {
        // 阶段1：MCS排序（生成顶点处理顺序）
        std::vector<int> temp_graph;
        for(size_t i = 0; i < graph.size(); ++i)
        {
            temp_graph.push_back(i);
        }

        std::vector<int> weight(num_vertices, 0);  // 顶点权重（初始为0）
        std::queue<int>  ordering;                 // 存储MCS排序结果

        // 随机打乱初始顺序（避免权重相同时的固定偏好）
        std::vector<int> coloringOrder(num_vertices);
        for(int i = 0; i < num_vertices; ++i)
        {
            coloringOrder[i] = i;
        }
        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(coloringOrder.begin(), coloringOrder.end(), std::default_random_engine(seed));

        int percentage = 0;
        for(int i = 0; i < num_vertices; ++i)
        {
            int max_weight = -1;
            int max_vertex = -1;
            int maxWId     = -1;

            // 从剩余顶点中找权重最大的顶点
            for(int j = 0; j < temp_graph.size(); ++j)
            {
                int nodeId = temp_graph[j];
                if(nodeId < 0)
                    continue;  // 已处理的顶点标记为-1
                if(weight[nodeId] > max_weight)
                {
                    max_weight = weight[nodeId];
                    max_vertex = nodeId;
                    maxWId     = j;
                }
            }

            if(max_vertex == -1)
            {
                std::cerr << "MCS Error: No max weight vertex found" << std::endl;
                graph_colors.clear();
                return graph_colors;
            }

            // 将最大权重顶点加入排序队列，并更新其邻居权重
            ordering.push(max_vertex);
            for(int neighbor : graph[max_vertex])
            {
                weight[neighbor] += 1;  // 邻居权重+1
            }

            // 标记该顶点为已处理
            temp_graph[maxWId] = -1;

            // 优化临时顶点列表（移除已处理的顶点）
            if(i % 100 == 0)
            {  // 每100步清理一次，避免频繁操作
                std::vector<int> temp_graph_new;
                for(int node : temp_graph)
                {
                    if(node >= 0)
                        temp_graph_new.push_back(node);
                }
                temp_graph = std::move(temp_graph_new);
            }
        }

        // 阶段2：按MCS排序进行贪心着色
        percentage = 0;
        int total  = ordering.size();
        while(!ordering.empty())
        {
            int current_vertex = ordering.front();
            ordering.pop();

            // 收集邻居已使用的颜色
            std::vector<int> used_colors;
            for(int neighbor : graph[current_vertex])
            {
                int neighbor_color = graph_colors[neighbor];
                if(neighbor_color != -1)
                {  // 邻居已着色
                    used_colors.push_back(neighbor_color);
                }
            }

            // 去重并排序
            std::sort(used_colors.begin(), used_colors.end());
            used_colors.erase(std::unique(used_colors.begin(), used_colors.end()),
                              used_colors.end());

            // 找到最小可用颜色
            int min_color = 0;
            for(int c : used_colors)
            {
                if(c == min_color)
                {
                    min_color++;
                }
                else if(c > min_color)
                {
                    break;
                }
            }

            graph_colors[current_vertex] = min_color;
        }
        // 新增：根据配置启用负载均衡
        if(balance_enabled_)
        {  // 可通过setter设置（如set_balance_enabled(true)）
            balanceColoredCategories(goal_max_min_ratio_);  // 目标比例（如1.2）
            // 验证负载均衡后的有效性
            if(!is_valid())
            {
                std::cerr << "Error: Invalid coloring after balancing!" << std::endl;
            }
        }
        return graph_colors;
    }

    public:
    // 负载均衡：调整颜色组，使最大/最小比例≤goalMaxMinRatio
    void balanceColoredCategories(float goalMaxMinRatio)
    {
        // 先初始化分组（必须在负载均衡前调用）
        convertToColoredCategories();

        float maxMinRatio = -1.0f;
        do
        {
            int biggestCategory  = -1;
            int smallestCategory = -1;
            maxMinRatio = findLargestSmallestCategories(biggestCategory, smallestCategory);

            // 尝试从最大组移动顶点到最小组
            int changableId = findChangableNodeInCategory(biggestCategory, smallestCategory);
            if(changableId == -1)
            {
                // 最大组无可用顶点，尝试从其他组移动
                for(size_t i = 0; i < categories.size(); ++i)
                {
                    if(i == biggestCategory || i == smallestCategory)
                    {
                        continue;
                    }
                    changableId = findChangableNodeInCategory(i, smallestCategory);
                    if(changableId != -1)
                    {
                        biggestCategory = i;  // 切换源组
                        break;
                    }
                }
            }

            if(changableId == -1)
            {
                // 无法继续优化
                std::cout << "Graph optimization stopped. Max/min ratio: " << maxMinRatio
                          << std::endl;
                return;
            }
            // 移动顶点到目标组
            changeColor(biggestCategory, changableId, smallestCategory);

        } while(maxMinRatio > goalMaxMinRatio);  // 直到满足目标比例

        std::cout << "Graph optimization completed. Max/min ratio: " << maxMinRatio
                  << std::endl;
    }

    // 控制是否打印进度信息
    void set_verbose(bool v) { verbose = v; }

  private:
    const std::vector<std::vector<int>>& graph;  // 邻接表（引用，不 ownership）
    std::vector<int>                     graph_colors;  // 颜色数组
    int                                  num_vertices;  // 顶点总数
    std::vector<std::vector<int>> categories;  // 新增：每个颜色的顶点列表（color → [顶点索引]）
    bool                                 verbose;       // 是否打印进度

  private:
    // 1. 计算颜色总数
    int get_num_colors() const
    {
        int numColors = 0;
        for(int color : graph_colors)
        {
            if(color + 1 > numColors)
            {
                numColors = color + 1;
            }
        }
        return numColors;
    }

    // 2. 验证着色有效性（相邻顶点颜色不同）
    bool is_valid() const
    {
        if(graph_colors.empty() || graph.size() != graph_colors.size())
        {
            return false;
        }
        for(size_t i = 0; i < graph.size(); ++i)
        {
            if(graph_colors[i] == -1)
            {  // 假设-1表示未着色
                return false;
            }
            for(int neighbor : graph[i])
            {
                if(graph_colors[i] == graph_colors[neighbor])
                {
                    return false;
                }
            }
        }
        return true;
    }

    // 3. 将颜色转换为分组（categories）
    void convertToColoredCategories()
    {
        categories.clear();
        int numColors = get_num_colors();
        categories.resize(numColors);
        for(size_t i = 0; i < graph.size(); ++i)
        {
            int color = graph_colors[i];
            categories[color].push_back(i);  // 按颜色分组顶点
        }
    }

    // 4. 找到最大/最小颜色组
    float findLargestSmallestCategories(int& biggestCategory, int& smallestCategory) const
    {
        if(categories.empty())
        {
            biggestCategory  = -1;
            smallestCategory = -1;
            return 1.0f;
        }
        size_t maxSize   = categories[0].size();
        biggestCategory  = 0;
        size_t minSize   = categories[0].size();
        smallestCategory = 0;

        for(size_t i = 0; i < categories.size(); ++i)
        {
            if(categories[i].size() > maxSize)
            {
                maxSize         = categories[i].size();
                biggestCategory = i;
            }
            if(categories[i].size() < minSize)
            {
                minSize          = categories[i].size();
                smallestCategory = i;
            }
        }
        return static_cast<float>(maxSize) / minSize;
    }

    // 5. 检查顶点是否可移动到目标颜色组（与目标颜色的顶点不相邻）
    bool changable(int node, int destinationColor) const
    {
        for(int neighbor : graph[node])
        {
            if(graph_colors[neighbor] == destinationColor)
            {
                return false;  // 邻居有目标颜色，不可移动
            }
        }
        return true;
    }

    // 6. 在源颜色组中找可移动到目标颜色组的顶点
    int findChangableNodeInCategory(int sourceColor, int destinationColor) const
    {
        for(size_t i = 0; i < categories[sourceColor].size(); ++i)
        {
            int node = categories[sourceColor][i];
            if(changable(node, destinationColor))
            {
                return i;  // 返回顶点在源组中的索引
            }
        }
        return -1;  // 无可用顶点
    }

    // 7. 移动顶点颜色（从源组到目标组）
    void changeColor(int sourceColor, int nodeIndexInSource, int destinationColor)
    {
        int nodeId           = categories[sourceColor][nodeIndexInSource];
        graph_colors[nodeId] = destinationColor;

        if(categories.size())
        {
            categories[sourceColor].erase(categories[sourceColor].begin() + nodeIndexInSource);
            categories[destinationColor].push_back(nodeId);
        }

        //int node = categories[sourceColor][nodeIndexInSource];
        //// 更新顶点颜色
        //graph_colors[node] = destinationColor;
        //// 更新分组：从源组移除，添加到目标组
        //categories[sourceColor].erase(categories[sourceColor].begin() + nodeIndexInSource);
        //categories[destinationColor].push_back(node);
    }
};

void FEMLinearSubsystem::do_build(DiagLinearSubsystem::BuildInfo&)
{
    m_impl.finite_element_method = require<FiniteElementMethod>();
    m_impl.finite_element_vertex_reporter = require<FiniteElementVertexReporter>();
    m_impl.sim_engine = &engine();
    auto dt_attr      = world().scene().config().find<Float>("dt");
    m_impl.dt         = dt_attr->view()[0];

    m_impl.dytopo_effect_receiver  = find<FEMDyTopoEffectReceiver>();
    m_impl.finite_element_animator = find<FiniteElementAnimator>();
    m_impl.converter.reserve_ratio(1.1);
}

void FEMLinearSubsystem::do_init(DiagLinearSubsystem::InitInfo& info) {}

void FEMLinearSubsystem::Impl::report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info)
{
    info.extent(fem().xs.size() * 3);
}

void FEMLinearSubsystem::Impl::receive_init_dof_info(WorldVisitor& w,
                                                     GlobalLinearSystem::InitDofInfo& info)
{
    auto& geo_infos = fem().geo_infos;
    auto  geo_slots = w.scene().geometries();

    IndexT offset = info.dof_offset();

    finite_element_method->for_each(
        geo_slots,
        [&](const FiniteElementMethod::ForEachInfo& foreach_info, geometry::SimplicialComplex& sc)
        {
            auto I          = foreach_info.global_index();
            auto dof_offset = sc.meta().find<IndexT>(builtin::dof_offset);
            UIPC_ASSERT(dof_offset, "dof_offset not found on FEM mesh why can it happen?");
            auto dof_count = sc.meta().find<IndexT>(builtin::dof_count);
            UIPC_ASSERT(dof_count, "dof_count not found on FEM mesh why can it happen?");

            IndexT this_dof_count = 3 * sc.vertices().size();
            view(*dof_offset)[0]  = offset;
            view(*dof_count)[0]   = this_dof_count;

            offset += this_dof_count;
        });

    UIPC_ASSERT(offset == info.dof_offset() + info.dof_count(), "dof size mismatch");
}

void FEMLinearSubsystem::Impl::report_extent(GlobalLinearSystem::DiagExtentInfo& info)
{
    UIPC_ASSERT(info.storage_type() == GlobalLinearSystem::HessianStorageType::Full,
                "Now only support Full Hessian");

    // 1) Hessian Count
    energy_producer_hessian_offset = 0;
    energy_producer_hessian_count  = fem().energy_producer_total_hessian_count;
    auto hessian_block_count       = energy_producer_hessian_count;

    if(dytopo_effect_receiver)  // if dytopo_effect enabled
    {
        dytopo_effect_hessian_offset = hessian_block_count;
        dytopo_effect_hessian_count = dytopo_effect_receiver->hessians().triplet_count();
        hessian_block_count += dytopo_effect_hessian_count;
    }

    if(finite_element_animator)
    {
        FiniteElementAnimator::ExtentInfo extent_info;
        finite_element_animator->report_extent(extent_info);
        animator_hessian_offset = hessian_block_count;
        animator_hessian_count  = extent_info.hessian_block_count;
        hessian_block_count += animator_hessian_count;
    }

    // 2) Gradient Count
    auto dof_count = fem().dxs.size() * 3;

    info.extent(hessian_block_count, dof_count);
}

    template <typename DType>
__forceinline__ __device__ __host__ bool solve3x3_psd_stable(const DType* m,
                                                             const DType* b,
                                                             DType*       out)
{
    const DType a11 = m[0];
    const DType a12 = m[3];
    const DType a13 = m[6];
    const DType a21 = m[1];
    const DType a22 = m[4];
    const DType a23 = m[7];
    const DType a31 = m[2];
    const DType a32 = m[5];
    const DType a33 = m[8];

    const DType i11 = a33 * a22 - a32 * a23;
    const DType i12 = -(a33 * a12 - a32 * a13);
    const DType i13 = a23 * a12 - a22 * a13;

    const DType det = (a11 * i11 + a21 * i12 + a31 * i13);

    if(abs(det) < 1e-5 * (abs(a11 * i11) + abs(a21 * i12) + abs(a31 * i13)))
    {
        out[0] = b[0];
        out[1] = b[1];
        out[2] = b[2];
        return false;
    }


    const DType deti = 1.0 / det;

    const DType i21 = -(a33 * a21 - a31 * a23);
    const DType i22 = a33 * a11 - a31 * a13;
    const DType i23 = -(a23 * a11 - a21 * a13);

    const DType i31 = a32 * a21 - a31 * a22;
    const DType i32 = -(a32 * a11 - a31 * a12);
    const DType i33 = a22 * a11 - a21 * a12;

    out[0] = deti * (i11 * b[0] + i12 * b[1] + i13 * b[2]);
    out[1] = deti * (i21 * b[0] + i22 * b[1] + i23 * b[2]);
    out[2] = deti * (i31 * b[0] + i32 * b[1] + i33 * b[2]);
    return true;
}

// 从 CPU 端全局梯度向量中提取顶点 vertexId 的局部梯度（3 个分量）
Vector3 get_vertex_gradient(const std::vector<Float>& gradients_h, int vertexId)
{
    // 计算起始索引（每个顶点占 3 个元素）
    size_t start = static_cast<size_t>(vertexId) * 3;
    // 检查索引是否越界（避免访问错误）
    if(start + 2 >= gradients_h.size())
    {
        throw std::out_of_range("vertexId out of range in gradients_h");
    }
    // 提取 x/y/z 分量
    return Vector3(gradients_h[start],      // x 分量
                gradients_h[start + 1],  // y 分量
                gradients_h[start + 2]   // z 分量
    );
}

void FEMLinearSubsystem::Impl::assemble(GlobalLinearSystem::DiagInfo& info)
{
    // 0) record dof info
    auto frame = sim_engine->frame();
    fem().set_dof_info(frame, info.gradients().offset(), info.gradients().size());

    // 1) Clear Gradient
    info.gradients().buffer_view().fill(0);

    // 2) Assemble Gradient and Hessian
    _assemble_producers(info);
    _assemble_dytopo_effect(info);
    _assemble_animation(info);

    using namespace muda;

    // 3) Clear Fixed Vertex Gradient
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(fem().xs.size(),
               [is_fixed = fem().is_fixed.cviewer().name("is_fixed"),
                gradients = info.gradients().viewer().name("gradients")] __device__(int i) mutable
               {
                   if(is_fixed(i))
                   {
                       gradients.segment<3>(i * 3).as_eigen().setZero();
                   }
               });

    // 4) Clear Fixed Vertex hessian
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(info.hessians().triplet_count(),
               [is_fixed = fem().is_fixed.cviewer().name("is_fixed"),
                hessians = info.hessians().viewer().name("hessians")] __device__(int I) mutable
               {
                   auto&& [i, j, H3] = hessians(I).read();

                   if(is_fixed(i) || is_fixed(j))
                   {
                       if(i != j)
                           hessians(I).write(i, j, Matrix3x3::Zero());
                   }
               })
        .wait();

    ///////////###############################TEST CONTACT,  where is external force??????
}

void FEMLinearSubsystem::Impl::vertices_Coloring()
{
    Timer timer{"Vertex coloring (supports internal fem data or external file)"};

    std::vector<std::vector<int>> adjacency;  // 邻接表（统一存储，无论数据源）
    int num_vertices = 0;                     // 顶点总数（根据数据源动态确定）
    bool use_external_file = false;
    // 分支1：使用外部文件（.t格式四面体网格）
    std::string external_filename = 
        "D:/Programming/Gaia/Simulator/GraphColoring/build/bar990.t";
    if(use_external_file)
    {
        std::cout << "Using external file: " << external_filename
                  << " to build adjacency" << std::endl;
        try
        {
            // 调用外部文件读取函数（复用之前实现的readTetFileAndBuildAdjacency）
            adjacency = readTetFileAndBuildAdjacency(external_filename);
            num_vertices = adjacency.size();  // 外部文件的顶点总数由邻接表大小确定
            if(num_vertices == 0)
            {
                std::cerr << "External file contains no vertices." << std::endl;
                return;
            }
        }
        catch(const std::exception& e)
        {
            std::cerr << "Failed to read external file: " << e.what() << std::endl;
            return;
        }
    }
    // 分支2：使用内部fem()数据
    else
    {
        std::cout << "Using internal fem data to build adjacency" << std::endl;
        const auto& fem_data = fem();
        num_vertices         = fem_data.xs.size();  // 从内部数据获取顶点总数
        if(num_vertices == 0)
        {
            std::cout << "No vertices in internal fem data." << std::endl;
            return;
        }

        adjacency.resize(num_vertices);  // 初始化邻接表

        // 2.1 处理内部四面体网格
        if(fem_data.dim_infos[3].primitive_count > 0)
        {
            std::cout << "Building adjacency from internal tetrahedrons..." << std::endl;
            std::vector<Vector4i> tets_host;
            tets_host.resize(fem_data.tets.size());
            fem_data.tets.view().copy_to(tets_host.data());

            for(const auto& tet : tets_host)
            {
                int v0 = tet[0], v1 = tet[1], v2 = tet[2], v3 = tet[3];
                std::vector<std::pair<int, int>> pairs = {
                    {v0, v1}, {v0, v2}, {v0, v3}, {v1, v2}, {v1, v3}, {v2, v3}};
                for(auto [u, v] : pairs)
                {
                    if(u < 0 || u >= num_vertices || v < 0 || v >= num_vertices)
                        continue;
                    if(std::find(adjacency[u].begin(), adjacency[u].end(), v)
                       == adjacency[u].end())
                        adjacency[u].push_back(v);
                    if(std::find(adjacency[v].begin(), adjacency[v].end(), u)
                       == adjacency[v].end())
                        adjacency[v].push_back(u);
                }
            }
        }
        // 2.2 处理内部表面网格（三角形）
        else if(fem_data.dim_infos[2].primitive_count > 0)
        {
            std::cout << "Building adjacency from internal surface triangles..."
                      << std::endl;
            std::vector<Vector3i> tris_host;
            tris_host.resize(fem_data.codim_2ds.size());
            fem_data.codim_2ds.view().copy_to(tris_host.data());

            for(const auto& tri : tris_host)
            {
                int v0 = tri[0], v1 = tri[1], v2 = tri[2];
                std::vector<std::pair<int, int>> pairs = {{v0, v1}, {v0, v2}, {v1, v2}};
                for(auto [u, v] : pairs)
                {
                    if(u < 0 || u >= num_vertices || v < 0 || v >= num_vertices)
                        continue;
                    if(std::find(adjacency[u].begin(), adjacency[u].end(), v)
                       == adjacency[u].end())
                        adjacency[u].push_back(v);
                    if(std::find(adjacency[v].begin(), adjacency[v].end(), u)
                       == adjacency[v].end())
                        adjacency[v].push_back(u);
                }
            }
        }
        else
        {
            std::cerr << "No valid internal mesh data (tets or codim_2ds) for adjacency building."
                      << std::endl;
            return;
        }
    }

    // 3. 调用自定义MCS算法进行着色
    uipc::backend::cuda::McsGraphColoring mcs(adjacency);
    mcs.set_verbose(true);       // 打印进度（可选）
    auto& colors = mcs.color();  // 执行着色

    // 4. 验证着色结果（可选，检查相邻顶点颜色是否冲突）
    bool valid = true;
    for(int v = 0; v < num_vertices; ++v)
    {
        for(int neighbor : adjacency[v])
        {
            if(colors[v] == colors[neighbor])
            {
                std::cerr << "Color conflict: vertex " << v << " and neighbor " << neighbor
                          << " have the same color " << colors[v] << std::endl;
                valid = false;
                break;
            }
        }
        if(!valid)
            break;
    }
    if(valid)
    {
        std::cout << "Coloring is valid." << std::endl;
    }

    // 替换你的打印逻辑
    std::cout << "Coloring completed. " << std::endl;
    std::cout << "Total vertices: " << num_vertices << std::endl;
    std::cout << "Total colors used: " << mcs.get_categories().size() << std::endl;  // 使用categories的大小

    std::cout << "Vertex distribution per color group (with indices):" << std::endl;
    const auto& categories = mcs.get_categories();  // 获取负载均衡中维护的categories
    for(size_t c = 0; c < categories.size(); ++c)
    {
        const auto& vertices = categories[c];
        std::cout << "  Color " << c << ": " << vertices.size() << " vertices -> [";
        for(size_t i = 0; i < vertices.size(); ++i)
        {
            std::cout << vertices[i];
            if(i != vertices.size() - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    vertex_group  = std::move(categories);
    //// 【关键补充】将着色结果移动到成员变量vertex_colors中

    //vertex_colors = std::move(colors);  // 必须添加这一行！

    //// 5. 统计每个颜色组的顶点数量和具体索引，并输出详细信息
    //int max_color = *std::max_element(vertex_colors.begin(), vertex_colors.end());
    //int total_colors = max_color + 1;

    //// 存储每个颜色对应的顶点索引（类似GraphColor的categories）
    //std::vector<std::vector<int>> color_vertices(total_colors);
    //for(int v = 0; v < num_vertices; ++v)
    //{
    //    int color = vertex_colors[v];
    //    color_vertices[color].push_back(v);  // 将顶点v加入其颜色对应的组
    //}

    //// 输出总统计信息
    //std::cout << "Coloring completed. " << std::endl;
    //std::cout << "Total vertices: " << num_vertices << std::endl;
    //std::cout << "Total colors used: " << total_colors << std::endl;

    //// 输出每个颜色组的顶点索引和数量
    //std::cout << "Vertex distribution per color group (with indices):" << std::endl;
    //for(int c = 0; c < total_colors; ++c)
    //{
    //    const auto& vertices = color_vertices[c];  // 当前颜色的顶点索引列表
    //    std::cout << "  Color " << c << ": " << vertices.size() << " vertices -> [";
    //    // 遍历打印顶点索引（用逗号分隔）
    //    for(size_t i = 0; i < vertices.size(); ++i)
    //    {
    //        std::cout << vertices[i];
    //        if(i != vertices.size() - 1)
    //        {
    //            std::cout << ", ";
    //        }
    //    }
    //    std::cout << "]" << std::endl;
    //}

    // （可选）验证总顶点数是否匹配（用于调试）
    //int sum = std::accumulate(color_vertices.begin(),
    //                          color_vertices.end(),
    //                          0,
    //                          [](int total, const std::vector<int>& group)
    //                          { return total + group.size(); });
    //if(sum != num_vertices)
    //{
    //    std::cerr << "Warning: Vertex count mismatch! Sum of groups: " << sum
    //              << ", Total vertices: " << num_vertices << std::endl;
    //}

    int stophere=1;
}

void FEMLinearSubsystem::Impl::solve_system_vertex(GlobalLinearSystem::DiagInfo& info)
{
    //vertices_Coloring();
    // 总函数时间计时
    Timer totalTimer{"Total solve_system_vertex time"};

    using namespace muda;
    // 变量声明
    auto                 N           = fem().xs.size();
    IndexT               vertex_size = fem().xs.size();
    auto                 info_x_size = info.x_update().size();
    auto                 xs_size     = fem().xs.size();
    std::vector<Float>   x_update_h_3v;
    std::vector<Vector3> x_update_h_global;
    std::vector<Vector3> xs_previous;

    // 1. 初始化向量（内存分配+数据拷贝）
    {
        Timer timer{"Initialize vectors (x_update_h_3v, x_update_h_global, xs_previous)"};
        x_update_h_3v.resize(info_x_size);
        x_update_h_global.resize(xs_size, Vector3::Zero());
        xs_previous.resize(xs_size);
        fem().xs.copy_to(xs_previous);
    }

    // 顶点循环整体计时
    {
        Timer vertexLoopTimer{"Total vertex loop time (all vertices)"};

        for(size_t iGroup = 0; iGroup < vertex_group.size(); iGroup++)
        {
            auto & parallelGroup = vertex_group[iGroup];

            // 我们这里没有分为mesh id 和 vertex id, 直接用vertex id
            
            //size_t numVertices = parallelGroup.size() / 2;
            //cpu_parallel_for(0,
            //                 numVertices,
            //                 [&](int iV)
            //                 {
            //                     IdType iMesh = parallelGroup[iV * 2];
            //                     int    vId   = parallelGroup[2 * iV + 1];

            //                     VBDTetMeshNeoHookean* pMesh =
            //                         (VBDTetMeshNeoHookean*)tMeshes[iMesh].get();
            //                     if(!pMesh->fixedMask[vId]
            //                        && !pMesh->activeCollisionMask[vId]
            //                        && pMesh->activeForMaterialSolve)
            //                     //if (!pMesh->fixedMask[vId])
            //                     {
            //                         //pMesh->VBDStep(vId);
            //                         VBDStepWithCollision(pMesh, iMesh, vId, apply_friction);
            //                     }
            //                 });
        }

        for(int vertexId = 0; vertexId < vertex_size; ++vertexId)
        {
            // 2. 清除梯度
            {
                Timer timer{"Clear Gradient (all vertices)"};
                info.gradients().buffer_view().fill(0);
            }

            // 3. 组装梯度和海森矩阵（分步骤）
            {
                Timer timer{"Assemble producers (all vertices)"};
                _assemble_producers(info);
            }
            {
                Timer timer{"Assemble dytopo effect (all vertices)"};
                _assemble_dytopo_effect(info);
            }
            {
                Timer timer{"Assemble animation (all vertices)"};
                _assemble_animation(info);
            }

            // 4. 清除固定顶点梯度（并行操作）
            {
                Timer timer{"Clear Fixed Vertex Gradient (ParallelFor, all vertices)"};
                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply(fem().xs.size(),
                           [is_fixed = fem().is_fixed.cviewer().name("is_fixed"),
                            gradients = info.gradients().viewer().name(
                                "gradients")] __device__(int i) mutable
                           {
                               if(is_fixed(i))
                               {
                                   gradients.segment<3>(i * 3).as_eigen().setZero();
                               }
                           });
            }

            // 5. 清除固定顶点海森矩阵（并行操作）
            {
                Timer timer{"Clear Fixed Vertex hessian (ParallelFor, all vertices)"};
                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply(info.hessians().triplet_count(),
                           [is_fixed = fem().is_fixed.cviewer().name("is_fixed"),
                            hessians = info.hessians().viewer().name("hessians")] __device__(int I) mutable
                           {
                               auto&& [i, j, H3] = hessians(I).read();

                               if(is_fixed(i) || is_fixed(j))
                               {
                                   if(i != j)
                                       hessians(I).write(i, j, Matrix3x3::Zero());
                               }
                           })
                    .wait();
            }

            // 6. 提取梯度数据到CPU并计算力
            std::vector<Vector3> force_h;
            std::vector<Float>   gradients_h;
            int                  gradients_size;
            {
                Timer timer{"Extract gradients to CPU and compute force (all vertices)"};
                force_h.resize(vertex_size);
                gradients_size = info.gradients().size();
                gradients_h.resize(gradients_size);
                info.gradients().buffer_view().copy_to(gradients_h.data());

                // 提取当前顶点梯度并计算力
                Vector3 gradient  = get_vertex_gradient(gradients_h, vertexId);
                force_h[vertexId] = -gradient;
            }

            // 7. 提取海森矩阵三元组到CPU
            IndexT                 triplet_size;
            std::vector<IndexT>    host_rows;
            std::vector<IndexT>    host_cols;
            std::vector<Matrix3x3> host_values;
            int                    hessian_size;
            {
                Timer timer{"Extract Hessian triplets to CPU (all vertices)"};
                triplet_size = info.hessians().row_indices().size();
                host_rows.resize(triplet_size);
                host_cols.resize(triplet_size);
                host_values.resize(triplet_size);
                info.hessians().row_indices().copy_to(host_rows.data());
                info.hessians().col_indices().copy_to(host_cols.data());
                info.hessians().values().copy_to(host_values.data());
                hessian_size = info.hessians().total_triplet_count();
            }

            // 8. 遍历三元组获取当前顶点海森矩阵
            Matrix3x3 h = Matrix3x3::Zero();
            {
                Timer timer{"Loop through triplets to get hessian h (all vertices)"};
                for(int I = 0; I < triplet_size; ++I)
                {
                    int i_vertex = host_rows[I];
                    int j_vertex = host_cols[I];
                    if(i_vertex == vertexId && j_vertex == vertexId)
                    {
                        h += host_values[I];
                    }
                }
            }

            // 9. 求解3x3矩阵及相关处理
            {
                Timer  timer{"Solve 3x3 PSD system (all vertices)"};
                auto&  force     = force_h[vertexId];
                double ForceNorm = force.squaredNorm();

                if(1)  // 保留原条件
                {
                    if(force.isZero())
                    {
                        continue;
                    }
                    if(h.isZero())
                    {
                        h = Matrix3x3::Identity() * 1e-6;
                        continue;
                    }

                    Vector3 descentDirection;
                    Float   stepSize               = 1;
                    Float   lineSearchShrinkFactor = 0.8;
                    bool    solverSuccess;

                    bool useDouble3x3 = 1;
                    if(useDouble3x3)
                    {
                        double H[9] = {h(0, 0),
                                       h(1, 0),
                                       h(2, 0),
                                       h(0, 1),
                                       h(1, 1),
                                       h(2, 1),
                                       h(0, 2),
                                       h(1, 2),
                                       h(2, 2)};

                        double F[3]      = {force(0), force(1), force(2)};
                        double dx[3]     = {0, 0, 0};
                        solverSuccess    = solve3x3_psd_stable(H, F, dx);
                        descentDirection = Vector3(dx[0], dx[1], dx[2]);

                        // 验证计算
                        auto   TestOuput = h * descentDirection;
                        auto   diff      = TestOuput - force;
                        double diff_norm = diff.norm();
                        if(diff_norm > 1e-6)
                        {
                            std::cout << "Warning: h * descentDirection does not match force (diff_norm = "
                                      << diff_norm << ")" << std::endl;
                        }
                    }
                    else
                    {
                        solverSuccess = false;  // 未使用分支
                    }

                    // 处理求解失败
                    if(!solverSuccess)
                    {
                        stepSize               = 1;
                        descentDirection       = force;
                        lineSearchShrinkFactor = 0.8;
                        std::cout << "Solver failed at vertex " << vertexId << std::endl;
                    }

                    // 检查数值异常
                    if(descentDirection.hasNaN())
                    {
                        std::cout << "force: " << force.transpose() << "\nHessian:\n"
                                  << h;
                        std::cout << "descentDirection has NaN at vertex "
                                  << vertexId << std::endl;
                        std::exit(-1);
                    }

                    // 10. 更新x_update数组
                    {
                        Timer timer{"Update x_update arrays (all vertices)"};
                        for(int k = 0; k < 3; ++k)
                        {
                            x_update_h_3v[vertexId * 3 + k] -= descentDirection[k];
                        }
                        x_update_h_global[vertexId] += descentDirection;
                    }

                    // 11. 更新顶点位置
                    {
                        Timer timer{"Update xs_temp and copy to fem().xs (all vertices)"};
                        std::vector<Vector3> xs_temp(xs_previous.size());
                        for(size_t i = 0; i < xs_previous.size(); ++i)
                        {
                            xs_temp[i] = xs_previous[i] + x_update_h_global[i];
                        }
                        fem().xs.copy_from(xs_temp);
                    }
                }
            }
        }
    }

    // 12. 同步回GPU
    {
        Timer timer{"Copy x_update_h_3v to GPU (info.x_update())"};
        info.x_update().buffer_view().copy_from(x_update_h_3v.data());
    }

    std::cout << "###########################################################" << std::endl;
}

void FEMLinearSubsystem::Impl::_assemble_producers(GlobalLinearSystem::DiagInfo& info)
{
    FiniteElementEnergyProducer::AssemblyInfo assembly_info;
    assembly_info.hessians = info.hessians().subview(energy_producer_hessian_offset,
                                                     energy_producer_hessian_count);
    assembly_info.dt = dt;

    for(auto& producer : fem().energy_producers)
    {
        producer->assemble_gradient_hessian(assembly_info);
    }

    using namespace muda;

    // need to assemble doublet gradient to dense gradient
    const auto& producer_gradients = fem().energy_producer_gradients;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(producer_gradients.doublet_count(),
               [dst_gradient = info.gradients().viewer().name("dst_gradient"),
                src_gradient = producer_gradients.viewer().name("src_gradient")] __device__(int I) mutable
               {
                   auto&& [i, G3] = src_gradient(I);  // i是当前梯度所属的顶点索引
                   dst_gradient.segment<3>(i * 3).atomic_add(G3);
               });
}

void FEMLinearSubsystem::Impl::_assemble_dytopo_effect(GlobalLinearSystem::DiagInfo& info)
{
    using namespace muda;

    if(dytopo_effect_receiver)  //  if dytopo_effect enabled
    {
        auto dytopo_effect_gradient_count =
            dytopo_effect_receiver->gradients().doublet_count();

        // 1) Assemble DyTopoEffect Gradient to Gradient
        if(dytopo_effect_gradient_count)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(dytopo_effect_gradient_count,
                       [dytopo_effect_gradient =
                            dytopo_effect_receiver->gradients().cviewer().name("dytopo_effect_gradient"),
                        gradients = info.gradients().viewer().name("gradients"),
                        vertex_offset = finite_element_vertex_reporter->vertex_offset(),
                        is_fixed = fem().is_fixed.cviewer().name("is_fixed")] __device__(int I) mutable
                       {
                           const auto& [g_i, G3] = dytopo_effect_gradient(I);
                           auto i = g_i - vertex_offset;  // from global to local
                           gradients.segment<3>(i * 3).atomic_add(G3);
                       });
        }

        // 2) Assemble DyTopoEffect Hessian to Hessian
        if(dytopo_effect_hessian_count)
        {
            auto dst_H3x3s = info.hessians().subview(dytopo_effect_hessian_offset,
                                                     dytopo_effect_hessian_count);

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(dytopo_effect_hessian_count,
                       [dytopo_effect_hessian =
                            dytopo_effect_receiver->hessians().cviewer().name("dytopo_effect_hessian"),
                        hessians = dst_H3x3s.viewer().name("hessians"),
                        vertex_offset =
                            finite_element_vertex_reporter->vertex_offset()] __device__(int I) mutable
                       {
                           const auto& [g_i, g_j, H3] = dytopo_effect_hessian(I);
                           auto i = g_i - vertex_offset;
                           auto j = g_j - vertex_offset;

                           hessians(I).write(i, j, H3);
                       });
        }
    }
}

void FEMLinearSubsystem::Impl::_assemble_animation(GlobalLinearSystem::DiagInfo& info)
{
    using namespace muda;
    if(finite_element_animator)
    {
        auto hessians = info.hessians().subview(animator_hessian_offset, animator_hessian_count);
        FiniteElementAnimator::AssembleInfo this_info{info.gradients(), hessians, dt};
        finite_element_animator->assemble(this_info);
    }
}

void FEMLinearSubsystem::Impl::accuracy_check(GlobalLinearSystem::AccuracyInfo& info)
{
    info.statisfied(true);
}

void FEMLinearSubsystem::Impl::retrieve_solution(GlobalLinearSystem::SolutionInfo& info)
{
    using namespace muda;

    auto dxs = fem().dxs.view();
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(fem().xs.size(),
               [dxs = dxs.viewer().name("dxs"),
                result = info.solution().viewer().name("result")] __device__(int i) mutable
               {
                   dxs(i) = -result.segment<3>(i * 3).as_eigen();

                   // cout << "solution dx(" << i << "):" << dxs(i).transpose().eval() << "\n";
               });
    // This is retrive solution section
}

void FEMLinearSubsystem::do_report_extent(GlobalLinearSystem::DiagExtentInfo& info)
{
    m_impl.report_extent(info);
}

void FEMLinearSubsystem::do_assemble(GlobalLinearSystem::DiagInfo& info)
{
    m_impl.assemble(info);
}

void FEMLinearSubsystem::do_solve_system_vertex(GlobalLinearSystem::DiagInfo& info)
{
    m_impl.solve_system_vertex(info);
}

void FEMLinearSubsystem::do_accuracy_check(GlobalLinearSystem::AccuracyInfo& info)
{
    m_impl.accuracy_check(info);
}

void FEMLinearSubsystem::do_retrieve_solution(GlobalLinearSystem::SolutionInfo& info)
{
    m_impl.retrieve_solution(info);
}

void FEMLinearSubsystem::do_report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info)
{
    m_impl.report_init_extent(info);
}

void FEMLinearSubsystem::do_receive_init_dof_info(GlobalLinearSystem::InitDofInfo& info)
{
    m_impl.receive_init_dof_info(world(), info);
}

}  // namespace uipc::backend::cuda
