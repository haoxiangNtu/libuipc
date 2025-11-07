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

// 对比单个顶点的对角Hessian矩阵
void compare_hessian_hii(IndexT vertexId, const Matrix3x3& cpu_h, const Matrix3x3& gpu_h)
{
    const Float eps   = 1e-6f;  // 浮点误差容忍度
    bool        match = true;
    for(int r = 0; r < 3; ++r)
    {
        for(int c = 0; c < 3; ++c)
        {
            Float diff = std::abs(cpu_h(r, c) - gpu_h(r, c));
            if(diff > eps)
            {
                match = false;
                std::cout << "Hii mismatch at vertex " << vertexId << ", (" << r
                          << "," << c << "): "
                          << "CPU=" << cpu_h(r, c) << ", GPU=" << gpu_h(r, c)
                          << ", diff=" << diff << std::endl;
            }
        }
    }
    if(match)
    {
        std::cout << "Vertex " << vertexId << " Hii matches (CPU vs GPU)." << std::endl;
    }
    else
    {
        std::cout << "+++++++++++++++++++++Vertex " << vertexId
                  << " Hii DO NOT match!" << std::endl;
    }
}

// 读取.t文件并构建邻接表
std::vector<std::vector<int>> readTetFileAndBuildAdjacency(const std::string& filename)
{
    // 数据结构定义
    std::unordered_map<int, int> external_to_internal;  // 外部索引 to 内部ID（0,1,2...）
    std::vector<int> internal_to_external;  // 内部ID to 外部索引（可选，用于调试）
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
    std::vector<std::vector<int>> categories;  // 新增：每个颜色的顶点列表（color  to  [顶点索引]）
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
    //std::cout << "Total colors used: " << mcs.get_categories().size() << std::endl;  // 使用categories的大小

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

    vertex_colors = std::move(colors);  // 必须添加这一行！

    // 5. 统计每个颜色组的顶点数量和具体索引，并输出详细信息
    int max_color = *std::max_element(vertex_colors.begin(), vertex_colors.end());
    int total_colors = max_color + 1;

    // 存储每个颜色对应的顶点索引（类似GraphColor的categories）
    std::vector<std::vector<int>> color_vertices(total_colors);
    for(int v = 0; v < num_vertices; ++v)
    {
        int color = vertex_colors[v];
        color_vertices[color].push_back(v);  // 将顶点v加入其颜色对应的组
    }

    // 输出总统计信息
    std::cout << "Coloring completed. " << std::endl;
    std::cout << "Total vertices: " << num_vertices << std::endl;
    std::cout << "Total colors used: " << total_colors << std::endl;

    // 输出每个颜色组的顶点索引和数量
    std::cout << "Vertex distribution per color group (with indices):" << std::endl;
    for(int c = 0; c < total_colors; ++c)
    {
        const auto& vertices = color_vertices[c];  // 当前颜色的顶点索引列表
        std::cout << "  Color " << c << ": " << vertices.size() << " vertices -> [";
        // 遍历打印顶点索引（用逗号分隔）
        for(size_t i = 0; i < vertices.size(); ++i)
        {
            std::cout << vertices[i];
            if(i != vertices.size() - 1)
            {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }

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
}

void FEMLinearSubsystem::Impl::solve_system_vertex_test(GlobalLinearSystem::DiagInfo& info)
{
    //if(vertex_group.size()==0)
    //{
    //    vertices_Coloring();
    //}
    //else {
    //    int jumpOver = 0;
    //}
    vertices_Coloring();
    // 总函数时间计时
    Timer totalTimer{"Total solve_system_vertex time"};

    using namespace muda;
    // 变量声明
    auto                 N           = fem().xs.size();
    IndexT               vertex_size = fem().xs.size();
    auto                 info_x_size = info.x_update().size();
    auto                 xs_size     = fem().xs.size();
    std::vector<Float>   x_update_h_3v;
    std::vector<Float>   xs_update_3v;
    std::vector<Vector3> x_update_h_global;
    std::vector<Vector3> xs_previous;
    muda::DeviceBuffer<Vector3> xs_previous_gpu;
    std::vector<Vector3> xs_initial;
    std::vector<Vector3> xs_temp_global;
    // 1. 初始化向量（内存分配+数据拷贝）
    {
        Timer timer{"Initialize vectors (x_update_h_3v, x_update_h_global, xs_previous)"};
        x_update_h_3v.resize(info_x_size);
        xs_update_3v.resize(info_x_size);
        x_update_h_global.resize(xs_size, Vector3::Zero());
        xs_previous.resize(xs_size);
        fem().xs.copy_to(xs_previous);
        xs_previous_gpu = fem().xs;  // 直接拷贝到GPU
        xs_initial.resize(xs_size);
        fem().xs.copy_to(xs_initial);
    }

    // 顶点循环整体计时
    {
        Timer vertexLoopTimer{"Total vertex loop time (all vertices)"};

        ////////////////////////
        //// Host 侧打印该颜色组的顶点索引（注意：大量输出会很慢，可限制条目）
        //{
        //    const size_t max_print = 64;  // 避免输出过多
        //    std::cout << "[Color " << iGroup
        //              << "] size=" << group_indices.size() << " -> [";
        //    for(size_t i = 0; i < group_indices.size() && i < max_print; ++i)
        //    {
        //        std::cout << group_indices[i];
        //        if(i + 1 < group_indices.size() && i + 1 < max_print)
        //            std::cout << ", ";
        //    }
        //    if(group_indices.size() > max_print)
        //        std::cout << ", ...";
        //    std::cout << "]" << std::endl;
        //}
        ///////////////////////
        // 
        // 组外：一次性累积所有组的位移增量（GPU端）
        {
            //static thread_local bool                        inited = false;
            //static thread_local muda::DeviceBuffer<Vector3> d_x_update_global;
            //if(!inited)
            //{
            //    d_x_update_global.resize((int)xs_size);
            //    d_x_update_global.view().fill(Vector3::Zero());
            //    inited = true;
            //}
            bool inited = false;
            muda::DeviceBuffer<Vector3> d_x_update_global;  // 对应 x_update_h_3v（外部传输用）
            muda::DeviceBuffer<Vector3> d_x_update_pos;  // 新增：对应 x_update_h_global（位置更新用）
            if(!inited)
            {
                d_x_update_global.resize((int)xs_size);
                d_x_update_global.view().fill(Vector3::Zero());
                d_x_update_pos.resize((int)xs_size);  // 初始化新增缓冲区
                d_x_update_pos.view().fill(Vector3::Zero());
                inited = true;
            }

            muda::DeviceBuffer<IndexT> d_color_vertices;
            for(size_t iGroup = 0; iGroup < vertex_group.size(); iGroup++)
            {
                // 2. clear gradient and hessian
                {
                    Timer timer{"Clear Gradient (all vertices)"};
                    info.gradients().buffer_view().fill(0);
                    info.hessians().values().fill(Matrix3x3::Zero());
                }

                auto& parallelGroup = vertex_group[iGroup];
                // host 转换为 IndexT 并拷到 GPU
                std::vector<IndexT> group_indices(parallelGroup.begin(),
                                                  parallelGroup.end());
                d_color_vertices.resize(static_cast<int>(group_indices.size()));
                if(!group_indices.empty())
                    d_color_vertices.view().copy_from(group_indices.data());
                // 3. 组装梯度和海森矩阵（分步骤）
                {
                    Timer timer{"GPU Assemble producers (all vertices)"};
                    //_assemble_producers(info);
                    _assemble_producers_by_color(info, d_color_vertices);
                }
                {
                    //但是这里的基础的gradient 和 hessian 是之前就计算过了的,再计算也不会变
                    Timer timer{"Assemble dytopo effect (all vertices)"};
                    _assemble_dytopo_effect(info);
                }
                {
                    Timer timer{"Assemble animation (all vertices)"};
                    _assemble_animation(info);
                }
                ////////////////////////===============after this code we do not need to verify first
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
                                hessians = info.hessians().viewer().name(
                                    "hessians")] __device__(int I) mutable
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

                //////////================================GPU 遍历这些color 中的顶点:
                // 1) 构建对角 H(ii)（GPU端），避免回传 triplets
                muda::DeviceBuffer<Matrix3x3> d_Hii;
                d_Hii.resize((int)xs_size);
                d_Hii.view().fill(Matrix3x3::Zero());

                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply(info.hessians().triplet_count(),
                           [hessians = info.hessians().viewer().name("hessians"),
                            Hii = d_Hii.viewer().name("Hii")] __device__(int I) mutable
                           {
                               auto&& [i, j, H3] = hessians(I).read();
                               if(i == j)
                               {
                                   // 原子累加 3x3 到 Hii(i)
                                   // 注意：若 Float 为 float 则 atomicAdd 可用；若为 double 请改为分块规约或使用自定义原子
                                   muda::atomic_add(&Hii(i)(0, 0), H3(0, 0));
                                   muda::atomic_add(&Hii(i)(0, 1), H3(0, 1));
                                   muda::atomic_add(&Hii(i)(0, 2), H3(0, 2));
                                   muda::atomic_add(&Hii(i)(1, 0), H3(1, 0));
                                   muda::atomic_add(&Hii(i)(1, 1), H3(1, 1));
                                   muda::atomic_add(&Hii(i)(1, 2), H3(1, 2));
                                   muda::atomic_add(&Hii(i)(2, 0), H3(2, 0));
                                   muda::atomic_add(&Hii(i)(2, 1), H3(2, 1));
                                   muda::atomic_add(&Hii(i)(2, 2), H3(2, 2));
                               }
                           }).wait();
                /////////////=============这里是进一步计算更新方向的问题，其他先不用真的更新gpu中的数据
                // 2) 对本颜色组的每个顶点并行：计算后同时更新两个数组
                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply((int)d_color_vertices.size(),
                           [verts = d_color_vertices.cviewer().name("color_vertices"),
                            is_fixed = fem().is_fixed.cviewer().name("is_fixed"),
                            gradients = info.gradients().cviewer().name("gradients"),
                            Hii = d_Hii.cviewer().name("Hii"),
                            xupd = d_x_update_global.viewer().name("xupd"),  // 对应 x_update_h_3v
                            xpos = d_x_update_pos.viewer().name("xpos"),  // 对应 x_update_h_global
                            xs = fem().xs.viewer().name("xs"),
                            xs_previous = xs_previous_gpu.viewer().name("xs_previous")
                           ] __device__(int k) mutable  // 用于实时更新顶点位置
                           {
                               const int v = (int)verts(k);
                               if(v < 0 || is_fixed(v))
                                   return;

                               // 1. 计算 descent（原有逻辑不变）
                               const Vector3 G = gradients.segment<3>(v * 3).as_eigen();
                               Vector3   force = -G;
                               Matrix3x3 H     = Hii(v);
                               if(H.isZero())
                                   H = Matrix3x3::Identity() * 1e-6f;
                               Float m[9]  = {H(0, 0),
                                              H(1, 0),
                                              H(2, 0),
                                              H(0, 1),
                                              H(1, 1),
                                              H(2, 1),
                                              H(0, 2),
                                              H(1, 2),
                                              H(2, 2)};
                               Float b[3]  = {force(0), force(1), force(2)};
                               Float dx[3] = {0, 0, 0};

                               bool solver_success =
                                   solve3x3_psd_stable<Float>(m, b, dx);
                               Vector3 descent;
                               if(!solver_success)
                               {
                                   descent = force;  // 与CPU一致：失败时用force作为方向
                                   printf("Solver failed at vertex %d (GPU)\n", v);
                               }
                               else
                               {
                                   descent = Vector3(dx[0], dx[1], dx[2]);
                               }

                               // 2. 同步更新两个数组（关键修改）
                               // 对应 CPU 的 x_update_h_3v：-= descent*2（等价于 += (-descent*2)）
                               xupd(v) -= descent * 2;
                               // 对应 CPU 的 x_update_h_global：+= descent
                               xpos(v) += descent;
                               //// 3. 实时更新顶点位置（延续之前的中间更新逻辑，基于 xpos 的累积）
                               //xs(v) = xs_previous(v) + xpos(v);  // 注意：xs_previous 需是初始位置的设备副本

                               //// 10. 更新x_update数组
                               //{
                               //    Timer timer{"Update x_update arrays (all vertices)"};
                               //    for(int k = 0; k < 3; ++k)
                               //    {
                               //        //x_update_h_3v[vertexId * 3 + k] -= descentDirection[k];
                               //        x_update_h_3v[vertexId * 3 + k] -=
                               //            descentDirection[k] * 2;
                               //    }
                               //    x_update_h_global[vertexId] += descentDirection;
                               //}

                               //// 11. 更新顶点位置
                               //{
                               //    Timer timer{"Update xs_temp and copy to fem().xs (all vertices)"};
                               //    std::vector<Vector3> xs_temp(xs_previous.size());
                               //    for(size_t i = 0; i < xs_previous.size(); ++i)
                               //    {
                               //        xs_temp[i] =
                               //            xs_previous[i] + x_update_h_global[i];
                               //    }
                               //    xs_temp_global = xs_temp;
                               //    fem().xs.copy_from(xs_temp);
                               //}
                               xs(v) = xs_previous(v) + xpos(v);  // 注意：xs_previous 需是初始位置的设备副本
                           })
                    .wait();
                
                ////=================================================================
                //// GPU并行更新完成后，拷贝两个数组到CPU
                //std::vector<Vector3> gpu_x_update_global;  // 对应CPU的x_update_h_3v（向量形式）
                //std::vector<Vector3> gpu_x_update_pos;  // 对应CPU的x_update_h_global
                //{
                //    Timer timer{"Copy GPU update arrays to CPU"};
                //    gpu_x_update_global.resize(xs_size);
                //    gpu_x_update_pos.resize(xs_size);
                //    d_x_update_global.copy_to(gpu_x_update_global);  // 拷贝GPU的d_x_update_global
                //    d_x_update_pos.copy_to(gpu_x_update_pos);  // 拷贝GPU的d_x_update_pos
                //}
                //// 我们这里没有分为mesh id 和 vertex id, 直接用vertex id
                //////////////////////////////////we need to compare gradietn with or without 
                //// GPU端：组装完梯度后，立即记录gradient结果
                //std::vector<Float> gpu_gradients;  // 保存GPU计算的梯度
                //{
                //    Timer timer{"Save GPU gradients"};
                //    gpu_gradients.resize(info.gradients().size());
                //    info.gradients().buffer_view().copy_to(gpu_gradients.data());  // 从GPU拷贝到CPU
                //}
                ////GPU端：组装完梯度后，立即记录hessain结果
                //struct HessianTriplet
                //{
                //    IndexT    i;
                //    IndexT    j;
                //    Matrix3x3 H3;
                //};
                //std::vector<HessianTriplet> gpu_hessian_triplets;
                //{
                //    Timer timer{"Save GPU Hessian triplets"};
                //    int   triplet_count = info.hessians().triplet_count();
                //    gpu_hessian_triplets.resize(triplet_count);

                //    // 拷贝行、列索引和值
                //    std::vector<IndexT>    rows(triplet_count);
                //    std::vector<IndexT>    cols(triplet_count);
                //    std::vector<Matrix3x3> values(triplet_count);
                //    info.hessians().row_indices().copy_to(rows.data());
                //    info.hessians().col_indices().copy_to(cols.data());
                //    info.hessians().values().copy_to(values.data());

                //    // 封装为HessianTriplet
                //    for(int i = 0; i < triplet_count; ++i)
                //    {
                //        gpu_hessian_triplets[i] = {rows[i], cols[i], values[i]};
                //    }
                //}
                //// 在GPU构建d_Hii之后，拷贝到CPU
                //std::vector<Matrix3x3> gpu_Hii;
                //{
                //    Timer timer{"Save GPU Hii"};
                //    gpu_Hii.resize(xs_size);
                //    d_Hii.copy_to(gpu_Hii);  // 从GPU拷贝d_Hii到CPU
                //}
                //std::vector<Matrix3x3> cpu_Hii(xs_size, Matrix3x3::Zero());

                ////=================================================================CPU逐个处理该颜色组的顶点，重复GPU的组装步骤
                //for(auto vertexId : group_indices)  // group_indices是当前颜色组的顶点ID列表
                //{
                //    const int vtttt0 = static_cast<int>(vertexId);
                //    // 清零梯度，准备CPU验证
                //    info.gradients().buffer_view().fill(0);
                //    info.hessians().values().fill(Matrix3x3::Zero());  // 海森矩阵也需清零，避免干扰
                //    // 重复GPU的三个组装步骤（确保逻辑完全一致）
                //    {
                //        Timer timer{"CPU Assemble producers (single vertex)"};
                //        _assemble_producers_by_vertex(info, vertexId);  // CPU逐个顶点组装
                //    }
                //    {
                //        Timer timer{"CPU Assemble dytopo effect (single vertex)"};
                //        _assemble_dytopo_effect(info);  // 与GPU步骤一致
                //    }
                //    {
                //        Timer timer{"CPU Assemble animation (single vertex)"};
                //        _assemble_animation(info);  // 与GPU步骤一致
                //    }
                //    // 4. 清除固定顶点梯度（并行操作）
                //    {
                //        Timer timer{"Clear Fixed Vertex Gradient (ParallelFor, all vertices)"};
                //        ParallelFor()
                //            .file_line(__FILE__, __LINE__)
                //            .apply(
                //                fem().xs.size(),
                //                [is_fixed = fem().is_fixed.cviewer().name("is_fixed"),
                //                 gradients = info.gradients().viewer().name(
                //                     "gradients")] __device__(int i) mutable
                //                {
                //                    if(is_fixed(i))
                //                    {
                //                        gradients.segment<3>(i * 3).as_eigen().setZero();
                //                    }
                //                });
                //    }
                //    // 5. 清除固定顶点海森矩阵（并行操作）
                //    {
                //        Timer timer{"Clear Fixed Vertex hessian (ParallelFor, all vertices)"};
                //        ParallelFor()
                //            .file_line(__FILE__, __LINE__)
                //            .apply(info.hessians().triplet_count(),
                //                   [is_fixed = fem().is_fixed.cviewer().name("is_fixed"),
                //                    hessians = info.hessians().viewer().name(
                //                        "hessians")] __device__(int I) mutable
                //                   {
                //                       auto&& [i, j, H3] = hessians(I).read();

                //                       if(is_fixed(i) || is_fixed(j))
                //                       {
                //                           if(i != j)
                //                               hessians(I).write(i, j, Matrix3x3::Zero());
                //                       }
                //                   })
                //            .wait();
                //    }

                //    std::vector<HessianTriplet> cpu_hessian_triplets;
                //    {
                //        Timer timer{"Save CPU Hessian triplets (single vertex)"};
                //        int triplet_count = info.hessians().triplet_count();
                //        cpu_hessian_triplets.resize(triplet_count);

                //        std::vector<IndexT>    rows(triplet_count);
                //        std::vector<IndexT>    cols(triplet_count);
                //        std::vector<Matrix3x3> values(triplet_count);
                //        info.hessians().row_indices().copy_to(rows.data());
                //        info.hessians().col_indices().copy_to(cols.data());
                //        info.hessians().values().copy_to(values.data());

                //        for(int i = 0; i < triplet_count; ++i)
                //        {
                //            cpu_hessian_triplets[i] = {rows[i], cols[i], values[i]};
                //        }
                //    }


                //    // 6. 提取梯度数据到CPU并计算力
                //    std::vector<Vector3> force_h;
                //    std::vector<Float>   gradients_h;
                //    int                  gradients_size;
                //    {
                //        Timer timer{"Extract gradients to CPU and compute force (all vertices)"};
                //        force_h.resize(vertex_size);
                //        gradients_size = info.gradients().size();
                //        gradients_h.resize(gradients_size);
                //        info.gradients().buffer_view().copy_to(gradients_h.data());

                //        // 提取当前顶点梯度并计算力
                //        Vector3 gradient = get_vertex_gradient(gradients_h, vertexId);
                //        force_h[vertexId] = -gradient;
                //    }

                //    // 7. 提取海森矩阵三元组到CPU
                //    IndexT                 triplet_size;
                //    std::vector<IndexT>    host_rows;
                //    std::vector<IndexT>    host_cols;
                //    std::vector<Matrix3x3> host_values;
                //    int                    hessian_size;
                //    {
                //        Timer timer{"Extract Hessian triplets to CPU (all vertices)"};
                //        triplet_size = info.hessians().row_indices().size();
                //        host_rows.resize(triplet_size);
                //        host_cols.resize(triplet_size);
                //        host_values.resize(triplet_size);
                //        info.hessians().row_indices().copy_to(host_rows.data());
                //        info.hessians().col_indices().copy_to(host_cols.data());
                //        info.hessians().values().copy_to(host_values.data());
                //        hessian_size = info.hessians().total_triplet_count();
                //    }

                //    // 8. 遍历三元组获取当前顶点海森矩阵
                //    Matrix3x3 h = Matrix3x3::Zero();
                //    {
                //        Timer timer{"Loop through triplets to get hessian h (all vertices)"};
                //        for(int I = 0; I < triplet_size; ++I)
                //        {
                //            int i_vertex = host_rows[I];
                //            int j_vertex = host_cols[I];
                //            if(i_vertex == vertexId && j_vertex == vertexId)
                //            {
                //                h += host_values[I];
                //            }
                //        }
                //    }

                //    // 9. 求解3x3矩阵及相关处理
                //    {
                //        Timer  timer{"Solve 3x3 PSD system (all vertices)"};
                //        auto&  force     = force_h[vertexId];
                //        double ForceNorm = force.squaredNorm();

                //        if(1)  // 保留原条件
                //        {
                //            if(force.isZero())
                //            {
                //                continue;
                //            }
                //            if(h.isZero())
                //            {
                //                h = Matrix3x3::Identity() * 1e-6;
                //                continue;
                //            }

                //            Vector3 descentDirection;
                //            Float   stepSize               = 1;
                //            Float   lineSearchShrinkFactor = 0.8;
                //            bool    solverSuccess;

                //            bool useDouble3x3 = 1;
                //            if(useDouble3x3)
                //            {
                //                double H[9] = {h(0, 0),
                //                               h(1, 0),
                //                               h(2, 0),
                //                               h(0, 1),
                //                               h(1, 1),
                //                               h(2, 1),
                //                               h(0, 2),
                //                               h(1, 2),
                //                               h(2, 2)};

                //                double F[3]   = {force(0), force(1), force(2)};
                //                double dx[3]  = {0, 0, 0};
                //                solverSuccess = solve3x3_psd_stable(H, F, dx);
                //                descentDirection = Vector3(dx[0], dx[1], dx[2]);

                //                // 验证计算
                //                auto   TestOuput = h * descentDirection;
                //                auto   diff      = TestOuput - force;
                //                double diff_norm = diff.norm();
                //                if(diff_norm > 1e-6)
                //                {
                //                    std::cout << "Warning: h * descentDirection does not match force (diff_norm = "
                //                              << diff_norm << ")" << std::endl;
                //                }
                //            }
                //            else
                //            {
                //                solverSuccess = false;  // 未使用分支
                //            }

                //            // 处理求解失败
                //            if(!solverSuccess)
                //            {
                //                stepSize               = 1;
                //                descentDirection       = force;
                //                lineSearchShrinkFactor = 0.8;
                //                std::cout << "Solver failed at vertex "
                //                          << vertexId << std::endl;
                //            }

                //            // 检查数值异常
                //            if(descentDirection.hasNaN())
                //            {
                //                std::cout << "force: " << force.transpose() << "\nHessian:\n"
                //                          << h;
                //                std::cout << "descentDirection has NaN at vertex "
                //                          << vertexId << std::endl;
                //                std::exit(-1);
                //            }

                //            // 10. 更新x_update数组
                //            {
                //                Timer timer{"Update x_update arrays (all vertices)"};
                //                for(int k = 0; k < 3; ++k)
                //                {
                //                    //x_update_h_3v[vertexId * 3 + k] -= descentDirection[k];
                //                    x_update_h_3v[vertexId * 3 + k] -=
                //                        descentDirection[k] * 2;
                //                }
                //                x_update_h_global[vertexId] += descentDirection;
                //            }

                //            // 11. 更新顶点位置
                //            {
                //                Timer timer{"Update xs_temp and copy to fem().xs (all vertices)"};
                //                std::vector<Vector3> xs_temp(xs_previous.size());
                //                for(size_t i = 0; i < xs_previous.size(); ++i)
                //                {
                //                    xs_temp[i] = xs_previous[i] + x_update_h_global[i];
                //                }
                //                xs_temp_global = xs_temp;
                //                fem().xs.copy_from(xs_temp);
                //            }
                //        }
                //    }

                //    /////////////////从此之后是比较gradient 和hessian的 差异
                //    cpu_Hii[vertexId] = h;  // 保存当前顶点的CPU对角Hessian
                //    // 记录CPU组装后的梯度结果
                //    std::vector<Float> cpu_gradients;
                //    {
                //        Timer timer{"Save CPU gradients"};
                //        cpu_gradients.resize(info.gradients().size());
                //        info.gradients().buffer_view().copy_to(cpu_gradients.data());  // 从设备/内存拷贝（根据info存储位置）
                //    }

                //    // 仅对当前 vertexId 的3个分量做比较
                //    bool        gradients_match = true;
                //    const Float eps_abs         = 1e6f;  // 绝对误差阈值
                //    const int   base = static_cast<int>(vertexId) * 3;

                //    for(int comp = 0; comp < 3; ++comp)
                //    {
                //        const Float g_gpu = gpu_gradients[base + comp];
                //        const Float g_cpu = cpu_gradients[base + comp];
                //        const Float diff  = std::abs(g_gpu - g_cpu);
                //        if(diff > eps_abs)
                //        {
                //            gradients_match = false;
                //            std::cout << "Gradient mismatch at vertex " << vertexId
                //                      << ", component " << comp << ": GPU=" << g_gpu
                //                      << ", CPU=" << g_cpu << ", diff=" << diff;
                //        }
                //    }

                //    if(gradients_match)
                //        std::cout << "Color group " << iGroup << " vertex " << vertexId
                //                  << " gradients match (GPU vs CPU)." << std::endl;
                //    else
                //        std::cout << "+++++++++++++++++++++Color group " << iGroup << " vertex " << vertexId
                //                  << " gradients DO NOT match!" << std::endl;

                //    // 对比当前顶点的CPU和GPU对角Hii
                //    compare_hessian_hii(vertexId, cpu_Hii[vertexId], gpu_Hii[vertexId]);
                //    int ContinueCompare = 0;
                //    // --------------------------
                //    // 对比 d_x_update_global vs x_update_h_3v
                //    // --------------------------
                //    const int v = static_cast<int>(vertexId);
                //    bool xupd_match = true;
                //    const Vector3& gpu_xupd = gpu_x_update_global[v];  // GPU的向量结果
                //    // 从CPU的扁平数组中提取当前顶点的3个分量
                //    Vector3 cpu_xupd(x_update_h_3v[v * 3 + 0],
                //                     x_update_h_3v[v * 3 + 1],
                //                     x_update_h_3v[v * 3 + 2]);
                //    // 对比每个分量
                //    const Float eps_xupd = std::is_same_v<Float, float> ? 1e-5f : 1e-10;  // 精度阈值
                //    for(int comp = 0; comp < 3; ++comp)
                //    {
                //        Float diff = std::abs(gpu_xupd[comp] - cpu_xupd[comp]);
                //        if(diff > eps_xupd)
                //        {
                //            xupd_match = false;
                //            std::cout << "x_update_global mismatch at vertex "
                //                      << v << ", component " << comp << ": "
                //                      << "GPU=" << gpu_xupd[comp]
                //                      << ", CPU=" << cpu_xupd[comp]
                //                      << ", diff=" << diff << std::endl;
                //        }
                //    }

                //    // --------------------------
                //    // 对比 d_x_update_pos vs x_update_h_global
                //    // --------------------------
                //    bool xpos_match = true;
                //    const Vector3& gpu_xpos = gpu_x_update_pos[v];  // GPU的累积更新
                //    const Vector3& cpu_xpos = x_update_h_global[v];  // CPU的累积更新
                //    for(int comp = 0; comp < 3; ++comp)
                //    {
                //        Float diff = std::abs(gpu_xpos[comp] - cpu_xpos[comp]);
                //        if(diff > eps_xupd)
                //        {  // 同精度阈值
                //            xpos_match = false;
                //            std::cout << "x_update_pos mismatch at vertex " << v
                //                      << ", component " << comp << ": "
                //                      << "GPU=" << gpu_xpos[comp]
                //                      << ", CPU=" << cpu_xpos[comp]
                //                      << ", diff=" << diff << std::endl;
                //        }
                //    }

                //    // 输出对比结果
                //    if(xupd_match && xpos_match)
                //    {
                //        std::cout << "Vertex " << v << " update arrays match (GPU vs CPU)."
                //                  << std::endl;
                //    }
                //    else
                //    {
                //        std::cout << "+++++++++++++++++++++Vertex " << v
                //                  << " update arrays DO NOT match!" << std::endl;
                //    }
                //    int testVertex = vertexId;
                //}

                int StopColorEnd = 0;
            }

            int endEachGroup = 0;
            ///////////==============================gpu 版本的更新逻辑
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(fem().xs.size(),  // 按顶点数量启动线程，每个线程处理一个顶点v
                       [d_x_update_global = d_x_update_global.viewer().name("d_x_update_global"),
                        d_x_update_pos = d_x_update_pos.viewer().name("d_x_update_pos"),
                        x_update = info.x_update().viewer().name("x_update"),  // 扁平float数组
                        xs_position = info.xs_position().viewer().name("xs_position"),  // 扁平float数组
                        xs_previous = xs_previous_gpu.viewer().name("xs_previous"),
                        xs = fem().xs.viewer().name("xs")]  // 顶点位置缓冲区
                       __device__(int v) mutable  // v：当前处理的顶点索引
                       {
                           // 1. 扁平化d_x_update_global到info.x_update()（对应CPU的x_update_h_3v）
                           const Vector3& upd = d_x_update_global(v);  // 取当前顶点的更新量（Vector3）
                           x_update(v * 3 + 0) = upd.x();  // 存储x分量
                           x_update(v * 3 + 1) = upd.y();  // 存储y分量
                           x_update(v * 3 + 2) = upd.z();  // 存储z分量

                           // 2. 扁平化xs_previous到info.xs_position()（对应CPU的xs_update_3v）
                           const Vector3& prev_pos = xs_previous(v);  // 取当前顶点的历史位置
                           xs_position(v * 3 + 0) = prev_pos.x();
                           xs_position(v * 3 + 1) = prev_pos.y();
                           xs_position(v * 3 + 2) = prev_pos.z();

                           // 3. 更新fem().xs为：历史位置 + 位置更新量（对应CPU的fem().xs.copy_from）
                           xs(v) = xs_previous(v);
                       })
                .wait();  // 等待所有线程完成
            ///////////==============================cpu 版本的更新逻辑
            //// 1) 从 GPU 读取 d_x_update_global 到 host
            //std::vector<Vector3> xupd_host(xs_size);
            //d_x_update_global.copy_to(xupd_host);

            //// 2) 扁平化为 CPU 侧的 x_update_h_3v（3*vertex_size）
            //x_update_h_3v.resize(3 * xs_size);
            //for(int i = 0; i < (int)xs_size; ++i)
            //{
            //    x_update_h_3v[i * 3 + 0] = xupd_host[i][0];
            //    x_update_h_3v[i * 3 + 1] = xupd_host[i][1];
            //    x_update_h_3v[i * 3 + 2] = xupd_host[i][2];
            //}
            ////这里也是组外
            //// 12. 同步回GPU
            //{
            //    Timer timer{"Copy xs_update_3v to GPU (info.x_update())"};
            //    for(int vertexId = 0; vertexId < vertex_size; ++vertexId)
            //    {
            //        for(int k = 0; k < 3; ++k)
            //        {
            //            xs_update_3v[vertexId * 3 + k] = xs_previous[vertexId][k];
            //        }
            //    }
            //    fem().xs.copy_from(xs_previous);
            //    info.xs_position().buffer_view().copy_from(xs_update_3v.data());
            //    info.x_update().buffer_view().copy_from(x_update_h_3v.data());
            //}
            //std::cout << "###########################################################"<< std::endl;
        }
    }
}

void FEMLinearSubsystem::Impl::solve_system_vertex_cpu(GlobalLinearSystem::DiagInfo& info)
{
    // 顶点染色初始化（单独计时，染色可能耗时）
    bool use_vertex_coloring = true;
    if(use_vertex_coloring && vertex_group.empty())
    {
        Timer timer{"CPU Vertex coloring initialization"};
        vertices_Coloring();
    }

    using namespace muda;
    // 总函数耗时统计
    Timer totalTimer{"Total CPU solve_system_vertex time"};

    auto N           = fem().xs.size();
    auto vertex_size = N;
    auto info_x_size = info.x_update().size();

    // 1. 同步GPU数据到CPU（分步骤计时，数据传输是CPU版本的关键瓶颈）
    std::vector<IndexT> is_fixed_host;
    {
        Timer timer{"CPU Sync is_fixed (GPU to CPU)"};
        fem().is_fixed.copy_to(is_fixed_host);  // 同步固定顶点标记
    }

    std::vector<Vector3> xs_previous;
    {
        Timer timer{"CPU Sync xs_previous (GPU to CPU)"};
        fem().xs.copy_to(xs_previous);  // 同步初始顶点位置
    }

    // 初始化CPU端数组（明确大小，计时内存分配）
    std::vector<Float>   x_update_h_3v;
    std::vector<Vector3> x_update_h_global;
    {
        Timer timer{"CPU Initialize x_update arrays"};
        x_update_h_3v.resize(info_x_size, 0.0f);       // 大小为info_x_size
        x_update_h_global.resize(N, Vector3::Zero());  // 大小为N
    }

    // 2. 处理顶点分组（明确计时分组逻辑，尤其是大顶点数时）
    std::vector<std::vector<IndexT>> groups;
    {
        Timer timer{"CPU Prepare vertex groups"};
        if(use_vertex_coloring && !vertex_group.empty())
        {
            groups = vertex_group;  // 复用已有分组
        }
        else
        {
            groups.resize(1);
            groups[0].resize(N);
            std::iota(groups[0].begin(), groups[0].end(), 0);  // 生成0~N-1
        }
    }

    // 3. 按组处理顶点（每组总耗时统计）
    for(size_t iGroup = 0; iGroup < groups.size(); ++iGroup)
    {
        Timer groupTimer{"CPU Group total time"};

        auto& group_indices = groups[iGroup];
        if(group_indices.empty())
            continue;

        // 逐个顶点计算（核心逻辑，总计时）
        {
            Timer timer{"CPU Per-vertex computation"};
            for(auto vertexId : group_indices)
            {
                // 跳过固定顶点（简单判断，无需单独计时）
                if(vertexId < 0 || is_fixed_host[vertexId])
                    continue;

                // 3.1 清除当前梯度和海森矩阵（GPU端操作，计时GPU清零）
                {
                    Timer timer{"CPU Trigger GPU clear Gradient/Hessian (vertex)"};
                    info.gradients().buffer_view().fill(0);
                    info.hessians().values().fill(Matrix3x3::Zero());
                }

                // 3.2 组装梯度和海森矩阵（分步骤计时CPU组装逻辑）
                {
                    Timer timer{"CPU Assemble producers (vertex)"};
                    _assemble_producers_by_vertex(info, vertexId);
                }
                {
                    Timer timer{"CPU Assemble dytopo effect (vertex)"};
                    _assemble_dytopo_effect(info);
                }
                {
                    Timer timer{"CPU Assemble animation (vertex)"};
                    _assemble_animation(info);
                }

                // 3.3 清除固定顶点的梯度（GPU并行操作，计时GPU耗时）
                {
                    Timer timer{"CPU Trigger GPU clear fixed gradients (vertex)"};
                    ParallelFor()
                        .file_line(__FILE__, __LINE__)
                        .apply(fem().xs.size(),
                               [is_fixed = fem().is_fixed.cviewer().name("is_fixed"),
                                gradients = info.gradients().viewer().name(
                                    "gradients")] __device__(int i)
                               {
                                   if(is_fixed(i))
                                       gradients.segment<3>(i * 3).as_eigen().setZero();
                               })
                        .wait();
                }

                // 3.4 清除固定顶点的非对角海森矩阵（GPU并行操作，计时）
                {
                    Timer timer{"CPU Trigger GPU clear fixed hessians (vertex)"};
                    ParallelFor()
                        .file_line(__FILE__, __LINE__)
                        .apply(info.hessians().triplet_count(),
                               [is_fixed = fem().is_fixed.cviewer().name("is_fixed"),
                                hessians = info.hessians().viewer().name("hessians")] __device__(int I)
                               {
                                   auto&& [i, j, H3] = hessians(I).read();
                                   if((is_fixed(i) || is_fixed(j)) && i != j)
                                       hessians(I).write(i, j, Matrix3x3::Zero());
                               })
                        .wait();
                }

                // 3.5.1 提取梯度 to 力（GPU to CPU传输，计时数据传输）
                std::vector<Float> gradients_h;
                {
                    Timer  timer{"CPU Copy gradients (GPU to CPU) (vertex)"};
                    size_t grad_size = info.gradients().size();
                    gradients_h.resize(grad_size);
                    info.gradients().buffer_view().copy_to(gradients_h.data());
                }
                Vector3 gradient(gradients_h[vertexId * 3 + 0],
                                 gradients_h[vertexId * 3 + 1],
                                 gradients_h[vertexId * 3 + 2]);
                Vector3 force = -gradient;

                // 3.5.2 提取对角Hessian（GPU to CPU传输+CPU组装，分步骤计时）
                std::vector<IndexT>    rows;
                std::vector<IndexT>    cols;
                std::vector<Matrix3x3> values;
                {
                    Timer  timer{"CPU Copy hessian triplets (GPU to CPU) (vertex)"};
                    size_t triplet_count = info.hessians().triplet_count();
                    rows.resize(triplet_count);
                    cols.resize(triplet_count);
                    values.resize(triplet_count);
                    info.hessians().row_indices().copy_to(rows.data());
                    info.hessians().col_indices().copy_to(cols.data());
                    info.hessians().values().copy_to(values.data());
                }

                // 3.5.2.1 CPU端组装对角Hessian（单独计时计算逻辑）
                Matrix3x3 h = Matrix3x3::Zero();
                {
                    Timer timer{"CPU Assemble diagonal Hii (vertex)"};
                    for(int I = 0; I < (int)values.size(); ++I)
                    {
                        if(rows[I] == vertexId && cols[I] == vertexId)
                            h += values[I];
                    }
                    if(h.isZero())
                        h = Matrix3x3::Identity() * 1e-6f;
                }

                // 3.5.3 求解更新方向（计时线性求解器）
                Vector3 descentDirection;
                bool    solverSuccess;
                {
                    Timer  timer{"CPU Solve 3x3 system (vertex)"};
                    double H[9]   = {h(0, 0),
                                     h(1, 0),
                                     h(2, 0),
                                     h(0, 1),
                                     h(1, 1),
                                     h(2, 1),
                                     h(0, 2),
                                     h(1, 2),
                                     h(2, 2)};
                    double F[3]   = {force(0), force(1), force(2)};
                    double dx[3]  = {0, 0, 0};
                    solverSuccess = solve3x3_psd_stable(H, F, dx);
                    if(!solverSuccess)
                    {
                        descentDirection = force;
                        std::cout << "Solver failed at vertex " << vertexId << " (CPU)\n";
                    }
                    else
                    {
                        descentDirection = Vector3(dx[0], dx[1], dx[2]);
                    }
                }

                // 3.5.4 更新CPU端数组（简单内存操作，无需单独计时，包含在循环总计时中）
                for(int k = 0; k < 3; ++k)
                    x_update_h_3v[vertexId * 3 + k] -= descentDirection[k] * 2;
                x_update_h_global[vertexId] += descentDirection;

                // 4.1 实时更新顶点位置（CPU to GPU传输，计时数据回传）
                {
                    Timer timer{"CPU Update vertex positions (CPU to GPU) (vertex)"};
                    std::vector<Vector3> xs_temp(N);
                    for(size_t i = 0; i < N; ++i)
                        xs_temp[i] = xs_previous[i] + x_update_h_global[i];
                    fem().xs.copy_from(xs_temp);
                }
            }
        }
    }

    // 4. 同步结果到info结构（最终数据整理+传输，计时）
    {
        Timer              timer{"CPU Final sync to info (CPU to GPU)"};
        std::vector<Float> xs_update_3v(3 * N);
        for(size_t i = 0; i < N; ++i)
        {
            xs_update_3v[i * 3 + 0] = xs_previous[i].x();
            xs_update_3v[i * 3 + 1] = xs_previous[i].y();
            xs_update_3v[i * 3 + 2] = xs_previous[i].z();
        }
        fem().xs.copy_from(xs_previous);  // 恢复初始位置
        info.xs_position().buffer_view().copy_from(xs_update_3v.data());
        info.x_update().buffer_view().copy_from(x_update_h_3v.data());
    }
}

void FEMLinearSubsystem::Impl::solve_system_vertex_gpu(GlobalLinearSystem::DiagInfo& info)
{
    // 顶点染色控制（按需执行）
    bool use_vertex_coloring = true;
    if(use_vertex_coloring && vertex_group.empty())
    {
        Timer timer{"Vertex coloring initialization"};
        vertices_Coloring();
    }

    // 总函数耗时统计
    Timer totalTimer{"Total solve_system_vertex_gpu time"};

    using namespace muda;
    auto vertex_size = fem().xs.size();         // 顶点总数
    auto info_x_size = info.x_update().size();  // info更新数组大小

    // 1. 初始化关键设备缓冲区（存储初始顶点位置）
    muda::DeviceBuffer<Vector3> xs_previous_gpu;
    {
        Timer timer{"Initialize xs_previous_gpu (GPU memory + copy)"};
        xs_previous_gpu = fem().xs;  // 从fem().xs拷贝初始位置到设备缓冲区
    }

    // 顶点循环整体耗时统计
    {
        Timer vertexLoopTimer{"Total vertex loop time (all groups)"};

        // 2. 初始化更新用设备缓冲区（x_update_h_3v和x_update_h_global的GPU对应）
        muda::DeviceBuffer<Vector3> d_x_update_global;  // 对应CPU的x_update_h_3v
        muda::DeviceBuffer<Vector3> d_x_update_pos;  // 对应CPU的x_update_h_global
        {
            Timer timer{"Initialize d_x_update buffers (resize + zero fill)"};
            d_x_update_global.resize((int)vertex_size);
            d_x_update_global.view().fill(Vector3::Zero());  // 初始化为零
            d_x_update_pos.resize((int)vertex_size);
            d_x_update_pos.view().fill(Vector3::Zero());  // 初始化为零
        }

        // 设备缓冲区：存储当前颜色组的顶点索引
        muda::DeviceBuffer<IndexT> d_color_vertices;

        // 3. 按颜色组遍历顶点
        for(size_t iGroup = 0; iGroup < vertex_group.size(); ++iGroup)
        {
            Timer groupTimer{"Group total time"};

            // 3.1 清除当前组的梯度和海森矩阵（GPU全局清零）
            {
                Timer timer{"Clear gradient and hessian (GPU fill zero)"};
                info.gradients().buffer_view().fill(0);            // 梯度清零
                info.hessians().values().fill(Matrix3x3::Zero());  // 海森矩阵值清零
            }

            // 3.2 处理当前颜色组的顶点索引（CPU to GPU拷贝）
            auto&               parallelGroup = vertex_group[iGroup];
            std::vector<IndexT> group_indices(parallelGroup.begin(),
                                              parallelGroup.end());
            {
                Timer timer{"Copy group indices to GPU (d_color_vertices)"};
                d_color_vertices.resize(static_cast<int>(group_indices.size()));
                if(!group_indices.empty())
                    d_color_vertices.view().copy_from(group_indices.data());  // CPU to GPU拷贝
            }

            // 3.3 组装梯度和海森矩阵（分步骤计时）
            {
                Timer timer{"Assemble producers (by color group)"};
                _assemble_producers_by_color(info, d_color_vertices);  // 按颜色组并行组装
            }
            {
                Timer timer{"Assemble dytopo effect"};
                _assemble_dytopo_effect(info);  // 动态拓扑影响组装
            }
            {
                Timer timer{"Assemble animation effect"};
                _assemble_animation(info);  // 动画影响组装
            }

            // 3.4 清除固定顶点的梯度（GPU并行）
            {
                Timer timer{"Clear fixed vertex gradients (GPU ParallelFor)"};
                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply(fem().xs.size(),
                           [is_fixed = fem().is_fixed.cviewer().name("is_fixed"),
                            gradients = info.gradients().viewer().name("gradients")] __device__(int i)
                           {
                               if(is_fixed(i))  // 固定顶点梯度置零
                                   gradients.segment<3>(i * 3).as_eigen().setZero();
                           })
                    .wait();  // 等待GPU操作完成
            }

            // 3.5 清除固定顶点的非对角海森矩阵（GPU并行）
            {
                Timer timer{"Clear fixed vertex non-diag hessians (GPU ParallelFor)"};
                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply(info.hessians().triplet_count(),
                           [is_fixed = fem().is_fixed.cviewer().name("is_fixed"),
                            hessians = info.hessians().viewer().name("hessians")] __device__(int I)
                           {
                               auto&& [i, j, H3] = hessians(I).read();
                               if((is_fixed(i) || is_fixed(j)) && i != j)
                                   hessians(I).write(i, j, Matrix3x3::Zero());  // 非对角项置零
                           })
                    .wait();  // 等待GPU操作完成
            }

            // 3.6 构建对角Hessian矩阵Hii（初始化+原子累加）
            muda::DeviceBuffer<Matrix3x3> d_Hii;
            {
                Timer timer{"Build diagonal Hii (resize + fill + atomic add)"};
                d_Hii.resize((int)vertex_size);
                d_Hii.view().fill(Matrix3x3::Zero());  // 初始化Hii为零

                // 并行累加对角项到Hii
                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply(info.hessians().triplet_count(),
                           [hessians = info.hessians().viewer().name("hessians"),
                            Hii = d_Hii.viewer().name("Hii")] __device__(int I)
                           {
                               auto&& [i, j, H3] = hessians(I).read();
                               if(i == j)  // 仅处理对角项
                               {
                                   // 原子累加3x3矩阵（适配float类型）
                                   atomic_add(&Hii(i)(0, 0), H3(0, 0));
                                   atomic_add(&Hii(i)(0, 1), H3(0, 1));
                                   atomic_add(&Hii(i)(0, 2), H3(0, 2));
                                   atomic_add(&Hii(i)(1, 0), H3(1, 0));
                                   atomic_add(&Hii(i)(1, 1), H3(1, 1));
                                   atomic_add(&Hii(i)(1, 2), H3(1, 2));
                                   atomic_add(&Hii(i)(2, 0), H3(2, 0));
                                   atomic_add(&Hii(i)(2, 1), H3(2, 1));
                                   atomic_add(&Hii(i)(2, 2), H3(2, 2));
                               }
                           })
                    .wait();
            }

            // 3.7 并行计算更新方向并更新数组（当前颜色组）
            {
                Timer timer{"Compute descent & update arrays (GPU ParallelFor)"};
                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply((int)d_color_vertices.size(),
                           [verts = d_color_vertices.cviewer().name("color_vertices"),
                            is_fixed = fem().is_fixed.cviewer().name("is_fixed"),
                            gradients = info.gradients().cviewer().name("gradients"),
                            Hii = d_Hii.cviewer().name("Hii"),
                            xupd = d_x_update_global.viewer().name("xupd"),  // 对应x_update_h_3v
                            xpos = d_x_update_pos.viewer().name("xpos"),  // 对应x_update_h_global
                            xs          = fem().xs.viewer().name("xs"),
                            xs_previous = xs_previous_gpu.viewer().name(
                                "xs_previous")] __device__(int k)  // k：当前组内顶点索引
                           {
                               const int v = (int)verts(k);  // 全局顶点ID
                               if(v < 0 || is_fixed(v))
                                   return;  // 跳过固定顶点

                               // 提取梯度并计算力
                               const Vector3 G = gradients.segment<3>(v * 3).as_eigen();
                               Vector3 force = -G;

                               // 提取对角Hessian（Hii）
                               Matrix3x3 H = Hii(v);
                               if(H.isZero())
                                   H = Matrix3x3::Identity() * 1e-6f;  // 避免奇异矩阵

                               // 求解3x3线性系统（获取更新方向）
                               Float m[9]  = {H(0, 0),
                                              H(1, 0),
                                              H(2, 0),
                                              H(0, 1),
                                              H(1, 1),
                                              H(2, 1),
                                              H(0, 2),
                                              H(1, 2),
                                              H(2, 2)};
                               Float b[3]  = {force(0), force(1), force(2)};
                               Float dx[3] = {0, 0, 0};
                               bool  solver_success =
                                   solve3x3_psd_stable<Float>(m, b, dx);

                               Vector3 descent;
                               if(!solver_success)
                               {
                                   descent = force;  // 求解失败时用 force 作为方向
                                   printf("Solver failed at vertex %d (GPU)\n", v);
                               }
                               else
                               {
                                   descent = Vector3(dx[0], dx[1], dx[2]);
                               }

                               // 更新两个全局数组
                               xupd(v) -= descent * 2;  // 对应CPU的x_update_h_3v
                               xpos(v) += descent;  // 对应CPU的x_update_h_global

                               // 实时更新顶点位置（VBD迭代依赖）
                               xs(v) = xs_previous(v) + xpos(v);
                           })
                    .wait();  // 等待当前组计算完成
            }
        }

        // 4. 最终同步结果到info结构（GPU内并行操作）
        {
            Timer timer{"Sync results to info (x_update + xs_position)"};
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(fem().xs.size(),
                       [d_x_update_global = d_x_update_global.viewer().name("d_x_update_global"),
                        d_x_update_pos = d_x_update_pos.viewer().name("d_x_update_pos"),
                        x_update = info.x_update().viewer().name("x_update"),
                        xs_position = info.xs_position().viewer().name("xs_position"),
                        xs_previous = xs_previous_gpu.viewer().name("xs_previous"),
                        xs = fem().xs.viewer().name("xs")] __device__(int v)
                       {
                           // 1. 同步x_update（扁平float数组）
                           const Vector3& upd  = d_x_update_global(v);
                           x_update(v * 3 + 0) = upd.x();
                           x_update(v * 3 + 1) = upd.y();
                           x_update(v * 3 + 2) = upd.z();

                           // 2. 同步xs_position（初始位置的扁平数组）
                           const Vector3& prev_pos = xs_previous(v);
                           xs_position(v * 3 + 0)  = prev_pos.x();
                           xs_position(v * 3 + 1)  = prev_pos.y();
                           xs_position(v * 3 + 2)  = prev_pos.z();

                           // 3. 恢复fem().xs为初始位置（仅保留更新方向，不修改原始位置）
                           xs(v) = xs_previous(v);
                       })
                .wait();  // 等待所有同步操作完成
        }
    }

    //bool use_vertex_coloring = true;
    //if(use_vertex_coloring && vertex_group.empty())
    //    vertices_Coloring();

    //// 总函数时间计时
    //Timer totalTimer{"Total solve_system_vertex time"};

    //using namespace muda;
    //// 变量声明
    //auto                        vertex_size = fem().xs.size();
    //auto                        info_x_size = info.x_update().size();
    //muda::DeviceBuffer<Vector3> xs_previous_gpu;
    //// 1. 初始化向量（内存分配+数据拷贝）
    //{
    //    Timer timer{"Initialize vectors (xs_previous)"};
    //    xs_previous_gpu = fem().xs;  // 直接拷贝到GPU
    //}

    //// 顶点循环整体计时
    //{
    //    Timer vertexLoopTimer{"Total vertex loop time (all vertices)"};
    //    {
    //        muda::DeviceBuffer<Vector3> d_x_update_global;  // 对应 x_update_h_3v（外部传输用）
    //        muda::DeviceBuffer<Vector3> d_x_update_pos;  // 新增：对应 x_update_h_global（位置更新用）
    //        {
    //            d_x_update_global.resize((int)vertex_size);
    //            d_x_update_global.view().fill(Vector3::Zero());
    //            d_x_update_pos.resize((int)vertex_size);  // 初始化新增缓冲区
    //            d_x_update_pos.view().fill(Vector3::Zero());
    //        }

    //        muda::DeviceBuffer<IndexT> d_color_vertices;
    //        for(size_t iGroup = 0; iGroup < vertex_group.size(); iGroup++)
    //        {
    //            // 2. clear gradient and hessian
    //            {
    //                Timer timer{"Clear Gradient (all vertices)"};
    //                info.gradients().buffer_view().fill(0);
    //                info.hessians().values().fill(Matrix3x3::Zero());
    //            }

    //            auto& parallelGroup = vertex_group[iGroup];
    //            // host 转换为 IndexT 并拷到 GPU
    //            std::vector<IndexT> group_indices(parallelGroup.begin(),
    //                                              parallelGroup.end());
    //            d_color_vertices.resize(static_cast<int>(group_indices.size()));
    //            if(!group_indices.empty())
    //                d_color_vertices.view().copy_from(group_indices.data());
    //            // 3. 组装梯度和海森矩阵（分步骤）
    //            {
    //                Timer timer{"GPU Assemble producers (all vertices)"};
    //                //_assemble_producers(info);
    //                _assemble_producers_by_color(info, d_color_vertices);
    //            }
    //            {
    //                //但是这里的基础的gradient 和 hessian 是之前就计算过了的,再计算也不会变
    //                Timer timer{"Assemble dytopo effect (all vertices)"};
    //                _assemble_dytopo_effect(info);
    //            }
    //            {
    //                Timer timer{"Assemble animation (all vertices)"};
    //                _assemble_animation(info);
    //            }
    //            ////////////////////////===============after this code we do not need to verify first
    //            // 4. 清除固定顶点梯度（并行操作）
    //            {
    //                Timer timer{"Clear Fixed Vertex Gradient (ParallelFor, all vertices)"};
    //                ParallelFor()
    //                    .file_line(__FILE__, __LINE__)
    //                    .apply(fem().xs.size(),
    //                           [is_fixed = fem().is_fixed.cviewer().name("is_fixed"),
    //                            gradients = info.gradients().viewer().name(
    //                                "gradients")] __device__(int i) mutable
    //                           {
    //                               if(is_fixed(i))
    //                               {
    //                                   gradients.segment<3>(i * 3).as_eigen().setZero();
    //                               }
    //                           });
    //            }

    //            // 5. 清除固定顶点海森矩阵（并行操作）
    //            {
    //                Timer timer{"Clear Fixed Vertex hessian (ParallelFor, all vertices)"};
    //                ParallelFor()
    //                    .file_line(__FILE__, __LINE__)
    //                    .apply(info.hessians().triplet_count(),
    //                           [is_fixed = fem().is_fixed.cviewer().name("is_fixed"),
    //                            hessians = info.hessians().viewer().name(
    //                                "hessians")] __device__(int I) mutable
    //                           {
    //                               auto&& [i, j, H3] = hessians(I).read();

    //                               if(is_fixed(i) || is_fixed(j))
    //                               {
    //                                   if(i != j)
    //                                       hessians(I).write(i, j, Matrix3x3::Zero());
    //                               }
    //                           })
    //                    .wait();
    //            }

    //            //////////================================GPU 遍历这些color 中的顶点:
    //            // 1) 构建对角 H(ii)（GPU端），避免回传 triplets
    //            muda::DeviceBuffer<Matrix3x3> d_Hii;
    //            d_Hii.resize((int)vertex_size);
    //            d_Hii.view().fill(Matrix3x3::Zero());

    //            ParallelFor()
    //                .file_line(__FILE__, __LINE__)
    //                .apply(info.hessians().triplet_count(),
    //                       [hessians = info.hessians().viewer().name("hessians"),
    //                        Hii = d_Hii.viewer().name("Hii")] __device__(int I) mutable
    //                       {
    //                           auto&& [i, j, H3] = hessians(I).read();
    //                           if(i == j)
    //                           {
    //                               // 原子累加 3x3 到 Hii(i)
    //                               // 注意：若 Float 为 float 则 atomicAdd 可用；若为 double 请改为分块规约或使用自定义原子
    //                               muda::atomic_add(&Hii(i)(0, 0), H3(0, 0));
    //                               muda::atomic_add(&Hii(i)(0, 1), H3(0, 1));
    //                               muda::atomic_add(&Hii(i)(0, 2), H3(0, 2));
    //                               muda::atomic_add(&Hii(i)(1, 0), H3(1, 0));
    //                               muda::atomic_add(&Hii(i)(1, 1), H3(1, 1));
    //                               muda::atomic_add(&Hii(i)(1, 2), H3(1, 2));
    //                               muda::atomic_add(&Hii(i)(2, 0), H3(2, 0));
    //                               muda::atomic_add(&Hii(i)(2, 1), H3(2, 1));
    //                               muda::atomic_add(&Hii(i)(2, 2), H3(2, 2));
    //                           }
    //                       })
    //                .wait();
    //            /////////////=============这里是进一步计算更新方向的问题，其他先不用真的更新gpu中的数据
    //            // 2) 对本颜色组的每个顶点并行：计算后同时更新两个数组
    //            ParallelFor()
    //                .file_line(__FILE__, __LINE__)
    //                .apply((int)d_color_vertices.size(),
    //                       [verts = d_color_vertices.cviewer().name("color_vertices"),
    //                        is_fixed = fem().is_fixed.cviewer().name("is_fixed"),
    //                        gradients = info.gradients().cviewer().name("gradients"),
    //                        Hii = d_Hii.cviewer().name("Hii"),
    //                        xupd = d_x_update_global.viewer().name("xupd"),  // 对应 x_update_h_3v
    //                        xpos = d_x_update_pos.viewer().name("xpos"),  // 对应 x_update_h_global
    //                        xs          = fem().xs.viewer().name("xs"),
    //                        xs_previous = xs_previous_gpu.viewer().name(
    //                            "xs_previous")] __device__(int k) mutable  // 用于实时更新顶点位置
    //                       {
    //                           const int v = (int)verts(k);
    //                           if(v < 0 || is_fixed(v))
    //                               return;

    //                           // 1. 计算 descent（原有逻辑不变）
    //                           const Vector3 G = gradients.segment<3>(v * 3).as_eigen();
    //                           Vector3   force = -G;
    //                           Matrix3x3 H     = Hii(v);
    //                           if(H.isZero())
    //                               H = Matrix3x3::Identity() * 1e-6f;
    //                           Float m[9]  = {H(0, 0),
    //                                          H(1, 0),
    //                                          H(2, 0),
    //                                          H(0, 1),
    //                                          H(1, 1),
    //                                          H(2, 1),
    //                                          H(0, 2),
    //                                          H(1, 2),
    //                                          H(2, 2)};
    //                           Float b[3]  = {force(0), force(1), force(2)};
    //                           Float dx[3] = {0, 0, 0};

    //                           bool solver_success =
    //                               solve3x3_psd_stable<Float>(m, b, dx);
    //                           Vector3 descent;
    //                           if(!solver_success)
    //                           {
    //                               descent = force;  // 与CPU一致：失败时用force作为方向
    //                               printf("Solver failed at vertex %d (GPU)\n", v);
    //                           }
    //                           else
    //                           {
    //                               descent = Vector3(dx[0], dx[1], dx[2]);
    //                           }

    //                           // 2. 同步更新两个数组（关键修改）
    //                           // 对应 CPU 的 x_update_h_3v：-= descent*2（等价于 += (-descent*2)）
    //                           xupd(v) -= descent * 2;
    //                           // 对应 CPU 的 x_update_h_global：+= descent
    //                           xpos(v) += descent;
    //                           //// 3. 实时更新顶点位置（延续之前的中间更新逻辑，基于 xpos 的累积）
    //                           xs(v) = xs_previous(v) + xpos(v);  // 注意：xs_previous 需是初始位置的设备副本
    //                       })
    //                .wait();
    //            int StopColorEnd = 0;
    //        }

    //        int endEachGroup = 0;
    //        ///////////==============================gpu 版本的更新逻辑
    //        ParallelFor()
    //            .file_line(__FILE__, __LINE__)
    //            .apply(fem().xs.size(),  // 按顶点数量启动线程，每个线程处理一个顶点v
    //                   [d_x_update_global = d_x_update_global.viewer().name("d_x_update_global"),
    //                    d_x_update_pos = d_x_update_pos.viewer().name("d_x_update_pos"),
    //                    x_update = info.x_update().viewer().name("x_update"),  // 扁平float数组
    //                    xs_position = info.xs_position().viewer().name("xs_position"),  // 扁平float数组
    //                    xs_previous = xs_previous_gpu.viewer().name("xs_previous"),
    //                    xs = fem().xs.viewer().name("xs")]  // 顶点位置缓冲区
    //                   __device__(int v) mutable  // v：当前处理的顶点索引
    //                   {
    //                       // 1. 扁平化d_x_update_global到info.x_update()（对应CPU的x_update_h_3v）
    //                       const Vector3& upd = d_x_update_global(v);  // 取当前顶点的更新量（Vector3）
    //                       x_update(v * 3 + 0) = upd.x();  // 存储x分量
    //                       x_update(v * 3 + 1) = upd.y();  // 存储y分量
    //                       x_update(v * 3 + 2) = upd.z();  // 存储z分量

    //                       // 2. 扁平化xs_previous到info.xs_position()（对应CPU的xs_update_3v）
    //                       const Vector3& prev_pos = xs_previous(v);  // 取当前顶点的历史位置
    //                       xs_position(v * 3 + 0) = prev_pos.x();
    //                       xs_position(v * 3 + 1) = prev_pos.y();
    //                       xs_position(v * 3 + 2) = prev_pos.z();

    //                       // 3. 更新fem().xs为：历史位置 + 位置更新量（对应CPU的fem().xs.copy_from）
    //                       xs(v) = xs_previous(v);
    //                   })
    //            .wait();  // 等待所有线程完成
    //    }
    //}
}

void FEMLinearSubsystem::Impl::update_info(GlobalLinearSystem::DiagInfo& info)
{
    IndexT               vertex_size = fem().xs.size();
    auto                 info_x_size = info.x_update().size();
    auto                 xs_size     = fem().xs.size();
    std::vector<Float>   dxs_update_h_3v;
    std::vector<Float>   xs_update_3v;
    std::vector<Vector3> x_update_h_global;
    std::vector<Vector3> xs_initial;
    std::vector<Vector3> dxs_initial;
    // 1. 初始化向量（内存分配+数据拷贝）
    {
        Timer timer{"Initialize vectors (x_update_h_3v, x_update_h_global, xs_previous)"};
        dxs_update_h_3v.resize(info_x_size);
        xs_update_3v.resize(info_x_size);

        xs_initial.resize(xs_size);
        dxs_initial.resize(xs_size);
        fem().xs.copy_to(xs_initial);
        fem().dxs.copy_to(dxs_initial);
    }
    {
        Timer timer{"Copy x_update_h_3v to GPU (info.x_update())"};
        fem().xs.copy_from(xs_initial);
        fem().dxs.copy_from(dxs_initial);
        for(int vertexId = 0; vertexId < vertex_size; ++vertexId)
        {
            for(int k = 0; k < 3; ++k)
            {
                xs_update_3v[vertexId * 3 + k] = xs_initial[vertexId][k];
                dxs_update_h_3v[vertexId * 3 + k] = dxs_initial[vertexId][k];
            }
        }
        info.xs_position().buffer_view().copy_from(xs_update_3v.data());
        info.x_update().buffer_view().copy_from(dxs_update_h_3v.data());
    }
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

void FEMLinearSubsystem::Impl::_assemble_producers_by_vertex(GlobalLinearSystem::DiagInfo& info, IndexT vertexId)
{
    FiniteElementEnergyProducer::AssemblyInfo assembly_info;
    assembly_info.hessians = info.hessians().subview(energy_producer_hessian_offset,
                                                     energy_producer_hessian_count);
    assembly_info.dt = dt;

    for(auto& producer : fem().energy_producers)
    {
        producer->assemble_gradient_hessian_by_vertex(assembly_info, vertexId);
        //producer->assemble_gradient_hessian(assembly_info);

        //auto& info_fem = producer->m_impl.finite_element_method;
        ////// 这里需要构造一个只包含当前vertexId的info

        ////producer->m_impl.finite_element_method->is_fixed();
        //////ComputeGradientHessianInfo this_info{info.dt,
        //////    global_gradient_view.subview(m_impl.gradient_offset, m_impl.gradient_count),
        //////    info.hessians.subview(m_impl.hessian_offset, m_impl.hessian_count)};

        //auto global_gradient_view = producer->m_impl.finite_element_method->m_impl.energy_producer_gradients.view();
        //auto subview_gradient = global_gradient_view.subview(producer->m_impl.gradient_offset,
        //                                 producer->m_impl.gradient_count);
        //subview_gradient.viewer();
        //using namespace muda;
        ////info_fem.gradient();
        //ParallelFor()
        //    .file_line(__FILE__, __LINE__)
        //    .apply(info_fem->xs().size(),
        //           [is_fixed = info_fem->is_fixed().cviewer().name("is_fixed"),
        //            xs       = info_fem->xs().cviewer().name("xs"),
        //            x_tildes = info_fem->x_tildes().viewer().name("x_tildes"),
        //            masses   = info_fem->masses().cviewer().name("masses"),
        //            G3s      = subview_gradient.viewer().name("G3s"),
        //            H3x3s = assembly_info.hessians.viewer().name("H3x3s")] __device__(int i) mutable
        //           {
        //               auto& m       = masses(i);
        //               auto& x       = xs(i);
        //               auto& x_tilde = x_tildes(i);

        //               Vector3   G;
        //               Matrix3x3 H;

        //               if(is_fixed(i))  // fixed
        //               {
        //                   G = Vector3::Zero();
        //               }
        //               else
        //               {
        //                   G = m * (x - x_tilde);
        //               }

        //               H = masses(i) * Matrix3x3::Identity();

        //               G3s(i).write(i, G);
        //               H3x3s(i).write(i, i, H);
        //           });
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

void FEMLinearSubsystem::Impl::_assemble_producers_by_color(GlobalLinearSystem::DiagInfo& info, muda::CBufferView<IndexT> color_vertices)
        {
    FiniteElementEnergyProducer::AssemblyInfo assembly_info;
    assembly_info.hessians = info.hessians().subview(energy_producer_hessian_offset,
                                                     energy_producer_hessian_count);
    assembly_info.dt = dt;

    for(auto& producer : fem().energy_producers)
    {
        //producer->assemble_gradient_hessian_by_vertex(assembly_info, vertexId);
        //oducer->assemble_gradient_hessian(assembly_info);
        producer->assemble_gradient_hessian_by_color(assembly_info, color_vertices);
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

    ///////////===========================================
    //std::vector<Float> gpu_gradients;  // 保存GPU计算的梯度
    //{
    //    Timer timer{"Save GPU gradients"};
    //    gpu_gradients.resize(info.gradients().size());
    //    info.gradients().buffer_view().copy_to(gpu_gradients.data());  // 从GPU拷贝到CPU
    //}
    //// 对比梯度结果
    //bool        gradients_match = true;
    //const Float eps             = 1e-6f;  // 浮点误差容忍度
    //for(size_t i = 0; i < gpu_gradients.size(); ++i)
    //{
    //    gradients_match = false;
    //    // 输出差异位置和数值（方便定位）
    //    int vertexId = i / 3;  // 假设梯度按顶点存储，每个顶点3个分量
    //    int comp     = i % 3;
    //    std::cout << "Gradient mismatch at vertex " << vertexId
    //              << ", component " << comp << ": "
    //              << "GPU=" << gpu_gradients[i] << std::endl;
    //}
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
            int testCollision = 0;
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
    m_impl.solve_system_vertex_cpu(info);
    //m_impl.solve_system_vertex_gpu(info);
    //impl.solve_system_vertex_test(info);
}

void FEMLinearSubsystem::do_update_info(GlobalLinearSystem::DiagInfo& info)
{
    m_impl.update_info(info);
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


        /////////////////////////========================================///cpu side  vertex loop
//{
//    for(int vertexId = 0; vertexId < vertex_size; ++vertexId)
//    {
//        // 2. 清除梯度 和hessian????
//        {
//            Timer timer{"Clear Gradient (all vertices)"};
//            info.gradients().buffer_view().fill(0);
//            //fill hessians also with matrix 0
//            info.hessians().values().fill(Matrix3x3::Zero());
//        }

//        // 3. 组装梯度和海森矩阵（分步骤）
//        {
//            Timer timer{"Assemble producers (all vertices)"};
//            //_assemble_producers(info);
//            _assemble_producers_by_vertex(info, vertexId);
//        }
//        {
//            //但是这里的基础的gradient 和 hessian 是之前就计算过了的,再计算也不会变
//            Timer timer{"Assemble dytopo effect (all vertices)"};
//            _assemble_dytopo_effect(info);
//        }
//        {
//            Timer timer{"Assemble animation (all vertices)"};
//            _assemble_animation(info);
//        }

//        // 4. 清除固定顶点梯度（并行操作）
//        {
//            Timer timer{"Clear Fixed Vertex Gradient (ParallelFor, all vertices)"};
//            ParallelFor()
//                .file_line(__FILE__, __LINE__)
//                .apply(fem().xs.size(),
//                       [is_fixed = fem().is_fixed.cviewer().name("is_fixed"),
//                        gradients = info.gradients().viewer().name(
//                            "gradients")] __device__(int i) mutable
//                       {
//                           if(is_fixed(i))
//                           {
//                               gradients.segment<3>(i * 3).as_eigen().setZero();
//                           }
//                       });
//        }

//        // 5. 清除固定顶点海森矩阵（并行操作）
//        {
//            Timer timer{"Clear Fixed Vertex hessian (ParallelFor, all vertices)"};
//            ParallelFor()
//                .file_line(__FILE__, __LINE__)
//                .apply(info.hessians().triplet_count(),
//                       [is_fixed = fem().is_fixed.cviewer().name("is_fixed"),
//                        hessians = info.hessians().viewer().name(
//                            "hessians")] __device__(int I) mutable
//                       {
//                           auto&& [i, j, H3] = hessians(I).read();

//                           if(is_fixed(i) || is_fixed(j))
//                           {
//                               if(i != j)
//                                   hessians(I).write(i, j, Matrix3x3::Zero());
//                           }
//                       })
//                .wait();
//        }

//        // 6. 提取梯度数据到CPU并计算力
//        std::vector<Vector3> force_h;
//        std::vector<Float>   gradients_h;
//        int                  gradients_size;
//        {
//            Timer timer{"Extract gradients to CPU and compute force (all vertices)"};
//            force_h.resize(vertex_size);
//            gradients_size = info.gradients().size();
//            gradients_h.resize(gradients_size);
//            info.gradients().buffer_view().copy_to(gradients_h.data());

//            // 提取当前顶点梯度并计算力
//            Vector3 gradient = get_vertex_gradient(gradients_h, vertexId);
//            force_h[vertexId] = -gradient;
//        }

//        // 7. 提取海森矩阵三元组到CPU
//        IndexT                 triplet_size;
//        std::vector<IndexT>    host_rows;
//        std::vector<IndexT>    host_cols;
//        std::vector<Matrix3x3> host_values;
//        int                    hessian_size;
//        {
//            Timer timer{"Extract Hessian triplets to CPU (all vertices)"};
//            triplet_size = info.hessians().row_indices().size();
//            host_rows.resize(triplet_size);
//            host_cols.resize(triplet_size);
//            host_values.resize(triplet_size);
//            info.hessians().row_indices().copy_to(host_rows.data());
//            info.hessians().col_indices().copy_to(host_cols.data());
//            info.hessians().values().copy_to(host_values.data());
//            hessian_size = info.hessians().total_triplet_count();
//        }

//        // 8. 遍历三元组获取当前顶点海森矩阵
//        Matrix3x3 h = Matrix3x3::Zero();
//        {
//            Timer timer{"Loop through triplets to get hessian h (all vertices)"};
//            for(int I = 0; I < triplet_size; ++I)
//            {
//                int i_vertex = host_rows[I];
//                int j_vertex = host_cols[I];
//                if(i_vertex == vertexId && j_vertex == vertexId)
//                {
//                    h += host_values[I];
//                }
//            }
//        }

//        // 9. 求解3x3矩阵及相关处理
//        {
//            Timer  timer{"Solve 3x3 PSD system (all vertices)"};
//            auto&  force     = force_h[vertexId];
//            double ForceNorm = force.squaredNorm();

//            if(1)  // 保留原条件
//            {
//                if(force.isZero())
//                {
//                    continue;
//                }
//                if(h.isZero())
//                {
//                    h = Matrix3x3::Identity() * 1e-6;
//                    continue;
//                }

//                Vector3 descentDirection;
//                Float   stepSize               = 1;
//                Float   lineSearchShrinkFactor = 0.8;
//                bool    solverSuccess;

//                bool useDouble3x3 = 1;
//                if(useDouble3x3)
//                {
//                    double H[9] = {h(0, 0),
//                                   h(1, 0),
//                                   h(2, 0),
//                                   h(0, 1),
//                                   h(1, 1),
//                                   h(2, 1),
//                                   h(0, 2),
//                                   h(1, 2),
//                                   h(2, 2)};

//                    double F[3]      = {force(0), force(1), force(2)};
//                    double dx[3]     = {0, 0, 0};
//                    solverSuccess    = solve3x3_psd_stable(H, F, dx);
//                    descentDirection = Vector3(dx[0], dx[1], dx[2]);

//                    // 验证计算
//                    auto   TestOuput = h * descentDirection;
//                    auto   diff      = TestOuput - force;
//                    double diff_norm = diff.norm();
//                    if(diff_norm > 1e-6)
//                    {
//                        std::cout << "Warning: h * descentDirection does not match force (diff_norm = "
//                                  << diff_norm << ")" << std::endl;
//                    }
//                }
//                else
//                {
//                    solverSuccess = false;  // 未使用分支
//                }

//                // 处理求解失败
//                if(!solverSuccess)
//                {
//                    stepSize               = 1;
//                    descentDirection       = force;
//                    lineSearchShrinkFactor = 0.8;
//                    std::cout << "Solver failed at vertex " << vertexId
//                              << std::endl;
//                }

//                // 检查数值异常
//                if(descentDirection.hasNaN())
//                {
//                    std::cout << "force: " << force.transpose() << "\nHessian:\n"
//                              << h;
//                    std::cout << "descentDirection has NaN at vertex "
//                              << vertexId << std::endl;
//                    std::exit(-1);
//                }

//                // 10. 更新x_update数组
//                {
//                    Timer timer{"Update x_update arrays (all vertices)"};
//                    for(int k = 0; k < 3; ++k)
//                    {
//                        //x_update_h_3v[vertexId * 3 + k] -= descentDirection[k];
//                        x_update_h_3v[vertexId * 3 + k] -= descentDirection[k] * 2;
//                    }
//                    x_update_h_global[vertexId] += descentDirection;
//                }

//                // 11. 更新顶点位置
//                {
//                    Timer timer{"Update xs_temp and copy to fem().xs (all vertices)"};
//                    std::vector<Vector3> xs_temp(xs_previous.size());
//                    for(size_t i = 0; i < xs_previous.size(); ++i)
//                    {
//                        xs_temp[i] = xs_previous[i] + x_update_h_global[i];
//                    }
//                    xs_temp_global = xs_temp;
//                    fem().xs.copy_from(xs_temp);
//                }
//            }
//        }
//    }
//}