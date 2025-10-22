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

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(FEMLinearSubsystem);

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

    if(abs(det) < 1e-4 * (abs(a11 * i11) + abs(a21 * i12) + abs(a31 * i13)))
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

void FEMLinearSubsystem::Impl::solve_system_vertex(GlobalLinearSystem::DiagInfo& info)
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
    //////##############这里一定是能改的，就是哪里的更新的有问题????????????????
    // 遍历所有顶点
    auto N = fem().xs.size();
    // 1. 定义 host 端的 vector（类型与 DeviceBuffer 一致，这里是 bool）
    std::vector<IndexT> is_fixed_h;  // 无需提前 resize，copy_to 会自动处理
    // 2. 调用 copy_to 复制设备数据到 host
    fem().is_fixed.copy_to(is_fixed_h);

    //std::vector<Float> gradient_h, hessian_h;
    //info.gradients().buffer_view().copy_to(gradient_h.data());
    //info.hessians();
    //IndexT;
    //Float;
    //Float;
    //fem().is_fixed.copy_to(is_fixed_h);
    IndexT               vertex_size = fem().xs.size();
    std::vector<Vector3> force_h(vertex_size);
    for(int vertexId = 0; vertexId < vertex_size; ++vertexId)
    {
        // 跳过固定顶点（已在梯度/海森中处理，无需更新）
        if(is_fixed_h[vertexId])
        {
            continue;
        }

        // 后续步骤：提取该顶点的力和海森矩阵...
        // 提取该顶点的梯度（3D向量）
        //auto gradient_h = info.gradients().viewer().as_eigen();
        //auto gradient_h = info.gradients().subview(1,3).viewer().as_eigen();

        //auto gradients_vertexId  = info.gradients()
        //                .viewer()
        //                .segment<3>(vertexId * 3)  // 每个顶点3个自由度，偏移量为vertexId*3
        //                .as_eigen();  // 转换为Eigen向量（假设支持as_eigen()）

        //gradients.segment<3>(i * 3).as_eigen().setZero();
        // 计算力：force = -gradient（能量的负梯度）

        std::vector<Float> gradients_h;
        int                gradients_size = info.gradients().size();
        gradients_h.resize(gradients_size);
        info.gradients().buffer_view().copy_to(gradients_h.data());
        // 提取该顶点的梯度（CPU 端，对应 GPU 的 segment<3>）
        Vector3 gradient = get_vertex_gradient(gradients_h, vertexId);
        // 直接用cout输出
        //std::cout << "gradient: " << gradient << std::endl; 

        // 力 = -梯度
        force_h[vertexId] = -gradient;
        // （可选）如果有额外外力，需叠加（类似示例中的external force）
        // force += pMesh->vertexExternalForces.col(vertexId);  // 根据你的网格接口调整
        Matrix3x3 h = Matrix3x3::Zero();  // 初始化3x3海森矩阵
        //Eigen::Matrix3f h = Eigen::Matrix3f::Zero();  // 初始化3x3海森矩阵

        // 假设 info.hessians() 是 TripletMatrixView<float, 3, 3> 类型
        //auto& hessian_view = info.hessians().viewer().;

        // 复制行索引、列索引、值到CPU端vector
        IndexT              triplet_size = info.hessians().row_indices().size();
        std::vector<IndexT> host_rows;
        std::vector<IndexT> host_cols;
        std::vector<Matrix3x3> host_values;  // 3x3块值
        host_rows.resize(triplet_size);
        host_cols.resize(triplet_size);
        host_values.resize(triplet_size);
        info.hessians().row_indices().copy_to(host_rows.data());
        info.hessians().col_indices().copy_to(host_cols.data());
        info.hessians().values().copy_to(host_values.data());

        // 遍历CPU端的三元组数据（此时所有数据都在CPU内存中）
        for(int I = 0; I < host_rows.size(); ++I)
        {
            int  i  = host_rows[I];    // 行索引（CPU端访问，安全）
            int  j  = host_cols[I];    // 列索引
            auto H3 = host_values[I];  // 3x3子块
            // 处理[i, j, H3]...
            // 只关注当前顶点的对角块（i=j=vertexId）
            if(i == vertexId && j == vertexId)
            {
                h = H3;  // 提取对角块作为局部海森矩阵
                break;   // 假设每个顶点只有一个对角块，找到后退出
            }
        }

        //// 遍历海森矩阵的三元组，找到当前顶点的对角块
        //for(int I = 0; I < info.hessians().triplet_count(); ++I)
        //{
        //    auto&& [i, j, H3] = info.hessians().viewer()(I).read();  // 读取三元组(i, j, 3x3子块)

        //    // 只关注当前顶点的对角块（i=j=vertexId）
        //    if(i == vertexId && j == vertexId)
        //    {
        //        h = H3;  // 提取对角块作为局部海森矩阵
        //        break;   // 假设每个顶点只有一个对角块，找到后退出
        //    }
        //}

        // （可选）如果未找到对角块，可初始化一个默认矩阵（如单位矩阵，避免奇异）
        if(h.isZero())
        {
            h = Matrix3x3::Identity() * 1e-6;  // 微小值避免求解失败
        }

        auto& force = force_h[vertexId];
        // 检查力是否足够大（避免微小力的无效更新）
        if(force.squaredNorm() > 1e-4)
        {
            Vector3 descentDirection;
            Float   stepSize = 1;
            //Float lineSearchShrinkFactor = tau;
            Float lineSearchShrinkFactor = 0.8;
            bool  solverSuccess;

            bool useDouble3x3 = 1;
            // 调用3x3 PSD矩阵求解器（与示例一致）
            if(useDouble3x3)
            {
                // 转换为double数组（适配求解器接口）
                double H[9] = {
                    h(0, 0), h(1, 0), h(2, 0), h(0, 1), h(1, 1), h(2, 1), h(0, 2), h(1, 2), h(2, 2)};
                double F[3]      = {force(0), force(1), force(2)};
                double dx[3]     = {0, 0, 0};
                solverSuccess    = solve3x3_psd_stable(H, F, dx);
                descentDirection = Vector3(dx[0], dx[1], dx[2]);
            }
            else
            {
                //force
                //solverSuccess = solve3x3_psd_stable(h.data(), force.data(), descentDirection.data());
            }

            // 处理求解失败（如矩阵奇异），退化为梯度下降
            if(!solverSuccess)
            {
                //stepSize         = stepSizeGD;  // 梯度下降步长
                stepSize         = 1;      // 梯度下降步长
                descentDirection = force;  // 直接用力（负梯度）作为方向
                //lineSearchShrinkFactor = physicsParams().tau_GD;
                lineSearchShrinkFactor = 0.8;
                std::cout << "Solver failed at vertex " << vertexId << std::endl;
            }

            // 检查数值异常（NaN）
            if(descentDirection.hasNaN())
            {
                std::cout << "force: " << force.transpose() << "\nHessian:\n"
                          << h;
                std::cout << "descentDirection has NaN at vertex " << vertexId << std::endl;
                std::exit(-1);
            }
            //////顶点信息更新， 实际上之后使用retrive就可以了

            //Vector3             host_data = Vector3::Zero();
            descentDirection = stepSize * descentDirection;
            // 1. Copy x_update to a host vector before the call
            std::vector<Float> x_update_before(info.x_update().size());
            info.x_update().buffer_view().copy_to(x_update_before.data());

            //#####################################################
            // 2. Call the function that may modify x_update
            // 1. 从 GPU 拷贝到 CPU
            std::vector<Float> x_update_h(info.x_update().size());
            info.x_update().buffer_view().copy_to(x_update_h.data());

            // 2. 在 CPU 端做加法
            for(int k = 0; k < 3; ++k)
            {
                x_update_h[vertexId * 3 + k] += descentDirection[k];
            }

            // 3. 同步回 GPU
            info.x_update().buffer_view().copy_from(x_update_h.data());

            //#####################################################
            // 3. Copy x_update to another host vector after the call
            std::vector<Float> x_update_after(info.x_update().size());
            info.x_update().buffer_view().copy_to(x_update_after.data());

            // 4. Compare the two vectors
            bool modified = !std::equal(x_update_before.begin(),
                                        x_update_before.end(),
                                        x_update_after.begin());

            if(modified)
                std::cout << "x_update has been modified." << std::endl;
            else
                std::cout << "x_update has not been modified." << std::endl;


        }
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

    //auto dxs = fem().dxs.view();
    //ParallelFor()
    //    .file_line(__FILE__, __LINE__)
    //    .apply(fem().xs.size(),
    //           [dxs = dxs.viewer().name("dxs"),
    //            result = info.solution().viewer().name("result")] __device__(int i) mutable
    //           {
    //               dxs(i) = -result.segment<3>(i * 3).as_eigen();

    //               // cout << "solution dx(" << i << "):" << dxs(i).transpose().eval() << "\n";
    //           });
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
