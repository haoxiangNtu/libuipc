#include <sim_engine.h>
#include <uipc/common/range.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
#include <contact_system/global_contact_manager.h>
#include <collision_detection/global_trajectory_filter.h>
#include <line_search/line_searcher.h>
#include <linear_system/global_linear_system.h>
#include <animator/global_animator.h>
#include <diff_sim/global_diff_sim_manager.h>
#include <newton_tolerance/newton_tolerance_manager.h>
#include <time_integrator/time_integrator_manager.h>


#include <uipc/uipc.h>

namespace uipc::backend::cuda
{
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
std::vector<IndexT> newton_iters_record;
std::vector<Float> Iters_energy;
std::vector<Float>  Iters_line_search;
void SimEngine::do_advance()
{
    Float alpha     = 1.0;
    Float ccd_alpha = 1.0;
    Float cfl_alpha = 1.0;

    /***************************************************************************************
    *                                  Function Shortcuts
    ***************************************************************************************/

    auto detect_dcd_candidates = [this]
    {
        if(m_global_trajectory_filter)
        {
            Timer timer{"Detect DCD Candidates"};
            m_global_trajectory_filter->detect(0.0);
            m_global_trajectory_filter->filter_active();
        }
    };

    auto detect_trajectory_candidates = [this](Float alpha)
    {
        if(m_global_trajectory_filter)
        {
            Timer timer{"Detect Trajectory Candidates"};
            m_global_trajectory_filter->detect(alpha);
        }
    };

    auto filter_dcd_candidates = [this]
    {
        if(m_global_trajectory_filter)
        {
            Timer timer{"Filter Contact Candidates"};
            m_global_trajectory_filter->filter_active();
        }
    };

    auto record_friction_candidates = [this]
    {
        if(m_global_trajectory_filter && m_friction_enabled)
        {
            m_global_trajectory_filter->record_friction_candidates();
        }
    };

    auto compute_adaptive_kappa = [this]
    {
        // TODO: now no effect
        if(m_global_contact_manager)
            m_global_contact_manager->compute_adaptive_kappa();
    };

    auto compute_dytopo_effect = [this]
    {
        // compute the dytopo effect gradient and hessian, containing:
        // 1) contact effect from contact pairs
        // 2) other dynamic topo effects, e.g. point picker, vertex stitch ...
        if(m_global_dytopo_effect_manager)
        {
            Timer timer{"Compute DyTopo Effect"};
            m_global_dytopo_effect_manager->compute_dytopo_effect();
        }
    };

    auto cfl_condition = [&cfl_alpha, this](Float alpha)
    {
        if(m_global_contact_manager)
        {
            cfl_alpha = m_global_contact_manager->compute_cfl_condition();
            if(cfl_alpha < alpha)
            {
                spdlog::info("CFL Filter: {} < {}", cfl_alpha, alpha);
                return cfl_alpha;
            }
        }

        return alpha;
    };

    auto filter_toi = [&ccd_alpha, this](Float alpha)
    {
        if(m_global_trajectory_filter)
        {
            Timer timer{"Filter CCD TOI"};
            ccd_alpha = m_global_trajectory_filter->filter_toi(alpha);
            if(ccd_alpha < alpha)
            {
                spdlog::info("CCD Filter: {} < {}", ccd_alpha, alpha);
                return ccd_alpha;
            }
        }

        return alpha;
    };

    auto compute_energy = [this, filter_dcd_candidates](Float alpha) -> Float
    {
        // Step Forward => x = x_0 + alpha * dx
        m_global_vertex_manager->step_forward(alpha);
        m_line_searcher->step_forward(alpha);

        // Update the collision pairs
        filter_dcd_candidates();

        // Compute New Energy => E
        return m_line_searcher->compute_energy(false);
    };

    auto step_animation = [this]()
    {
        if(m_global_animator)
        {
            Timer timer{"Step Animation"};
            m_global_animator->step();
        }
    };

    auto compute_animation_substep_ratio = [this](SizeT newton_iter)
    {
        // compute the ratio to the aim position.
        // dst = prev_position + ratio * (position - prev_position)
        if(m_global_animator)
        {
            m_global_animator->compute_substep_ratio(newton_iter);
            spdlog::info("Animation Substep Ratio: {}", m_global_animator->substep_ratio());
        }
    };

    auto animation_reach_target = [this]()
    {
        if(m_global_animator)
        {
            return m_global_animator->substep_ratio() >= 1.0;
        }
        return true;
    };

    auto convergence_check = [&](SizeT newton_iter) -> bool
    {
        if(m_dump_surface->view()[0])
        {
            dump_global_surface(fmt::format("dump_surface.{}.{}", m_current_frame, newton_iter));
        }

        NewtonToleranceManager::ResultInfo result_info;
        result_info.frame(m_current_frame);
        result_info.newton_iter(newton_iter);
        m_newton_tolerance_manager->check(result_info);

        if(!result_info.converged())
            return false;

        // ccd alpha should close to 1.0
        if(ccd_alpha < m_ccd_tol->view()[0])
            return false;

        if(!animation_reach_target())
            return false;

        return true;
    };

    auto update_diff_parm = [this]()
    {
        if(m_global_diff_sim_manager)
        {
            Timer timer{"Update Diff Parm"};
            m_global_diff_sim_manager->update();
        }
    };

    auto check_line_search_iter = [this](SizeT iter)
    {
        if(iter >= m_line_searcher->max_iter())
        {
            spdlog::warn("Line Search Exits with Max Iteration: {} (Frame={})",
                         m_line_searcher->max_iter(),
                         m_current_frame);

            if(m_strict_mode->view()[0])
            {
                throw SimEngineException("StrictMode: Line Search Exits with Max Iteration");
            }
        }
    };

    auto check_newton_iter = [this](SizeT iter)
    {
        if(iter >= m_newton_max_iter->view()[0])
        {
            spdlog::warn("Newton Iteration Exits with Max Iteration: {} (Frame={})",
                         m_newton_max_iter->view()[0],
                         m_current_frame);

            if(m_strict_mode->view()[0])
            {
                throw SimEngineException("StrictMode: Newton Iteration Exits with Max Iteration");
            }
        }
        else
        {
            spdlog::info("Newton Iteration Converged with Iteration Count: {}, Bound: [{}, {}]",
                         iter,
                         m_newton_min_iter->view()[0],
                         m_newton_max_iter->view()[0]);
        }
    };

    //auto solve3x3_psd_stable = []<typename DType>(const DType* m, const DType* b, DType* out)
    //__forceinline__ __device__ __host__ -> bool
    //{
    //    const DType a11 = m[0];
    //    const DType a12 = m[3];
    //    const DType a13 = m[6];
    //    const DType a21 = m[1];
    //    const DType a22 = m[4];
    //    const DType a23 = m[7];
    //    const DType a31 = m[2];
    //    const DType a32 = m[5];
    //    const DType a33 = m[8];
    //    const DType i11 = a33 * a22 - a32 * a23;
    //    const DType i12 = -(a33 * a12 - a32 * a13);
    //    const DType i13 = a23 * a12 - a22 * a13;

    //    const DType det = (a11 * i11 + a21 * i12 + a31 * i13);

    //    if(abs(det) < CMP_EPSILON * (abs(a11 * i11) + abs(a21 * i12) + abs(a31 * i13)))
    //    {
    //        out[0] = b[0];
    //        out[1] = b[1];
    //        out[2] = b[2];
    //        return false;
    //    }

    //    const DType deti = 1.0 / det;

    //    const DType i21 = -(a33 * a21 - a31 * a23);
    //    const DType i22 = a33 * a11 - a31 * a13;
    //    const DType i23 = -(a23 * a11 - a21 * a13);
    //    const DType i31 = a32 * a21 - a31 * a22;
    //    const DType i32 = -(a32 * a11 - a31 * a12);
    //    const DType i33 = a22 * a11 - a21 * a12;

    //    out[0] = deti * (i11 * b[0] + i12 * b[1] + i13 * b[2]);
    //    out[1] = deti * (i21 * b[0] + i22 * b[1] + i23 * b[2]);
    //    out[2] = deti * (i31 * b[0] + i32 * b[1] + i33 * b[2]);
    //    return true;
    //};

    /***************************************************************************************
    *                                  Core Pipeline
    ***************************************************************************************/

    // Abort on exception if the runtime check is enabled for debugging
    constexpr bool AbortOnException = uipc::RUNTIME_CHECK;

    auto pipeline_ipc = [&]() noexcept(AbortOnException)
    {
        Timer timer{"Pipeline"};

        ++m_current_frame;

        spdlog::info(R"(>>> Begin Frame: {})", m_current_frame);

        // Rebuild Scene
        {
            Timer timer{"Rebuild Scene"};
            // Trigger the rebuild_scene event, systems register their actions will be called here
            m_state = SimEngineState::RebuildScene;
            {
                event_rebuild_scene();

                // TODO: rebuild the vertex and surface info
                // m_global_vertex_manager->rebuild_vertex_info();
                // m_global_surface_manager->rebuild_surface_info();
            }

            // After the rebuild_scene event, the pending creation or deletion can be solved
            world().scene().solve_pending();

            // Update the diff parms
            update_diff_parm();
        }

        // Simulation:
        {
            //auto subtimestep_iter = 0;
            //auto subtiemstep_max = 10;
            //for(; subtimestep_iter < subtiemstep_max; ++subtimestep_iter)
            //{
            //}
            Timer timer{"Simulation"};
            // 1. Adaptive Parameter Calculation
            AABB vertex_bounding_box =
                m_global_vertex_manager->compute_vertex_bounding_box();
            detect_dcd_candidates();
            compute_adaptive_kappa();

            // 2. Record Friction Candidates at the beginning of the frame
            record_friction_candidates();
            m_global_vertex_manager->record_prev_positions();

            // 3. Predict Motion => x_tilde = x + v * dt
            m_state = SimEngineState::PredictMotion;
            m_time_integrator_manager->predict_dof();
            step_animation();

            // 4. Nonlinear-Newton Iteration
            Float box_size = vertex_bounding_box.diagonal().norm();
            Float tol      = m_newton_scene_tol * box_size;
            m_newton_tolerance_manager->pre_newton(m_current_frame);

            auto   newton_max_iter = m_newton_max_iter->view()[0];
            auto   newton_min_iter = m_newton_min_iter->view()[0];
            newton_max_iter        = 20;
            IndexT newton_iter     = 0;
            Float  Itres_Energy_each=0;
            for(; newton_iter < newton_max_iter; ++newton_iter)
            {
                Timer timer{"Newton Iteration"};

                // 1) Compute animation substep ratio
                compute_animation_substep_ratio(newton_iter);

                // 2) Build Collision Pairs
                if(newton_iter > 0)
                    detect_dcd_candidates();

                // 3) Compute Dynamic Topo Effect Gradient and Hessian => G:Vector3, H:Matrix3x3
                //    Including Contact Effect
                m_state = SimEngineState::ComputeDyTopoEffect;
                compute_dytopo_effect();

                // 4) Solve Global Linear System => dx = A^-1 * b

                // 2. 调用可能修改 xs_position 和 x_update 的函数
                m_state = SimEngineState::SolveGlobalLinearSystem;
                {
                    Timer timer{"Solve Global Linear System"};
                    //m_global_linear_system->solve();
                    m_global_linear_system->solve_by_vertex();
                }


                // 5) Collect Vertex Displacements Globally, 重新计算更新方向？？？？只要更新方向dxs足够小就叫做收敛??????????
                m_global_vertex_manager->collect_vertex_displacements();


                // 6) Check Termination Condition
                bool converged  = convergence_check(newton_iter);
                bool terminated = converged && (newton_iter >= newton_min_iter);
                auto testCC     = newton_iter;
                if(terminated)
                    break;


                // 7) Begin Line Search
                auto line_search_iter_global = 0;
                m_state = SimEngineState::LineSearch;
                {
                    Timer timer{"Line Search"};

                    // Reset Alpha
                    alpha = 1.0;

                    // Record Current State x to x_0
                    m_line_searcher->record_start_point();
                    m_global_vertex_manager->record_start_point();
                    detect_trajectory_candidates(alpha);

                    // Compute Current Energy => E_0
                    Float E0 = m_line_searcher->compute_energy(true);  // initial energy
                    // spdlog::info("Initial Energy: {}", E0);

                    // CCD filter
                    alpha = filter_toi(alpha);

                    // CFL Condition
                    alpha = cfl_condition(alpha);

                    // * Step Forward => x = x_0 + alpha * dx
                    // Compute Test Energy => E
                    Float E = compute_energy(alpha);

                    if(!converged)
                    {
                        SizeT line_search_iter = 0;
                        while(line_search_iter < m_line_searcher->max_iter())
                        {
                            Timer timer{"Line Search Iteration"};

                            // Check Energy Decrease
                            // TODO: maybe better condition like Wolfe condition/Armijo condition in the future
                            bool energy_decrease = (E <= E0);

                            // Check Inversion
                            // TODO: Inversion check if needed
                            bool no_inversion = true;

                            bool success = energy_decrease && no_inversion;

                            if(success)
                                break;

                            // If not success, then shrink alpha
                            alpha /= 2;
                            E = compute_energy(alpha);
                            Itres_Energy_each = E;
                            line_search_iter++;
                        }

                        // Check Line Search Iteration
                        // report warnings or throw exceptions if needed
                        check_line_search_iter(line_search_iter);
                        line_search_iter_global = line_search_iter;
                    }
                }
                Iters_line_search.push_back(line_search_iter_global);
            }
            newton_iters_record.push_back(newton_iter);
            Iters_energy.push_back(Itres_Energy_each);
            // 5. Update Velocity => v = (x - x_0) / dt
            m_state = SimEngineState::UpdateVelocity;
            {
                Timer timer{"Update Velocity"};
                m_time_integrator_manager->update_state();
            }
            //std::cout << "Newton iteration counts: [";
            //for(size_t i = 0; i < newton_iters_record.size(); ++i)
            //{
            //    std::cout << newton_iters_record[i];
            //    if(i + 1 < newton_iters_record.size())
            //        std::cout << ", ";
            //}
            //std::cout << "]" << std::endl;

            //std::cout << "Line search counts: [";
            //for(size_t i = 0; i < Iters_line_search.size(); ++i)
            //{
            //    std::cout << Iters_line_search[i];
            //    if(i + 1 < Iters_line_search.size())
            //        std::cout << ", ";
            //}
            //std::cout << "]" << std::endl;

            //std::cout << "Energy iteration counts: [";
            //for(size_t i = 0; i < Iters_energy.size(); ++i)
            //{
            //    std::cout << Iters_energy[i];
            //    if(i + 1 < Iters_energy.size())
            //        std::cout << ", ";
            //}
            //std::cout << "]" << std::endl;

            // Check Newton Iteration
            // report warnings or throw exceptions if needed
            check_newton_iter(newton_iter);
        }

        spdlog::info("<<< End Frame: {}", m_current_frame);
    };


    auto pipeline_vbd = [&]() noexcept(AbortOnException)
    {
        Timer timer{"Pipeline"};

        ++m_current_frame;

        spdlog::info(R"(>>> Begin Frame: {})", m_current_frame);


        // Rebuild Scene
        {
            Timer timer{"Rebuild Scene"};
            // Trigger the rebuild_scene event, systems register their actions will be called here
            m_state = SimEngineState::RebuildScene;
            {
                event_rebuild_scene();

                // TODO: rebuild the vertex and surface info
                // m_global_vertex_manager->rebuild_vertex_info();
                // m_global_surface_manager->rebuild_surface_info();
            }

            // After the rebuild_scene event, the pending creation or deletion can be solved
            world().scene().solve_pending();

            // Update the diff parms
            update_diff_parm();
        }
        //m_global_vertex_manager->m_impl.body_ids();
        //world().scene().solve_pending();
        //world().scene();
        //world().scene().geometries().data();
        //auto obj_all = scene.objects().find("balls")[0];
        //auto ids     = obj_all->geometries().ids();
        //auto geo     = scene.geometries()
        //               .find(ids[0])
        //               .rest_geometry->geometry()
        //               .as<SimplicialComplex>()
        //               ->vertices();

        //GlobalVertexManager* m_global_vertex_manager = nullptr;
        //GlobalSimplicialSurfaceManager* m_global_simplicial_surface_manager = nullptr;
        //GlobalBodyManager*         m_global_body_manager          = nullptr;
        //GlobalContactManager*      m_global_contact_manager       = nullptr;
        //GlobalDyTopoEffectManager* m_global_dytopo_effect_manager = nullptr;
        //GlobalTrajectoryFilter*    m_global_trajectory_filter     = nullptr;

        //// Newton Solver Systems
        //TimeIntegratorManager*  m_time_integrator_manager  = nullptr;
        //LineSearcher*           m_line_searcher            = nullptr;
        //GlobalLinearSystem*     m_global_linear_system     = nullptr;
        //NewtonToleranceManager* m_newton_tolerance_manager = nullptr;

        spdlog::info("<<< End Frame: {}", m_current_frame);

        // Simulation:
        {
            Timer timer{"Simulation"};
            // 1. Adaptive Parameter Calculation
            AABB vertex_bounding_box = m_global_vertex_manager->compute_vertex_bounding_box();
            size_t size = m_global_vertex_manager->body_ids().size();
            std::vector<IndexT> body_ids_host(size);
            m_global_vertex_manager->body_ids().copy_to(body_ids_host.data());
            //DeviceBuffer<Vector2i> body_ids_test;
            //std::vector<Vector2i>  body_ids_test_host;
            //body_ids_test.copy_to(body_ids_test_host);
            //using namespace uipc;
            //using namespace uipc::core;
            using namespace uipc::geometry;

            //uipc::geometry::GeometrySlot

            auto geo = world().scene().geometries()[0];

            auto & verticesColoringCategories = geo->geometry().edgesColoringCategories;
            auto& edgesColoringCategories = geo->geometry().edgesColoringCategories;
            auto& tetsColoringCategories = geo->geometry().tetsColoringCategories;
            /////////############################################## read graph color method:

            /////////############################################## run color mapping graph
            //// generate parallelization groups
            size_t numberOfParallelGroups = 0;
            //for(size_t iMesh = 0; iMesh < basetetMeshes.size(); iMesh++)
            //{
            //    if(numberOfParallelGroups
            //       < basetetMeshes[iMesh]->verticesColoringCategories().size())
            //    {
            //        numberOfParallelGroups =
            //            basetetMeshes[iMesh]->verticesColoringCategories().size();
            //    }
            //}
            //########################## multi-mesh cases to load color into same region
            std::vector<std::vector<int32_t>> vertexParallelGroups;
            vertexParallelGroups.resize(numberOfParallelGroups);
            //for(int iMesh = 0; iMesh < tMeshes.size(); iMesh++)
            //{
            //    VBDBaseTetMesh::SharedPtr pMesh = tMeshes[iMesh];
            //    size_t numColors = pMesh->verticesColoringCategories().size();
            //    std::vector<IdType> availableColors(numColors, -1);
            //    numAllVertices += pMesh->numVertices();

            //    for(int iColor = 0; iColor < numColors; iColor++)
            //    {
            //        availableColors[iColor] = iColor;
            //    }

            //    for(int iColor = 0; iColor < numColors; iColor++)
            //    {
            //        const std::vector<IdType>& currentColorGroup =
            //            pMesh->verticesColoringCategories()[iColor];

            //        int smallestGroupId =
            //            findSmallestParallelGroup(availableColors, vertexParallelGroups, true);
            //        // add the current color group to this parallel group
            //        for(int iVertex = 0; iVertex < currentColorGroup.size(); iVertex++)
            //        {
            //            vertexParallelGroups[smallestGroupId].push_back(iMesh);
            //            vertexParallelGroups[smallestGroupId].push_back(
            //                currentColorGroup[iVertex]);
            //        }
            //        // color group from the same mesh cannot be added to this group again
            //        // std::cout << "Color " << iColor << " of mesh " << iMesh << " was added to parallel group " << smallestGroupId << "\n";
            //    }
            //}
            ///////////////////////////////////////////////////////////////////////////////
            world().scene().geometries()[0]->geometry().instances().find<IndexT>(builtin::is_fixed);
            world().scene().find_geometry(0)->geometry().instances().find<IndexT>(builtin::is_fixed);


            auto geoT = world()
                           .scene()
                           .get()
                           .geometries()
                           .find(0)
                           .rest_geometry->geometry()
                           .as<SimplicialComplex>();


            auto pos = geoT->vertices().find<Vector3>(builtin::position);
            span<Vector3>       non_const_view = view(*pos);
            for(size_t i = 0; i < non_const_view.size(); ++i)
            {
                const auto& vec = non_const_view[i];
                spdlog::info("vertex info {}: ({}, {}, {})",
                             i,
                             vec.x(),
                             vec.y(),
                             vec.z());
            }


            /////##############################################CPU serial
            int iterations = 100;
            for(int iIter = 0; iIter < iterations; iIter++)
            {
                //bool apply_friction = iIter >= physicsParams().frictionStartIter;
                //solveCollisionsSequentially();
                for(size_t iGroup = 0; iGroup < vertexParallelGroups.size(); iGroup++)
                {
                    const std::vector<int32_t>& parallelGroup = vertexParallelGroups[iGroup];
                    size_t numVertices = parallelGroup.size() / 2;  // 每个顶点存储两个元素（iMesh和vId）

                    // 将并行循环改为串行for循环，逐个处理组内的顶点
                    for(int iV = 0; iV < numVertices; ++iV)
                    {
                        int32_t iMesh = parallelGroup[iV * 2];  // 获取顶点所属的网格索引
                        int vId = parallelGroup[2 * iV + 1];  // 获取顶点在网格中的索引

                        // 获取网格对象并执行处理逻辑
                        //VBDTetMeshNeoHookean* pMesh = (VBDTetMeshNeoHookean*)tMeshes[iMesh].get();
                        //if(!pMesh->fixedMask[vId] && !pMesh->activeCollisionMask[vId]
                        //   && pMesh->activeForMaterialSolve)
                        {
                            //VBDStepWithCollision(pMesh, iMesh, vId, apply_friction);
                            //VBDTetMeshNeoHookean* pMesh = (VBDTetMeshNeoHookean*)pMesh_;
                            Matrix3x3 h;
                            Vector3   force;
                            h.setZero();
                            force.setZero();

                            // 累加材料力和海森矩阵（核心逻辑保留）
                            //pMesh->accumlateMaterialForceAndHessian2(vertexId, force, h);

                            // 累加惯性力和海森矩阵（核心逻辑保留，删除debugOperation）
                            //pMesh->accumlateInertiaForceAndHessian(vertexId, force, h);

                            /*Mat3 tmp_h = h; 
                            pMesh->accumlateMaterialForceAndHessian(vertexId, force, h);
                            Mat3 K = h - tmp_h; 
                            accumlateDampingForceAndHessian(pMesh, vertexId, force, h, K);*/

                            // 累加边界力、碰撞力和海森矩阵（核心逻辑保留）
                            //accumlateBoundaryForceAndHessian(pMesh, meshId, vertexId, force, h, apply_friction);
                            // 碰撞立这里可以先不添加试一试
                            // 
                            //accumlateCollisionForceAndHessian(
                            //    pMesh, meshId, vertexId, force, h, apply_friction);

                            // 叠加外力（核心逻辑保留）
                            //force += pMesh->vertexExternalForces.col(vertexId);

                            // 线搜索与线性求解， 这里恐怕得需要自己去求解了
                            double specified_error = 1e-4;
                            if(force.squaredNorm() > specified_error)
                            {
                                Matrix3x3    h;
                                Vector3      force;

                                Vector3 descentDirection;
                                //float   stepSize = physicsParams().stepSize;
                                //float lineSearchShrinkFactor = physicsParams().tau;

                                bool solverSuccess;
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
                                    double F[3] = {force(0), force(1), force(2)};
                                    double dx[3] = {0, 0, 0};
                                    solverSuccess = solve3x3_psd_stable(H, F, dx);
                                    descentDirection = Vector3(dx[0], dx[1], dx[2]);

                                    ////CuMatrix 是什么库？？？？
                                }
                                else
                                {
                                    solverSuccess = solve3x3_psd_stable(
                                        h.data(), force.data(), descentDirection.data());
                                }

                                if(!solverSuccess)
                                {
                                    //stepSize = physicsParams().stepSizeGD;
                                    descentDirection = force;
                                    //lineSearchShrinkFactor = physicsParams().tau_GD;
                                    //std::cout << "Solver failed at vertex "
                                    //          << vertexId << std::endl;
                                    // std::exit(-1);
                                }
                                if(descentDirection.hasNaN())
                                {
                                    std::cout << "force: " << force.transpose() << "\nHessian:\n"
                                              << h;
                                    //std::cout << "descentDirection has NaN at vertex "
                                    //          << vertexId << std::endl;
                                    std::exit(-1);
                                }

                                // 更新顶点位置（核心逻辑保留）
                                //pMesh->vertex(vertexId) += stepSize * descentDirection;
                            }

                        }
                    }
                }
            }  // iteration
            /////##############################################CPU parallel	
            //for(size_t iGroup = 0; iGroup < vertexParallelGroups.size(); iGroup++)
            //{
            //    const std::vector<IdType>& parallelGroup = vertexParallelGroups[iGroup];

            //    size_t numVertices = parallelGroup.size() / 2;
            //    //cpu_parallel_for(0,
            //    //                 numVertices,
            //    //                 [&](int iV)
            //    //                 {
            //    //                     IdType iMesh = parallelGroup[iV * 2];
            //    //                     int    vId   = parallelGroup[2 * iV + 1];

            //    //                     VBDTetMeshNeoHookean* pMesh =
            //    //                         (VBDTetMeshNeoHookean*)tMeshes[iMesh].get();
            //    //                     if(!pMesh->fixedMask[vId]
            //    //                        && !pMesh->activeCollisionMask[vId]
            //    //                        && pMesh->activeForMaterialSolve)
            //    //                     //if (!pMesh->fixedMask[vId])
            //    //                     {
            //    //                         //pMesh->VBDStep(vId);
            //    //                         VBDStepWithCollision(pMesh, iMesh, vId, apply_friction);
            //    //                     }
            //    //                 });
            //}
            /////##############################################


            //std::vector<Vector3> positions_host;
            //m_global_vertex_manager->positions().copy_to(positions_host);


            //std::vector<Vector2i> pairs_host;
            //pairs.copy_to(pairs_host);
            //positions
            //GlobalVertexManager::body_ids();
            //m_global_vertex_manager->m_impl.body_ids();
            //detect_dcd_candidates();
            //compute_adaptive_kappa();

            // 2. Record Friction Candidates at the beginning of the frame
            record_friction_candidates();
            m_global_vertex_manager->record_prev_positions();

            // 3. Predict Motion => x_tilde = x + v * dt
            m_state = SimEngineState::PredictMotion;
            m_time_integrator_manager->predict_dof();
            step_animation();

            // 4. Nonlinear-Newton Iteration
            Float box_size = vertex_bounding_box.diagonal().norm();
            Float tol      = m_newton_scene_tol * box_size;
            m_newton_tolerance_manager->pre_newton(m_current_frame);


            // parallel for vbd process 

            //
            auto   newton_max_iter = m_newton_max_iter->view()[0];
            auto   newton_min_iter = m_newton_min_iter->view()[0];
            IndexT newton_iter     = 0;
            for(; newton_iter < newton_max_iter; ++newton_iter)
            {
                Timer timer{"Newton Iteration"};

                // 1) Compute animation substep ratio
                compute_animation_substep_ratio(newton_iter);

                // 2) Build Collision Pairs
                if(newton_iter > 0)
                    detect_dcd_candidates();

                // 3) Compute Dynamic Topo Effect Gradient and Hessian => G:Vector3, H:Matrix3x3
                //    Including Contact Effect
                m_state = SimEngineState::ComputeDyTopoEffect;
                compute_dytopo_effect();

                // 4) Solve Global Linear System => dx = A^-1 * b
                m_state = SimEngineState::SolveGlobalLinearSystem;
                {
                    Timer timer{"Solve Global Linear System"};
                    m_global_linear_system->solve();
                }


                // 5) Collect Vertex Displacements Globally
                m_global_vertex_manager->collect_vertex_displacements();


                // 6) Check Termination Condition
                bool converged  = convergence_check(newton_iter);
                bool terminated = converged && (newton_iter >= newton_min_iter);
                if(terminated)
                    break;


                // 7) Begin Line Search
                m_state = SimEngineState::LineSearch;
                {
                    Timer timer{"Line Search"};

                    // Reset Alpha
                    alpha = 1.0;

                    // Record Current State x to x_0
                    m_line_searcher->record_start_point();
                    m_global_vertex_manager->record_start_point();
                    detect_trajectory_candidates(alpha);

                    // Compute Current Energy => E_0
                    Float E0 = m_line_searcher->compute_energy(true);  // initial energy
                    // spdlog::info("Initial Energy: {}", E0);

                    // CCD filter
                    alpha = filter_toi(alpha);

                    // CFL Condition
                    alpha = cfl_condition(alpha);

                    // * Step Forward => x = x_0 + alpha * dx
                    // Compute Test Energy => E
                    Float E = compute_energy(alpha);

                    if(!converged)
                    {
                        SizeT line_search_iter = 0;
                        while(line_search_iter < m_line_searcher->max_iter())
                        {
                            Timer timer{"Line Search Iteration"};

                            // Check Energy Decrease
                            // TODO: maybe better condition like Wolfe condition/Armijo condition in the future
                            bool energy_decrease = (E <= E0);

                            // Check Inversion
                            // TODO: Inversion check if needed
                            bool no_inversion = true;

                            bool success = energy_decrease && no_inversion;

                            if(success)
                                break;

                            // If not success, then shrink alpha
                            alpha /= 2;
                            E = compute_energy(alpha);

                            line_search_iter++;
                        }

                        // Check Line Search Iteration
                        // report warnings or throw exceptions if needed
                        check_line_search_iter(line_search_iter);
                    }
                }
            }
            //SizeT AnimatorVisitor::substep() noexcept
            //{
            //    return m_animator.substep();
            //}
          // namespace uipc::backend
            ////////////###########now you need to test a coloring graph on a mesh level#############////////////
            auto substep_Num = world().animator().substep();

            for(int substep = 0; substep < int(substep_Num); substep++)
            {
                //world().animator().
                //curTime += physicsParams().dt;
                //debugOperation(
                //    DEBUG_LVL_DEBUG,
                //    [&]()
                //    { std::cout << "Substep step: " << substep << std::endl; });

                //dcd();
                //################################################################
                detect_dcd_candidates();
                //################################################################
                //cpu_parallel_for(0,
                //                 numTetMeshes(),
                //                 [&](int iMesh)
                //                 {
                //                     if(tMeshes[iMesh]->activeForMaterialSolve)
                //                     {
                //                         tMeshes[iMesh]->evaluateExternalForce();
                //                         tMeshes[iMesh]->applyInitialStep();
                //                     }
                //                 });
                //applyDeformers();

                //ccd();
                //################################################################
                // Reset Alpha
                alpha = 1.0;

                // Record Current State x to x_0
                m_line_searcher->record_start_point();
                m_global_vertex_manager->record_start_point();
                detect_trajectory_candidates(alpha);

                // Compute Current Energy => E_0
                Float E0 = m_line_searcher->compute_energy(true);  // initial energy
                // spdlog::info("Initial Energy: {}", E0);

                // CCD filter
                alpha = filter_toi(alpha);
                //################################################################
                int iterations = 20;
                for(int iIter = 0; iIter < iterations; iIter++)
                {
                    bool apply_friction = true;
                    //bool apply_friction = iIter >= physicsParams().frictionStartIter;
                    // apply_friction = true;
                    //debugOperation(
                    //    DEBUG_LVL_DEBUG_VEBOSE,
                    //    [&]() { std::cout << "iIter: " << iIter << std::endl; });
                    //debugOperation(DEBUG_LVL_DEBUG_VEBOSE,
                    //               std::bind(&VBDPhysics::outputPosVel, this));
                    //debugOperation(DEBUG_LVL_DEBUG_VEBOSE,
                    //               std::bind(&VBDPhysics::clearForces, this));
                    std::vector<std::vector<int32_t>> vertexParallelGroups;
                    for(size_t iGroup = 0; iGroup < vertexParallelGroups.size(); iGroup++)
                    {
                        const std::vector<int32_t>& parallelGroup = vertexParallelGroups[iGroup];

                        size_t numVertices = parallelGroup.size() / 2;
                        //size_t numCollisionParallelGroup =
                        //    activeColllisionList.numActiveCollisionsEachParallelGroup[iGroup];
                        //// update collision info for parallel group
                        //cpu_parallel_for(
                        //    0,
                        //    numCollisionParallelGroup,
                        //    [&](int iCollision)
                        //    {
                        //        IdType iMesh =
                        //            activeColllisionList.activeCollisionsEachParallelGroup[iGroup][iCollision * 2];
                        //        int vId =
                        //            activeColllisionList.activeCollisionsEachParallelGroup[iGroup][2 * iCollision + 1];
                        //        updateCollisionInfo(
                        //            getCollisionDetectionResultFromTetMeshId(iMesh, vId));
                        //    });

                        //cpu_parallel_for(
                        //    0,
                        //    numVertices,
                        //    [&](int iV)
                        //    {
                        //        IdType iMesh = parallelGroup[iV * 2];
                        //        int    vId   = parallelGroup[2 * iV + 1];

                        //        VBDTetMeshNeoHookean* pMesh =
                        //            (VBDTetMeshNeoHookean*)tMeshes[iMesh].get();
                        //        if(!pMesh->fixedMask[vId] && pMesh->activeForMaterialSolve)
                        //        //if (!pMesh->fixedMask[vId])
                        //        {
                        //            //pMesh->VBDStep(vId);
                        //            VBDStepWithCollision(pMesh, iMesh, vId, apply_friction);
                        //        }
                        //    });


                        // VBDTetMeshNeoHookean* pMesh = (VBDTetMeshNeoHookean*)tMeshes[0].get();
                        //int sum = checksum(reinterpret_cast<int*>(pMesh->mVertPos.data()), sizeof(FloatingType) / sizeof(int) * pMesh->mVertPos.size());
                        //std::cout << "substep: " << substep << "iteration: " << iteration << "iGroup: " << iGroup << ", checksum: " << sum << std::endl;
                    }
                    //debugOperation(DEBUG_LVL_DEBUG_VEBOSE,
                    //               std::bind(&VBDPhysics::outputForces, this));
                    //if(physicsParams().intermediateCollisionIterations > 0
                    //   && iIter % physicsParams().intermediateCollisionIterations
                    //          == physicsParams().intermediateCollisionIterations - 1)
                    //{
                    //    intermediateCollisionDetection();
                    //}
                }  // iteration

                //updateVelocities();
            }  // substep

            ////////////////////////////////////
            // 5. Update Velocity => v = (x - x_0) / dt
            m_state = SimEngineState::UpdateVelocity;
            {
                Timer timer{"Update Velocity"};
                m_time_integrator_manager->update_state();
            }

            // Check Newton Iteration
            // report warnings or throw exceptions if needed
            check_newton_iter(newton_iter);
        }

        spdlog::info("<<< End Frame: {}", m_current_frame);
    };

    try
    {
        pipeline_ipc();
        //pipeline_vbd();
        m_last_solved_frame = m_current_frame;
    }
    catch(const SimEngineException& e)
    {
        spdlog::error("Engine Advance Error: {}", e.what());
        status().push_back(core::EngineStatus::error(e.what()));
    }
}
}  // namespace uipc::backend::cuda
