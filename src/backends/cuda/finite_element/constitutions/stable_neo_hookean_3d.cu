#include <finite_element/fem_3d_constitution.h>
#include <finite_element/constitutions/stable_neo_hookean_3d_function.h>
#include <finite_element/fem_utils.h>
#include <kernel_cout.h>
#include <muda/ext/eigen/log_proxy.h>
#include <Eigen/Dense>
#include <muda/ext/eigen/evd.h>
#include <utils/make_spd.h>
#include <utils/matrix_assembler.h>

namespace uipc::backend::cuda
{
class StableNeoHookean3D final : public FEM3DConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64 ConstitutionUID = 10;

    using FEM3DConstitution::FEM3DConstitution;

    vector<Float> h_mus;
    vector<Float> h_lambdas;

    muda::DeviceBuffer<Float> mus;
    muda::DeviceBuffer<Float> lambdas;

    virtual U64 get_uid() const noexcept override { return ConstitutionUID; }

    virtual void do_build(BuildInfo& info) override {}
    ////////////////============build neibour info
    // add members
    muda::DeviceBuffer<int> d_v2t_offsets;  // size V+1
    muda::DeviceBuffer<int> d_v2t_list;     // size 4*T
    muda::DeviceBuffer<int> d_v2t_cursor;   // size V
    muda::DeviceBuffer<int> d_v_deg;        // size V
    bool                    v2t_built_ = false;

    // build v2t on GPU with a single thread kernel
    void build_v2t_on_device_serial(const ComputeGradientHessianInfo& info)
    {
        using namespace muda;
        const int V = static_cast<int>(info.xs().size());
        const int T = static_cast<int>(info.indices().size());
        if(V <= 0 || T <= 0)
        {
            v2t_built_ = true;
            return;
        }

        d_v_deg.resize(V);
        d_v2t_offsets.resize(V + 1);
        d_v2t_list.resize(4 * T);
        d_v2t_cursor.resize(V);

        Launch(1, 1)
            .file_line(__FILE__, __LINE__)
            .apply(
                [indices = info.indices().viewer().name("indices"),
                 deg     = d_v_deg.viewer().name("deg"),
                 offs    = d_v2t_offsets.viewer().name("offs"),
                 cursor  = d_v2t_cursor.viewer().name("cursor"),
                 v2t     = d_v2t_list.viewer().name("v2t"),
                 V,
                 T] __device__() mutable
                {
                    // 1) zero deg
                    for(int v = 0; v < V; ++v)
                        deg(v) = 0;

                    // 2) count incidence
                    for(int ti = 0; ti < T; ++ti)
                    {
                        const Vector4i& t  = indices(ti);
                        int             v0 = t(0);
                        if(0 <= v0 && v0 < V)
                            ++deg(v0);
                        int v1 = t(1);
                        if(0 <= v1 && v1 < V)
                            ++deg(v1);
                        int v2 = t(2);
                        if(0 <= v2 && v2 < V)
                            ++deg(v2);
                        int v3 = t(3);
                        if(0 <= v3 && v3 < V)
                            ++deg(v3);
                    }

                    // 3) exclusive scan deg -> offs
                    int acc = 0;
                    for(int v = 0; v < V; ++v)
                    {
                        offs(v) = acc;
                        acc += deg(v);
                    }
                    offs(V) = acc;  // total

                    // 4) init cursor
                    for(int v = 0; v < V; ++v)
                        cursor(v) = offs(v);

                    // 5) fill CSR list
                    for(int ti = 0; ti < T; ++ti)
                    {
                        const Vector4i& t  = indices(ti);
                        int             v0 = t(0);
                        if(0 <= v0 && v0 < V)
                        {
                            int p  = cursor(v0)++;
                            v2t(p) = ti;
                        }
                        int v1 = t(1);
                        if(0 <= v1 && v1 < V)
                        {
                            int p  = cursor(v1)++;
                            v2t(p) = ti;
                        }
                        int v2 = t(2);
                        if(0 <= v2 && v2 < V)
                        {
                            int p  = cursor(v2)++;
                            v2t(p) = ti;
                        }
                        int v3 = t(3);
                        if(0 <= v3 && v3 < V)
                        {
                            int p  = cursor(v3)++;
                            v2t(p) = ti;
                        }
                    }
                });

        v2t_built_ = true;
    }

    /////////////////////////==========================================
    /// <summary>
    /// /////////////////
    /// </summary>
    /// <param name="info"></param>
    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override
    {
        using ForEachInfo = FiniteElementMethod::ForEachInfo;

        auto geo_slots = world().scene().geometries();

        auto N = info.primitive_count();

        h_mus.resize(N);
        h_lambdas.resize(N);

        info.for_each(
            geo_slots,
            [](geometry::SimplicialComplex& sc) -> auto
            {
                auto mu     = sc.tetrahedra().find<Float>("mu");
                auto lambda = sc.tetrahedra().find<Float>("lambda");

                return zip(mu->view(), lambda->view());
            },
            [&](const ForEachInfo& I, auto mu_and_lambda)
            {
                auto&& [mu, lambda] = mu_and_lambda;

                auto vI = I.global_index();

                h_mus[vI]     = mu;
                h_lambdas[vI] = lambda;
            });

        mus.resize(N);
        mus.view().copy_from(h_mus.data());

        lambdas.resize(N);
        lambdas.view().copy_from(h_lambdas.data());
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        namespace SNH = sym::stable_neo_hookean_3d;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.indices().size(),
                   [mus      = mus.cviewer().name("mus"),
                    lambdas  = lambdas.cviewer().name("lambdas"),
                    energies = info.energies().viewer().name("energies"),
                    indices  = info.indices().viewer().name("indices"),
                    xs       = info.xs().viewer().name("xs"),
                    Dm_invs  = info.Dm_invs().viewer().name("Dm_invs"),
                    volumes  = info.rest_volumes().viewer().name("volumes"),
                    dt       = info.dt()] __device__(int I)
                   {
                       const Vector4i&  tet    = indices(I);
                       const Matrix3x3& Dm_inv = Dm_invs(I);
                       Float            mu     = mus(I);
                       Float            lambda = lambdas(I);

                       const Vector3& x0 = xs(tet(0));
                       const Vector3& x1 = xs(tet(1));
                       const Vector3& x2 = xs(tet(2));
                       const Vector3& x3 = xs(tet(3));

                       auto F = fem::F(x0, x1, x2, x3, Dm_inv);

                       auto J = F.determinant();

                       //auto VecF = flatten(F);

                       Float E;

                       SNH::E(E, mu, lambda, F);
                       E *= dt * dt * volumes(I);
                       energies(I) = E;
                   });
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override

    {
        using namespace muda;
        namespace SNH = sym::stable_neo_hookean_3d;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.indices().size(),
                   [mus     = mus.cviewer().name("mus"),
                    lambdas = lambdas.cviewer().name("lambdas"),
                    indices = info.indices().viewer().name("indices"),
                    xs      = info.xs().viewer().name("xs"),
                    Dm_invs = info.Dm_invs().viewer().name("Dm_invs"),
                    G3s     = info.gradients().viewer().name("gradients"),
                    H3x3s   = info.hessians().viewer().name("hessians"),
                    volumes = info.rest_volumes().viewer().name("volumes"),
                    dt      = info.dt()] __device__(int I) mutable
                   {
                       const Vector4i&  tet    = indices(I);
                       const Matrix3x3& Dm_inv = Dm_invs(I);
                       Float            mu     = mus(I);
                       Float            lambda = lambdas(I);

                       const Vector3& x0 = xs(tet(0));
                       const Vector3& x1 = xs(tet(1));
                       const Vector3& x2 = xs(tet(2));
                       const Vector3& x3 = xs(tet(3));

                       auto F = fem::F(x0, x1, x2, x3, Dm_inv);

                       auto J = F.determinant();

                       //auto VecF = flatten(F);

                       auto Vdt2 = volumes(I) * dt * dt;

                       Matrix3x3 dEdF;
                       Matrix9x9 ddEddF;
                       SNH::dEdVecF(dEdF, mu, lambda, F);
                       SNH::ddEddVecF(ddEddF, mu, lambda, F);

                       auto VecdEdF = flatten(dEdF);

                       VecdEdF *= Vdt2;
                       ddEddF *= Vdt2;

                       make_spd(ddEddF);
                       Matrix9x12  dFdx = fem::dFdx(Dm_inv);
                       Vector12    G    = dFdx.transpose() * VecdEdF;
                       Matrix12x12 H    = dFdx.transpose() * ddEddF * dFdx;

                       DoubletVectorAssembler DVA{G3s};
                       DVA.segment<4>(I * 4).write(tet, G);

                       TripletMatrixAssembler TMA{H3x3s};
                       TMA.block<4, 4>(I * 4 * 4).write(tet, H);
                   });
    }


    virtual void do_compute_gradient_hessian_by_vertex(ComputeGradientHessianInfo& info, IndexT vertexId) override
    {
        //gpu 并行化 四面体的处理？？？？感觉没有必要，暂时先用cpu的思路，遍历与该顶点相关的四面体
        //using namespace muda;
        //namespace SNH = sym::stable_neo_hookean_3d;

        //if(vertexId < 0)
        //    return;
        //if(!v2t_built_)
        //    build_v2t_on_device_serial(info);  // 保持前面用 Launch(1,1) 的序列化构建

        //// 用很小的 host 读回获取 begin/end（仅2个int），然后用 ParallelFor
        //int off2[2] = {0, 0};
        //// 读取 offsets[vertexId], offsets[vertexId+1]
        //d_v2t_offsets.view().subview(static_cast<int>(vertexId), 2).copy_to(off2);
        //const int begin = off2[0];
        //const int end   = off2[1];
        //const int cnt   = end - begin;
        //if(cnt <= 0)
        //    return;

        //ParallelFor()
        //    .file_line(__FILE__, __LINE__)
        //    .apply(cnt,
        //           [mus      = mus.cviewer().name("mus"),
        //            lambdas  = lambdas.cviewer().name("lambdas"),
        //            indices  = info.indices().viewer().name("indices"),
        //            xs       = info.xs().viewer().name("xs"),
        //            Dm_invs  = info.Dm_invs().viewer().name("Dm_invs"),
        //            G3s      = info.gradients().viewer().name("gradients"),
        //            H3x3s    = info.hessians().viewer().name("hessians"),
        //            volumes  = info.rest_volumes().viewer().name("volumes"),
        //            dt       = info.dt(),
        //            v2t_list = d_v2t_list.cviewer().name("v2t_list"),
        //            begin] __device__(int k) mutable
        //           {
        //               const int I = v2t_list(begin + k);  // incident tet index

        //               const Vector4i&  tet    = indices(I);
        //               const Matrix3x3& Dm_inv = Dm_invs(I);
        //               const Float      mu     = mus(I);
        //               const Float      lambda = lambdas(I);

        //               const Vector3& x0 = xs(tet(0));
        //               const Vector3& x1 = xs(tet(1));
        //               const Vector3& x2 = xs(tet(2));
        //               const Vector3& x3 = xs(tet(3));

        //               const auto F    = fem::F(x0, x1, x2, x3, Dm_inv);
        //               const auto Vdt2 = volumes(I) * dt * dt;

        //               Matrix3x3 dEdF;
        //               Matrix9x9 ddEddF;
        //               SNH::dEdVecF(dEdF, mu, lambda, F);
        //               SNH::ddEddVecF(ddEddF, mu, lambda, F);

        //               auto VecdEdF = flatten(dEdF);
        //               VecdEdF *= Vdt2;
        //               ddEddF *= Vdt2;

        //               make_spd(ddEddF);
        //               Matrix9x12  dFdx = fem::dFdx(Dm_inv);
        //               Vector12    G    = dFdx.transpose() * VecdEdF;
        //               Matrix12x12 H    = dFdx.transpose() * ddEddF * dFdx;

        //               DoubletVectorAssembler DVA{G3s};
        //               DVA.segment<4>(I * 4).write(tet, G);

        //               TripletMatrixAssembler TMA{H3x3s};
        //               TMA.block<4, 4>(I * 16).write(tet, H);
        //           });

        //在gpu中暂时先用cpu的思路，遍历与该顶点相关的四面体
        using namespace muda;
        namespace SNH = sym::stable_neo_hookean_3d;

        if(vertexId < 0)
            return;
        if(!v2t_built_)
            build_v2t_on_device_serial(info);  // 确保已在 GPU 上构建 v2t

        Launch(1, 1)
            .file_line(__FILE__, __LINE__)
            .apply(
                [mus      = mus.cviewer().name("mus"),
                 lambdas  = lambdas.cviewer().name("lambdas"),
                 indices  = info.indices().viewer().name("indices"),
                 xs       = info.xs().viewer().name("xs"),
                 Dm_invs  = info.Dm_invs().viewer().name("Dm_invs"),
                 G3s      = info.gradients().viewer().name("gradients"),
                 H3x3s    = info.hessians().viewer().name("hessians"),
                 volumes  = info.rest_volumes().viewer().name("volumes"),
                 dt       = info.dt(),
                 v2t_offs = d_v2t_offsets.cviewer().name("v2t_offs"),
                 v2t_list = d_v2t_list.cviewer().name("v2t_list"),
                 vId      = static_cast<int>(vertexId)] __device__() mutable
                {
                    const int begin = v2t_offs(vId);
                    const int end   = v2t_offs(vId + 1);
                    if(end <= begin)
                        return;

                    for(int k = begin; k < end; ++k)
                    {
                        const int I = v2t_list(k);  // incident tet index

                        const Vector4i&  tet    = indices(I);
                        const Matrix3x3& Dm_inv = Dm_invs(I);
                        const Float      mu     = mus(I);
                        const Float      lambda = lambdas(I);

                        const Vector3& x0 = xs(tet(0));
                        const Vector3& x1 = xs(tet(1));
                        const Vector3& x2 = xs(tet(2));
                        const Vector3& x3 = xs(tet(3));

                        const auto F    = fem::F(x0, x1, x2, x3, Dm_inv);
                        const auto Vdt2 = volumes(I) * dt * dt;

                        Matrix3x3 dEdF;
                        Matrix9x9 ddEddF;
                        SNH::dEdVecF(dEdF, mu, lambda, F);
                        SNH::ddEddVecF(ddEddF, mu, lambda, F);

                        auto VecdEdF = flatten(dEdF);
                        VecdEdF *= Vdt2;
                        ddEddF *= Vdt2;

                        make_spd(ddEddF);
                        Matrix9x12  dFdx = fem::dFdx(Dm_inv);
                        Vector12    G    = dFdx.transpose() * VecdEdF;
                        Matrix12x12 H    = dFdx.transpose() * ddEddF * dFdx;

                        DoubletVectorAssembler DVA{G3s};
                        DVA.segment<4>(I * 4).write(tet, G);

                        TripletMatrixAssembler TMA{H3x3s};
                        TMA.block<4, 4>(I * 16).write(tet, H);
                    }
                });

        ////// 旧的实现：遍历所有四面体，跳过不包含目标顶点的四面体
        //using namespace muda;
        //namespace SNH = sym::stable_neo_hookean_3d;

        ////// 仅计算与指定顶点（vertexId）相关的四面体
        //Launch(1, 1)
        //    .apply(
        //        [mus     = mus.cviewer().name("mus"),
        //         lambdas = lambdas.cviewer().name("lambdas"),
        //         indices = info.indices().viewer().name("indices"),  // 四面体顶点索引（每个四面体含4个顶点）
        //         xs       = info.xs().viewer().name("xs"),
        //         Dm_invs  = info.Dm_invs().viewer().name("Dm_invs"),
        //         G3s      = info.gradients().viewer().name("gradients"),
        //         H3x3s    = info.hessians().viewer().name("hessians"),
        //         volumes  = info.rest_volumes().viewer().name("volumes"),
        //         dt       = info.dt(),
        //         vertexId = vertexId,                             // 目标顶点ID
        //         n = info.indices().size()] __device__() mutable  // n为四面体总数
        //        {
        //            // 边界检查：确保目标顶点ID有效（假设顶点ID范围是0~max_vertex）
        //            // （若有顶点总数信息，可添加 vertexId < max_vertex 检查）
        //            if(vertexId < 0)
        //            {
        //                print("Invalid vertexId+++++++++++++++++++++++++++++++++++++++++++++++++++++++++: %d\n", vertexId);
        //                return;
        //            }

        //            // 遍历所有四面体，仅处理包含目标顶点（vertexId）的四面体
        //            for(int I = 0; I < n; ++I)  // I是四面体索引
        //            {
        //                // 获取当前四面体的4个顶点索引
        //                const Vector4i& tet = indices(I);

        //                // 检查当前四面体是否包含目标顶点（vertexId）
        //                bool contains_target =
        //                    (tet(0) == vertexId) || (tet(1) == vertexId)
        //                    || (tet(2) == vertexId) || (tet(3) == vertexId);

        //                // 若不包含目标顶点，跳过当前四面体
        //                if(!contains_target)
        //                    continue;

        //                // 以下是原ParallelFor中针对包含目标顶点的四面体的计算逻辑
        //                const Matrix3x3& Dm_inv = Dm_invs(I);
        //                Float            mu     = mus(I);
        //                Float            lambda = lambdas(I);

        //                const Vector3& x0 = xs(tet(0));
        //                const Vector3& x1 = xs(tet(1));
        //                const Vector3& x2 = xs(tet(2));
        //                const Vector3& x3 = xs(tet(3));

        //                auto F = fem::F(x0, x1, x2, x3, Dm_inv);
        //                auto J = F.determinant();

        //                auto Vdt2 = volumes(I) * dt * dt;

        //                Matrix3x3 dEdF;
        //                Matrix9x9 ddEddF;
        //                SNH::dEdVecF(dEdF, mu, lambda, F);
        //                SNH::ddEddVecF(ddEddF, mu, lambda, F);

        //                auto VecdEdF = flatten(dEdF);
        //                VecdEdF *= Vdt2;
        //                ddEddF *= Vdt2;

        //                make_spd(ddEddF);
        //                Matrix9x12  dFdx = fem::dFdx(Dm_inv);
        //                Vector12    G    = dFdx.transpose() * VecdEdF;
        //                Matrix12x12 H    = dFdx.transpose() * ddEddF * dFdx;

        //                // 组装梯度和Hessian（仅针对包含目标顶点的四面体）
        //                DoubletVectorAssembler DVA{G3s};
        //                DVA.segment<4>(I * 4).write(tet, G);

        //                TripletMatrixAssembler TMA{H3x3s};
        //                TMA.block<4, 4>(I * 4 * 4).write(tet, H);
        //            }
        //        });

        //ParallelFor()
        //    .file_line(__FILE__, __LINE__)
        //    .apply(info.indices().size(),
        //           [mus     = mus.cviewer().name("mus"),
        //            lambdas = lambdas.cviewer().name("lambdas"),
        //            indices = info.indices().viewer().name("indices"),
        //            xs      = info.xs().viewer().name("xs"),
        //            Dm_invs = info.Dm_invs().viewer().name("Dm_invs"),
        //            G3s     = info.gradients().viewer().name("gradients"),
        //            H3x3s   = info.hessians().viewer().name("hessians"),
        //            volumes = info.rest_volumes().viewer().name("volumes"),
        //            dt      = info.dt()] __device__(int I) mutable
        //           {
        //               const Vector4i&  tet    = indices(I);
        //               const Matrix3x3& Dm_inv = Dm_invs(I);
        //               Float            mu     = mus(I);
        //               Float            lambda = lambdas(I);

        //               const Vector3& x0 = xs(tet(0));
        //               const Vector3& x1 = xs(tet(1));
        //               const Vector3& x2 = xs(tet(2));
        //               const Vector3& x3 = xs(tet(3));

        //               auto F = fem::F(x0, x1, x2, x3, Dm_inv);

        //               auto J = F.determinant();

        //               //auto VecF = flatten(F);

        //               auto Vdt2 = volumes(I) * dt * dt;

        //               Matrix3x3 dEdF;
        //               Matrix9x9 ddEddF;
        //               SNH::dEdVecF(dEdF, mu, lambda, F);
        //               SNH::ddEddVecF(ddEddF, mu, lambda, F);

        //               auto VecdEdF = flatten(dEdF);

        //               VecdEdF *= Vdt2;
        //               ddEddF *= Vdt2;

        //               make_spd(ddEddF);
        //               Matrix9x12  dFdx = fem::dFdx(Dm_inv);
        //               Vector12    G    = dFdx.transpose() * VecdEdF;
        //               Matrix12x12 H    = dFdx.transpose() * ddEddF * dFdx;

        //               DoubletVectorAssembler DVA{G3s};
        //               DVA.segment<4>(I * 4).write(tet, G);

        //               TripletMatrixAssembler TMA{H3x3s};
        //               TMA.block<4, 4>(I * 4 * 4).write(tet, H);
        //           });
    }

    virtual void do_compute_gradient_hessian_by_color(ComputeGradientHessianInfo& info,
                                                       muda::CBufferView<IndexT> color_vertices) override
    {

        // 并行维度：color_vertices 中的每个顶点各一个线程
        using namespace muda;
        namespace SNH = sym::stable_neo_hookean_3d;

        const int m = static_cast<int>(color_vertices.size());
        if(m == 0)
            return;
        if(!v2t_built_)
            build_v2t_on_device_serial(info);  // 确保已在 GPU 上构建 v2t

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(m,
                   [mus      = mus.cviewer().name("mus"),
                    lambdas  = lambdas.cviewer().name("lambdas"),
                    indices  = info.indices().viewer().name("indices"),
                    xs       = info.xs().viewer().name("xs"),
                    Dm_invs  = info.Dm_invs().viewer().name("Dm_invs"),
                    G3s      = info.gradients().viewer().name("gradients"),
                    H3x3s    = info.hessians().viewer().name("hessians"),
                    volumes  = info.rest_volumes().viewer().name("volumes"),
                    dt       = info.dt(),
                    v2t_offs = d_v2t_offsets.cviewer().name("v2t_offs"),
                    v2t_list = d_v2t_list.cviewer().name("v2t_list"),
                    verts = color_vertices.viewer().name("verts")] __device__(int ki) mutable
                   {
                       const int v = static_cast<int>(verts(ki));
                       if(v < 0)
                           return;

                       const int begin = v2t_offs(v);
                       const int end   = v2t_offs(v + 1);
                       if(end <= begin)
                           return;

                       // 遍历该顶点的一环四面体并装配
                       for(int p = begin; p < end; ++p)
                       {
                           const int I = v2t_list(p);  // 该顶点相邻的四面体索引

                           const Vector4i&  tet    = indices(I);
                           const Matrix3x3& Dm_inv = Dm_invs(I);
                           const Float      mu     = mus(I);
                           const Float      lambda = lambdas(I);

                           const Vector3& x0 = xs(tet(0));
                           const Vector3& x1 = xs(tet(1));
                           const Vector3& x2 = xs(tet(2));
                           const Vector3& x3 = xs(tet(3));

                           const auto F    = fem::F(x0, x1, x2, x3, Dm_inv);
                           const auto Vdt2 = volumes(I) * dt * dt;

                           Matrix3x3 dEdF;
                           Matrix9x9 ddEddF;
                           SNH::dEdVecF(dEdF, mu, lambda, F);
                           SNH::ddEddVecF(ddEddF, mu, lambda, F);

                           auto VecdEdF = flatten(dEdF);
                           VecdEdF *= Vdt2;
                           ddEddF *= Vdt2;

                           make_spd(ddEddF);
                           Matrix9x12  dFdx = fem::dFdx(Dm_inv);
                           Vector12    G    = dFdx.transpose() * VecdEdF;
                           Matrix12x12 H    = dFdx.transpose() * ddEddF * dFdx;

                           // 假设颜色组内不会出现同一四面体的多个顶点（无写冲突）
                           DoubletVectorAssembler DVA{G3s};
                           DVA.segment<4>(I * 4).write(tet, G);

                           TripletMatrixAssembler TMA{H3x3s};
                           TMA.block<4, 4>(I * 4 * 4).write(tet, H);
                       }
                   })
            .wait();

        ////在gpu中暂时先用cpu的思路，遍历与该顶点相关的四面体
        //using namespace muda;
        //namespace SNH = sym::stable_neo_hookean_3d;

        //if(vertexId < 0)
        //    return;
        //if(!v2t_built_)
        //    build_v2t_on_device_serial(info);  // 确保已在 GPU 上构建 v2t

        //Launch(1, 1)
        //    .file_line(__FILE__, __LINE__)
        //    .apply(
        //        [mus      = mus.cviewer().name("mus"),
        //         lambdas  = lambdas.cviewer().name("lambdas"),
        //         indices  = info.indices().viewer().name("indices"),
        //         xs       = info.xs().viewer().name("xs"),
        //         Dm_invs  = info.Dm_invs().viewer().name("Dm_invs"),
        //         G3s      = info.gradients().viewer().name("gradients"),
        //         H3x3s    = info.hessians().viewer().name("hessians"),
        //         volumes  = info.rest_volumes().viewer().name("volumes"),
        //         dt       = info.dt(),
        //         v2t_offs = d_v2t_offsets.cviewer().name("v2t_offs"),
        //         v2t_list = d_v2t_list.cviewer().name("v2t_list"),
        //         vId      = static_cast<int>(vertexId)] __device__() mutable
        //        {
        //            const int begin = v2t_offs(vId);
        //            const int end   = v2t_offs(vId + 1);
        //            if(end <= begin)
        //                return;

        //            for(int k = begin; k < end; ++k)
        //            {
        //                const int I = v2t_list(k);  // incident tet index

        //                const Vector4i&  tet    = indices(I);
        //                const Matrix3x3& Dm_inv = Dm_invs(I);
        //                const Float      mu     = mus(I);
        //                const Float      lambda = lambdas(I);

        //                const Vector3& x0 = xs(tet(0));
        //                const Vector3& x1 = xs(tet(1));
        //                const Vector3& x2 = xs(tet(2));
        //                const Vector3& x3 = xs(tet(3));

        //                const auto F    = fem::F(x0, x1, x2, x3, Dm_inv);
        //                const auto Vdt2 = volumes(I) * dt * dt;

        //                Matrix3x3 dEdF;
        //                Matrix9x9 ddEddF;
        //                SNH::dEdVecF(dEdF, mu, lambda, F);
        //                SNH::ddEddVecF(ddEddF, mu, lambda, F);

        //                auto VecdEdF = flatten(dEdF);
        //                VecdEdF *= Vdt2;
        //                ddEddF *= Vdt2;

        //                make_spd(ddEddF);
        //                Matrix9x12  dFdx = fem::dFdx(Dm_inv);
        //                Vector12    G    = dFdx.transpose() * VecdEdF;
        //                Matrix12x12 H    = dFdx.transpose() * ddEddF * dFdx;

        //                DoubletVectorAssembler DVA{G3s};
        //                DVA.segment<4>(I * 4).write(tet, G);

        //                TripletMatrixAssembler TMA{H3x3s};
        //                TMA.block<4, 4>(I * 16).write(tet, H);
        //            }
        //        });
    }
};

REGISTER_SIM_SYSTEM(StableNeoHookean3D);
}  // namespace uipc::backend::cuda
