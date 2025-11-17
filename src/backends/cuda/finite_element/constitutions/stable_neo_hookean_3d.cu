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

        virtual void do_compute_gradient_hessian_by_vertex(ComputeGradientHessianInfo& info,
                                                       IndexT vertexId) override
    {
        // GPU processing for tetrahedrons associated with a specific vertex
        // Temporarily adopts CPU-like logic: iterate over tetrahedrons related to the target vertex
        using namespace muda;
        namespace SNH = sym::stable_neo_hookean_3d;

        if(vertexId < 0)
            return;
        if(!v2t_built_)
            build_v2t_on_device_serial(info);  // Ensure vertex-to-tetrahedron mapping is built on GPU

        // Launch a single GPU thread to process tetrahedrons associated with the target vertex (serial execution)
        Launch(1, 1)
            .file_line(__FILE__, __LINE__)
            .apply(
                [mus = mus.cviewer().name("mus"),  // Material parameter: shear modulus
                 lambdas = lambdas.cviewer().name("lambdas"),  // Material parameter: Lame's first parameter
                 indices = info.indices().viewer().name("indices"),  // Tetrahedron vertex indices (4 vertices per tetrahedron)
                 xs = info.xs().viewer().name("xs"),  // Current vertex positions
                 Dm_invs = info.Dm_invs().viewer().name("Dm_invs"),  // Inverse of reference deformation gradient
                 G3s = info.gradients().viewer().name("gradients"),  // Gradient buffer (to be assembled)
                 H3x3s = info.hessians().viewer().name("hessians"),  // Hessian matrix buffer (to be assembled)
                 volumes = info.rest_volumes().viewer().name("volumes"),  // Rest volumes of tetrahedrons
                 dt = info.dt(),  // Time step
                 v2t_offs = d_v2t_offsets.cviewer().name("v2t_offs"),  // Offsets for vertex-to-tetrahedron mapping
                 v2t_list = d_v2t_list.cviewer().name("v2t_list"),  // List of tetrahedrons for each vertex
                 vId = static_cast<int>(vertexId)] __device__() mutable  // Target vertex ID
                {
                    // Get the range of tetrahedrons associated with the target vertex
                    const int begin = v2t_offs(vId);
                    const int end   = v2t_offs(vId + 1);
                    if(end <= begin)  // No associated tetrahedrons, exit early
                        return;

                    // Iterate over all tetrahedrons associated with the target vertex
                    for(int k = begin; k < end; ++k)
                    {
                        const int I = v2t_list(k);  // Index of the current incident tetrahedron

                        const Vector4i& tet = indices(I);  // 4 vertices of the current tetrahedron
                        const Matrix3x3& Dm_inv = Dm_invs(I);  // Inverse reference deformation gradient of the tetrahedron
                        const Float mu = mus(I);  // Shear modulus of the tetrahedron
                        const Float lambda = lambdas(I);  // Lame's parameter of the tetrahedron

                        // Current positions of the tetrahedron's vertices
                        const Vector3& x0 = xs(tet(0));
                        const Vector3& x1 = xs(tet(1));
                        const Vector3& x2 = xs(tet(2));
                        const Vector3& x3 = xs(tet(3));

                        // Compute deformation gradient F
                        const auto F = fem::F(x0, x1, x2, x3, Dm_inv);
                        // Compute volume scaling factor (rest volume * dt square)
                        const auto Vdt2 = volumes(I) * dt * dt;

                        // Compute first and second derivatives of strain energy w.r.t. F
                        Matrix3x3 dEdF;  // First derivative (Cauchy stress related)
                        Matrix9x9 ddEddF;  // Second derivative (stiffness related)
                        SNH::dEdVecF(dEdF, mu, lambda, F);  // Compute dEdF using stable neo-Hookean model
                        SNH::ddEddVecF(ddEddF, mu, lambda, F);  // Compute ddEddF using stable neo-Hookean model

                        // Scale derivatives by volume and time step factor
                        auto VecdEdF = flatten(dEdF);  // Flatten 3x3 matrix to 9x1 vector
                        VecdEdF *= Vdt2;
                        ddEddF *= Vdt2;

                        // Ensure the Hessian is symmetric positive definite
                        make_spd(ddEddF);
                        // Compute derivative of F w.r.t. vertex positions (9x12 matrix)
                        Matrix9x12 dFdx = fem::dFdx(Dm_inv);
                        // Compute gradient (12x1 vector: 3 components per vertex in the tetrahedron)
                        Vector12 G = dFdx.transpose() * VecdEdF;
                        // Compute Hessian (12x12 matrix: second derivatives between vertices)
                        Matrix12x12 H = dFdx.transpose() * ddEddF * dFdx;

                        // Assemble gradient into global buffer
                        DoubletVectorAssembler DVA{G3s};
                        DVA.segment<4>(I * 4).write(tet, G);  // Write gradient for the 4 vertices of the tetrahedron

                        // Assemble Hessian into global buffer
                        TripletMatrixAssembler TMA{H3x3s};
                        TMA.block<4, 4>(I * 16).write(tet, H);  // Write Hessian block for the tetrahedron's vertices
                    }
                });

        ////// Old implementation: Iterate all tetrahedrons, skip those not containing the target vertex
        //using namespace muda;
        //namespace SNH = sym::stable_neo_hookean_3d;

        ////// Only compute for tetrahedrons containing the specified vertex (vertexId)
        //Launch(1, 1)
        //    .apply(
        //        [mus     = mus.cviewer().name("mus"),
        //         lambdas = lambdas.cviewer().name("lambdas"),
        //         indices = info.indices().viewer().name("indices"),  // Tetrahedron vertex indices (4 per tetrahedron)
        //         xs       = info.xs().viewer().name("xs"),
        //         Dm_invs  = info.Dm_invs().viewer().name("Dm_invs"),
        //         G3s      = info.gradients().viewer().name("gradients"),
        //         H3x3s    = info.hessians().viewer().name("hessians"),
        //         volumes  = info.rest_volumes().viewer().name("volumes"),
        //         dt       = info.dt(),
        //         vertexId = vertexId,                             // Target vertex ID
        //         n = info.indices().size()] __device__() mutable  // Total number of tetrahedrons
        //        {
        //            // Boundary check: Ensure target vertex ID is valid
        //            if(vertexId < 0)
        //            {
        //                print("Invalid vertexId+++++++++++++++++++++++++++++++++++++++++++++++++++++++++: %d\n", vertexId);
        //                return;
        //            }

        //            // Iterate all tetrahedrons, process only those containing the target vertex
        //            for(int I = 0; I < n; ++I)  // I is tetrahedron index
        //            {
        //                // Get 4 vertex indices of the current tetrahedron
        //                const Vector4i& tet = indices(I);

        //                // Check if current tetrahedron contains the target vertex
        //                bool contains_target =
        //                    (tet(0) == vertexId) || (tet(1) == vertexId)
        //                    || (tet(2) == vertexId) || (tet(3) == vertexId);

        //                // Skip if no target vertex
        //                if(!contains_target)
        //                    continue;

        //                // Computation logic for tetrahedrons containing the target vertex
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

        //                // Assemble gradient and Hessian (only for target-containing tetrahedrons)
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

        // Parallel dimension: one thread per vertex in color_vertices
        using namespace muda;
        namespace SNH = sym::stable_neo_hookean_3d;

        const int m = static_cast<int>(color_vertices.size());
        if(m == 0)  // No vertices in the color group, exit early
            return;
        if(!v2t_built_)
            build_v2t_on_device_serial(info);  // Ensure vertex-to-tetrahedron mapping is built on GPU

        // Launch parallel threads: each thread processes one vertex in the color group
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(m,
                   [mus = mus.cviewer().name("mus"),  // Shear modulus array
                    lambdas = lambdas.cviewer().name("lambdas"),  // Lame's first parameter array
                    indices = info.indices().viewer().name("indices"),  // Tetrahedron vertex indices
                    xs = info.xs().viewer().name("xs"),  // Current vertex positions
                    Dm_invs = info.Dm_invs().viewer().name("Dm_invs"),  // Inverse reference deformation gradients
                    G3s = info.gradients().viewer().name("gradients"),  // Gradient buffer
                    H3x3s = info.hessians().viewer().name("hessians"),  // Hessian buffer
                    volumes = info.rest_volumes().viewer().name("volumes"),  // Rest volumes of tetrahedrons
                    dt = info.dt(),  // Time step
                    v2t_offs = d_v2t_offsets.cviewer().name("v2t_offs"),  // Vertex-to-tetrahedron offsets
                    v2t_list = d_v2t_list.cviewer().name("v2t_list"),  // Vertex-to-tetrahedron list
                    verts = color_vertices.viewer().name("verts")] __device__(int ki) mutable  // Index in color_vertices
                   {
                       const int v = static_cast<int>(verts(ki));  // Global ID of the current vertex in color group
                       if(v < 0)  // Invalid vertex ID, skip
                           return;

                       // Get the range of tetrahedrons associated with the current vertex
                       const int begin = v2t_offs(v);
                       const int end   = v2t_offs(v + 1);
                       if(end <= begin)  // No associated tetrahedrons, skip
                           return;

                       // Iterate over all first-ring tetrahedrons of the vertex and assemble gradients/Hessians
                       for(int p = begin; p < end; ++p)
                       {
                           const int I = v2t_list(p);  // Index of the tetrahedron adjacent to the current vertex

                           const Vector4i& tet = indices(I);  // 4 vertices of the tetrahedron
                           const Matrix3x3& Dm_inv = Dm_invs(I);  // Inverse reference deformation gradient
                           const Float mu = mus(I);  // Shear modulus of the tetrahedron
                           const Float lambda = lambdas(I);  // Lame's parameter of the tetrahedron

                           // Current positions of the tetrahedron's vertices
                           const Vector3& x0 = xs(tet(0));
                           const Vector3& x1 = xs(tet(1));
                           const Vector3& x2 = xs(tet(2));
                           const Vector3& x3 = xs(tet(3));

                           // Compute deformation gradient F
                           const auto F = fem::F(x0, x1, x2, x3, Dm_inv);
                           // Volume scaling factor (rest volume * dt square)
                           const auto Vdt2 = volumes(I) * dt * dt;

                           // Compute first and second derivatives of strain energy
                           Matrix3x3 dEdF;    // First derivative (dEnergy/dF)
                           Matrix9x9 ddEddF;  // Second derivative (d squareEnergy/dF square)
                           SNH::dEdVecF(dEdF, mu, lambda, F);  // Compute using stable neo-Hookean model
                           SNH::ddEddVecF(ddEddF, mu, lambda, F);  // Compute using stable neo-Hookean model

                           // Scale derivatives by volume and time step
                           auto VecdEdF = flatten(dEdF);  // Flatten 3x3 to 9x1 vector
                           VecdEdF *= Vdt2;
                           ddEddF *= Vdt2;

                           // Ensure Hessian is symmetric positive definite
                           make_spd(ddEddF);
                           // Derivative of F w.r.t. vertex positions (9x12 matrix)
                           Matrix9x12 dFdx = fem::dFdx(Dm_inv);
                           // Gradient vector (12x1: 3 components for each of 4 vertices)
                           Vector12 G = dFdx.transpose() * VecdEdF;
                           // Hessian matrix (12x12: second derivatives between vertices)
                           Matrix12x12 H = dFdx.transpose() * ddEddF * dFdx;

                           // Assume no write conflicts for tetrahedrons in the same color group
                           // Assemble gradient into global buffer
                           DoubletVectorAssembler DVA{G3s};
                           DVA.segment<4>(I * 4).write(tet, G);

                           // Assemble Hessian into global buffer
                           TripletMatrixAssembler TMA{H3x3s};
                           TMA.block<4, 4>(I * 4 * 4).write(tet, H);
                       }
                   })
            .wait();  // Wait for all threads in the color group to complete

        //// GPU implementation using CPU-like logic: iterate over tetrahedrons related to vertices in the color group
        //using namespace muda;
        //namespace SNH = sym::stable_neo_hookean_3d;

        //if(vertexId < 0)
        //    return;
        //if(!v2t_built_)
        //    build_v2t_on_device_serial(info);  // Ensure vertex-to-tetrahedron mapping is built on GPU

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
