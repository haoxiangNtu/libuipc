#include <time_integrator/bdf1_flag.h>
#include <finite_element/finite_element_kinetic.h>

namespace uipc::backend::cuda
{
class FiniteElementBDF1Kinetic final : public FiniteElementKinetic
{
  public:
    using FiniteElementKinetic::FiniteElementKinetic;

    virtual void do_build(BuildInfo& info) override
    {
        // require BDF1 integration flag
        require<BDF1Flag>();
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        // Compute kinetic energy
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.xs().size(),
                   [is_fixed = info.is_fixed().cviewer().name("is_fixed"),
                    xs       = info.xs().cviewer().name("xs"),
                    x_tildes = info.x_tildes().viewer().name("x_tildes"),
                    masses   = info.masses().cviewer().name("masses"),
                    Ks = info.energies().viewer().name("kinetic_energy")] __device__(int i) mutable
                   {
                       auto& K = Ks(i);
                       if(is_fixed(i))
                       {
                           K = 0.0;
                       }
                       else
                       {
                           const Vector3& x       = xs(i);
                           const Vector3& x_tilde = x_tildes(i);
                           Float          M       = masses(i);
                           Vector3        dx      = x - x_tilde;
                           K                      = 0.5 * M * dx.dot(dx);
                       }
                   });
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;

        // Kinetic
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.xs().size(),
                   [is_fixed = info.is_fixed().cviewer().name("is_fixed"),
                    xs       = info.xs().cviewer().name("xs"),
                    x_tildes = info.x_tildes().viewer().name("x_tildes"),
                    masses   = info.masses().cviewer().name("masses"),
                    G3s      = info.gradients().viewer().name("G3s"),
                    H3x3s = info.hessians().viewer().name("H3x3s")] __device__(int i) mutable
                   {
                       auto& m       = masses(i);
                       auto& x       = xs(i);
                       auto& x_tilde = x_tildes(i);

                       Vector3   G;
                       Matrix3x3 H;

                       if(is_fixed(i))  // fixed
                       {
                           G = Vector3::Zero();
                       }
                       else
                       {
                           G = m * (x - x_tilde);
                       }

                       H = masses(i) * Matrix3x3::Identity();

                       G3s(i).write(i, G);
                       H3x3s(i).write(i, i, H);
                   });
    }

    virtual void do_compute_gradient_hessian_by_vertex(FiniteElementKinetic::ComputeGradientHessianInfo& info,
                                                       IndexT vertexId) override
    {
        using namespace muda;

        // Only compute gradient and Hessian for the specified vertex (vertexId)
        Launch(1, 1).apply(
            [is_fixed = info.is_fixed().cviewer().name("is_fixed"),
             xs       = info.xs().cviewer().name("xs"),
             x_tildes = info.x_tildes().viewer().name("x_tildes"),
             masses   = info.masses().cviewer().name("masses"),
             G3s      = info.gradients().viewer().name("G3s"),
             H3x3s    = info.hessians().viewer().name("H3x3s"),
             vertexId = vertexId,
             n = info.xs().size()] __device__() mutable  // Capture total vertex count for boundary check
            {
                // Boundary check: ensure vertexId is in the valid range (0 <= vertexId < total vertices)
                if(vertexId < 0 || vertexId >= n)
                {
                    // Optional: print debug info
                    // print("Invalid vertexId: %d (total vertices: %d)\n", vertexId, n);
                    return;
                }

                // Directly use the given vertexId instead of looping
                int i = vertexId;

                // Compute gradient G and Hessian H for this vertex (same logic as original, but only for vertexId)
                auto& m       = masses(i);
                auto& x       = xs(i);
                auto& x_tilde = x_tildes(i);

                Vector3   G;
                Matrix3x3 H;

                if(is_fixed(i))  // Fixed vertex
                {
                    G = Vector3::Zero();
                }
                else  // Non-fixed vertex
                {
                    G = m * (x - x_tilde);
                }

                {
                    //const int   fixed = is_fixed(i) ? 1 : 0;
                    //const Float m     = masses(i);
                    //print("[BDF1Kinetic by vertex] i=%d fixed=%d m=%.6g G=(%.6g, %.6g, %.6g)\n",
                    //      i,
                    //      fixed,
                    //      (double)m,
                    //      (double)G(0),
                    //      (double)G(1),
                    //      (double)G(2));
                }

                H = masses(i) * Matrix3x3::Identity();

                // Write back gradient and Hessian for this vertex only
                G3s(i).write(i, G);
                H3x3s(i).write(i, i, H);
            });
    }

    virtual void do_compute_gradient_hessian_by_color(FiniteElementKinetic::ComputeGradientHessianInfo& info,
                                                      muda::CBufferView<IndexT> color_vertices) override
    {
        using namespace muda;
        const int n = static_cast<int>(info.xs().size());
        const int m = static_cast<int>(color_vertices.size());
        if(m == 0)
            return;

        // Each thread handles one vertex in color_vertices
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(m,
                   [is_fixed = info.is_fixed().cviewer().name("is_fixed"),
                    xs       = info.xs().cviewer().name("xs"),
                    x_tildes = info.x_tildes().viewer().name("x_tildes"),
                    masses   = info.masses().cviewer().name("masses"),
                    G3s      = info.gradients().viewer().name("G3s"),
                    H3x3s    = info.hessians().viewer().name("H3x3s"),
                    verts    = color_vertices.viewer().name("verts"),
                    n] __device__(int k) mutable
                   {
                       const int i = static_cast<int>(verts(k));
                       if(i < 0 || i >= n)
                           return;

                       Vector3   G;
                       Matrix3x3 H;

                       if(is_fixed(i))
                       {
                           G = Vector3::Zero();
                           //H = Matrix3x3::Zero();  // Or keep the mass matrix depending on your needs
                       }
                       else
                       {
                           const auto& x       = xs(i);
                           const auto& x_tilde = x_tildes(i);
                           const Float m       = masses(i);

                           G = m * (x - x_tilde);
                           //H = m * Matrix3x3::Identity();
                       }

                       // DEBUG: print (guard removed)
                       if(1)
                       {
                           const int   fixed = is_fixed(i) ? 1 : 0;
                           const Float m     = masses(i);
                           print("[BDF1Kinetic by color] k=%d i=%d fixed=%d m=%.6g G=(%.6g, %.6g, %.6g)\n",
                                 k,
                                 i,
                                 fixed,
                                 (double)m,
                                 (double)G(0),
                                 (double)G(1),
                                 (double)G(2));
                       }

                       H = masses(i) * Matrix3x3::Identity();
                       G3s(i).write(i, G);
                       H3x3s(i).write(i, i, H);
                   })
            .wait();
    }
};

REGISTER_SIM_SYSTEM(FiniteElementBDF1Kinetic);
}  // namespace uipc::backend::cuda
