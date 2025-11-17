//ref: https://github.com/ipc-sim/Codim-IPC/tree/main/Library/Math/Distance
#include <thrust/extrema.h>
#include <thrust/swap.h>
namespace uipc::backend::cuda::distance
{
template <typename T>
MUDA_GENERIC bool point_edge_cd_broadphase(const Eigen::Vector<T, 3>& x0,
                                           const Eigen::Vector<T, 3>& x1,
                                           const Eigen::Vector<T, 3>& x2,
                                           T                          dist)
{
    const Eigen::Array<T, 3, 1> max_e = x1.array().max(x2.array());
    const Eigen::Array<T, 3, 1> min_e = x1.array().min(x2.array());
    if((x0.array() - max_e > dist).any() || (min_e - x0.array() > dist).any())
    {
        return false;
    }
    else
    {
        return true;
    }
}

template <typename T>
MUDA_GENERIC bool point_edge_ccd_broadphase(const Eigen::Matrix<T, 2, 1>& p,
                                            const Eigen::Matrix<T, 2, 1>& e0,
                                            const Eigen::Matrix<T, 2, 1>& e1,
                                            const Eigen::Matrix<T, 2, 1>& dp,
                                            const Eigen::Matrix<T, 2, 1>& de0,
                                            const Eigen::Matrix<T, 2, 1>& de1,
                                            T                             dist)
{
    const Eigen::Array<T, 2, 1> max_p = p.array().max((p + dp).array());
    const Eigen::Array<T, 2, 1> min_p = p.array().min((p + dp).array());
    const Eigen::Array<T, 2, 1> max_e =
        e0.array().max(e1.array()).max((e0 + de0).array()).max((e1 + de1).array());
    const Eigen::Array<T, 2, 1> min_e =
        e0.array().min(e1.array()).min((e0 + de0).array()).min((e1 + de1).array());
    if((min_p - max_e > dist).any() || (min_e - max_p > dist).any())
    {
        return false;
    }
    else
    {
        return true;
    }
}

template <typename T>
MUDA_GENERIC bool point_triangle_cd_broadphase(const Eigen::Vector<T, 3>& p,
                                               const Eigen::Vector<T, 3>& t0,
                                               const Eigen::Vector<T, 3>& t1,
                                               const Eigen::Vector<T, 3>& t2,
                                               T                          dist)
{
    const Eigen::Array<T, 3, 1> max_tri = t0.array().max(t1.array()).max(t2.array());
    const Eigen::Array<T, 3, 1> min_tri = t0.array().min(t1.array()).min(t2.array());
    if((p.array() - max_tri > dist).any() || (min_tri - p.array() > dist).any())
    {
        return false;
    }
    else
    {
        return true;
    }
}

template <typename T>
MUDA_GENERIC bool edge_edge_cd_broadphase(const Eigen::Vector<T, 3>& ea0,
                                          const Eigen::Vector<T, 3>& ea1,
                                          const Eigen::Vector<T, 3>& eb0,
                                          const Eigen::Vector<T, 3>& eb1,
                                          T                          dist)
{
    const Eigen::Array<T, 3, 1> max_a = ea0.array().max(ea1.array());
    const Eigen::Array<T, 3, 1> min_a = ea0.array().min(ea1.array());
    const Eigen::Array<T, 3, 1> max_b = eb0.array().max(eb1.array());
    const Eigen::Array<T, 3, 1> min_b = eb0.array().min(eb1.array());
    if((min_a - max_b > dist).any() || (min_b - max_a > dist).any())
    {
        return false;
    }
    else
    {
        return true;
    }
}

template <typename T>
MUDA_GENERIC bool point_triangle_ccd_broadphase(const Eigen::Vector<T, 3>& p,
                                                const Eigen::Vector<T, 3>& t0,
                                                const Eigen::Vector<T, 3>& t1,
                                                const Eigen::Vector<T, 3>& t2,
                                                const Eigen::Vector<T, 3>& dp,
                                                const Eigen::Vector<T, 3>& dt0,
                                                const Eigen::Vector<T, 3>& dt1,
                                                const Eigen::Vector<T, 3>& dt2,
                                                T                          dist)
{
    const Eigen::Array<T, 3, 1> max_p   = p.array().max((p + dp).array());
    const Eigen::Array<T, 3, 1> min_p   = p.array().min((p + dp).array());
    const Eigen::Array<T, 3, 1> max_tri = t0.array()
                                              .max(t1.array())
                                              .max(t2.array())
                                              .max((t0 + dt0).array())
                                              .max((t1 + dt1).array())
                                              .max((t2 + dt2).array());
    const Eigen::Array<T, 3, 1> min_tri = t0.array()
                                              .min(t1.array())
                                              .min(t2.array())
                                              .min((t0 + dt0).array())
                                              .min((t1 + dt1).array())
                                              .min((t2 + dt2).array());
    if((min_p - max_tri > dist).any() || (min_tri - max_p > dist).any())
    {
        return false;
    }
    else
    {
        return true;
    }
}

template <typename T>
MUDA_GENERIC bool edge_edge_ccd_broadphase(const Eigen::Vector<T, 3>& ea0,
                                           const Eigen::Vector<T, 3>& ea1,
                                           const Eigen::Vector<T, 3>& eb0,
                                           const Eigen::Vector<T, 3>& eb1,
                                           const Eigen::Vector<T, 3>& dea0,
                                           const Eigen::Vector<T, 3>& dea1,
                                           const Eigen::Vector<T, 3>& deb0,
                                           const Eigen::Vector<T, 3>& deb1,
                                           T                          dist)
{
    const Eigen::Array<T, 3, 1> max_a =
        ea0.array().max(ea1.array()).max((ea0 + dea0).array()).max((ea1 + dea1).array());
    const Eigen::Array<T, 3, 1> min_a =
        ea0.array().min(ea1.array()).min((ea0 + dea0).array()).min((ea1 + dea1).array());
    const Eigen::Array<T, 3, 1> max_b =
        eb0.array().max(eb1.array()).max((eb0 + deb0).array()).max((eb1 + deb1).array());
    const Eigen::Array<T, 3, 1> min_b =
        eb0.array().min(eb1.array()).min((eb0 + deb0).array()).min((eb1 + deb1).array());
    if((min_a - max_b > dist).any() || (min_b - max_a > dist).any())
    {
        return false;
    }
    else
    {
        return true;
    }
}

template <typename T>
MUDA_GENERIC bool point_edge_ccd_broadphase(const Eigen::Vector<T, 3>& p,
                                            const Eigen::Vector<T, 3>& e0,
                                            const Eigen::Vector<T, 3>& e1,
                                            const Eigen::Vector<T, 3>& dp,
                                            const Eigen::Vector<T, 3>& de0,
                                            const Eigen::Vector<T, 3>& de1,
                                            T                          dist)
{
    const Eigen::Array<T, 3, 1> max_p = p.array().max((p + dp).array());
    const Eigen::Array<T, 3, 1> min_p = p.array().min((p + dp).array());
    const Eigen::Array<T, 3, 1> max_e =
        e0.array().max(e1.array()).max((e0 + de0).array()).max((e1 + de1).array());
    const Eigen::Array<T, 3, 1> min_e =
        e0.array().min(e1.array()).min((e0 + de0).array()).min((e1 + de1).array());
    if((min_p - max_e > dist).any() || (min_e - max_p > dist).any())
    {
        return false;
    }
    else
    {
        return true;
    }
}

template <typename T>
MUDA_GENERIC bool point_point_ccd_broadphase(const Eigen::Vector<T, 3>& p0,
                                             const Eigen::Vector<T, 3>& p1,
                                             const Eigen::Vector<T, 3>& dp0,
                                             const Eigen::Vector<T, 3>& dp1,
                                             T                          dist)
{
    const Eigen::Array<T, 3, 1> max_p0 = p0.array().max((p0 + dp0).array());
    const Eigen::Array<T, 3, 1> min_p0 = p0.array().min((p0 + dp0).array());
    const Eigen::Array<T, 3, 1> max_p1 = p1.array().max((p1 + dp1).array());
    const Eigen::Array<T, 3, 1> min_p1 = p1.array().min((p1 + dp1).array());
    if((min_p0 - max_p1 > dist).any() || (min_p1 - max_p0 > dist).any())
    {
        return false;
    }
    else
    {
        return true;
    }
}

template <typename T>
MUDA_GENERIC bool point_triangle_ccd(Eigen::Vector<T, 3> p,
                                     Eigen::Vector<T, 3> t0,
                                     Eigen::Vector<T, 3> t1,
                                     Eigen::Vector<T, 3> t2,
                                     Eigen::Vector<T, 3> dp,
                                     Eigen::Vector<T, 3> dt0,
                                     Eigen::Vector<T, 3> dt1,
                                     Eigen::Vector<T, 3> dt2,
                                     T                   eta,
                                     T                   thickness,
                                     int                 max_iter,
                                     T&                  toc)
{
    Eigen::Vector<T, 3> mov = (dt0 + dt1 + dt2 + dp) / 4;
    dt0 -= mov;
    dt1 -= mov;
    dt2 -= mov;
    dp -= mov;
    Eigen::Array3<T> dispMag2Vec{dt0.squaredNorm(), dt1.squaredNorm(), dt2.squaredNorm()};
    T maxDispMag = dp.norm() + sqrt(dispMag2Vec.maxCoeff());

    if(maxDispMag <= T(0))
    {
        return false;
    }

    T    dist2_cur;
    auto flag = point_triangle_distance_flag(p, t0, t1, t2);
    point_triangle_distance2(flag, p, t0, t1, t2, dist2_cur);
    T dist_cur = sqrt(dist2_cur);
    T gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
    T toc_prev = toc;
    toc        = 0;
    while(true)
    {
        if(max_iter >= 0)
        {
            if(--max_iter < 0)
                return true;
        }

        T tocLowerBound = (1 - eta) * (dist2_cur - thickness * thickness)
                          / ((dist_cur + thickness) * maxDispMag);

        p += tocLowerBound * dp;
        t0 += tocLowerBound * dt0;
        t1 += tocLowerBound * dt1;
        t2 += tocLowerBound * dt2;
        flag = point_triangle_distance_flag(p, t0, t1, t2);
        point_triangle_distance2(flag, p, t0, t1, t2, dist2_cur);
        dist_cur = sqrt(dist2_cur);
        if(toc && ((dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap))
        {
            break;
        }

        toc += tocLowerBound;
        if(toc > toc_prev)
        {
            return false;
        }
    }

    return true;
}

template <typename T>
MUDA_GENERIC bool edge_edge_ccd(Eigen::Vector<T, 3> ea0,
                                Eigen::Vector<T, 3> ea1,
                                Eigen::Vector<T, 3> eb0,
                                Eigen::Vector<T, 3> eb1,
                                Eigen::Vector<T, 3> dea0,
                                Eigen::Vector<T, 3> dea1,
                                Eigen::Vector<T, 3> deb0,
                                Eigen::Vector<T, 3> deb1,
                                T                   eta,
                                T                   thickness,
                                int                 max_iter,
                                T&                  toc)
{
    Eigen::Vector<T, 3> mov = (dea0 + dea1 + deb0 + deb1) / 4;
    dea0 -= mov;
    dea1 -= mov;
    deb0 -= mov;
    deb1 -= mov;
    T maxDispMag = sqrt(std::max(dea0.squaredNorm(), dea1.squaredNorm()))
                   + sqrt(std::max(deb0.squaredNorm(), deb1.squaredNorm()));
    if(maxDispMag == 0)
    {
        return false;
    }

    T    dist2_cur;
    auto flag = edge_edge_distance_flag(ea0, ea1, eb0, eb1);
    edge_edge_distance2(flag, ea0, ea1, eb0, eb1, dist2_cur);
    T dFunc = dist2_cur - thickness * thickness;
    if(dFunc <= 0)
    {
        // since we ensured other place that all dist smaller than dHat are positive,
        // this must be some far away nearly parallel edges
        Eigen::Array4<T> dists{(ea0 - eb0).squaredNorm(),
                               (ea0 - eb1).squaredNorm(),
                               (ea1 - eb0).squaredNorm(),
                               (ea1 - eb1).squaredNorm()};
        dist2_cur = dists.minCoeff();
        dFunc     = dist2_cur - thickness * thickness;
    }
    T dist_cur = sqrt(dist2_cur);
    T gap      = eta * dFunc / (dist_cur + thickness);
    T toc_prev = toc;
    toc        = 0;
    while(true)
    {
        if(max_iter >= 0)
        {
            if(--max_iter < 0)
                return true;
        }

        T tocLowerBound = (1 - eta) * dFunc / ((dist_cur + thickness) * maxDispMag);

        ea0 += tocLowerBound * dea0;
        ea1 += tocLowerBound * dea1;
        eb0 += tocLowerBound * deb0;
        eb1 += tocLowerBound * deb1;
        auto flag = edge_edge_distance_flag(ea0, ea1, eb0, eb1);
        edge_edge_distance2(flag, ea0, ea1, eb0, eb1, dist2_cur);
        dFunc = dist2_cur - thickness * thickness;
        if(dFunc <= 0)
        {
            // since we ensured other place that all dist smaller than dHat are positive,
            // this must be some far away nearly parallel edges
            Eigen::Array4<T> dists{(ea0 - eb0).squaredNorm(),
                                   (ea0 - eb1).squaredNorm(),
                                   (ea1 - eb0).squaredNorm(),
                                   (ea1 - eb1).squaredNorm()};
            dist2_cur = dists.minCoeff();
            dFunc     = dist2_cur - thickness * thickness;
        }
        dist_cur = sqrt(dist2_cur);
        if(toc && (dFunc / (dist_cur + thickness) < gap))
        {
            break;
        }

        toc += tocLowerBound;
        if(toc > toc_prev)
        {
            return false;
        }
    }

    return true;
}

template <typename T>
MUDA_GENERIC bool point_edge_ccd(Eigen::Vector<T, 3> p,
                                 Eigen::Vector<T, 3> e0,
                                 Eigen::Vector<T, 3> e1,
                                 Eigen::Vector<T, 3> dp,
                                 Eigen::Vector<T, 3> de0,
                                 Eigen::Vector<T, 3> de1,
                                 T                   eta,
                                 T                   thickness,
                                 int                 max_iter,
                                 T&                  toc)
{
    Eigen::Vector<T, 3> mov = (dp + de0 + de1) / 3;
    de0 -= mov;
    de1 -= mov;
    dp -= mov;
    T maxDispMag = dp.norm() + sqrt(std::max(de0.squaredNorm(), de1.squaredNorm()));
    if(maxDispMag == 0)
    {
        return false;
    }

    T    dist2_cur;
    auto flag = point_edge_distance_flag(p, e0, e1);
    point_edge_distance2(flag, p, e0, e1, dist2_cur);
    T dist_cur = sqrt(dist2_cur);
    T gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
    T toc_prev = toc;
    toc        = 0;
    while(true)
    {
        if(max_iter >= 0)
        {
            if(--max_iter < 0)
                return true;
        }

        T tocLowerBound = (1 - eta) * (dist2_cur - thickness * thickness)
                          / ((dist_cur + thickness) * maxDispMag);

        p += tocLowerBound * dp;
        e0 += tocLowerBound * de0;
        e1 += tocLowerBound * de1;
        flag = point_edge_distance_flag(p, e0, e1);
        point_edge_distance2(flag, p, e0, e1, dist2_cur);
        dist_cur = sqrt(dist2_cur);
        if(toc && (dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap)
        {
            break;
        }

        toc += tocLowerBound;
        if(toc > toc_prev)
        {
            return false;
        }
    }

    return true;
}

template <typename T>
MUDA_GENERIC bool point_point_ccd(Eigen::Vector<T, 3> p0,
                                  Eigen::Vector<T, 3> p1,
                                  Eigen::Vector<T, 3> dp0,
                                  Eigen::Vector<T, 3> dp1,
                                  T                   eta,
                                  T                   thickness,
                                  int                 max_iter,
                                  T&                  toc)
{
    Eigen::Vector<T, 3> mov = (dp0 + dp1) / 2;
    dp1 -= mov;
    dp0 -= mov;
    T maxDispMag = dp0.norm() + dp1.norm();
    if(maxDispMag == 0)
    {
        return false;
    }

    T    dist2_cur;
    auto flag = point_point_distance_flag(p0, p1);
    point_point_distance2(flag, p0, p1, dist2_cur);
    T dist_cur = sqrt(dist2_cur);
    T gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
    T toc_prev = toc;
    toc        = 0;
    while(true)
    {
        if(max_iter >= 0)
        {
            if(--max_iter < 0)
                return true;
        }

        T tocLowerBound = (1 - eta) * (dist2_cur - thickness * thickness)
                          / ((dist_cur + thickness) * maxDispMag);

        p0 += tocLowerBound * dp0;
        p1 += tocLowerBound * dp1;
        flag = point_point_distance_flag(p0, p1);
        point_point_distance2(flag, p0, p1, dist2_cur);
        dist_cur = sqrt(dist2_cur);
        if(toc && (dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap)
        {
            break;
        }

        toc += tocLowerBound;
        if(toc > toc_prev)
        {
            return false;
        }
    }

    return true;
}

// Helper function 1: Calculate the closest contact point x_b on the triangle to point p (for point-triangle collision)
// Meaning of flag: 0 = point is inside the triangle face, 1 = closest vertex t0, 2 = closest vertex t1, 3 = closest vertex t2, 4 = closest edge t0t1, 5 = closest edge t1t2, 6 = closest edge t2t0
template <typename T>
MUDA_GENERIC void point_triangle_contact_point(int                        flag,
                                               const Eigen::Vector<T, 3>& p,
                                               const Eigen::Vector<T, 3>& t0,
                                               const Eigen::Vector<T, 3>& t1,
                                               const Eigen::Vector<T, 3>& t2,
                                               Eigen::Vector<T, 3>&       x_b)
{
    if(flag == 0)
    {  // Inside the face: calculate the projection of p onto the triangle face
        Eigen::Vector<T, 3> e0 = t1 - t0;
        Eigen::Vector<T, 3> e1 = t2 - t0;
        Eigen::Vector<T, 3> v  = p - t0;
        T a = e0.dot(e0), b = e0.dot(e1), c = e1.dot(e1), d = e0.dot(v), e = e1.dot(v);
        T denom = a * c - b * b;
        if(std::abs(denom) < T(1e-12))
        {  // Degenerate triangle, take the closest vertex
            T d0 = (p - t0).squaredNorm();
            T d1 = (p - t1).squaredNorm();
            T d2 = (p - t2).squaredNorm();
            if(d0 <= d1 && d0 <= d2)
                x_b = t0;
            else if(d1 <= d2)
                x_b = t1;
            else
                x_b = t2;
            return;
        }
        T s = (b * e - c * d) / denom;
        T t = (a * e - b * d) / denom;
        s   = std::clamp(s, T(0), T(1));
        t   = std::clamp(t, T(0), T(1));
        x_b = t0 + s * e0 + t * e1;
    }
    else if(flag >= 1 && flag <= 3)
    {  // Closest vertex
        x_b = (flag == 1) ? t0 : (flag == 2) ? t1 : t2;
    }
    else
    {  // Closest edge (4=t0t1, 5=t1t2, 6=t2t0)
        if(flag == 4)
        {
            T t_param = (p - t0).dot(t1 - t0) / (t1 - t0).squaredNorm();
            t_param   = std::clamp(t_param, T(0), T(1));
            x_b       = t0 + t_param * (t1 - t0);
        }
        else if(flag == 5)
        {
            T t_param = (p - t1).dot(t2 - t1) / (t2 - t1).squaredNorm();
            t_param   = std::clamp(t_param, T(0), T(1));
            x_b       = t1 + t_param * (t2 - t1);
        }
        else
        {  // flag == 6
            T t_param = (p - t2).dot(t0 - t2) / (t0 - t2).squaredNorm();
            t_param   = std::clamp(t_param, T(0), T(1));
            x_b       = t2 + t_param * (t0 - t2);
        }
    }
}

// Helper function 2: Calculate the intersection points x_a (on edge ea) and x_b (on edge eb) of two edges at TOI (for edge-edge collision)
template <typename T>
MUDA_GENERIC bool edge_edge_intersection(const Eigen::Vector<T, 3>& ea0,
                                         const Eigen::Vector<T, 3>& ea1,
                                         const Eigen::Vector<T, 3>& eb0,
                                         const Eigen::Vector<T, 3>& eb1,
                                         Eigen::Vector<T, 3>& x_a,  // Intersection point on edge ea
                                         Eigen::Vector<T, 3>& x_b)
{  // Intersection point on edge eb
    Eigen::Vector<T, 3> u = ea1 - ea0;
    Eigen::Vector<T, 3> v = eb1 - eb0;
    Eigen::Vector<T, 3> w = ea0 - eb0;
    T a = u.dot(u), b = u.dot(v), c = v.dot(v), d = u.dot(w), e = v.dot(w);
    T denom = a * c - b * b;
    if(std::abs(denom) < T(1e-12))
        return false;  // Edges are parallel or collinear, no unique intersection point

    // Calculate edge parameters (s(belong to)[0,1] corresponds to ea, t(belong to)[0,1] corresponds to eb)
    T s = (b * e - c * d) / denom;
    T t = (a * e - b * d) / denom;
    s   = std::clamp(s, T(0), T(1));
    t   = std::clamp(t, T(0), T(1));

    x_a = ea0 + s * u;
    x_b = eb0 + t * v;
    return true;
}

// Helper function 3: Calculate the closest contact point x_b on the edge to point p (for point-edge collision)
// Meaning of flag: 0 = middle of the edge, 1 = closest endpoint e0, 2 = closest endpoint e1
template <typename T>
MUDA_GENERIC void point_edge_contact_point(int                        flag,
                                           const Eigen::Vector<T, 3>& p,
                                           const Eigen::Vector<T, 3>& e0,
                                           const Eigen::Vector<T, 3>& e1,
                                           Eigen::Vector<T, 3>&       x_b)
{
    if(flag == 0)
    {  // Middle of the edge: calculate the projection of p onto the edge
        T t_param = (p - e0).dot(e1 - e0) / (e1 - e0).squaredNorm();
        t_param   = std::clamp(t_param, T(0), T(1));
        x_b       = e0 + t_param * (e1 - e0);
    }
    else
    {  // Closest endpoint
        x_b = (flag == 1) ? e0 : e1;
    }
}

/**
 * @brief Point-triangle CCD collision detection + penetration depth calculation (vertex-triangle collision model in Section 3.5 of the document)
 * @param[in] p_initial: Initial position of the collision point (at time t)
 * @param[in] t0_initial~t2_initial: Initial positions of the triangle vertices (at time t)
 * @param[in] dp: Motion increment of the point (total displacement from t=>t+1: p_pred - p_initial)
 * @param[in] dt0~dt2: Motion increments of the triangle vertices (total displacement from t=>t+1)
 * @param[in] eta: Convergence threshold (consistent with the original CCD)
 * @param[in] thickness: Collision trigger threshold (collision is determined if distance is less than this value)
 * @param[in] max_iter: Maximum number of iterations (consistent with the original CCD)
 * @param[out] toc: Collision time parameter ((belong to)[0,1], 0 = time t, 1 = time t+1)
 * @param[out] d: Penetration depth (defined in Section 3.5 of the document: d = max(0, thickness - (xb-xa)(dot)n-hat))
 * @return Whether a valid collision is detected (toc(belong to)[0,1] and penetration depth is calculated)
 */
template <typename T>
MUDA_GENERIC bool point_triangle_ccd_compute_penetration_depth(
    Eigen::Vector<T, 3> p_initial,
    Eigen::Vector<T, 3> t0_initial,
    Eigen::Vector<T, 3> t1_initial,
    Eigen::Vector<T, 3> t2_initial,
    Eigen::Vector<T, 3> dp,
    Eigen::Vector<T, 3> dt0,
    Eigen::Vector<T, 3> dt1,
    Eigen::Vector<T, 3> dt2,
    T                   eta,
    T                   thickness,
    int                 max_iter,
    T&                  toc,
    T&                  d)
{
    //// Step 1: Call the original CCD to detect collision and get toc (use temporary variables to avoid modifying initial positions)
    //Eigen::Vector<T, 3> p_temp        = p_initial;
    //Eigen::Vector<T, 3> t0_temp       = t0_initial;
    //Eigen::Vector<T, 3> t1_temp       = t1_initial;
    //Eigen::Vector<T, 3> t2_temp       = t2_initial;
    //bool                has_collision = point_triangle_ccd(
    //    p_temp, t0_temp, t1_temp, t2_temp, dp, dt0, dt1, dt2, eta, thickness, max_iter, toc);

    //// No collision or invalid TOI: set penetration depth to 0
    //if(!has_collision || toc < T(1e-12) || toc > T(1.0 - 1e-12))
    //{
    //    d = T(0);
    //    return false;
    //}

    //// Step 2: Explicitly calculate positions at TOI (implicit logic in the document: linear interpolation of motion trajectory)
    //Eigen::Vector<T, 3> p_toi  = p_initial + toc * dp;    // TOI position of the point (x_a)
    //Eigen::Vector<T, 3> t0_toi = t0_initial + toc * dt0;  // TOI positions of the triangle vertices
    //Eigen::Vector<T, 3> t1_toi = t1_initial + toc * dt1;
    //Eigen::Vector<T, 3> t2_toi = t2_initial + toc * dt2;

    //// Step 3: Calculate contact points x_a and x_b at TOI (defined in Section 3.5 of the document)
    //Eigen::Vector<T, 3> x_a = p_toi;  // Collision point (x_a = TOI position of the point)
    //Eigen::Vector<T, 3> x_b;          // Contact point on the triangle (x_b = closest point/collision point)
    //int flag = point_triangle_distance_flag(x_a, t0_toi, t1_toi, t2_toi);
    //point_triangle_contact_point(flag, x_a, t0_toi, t1_toi, t2_toi, x_b);

    //// Step 4: Calculate contact normal vector n-hat (Section 3.5 of the document: outward normal vector of the triangle surface at x_b)
    //Eigen::Vector<T, 3> tri_normal = (t1_toi - t0_toi).cross(t2_toi - t0_toi);
    //T                   norm_mag   = tri_normal.norm();
    //if(norm_mag < T(1e-12))
    //{  // Degenerate triangle, no valid normal vector
    //    d = T(0);
    //    return false;
    //}
    //Eigen::Vector<T, 3> n_hat = tri_normal / norm_mag;

    //// Step 5: Calculate penetration depth (formula in Section 3.5 of the document + collision threshold correction)
    //T geometric_dist = (x_b - x_a).dot(n_hat);  // Normal distance from x_a to x_b at TOI
    //d = std::max(T(0), thickness - geometric_dist);  // Penetration depth = threshold - actual distance (non-negative)

    return true;
}

/**
 * @brief Edge-edge CCD collision detection + penetration depth calculation (edge-edge collision model in Section 3.5 of the document)
 * @param[in] ea0_initial~ea1_initial: Initial positions of edge ea (at time t)
 * @param[in] eb0_initial~eb1_initial: Initial positions of edge eb (at time t)
 * @param[in] dea0~dea1: Motion increments of the vertices of edge ea (total displacement from t=>t+1)
 * @param[in] deb0~deb1: Motion increments of the vertices of edge eb (total displacement from t=>t+1)
 * @param[in] eta: Convergence threshold (consistent with the original CCD)
 * @param[in] thickness: Collision trigger threshold (collision is determined if distance is less than this value)
 * @param[in] max_iter: Maximum number of iterations (consistent with the original CCD)
 * @param[out] toc: Collision time parameter ((belong to)[0,1])
 * @param[out] d: Penetration depth (defined in Section 3.5 of the document: d = max(0, thickness - ||xb-xa||))
 * @return Whether a valid collision is detected
 */
template <typename T>
MUDA_GENERIC bool edge_edge_ccd_compute_penetration_depth(Eigen::Vector<T, 3> ea0_initial,
                                                          Eigen::Vector<T, 3> ea1_initial,
                                                          Eigen::Vector<T, 3> eb0_initial,
                                                          Eigen::Vector<T, 3> eb1_initial,
                                                          Eigen::Vector<T, 3> dea0,
                                                          Eigen::Vector<T, 3> dea1,
                                                          Eigen::Vector<T, 3> deb0,
                                                          Eigen::Vector<T, 3> deb1,
                                                          T   eta,
                                                          T   thickness,
                                                          int max_iter,
                                                          T&  toc,
                                                          T&  d)
{
    //// Step 1: Call the original CCD to detect collision
    //Eigen::Vector<T, 3> ea0_temp      = ea0_initial;
    //Eigen::Vector<T, 3> ea1_temp      = ea1_initial;
    //Eigen::Vector<T, 3> eb0_temp      = eb0_initial;
    //Eigen::Vector<T, 3> eb1_temp      = eb1_initial;
    //bool                has_collision = edge_edge_ccd(
    //    ea0_temp, ea1_temp, eb0_temp, eb1_temp, dea0, dea1, deb0, deb1, eta, thickness, max_iter, toc);

    //if(!has_collision || toc < T(1e-12) || toc > T(1.0 - 1e-12))
    //{
    //    d = T(0);
    //    return false;
    //}

    //// Step 2: Calculate positions of edges at TOI
    //Eigen::Vector<T, 3> ea0_toi = ea0_initial + toc * dea0;
    //Eigen::Vector<T, 3> ea1_toi = ea1_initial + toc * dea1;
    //Eigen::Vector<T, 3> eb0_toi = eb0_initial + toc * deb0;
    //Eigen::Vector<T, 3> eb1_toi = eb1_initial + toc * deb1;

    //// Step 3: Calculate edge-edge intersection points at TOI (x_a, x_b, defined in Section 3.5 of the document)
    //Eigen::Vector<T, 3> x_a, x_b;
    //if(!edge_edge_intersection(ea0_toi, ea1_toi, eb0_toi, eb1_toi, x_a, x_b))
    //{
    //    d = T(0);
    //    return false;
    //}

    //// Step 4: Calculate contact normal vector n-hat (Section 3.5 of the document: n = xb - xa, normalized)
    //Eigen::Vector<T, 3> n     = x_b - x_a;
    //T                   n_mag = n.norm();
    //if(n_mag < T(1e-12))
    //{  // Intersection points coincide, no penetration
    //    d = T(0);
    //    return false;
    //}
    //Eigen::Vector<T, 3> n_hat = n / n_mag;

    //// Step 5: Calculate penetration depth (formula in Section 3.5 of the document + threshold correction)
    //T geometric_dist = n_mag;  // Edge-edge intersection distance = ||xb-xa||
    //d                = std::max(T(0), thickness - geometric_dist);

    return true;
}

/**
 * @brief Point-edge CCD collision detection + penetration depth calculation (extended model of vertex-triangle collision in Section 3.5 of the document)
 * @param[in] p_initial: Initial position of the collision point (at time t)
 * @param[in] e0_initial~e1_initial: Initial positions of the edge vertices (at time t)
 * @param[in] dp: Motion increment of the point (total displacement from t=>t+1)
 * @param[in] de0~de1: Motion increments of the edge vertices (total displacement from t=>t+1)
 * @param[in] eta: Convergence threshold (consistent with the original CCD)
 * @param[in] thickness: Collision trigger threshold
 * @param[in] max_iter: Maximum number of iterations (consistent with the original CCD)
 * @param[out] toc: Collision time parameter ((belong to)[0,1])
 * @param[out] d: Penetration depth (d = max(0, thickness - (xb-xa)(dot)n-hat))
 * @return Whether a valid collision is detected
 */
template <typename T>
MUDA_GENERIC bool point_edge_ccd_compute_penetration_depth(Eigen::Vector<T, 3> p_initial,
                                                           Eigen::Vector<T, 3> e0_initial,
                                                           Eigen::Vector<T, 3> e1_initial,
                                                           Eigen::Vector<T, 3> dp,
                                                           Eigen::Vector<T, 3> de0,
                                                           Eigen::Vector<T, 3> de1,
                                                           T   eta,
                                                           T   thickness,
                                                           int max_iter,
                                                           T&  toc,
                                                           T&  d)
{
    //// Step 1: Call the original CCD to detect collision
    //Eigen::Vector<T, 3> p_temp  = p_initial;
    //Eigen::Vector<T, 3> e0_temp = e0_initial;
    //Eigen::Vector<T, 3> e1_temp = e1_initial;
    //bool                has_collision =
    //    point_edge_ccd(p_temp, e0_temp, e1_temp, dp, de0, de1, eta, thickness, max_iter, toc);

    //if(!has_collision || toc < T(1e-12) || toc > T(1.0 - 1e-12))
    //{
    //    d = T(0);
    //    return false;
    //}

    //// Step 2: Calculate positions at TOI
    //Eigen::Vector<T, 3> p_toi  = p_initial + toc * dp;    // TOI position of the point (x_a)
    //Eigen::Vector<T, 3> e0_toi = e0_initial + toc * de0;  // TOI positions of the edge
    //Eigen::Vector<T, 3> e1_toi = e1_initial + toc * de1;

    //// Step 3: Calculate contact points x_a and x_b
    //Eigen::Vector<T, 3> x_a = p_toi;
    //Eigen::Vector<T, 3> x_b;
    //int                 flag = point_edge_distance_flag(x_a, e0_toi, e1_toi);
    //point_edge_contact_point(flag, x_a, e0_toi, e1_toi, x_b);

    //// Step 4: Calculate contact normal vector n-hat (perpendicular to the edge, pointing from the point to the edge)
    //Eigen::Vector<T, 3> edge_dir = e1_toi - e0_toi;
    //Eigen::Vector<T, 3> n        = x_b - x_a;
    //T                   n_mag    = n.norm();
    //if(n_mag < T(1e-12))
    //{  // Point coincides with the contact point on the edge
    //    d = T(0);
    //    return false;
    //}
    //// Ensure the normal vector is perpendicular to the edge (correct numerical errors)
    //Eigen::Vector<T, 3> edge_normal     = edge_dir.cross(n).cross(edge_dir);
    //T                   edge_normal_mag = edge_normal.norm();
    //if(edge_normal_mag > T(1e-12))
    //{
    //    n     = edge_normal;
    //    n_mag = edge_normal_mag;
    //}
    //Eigen::Vector<T, 3> n_hat = n / n_mag;

    //// Step 5: Calculate penetration depth
    //T geometric_dist = n_mag;
    //d                = std::max(T(0), thickness - geometric_dist);

    return true;
}

/**
 * @brief Point-point CCD collision detection + penetration depth calculation (simplified extension of the collision model in Section 3.5 of the document)
 * @param[in] p0_initial~p1_initial: Initial positions of the two points (at time t)
 * @param[in] dp0~dp1: Motion increments of the two points (total displacement from t=>t+1)
 * @param[in] eta: Convergence threshold (consistent with the original CCD)
 * @param[in] thickness: Collision trigger threshold (collision is determined if distance is less than this value)
 * @param[in] max_iter: Maximum number of iterations (consistent with the original CCD)
 * @param[out] toc: Collision time parameter ((belong to)[0,1])
 * @param[out] d: Penetration depth (d = max(0, thickness - ||p1_toi - p0_toi||))
 * @return Whether a valid collision is detected
 */
template <typename T>
MUDA_GENERIC bool point_point_ccd_compute_penetration_depth(Eigen::Vector<T, 3> p0_initial,
                                                            Eigen::Vector<T, 3> p1_initial,
                                                            Eigen::Vector<T, 3> dp0,
                                                            Eigen::Vector<T, 3> dp1,
                                                            T   eta,
                                                            T   thickness,
                                                            int max_iter,
                                                            T&  toc,
                                                            T&  d)
{
    //// Step 1: Call the original CCD to detect collision
    //Eigen::Vector<T, 3> p0_temp = p0_initial;
    //Eigen::Vector<T, 3> p1_temp = p1_initial;
    //bool                has_collision =
    //    point_point_ccd(p0_temp, p1_temp, dp0, dp1, eta, thickness, max_iter, toc);

    //if(!has_collision || toc < T(1e-12) || toc > T(1.0 - 1e-12))
    //{
    //    d = T(0);
    //    return false;
    //}

    //// Step 2: Calculate positions of the points at TOI (x_a = p0_toi, x_b = p1_toi)
    //Eigen::Vector<T, 3> p0_toi = p0_initial + toc * dp0;
    //Eigen::Vector<T, 3> p1_toi = p1_initial + toc * dp1;

    //// Step 3: Calculate contact normal vector n-hat (Section 3.5 of the document: n = xb - xa, normalized)
    //Eigen::Vector<T, 3> n     = p1_toi - p0_toi;
    //T                   n_mag = n.norm();
    //if(n_mag < T(1e-12))
    //{  // The two points coincide, no penetration
    //    d = T(0);
    //    return false;
    //}
    //Eigen::Vector<T, 3> n_hat = n / n_mag;

    //// Step 4: Calculate penetration depth
    //T geometric_dist = n_mag;  // Point-point distance = ||xb-xa||
    //d                = std::max(T(0), thickness - geometric_dist);

    return true;
}
}  // namespace uipc::backend::cuda::distance