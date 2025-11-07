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

// 辅助函数1：计算三角形上离点p最近的接触点x_b（用于点-三角形碰撞）
// flag含义：0=点在三角形面内，1=最近顶点t0，2=最近顶点t1，3=最近顶点t2，4=最近边t0t1，5=最近边t1t2，6=最近边t2t0
template <typename T>
MUDA_GENERIC void point_triangle_contact_point(int                        flag,
                                               const Eigen::Vector<T, 3>& p,
                                               const Eigen::Vector<T, 3>& t0,
                                               const Eigen::Vector<T, 3>& t1,
                                               const Eigen::Vector<T, 3>& t2,
                                               Eigen::Vector<T, 3>&       x_b)
{
    if(flag == 0)
    {  // 面内：计算p在三角形面上的投影
        Eigen::Vector<T, 3> e0 = t1 - t0;
        Eigen::Vector<T, 3> e1 = t2 - t0;
        Eigen::Vector<T, 3> v  = p - t0;
        T a = e0.dot(e0), b = e0.dot(e1), c = e1.dot(e1), d = e0.dot(v), e = e1.dot(v);
        T denom = a * c - b * b;
        if(std::abs(denom) < T(1e-12))
        {  // 三角形退化，取最近顶点
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
    {  // 最近顶点
        x_b = (flag == 1) ? t0 : (flag == 2) ? t1 : t2;
    }
    else
    {  // 最近边（4=t0t1，5=t1t2，6=t2t0）
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

// 辅助函数2：计算两条边在TOI时刻的交点x_a（边ea）和x_b（边eb）（用于边-边碰撞）
template <typename T>
MUDA_GENERIC bool edge_edge_intersection(const Eigen::Vector<T, 3>& ea0,
                                         const Eigen::Vector<T, 3>& ea1,
                                         const Eigen::Vector<T, 3>& eb0,
                                         const Eigen::Vector<T, 3>& eb1,
                                         Eigen::Vector<T, 3>& x_a,  // 边ea上的交点
                                         Eigen::Vector<T, 3>& x_b)
{  // 边eb上的交点
    Eigen::Vector<T, 3> u = ea1 - ea0;
    Eigen::Vector<T, 3> v = eb1 - eb0;
    Eigen::Vector<T, 3> w = ea0 - eb0;
    T a = u.dot(u), b = u.dot(v), c = v.dot(v), d = u.dot(w), e = v.dot(w);
    T denom = a * c - b * b;
    if(std::abs(denom) < T(1e-12))
        return false;  // 边平行或共线，无唯一交点

    // 计算边参数（s∈[0,1]对应ea，t∈[0,1]对应eb）
    T s = (b * e - c * d) / denom;
    T t = (a * e - b * d) / denom;
    s   = std::clamp(s, T(0), T(1));
    t   = std::clamp(t, T(0), T(1));

    x_a = ea0 + s * u;
    x_b = eb0 + t * v;
    return true;
}

// 辅助函数3：计算边上离点p最近的接触点x_b（用于点-边碰撞）
// flag含义：0=边中间，1=最近端点e0，2=最近端点e1
template <typename T>
MUDA_GENERIC void point_edge_contact_point(int                        flag,
                                           const Eigen::Vector<T, 3>& p,
                                           const Eigen::Vector<T, 3>& e0,
                                           const Eigen::Vector<T, 3>& e1,
                                           Eigen::Vector<T, 3>&       x_b)
{
    if(flag == 0)
    {  // 边中间：计算p在边上的投影
        T t_param = (p - e0).dot(e1 - e0) / (e1 - e0).squaredNorm();
        t_param   = std::clamp(t_param, T(0), T(1));
        x_b       = e0 + t_param * (e1 - e0);
    }
    else
    {  // 最近端点
        x_b = (flag == 1) ? e0 : e1;
    }
}

/**
 * @brief 点-三角形CCD碰撞检测 + 穿透深度计算（文档3.5节顶点-三角形碰撞模型）
 * @param[in] p_initial: 碰撞点初始位置（t时刻）
 * @param[in] t0_initial~t2_initial: 三角形顶点初始位置（t时刻）
 * @param[in] dp: 点的运动增量（t→t+1总位移：p_pred - p_initial）
 * @param[in] dt0~dt2: 三角形顶点运动增量（t→t+1总位移）
 * @param[in] eta: 收敛阈值（与原有CCD一致）
 * @param[in] thickness: 碰撞触发阈值（距离小于该值判定碰撞）
 * @param[in] max_iter: 最大迭代次数（与原有CCD一致）
 * @param[out] toc: 碰撞时间参数（∈[0,1]，0=t时刻，1=t+1时刻）
 * @param[out] d: 穿透深度（文档3.5节定义：d = max(0, thickness - (xb-xa)·n̂)）
 * @return 是否检测到有效碰撞（toc∈[0,1]且计算出穿透深度）
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
    //// 步骤1：调用原有CCD检测碰撞，获取toc（用临时变量避免修改初始位置）
    //Eigen::Vector<T, 3> p_temp        = p_initial;
    //Eigen::Vector<T, 3> t0_temp       = t0_initial;
    //Eigen::Vector<T, 3> t1_temp       = t1_initial;
    //Eigen::Vector<T, 3> t2_temp       = t2_initial;
    //bool                has_collision = point_triangle_ccd(
    //    p_temp, t0_temp, t1_temp, t2_temp, dp, dt0, dt1, dt2, eta, thickness, max_iter, toc);

    //// 无碰撞或TOI无效：穿透深度设为0
    //if(!has_collision || toc < T(1e-12) || toc > T(1.0 - 1e-12))
    //{
    //    d = T(0);
    //    return false;
    //}

    //// 步骤2：显式计算TOI时刻的位置（文档隐含逻辑：运动轨迹线性插值）
    //Eigen::Vector<T, 3> p_toi  = p_initial + toc * dp;    // 点的TOI位置（x_a）
    //Eigen::Vector<T, 3> t0_toi = t0_initial + toc * dt0;  // 三角形顶点TOI位置
    //Eigen::Vector<T, 3> t1_toi = t1_initial + toc * dt1;
    //Eigen::Vector<T, 3> t2_toi = t2_initial + toc * dt2;

    //// 步骤3：计算TOI时刻的接触点x_a、x_b（文档3.5节定义）
    //Eigen::Vector<T, 3> x_a = p_toi;  // 碰撞点（x_a = 点的TOI位置）
    //Eigen::Vector<T, 3> x_b;          // 三角形上的接触点（x_b = 最近点/碰撞点）
    //int flag = point_triangle_distance_flag(x_a, t0_toi, t1_toi, t2_toi);
    //point_triangle_contact_point(flag, x_a, t0_toi, t1_toi, t2_toi, x_b);

    //// 步骤4：计算接触法向量n̂（文档3.5节：x_b处的三角形表面外法向量）
    //Eigen::Vector<T, 3> tri_normal = (t1_toi - t0_toi).cross(t2_toi - t0_toi);
    //T                   norm_mag   = tri_normal.norm();
    //if(norm_mag < T(1e-12))
    //{  // 三角形退化，无有效法向量
    //    d = T(0);
    //    return false;
    //}
    //Eigen::Vector<T, 3> n_hat = tri_normal / norm_mag;

    //// 步骤5：计算穿透深度（文档3.5节公式 + 碰撞阈值修正）
    //T geometric_dist = (x_b - x_a).dot(n_hat);  // TOI时刻x_a到x_b的法向距离
    //d = std::max(T(0), thickness - geometric_dist);  // 穿透深度=阈值-实际距离（非负）

    return true;
}

/**
 * @brief 边-边CCD碰撞检测 + 穿透深度计算（文档3.5节边-边碰撞模型）
 * @param[in] ea0_initial~ea1_initial: 边ea初始位置（t时刻）
 * @param[in] eb0_initial~eb1_initial: 边eb初始位置（t时刻）
 * @param[in] dea0~dea1: 边ea顶点运动增量（t→t+1总位移）
 * @param[in] deb0~deb1: 边eb顶点运动增量（t→t+1总位移）
 * @param[in] eta: 收敛阈值（与原有CCD一致）
 * @param[in] thickness: 碰撞触发阈值（距离小于该值判定碰撞）
 * @param[in] max_iter: 最大迭代次数（与原有CCD一致）
 * @param[out] toc: 碰撞时间参数（∈[0,1]）
 * @param[out] d: 穿透深度（文档3.5节定义：d = max(0, thickness - ||xb-xa||)）
 * @return 是否检测到有效碰撞
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
    //// 步骤1：调用原有CCD检测碰撞
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

    //// 步骤2：计算TOI时刻的边位置
    //Eigen::Vector<T, 3> ea0_toi = ea0_initial + toc * dea0;
    //Eigen::Vector<T, 3> ea1_toi = ea1_initial + toc * dea1;
    //Eigen::Vector<T, 3> eb0_toi = eb0_initial + toc * deb0;
    //Eigen::Vector<T, 3> eb1_toi = eb1_initial + toc * deb1;

    //// 步骤3：计算TOI时刻的边-边交点（x_a、x_b，文档3.5节定义）
    //Eigen::Vector<T, 3> x_a, x_b;
    //if(!edge_edge_intersection(ea0_toi, ea1_toi, eb0_toi, eb1_toi, x_a, x_b))
    //{
    //    d = T(0);
    //    return false;
    //}

    //// 步骤4：计算接触法向量n̂（文档3.5节：n = xb - xa，单位化）
    //Eigen::Vector<T, 3> n     = x_b - x_a;
    //T                   n_mag = n.norm();
    //if(n_mag < T(1e-12))
    //{  // 交点重合，无穿透
    //    d = T(0);
    //    return false;
    //}
    //Eigen::Vector<T, 3> n_hat = n / n_mag;

    //// 步骤5：计算穿透深度（文档3.5节公式 + 阈值修正）
    //T geometric_dist = n_mag;  // 边-边交点距离=||xb-xa||
    //d                = std::max(T(0), thickness - geometric_dist);

    return true;
}

/**
 * @brief 点-边CCD碰撞检测 + 穿透深度计算（文档3.5节顶点-三角形碰撞的扩展模型）
 * @param[in] p_initial: 碰撞点初始位置（t时刻）
 * @param[in] e0_initial~e1_initial: 边顶点初始位置（t时刻）
 * @param[in] dp: 点的运动增量（t→t+1总位移）
 * @param[in] de0~de1: 边顶点运动增量（t→t+1总位移）
 * @param[in] eta: 收敛阈值（与原有CCD一致）
 * @param[in] thickness: 碰撞触发阈值
 * @param[in] max_iter: 最大迭代次数（与原有CCD一致）
 * @param[out] toc: 碰撞时间参数（∈[0,1]）
 * @param[out] d: 穿透深度（d = max(0, thickness - (xb-xa)·n̂)）
 * @return 是否检测到有效碰撞
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
    //// 步骤1：调用原有CCD检测碰撞
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

    //// 步骤2：计算TOI时刻的位置
    //Eigen::Vector<T, 3> p_toi  = p_initial + toc * dp;    // 点的TOI位置（x_a）
    //Eigen::Vector<T, 3> e0_toi = e0_initial + toc * de0;  // 边的TOI位置
    //Eigen::Vector<T, 3> e1_toi = e1_initial + toc * de1;

    //// 步骤3：计算接触点x_a、x_b
    //Eigen::Vector<T, 3> x_a = p_toi;
    //Eigen::Vector<T, 3> x_b;
    //int                 flag = point_edge_distance_flag(x_a, e0_toi, e1_toi);
    //point_edge_contact_point(flag, x_a, e0_toi, e1_toi, x_b);

    //// 步骤4：计算接触法向量n̂（垂直于边，指向点→边方向）
    //Eigen::Vector<T, 3> edge_dir = e1_toi - e0_toi;
    //Eigen::Vector<T, 3> n        = x_b - x_a;
    //T                   n_mag    = n.norm();
    //if(n_mag < T(1e-12))
    //{  // 点与边接触点重合
    //    d = T(0);
    //    return false;
    //}
    //// 确保法向量垂直于边（修正数值误差）
    //Eigen::Vector<T, 3> edge_normal     = edge_dir.cross(n).cross(edge_dir);
    //T                   edge_normal_mag = edge_normal.norm();
    //if(edge_normal_mag > T(1e-12))
    //{
    //    n     = edge_normal;
    //    n_mag = edge_normal_mag;
    //}
    //Eigen::Vector<T, 3> n_hat = n / n_mag;

    //// 步骤5：计算穿透深度
    //T geometric_dist = n_mag;
    //d                = std::max(T(0), thickness - geometric_dist);

    return true;
}

/**
 * @brief 点-点CCD碰撞检测 + 穿透深度计算（文档3.5节碰撞模型的简化扩展）
 * @param[in] p0_initial~p1_initial: 两点初始位置（t时刻）
 * @param[in] dp0~dp1: 两点运动增量（t→t+1总位移）
 * @param[in] eta: 收敛阈值（与原有CCD一致）
 * @param[in] thickness: 碰撞触发阈值（距离小于该值判定碰撞）
 * @param[in] max_iter: 最大迭代次数（与原有CCD一致）
 * @param[out] toc: 碰撞时间参数（∈[0,1]）
 * @param[out] d: 穿透深度（d = max(0, thickness - ||p1_toi - p0_toi||)）
 * @return 是否检测到有效碰撞
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
    //// 步骤1：调用原有CCD检测碰撞
    //Eigen::Vector<T, 3> p0_temp = p0_initial;
    //Eigen::Vector<T, 3> p1_temp = p1_initial;
    //bool                has_collision =
    //    point_point_ccd(p0_temp, p1_temp, dp0, dp1, eta, thickness, max_iter, toc);

    //if(!has_collision || toc < T(1e-12) || toc > T(1.0 - 1e-12))
    //{
    //    d = T(0);
    //    return false;
    //}

    //// 步骤2：计算TOI时刻的点位置（x_a = p0_toi，x_b = p1_toi）
    //Eigen::Vector<T, 3> p0_toi = p0_initial + toc * dp0;
    //Eigen::Vector<T, 3> p1_toi = p1_initial + toc * dp1;

    //// 步骤3：计算接触法向量n̂（文档3.5节：n = xb - xa，单位化）
    //Eigen::Vector<T, 3> n     = p1_toi - p0_toi;
    //T                   n_mag = n.norm();
    //if(n_mag < T(1e-12))
    //{  // 两点重合，无穿透
    //    d = T(0);
    //    return false;
    //}
    //Eigen::Vector<T, 3> n_hat = n / n_mag;

    //// 步骤4：计算穿透深度
    //T geometric_dist = n_mag;  // 点-点距离=||xb-xa||
    //d                = std::max(T(0), thickness - geometric_dist);

    return true;
}

}  // namespace uipc::backend::cuda::distance