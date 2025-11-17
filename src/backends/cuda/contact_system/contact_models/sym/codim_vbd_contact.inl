
// > Squared Version
// > D := d*d

// Example: Calculate the penetration depth d between vertex x_a and face contact point x_b
template <typename T>
__host__ __device__ T ComputePenetrationDepth(const T x_a[3], const T x_b[3], const T n_hat[3])
{
    // Calculate (x_b - x_a) (dot) n-hat
    T dot = (x_b[0] - x_a[0]) * n_hat[0] + (x_b[1] - x_a[1]) * n_hat[1]
            + (x_b[2] - x_a[2]) * n_hat[2];
    // Penetration depth d = max(0, dot) (only retain non-negative penetration)
    return dot > 1e-12 ? dot : static_cast<T>(0.0);
}

/*****************************************************************************************************************************
Function: Calculate collision energy (quadratic collision energy defined in the paper E_c = 0.5 * k_c * d square)
Function name: KappaQuadratic (replaces the original KappaBarrier)
Parameter description:
- R: Output parameter, stores the calculated collision energy
- k_c: Collision stiffness (denoted as k_c in the paper, refer to experimental parameters set to 1e6~1e7)
- d: Penetration depth (needs to be pre-calculated: d = max(0, (x_b - x_a) (dot) n-hat), only non-negative values are valid)
Template features: Supports any floating-point type (float/double), compatible with both CPU/GPU calls
*****************************************************************************************************************************/
template <typename T>
__host__ __device__ void KappaQuadratic(T& R, const T& k_c, const T& d, const T& dHat, const T& xi)
{
    // Numerical stability: Ignore minimal penetration (avoid invalid calculations due to floating-point errors)
    if(d > 1e-12)
    {
        R = 0.5 * k_c * d * d;  // Core formula: Quadratic collision energy E_c = 1/2 * k_c * d square
    }
    else
    {
        R = static_cast<T>(0.0);  // No penetration (d≤0), collision energy is 0
    }
}

/*****************************************************************************************************************************
Function: First derivative of collision energy with respect to penetration depth d (used to calculate collision force f_c = -dE_c/dd)
Function name: dKappaQuadraticdD (replaces the original dKappaBarrierdD)
Parameter description:
- R: Output parameter, stores the calculated first derivative
- k_c: Collision stiffness (consistent with k_c in energy calculation)
- d: Penetration depth (same as above, only non-negative values are valid)
Physical meaning: The derivative result is k_c*d, the negative sign should be added when calculating the collision force externally (f_c = -R)
*****************************************************************************************************************************/
template <typename T>
__host__ __device__ void dKappaQuadraticdD(T& R, const T& k_c, const T& d, const T& dHat, const T& xi)
{
    if(d > 1e-12)
    {
        R = k_c * d;  // First derivative: dE_c/dd = k_c * d
    }
    else
    {
        R = static_cast<T>(0.0);  // No penetration, derivative is 0 (no collision force)
    }
}

/*****************************************************************************************************************************
Function: Second derivative of collision energy with respect to penetration depth d (used to calculate the Hessian matrix of collision terms)
Function name: ddKappaQuadraticddD (replaces the original ddKappaBarrierddD)
Parameter description:
- R: Output parameter, stores the calculated second derivative
- k_c: Collision stiffness (consistent with k_c in energy calculation)
- d: Penetration depth (same as above, only non-negative values are valid)
Physical meaning: The second derivative is a constant k_c, used to construct the 3 by 3 Hessian matrix for VBD local iteration
*****************************************************************************************************************************/
template <typename T>
__host__ __device__ void ddKappaQuadraticddD(T& R, const T& k_c, const T& d, const T& dHat, const T& xi)
{
    if(d > 1e-12)
    {
        R = k_c;  // Second derivative: d squareE_c/dd square = k_c (the second derivative of a quadratic function is a constant)
    }
    else
    {
        R = static_cast<T>(0.0);  // No penetration, second derivative is 0 (collision term does not affect Hessian)
    }
}