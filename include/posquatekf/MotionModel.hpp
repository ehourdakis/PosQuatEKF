#ifndef FELICE_CRF_SYSTEM_MODEL
#define FELICE_CRF_SYSTEM_MODEL

#include <LinearizedSystemModel.hpp>
#include <iostream>

namespace FELICE
{
namespace ekf
{

/**
 * @brief System state vector for the ekf robot.
 *
 * This is a system state tracking position, velocity, 
 * acceleration, orientation, angular velocity and angular 
 * acceleration.
 *
 * @param T Numeric scalar type
 */
template<typename T>
class State : public Kalman::Vector<T, 19>
{
public:
    KALMAN_VECTOR(State, T, 19)
    
    T x() const { return (*this)[ 0 ]; }
    T y() const { return (*this)[ 1 ]; }
    T z() const { return (*this)[ 2 ]; }
    T vx() const { return (*this)[ 3 ]; }
    T vy() const { return (*this)[ 4 ]; }
    T vz() const { return (*this)[ 5 ]; }
    T ax() const { return (*this)[ 6 ]; }
    T ay() const { return (*this)[ 7 ]; }
    T az() const { return (*this)[ 8 ]; }
    T qw() const { return (*this)[ 9 ]; }
    T qx() const { return (*this)[ 10 ]; }
    T qy() const { return (*this)[ 11 ]; }
    T qz() const { return (*this)[ 12 ]; }
    T wx() const { return (*this)[ 13 ]; }
    T wy() const { return (*this)[ 14 ]; }
    T wz() const { return (*this)[ 15 ]; }
    T dwx() const { return (*this)[ 16 ]; }
    T dwy() const { return (*this)[ 17 ]; }
    T dwz() const { return (*this)[ 18 ]; }
    
    T& x() { return (*this)[ 0 ]; }
    T& y() { return (*this)[ 1 ]; }
    T& z() { return (*this)[ 2 ]; }
    T& vx() { return (*this)[ 3 ]; }
    T& vy() { return (*this)[ 4 ]; }
    T& vz() { return (*this)[ 5 ]; }
    T& ax() { return (*this)[ 6 ]; }
    T& ay() { return (*this)[ 7 ]; }
    T& az() { return (*this)[ 8 ]; }
    T& qw() { return (*this)[ 9 ]; }
    T& qx() { return (*this)[ 10 ]; }
    T& qy() { return (*this)[ 11 ]; }
    T& qz() { return (*this)[ 12 ]; }
    T& wx() { return (*this)[ 13 ]; }
    T& wy() { return (*this)[ 14 ]; }
    T& wz() { return (*this)[ 15 ]; }
    T& dwx() { return (*this)[ 16 ]; }
    T& dwy() { return (*this)[ 17 ]; }
    T& dwz() { return (*this)[ 18 ]; }
};

/**
 * @brief System control-input.
 *
 * The system control is not used currently. 
 *
 * @param T Numeric scalar type
 */
template<typename T>
class Control : public Kalman::Vector<T, 2>
{
public:
    KALMAN_VECTOR(Control, T, 2)
};

/**
 * @brief System model for the ekf robot.
 *
 * The system model defining how the system state evolves over time.
 *
 * @param T Numeric scalar type
 * @param CovarianceBase Class template to determine the covariance representation
 *                       (as covariance matrix (StandardBase) or as lower-triangular
 *                       coveriace square root (SquareRootBase))
 */
template<typename T, template<class> class CovarianceBase = Kalman::StandardBase>
class SystemModel : public Kalman::LinearizedSystemModel<State<T>, Control<T>, CovarianceBase>
{
public:
    //! State type shortcut definition
	typedef FELICE::ekf::State<T> S;
    
    //! Control type shortcut definition
    typedef FELICE::ekf::Control<T> C;
    
    /**
     * @brief Definition of (non-linear) state transition function
     *
     * This function defines how the system state is propagated through time,
     * i.e. it defines in which state \f$\hat{x}_{k+1}\f$ is system is expected to 
     * be in time-step \f$k+1\f$ given the current state \f$x_k\f$ in step \f$k\f$ and
     * the system control input \f$u\f$.
     *
     * @param [in] x The system state in current time-step
     * @param [in] u The control vector input
     * @returns The (predicted) system state in the next time-step
     */
    S f(const S& x, const C&)const // u) const
    {
        const double sx    = x.x();
        const double sy    = x.y();
        const double sz    = x.z();
        const double svx   = x.vx();
        const double svy   = x.vy();
        const double svz   = x.vz();
        const double sax   = x.ax();
        const double say   = x.ay();
        const double saz   = x.az();
        const double sqw   = x.qw();
        const double sqx   = x.qx();
        const double sqy   = x.qy();
        const double sqz   = x.qz();
        const double swx   = x.wx();
        const double swy   = x.wy();
        const double swz   = x.wz();
        const double sdwx  = x.dwx();
        const double sdwy  = x.dwy();
        const double sdwz  = x.dwz();

        //! Predicted state vector after transition
        S x_;
        x_.setZero();
        
        // Return transitioned state vector
        double dt_ = 0.05;

        // evolution of state
        x_.x() = sx + svx * dt_ + 0.5 * sax * dt_ * dt_;
        x_.y() = sy + svy * dt_ + 0.5 * say * dt_ * dt_;
        x_.z() = sz + svz * dt_ + 0.5 * saz * dt_ * dt_;
        x_.vx() = svx + sax * dt_;
        x_.vy() = svy + say * dt_;
        x_.vz() = svz + saz * dt_;
        x_.ax() = sax;
        x_.ay() = say;
        x_.az() = saz;
        x_.qw() = sqw - 0.5 * dt_ * (swx * sqx + swy * sqy + swz * sqz);
        x_.qx() = sqx + 0.5 * dt_ * (swx * sqw - swy * sqz + swz * sqy);
        x_.qy() = sqy + 0.5 * dt_ * (swx * sqz + swy * sqw - swz * sqx);
        x_.qz() = sqz - 0.5 * dt_ * (swx * sqy - swy * sqx - swz * sqw);
        x_.wx() = swx + sdwx * dt_;
        x_.wy() = swy + sdwy * dt_;
        x_.wz() = swz + sdwz * dt_;
        x_.dwx() = sdwx;
        x_.dwy() = sdwy;
        x_.dwz() = sdwz;

        // normalize the quaternion after computing state transition
        Eigen::Quaterniond q(x_.qw(), x_.qx(), x_.qy(), x_.qz());
        q.normalize();
        x_.qw() = q.w();
        x_.qx() = q.x();
        x_.qy() = q.y();
        x_.qz() = q.z();

        // std::cout << "Predicted: " << x_.transpose() << std::endl;
        return x_;
    }
    
protected:
    /**
     * @brief Update jacobian matrices for the system state transition function using current state
     *
     * This will re-compute the (state-dependent) elements of the jacobian matrices
     * to linearize the non-linear state transition function \f$f(x,u)\f$ around the
     * current state \f$x\f$.
     *
     * @note Please consult the readme for the derivation of these equations.
     *
     * @param x The current system state around which to linearize
     * @param u The current system control input
     */
    void updateJacobians( const S& x, const C& )
    {
        //! System model noise jacobian
        this->W.setIdentity();

        // normalize the quaternion before hand
        Eigen::Quaterniond q(x.qw(), x.qx(), x.qy(), x.qz());
        q.normalize();

        double sqw  = x.qw();
        double sqx  = x.qx();
        double sqy  = x.qy();
        double sqz  = x.qz();
        double swx  = x.wx();
        double swy  = x.wy();
        double swz  = x.wz();

        double dt_ = 0.05;

        //! System model jacobian
        this->F.setIdentity();

        this->F(0, 3) = dt_;
        this->F(1, 4) = dt_;
        this->F(2, 5) = dt_;

        this->F(0, 6) = 0.5 * dt_ * dt_;
        this->F(1, 7) = 0.5 * dt_ * dt_;
        this->F(2, 8) = 0.5 * dt_ * dt_;

        this->F(3, 6) = dt_;
        this->F(4, 7) = dt_;
        this->F(5, 8) = dt_;

        this->F(9, 9) = 1.0;
        this->F(9, 10) = -0.5 * dt_ * swx;
        this->F(9, 11) = -0.5 * dt_ * swy;
        this->F(9, 12) = -0.5 * dt_ * swz;
        this->F(9, 13) = -0.5 * dt_ * sqx;
        this->F(9, 14) = -0.5 * dt_ * sqy;
        this->F(9, 15) = -0.5 * dt_ * sqz;

        this->F(10, 9) = 0.5 * dt_ * swx;
        this->F(10, 10) = 1.0;
        this->F(10, 11) = 0.5 * dt_ * swz;
        this->F(10, 12) = -0.5 * dt_ * swy;
        this->F(10, 13) = 0.5 * dt_ * sqw;
        this->F(10, 14) = -0.5 * dt_ * sqz;
        this->F(10, 15) = 0.5 * dt_ * sqy;

        this->F(11, 9) = 0.5 * dt_ * swy;
        this->F(11, 10) = -0.5 * dt_ * swz;
        this->F(11, 11) = 1.0;
        this->F(11, 12) = 0.5 * dt_ * swx;
        this->F(11, 13) = 0.5 * dt_ * sqz;
        this->F(11, 14) = 0.5 * dt_ * sqw;
        this->F(11, 15) = -0.5 * dt_ * sqx;

        this->F(12, 9) = 0.5 * dt_ * swz;
        this->F(12, 10) = 0.5 * dt_ * swy;
        this->F(12, 11) = -0.5 * dt_ * swx;
        this->F(12, 12) = 1.0;
        this->F(12, 13) = -0.5 * dt_ * sqy;
        this->F(12, 14) = 0.5 * dt_ * sqx;
        this->F(12, 15) = 0.5 * dt_ * sqw;

        // Angular velocity Jacobian
        this->F(13, 16) = dt_;
        this->F(14, 17) = dt_;
        this->F(15, 18) = dt_;
    }
};

} // namespace ekf
} // namespace FELICE

#endif