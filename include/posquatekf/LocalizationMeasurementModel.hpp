#ifndef FELICE_CRF_OBSERVATION_MODEL
#define FELICE_CRF_OBSERVATION_MODEL

#include <LinearizedMeasurementModel.hpp>
#include "MotionModel.hpp"

namespace FELICE
{
namespace ekf
{

/**
 * @brief Measurement vector for a localization estimate from Poser.
 *
 * @param T Numeric scalar type
 */
template<typename T>
class LocalizationMeasurement : public Kalman::Vector<T, 7>
{
public:
    KALMAN_VECTOR(LocalizationMeasurement, T, 7)
    
    //! Localization Measurement.
    T x() const { return (*this)[0]; }
    T y() const { return (*this)[1]; }
    T z() const { return (*this)[2]; }
    T qW() const { return (*this)[3]; }
    T qX() const { return (*this)[4]; }
    T qY() const { return (*this)[5]; }
    T qZ() const { return (*this)[6]; }
    
    T& x() { return (*this)[0]; }
    T& y() { return (*this)[1]; }
    T& z() { return (*this)[2]; }
    T& qW() { return (*this)[3]; }
    T& qX() { return (*this)[4]; }
    T& qY() { return (*this)[5]; }
    T& qZ() { return (*this)[6]; }
};

/**
 * @brief Measurement model for integrating localization estimates.
 *
 * Integrates the estimates from a localization system into the filter. 
 *
 * @param T Numeric scalar type
 * @param CovarianceBase Class template to determine the covariance representation
 *                       (as covariance matrix (StandardBase) or as lower-triangular
 *                       coveriace square root (SquareRootBase))
 */
template<typename T, template<class> class CovarianceBase = Kalman::StandardBase>
class LocalizationMeasurementModel : public Kalman::LinearizedMeasurementModel<State<T>, LocalizationMeasurement<T>, CovarianceBase>
{
public:
    //! State type shortcut definition
    typedef FELICE::ekf::State<T> S;
    
    //! Measurement type shortcut definition
    typedef  FELICE::ekf::LocalizationMeasurement<T> M;
    
    LocalizationMeasurementModel()
    {
        //! Measurement noise
        this->V.setIdentity();

        //! Measurement model Jacobian 
        this->H.setZero();
        
        this->H(0, 0) = 1.0;
        this->H(1, 1) = 1.0;
        this->H(2, 2) = 1.0;
        this->H(3, 9) = 1.0;
        this->H(4, 10) = 1.0;
        this->H(5, 11) = 1.0;
        this->H(6, 12) = 1.0;
    }
    
    /**
     * @brief Definition of the measurement function.
     *
     * This function computes the measurement that is expected
     * from a certain system state.
     *
     * @param [in] x The system state in current time-step
     * @returns The (predicted) sensor measurement for the system state
     */
    M h(const S& x) const
    {
        M measurement;
        
        Eigen::Quaterniond q(x.qw(), x.qx(), x.qy(), x.qz());
        q.normalize();

        measurement.x()  = x.x();
        measurement.y()  = x.y();
        measurement.z()  = x.z();
        measurement.qW() = q.w();
        measurement.qX() = q.x();
        measurement.qY() = q.y();
        measurement.qZ() = q.z();
        
        return measurement;
    }

    
    /**
     * @brief Update jacobian matrices for the system state transition function using current state
     *
     * Linearize the non-linear measurement function \f$h(x)\f$ around the
     * current state \f$x\f$.
     *
     * @note Consult the readme file for the derivation of these equations.
     *
     * @param x The current system state around which to linearize
     */
    void updateJacobians( const S& x)
    {
        // H = dh/dx (Jacobian of measurement function w.r.t. the state)
        this->H.setZero();
        
        this->H(0, 0) = 1.0;
        this->H(1, 1) = 1.0;
        this->H(2, 2) = 1.0;
        this->H(3, 9) = 1.0;
        this->H(4, 10) = 1.0;
        this->H(5, 11) = 1.0;
        this->H(6, 12) = 1.0;
    }
};

} // namespace ekf
} // namespace FELICE

#endif