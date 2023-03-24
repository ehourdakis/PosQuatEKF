#include "ExtendedKalmanFilter.hpp"

namespace ekf
{

/**
 * @brief EKF filter for a position and quaternion state space system.
 *
 * The ekf tracks the state through a series of predict() and 
 * update() functions. It is robust to outliers, using the mahalanobis
 * distance to filter measurements.
 *
 */
class PoseQuaternionEKF {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    using EKF = ExtendedKalmanFilter;
    using State = EKF::State;
    using Measurement = EKF::Measurement;

    /**
     * @brief Initialize the EKF using a position and quaternion.
     *
     * @param position The initial system position to use for initialization
     * @param orientation The initial system orientation to use for initialization
     */
    PoseQuaternionEKF(const Eigen::Vector3d& position = Eigen::Vector3d::Zero(),
                      const Eigen::Quaterniond& orientation = Eigen::Quaterniond::Identity(),
                      const double state_covariance = 1e-5,
                      const double process_covariance = 1e-5,
                      const double measurement_covariance = 1e+2,
                      const double outlier_threshold = 0.02,
                      const bool reject_outliers = false)
    : _outlier_threshold(outlier_threshold),
      _reject_outliers(reject_outliers),
      _ekf(new EKF())
    {
        init_state(position, orientation);

        _ekf->setStateCovariance(EKF::Square::Identity()*state_covariance);
        _ekf->setProcessCovariance(EKF::Square::Identity()*process_covariance);
        _ekf->setMeasurementCovariance(EKF::MCovariance::Identity()*measurement_covariance);
    }
    /**
     * @brief Destructor. 
     *
     */
    ~PoseQuaternionEKF()
    {

    }

    /**
     * @brief Run the prediction of the EKF for a number of steps.
     *
     * This will compute the next state of the EKF based on its previous
     * state estimate and the state model transition. 
     *
     * @param dt The forward dt time to make prediction
     */
    const State predict(const double dt)
    {
        State ret;
        
        ret = _ekf->predict(dt);
        
        return ret;
    }
    
    /**
     * @brief Update the EKF given a position and orientation measurement.
     *
     * The function uses the measurement model to integrate a new measurement
     * to the EKF. It assumes that the orientation is normalized. 
     *
     * @param position The position component of the measurement.
     * @param orientation The orientation component represented as a normalized quaternion.
     */
    const State update(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation)
    {
        Measurement measurement;
        measurement(0) = position(0);
        measurement(1) = position(1);
        measurement(2) = position(2);
        measurement(3) = orientation.w();
        measurement(4) = orientation.x();
        measurement(5) = orientation.y();
        measurement(6) = orientation.z();

        auto x_ekf = _ekf->update(measurement, (_reject_outliers)?_outlier_threshold:-1.0); 

        return x_ekf;
    }

    /**
     * @brief Return the current EKF state.
     */
    inline const State& getState() 
    { 
        return _ekf->getState();
    }

    /**
     * @brief Initialize the EKF state using a position and quaternion.
     *
     * The initial pose is added to the vector of estimates.
     *
     * @param position The initial system position to use for initialization
     * @param orientation The initial system orientation to use for initialization
     */
    void init_state(const Eigen::Vector3d& position = Eigen::Vector3d::Zero(),
        const Eigen::Quaterniond& orientation = Eigen::Quaterniond::Identity())
    {
        State x;
        x.setZero();

        x(0)= position(0);
        x(1)= position(1);
        x(2)= position(2);
        x(9) = orientation.w();
        x(10) = orientation.x();
        x(11) = orientation.y();
        x(12) = orientation.z();

        _ekf->setState(x);
    }

    /**
     * @brief Return a constant reference to the EKF. 
     */
    const std::unique_ptr<EKF>& getEKF()
    {
        return _ekf;
    }

    /**
     * @brief Get the outlier threshold.
     */
    inline const double getOutlierThreshold()
    {
        return _outlier_threshold;
    }

    /**
     * @brief Set the outlier threshold.
     */
    inline void setOutlierThreshold(const double ot)
    {
        _outlier_threshold = ot;
    }

    /**
     * @brief Get the outlier threshold.
     */
    inline bool getRejectOutliers()
    {
        return _reject_outliers;
    }

    /**
     * @brief Set the outlier threshold.
     */
    inline void setRejectOutliers(bool ro)
    {
        _reject_outliers = ro;
    }
private:
    double _outlier_threshold;
    bool _reject_outliers;
    
    std::unique_ptr<EKF> _ekf; // the EKF model
};

}