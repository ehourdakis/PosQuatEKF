#include "ExtendedKalmanFilter.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/quaternion.hpp>

namespace ekf
{

/**
 * @brief EKF filter for a position and quaternion state space system.
 *
 * The ekf tracks the state through a series of predict() and 
 * update() functions. It is robust to outliers, using the mahalanobis
 * distance to filter measurements.
 */
class PoseQuaternionEKF {
public:
    using EKF = ExtendedKalmanFilter;
    using State = EKF::State;
    using Measurement = EKF::Measurement;

    /**
     * @brief Initialize the EKF using a position and quaternion.
     *
     * @param position The initial system position to use for initialization
     * @param orientation The initial system orientation to use for initialization
     */
    PoseQuaternionEKF(const cv::Vec3d& position = cv::Vec3d(0,0,0),
                      const cv::Quat<double> orientation = cv::Quat<double>(1,0,0,0),
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

        cv::Mat init_state_covariance = cv::Mat::eye(19, 19, CV_64F) * state_covariance;
        cv::Mat init_process_covariance = cv::Mat::eye(19, 19, CV_64F) * process_covariance;
        cv::Mat init_measurement_covariance = cv::Mat::eye(7, 7, CV_64F) * measurement_covariance;

        _ekf->setStateCovariance(init_state_covariance);
        _ekf->setProcessCovariance(init_process_covariance);
        _ekf->setMeasurementCovariance(init_measurement_covariance);
    }
    /**
     * @brief Destructor. 
     *
     */
    ~PoseQuaternionEKF()
    {

    }

    /**
     * @brief Predict the state and covariance using the process model
     *
     * This will compute the next state of the EKF based on its previous
     * state estimate and the state model transition. 
     *
     * @param dt The forward dt time to make prediction
     */
    const State predict(const double dt)
    {
        State ret;
        
        // predict the states and covariance
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
    const State update(const cv::Vec3d& position, const cv::Quat<double>& orientation)
    {
        cv::Mat measurement = cv::Mat::zeros(7, 1, CV_64F);

        // Initialize the measurement vector.
        measurement.at<double>(0,0) = position[0];
        measurement.at<double>(1,0) = position[1];
        measurement.at<double>(2,0) = position[2];
        measurement.at<double>(3,0) = orientation[0];
        measurement.at<double>(4,0) = orientation[1];
        measurement.at<double>(5,0) = orientation[2];
        measurement.at<double>(6,0) = orientation[3];

        // Update the EKF.
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
    void init_state(const cv::Vec3d& position = cv::Vec3d(0,0,0),
        const cv::Quat<double> orientation = cv::Quat<double>(1,0,0,0))
    {
        cv::Mat x = cv::Mat::zeros(19, 1, CV_64F);

        x.at<double>(0,0)  = position[0];
        x.at<double>(1,0)  = position[1];
        x.at<double>(2,0)  = position[2];
        x.at<double>(9,0)  = orientation[0];
        x.at<double>(10,0) = orientation[1];
        x.at<double>(11,0) = orientation[2];
        x.at<double>(12,0) = orientation[3];

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