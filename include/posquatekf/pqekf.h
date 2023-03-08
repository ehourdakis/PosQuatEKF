#include <ExtendedKalmanFilter.hpp>
#include "MotionModel.hpp"
#include "LocalizationMeasurementModel.hpp"

namespace FELICE
{
namespace ekf
{

/**
 * @brief EKF filter for position and quaternion.
 *
 * The ekf tracks the state through a series of predict() and 
 * update() functions.
 *
 * @param T Numeric scalar type
 */
template<class T>
class PoseQuaternionEKF {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    using State = ekf::State<T>;
    using Control = ekf::Control<T>;
    using SystemModel = ekf::SystemModel<T>;
    using LocalizationMeasurement = ekf::LocalizationMeasurement<T>;
    using LocalizationModel = ekf::LocalizationMeasurementModel<T>;
    using States = std::vector<State>;
    using EKF = Kalman::ExtendedKalmanFilter<State>;

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
                      const double outlier_threshold = 0.02)
    : _outlier_threshold(outlier_threshold),
      _ekf(new EKF())
    {
        init_state(position, orientation);

        _ekf->P = Eigen::MatrixXd::Identity(19, 19)*state_covariance;

        Eigen::MatrixXd proc_cov = Eigen::MatrixXd::Identity(19, 19)*process_covariance;
        sys.setCovariance(proc_cov);

        Eigen::MatrixXd meas_cov = Eigen::MatrixXd::Identity(7, 7)*measurement_covariance;
        om.setCovariance(meas_cov);
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
     * If the number of steps given is larger than 1, i.e. there is no 
     * subsequent call to update, the function will store the estimated pose, 
     *
     * @param num_steps The number of steps to run the prediction.
     */
    const State predict(unsigned int num_steps=1)
    {
        State ret;
        for(size_t i = 0; i < num_steps; i++)
        {
            ret = _ekf->predict(sys);
            if(num_steps>1) ekf_states.push_back(ret);
        }
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
        LocalizationMeasurement measurement;
        measurement.x()  = position(0);
        measurement.y()  = position(1);
        measurement.z()  = position(2);
        measurement.qW() = orientation.w();
        measurement.qX() = orientation.x();
        measurement.qY() = orientation.y();
        measurement.qZ() = orientation.z();

        auto x_ekf = _ekf->update(om, measurement, _outlier_threshold); 
        ekf_states.push_back(x_ekf);

        return x_ekf;
    }

    /**
     * @brief Return the current EKF state.
     */
    const State& getState() 
    { 
        return _ekf->getState();
    }

    /**
     * @brief Return all stored ekf States.
     */
    const States &getStates()
    {
        return ekf_states;
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
        x.x() = position(0);
        x.y() = position(1);
        x.z() = position(2);

        x.qw() = orientation.w();
        x.qx() = orientation.x();
        x.qy() = orientation.y();
        x.qz() = orientation.z();

        _ekf->init(x);
        ekf_states.push_back(x);
    }

    /**
     * @brief Return a constant reference to the EKF. 
     */
    const std::unique_ptr<EKF>& getEKF()
    {
        return _ekf;
    }

    /**
     * @brief Get the measurement covariance matrix.
     */
    const Eigen::MatrixXd getMeasurementCovariance()
    {
        return om.getCovariance();
    }

    /**
     * @brief Set the measurement covariance matrix.
     */
    const void setMeasurementCovariance(const Eigen::MatrixXd& cov)
    {
        return om.setCovariance(cov);
    }

    /**
     * @brief Get the process covariance matrix.
     */
    const Eigen::MatrixXd getProcessCovariance()
    {
        return sys.getCovariance();
    }

    /**
     * @brief Set the process covariance matrix.
     */
    const void setProcessCovariance(const Eigen::MatrixXd& cov)
    {
        return sys.setCovariance(cov);
    }
private:
    const double _outlier_threshold;

    Control u; // The control model
    SystemModel sys; // The system model
    LocalizationModel om; // The localization measurement model
    
    std::unique_ptr<EKF> _ekf; // the EKF model

    //TODO: Remove storing states
    States ekf_states; // the stored states of the EKF
};

}

}