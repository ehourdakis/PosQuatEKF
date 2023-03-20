#ifndef EXTENDEDKALMANFILTER_
#define EXTENDEDKALMANFILTER_

#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>

namespace ekf {
    
    /**
     * @brief Extended Kalman Filter (EKF)
     *
     * @param BaseType The base type for variables
     */
    template<class BaseType>
    class ExtendedKalmanFilter
    {
    public:
        using State = cv::Mat;//<BaseType, 19, 1>;
        using Measurement = cv::Mat;//<BaseType, 7, 1>;
        using Square = cv::Mat;//<BaseType, 19, 19>;
        using MCovariance = cv::Mat;//<BaseType, 7, 7>;

        /**
         * @brief Default constructor
         */
        ExtendedKalmanFilter()
        {
            x = cv::Mat::zeros(19, 1, CV_64F);

            // Setup state and covariance
            P = cv::Mat::eye(19, 19, CV_64F);

            W = cv::Mat::eye(19, 19, CV_64F);
            PCov = cv::Mat::eye(19, 19, CV_64F);

            V = cv::Mat::eye(7, 7, CV_64F);
            MCov = cv::Mat::eye(7, 7, CV_64F);

            H = cv::Mat::zeros(7,19, CV_64F);

            IC = cv::Mat::eye(7, 7, CV_64F);
        }

        /**
         * @brief Perform filter prediction step using control input \f$u\f$ and corresponding system model
         *
         * @param [in] s The System model
         * @param [in] u The Control input vector
         * @return The updated state estimate
         */
        const State& predict( const double dt )
        {
            updateStateJacobians( x, dt );
            
            // predict state
            x = f(x, dt);
            
            // predict covariance
            P  = ( F * P * F.t() ) + ( W * PCov * W.t() );

            return this->getState();
        }

        /**
         * @brief Perform filter update step using measurement \f$z\f$ and corresponding measurement model
         *
         * @param [in] z The measurement vector
         * @param [in] outlier_threshold Threshold for mahalanobis distance
         * @return The updated state estimate
         */
        const State& update( const Measurement& z, double outlier_threshold )
        {
            updateMeasurementJacobians( x );
            
            // compute innovation covariance
            IC = ( H * P * H.t() ) + ( V * MCov * V.t() );
        
            Measurement spred = h(x);
            auto mah = mahalanobis_position(z, spred, IC);
            Measurement diff = (z - spred);
            
            if(outlier_threshold>0.0 && mah>outlier_threshold) {
                // std::cout << "Mah: " << mah << "\nDiff: " << diff.t() << std::endl;
                std::cout << "Outlier Detected\n";

                return getState();
            }

            // compute kalman gain
            cv::Mat K = P * H.t() * IC.inv(); //<BaseType, 19, 7> 

            // Update state using computed kalman gain and innovation
            x += K * ( z - spred );
            
            // Update covariance
            P -= K * H * P;
            
            // return updated state estimate
            return getState();
        }

        /**
         * @brief Returns the innovation covariance matrix
         */
        inline const MCovariance& getInnovationCovariance()
        {
            return IC;
        }

        /**
         * @brief Get the measurement covariance matrix.
         */
        inline const MCovariance getMeasurementCovariance()
        {
            return MCov;
        }

        /**
         * @brief Set the measurement covariance matrix.
         */
        inline void setMeasurementCovariance(const MCovariance& cov)
        {
            MCov = cov;
        }

        /**
         * @brief Get the process covariance matrix.
         */
        inline const Square getProcessCovariance()
        {
            return PCov;
        }

        /**
         * @brief Set the process covariance matrix.
         */
        inline void setProcessCovariance(const Square& cov)
        {
            PCov = cov;
        }

        /**
         * @brief Get the state covariance matrix.
         */
        inline const Square getStateCovariance()
        {
            return P;
        }

        /**
         * @brief Set the process covariance matrix.
         */
        inline void setStateCovariance(const Square& cov)
        {
            P = cov;
        }

        /**
         * @brief Get the EKF state
         */
        inline const State& getState() {
            return x;
        }

        /**
         * @brief Set the EKF state
         */
        inline void setState(const State& rx) {
            x = rx;
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
         * @param dt The dt used to evaluate the Jacobians
         */
        void updateStateJacobians( const State &x, const double dt )
        {
            // normalize the quaternion before hand
            Eigen::Quaterniond q(x.at<BaseType>(9,0), x.at<BaseType>(10,0), x.at<BaseType>(11,0), x.at<BaseType>(12,0));
            q.normalize();

            double sqw  = q.w();
            double sqx  = q.x();
            double sqy  = q.y();
            double sqz  = q.z();
            double swx  = x.at<BaseType>(13,0);
            double swy  = x.at<BaseType>(14,0);
            double swz  = x.at<BaseType>(15,0);

            F = cv::Mat::eye(19, 19, CV_64F);

            F.at<double>(0, 3) = dt;
            F.at<double>(1, 4) = dt;
            F.at<double>(2, 5) = dt;

            F.at<double>(0, 6) = 0.5 * dt * dt;
            F.at<double>(1, 7) = 0.5 * dt * dt;
            F.at<double>(2, 8) = 0.5 * dt * dt;

            F.at<double>(3, 6) = dt;
            F.at<double>(4, 7) = dt;
            F.at<double>(5, 8) = dt;

            F.at<double>(9, 9) = 1.0;
            F.at<double>(9, 10) = -0.5 * dt * swx;
            F.at<double>(9, 11) = -0.5 * dt * swy;
            F.at<double>(9, 12) = -0.5 * dt * swz;
            F.at<double>(9, 13) = -0.5 * dt * sqx;
            F.at<double>(9, 14) = -0.5 * dt * sqy;
            F.at<double>(9, 15) = -0.5 * dt * sqz;

            F.at<double>(10, 9) = 0.5 * dt * swx;
            F.at<double>(10, 10) = 1.0;
            F.at<double>(10, 11) = 0.5 * dt * swz;
            F.at<double>(10, 12) = -0.5 * dt * swy;
            F.at<double>(10, 13) = 0.5 * dt * sqw;
            F.at<double>(10, 14) = -0.5 * dt * sqz;
            F.at<double>(10, 15) = 0.5 * dt * sqy;

            F.at<double>(11, 9) = 0.5 * dt * swy;
            F.at<double>(11, 10) = -0.5 * dt * swz;
            F.at<double>(11, 11) = 1.0;
            F.at<double>(11, 12) = 0.5 * dt * swx;
            F.at<double>(11, 13) = 0.5 * dt * sqz;
            F.at<double>(11, 14) = 0.5 * dt * sqw;
            F.at<double>(11, 15) = -0.5 * dt * sqx;

            F.at<double>(12, 9) = 0.5 * dt * swz;
            F.at<double>(12, 10) = 0.5 * dt * swy;
            F.at<double>(12, 11) = -0.5 * dt * swx;
            F.at<double>(12, 12) = 1.0;
            F.at<double>(12, 13) = -0.5 * dt * sqy;
            F.at<double>(12, 14) = 0.5 * dt * sqx;
            F.at<double>(12, 15) = 0.5 * dt * sqw;

            // Angular velocity Jacobian
            F.at<double>(13, 16) = dt;
            F.at<double>(14, 17) = dt;
            F.at<double>(15, 18) = dt;
        }

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
        State f(const State& x, const double dt) const
        {
            const double sx    = x.at<double>(0,0);
            const double sy    = x.at<double>(1,0);
            const double sz    = x.at<double>(2,0);
            const double svx   = x.at<double>(3,0);
            const double svy   = x.at<double>(4,0);
            const double svz   = x.at<double>(5,0);
            const double sax   = x.at<double>(6,0);
            const double say   = x.at<double>(7,0);
            const double saz   = x.at<double>(8,0);
            const double sqw   = x.at<double>(9,0);
            const double sqx   = x.at<double>(10,0);
            const double sqy   = x.at<double>(11,0);
            const double sqz   = x.at<double>(12,0);
            const double swx   = x.at<double>(13,0);
            const double swy   = x.at<double>(14,0);
            const double swz   = x.at<double>(15,0);
            const double sdwx  = x.at<double>(16,0);
            const double sdwy  = x.at<double>(17,0);
            const double sdwz  = x.at<double>(18,0);

            //! Predicted state vector after transition
            State x_;
            x_ = cv::Mat::zeros(19, 1, CV_64F);
            
            // evolution of state
            x_.at<double>(0,0)  = sx + svx * dt + 0.5 * sax * dt * dt;
            x_.at<double>(1,0)  = sy + svy * dt + 0.5 * say * dt * dt;
            x_.at<double>(2,0)  = sz + svz * dt + 0.5 * saz * dt * dt;
            x_.at<double>(3,0)  = svx + sax * dt;
            x_.at<double>(4,0)  = svy + say * dt;
            x_.at<double>(5,0)  = svz + saz * dt;
            x_.at<double>(6,0)  = sax;
            x_.at<double>(7,0)  = say;
            x_.at<double>(8,0)  = saz;
            x_.at<double>(9,0)  = sqw - 0.5 * dt * (swx * sqx + swy * sqy + swz * sqz);
            x_.at<double>(10,0) = sqx + 0.5 * dt * (swx * sqw - swy * sqz + swz * sqy);
            x_.at<double>(11,0) = sqy + 0.5 * dt * (swx * sqz + swy * sqw - swz * sqx);
            x_.at<double>(12,0) = sqz - 0.5 * dt * (swx * sqy - swy * sqx - swz * sqw);
            x_.at<double>(13,0) = swx + sdwx * dt;
            x_.at<double>(14,0) = swy + sdwy * dt;
            x_.at<double>(15,0) = swz + sdwz * dt;
            x_.at<double>(16,0) = sdwx;
            x_.at<double>(17,0) = sdwy;
            x_.at<double>(18,0) = sdwz;

            // normalize the quaternion after computing state transition
            Eigen::Quaterniond q(x_.at<double>(9,0), x_.at<double>(10,0), x_.at<double>(11,0), x_.at<double>(12,0));
            q.normalize();
            x_.at<double>(9,0) = q.w();
            x_.at<double>(10,0) = q.x();
            x_.at<double>(11,0) = q.y();
            x_.at<double>(12,0) = q.z();

            return x_;
        }
        
        /**
         * @brief Compute the mahalanobis distance of the measurement.
         * 
         * @param x The x measurement vector
         * @param mean The state projected as measurement
         * @param cov The innovation covariance matrix
         */
        double mahalanobis( const cv::Mat& x,
                            const cv::Mat& mean,
                            const cv::Mat& cov) {
            cv::Mat diff = x - mean;
            cv::Mat inv_cov = cov.inv();
            cv::Mat mah = diff.t() * cov * diff;

            auto squared_distance = mah.at<double>(0,0);
            double md = std::sqrt(squared_distance);
            
            return md;
        }

        /**
         * @brief Compute the mahalanobis distance of the position component of the measurement.
         * 
         * @param x The x measurement vector
         * @param mean The state projected as measurement
         * @param cov The innovation covariance matrix
         */
        double mahalanobis_position(const cv::Mat& x,
                                    const cv::Mat& mean,
                                    const cv::Mat& cov) {
            // Extract the position components of the measurement and the state mean
            cv::Mat x_pos = x.rowRange(0,3).clone();//head<3>();
            cv::Mat mean_pos = mean.rowRange(0,3).clone();//.head<3>();

            // Extract the submatrix of the covariance matrix corresponding to position
            cv::Mat cov_pos = cov(cv::Range(0, 3), cv::Range(0, 3)).clone();//cov.topLeftCorner<3, 3>();

            // Compute the Mahalanobis distance between the position components
            cv::Mat diff_pos = x_pos - mean_pos;
            cv::Mat inv_cov_pos = cov_pos.inv();

            cv::Mat mah = diff_pos.t() * inv_cov_pos * diff_pos;

            auto squared_distance = mah.at<double>(0,0);
            double md = std::sqrt(squared_distance);

            return md;
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
        inline void updateMeasurementJacobians( const State& x)
        {
            // H = dh/dx (Jacobian of measurement function w.r.t. the state)
            H = cv::Mat::zeros(7, 19, CV_64F);
            
            H.at<double>(0, 0) = 1.0;
            H.at<double>(1, 1) = 1.0;
            H.at<double>(2, 2) = 1.0;
            H.at<double>(3, 9) = 1.0;
            H.at<double>(4, 10) = 1.0;
            H.at<double>(5, 11) = 1.0;
            H.at<double>(6, 12) = 1.0;
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
        Measurement h(const State& x) const
        {
            Measurement measurement = cv::Mat::zeros(7, 1, CV_64F);
            
            Eigen::Quaterniond q(x.at<double>(9,0), x.at<double>(10,0), x.at<double>(11,0), x.at<double>(12,0));
            q.normalize();

            measurement.at<double>(0,0) = x.at<double>(0,0);
            measurement.at<double>(1,0) = x.at<double>(1,0);
            measurement.at<double>(2,0) = x.at<double>(2,0);
            measurement.at<double>(3,0) = q.w();
            measurement.at<double>(4,0) = q.x();
            measurement.at<double>(5,0) = q.y();
            measurement.at<double>(6,0) = q.z();
            
            return measurement;
        }
    private:
        //! Innovation covariance
        MCovariance IC;

        //! System model jacobian
        Square F;

        //! State covariance
        Square P;

        //! System model process covariance
        Square PCov;

        //! System model noise jacobian
        Square W;

        //! Measurement covariance
        MCovariance MCov;

        //! Measurement model noise jacobian
        MCovariance V;

        //! EKF state
        State x;

        //! Measurement model jacobian
        cv::Mat H;
    };
}

#endif
