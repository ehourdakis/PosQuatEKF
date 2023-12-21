#ifndef EXTENDEDKALMANFILTER_
#define EXTENDEDKALMANFILTER_

#include <optional>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace ekf {

    // Check if a quaternion is normalized
    std::optional<Eigen::Quaterniond> safe_normalize(const Eigen::Quaterniond& qn) {
        double epsilon = 1e-5; // Tolerance for the normalization check

        Eigen::Quaterniond q = qn;
        // Check if the quaternion is close to zero
        if (q.norm() < epsilon) {
            std::cerr << "Error: Quaternion has near-zero norm." << std::endl;
            return std::nullopt;
        }

        // Normalize the quaternion
        q.normalize();

        // Check if the quaternion is now normalized
        if (std::abs(q.norm() - 1.0) > epsilon) {
            std::cerr << "Error: Quaternion normalization failed." << std::endl;
            return std::nullopt;
        }

        return q;
    }

    // Check if a matrix is invertible, specifiy a tolerance
    bool is_invertible(const Eigen::MatrixXd& mat, double epsilon = 1e-15) {
        double det = mat.determinant();
        return std::abs(det) > epsilon;
    }

    /**
     * @brief Extended Kalman Filter (EKF)
     *
     */
    class ExtendedKalmanFilter
    {
    public:
        using State = Eigen::Matrix<double, 25, 1>;
        using Measurement = Eigen::Matrix<double, 13, 1>;
        using Square = Eigen::Matrix<double, 25, 25>;
        using MCovariance = Eigen::Matrix<double, 13, 13>;

        /**
         * @brief Default constructor
         */
        ExtendedKalmanFilter()
        {
            // Initialize state and covariance
            x = State::Zero();

            P.setIdentity();

            // Initialize process and measurement noise
            W.setIdentity();
            PCov.setIdentity();

            V.setIdentity();
            MCov.setIdentity();

            IC.setIdentity();
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
            P  = ( F * P * F.transpose() ) + ( W * PCov * W.transpose() );

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
            IC = ( H * P * H.transpose() ) + ( V * MCov * V.transpose() );
        
            Measurement spred = h(x);
            auto mah = mahalanobis_position(z, spred, IC);
            Measurement diff = (z - spred);
            
            if(outlier_threshold>0.0 && mah>outlier_threshold) {
                // std::cout << "Mah: " << mah << "\nDiff: " << diff.transpose() << std::endl;
                std::cout << "Outlier Detected\n";

                return getState();
            }

            // check if innovation covariance is invertible
            if (!is_invertible(IC)) {
                std::cout << IC <<std::endl;
                throw std::runtime_error("innovation covariance is not invertible");
            }

            // compute kalman gain
            // cholesky decomposition of the innovation covariance
            auto inv_IC = IC.llt().solve(MCovariance::Identity());
            Eigen::Matrix<double, 25, 13> K = P * H.transpose() * inv_IC;

            // Update state using computed kalman gain and innovation
            x += K * ( z - spred );
            
            // Update covariance
            P -= K * H * P;
            
            // return updated state estimate
            return getState();
        }

        /**
         * @brief Updates the state with only a SLAM measurement.
         * 
         * This function updates the state of the system using a measurement from a Simultaneous Localization 
         * and Mapping (SLAM) system. It integrates the new measurement into the current state estimate.
         * The function also includes a check for measurement outliers based on the Mahalanobis distance.
         * 
         * @param slam_measurement The SLAM measurement as an Eigen::VectorXd. It is expected to contain the 
         *                         position (sx, sy, sz) and orientation (quaternion sqw, sqx, sqy, sqz).
         * @param outlier_threshold Threshold for the Mahalanobis distance to consider a measurement as an 
         *                          outlier. Measurements with a Mahalanobis distance greater than this 
         *                          threshold will be rejected.
         * 
         * @details The function first constructs the measurement model matrix (H_slam) and the innovation 
         *          covariance matrix (R_slam) for the SLAM measurement. It then computes the innovation, 
         *          the Kalman gain, and performs the state update. If the innovation covariance matrix (S) 
         *          is not invertible or if the Mahalanobis distance of the innovation exceeds the outlier 
         *          threshold, the function will abort the update and return early.
         */
        void updateWithSLAM(const Eigen::VectorXd& slam_measurement, double outlier_threshold) 
        {
            // Update measurement model and Jacobian for SLAM
            Eigen::Matrix<double, 7, 25> H_slam;
            H_slam.setZero();

            // Mapping position from the state vector
            H_slam(0, 0) = 1.0; // sx
            H_slam(1, 1) = 1.0; // sy
            H_slam(2, 2) = 1.0; // sz

            // Mapping orientation (quaternion) from the state vector
            H_slam(3, 9) = 1.0;  // sqw
            H_slam(4, 10) = 1.0; // sqx
            H_slam(5, 11) = 1.0; // sqy
            H_slam(6, 12) = 1.0; // sqz
            
            // Compute the innovation covariance for SLAM
            Eigen::Matrix<double, 7, 7> R_slam = Eigen::Matrix<double, 7, 7>::Identity() * 1e-3; // Define the measurement noise covariance for SLAM
            Eigen::Matrix<double, 7, 7> S = H_slam * P * H_slam.transpose() + R_slam;

            // Check if S is invertible
            if (!is_invertible(S)) 
            {
                std::cerr << "Innovation covariance is not invertible for SLAM update." << std::endl;
                return;
            }

            // Compute the Kalman gain
            Eigen::Matrix<double, 25, 7> K = P * H_slam.transpose() * S.inverse();
            std::cout << "SLAM Kalman Gain (K): \n" << K << "\n\n";

            // Update the state and covariance
            Eigen::VectorXd y = slam_measurement - h(x).head<7>(); // Innovation
            
            std::cout << "SLAM Innovation (y): \n" << y << "\n\n";

            // Compute Mahalanobis distance for outlier detection
            double mahalanobis_dist = std::sqrt(y.transpose() * S.inverse() * y);
            if (mahalanobis_dist > outlier_threshold) 
            {
                std::cout << "Outlier Detected in SLAM Update " << " Mah: " << mahalanobis_dist << " Outlier threshold: " << outlier_threshold << std::endl;
                return;
            }

            x += K * y;
            P -= K * H_slam * P;
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
            Eigen::Quaterniond q(x(9), x(10), x(11), x(12));
            std::optional<Eigen::Quaterniond> normalized_quat = safe_normalize(q);

            if (normalized_quat.has_value()) {
                q = normalized_quat.value();
            } else {
                std::cout << "Failed to normalize quaternion." << std::endl;
            }

            double sqw  = q.w();
            double sqx  = q.x();
            double sqy  = q.y();
            double sqz  = q.z();
            double swx  = x(13);
            double swy  = x(14);
            double swz  = x(15);

            F.setIdentity();

            F(0, 3) = dt;
            F(1, 4) = dt;
            F(2, 5) = dt;

            F(0, 6) = 0.5 * dt * dt;
            F(1, 7) = 0.5 * dt * dt;
            F(2, 8) = 0.5 * dt * dt;

            F(3, 6) = dt;
            F(4, 7) = dt;
            F(5, 8) = dt;

            F(9, 9) = 1.0;
            F(9, 10) = -0.5 * dt * swx;
            F(9, 11) = -0.5 * dt * swy;
            F(9, 12) = -0.5 * dt * swz;
            F(9, 13) = -0.5 * dt * sqx;
            F(9, 14) = -0.5 * dt * sqy;
            F(9, 15) = -0.5 * dt * sqz;

            F(10, 9) = 0.5 * dt * swx;
            F(10, 10) = 1.0;
            F(10, 11) = 0.5 * dt * swz;
            F(10, 12) = -0.5 * dt * swy;
            F(10, 13) = 0.5 * dt * sqw;
            F(10, 14) = -0.5 * dt * sqz;
            F(10, 15) = 0.5 * dt * sqy;

            F(11, 9) = 0.5 * dt * swy;
            F(11, 10) = -0.5 * dt * swz;
            F(11, 11) = 1.0;
            F(11, 12) = 0.5 * dt * swx;
            F(11, 13) = 0.5 * dt * sqz;
            F(11, 14) = 0.5 * dt * sqw;
            F(11, 15) = -0.5 * dt * sqx;

            F(12, 9) = 0.5 * dt * swz;
            F(12, 10) = 0.5 * dt * swy;
            F(12, 11) = -0.5 * dt * swx;
            F(12, 12) = 1.0;
            F(12, 13) = -0.5 * dt * sqy;
            F(12, 14) = 0.5 * dt * sqx;
            F(12, 15) = 0.5 * dt * sqw;

            // Angular velocity Jacobian
            F(13, 16) = dt;
            F(14, 17) = dt;
            F(15, 18) = dt;

            // Partial derivatives of quaternion components with respect to gyroscope biases
            F(10, 19) = -0.5 * dt * sqw; // Partial derivative of sqx with respect to b_gx
            F(10, 20) = 0.5 * dt * sqz;  // Partial derivative of sqx with respect to b_gy
            F(10, 21) = -0.5 * dt * sqy; // Partial derivative of sqx with respect to b_gz

            F(11, 19) = -0.5 * dt * sqz; // Partial derivative of sqy with respect to b_gx
            F(11, 20) = -0.5 * dt * sqw; // Partial derivative of sqy with respect to b_gy
            F(11, 21) = 0.5 * dt * sqx;  // Partial derivative of sqy with respect to b_gz

            F(12, 19) = 0.5 * dt * sqy;  // Partial derivative of sqz with respect to b_gx
            F(12, 20) = -0.5 * dt * sqx; // Partial derivative of sqz with respect to b_gy
            F(12, 21) = -0.5 * dt * sqw; // Partial derivative of sqz with respect to b_gz

            // Partial derivatives of velocity components with respect to accelerometer biases
            F(3, 22) = -dt; // Partial derivative of vx with respect to b_ax
            F(4, 23) = -dt; // Partial derivative of vy with respect to b_ay
            F(5, 24) = -dt; // Partial derivative of vz with respect to b_az
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
            const double sx    = x(0);
            const double sy    = x(1);
            const double sz    = x(2);
            const double svx   = x(3);
            const double svy   = x(4);
            const double svz   = x(5);
            const double sax   = x(6);
            const double say   = x(7);
            const double saz   = x(8);
            const double sqw   = x(9);
            const double sqx   = x(10);
            const double sqy   = x(11);
            const double sqz   = x(12);
            const double swx   = x(13);
            const double swy   = x(14);
            const double swz   = x(15);
            const double sdwx  = x(16);
            const double sdwy  = x(17);
            const double sdwz  = x(18);
            const double b_ax  = x(19);
            const double b_ay  = x(20);
            const double b_az  = x(21);
            const double b_gx  = x(22);
            const double b_gy  = x(23);
            const double b_gz  = x(24);

            //! Predicted state vector after transition
            State x_;
            x_.setZero();
            
            /**
             * @brief Evolution of state 
             * @note Effect of Biases on Measurements
             * In the context of IMU sensors, biases are systematic errors that are added to the true sensor readings. 
             * For instance, if an accelerometer has a bias `b_ax` in the x-direction, the actual measured acceleration 
             * `ax` is the true acceleration `sax` plus the bias: `ax_measured = sax + b_ax`
             * In the EKF, the state estimate reflects the true acceleration, not the biased measurement. Therefore, 
             * we correct the measured acceleration by subtracting the estimated bias: `sax_estimated = ax_measured - b_ax`.
             */
            x_(0)  = sx + svx * dt + 0.5 * sax * dt * dt;
            x_(1)  = sy + svy * dt + 0.5 * say * dt * dt;
            x_(2)  = sz + svz * dt + 0.5 * saz * dt * dt;
            x_(3)  = svx + (sax - b_ax) * dt; // b_ax is the accelerometer bias
            x_(4)  = svy + (say - b_ay) * dt;
            x_(5)  = svz + (saz - b_az) * dt;
            x_(6)  = sax;
            x_(7)  = say;
            x_(8)  = saz;
            x_(9)  = sqw - 0.5 * dt * ((swx - b_gx) * sqx + (swy - b_gy) * sqy + (swz - b_gz) * sqz); // b_gx are the gyroscope biases
            x_(10) = sqx + 0.5 * dt * ((swx - b_gx) * sqw - (swy - b_gy) * sqz + (swz - b_gz) * sqy);
            x_(11) = sqy + 0.5 * dt * ((swx - b_gx) * sqz + (swy - b_gy) * sqw - (swz - b_gz) * sqx);
            x_(12) = sqz - 0.5 * dt * ((swx - b_gx) * sqy - (swy - b_gy) * sqx - (swz - b_gz) * sqw);
            x_(13) = swx + sdwx * dt;
            x_(14) = swy + sdwy * dt;
            x_(15) = swz + sdwz * dt;
            x_(16) = sdwx;
            x_(17) = sdwy;
            x_(18) = sdwz;
            x_(19) = b_ax;
            x_(20) = b_ay;
            x_(21) = b_az;
            x_(22) = b_gx;
            x_(23) = b_gy;
            x_(24) = b_gz;

            // normalize the quaternion after computing state transition
            Eigen::Quaterniond q(x_(9), x_(10), x_(11), x_(12));
            std::optional<Eigen::Quaterniond> normalized_quat = safe_normalize(q);

            if (normalized_quat.has_value()) {
                q = normalized_quat.value();
            } else {
                std::cout << "Failed to normalize quaternion." << std::endl;
            }

            x_(9)  = q.w();
            x_(10) = q.x();
            x_(11) = q.y();
            x_(12) = q.z();

            return x_;
        }
        
        /**
         * @brief Compute the mahalanobis distance of the measurement.
         * 
         * @param x The x measurement vector
         * @param mean The state projected as measurement
         * @param cov The innovation covariance matrix
         */
        double mahalanobis( const Eigen::VectorXd& x,
                            const Eigen::VectorXd& mean,
                            const Eigen::MatrixXd& cov) {
            Eigen::VectorXd diff = x - mean;

            if (!is_invertible(cov))
                throw std::runtime_error("covariance matrix is not invertible");

            auto inv_cov = cov.llt().solve(Eigen::Matrix<double, 7, 7>::Identity());
            double md = std::sqrt(diff.transpose() * inv_cov * diff);
            
            return md;
        }

        /**
         * @brief Compute the mahalanobis distance of the position component of the measurement.
         * 
         * @param x The x measurement vector
         * @param mean The state projected as measurement
         * @param cov The innovation covariance matrix
         */
        double mahalanobis_position(const Eigen::VectorXd& x,
                                    const Eigen::VectorXd& mean,
                                    const Eigen::MatrixXd& cov) {
            // Extract the position components of the measurement and the state mean
            Eigen::Vector3d x_pos = x.head<3>();
            Eigen::Vector3d mean_pos = mean.head<3>();

            // Extract the submatrix of the covariance matrix corresponding to position
            Eigen::MatrixXd cov_pos = cov.topLeftCorner<3, 3>();

            // Compute the Mahalanobis distance between the position components
            Eigen::VectorXd diff_pos = x_pos - mean_pos;
            if (!is_invertible(cov_pos))
                throw std::runtime_error("covariance matrix is not invertible");

            auto inv_cov_pos = cov_pos.llt().solve(Eigen::Matrix3d::Identity());
            double md = std::sqrt(diff_pos.transpose() * inv_cov_pos * diff_pos);

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
         * @note The Jacobian matrix H represents how changes in the state vector 
         * affect the expected measurements.
         * @param x The current system state around which to linearize
         */
        inline void updateMeasurementJacobians(const State& x)
        {
            // H = dh/dx (Jacobian of measurement function w.r.t. the state)
            H.setZero();

            // Position from SLAM system
            H(0, 0) = 1.0; // Partial derivative of x position w.r.t. x
            H(1, 1) = 1.0; // Partial derivative of y position w.r.t. y
            H(2, 2) = 1.0; // Partial derivative of z position w.r.t. z

            // Orientation from SLAM system
            H(3, 9) = 1.0;  // Partial derivative of qw w.r.t. qw
            H(4, 10) = 1.0; // Partial derivative of qx w.r.t. qx
            H(5, 11) = 1.0; // Partial derivative of qy w.r.t. qy
            H(6, 12) = 1.0; // Partial derivative of qz w.r.t. qz

            /**
             * @brief Linear acceleration and angular velocity from IMU (subtracting biases)
             * The Jacobian matrix `H` represents how changes in the state vector affect 
             * the expected measurements. 
             * For example, `H(7, 6) = 1.0`, means a unit change in the true acceleration 
             * `sax` (state) results in the same unit change in the measured acceleration 
             * `ax` (measurement).
             * Conversely, `H(7, 19) = -1.0` means a unit change in the estimated bias `b_ax` 
             * (state) results in an opposite unit change in the measured acceleration `ax` 
             * (measurement). This is the result of the bias is subtracted from the measured  
             * value, in the state estimate, to estimate the true value.
             */
            H(7, 6) = 1.0;  // Partial derivative of ax w.r.t. sax
            H(7, 19) = -1.0; // Partial derivative of ax w.r.t. b_ax
            H(8, 7) = 1.0;  // Partial derivative of ay w.r.t. say
            H(8, 20) = -1.0; // Partial derivative of ay w.r.t. b_ay
            H(9, 8) = 1.0;  // Partial derivative of az w.r.t. saz
            H(9, 21) = -1.0; // Partial derivative of az w.r.t. b_az

            H(10, 13) = 1.0; // Partial derivative of wx w.r.t. swx
            H(10, 22) = -1.0; // Partial derivative of wx w.r.t. b_gx
            H(11, 14) = 1.0; // Partial derivative of wy w.r.t. swy
            H(11, 23) = -1.0; // Partial derivative of wy w.r.t. b_gy
            H(12, 15) = 1.0; // Partial derivative of wz w.r.t. swz
            H(12, 24) = -1.0; // Partial derivative of wz w.r.t. b_gz
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
            Measurement measurement;
            measurement.setZero();
            
            Eigen::Quaterniond q(x(9), x(10), x(11), x(12));
            std::optional<Eigen::Quaterniond> normalized_quat = safe_normalize(q);

            if (normalized_quat.has_value()) {
                q = normalized_quat.value();
            } else {
                std::cout << "Failed to normalize quaternion." << std::endl;
            }

            measurement(0) = x.x();
            measurement(1) = x.y();
            measurement(2) = x.z();
            measurement(3) = q.w();
            measurement(4) = q.x();
            measurement(5) = q.y();
            measurement(6) = q.z();
            
            // Linear acceleration from IMU (subtracting biases)
            measurement(7) = x(6) - x(19); // ax - b_ax
            measurement(8) = x(7) - x(20); // ay - b_ay
            measurement(9) = x(8) - x(21); // az - b_az

            // Angular velocity from IMU (subtracting biases)
            measurement(10) = x(13) - x(22); // wx - b_gx
            measurement(11) = x(14) - x(23); // wy - b_gy
            measurement(12) = x(15) - x(24); // wz - b_gz
            
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
        Eigen::Matrix<double, 13, 25> H;
    };
}

#endif
