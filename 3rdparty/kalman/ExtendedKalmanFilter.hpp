// The MIT License (MIT)
//
// Copyright (c) 2015 Markus Herb
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#ifndef KALMAN_EXTENDEDKALMANFILTER_HPP_
#define KALMAN_EXTENDEDKALMANFILTER_HPP_

#include "KalmanFilterBase.hpp"
#include "StandardFilterBase.hpp"
#include "LinearizedSystemModel.hpp"
#include "LinearizedMeasurementModel.hpp"

namespace Kalman {
    
    /**
     * @brief Extended Kalman Filter (EKF)
     * 
     * This implementation is based upon [An Introduction to the Kalman Filter](https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf)
     * by Greg Welch and Gary Bishop.
     *
     * @param StateType The vector-type of the system state (usually some type derived from Kalman::Vector)
     */
    template<class StateType>
    class ExtendedKalmanFilter : public KalmanFilterBase<StateType>,
                                 public StandardFilterBase<StateType>
    {
    public:
        //! Kalman Filter base type
        typedef KalmanFilterBase<StateType> KalmanBase;
        //! Standard Filter base type
        typedef StandardFilterBase<StateType> StandardBase;
        
        //! Numeric Scalar Type inherited from base
        using typename KalmanBase::T;
        
        //! State Type inherited from base
        using typename KalmanBase::State;
        
        //! Linearized Measurement Model Type
        template<class Measurement, template<class> class CovarianceBase>
        using MeasurementModelType = LinearizedMeasurementModel<State, Measurement, CovarianceBase>;
        
        //! Linearized System Model Type
        template<class Control, template<class> class CovarianceBase>
        using SystemModelType = LinearizedSystemModel<State, Control, CovarianceBase>;
        
    protected:
        //! Kalman Gain Matrix Type
        template<class Measurement>
        using KalmanGain = Kalman::KalmanGain<State, Measurement>;
        
    public:
        //! State Estimate
        using KalmanBase::x;
        //! State Covariance Matrix
        using StandardBase::P;
        
    public:
        /**
         * @brief Constructor
         */
        ExtendedKalmanFilter()
        {
            // Setup state and covariance
            P.setIdentity();
        }
        
        /**
         * @brief Perform filter prediction step using system model and no control input (i.e. \f$ u = 0 \f$)
         *
         * @param [in] s The System model
         * @return The updated state estimate
         */
        template<class Control, template<class> class CovarianceBase>
        const State& predict( SystemModelType<Control, CovarianceBase>& s )
        {
            // predict state (without control)
            Control u;
            u.setZero();
            return predict( s, u );
        }
        
        /**
         * @brief Perform filter prediction step using control input \f$u\f$ and corresponding system model
         *
         * @param [in] s The System model
         * @param [in] u The Control input vector
         * @return The updated state estimate
         */
        template<class Control, template<class> class CovarianceBase>
        const State& predict( SystemModelType<Control, CovarianceBase>& s, const Control& u )
        {
            s.updateJacobians( x, u );
            
            // predict state
            x = s.f(x, u);
            
            // predict covariance
            P  = ( s.F * P * s.F.transpose() ) + ( s.W * s.getCovariance() * s.W.transpose() );
            
            // return state prediction
            return this->getState();
        }
        
        /**
         * @brief Compute the mahalanobis distance of the measurement.
         * 
         * @param x The x measurement vector
         * @param mean The state projected as measurement
         * @param cov The innovation covariance matrix
         */
        double mahalanobis(const Eigen::VectorXd& x,
                                    const Eigen::VectorXd& mean,
                                    const Eigen::MatrixXd& cov) {
            Eigen::VectorXd diff = x - mean;
            // Eigen::MatrixXd inv_cov = cov.inverse();
            double md = std::sqrt(diff.transpose() * cov * diff);
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
            Eigen::MatrixXd inv_cov_pos = cov_pos.inverse();
            double md = std::sqrt(diff_pos.transpose() * inv_cov_pos * diff_pos);
            return md;
        }

        /**
         * @brief Perform filter update step using measurement \f$z\f$ and corresponding measurement model
         *
         * @param [in] m The Measurement model
         * @param [in] z The measurement vector
         * @return The updated state estimate
         */
        template<class Measurement, template<class> class CovarianceBase>
        const State& update( MeasurementModelType<Measurement, CovarianceBase>& m, const Measurement& z )
        {
            m.updateJacobians( x );
            
            // COMPUTE KALMAN GAIN
            // compute innovation covariance
            Covariance<Measurement> S = ( m.H * P * m.H.transpose() ) + ( m.V * m.getCovariance() * m.V.transpose() );
        
            auto mah = mahalanobis_position(z, m.h(x), S);
            Measurement diff = (z - m.h( x ));
            std::cout << "Mah: " << mah << "\nDiff: " << diff.transpose() << std::endl;
            if(mah>0.02) {
                std::cout << "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT\n";
            }

            // compute kalman gain
            KalmanGain<Measurement> K = P * m.H.transpose() * S.inverse();
            
            // UPDATE STATE ESTIMATE AND COVARIANCE
            // Update state using computed kalman gain and innovation
            x += K * ( z - m.h( x ) );
            
            // Update covariance
            P -= K * m.H * P;
            
            // return updated state estimate
            return this->getState();
        }
    };
}

#endif
