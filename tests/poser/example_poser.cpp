
// The example runs pqekf for the poser data. 
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>

#include <iostream>
#include <random>
#include <chrono>

#include "PoseStruct.h"
#include <pqekf.hpp>
#include <imu.hpp>

#if(USE_GNUPLOT)
#include "plots.h"
#endif

#include <stdio.h>
#include <sys/ioctl.h> // For FIONREAD
#include <termios.h>
#include <stdbool.h>

/**
 * @brief Non-blocking character reading.
 * @ref https://stackoverflow.com/a/33201364
 */
int kbhit() {
    static bool initflag = false;
    static const int STDIN = 0;

    if (!initflag) {
        // Use termios to turn off line buffering
        struct termios term;
        tcgetattr(STDIN, &term);
        term.c_lflag &= ~ICANON;
        tcsetattr(STDIN, TCSANOW, &term);
        setbuf(stdin, NULL);
        initflag = true;
    }

    int nbbytes;
    ioctl(STDIN, FIONREAD, &nbbytes);  // 0 is STDIN
    return nbbytes;
}

int main(int argc, char** argv)
{
    std::vector<poses::Pose> measurements;
    std::vector<ekf::PoseQuaternionEKF::State > ekf_states; // the stored states of the EKF

    bool paused = false; // Pause using spacebar
    auto calibrator = IMUCalibrator("data/imu/imu_calibration.txt");
    calibrator.cout();

    /**
     * @brief The IMU in realsense cameras has some offset from the Depth sensor, and 
     * zero orientation.
     */
    Eigen::Vector3d extrinsic_calibration(-0.0302200000733137, 0.00740000000223517, 0.0160199999809265);

    // load and interpolate data
    poses::DataLoader loader("data/imu/poses.txt", "data/imu/rgb_aligned.txt", "data/imu/imu_aligned.txt");
    auto dataIterations = loader.getDataIterations();

    if (dataIterations.empty()) 
    {
        std::cerr << "No data loaded.\n";
        return 1;
    }

    Eigen::Quaterniond prev_orientation_continous = dataIterations.front().pose.orientation.normalized();
    // Iterate through all data iterations to correct the quaternion orientation
    for (size_t i = 1; i < dataIterations.size(); ++i) 
    {
        auto& iteration = dataIterations[i];
        iteration.pose.orientation = correctQuaternionFlip(iteration.pose.orientation.normalized(), prev_orientation_continous);
        prev_orientation_continous = iteration.pose.orientation;  // Update the previous orientation
    }

#if(USE_GNUPLOT)
    // Initialize gnuplot figure for trajectory
    Gnuplot cov_draw;
#endif

    // Initialize pqekf
    auto& firstPose = dataIterations.front().pose;
    std::unique_ptr<ekf::PoseQuaternionEKF> pqekf(new ekf::PoseQuaternionEKF(
        firstPose.position,  // Initial position
        firstPose.orientation, // Initial orientation
        calibrator.getGyroBias(),
        1e-2, // Scale initial state covariance
        1e-2, // Scale process covariance
        1e-2, // Scale measurement covariance
        2)); // Outlier threshold

    Eigen::Quaterniond prev_orientation = firstPose.orientation.normalized();

    // Initialize a variable to store the previous IMU timestamp
    double previousIMUTimestamp = dataIterations.front().imuData.empty() ? 0 : dataIterations.front().imuData.front().timestamp;

        // run a series of predict-update steps given the measurements
    for (size_t i = 1; i < dataIterations.size(); i++) 
    {
        const auto& currentIteration = dataIterations[i];
        const auto& previousIteration = dataIterations[i - 1];

        // Calculate the time difference in seconds
        auto dt = (currentIteration.pose.timestamp - previousIteration.pose.timestamp) * 1e-3;
        auto current_state = pqekf->predict(dt);

        // Allow the process model to build up before rejecting outliers
        if (i == 30) pqekf->setRejectOutliers(true);
            
        // skip update every nskip
        const auto nskip = 15;
        if((i<30)||(i%nskip)==0)
        {
            current_state = pqekf->update_slam(currentIteration.pose.position, currentIteration.pose.orientation);
        }

        // Process IMU data within the current iteration
        for (const auto& imu : currentIteration.imuData) 
        {
            double dtIMU = (imu.timestamp - previousIMUTimestamp) * 1e-3;
            previousIMUTimestamp = imu.timestamp; // Update the previous timestamp
    
            // Apply calibration to IMU data
            Eigen::Vector3d calibratedAccel = imu.acceleration - calibrator.getGravityVector();
            Eigen::Vector3d calibratedGyro = imu.gyroscope     - calibrator.getGyroBias();
            
            pqekf->update_accelerometer_gyroscope(calibratedAccel, calibratedGyro, dtIMU);break;
        }

        ekf_states.push_back(current_state);

#if(USE_GNUPLOT)
        // Plot trajectory, measurement and process covariance.
        // if(((i-1)%50)==0){
        //     poses::plot_demo(measurements, ekf_states, pqekf, i, cov_draw);
        // }
        // std::cout << "IC :" << std::endl << pqekf->getEKF()->getInnovationCovariance().topLeftCorner<3, 3>() << std::endl;
        // std::cout << "P :"  << std::endl << pqekf->getEKF()->getStateCovariance.topLeftCorner<3, 3>() << std::endl;
#endif
        std::cout << " Timestamp: " << std::fixed << std::setprecision(0) << currentIteration.pose.timestamp 
            << std::setprecision(8) << " dt: " << dt << " Step: " << i 
            << std::endl << std::setprecision(4) 
            << " Position: " << pqekf->getState().block<3,1>(0,0).transpose() << std::endl
            << "Orientation: " << pqekf->getState().block<4,1>(9,0).transpose() << std::endl;

        // Implement pause on space
        if (kbhit()) 
        {
            int ch = getchar();
            if (ch == 32) paused = !paused;
            std::cout << (paused?"":"Not ") << "Paused...\n";
        }

        if(paused)
        {
            fflush(stdin);
            if(getchar()==32) paused = false;
        }
    }
    
#if(USE_GNUPLOT)
    // Plot graphs
    std::vector<Eigen::VectorXd> orientations, otargets;
    std::vector<Eigen::VectorXd> positions, ptargets;
    std::vector<Eigen::VectorXd> acc_biases, gyro_biases;

    for(unsigned i = 0; i < ekf_states.size(); i++) 
    {
        // Ensure that we do not access out of bounds
        if (i < dataIterations.size()) {
            const auto& currentIteration = dataIterations[i];

            // Extracting the target orientation and position from the Pose in dataIterations
            otargets.push_back(poses::quaternionToVectorXd(currentIteration.pose.orientation));
            ptargets.push_back(currentIteration.pose.position);
        }

        // Extracting the estimated orientation and position from the EKF states
        orientations.push_back(ekf_states[i].block<4,1>(9,0));
        positions.push_back(ekf_states[i].block<3,1>(0,0));

        acc_biases.push_back(ekf_states[i].block<3,1>(19,0));
        gyro_biases.push_back(ekf_states[i].block<3,1>(22,0));
    }

    poses::graph_plot_quaternion(orientations, otargets);
    poses::graph_plot_position(positions, ptargets);
    poses::graph_plot_biases(acc_biases, gyro_biases);
#endif

    return 0;
}