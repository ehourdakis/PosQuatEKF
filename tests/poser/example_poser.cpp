
// The example runs pqekf for the poser data. 
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>

#include <iostream>
#include <random>
#include <chrono>

#include "PoseStruct.h"
#include "pqekf.hpp"

#if(USE_GNUPLOT)
#include "plots.h"
#endif

#include <stdio.h>
#include <sys/ioctl.h> // For FIONREAD
#include <termios.h>
#include <stdbool.h>

using PoseQuaternionEKFd = ekf::PoseQuaternionEKF<double>;
	
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
    std::vector<PoseQuaternionEKFd::State > ekf_states; // the stored states of the EKF

    bool paused = false; // Pause using spacebar

    // load and interpolate data
    // bool loaded = poses::load_and_interpolate_poses("data/poser.csv", measurements);
    // bool loaded = poses::read_poses_from_csv("data/slam.csv", measurements);
    bool loaded = poses::read_poses_from_csv("data/outliers.csv", measurements);
    if(!loaded) {
        std::cerr << "Could not load .csv file\n";
        return 0;
    }
#if(USE_GNUPLOT)
    // Initialize gnuplot figure for trajectory
    Gnuplot cov_draw;
#endif

    // initialize pqekf
    std::unique_ptr<PoseQuaternionEKFd> pqekf(new PoseQuaternionEKFd(
        measurements[0].position,  // Initial position
        measurements[0].orientation, // Initial orientation
        1e-2, // Scale initial state covariance
        1e-2, // Scale process covariance
        1e-2, // Scale measurement covariance
        0.2)); // Outlier threshold
    
    // run a series of predict-update steps given the measurements
    for(size_t i = 1; i <= measurements.size(); i++)
    {
        // The diff between two ROS timestamps is in nanoseconds
        auto dt = (measurements[i].timestamp - measurements[i-1].timestamp)*1e-9;
        auto current_state = pqekf->predict(dt);

        // allow the process model to build up before rejecting outliers
        if(i==30) pqekf->setRejectOutliers(true);
        
        // skip update every nskip
        const auto nskip = 1;
        if((i<30)||(i%nskip)==0){
            measurements[i].orientation.normalize();
            current_state = pqekf->update(measurements[i].position, measurements[i].orientation);
        }

        ekf_states.push_back(current_state);

#if(USE_GNUPLOT)
        // Plot trajectory, measurement and process covariance.
        if(((i-1)%50)==0){
            poses::plot_demo(measurements, ekf_states, pqekf, i, cov_draw);
        }
        // std::cout << "IC :" << std::endl << pqekf->getEKF()->getInnovationCovariance().topLeftCorner<3, 3>() << std::endl;
        // std::cout << "P :"  << std::endl << pqekf->getEKF()->getStateCovariance.topLeftCorner<3, 3>() << std::endl;
#endif
        std::cout << " Timestamp: " << std::fixed << std::setprecision(0) << measurements[i].timestamp 
            << std::setprecision(8) << " dt: " << dt << " Step: " << i 
            << std::endl << std::setprecision(4) 
            << " Position: " << pqekf->getState().block<3,1>(0,0).transpose() << std::endl
            << "Orientation: " << pqekf->getState().block<4,1>(9,0).transpose() << std::endl;

        // Implement pause on space
        if (kbhit()) {
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
    std::vector<double> values;

    for(unsigned i = 0; i < ekf_states.size(); i++) {
        values.push_back(ekf_states[i](18));
        
        orientations.push_back(ekf_states[i].block<4,1>(9,0));
        otargets.push_back(poses::quaternionToVectorXd(measurements[i].orientation));

        positions.push_back(ekf_states[i].block<3,1>(0,0));
        ptargets.push_back(measurements[i].position);
    }

    poses::graph_plot_quaternion(orientations, otargets);
    poses::graph_plot_position(positions, ptargets);
#endif

    return 0;
}