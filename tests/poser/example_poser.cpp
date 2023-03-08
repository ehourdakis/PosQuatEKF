
// The example runs pqekf for the poser data. 
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>

#include <iostream>
#include <random>
#include <chrono>

#include "PoseStruct.h"
#include "pqekf.h"

#if(USE_GNUPLOT)
#include "plots.h"
#endif

#include <stdio.h>
#include <sys/ioctl.h> // For FIONREAD
#include <termios.h>
#include <stdbool.h>

using namespace FELICE::ekf;
using PoseQuaternionEKFd = PoseQuaternionEKF<double>;
	
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
    std::vector<Pose> measurements;
    
    bool paused = false; // Pause using spacebar

    // load and interpolate data
    bool loaded = load_and_interpolate_poses("/files/Projects/UnderDev/PosQuatEKF/data/outliers.csv", measurements);
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
        1e-8, // Scale initial state covariance
        1e-5, // Scale process covariance
        1e+2)); // Scale measurement covariance
    
    // run a series of predict-update steps given the measurements
    for(size_t i = 1; i <= measurements.size(); i++)
    {
        pqekf->predict();
 
        if((i<30)||(i%1)==0){
            measurements[i].orientation.normalize();
            pqekf->update(measurements[i].position, measurements[i].orientation);
        }

#if(USE_GNUPLOT)
        // Plot trajectory, measurement and process covariance.
        if((i%15)==0){
            plot_demo(measurements, pqekf, i, cov_draw);
        }
        std::cout << "IC :" << std::endl << pqekf->getEKF()->getInnovationCovariance().topLeftCorner<3, 3>() << std::endl;
        std::cout << "P :"  << std::endl << pqekf->getEKF()->P.topLeftCorner<3, 3>() << std::endl;
#endif

        std::cout << "Step: " << i << ":" << std::setprecision(3) << pqekf->getState().transpose() << std::endl;

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

    for(unsigned i = 0; i < pqekf->getStates().size(); i++) {
        values.push_back(pqekf->getStates()[i](18));
        
        orientations.push_back(pqekf->getStates()[i].block<4,1>(9,0));
        otargets.push_back(quaternionToVectorXd(measurements[i].orientation));

        positions.push_back(pqekf->getStates()[i].block<3,1>(0,0));
        ptargets.push_back(measurements[i].position);
    }

    // graph_plot_quaternion(orientations, otargets);
    // graph_plot_position(positions, ptargets);
#endif

    return 0;
}
