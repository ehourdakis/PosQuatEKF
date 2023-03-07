
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
    std::vector<Pose> poses;
    
    bool paused = false;

    // load and interpolate data
    bool loaded = load_and_interpolate_poses("data/test.csv", poses);
    if(!loaded) {
        std::cerr << "Could not load poser.csv file\n";
        return 0;
    }
#if(USE_GNUPLOT)
    // Initialize gnuplot figure for trajectory
    Gnuplot cov_draw;
    cov_draw << "set xrange [-5:22]" << std::endl;
    cov_draw << "set yrange [-2:8]" << std::endl;
    cov_draw << "set zrange [-1:3]" << std::endl;
    // Setup plot title and labels
    cov_draw << "set title 'Trajectory, process and measurement Covariance'" << std::endl;
    cov_draw << "set xlabel 'x'" << std::endl;
    cov_draw << "set ylabel 'y'" << std::endl;
    cov_draw << "set zlabel 'z'" << std::endl;
    cov_draw << "set view 49, 48, 3, 1" << std::endl;
    cov_draw << "set view equal xyz" << std::endl;
#endif

    // initialize pqekf
    PoseQuaternionEKF<double> pqekf(poses[0].position, poses[0].orientation);
    
    // run a series of predict-update steps given the measurements
    for(size_t i = 1; i <= poses.size(); i++)
    {
        pqekf.predict();
 
        if((i%1)==0){
            poses[i].orientation.normalize();
            pqekf.update(poses[i].position, poses[i].orientation, 0.02);
        }

#if(USE_GNUPLOT)
        // Plot trajectory, measurement and process covariance.
        if((i%5)==0){
            cov_draw << "splot ";
            cov_draw << "'-' with linespoints pointtype 7 linecolor rgb 'blue' lw 1 ps 0.8 title 'EKF'";
            cov_draw << ", '-' with lines lw 1.0 linecolor rgb 'red' title 'Process Covariance'";
            cov_draw << ", '-' with lines lw 0.8 title 'Measurement Covariance'";
            cov_draw << std::endl;
            plot_ekf(pqekf.getStates(), cov_draw);
            plot_covariance(pqekf.getEKF().P.topLeftCorner<3, 3>(),
                            pqekf.getState().head<3>(), cov_draw);
            plot_covariance(pqekf.getMeasurementCovariance().topLeftCorner<3, 3>(), 
                            poses[i].position, cov_draw);
        }
#endif

        std::cout << "Step: " << i << ":" << std::setprecision(3) << pqekf.getState().transpose() << std::endl;

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
    std::vector<Eigen::VectorXd> orientations, targets;
    std::vector<double> values;
    for(unsigned i = 0; i < pqekf.getStates().size(); i++) {
        values.push_back(pqekf.getStates()[i](18));
        orientations.push_back(pqekf.getStates()[i].block<4,1>(9,0));
        targets.push_back(quaternionToVectorXd(poses[i].orientation));
    }
    graph_plot_quaternion(orientations, targets);
#endif

    return 0;
}
