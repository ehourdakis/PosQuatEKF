
// The example runs the pose quaternion EKF for the poser data. 
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>

#include <iostream>
#include <random>
#include <chrono>

#include "PoseStruct.h"
#include "pqekf.h"

#if(USE_MATPLOTLIBCPP)
#include "plots.h"
#endif

using namespace FELICE::ekf;

int main(int argc, char** argv)
{
    std::vector<Pose> poses;

    // load and interpolate data
    bool loaded = load_and_interpolate_poses("data/test.csv", poses);
    if(!loaded) {
        std::cerr << "Could not load poser.csv file\n";
        return 0;
    }

    // initialize the ekf
    PoseQuaternionEKF<double> ekf(poses[0].position, poses[0].orientation);
    
    // run a series of predict-update steps given the measurements
    for(size_t i = 1; i <= poses.size(); i++)
    {
        ekf.predict();

        // if((i%1)==0){
            poses[i].orientation.normalize();
            ekf.update(poses[i].position, poses[i].orientation);
        // }
        std::cout << "Step: " << i << ":" << std::setprecision(3) << ekf.getState().transpose() << std::endl;
    }
    
#if(USE_MATPLOTLIBCPP)
    auto fig_number = plt::figure();
    plot_ekf(ekf.getStates(), fig_number);
    plot_trajectory(poses, fig_number);

    std::vector<Eigen::VectorXd> orientations, targets;
    std::vector<double> values;
    for(unsigned i = 0; i < ekf.getStates().size(); i++) {
        values.push_back(ekf.getStates()[i](18));
        orientations.push_back(ekf.getStates()[i].block<4,1>(9,0));
        targets.push_back(quaternionToVectorXd(poses[i].orientation));
    }
    graph_plot_quaternion(orientations, targets);
    plt::show();
#endif

    return 0;
}
