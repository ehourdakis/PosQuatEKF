
// The example runs pqekf for the poser data. 
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
    
    plt::ion();
    plt::backend("TkAgg");

    // load and interpolate data
    bool loaded = load_and_interpolate_poses("data/test.csv", poses);
    if(!loaded) {
        std::cerr << "Could not load poser.csv file\n";
        return 0;
    }

    auto cov_fig_number = plt::figure();
    plt::xlabel("X-axis");
    plt::ylabel("Y-axis");
    plt::title("Continuous update"); 
    plt::grid(true);
    // plt::show();
    plt::clf();

    // initialize pqekf
    PoseQuaternionEKF<double> pqekf(poses[0].position, poses[0].orientation);
    
    // run a series of predict-update steps given the measurements
    for(size_t i = 1; i <= poses.size(); i++)
    {
        pqekf.predict();
 
        if((i%2)==0){
            poses[i].orientation.normalize();
            pqekf.update(poses[i].position, poses[i].orientation, 0.02);
        }
#if(USE_MATPLOTLIBCPP)
        // plot_covariance_3d(pqekf.getEKF().P.topLeftCorner<3, 3>(), cov_fig_number);
        // plot_covariance(pqekf.getProcessCovariance().topLeftCorner<3, 3>(), cov_fig_number);
#endif

        std::cout << "Step: " << i << ":" << std::setprecision(3) << pqekf.getState().transpose() << std::endl;
    }
    
#if(USE_MATPLOTLIBCPP)
    auto fig_number = plt::figure();
    plot_ekf(pqekf.getStates(), fig_number);
    plot_trajectory(poses, fig_number);

    std::vector<Eigen::VectorXd> orientations, targets;
    std::vector<double> values;
    for(unsigned i = 0; i < pqekf.getStates().size(); i++) {
        values.push_back(pqekf.getStates()[i](18));
        orientations.push_back(pqekf.getStates()[i].block<4,1>(9,0));
        targets.push_back(quaternionToVectorXd(poses[i].orientation));
    }
    graph_plot_quaternion(orientations, targets);
    plt::show();
#endif

    return 0;
}
