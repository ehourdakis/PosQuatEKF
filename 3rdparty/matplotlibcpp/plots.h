
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

#define WITHOUT_NUMPY
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

#include <Eigen/Geometry>

namespace FELICE
{
namespace ekf
{

/**
 * @brief Converts a quaternion to an Eigen::VectorXd
 *
 * @param [in] quat The quaternion to be converted
 * @return The quaternion as an Eigen::VectorXd.
 */
Eigen::VectorXd quaternionToVectorXd(const Eigen::Quaterniond& quat) {
    Eigen::VectorXd quatVec(4);

    quatVec(0) = quat.w();
    quatVec(1) = quat.x();
    quatVec(2) = quat.y();
    quatVec(3) = quat.z();

    return quatVec;
}

/**
 * @brief Plots four graphs, one for each quaternion coefficient.
 *
 * @param [in] matrices Vector of quaternions as Eigen::VectorXd
 * @param [in] targets Vector of targets as Eigen::VectorXd
 */
void graph_plot_quaternion(const std::vector<Eigen::VectorXd>& matrices, const std::vector<Eigen::VectorXd>& targets) {
    // Create four separate subplots for each row of the matrices
    plt::figure();

    plt::subplot(2, 2, 1);
    std::vector<double> a, aa;
    for (const auto& mat : matrices) {
        a.push_back(mat(0, 0));
    }
    for (const auto& mat : targets) {
        aa.push_back(mat(0, 0));
    }
    plt::plot(a);
    plt::plot(aa);
    plt::title("qw");

    plt::subplot(2, 2, 2);
    std::vector<double> b, bb;
    for (const auto& mat : matrices) {
        b.push_back(mat(1, 0));
    }
    for (const auto& mat : targets) {
        bb.push_back(mat(1, 0));
    }
    plt::plot(b);
    plt::plot(bb);
    plt::title("qx");

    plt::subplot(2, 2, 3);
    std::vector<double> c, cc;
    for (const auto& mat : matrices) {
        c.push_back(mat(2, 0));
    }
    for (const auto& mat : targets) {
        cc.push_back(mat(2, 0));
    }
    plt::plot(c);
    plt::plot(cc);
    plt::title("qy");

    plt::subplot(2, 2, 4);
    std::vector<double> d, dd;
    for (const auto& mat : matrices) {
        d.push_back(mat(3, 0));
    }
    for (const auto& mat : targets) {
        dd.push_back(mat(3, 0));
    }
    plt::plot(d);
    plt::plot(dd);
    plt::title("qz");

    // Show the plots
    plt::show();
}

/**
 * @brief Plots the ekf position estimates.
 *
 * @param [in] poses A vector of ekf estimates
 * @param [in] fig_number The Matplotlibcpp figure number
 */
void plot_ekf(const std::vector<FELICE::ekf::State<double>>& poses, long fig_number) {
    std::vector<double> x, y, z, qx, qy, qz, qw;

    auto skip = 0; // skip mod

    // Extract position and orientation data from the Pose struct
    for (const auto& pose : poses) {
        if(skip++%15!=0) continue;
        x.push_back(pose(0));
        y.push_back(pose(1));
        z.push_back(pose(2));

        qw.push_back(pose(9));
        qx.push_back(pose(10));
        qy.push_back(pose(11));
        qz.push_back(pose(12));
    }

    // Plot red circles for position
    std::map<std::string, std::string> marker_properties = {{"color", "red"}};
    plt::scatter<double>(x, y, z, 2, marker_properties, fig_number);

    // Set the title and axis labels
    plt::title("Data", std::map<std::string, std::string>{{"fontsize", "16"}});
    plt::xlabel("x", std::map<std::string, std::string>{{"fontsize", "16"}});
    plt::ylabel("y", std::map<std::string, std::string>{{"fontsize", "16"}});
}

/**
 * @brief Plots the measurements. 
 *
 * @param [in] poses A vector of ekf::Pose poses.
 * @param [in] fig_number The Matplotlibcpp figure number
 */
void plot_trajectory(const std::vector<ekf::Pose>& poses, long fig_number) {
    std::vector<double> x, y, z, qx, qy, qz, qw;

    auto skip = 0;

    // Extract position and orientation data from the Pose struct
    for (const auto& pose : poses) {
        if(skip++%20!=0) continue;
        x.push_back(pose.position.x());
        y.push_back(pose.position.y());
        z.push_back(pose.position.z());

        qw.push_back(pose.orientation.w());
        qx.push_back(pose.orientation.x());
        qy.push_back(pose.orientation.y());
        qz.push_back(pose.orientation.z());
    }

    // Plot small circles for position
    std::map<std::string, std::string> marker_properties = {{"color", "blue"}};
    plt::scatter<double>(x, y, z, 1, marker_properties, fig_number);

    // Set the title and axis labels
    plt::title("Data", std::map<std::string, std::string>{{"fontsize", "16"}});
    plt::xlabel("x", std::map<std::string, std::string>{{"fontsize", "16"}});
    plt::ylabel("y", std::map<std::string, std::string>{{"fontsize", "16"}});
}

/**
 * @brief Plots the covariance. 
 *
 * @param [in] covariance A 3x3 Covariance matrix
 */
void plot_covariance(const Eigen::MatrixXd& covariance, long fig_number) {
    // Compute eigenvalues and eigenvectors of covariance matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(covariance);
    const Eigen::MatrixXd& eigenvalues = eig.eigenvalues();
    const Eigen::MatrixXd& eigenvectors = eig.eigenvectors();

    // Compute length of eigenvalues for scaling the eigenvectors
    const double eigenvalues_length = eigenvalues.maxCoeff() - eigenvalues.minCoeff();

    // Plot covariance ellipsoid
    std::vector<double> xs, ys, zs;
    for (double theta = 0; theta <= 2 * M_PI; theta += M_PI / 20) {
        for (double phi = 0; phi <= M_PI; phi += M_PI / 20) {
            double x = eigenvectors(0, 0) * eigenvalues(0) * sin(phi) * cos(theta) +
                        eigenvectors(0, 1) * eigenvalues(1) * sin(phi) * sin(theta) +
                        eigenvectors(0, 2) * eigenvalues(2) * cos(phi);
            double y = eigenvectors(1, 0) * eigenvalues(0) * sin(phi) * cos(theta) +
                        eigenvectors(1, 1) * eigenvalues(1) * sin(phi) * sin(theta) +
                        eigenvectors(1, 2) * eigenvalues(2) * cos(phi);
            double z = eigenvectors(2, 0) * eigenvalues(0) * sin(phi) * cos(theta) +
                        eigenvectors(2, 1) * eigenvalues(1) * sin(phi) * sin(theta) +
                        eigenvectors(2, 2) * eigenvalues(2) * cos(phi);
            xs.push_back(x);
            ys.push_back(y);
            zs.push_back(z);
        }
    }
    std::cout << covariance << std::endl;
    plt::clf();
    plt::plot3(xs, ys, zs, std::map<std::string, std::string>(), fig_number);
    plt::pause(0.05);
    plt::draw();
}
}
}