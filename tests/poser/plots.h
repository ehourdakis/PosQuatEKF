
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <gnuplot-iostream.h>

namespace poses
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
        // Create a new gnuplot object
        Gnuplot gp;

        // Create four separate subplots for each row of the matrices
        gp << "set multiplot layout 2,2\n";

        gp << "set title 'qw'\n";
        gp << "plot '-' with lines lw 2 linecolor 'blue' title 'Measurements w outliers', '-' with lines linecolor 'red' lw 1 title 'EKF Estimate'\n";
        for (const auto& mat : targets) {
            gp << mat(0, 0) << "\n";
        }
        gp << "e\n";
        for (const auto& mat : matrices) {
            gp << mat(0, 0) << "\n";
        }
        gp << "e\n";

        gp << "set title 'qx'\n";
        gp << "plot '-' with lines lw 2 linecolor 'blue' title 'Measurements w outliers', '-' with lines linecolor 'red' lw 1 title 'EKF Estimate'\n";
        for (const auto& mat : targets) {
            gp << mat(1, 0) << "\n";
        }
        gp << "e\n";
        for (const auto& mat : matrices) {
            gp << mat(1, 0) << "\n";
        }
        gp << "e\n";

        gp << "set title 'qy'\n";
        gp << "plot '-' with lines lw 2 linecolor 'blue' title 'Measurements w outliers', '-' with lines linecolor 'red' lw 1 title 'EKF Estimate'\n";
        for (const auto& mat : targets) {
            gp << mat(2, 0) << "\n";
        }
        gp << "e\n";
        for (const auto& mat : matrices) {
            gp << mat(2, 0) << "\n";
        }
        gp << "e\n";

        gp << "set title 'qz'\n";
        gp << "plot '-' with lines lw 2 linecolor 'blue' title 'Measurements w outliers', '-' with lines linecolor 'red' lw 1 title 'EKF Estimate'\n";
        for (const auto& mat : targets) {
            gp << mat(3, 0) << "\n";
        }
        gp << "e\n";
        for (const auto& mat : matrices) {
            gp << mat(3, 0) << "\n";
        }
        gp << "e\n";

        // Reset the layout to a single plot
        gp << "unset multiplot\n";
    }

    /**
     * @brief Plots three graphs, one for each position component
     *
     * @param [in] matrices Vector of positions as Eigen::VectorXd
     * @param [in] targets Vector of targets as Eigen::VectorXd
     */
    void graph_plot_position(const std::vector<Eigen::VectorXd>& matrices, const std::vector<Eigen::VectorXd>& targets) {
        // Create a new gnuplot object
        Gnuplot gp;

        // Create four separate subplots for each row of the matrices
        gp << "set multiplot layout 2,2\n";

        gp << "set title 'X'\n";
        gp << "plot '-' with lines lw 2 linecolor 'blue' title 'Measurements w outliers', '-' with lines linecolor 'red' lw 1 title 'EKF Estimate'\n";
        for (const auto& mat : targets) {
            gp << mat(0, 0) << "\n";
        }
        gp << "e\n";
        for (const auto& mat : matrices) {
            gp << mat(0, 0) << "\n";
        }
        gp << "e\n";

        gp << "set title 'Y'\n";
        gp << "plot '-' with lines lw 2 linecolor 'blue' title 'Measurements w outliers', '-' with lines linecolor 'red' lw 1 title 'EKF Estimate'\n";
        for (const auto& mat : targets) {
            gp << mat(1, 0) << "\n";
        }
        gp << "e\n";
        for (const auto& mat : matrices) {
            gp << mat(1, 0) << "\n";
        }
        gp << "e\n";

        gp << "set title 'Z'\n";
        gp << "plot '-' with lines lw 2 linecolor 'blue' title 'Measurements w outliers', '-' with lines linecolor 'red' lw 1 title 'EKF Estimate'\n";
        for (const auto& mat : targets) {
            gp << mat(2, 0) << "\n";
        }
        gp << "e\n";
        for (const auto& mat : matrices) {
            gp << mat(2, 0) << "\n";
        }
        gp << "e\n";

        // Reset the layout to a single plot
        gp << "unset multiplot\n";
    }

    void graph_plot_biases(const std::vector<Eigen::VectorXd>& accelerometer_biases, const std::vector<Eigen::VectorXd>& gyro_biases) {
        // Create a new gnuplot object
        Gnuplot gp;

        // Create four separate subplots for each row of the matrices
        gp << "set multiplot layout 2,1\n";

        gp << "set title 'Accelerometer Biases'\n";
        gp << "plot '-' with lines lw 2 linecolor 'blue' title 'Ax', '-' with lines linecolor 'red' lw 1 title 'Ay', '-' with lines linecolor 'green' lw 1  title 'Az'\n";
        for (const auto& mat : accelerometer_biases) {
            gp << mat(0, 0) << "\n";
        }
        gp << "e\n";
        for (const auto& mat : accelerometer_biases) {
            gp << mat(1, 0) << "\n";
        }
        gp << "e\n";
        for (const auto& mat : accelerometer_biases) {
            gp << mat(2, 0) << "\n";
        }
        gp << "e\n";

        gp << "set title 'Gyroscope Biases'\n";
        gp << "plot '-' with lines lw 2 linecolor 'blue' title 'Gx', '-' with lines linecolor 'red' lw 1 title 'Gy', '-' with lines linecolor 'green' lw 1 title 'Gz'\n";
        for (const auto& mat : gyro_biases) {
            gp << mat(0, 0) << "\n";
        }
        gp << "e\n";
        for (const auto& mat : gyro_biases) {
            gp << mat(1, 0) << "\n";
        }
        gp << "e\n";
        for (const auto& mat : gyro_biases) {
            gp << mat(2, 0) << "\n";
        }
        gp << "e\n";

        // Reset the layout to a single plot
        gp << "unset multiplot\n";
    }

    /**
     * @brief Plots the ekf position estimates.
     *
     * @param [in] poses A vector of ekf estimates
     * @param [in] fig_number The Matplotlibcpp figure number
     */
    void plot_ekf(const std::vector<Eigen::Matrix<double, 25, 1> >& poses, Gnuplot &gp) {
        std::vector<double> x, y, z, qx, qy, qz, qw;

        auto skip = 0; // skip mod

        // Extract position and orientation data from the Pose struct
        for (const auto& pose : poses) {
            x.push_back(pose(0));
            y.push_back(pose(1));
            z.push_back(pose(2));

            qw.push_back(pose(9));
            qx.push_back(pose(10));
            qy.push_back(pose(11));
            qz.push_back(pose(12));
        }

        gp.send1d(boost::make_tuple(x, y, z));
    }

    /**
     * @brief Plots the measurements. 
     *
     * @param [in] poses A vector of poses::Pose poses.
     * @param [in] fig_number The Matplotlibcpp figure number
     */
    void plot_trajectory(const std::vector<poses::Pose>& poses, Gnuplot &gp) {
        std::vector<double> x, y, z, qx, qy, qz, qw;

        auto skip = 0;

        // Extract position and orientation data from the Pose struct
        for (const auto& pose : poses) {
            x.push_back(pose.position.x());
            y.push_back(pose.position.y());
            z.push_back(pose.position.z());

            qw.push_back(pose.orientation.w());
            qx.push_back(pose.orientation.x());
            qy.push_back(pose.orientation.y());
            qz.push_back(pose.orientation.z());
        }

        gp.send1d(boost::make_tuple(x, y, z));
    }

    /**
     * @brief Plots the covariance. 
     *
     * @param [in] covariance A 3x3 Covariance matrix
     */
    void plot_covariance(const Eigen::MatrixXd& covariance, const Eigen::VectorXd& mean, Gnuplot &gp, double scale = 1.0) {
        // Compute eigenvalues and eigenvectors of covariance matrix
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(covariance);
        const Eigen::MatrixXd& eigenvalues = eig.eigenvalues();
        const Eigen::MatrixXd& eigenvectors = eig.eigenvectors();

        // Compute length of eigenvalues for scaling the eigenvectors
        const double eigenvalues_length = eigenvalues.maxCoeff() - eigenvalues.minCoeff();

        // Compute x, y, and z coordinates of points on the ellipsoid
        std::vector<double> xx, yy, zz;
        for (double theta = 0; theta <= 2 * M_PI; theta += M_PI / 20) {
            for (double phi = 0; phi <= M_PI; phi += M_PI / 20) {
                double x = scale * (eigenvectors(0, 0) * eigenvalues(0) * sin(phi) * cos(theta) +
                                eigenvectors(0, 1) * eigenvalues(1) * sin(phi) * sin(theta) +
                                eigenvectors(0, 2) * eigenvalues(2) * cos(phi)) + mean(0);
                double y = scale * (eigenvectors(1, 0) * eigenvalues(0) * sin(phi) * cos(theta) +
                                eigenvectors(1, 1) * eigenvalues(1) * sin(phi) * sin(theta) +
                                eigenvectors(1, 2) * eigenvalues(2) * cos(phi)) + mean(1);
                double z = scale * (eigenvectors(2, 0) * eigenvalues(0) * sin(phi) * cos(theta) +
                                eigenvectors(2, 1) * eigenvalues(1) * sin(phi) * sin(theta) +
                                eigenvectors(2, 2) * eigenvalues(2) * cos(phi)) + mean(2);
                xx.push_back(x);
                yy.push_back(y);
                zz.push_back(z);
            }
        }

        // Send x, y, and z coordinates to Gnuplot iostream
        gp.send1d(boost::make_tuple(xx, yy, zz));
    }

    /**
     * @brief Plots the poser demo, i.e. the trajectory along with the process
     * and measurement covariances.
     *
     * @param [in] targets The measurements 
     * @param [in] states The EKF estimates
     * @param [in] pqekf A smart pointer to pqekf
     * @param [in] index The simulation index
     * @param [in] gp Gnuplot reference
     */
    void plot_demo( const std::vector<poses::Pose>& targets, 
                    const std::vector< ekf::PoseQuaternionEKF::State >& states,
                    std::unique_ptr<ekf::PoseQuaternionEKF>& pqekf,
                    const int index,
                    Gnuplot &gp)
    {
        if(index == 1)
        {
            auto x_comp = [](const poses::Pose& a, const poses::Pose& b) { return a.position.x() < b.position.x(); };
            auto y_comp = [](const poses::Pose& a, const poses::Pose& b) { return a.position.y() < b.position.y(); };
            auto z_comp = [](const poses::Pose& a, const poses::Pose& b) { return a.position.z() < b.position.z(); };

            auto minmax_x = std::minmax_element(targets.begin(), targets.end(), x_comp);
            auto minmax_y = std::minmax_element(targets.begin(), targets.end(), y_comp);
            auto minmax_z = std::minmax_element(targets.begin(), targets.end(), z_comp);

            // Access the min/max values
            double min_x = minmax_x.first->position.x();
            double max_x = minmax_x.second->position.x();

            double min_y = minmax_y.first->position.y();
            double max_y = minmax_y.second->position.y();

            double min_z = minmax_z.first->position.z() - 2.0;
            double max_z = minmax_z.second->position.z() + 2.0;

            gp << "set xrange [" << min_x << ":" << max_x << "]" << std::endl;
            gp << "set yrange [" << min_y << ":" << max_y << "]" << std::endl;
            gp << "set zrange [" << min_z << ":" << max_z << "]" << std::endl;

            gp << "set title 'Trajectory, process and measurement Covariance'" << std::endl;
            gp << "set xlabel 'x'" << std::endl;
            gp << "set ylabel 'y'" << std::endl;
            gp << "set zlabel 'z'" << std::endl;
            // gp << "set view 49, 48, 3, 1"<< std::endl;
        }

        gp << "splot ";
        gp << "'-' with linespoints pointtype 7 linecolor rgb 'red' lw 1 ps 0.8 title 'Measurements w outliers'";
        gp << ",'-' with linespoints pointtype 7 linecolor rgb 'blue' lw 1 ps 0.8 title 'EKF'";
        gp << ", '-' with lines lw 1.0 linecolor rgb 'blue' title 'Process Covariance'";
        // plot transparent line. In #90D13030, first two digits (90) are transparency, remaining RGB code
        gp << ", '-' with lines lw 0.5 lc rgb \"#95FF6666\" title 'Measurement Covariance'";
        gp << std::endl;

        std::vector<Pose>::const_iterator first = targets.begin();
        std::vector<Pose>::const_iterator last = targets.begin() + index;
        plot_trajectory(std::vector<Pose>(first, last), gp);
        
        plot_ekf(states, gp);

        plot_covariance(pqekf->getEKF()->getStateCovariance().topLeftCorner<3, 3>(),
                        pqekf->getState().head<3>(), gp, 10);
        plot_covariance(pqekf->getEKF()->getInnovationCovariance().topLeftCorner<3, 3>(), 
                        targets[index].position, gp, 10);
    }

}