
#include <vector>
#include <gnuplot-iostream.h>

namespace poses
{
    /**
     * @brief Plots four graphs, one for each quaternion coefficient.
     *
     * @param [in] matrices Vector of quaternions as Eigen::VectorXd
     * @param [in] targets Vector of targets as Eigen::VectorXd
     */
    void graph_plot_quaternion(const std::vector<cv::Quat<double> >& matrices, const std::vector<cv::Quat<double> >& targets) {
        // Create a new gnuplot object
        Gnuplot gp;

        // Create four separate subplots for each row of the matrices
        gp << "set multiplot layout 2,2\n";

        gp << "set title 'qw'\n";
        gp << "plot '-' with lines lw 1 linecolor 'blue' title 'Measurements w outliers', '-' with lines linecolor 'red' lw 2 title 'EKF Estimate'\n";
        for (const auto& mat : targets) {
            gp << mat[0] << "\n";
        }
        gp << "e\n";
        for (const auto& mat : matrices) {
            gp << mat[0] << "\n";
        }
        gp << "e\n";

        gp << "set title 'qx'\n";
        gp << "plot '-' with lines lw 1 linecolor 'blue' title 'Measurements w outliers', '-' with lines linecolor 'red' lw 2 title 'EKF Estimate'\n";
        for (const auto& mat : targets) {
            gp << mat[1] << "\n";
        }
        gp << "e\n";
        for (const auto& mat : matrices) {
            gp << mat[1] << "\n";
        }
        gp << "e\n";

        gp << "set title 'qy'\n";
        gp << "plot '-' with lines lw 1 linecolor 'blue' title 'Measurements w outliers', '-' with lines linecolor 'red' lw 2 title 'EKF Estimate'\n";
        for (const auto& mat : targets) {
            gp << mat[2] << "\n";
        }
        gp << "e\n";
        for (const auto& mat : matrices) {
            gp << mat[2] << "\n";
        }
        gp << "e\n";

        gp << "set title 'qz'\n";
        gp << "plot '-' with lines lw 1 linecolor 'blue' title 'Measurements w outliers', '-' with lines linecolor 'red' lw 2 title 'EKF Estimate'\n";
        for (const auto& mat : targets) {
            gp << mat[3] << "\n";
        }
        gp << "e\n";
        for (const auto& mat : matrices) {
            gp << mat[3] << "\n";
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
    void graph_plot_position(const std::vector<cv::Mat>& matrices, const std::vector<cv::Mat>& targets) {
        // Create a new gnuplot object
        Gnuplot gp;

        // Create four separate subplots for each row of the matrices
        gp << "set multiplot layout 2,2\n";

        gp << "set title 'X'\n";
        gp << "plot '-' with lines lw 1 linecolor 'blue' title 'Measurements w outliers', '-' with lines linecolor 'red' lw 2 title 'EKF Estimate'\n";
        for (const auto& mat : targets) {
            gp << mat.at<double>(0, 0) << "\n";
        }
        gp << "e\n";
        for (const auto& mat : matrices) {
            gp << mat.at<double>(0, 0) << "\n";
        }
        gp << "e\n";

        gp << "set title 'Y'\n";
        gp << "plot '-' with lines lw 1 linecolor 'blue' title 'Measurements w outliers', '-' with lines linecolor 'red' lw 2 title 'EKF Estimate'\n";
        for (const auto& mat : targets) {
            gp << mat.at<double>(0,1) << "\n";
        }
        gp << "e\n";
        for (const auto& mat : matrices) {
            gp << mat.at<double>(0,1) << "\n";
        }
        gp << "e\n";

        gp << "set title 'Z'\n";
        gp << "plot '-' with lines lw 1 linecolor 'blue' title 'Measurements w outliers', '-' with lines linecolor 'red' lw 2 title 'EKF Estimate'\n";
        for (const auto& mat : targets) {
            gp << mat.at<double>(0,2) << "\n";
        }
        gp << "e\n";
        for (const auto& mat : matrices) {
            gp << mat.at<double>(0,2) << "\n";
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
    void plot_ekf(const std::vector<cv::Mat>& poses, Gnuplot &gp) {
        std::vector<double> x, y, z, qx, qy, qz, qw;

        // Extract position and orientation data from the Pose struct
        for (const auto& pose : poses) {
            x.push_back(pose.at<double>(0,0));
            y.push_back(pose.at<double>(1,0));
            z.push_back(pose.at<double>(2,0));

            qw.push_back(pose.at<double>(9,0));
            qx.push_back(pose.at<double>(10,0));
            qy.push_back(pose.at<double>(11,0));
            qz.push_back(pose.at<double>(12,0));
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

        // Extract position and orientation data from the Pose struct
        for (const auto& pose : poses) {
            x.push_back(pose.position[0]);
            y.push_back(pose.position[1]);
            z.push_back(pose.position[2]);

            qw.push_back(pose.orientation[0]);
            qx.push_back(pose.orientation[1]);
            qy.push_back(pose.orientation[2]);
            qz.push_back(pose.orientation[3]);
        }

        gp.send1d(boost::make_tuple(x, y, z));
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
                    const std::vector< ekf::PoseQuaternionEKF<double>::State >& states,
                    std::unique_ptr<ekf::PoseQuaternionEKF<double> >& pqekf,
                    const int index,
                    Gnuplot &gp)
    {
        if(index == 1)
        {
            auto x_comp = [](const poses::Pose& a, const poses::Pose& b) { return a.position[0] < b.position[0]; };
            auto y_comp = [](const poses::Pose& a, const poses::Pose& b) { return a.position[1] < b.position[1]; };
            auto z_comp = [](const poses::Pose& a, const poses::Pose& b) { return a.position[2] < b.position[2]; };

            auto minmax_x = std::minmax_element(targets.begin(), targets.end(), x_comp);
            auto minmax_y = std::minmax_element(targets.begin(), targets.end(), y_comp);
            auto minmax_z = std::minmax_element(targets.begin(), targets.end(), z_comp);

            // Access the min/max values
            double min_x = minmax_x.first->position[0];
            double max_x = minmax_x.second->position[0];

            double min_y = minmax_y.first->position[1];
            double max_y = minmax_y.second->position[1];

            double min_z = minmax_z.first->position[2] - 2.0;
            double max_z = minmax_z.second->position[2] + 2.0;

            gp << "set xrange [" << min_x << ":" << max_x << "]" << std::endl;
            gp << "set yrange [" << min_y << ":" << max_y << "]" << std::endl;
            gp << "set zrange [" << min_z << ":" << max_z << "]" << std::endl;

            gp << "set title 'Trajectory, process and measurement Covariance'" << std::endl;
            gp << "set xlabel 'x'" << std::endl;
            gp << "set ylabel 'y'" << std::endl;
            gp << "set zlabel 'z'" << std::endl;
        }

        gp << "splot ";
        gp << "'-' with linespoints pointtype 7 linecolor rgb 'red' lw 1 ps 0.8 title 'Measurements w outliers'";
        gp << ",'-' with linespoints pointtype 7 linecolor rgb 'blue' lw 1 ps 0.8 title 'EKF'";
        gp << std::endl;

        std::vector<Pose>::const_iterator first = targets.begin();
        std::vector<Pose>::const_iterator last = targets.begin() + index;
        plot_trajectory(std::vector<Pose>(first, last), gp);
        
        plot_ekf(states, gp);
    }

}