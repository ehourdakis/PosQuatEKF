#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip> // set precission cout

#include <Eigen/Dense>
#include <cmath>

namespace FELICE
{
namespace ekf
{
/**
 * @brief Holds a pose measurement (position and orientation quaternion)
 *
 */
struct Pose {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    Pose() = default;

    Pose(Eigen::Vector3d position_, Eigen::Quaterniond orientation_, double dt_)
    : position(position_), orientation(orientation_), timestamp(dt_) {}

    void cout() {
        std::cout << std::fixed << timestamp << " " << position.transpose() << " " 
            << orientation.w() << " " << orientation.vec().transpose() << std::endl;
    }
public:
    Eigen::Vector3d position = Eigen::Vector3d(0,0,0);
    Eigen::Quaterniond orientation = Eigen::Quaterniond();
    double timestamp;
};

/**
 * @brief Read a CSV file into a vector of Pose objects
 * 
 * Read the position, quaternion fields of a Poser csv file and 
 * output to a Pose vector.
 * 
 * @param [in] filename string of the filename
 * @param [out] poses the vector of poses read from the file
 */
bool read_poses_from_csv(const std::string& filename, std::vector<Pose> &poses) {

    std::ifstream file(filename);
    if (file.is_open()) {
        // Skip the first header row
        std::string line;
        std::getline(file, line);

        while (std::getline(file, line)) {
            std::stringstream ss(line);

            // Parse fields
            std::string field;
            Pose pose;
            for (int i = 0; i < 46; i++) {
                std::getline(ss, field, ',');
                //std::cout << "Field is: " << field << std::endl;
                if (i == 0) {
                    pose.timestamp = std::stod(field);
                } else if (i >= 4 && i < 7 ) {
                    pose.position(i - 4) = std::stod(field);
                } else if (i >= 7 && i < 11) {
                    pose.orientation.coeffs()(i - 7) = std::stod(field);
                }
            }
            // pose.cout();
            poses.push_back(pose);
        }

        file.close();
        return true;
    }

    return false;
}

/**
 * @brief Interpolate poses from a CSV file
 *
 * The function loads a series of localization measurements, 
 * and replaces repeating measurements by interpolating between
 * the previous and the current measurement. 
 * 
 * The positions are intepolated using:
 * (1.0 - alpha) * prev_pose + alpha * next_pose
 * where alpha is the linear interpolation between the current
 * and the next timestamp.
 * 
 * The quaternions are interpolated using SLERP.
 *
 * @param [in] filename string of the filename
 * @param [out] interp_poses The interpolated poses
 */
bool load_and_interpolate_poses(const std::string& filename, std::vector<Pose> &interp_poses) 
{
    // Read poses from CSV file
    std::vector<Pose> poses;
    auto bLoaded = read_poses_from_csv(filename, poses);
    
    if(!bLoaded) return false;

    // Replace zero-norm quaternions with previous quaternions
    for (size_t i = 1; i < poses.size(); i++) 
    {
        if (poses[i].orientation.norm() == 0.0) 
        {
            poses[i].orientation = poses[i-1].orientation;
        }
    }

    // Find start index and end index of each pose segment
    std::vector<int> pose_start_idxs = {0};
    for (size_t i = 1; i < poses.size(); i++) 
    {
        double position_change = (poses[i].position - poses[i-1].position).norm();
        if (position_change > 0.0) 
        {
            pose_start_idxs.push_back(i);
        }
    }
    pose_start_idxs.push_back(poses.size());

    //for(auto i : pose_start_idxs) std::cout << i << " \n";
    
    // Interpolate poses between start and end indices of each pose segment
    for (unsigned i = 0; i < pose_start_idxs.size() - 1; i++) 
    {
        int start_idx = pose_start_idxs[i];
        int end_idx = pose_start_idxs[i+1]+1;
        std::vector<double> interp_times;

        if (start_idx == end_idx - 1) 
        {
            // Only one measurement
            interp_poses.push_back(poses[start_idx]);
            continue;
        }

        // Compute interpolation times
        for (int j = start_idx; j < end_idx; j++) 
        {
            interp_times.push_back(poses[j].timestamp);
        }
        interp_times.push_back(poses[end_idx-1].timestamp);

        // Interpolate positions and orientations
        for (unsigned j = 0; j < interp_times.size(); j++) 
        {
            double t = interp_times[j];
            Eigen::Vector3d interp_pos;
            Eigen::Quaterniond interp_quat;
            if (j == 0) 
            {
                interp_pos = poses[start_idx].position;
                interp_quat = poses[start_idx].orientation;
            } else if (j == interp_times.size() - 1) 
            {
                interp_pos = poses[end_idx-1].position;
                interp_quat = poses[end_idx-1].orientation;
            } else 
            {
                double alpha = (t - poses[start_idx].timestamp) / (poses[end_idx-1].timestamp - poses[start_idx].timestamp);
                // std::cout <<"Intr " << alpha << " " << poses[end_idx-1].position.transpose() << " " << poses[start_idx].position.transpose() << "\n";
                interp_pos = (1.0 - alpha) * poses[start_idx].position + alpha * poses[end_idx].position;
                interp_quat = poses[start_idx].orientation.slerp(alpha, poses[end_idx].orientation);
                //std::cout << interp_pos.transpose() << std::endl;
            }
            interp_quat.normalize();
            interp_poses.push_back({interp_pos, interp_quat, t});
        }
    }

    //for(auto k : interp_poses) k.cout();

    return true;
}

/**
 * @brief Write a vector of poses to a CSV
 *
 * @param [in] filename The csv filename to write
 * @param [in] interp_poses The vector of poses
 * @returns Boolean to indicate whether filename is written succesfuly. 
 */
bool write_to_csv(std::string filename, const std::vector<Pose> &interp_poses)
{
    std::ofstream outfile(filename);
    if(!outfile.is_open()) 
        return false;

    outfile << "timestamp,x,y,z,qw,qx,qy,qz\n";
    for (const auto& pose : interp_poses) {
        outfile << std::fixed << std::setprecision(0) << pose.timestamp << "," << std::setprecision(5) 
                << pose.position.x() << "," << pose.position.y() << "," << pose.position.z() << ","
                << pose.orientation.w() << "," << pose.orientation.x() << ","
                << pose.orientation.y() << "," << pose.orientation.z() << "\n";
    }
    outfile.close();

    return true;
}

}
}