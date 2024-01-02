#include <Eigen/Dense>
#include <vector>

class IMUCalibration {
public:
    // Add gyro data for bias estimation
    void addGyroDataForBias(const Eigen::Vector3d& gyroData) {
        gyroDataForBias.push_back(gyroData);
    }

    // Estimate gyro bias based on collected data
    Eigen::Vector3d estimateGyroBias() {
        Eigen::Vector3d sum = Eigen::Vector3d::Zero();
        for (const auto& data : gyroDataForBias) {
            sum += data;
        }
        Eigen::Vector3d bias = sum / static_cast<double>(gyroDataForBias.size());
        gyroBias = bias; // Store the estimated bias
        return bias;
    }

    // Add accelerometer data for gravity estimation
    void addAccDataForGravity(const Eigen::Vector3d& accData) {
        accDataForGravity.push_back(accData);
    }

    // Estimate gravity vector based on collected data
    Eigen::Vector3d estimateGravityVector() {
        Eigen::Vector3d sum = Eigen::Vector3d::Zero();
        for (const auto& data : accDataForGravity) {
            sum += data;
        }
        Eigen::Vector3d gravity = sum / static_cast<double>(accDataForGravity.size());
        this->gravity = gravity; // Store the estimated gravity vector
        return gravity;
    }

    // Getters for the estimated values
    Eigen::Vector3d getGyroBias() const {
        return gyroBias;
    }

    Eigen::Vector3d getGravityVector() const {
        return gravity;
    }

private:
    std::vector<Eigen::Vector3d> gyroDataForBias;
    std::vector<Eigen::Vector3d> accDataForGravity;
    Eigen::Vector3d gyroBias = Eigen::Vector3d::Zero();
    Eigen::Vector3d gravity = Eigen::Vector3d::Zero();
};

class IMUCalibrator {
public:
    IMUCalibrator() = default; 

    IMUCalibrator(const std::string& indexFilePath) {
        std::vector<std::string> imuFiles = poses::loadIMUDataForPoses(indexFilePath);
        for (const auto& imuFile : imuFiles) {
            std::string l_imuFile = "data/imu/" + imuFile;
            auto imuData = poses::loadIMUData(l_imuFile);
            for (const auto& data : imuData) {
                imuCalibration.addAccDataForGravity(data.acceleration);
                imuCalibration.addGyroDataForBias(data.gyroscope);
            }
        }

        gravityVector = imuCalibration.estimateGravityVector();
        gyroBias = imuCalibration.estimateGyroBias();
    }

    void addAccData(const Eigen::Vector3d& accData) {
        imuCalibration.addAccDataForGravity(accData);
    }

    void addGyroData(const Eigen::Vector3d& gyroData) {
        imuCalibration.addGyroDataForBias(gyroData);
    }

    void estimateCalibration() {
        gravityVector = imuCalibration.estimateGravityVector();
        gyroBias = imuCalibration.estimateGyroBias();
    }

    Eigen::Vector3d getGravityVector() const {
        return gravityVector;
    }

    Eigen::Vector3d getGyroBias() const {
        return gyroBias;
    }

    void cout() const {
        std::cout << "Estimated Gravity Vector: " << gravityVector.transpose() << std::endl;
        std::cout << "Estimated Gyro Bias: " << gyroBias.transpose() << std::endl;
    }
private:
    IMUCalibration imuCalibration;
    Eigen::Vector3d gravityVector;
    Eigen::Vector3d gyroBias;
};