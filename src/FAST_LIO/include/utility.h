#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

// #include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>
#include "dynamic_map.h"
using namespace std;

typedef pcl::PointXYZI PointType;

class ParamServer
{
public:
    ros::NodeHandle nh;

    int PickedSurfNum;
    int skipSurf;

    int PickedCornerNum;
    int skipCorner;

    // Map Server
    string globalSurfMap_dirctory;
    string globalCornerMap_dirctory;
    string globalCornerMap_pcd;
    string globalSurfMap_pcd;
    int area_size;
    int margin;
    float updateMapFrequency;
    string ndt_neighbor_search_method;
    float ndt_resolution;
    string Matching_method;
    string intialMethod;
    enum eintialMethod
    {
        human = 0,
        gps = 1,
    } mintialMethod;
    bool optimization_with_GPS;
    Eigen::Vector3d Pil;
    Eigen::Vector3d Pli;
    vector<double> initialPose;
    int initial_count_num;

    std::string robot_id; //机器人ID,没有使用

    // Topics
    string pointCloudTopic;
    string imuTopic;
    string odomTopic;
    string gpsTopic;

    // Frames
    string lidarFrame;
    string baselinkFrame;
    string odometryFrame;
    string mapFrame;

    // GPS Settings
    bool useImuHeadingInitialization; //是否使用imu的初始朝向yaw作为整个地图的初始朝向 ，false则地图从0开始，true地图以imu的yaw作为初始朝向
    bool useGpsElevation;             // 是否使用gps估计z，如果不使用，则使用里程计的z代替gps的
    float gpsCovThreshold;            // m^2, threshold for using GPS data
    float poseCovThreshold;           // m^2, threshold for using GPS data

    // Save pcd
    bool savePCD;
    string savePCDDirectory;

    // Velodyne Sensor Configuration: Velodyne
    float min_range;
    float max_range;
    float Vertical_angle;
    float ang_bottom;
    string lidar_type; //

    enum elidar_type
    {
        Velodyne = 0,
        rslidar = 1,

    } mlidar_type;
    int N_SCAN;         //行线数
    int Horizon_SCAN;   // 列线数
    string timeField;   //点云时间戳字段
    int downsampleRate; //降采样率，是一个倒数   64->16  downsampleRate=4

    // IMU
    float imuAccNoise;
    float imuGyrNoise;
    float imuAccBiasN;
    float imuGyrBiasN;
    float imuGravity; //重力加速度的值
    int imuFrequency;
    int area_num;
    int iter_num;
    float distance_limit;

    // lidar -> imu的坐标变换
    vector<double> extRotV;
    vector<double> extRPYV;
    vector<double> extTransV;

    Eigen::Matrix3d extRot;
    Eigen::Matrix3d extRPY;
    Eigen::Vector3d extTrans;
    Eigen::Quaterniond extQRPY;

    // LOAM
    float edgeThreshold; // 特征提取的阈值
    float surfThreshold;
    int edgeFeatureMinValidNum;
    int surfFeatureMinValidNum;

    // voxel filter paprams
    float odometrySurfLeafSize;
    float mappingCornerLeafSize;
    float mappingSurfLeafSize;

    //机器人运动约束
    float z_tollerance;
    float rotation_tollerance;

    // CPU Params
    int numberOfCores;             //并行计算使用的cpu核心数
    double mappingProcessInterval; // 建图间隔

    // Surrounding map
    float surroundingkeyframeAddingDistThreshold;  // meters, regulate keyframe adding threshold
    float surroundingkeyframeAddingAngleThreshold; // radians, regulate keyframe adding threshold
    float surroundingKeyframeDensity;              // meters, downsample surrounding keyframe poses
    float surroundingKeyframeSearchRadius;         // 周围关键帧搜索半径  meters, within n meters scan-to-map optimization (when loop closure disabled)

    // Loop closure
    bool loopClosureEnableFlag;
    float loopClosureFrequency;          // 回环检测频率
    int surroundingKeyframeSize;         // submap size (when loop closure enabled)
    float historyKeyframeSearchRadius;   //闭环检测搜索半径
    float historyKeyframeSearchTimeDiff; //闭环检测时间阈值
    int historyKeyframeSearchNum;        // number of hostory key frames will be fused into a submap for loop closure
    float historyKeyframeFitnessScore;   // icp threshold, the smaller the better alignment

    // global map visualization radius
    float globalMapVisualizationSearchRadius; // meters, global map visualization radius
    float globalMapVisualizationPoseDensity;  //  meters, global map visualization keyframe density
    float globalMapVisualizationLeafSize;     // meters, global map visualization cloud density

    // ROS的参数服务
    ParamServer()
    {
        nh.param<float>("globalmap_server/updateMapFrequency", updateMapFrequency, 0.1);
        nh.param<std::string>("globalmap_server/globalSurfMap_dirctory", globalSurfMap_dirctory, "../data/map.pcd");
        nh.param<std::string>("globalmap_server/globalCornerMap_dirctory", globalCornerMap_dirctory, "../data/map.pcd");
        nh.param<std::string>("globalmap_server/globalCornerMap_pcd", globalCornerMap_pcd, "../data/map.pcd");
        nh.param<std::string>("globalmap_server/globalSurfMap_pcd", globalSurfMap_pcd, "../data/map.pcd");
        nh.param<int>("globalmap_server/area_size", area_size, -1);
        nh.param<int>("globalmap_server/margin", margin, -1);
        nh.param<std::string>("globalmap_server/ndt_neighbor_search_method", ndt_neighbor_search_method, "DIRECT7");
        nh.param<float>("globalmap_server/ndt_resolution", ndt_resolution, 1.0);
        nh.param<std::string>("globalmap_server/Matching_method", Matching_method, "loam");
        nh.param<vector<double>>("globalmap_server/initialPose", initialPose, vector<double>());
        nh.param<std::string>("globalmap_server/intialMethod", intialMethod, "gps");
        if (intialMethod == "human")
            mintialMethod = human;
        else if (intialMethod == "gps")
            mintialMethod = gps;
        else
        {
            std::cout << "Undefined intialMethod type " << std::endl;
            exit(-1);
        }
        nh.param<bool>("globalmap_server/optimization_with_GPS", optimization_with_GPS, "false");
        nh.param<int>("globalmap_server/initial_count_num", initial_count_num, -1);

        nh.param<std::string>("/robot_id", robot_id, "roboat");

        nh.param<std::string>("lio_sam/pointCloudTopic", pointCloudTopic, "points_raw");
        nh.param<std::string>("lio_sam/imuTopic", imuTopic, "imu_correct");
        nh.param<std::string>("lio_sam/odomTopic", odomTopic, "odometry/imu");
        nh.param<std::string>("lio_sam/gpsTopic", gpsTopic, "odometry/gps");

        nh.param<std::string>("lio_sam/lidarFrame", lidarFrame, "base_link");
        nh.param<std::string>("lio_sam/baselinkFrame", baselinkFrame, "base_link");
        nh.param<std::string>("lio_sam/odometryFrame", odometryFrame, "odom");
        nh.param<std::string>("lio_sam/mapFrame", mapFrame, "map");

        nh.param<bool>("lio_sam/useImuHeadingInitialization", useImuHeadingInitialization, false);
        nh.param<bool>("lio_sam/useGpsElevation", useGpsElevation, false);
        nh.param<float>("lio_sam/gpsCovThreshold", gpsCovThreshold, 2.0);
        nh.param<float>("lio_sam/poseCovThreshold", poseCovThreshold, 25.0);

        nh.param<bool>("lio_sam/savePCD", savePCD, false);
        nh.param<std::string>("lio_sam/savePCDDirectory", savePCDDirectory, "/Downloads/LOAM/");

        nh.param<float>("lio_sam/min_range", min_range, 1.0);
        nh.param<float>("lio_sam/max_range", max_range, 150.0);
        nh.param<float>("lio_sam/Vertical_angle", Vertical_angle, 30.0);
        nh.param<float>("lio_sam/ang_bottom", ang_bottom, 15.0);
        nh.param<string>("lio_sam/lidar_type", lidar_type, "rslidar");
        if (lidar_type == "Velodyne")
            mlidar_type = Velodyne;
        else if (lidar_type == "rslidar")
            mlidar_type = rslidar;
        // else if(lidar_type=="rslidar_16")
        //      mlidar_type=rslidar_16;
        // else if(lidar_type=="rslidar_32")
        //      mlidar_type=rslidar_32;
        else
        {
            std::cout << "Undefined lidar type " << std::endl;
            exit(-1);
        }

        nh.param<int>("lio_sam/N_SCAN", N_SCAN, 16);
        nh.param<int>("lio_sam/Horizon_SCAN", Horizon_SCAN, 1800);
        nh.param<std::string>("lio_sam/timeField", timeField, "time");
        nh.param<int>("lio_sam/downsampleRate", downsampleRate, 1);

        nh.param<float>("lio_sam/imuAccNoise", imuAccNoise, 0.01);
        nh.param<float>("lio_sam/imuGyrNoise", imuGyrNoise, 0.001);
        nh.param<float>("lio_sam/imuAccBiasN", imuAccBiasN, 0.0002);
        nh.param<float>("lio_sam/imuGyrBiasN", imuGyrBiasN, 0.00003);
        nh.param<float>("lio_sam/imuGravity", imuGravity, 9.80511);
        nh.param<int>("lio_sam/imuFrequency", imuFrequency, 200);
        nh.param<int>("lio_sam/area_num", area_num, 6);
        nh.param<int>("lio_sam/iter_num", iter_num, 30);
        nh.param<float>("lio_sam/distance_limit", distance_limit, 2500.0);

        nh.param<int>("lio_sam/PickedSurfNum", PickedSurfNum, 100);
        skipSurf = 300 / PickedSurfNum;
        nh.param<int>("lio_sam/PickedCornerNum", PickedCornerNum, 20);
        skipCorner = 300 / PickedCornerNum;

        nh.param<vector<double>>("lio_sam/extrinsicRot", extRotV, vector<double>());
        nh.param<vector<double>>("lio_sam/extrinsicRPY", extRPYV, vector<double>());
        nh.param<vector<double>>("lio_sam/extrinsicTrans", extTransV, vector<double>());
        extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
        extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
        extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
        extQRPY = Eigen::Quaterniond(extRPY.transpose());
        // extQRPY = Eigen::Quaterniond(extRPY);

        Pli << extTrans.x(), extTrans.y(), extTrans.z();
        Pil = extQRPY * (-Pli);
        nh.param<float>("lio_sam/edgeThreshold", edgeThreshold, 0.1);
        nh.param<float>("lio_sam/surfThreshold", surfThreshold, 0.1);
        nh.param<int>("lio_sam/edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);
        nh.param<int>("lio_sam/surfFeatureMinValidNum", surfFeatureMinValidNum, 100);

        nh.param<float>("lio_sam/odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
        nh.param<float>("lio_sam/mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
        nh.param<float>("lio_sam/mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

        nh.param<float>("lio_sam/z_tollerance", z_tollerance, FLT_MAX);
        nh.param<float>("lio_sam/rotation_tollerance", rotation_tollerance, FLT_MAX);

        nh.param<int>("lio_sam/numberOfCores", numberOfCores, 2);
        nh.param<double>("lio_sam/mappingProcessInterval", mappingProcessInterval, 0.15);

        nh.param<float>("lio_sam/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0);
        nh.param<float>("lio_sam/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
        nh.param<float>("lio_sam/surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
        nh.param<float>("lio_sam/surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);

        nh.param<bool>("lio_sam/loopClosureEnableFlag", loopClosureEnableFlag, false);
        nh.param<float>("lio_sam/loopClosureFrequency", loopClosureFrequency, 1.0);
        nh.param<int>("lio_sam/surroundingKeyframeSize", surroundingKeyframeSize, 50);
        nh.param<float>("lio_sam/historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
        nh.param<float>("lio_sam/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
        nh.param<int>("lio_sam/historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
        nh.param<float>("lio_sam/historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);

        nh.param<float>("lio_sam/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
        nh.param<float>("lio_sam/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
        nh.param<float>("lio_sam/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);

        usleep(100);
    }
    // std::mutex mtx;
    sensor_msgs::Imu imuConverter(const sensor_msgs::Imu &imu_in)
    {
        sensor_msgs::Imu imu_out = imu_in;
        // rotate acceleration
        Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
        acc = extRot * acc;
        imu_out.linear_acceleration.x = acc.x();
        imu_out.linear_acceleration.y = acc.y();
        imu_out.linear_acceleration.z = acc.z();
        // rotate gyroscope
        Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
        gyr = extRot * gyr;
        imu_out.angular_velocity.x = gyr.x();
        imu_out.angular_velocity.y = gyr.y();
        imu_out.angular_velocity.z = gyr.z();
        // rotate roll pitch yaw
        Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
        Eigen::Quaterniond q_final = (q_from * extQRPY).normalized();
        // Eigen::Quaterniond q_final=q_from.normalized();
        // std::lock_guard<std::mutex> lock(mtx);
        // cout<<"********************"<<endl;
        // cout<<"q_from * extQRPY"<<(q_from * extQRPY).matrix()<<endl;
        // cout<<"extQRPY* q_from"<<(extQRPY * q_from).matrix()<<endl;
        // usleep(1000);

        imu_out.orientation.x = q_final.x();
        imu_out.orientation.y = q_final.y();
        imu_out.orientation.z = q_final.z();
        imu_out.orientation.w = q_final.w();

        if (sqrt(q_final.x() * q_final.x() + q_final.y() * q_final.y() + q_final.z() * q_final.z() + q_final.w() * q_final.w()) < 0.1)
        {
            ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
            ros::shutdown();
        }

        return imu_out;
    }
};

/**
 * @details 发布点云
 * @param thisPub 发布者
 * @param thisCloud 点云指针
 * @param thisStamp 时间戳
 * @param thisFrame 坐标系
 * @return thisCloud对应的ros点云
 */
sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->getNumSubscribers() != 0)
        thisPub->publish(tempCloud);
    return tempCloud;
}

//获取msg的时间
template <typename T>
double ROS_TIME(T msg)
{
    return msg->header.stamp.toSec();
}

// ros -> typename T
template <typename T>
void imuAngular2rosAngular(sensor_msgs::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z)
{
    *angular_x = thisImuMsg->angular_velocity.x;
    *angular_y = thisImuMsg->angular_velocity.y;
    *angular_z = thisImuMsg->angular_velocity.z;
}

// ros -> typename T
template <typename T>
void imuAccel2rosAccel(sensor_msgs::Imu *thisImuMsg, T *acc_x, T *acc_y, T *acc_z)
{
    *acc_x = thisImuMsg->linear_acceleration.x;
    *acc_y = thisImuMsg->linear_acceleration.y;
    *acc_z = thisImuMsg->linear_acceleration.z;
}

// sensor_msgs::Imu四元数 -> ros::roll pitch yaw
template <typename T>
void imuRPY2rosRPY(sensor_msgs::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw)
{
    double imuRoll, imuPitch, imuYaw;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(thisImuMsg->orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

    *rosRoll = imuRoll;
    *rosPitch = imuPitch;
    *rosYaw = imuYaw;
}

float pointDistance(PointType p)
{
    return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}

#endif
