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

using namespace std;

class ParamServer
{
public:
    ros::NodeHandle nh;

    /*loop clousre*/
    bool startFlag = true;
    bool loopClosureEnableFlag;
    float loopClosureFrequency; //   回环检测频率
    int surroundingKeyframeSize;
    float historyKeyframeSearchRadius;   // 回环检测 radius kdtree搜索半径
    float historyKeyframeSearchTimeDiff; //  帧间时间阈值
    int historyKeyframeSearchNum;        //   回环时多少个keyframe拼成submap
    float historyKeyframeFitnessScore;   // icp 匹配阈值
    bool potentialLoopFlag = false;

    bool visulize_IkdtreeMap = false;            //  visual iktree submap

    // voxel filter paprams
    float odometrySurfLeafSize;
    float mappingCornerLeafSize;
    float mappingSurfLeafSize;

    float z_tollerance;
    float rotation_tollerance;

    // CPU Params
    int numberOfCores = 4;
    double mappingProcessInterval;

    bool useImuHeadingInitialization;
    bool useGpsElevation;   //  是否使用gps高层优化
    float gpsCovThreshold;  //  gps方向角和高度差的协方差阈值
    float poseCovThreshold; //  位姿协方差阈值  from isam2

    // global map visualization radius
    float globalMapVisualizationSearchRadius;
    float globalMapVisualizationPoseDensity;
    float globalMapVisualizationLeafSize;

    // Surrounding map
    float surroundingkeyframeAddingDistThreshold;  //  判断是否为关键帧的距离阈值
    float surroundingkeyframeAddingAngleThreshold; //  判断是否为关键帧的角度阈值
    float surroundingKeyframeDensity;
    float surroundingKeyframeSearchRadius;

    bool recontructKdTree = false;

    bool savePCD;            // 是否保存地图
    string savePCDDirectory; // 保存路径

    string gps_topic;

    vector<double> extrinT_Gnss2Lidar;
    vector<double> extrinR_Gnss2Lidar;

    // ROS的参数服务
    ParamServer()
    {
        nh.param<float>("odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
        nh.param<float>("mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
        nh.param<float>("mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

        nh.param<float>("z_tollerance", z_tollerance, FLT_MAX);
        nh.param<float>("rotation_tollerance", rotation_tollerance, FLT_MAX);

        nh.param<int>("numberOfCores", numberOfCores, 2);
        nh.param<double>("mappingProcessInterval", mappingProcessInterval, 0.15);

        // save keyframes
        nh.param<float>("surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 20.0);
        nh.param<float>("surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
        nh.param<float>("surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
        nh.param<float>("surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);

        // loop clousre
        nh.param<bool>("loopClosureEnableFlag", loopClosureEnableFlag, false);
        nh.param<float>("loopClosureFrequency", loopClosureFrequency, 1.0);
        nh.param<int>("surroundingKeyframeSize", surroundingKeyframeSize, 50);
        nh.param<float>("historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
        nh.param<float>("historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
        nh.param<int>("historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
        nh.param<float>("historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);

        // gnss
        nh.param<string>("common/gps_topic", gps_topic, "/gps/fix");
        nh.param<vector<double>>("mapping/extrinR_Gnss2Lidar", extrinR_Gnss2Lidar, vector<double>());
        nh.param<vector<double>>("mapping/extrinT_Gnss2Lidar", extrinT_Gnss2Lidar, vector<double>());
        nh.param<bool>("useImuHeadingInitialization", useImuHeadingInitialization, false);
        nh.param<bool>("useGpsElevation", useGpsElevation, false);
        nh.param<float>("gpsCovThreshold", gpsCovThreshold, 2.0);
        nh.param<float>("poseCovThreshold", poseCovThreshold, 25.0);

        // Visualization
        nh.param<float>("globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
        nh.param<float>("globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
        nh.param<float>("globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);

        // visual ikdtree map
        nh.param<bool>("visulize_IkdtreeMap", visulize_IkdtreeMap, false);

        // reconstruct ikdtree
        nh.param<bool>("recontructKdTree", recontructKdTree, false);

        // savMap
        nh.param<bool>("savePCD", savePCD, false);
        nh.param<std::string>("savePCDDirectory", savePCDDirectory, "/Downloads/LOAM/");

        usleep(100);
    }
};

#endif