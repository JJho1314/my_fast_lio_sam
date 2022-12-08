#include "fast_lio.hpp"
#include <utility.h>
// #include <omp.h>
// #include <mutex>
// #include <math.h>
// #include <thread>
// #include <fstream>
// #include <csignal>
// #include <unistd.h>
// #include <Python.h>
// #include <so3_math.h>
// #include <ros/ros.h>
// #include <Eigen/Core>
// #include "IMU_Processing.hpp"
// #include <nav_msgs/Odometry.h>
// #include <nav_msgs/Path.h>
// #include <visualization_msgs/Marker.h>
// #include <pcl_conversions/pcl_conversions.h>
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/io/pcd_io.h>
// #include <sensor_msgs/PointCloud2.h>
// #include <tf/transform_datatypes.h>
// #include <tf/transform_broadcaster.h>
// #include <geometry_msgs/Vector3.h>
// #include <livox_ros_driver/CustomMsg.h>
// #include "preprocess.h"
// #include <ikd-Tree/ikd_Tree.h>

// #include <std_msgs/Header.h>
// #include <std_msgs/Float64MultiArray.h>
// #include <sensor_msgs/Imu.h>
// #include <sensor_msgs/PointCloud2.h>
// #include <sensor_msgs/NavSatFix.h>
// #include <nav_msgs/Odometry.h>
// #include <nav_msgs/Path.h>
// #include <visualization_msgs/Marker.h>
// #include <visualization_msgs/MarkerArray.h>

// #include <pcl/search/impl/search.hpp>
// #include <pcl/range_image/range_image.h>
// #include <pcl/kdtree/kdtree_flann.h>
// #include <pcl/common/common.h>
// #include <pcl/common/transforms.h>
// #include <pcl/registration/icp.h>
// #include <pcl/io/pcd_io.h>
// #include <pcl/filters/filter.h>
// #include <pcl/filters/crop_box.h>
// #include <pcl_conversions/pcl_conversions.h>

// // gstam
// #include <gtsam/geometry/Rot3.h>
// #include <gtsam/geometry/Pose3.h>
// #include <gtsam/slam/PriorFactor.h>
// #include <gtsam/slam/BetweenFactor.h>
// #include <gtsam/navigation/GPSFactor.h>
// #include <gtsam/navigation/ImuFactor.h>
// #include <gtsam/navigation/CombinedImuFactor.h>
// #include <gtsam/nonlinear/NonlinearFactorGraph.h>
// #include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
// #include <gtsam/nonlinear/Marginals.h>
// #include <gtsam/nonlinear/Values.h>
// #include <gtsam/inference/Symbol.h>
// #include <gtsam/nonlinear/ISAM2.h>

// // gnss
// #include "GNSS_Processing.hpp"
// #include "sensor_msgs/NavSatFix.h"

// // save map
// #include "fast_lio_sam/save_map.h"
// #include "fast_lio_sam/save_pose.h"

// // save data in kitti format
// #include <sstream>
// #include <fstream>
// #include <iomanip>

// using namespace gtsam;

#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define MAXN (720000)
#define PUBFRAME_PERIOD (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
/**************************/

float res_last[100000] = {0.0}; //残差，点到面距离平方和

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
bool point_selected_surf[100000] = {0}; // 是否为平面特征点
bool lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;

vector<vector<int>> pointSearchInd_surf;
vector<BoxPointType> cub_needrm; // ikd-tree中，地图需要移除的包围盒序列
vector<PointVector> Nearest_Points;

deque<double> time_buffer;               // 记录lidar时间
deque<PointCloudXYZI::Ptr> lidar_buffer; //记录特征提取或间隔采样后的lidar（特征）数据
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());  //畸变纠正后降采样的单帧点云，lidar系
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI()); //畸变纠正后降采样的单帧点云，w系
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1)); //特征点在地图中对应点的，局部平面参数,w系
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1)); //对应点法相量？
PointCloudXYZI::Ptr _featsArray;                                  // ikd-tree中，map需要移除的点云

pcl::VoxelGrid<PointType> downSizeFilterSurf; //单帧内降采样使用voxel grid
pcl::VoxelGrid<PointType> downSizeFilterMap;  //未使用

KD_TREE ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d); // T lidar to imu (imu = r * lidar + t)
M3D Lidar_R_wrt_IMU(Eye3d);  // R lidar to imu (imu = r * lidar + t)

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf; // 状态，噪声维度，输入
state_ikfom state_point;
vect3 pos_lid; // world系下lidar坐标

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

/*back end*/
vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames; // 历史所有关键帧的角点集合（降采样）
vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;   // 历史所有关键帧的平面点集合（降采样）

pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D(new pcl::PointCloud<PointType>());         // 历史关键帧位姿（位置）
pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>()); // 历史关键帧位姿
pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>());

pcl::PointCloud<PointTypePose>::Ptr fastlio_unoptimized_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>()); //  存储fastlio 未优化的位姿
pcl::PointCloud<PointTypePose>::Ptr gnss_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>());                //  gnss 轨迹

ros::Publisher pubHistoryKeyFrames; //  发布 loop history keyframe submap
ros::Publisher pubIcpKeyFrames;
ros::Publisher pubRecentKeyFrames;
ros::Publisher pubRecentKeyFrame;
ros::Publisher pubCloudRegisteredRaw;
ros::Publisher pubLoopConstraintEdge;

bool aLoopIsClosed = false;
map<int, int> loopIndexContainer; // from new to old
vector<pair<int, int>> loopIndexQueue;
vector<gtsam::Pose3> loopPoseQueue;
vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
deque<std_msgs::Float64MultiArray> loopInfoVec;

// 局部关键帧构建的map点云，对应kdtree，用于scan-to-map找相邻点
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());

pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses(new pcl::KdTreeFLANN<PointType>());

// 降采样
pcl::VoxelGrid<PointType> downSizeFilterCorner;
// pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterICP;
pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization

float transformTobeMapped[6]; //  当前帧的位姿(world系下)

std::mutex mtx;
std::mutex mtxLoopInfo;

// gtsam
gtsam::NonlinearFactorGraph gtSAMgraph;
gtsam::Values initialEstimate;
gtsam::Values optimizedEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;
Eigen::MatrixXd poseCovariance;

ros::Publisher pubLaserCloudSurround;
ros::Publisher pubOptimizedGlobalMap; //   发布最后优化的地图

int updateKdtreeCount = 0;        //  每100次更新一次
bool visulize_IkdtreeMap = false; //  visual iktree submap

// gnss
double last_timestamp_gnss = -1.0;
deque<nav_msgs::Odometry> gnss_buffer;
geometry_msgs::PoseStamped msg_gnss_pose;

bool gnss_inited = false; //  是否完成gnss初始化
shared_ptr<GnssProcess> p_gnss(new GnssProcess());
GnssProcess gnss_data;
ros::Publisher pubGnssPath;
nav_msgs::Path gps_path;

// saveMap
ros::ServiceServer srvSaveMap;
ros::ServiceServer srvSavePose;

/*定义pose结构体*/
struct pose
{
    Eigen::Vector3d t;
    Eigen::Matrix3d R;
};

class mapOptimization : public ParamServer
{
public:

    void saveKeyFramesAndFactor()
    {
        //  计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
        if (saveFrame() == false)
            return;
        // 激光里程计因子(from fast-lio),  输入的是frame_relative pose  帧间位姿(body 系下)
        addOdomFactor();
        // GPS因子 (UTM -> WGS84)
        addGPSFactor();
        // 闭环因子 (rs-loop-detect)  基于欧氏距离的检测
        addLoopFactor();
        // 执行优化
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();
        if (aLoopIsClosed == true) // 有回环因子，多update几次
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }
        // update之后要清空一下保存的因子图，注：历史数据不会清掉，ISAM保存起来了
        gtSAMgraph.resize(0);
        initialEstimate.clear();

        PointType thisPose3D;
        PointTypePose thisPose6D;
        gtsam::Pose3 latestEstimate;

        // 优化结果
        isamCurrentEstimate = isam->calculateBestEstimate();
        // 当前帧位姿结果
        latestEstimate = isamCurrentEstimate.at<gtsam::Pose3>(isamCurrentEstimate.size() - 1);

        // cloudKeyPoses3D加入当前帧位置
        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        // 索引
        thisPose3D.intensity = cloudKeyPoses3D->size(); //  使用intensity作为该帧点云的index
        cloudKeyPoses3D->push_back(thisPose3D);         //  新关键帧帧放入队列中

        // cloudKeyPoses6D加入当前帧位姿
        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity;
        thisPose6D.roll = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw = latestEstimate.rotation().yaw();
        thisPose6D.time = lidar_end_time;
        cloudKeyPoses6D->push_back(thisPose6D);

        // 位姿协方差
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size() - 1);

        // ESKF状态和方差  更新
        state_ikfom state_updated = kf.get_x(); //  获取cur_pose (还没修正)
        Eigen::Vector3d pos(latestEstimate.translation().x(), latestEstimate.translation().y(), latestEstimate.translation().z());
        Eigen::Quaterniond q = EulerToQuat(latestEstimate.rotation().roll(), latestEstimate.rotation().pitch(), latestEstimate.rotation().yaw());

        //  更新状态量
        state_updated.pos = pos;
        state_updated.rot = q;
        state_point = state_updated; // 对state_point进行更新，state_point可视化用到
        // if(aLoopIsClosed == true )
        kf.change_x(state_updated); //  对cur_pose 进行isam2优化后的修正

        // TODO:  P的修正有待考察，按照yanliangwang的做法，修改了p，会跑飞
        // esekfom::esekf<state_ikfom, 12, input_ikfom>::cov P_updated = kf.get_P(); // 获取当前的状态估计的协方差矩阵
        // P_updated.setIdentity();
        // P_updated(6, 6) = P_updated(7, 7) = P_updated(8, 8) = 0.00001;
        // P_updated(9, 9) = P_updated(10, 10) = P_updated(11, 11) = 0.00001;
        // P_updated(15, 15) = P_updated(16, 16) = P_updated(17, 17) = 0.0001;
        // P_updated(18, 18) = P_updated(19, 19) = P_updated(20, 20) = 0.001;
        // P_updated(21, 21) = P_updated(22, 22) = 0.00001;
        // kf.change_P(P_updated);

        // 当前帧激光角点、平面点，降采样集合
        // pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        // pcl::copyPointCloud(*feats_undistort,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*feats_undistort, *thisSurfKeyFrame); // 存储关键帧,没有降采样的点云

        // 保存特征点降采样集合
        // cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        updatePath(thisPose6D); //  可视化update后的path
    }

    bool savePoseService(fast_lio_sam::save_poseRequest &req, fast_lio_sam::save_poseResponse &res)
    {
        pose pose_gnss;
        pose pose_optimized;
        pose pose_without_optimized;

        std::ofstream file_pose_gnss;
        std::ofstream file_pose_optimized;
        std::ofstream file_pose_without_optimized;

        string savePoseDirectory;
        cout << "****************************************************" << endl;
        cout << "Saving poses to pose files ..." << endl;
        if (req.destination.empty())
            savePoseDirectory = std::getenv("HOME") + savePCDDirectory;
        else
            savePoseDirectory = std::getenv("HOME") + req.destination;
        cout << "Save destination: " << savePoseDirectory << endl;

        // create file
        CreateFile(file_pose_gnss, savePoseDirectory + "/gnss_pose.txt");
        CreateFile(file_pose_optimized, savePoseDirectory + "/optimized_pose.txt");
        CreateFile(file_pose_without_optimized, savePoseDirectory + "/without_optimized_pose.txt");

        //  save optimize data
        for (int i = 0; i < cloudKeyPoses6D->size(); i++)
        {
            pose_optimized.t = Eigen::Vector3d(cloudKeyPoses6D->points[i].x, cloudKeyPoses6D->points[i].y, cloudKeyPoses6D->points[i].z);
            pose_optimized.R = Exp(double(cloudKeyPoses6D->points[i].roll), double(cloudKeyPoses6D->points[i].pitch), double(cloudKeyPoses6D->points[i].yaw));
            WriteText(file_pose_optimized, pose_optimized);
        }
        cout << "Sucess global optimized  poses to pose files ..." << endl;

        for (int i = 0; i < fastlio_unoptimized_cloudKeyPoses6D->size(); i++)
        {
            pose_without_optimized.t = Eigen::Vector3d(fastlio_unoptimized_cloudKeyPoses6D->points[i].x, fastlio_unoptimized_cloudKeyPoses6D->points[i].y, fastlio_unoptimized_cloudKeyPoses6D->points[i].z);
            pose_without_optimized.R = Exp(double(fastlio_unoptimized_cloudKeyPoses6D->points[i].roll), double(fastlio_unoptimized_cloudKeyPoses6D->points[i].pitch), double(fastlio_unoptimized_cloudKeyPoses6D->points[i].yaw));
            WriteText(file_pose_without_optimized, pose_without_optimized);
        }
        cout << "Sucess unoptimized  poses to pose files ..." << endl;

        for (int i = 0; i < gnss_cloudKeyPoses6D->size(); i++)
        {
            pose_gnss.t = Eigen::Vector3d(gnss_cloudKeyPoses6D->points[i].x, gnss_cloudKeyPoses6D->points[i].y, gnss_cloudKeyPoses6D->points[i].z);
            pose_gnss.R = Exp(double(gnss_cloudKeyPoses6D->points[i].roll), double(gnss_cloudKeyPoses6D->points[i].pitch), double(gnss_cloudKeyPoses6D->points[i].yaw));
            WriteText(file_pose_gnss, pose_gnss);
        }
        cout << "Sucess gnss  poses to pose files ..." << endl;

        file_pose_gnss.close();
        file_pose_optimized.close();
        file_pose_without_optimized.close();
        return true;
    }

    /**
     * 保存全局关键帧特征点集合
     */
    bool saveMapService(fast_lio_sam::save_mapRequest &req, fast_lio_sam::save_mapResponse &res)
    {
        string saveMapDirectory;

        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;
        if (req.destination.empty())
            saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
        else
            saveMapDirectory = std::getenv("HOME") + req.destination;
        cout << "Save destination: " << saveMapDirectory << endl;
        // 这个代码太坑了！！注释掉
        //   int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
        //   unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
        // 保存历史关键帧位姿
        pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);      // 关键帧位置
        pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D); // 关键帧位姿
        // 提取历史关键帧角点、平面点集合
        //   pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
        //   pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());

        // 注意：拼接地图时，keyframe是lidar系，而fastlio更新后的存到的cloudKeyPoses6D 关键帧位姿是body系下的，需要把
        // cloudKeyPoses6D  转换为T_world_lidar 。 T_world_lidar = T_world_body * T_body_lidar , T_body_lidar 是外参
        for (int i = 0; i < (int)cloudKeyPoses6D->size(); i++)
        {
            //   *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
            *globalSurfCloud += *transformPointCloud(surfCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
            cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
        }

        if (req.resolution != 0)
        {
            cout << "\n\nSave resolution: " << req.resolution << endl;

            // 降采样
            // downSizeFilterCorner.setInputCloud(globalCornerCloud);
            // downSizeFilterCorner.setLeafSize(req.resolution, req.resolution, req.resolution);
            // downSizeFilterCorner.filter(*globalCornerCloudDS);
            // pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloudDS);
            // 降采样
            downSizeFilterSurf.setInputCloud(globalSurfCloud);
            downSizeFilterSurf.setLeafSize(req.resolution, req.resolution, req.resolution);
            downSizeFilterSurf.filter(*globalSurfCloudDS);
            pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
        }
        else
        {
            //   downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
            downSizeFilterSurf.setInputCloud(globalSurfCloud);
            downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
            downSizeFilterSurf.filter(*globalSurfCloudDS);
            // pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloud);
            // pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);           //  稠密点云地图
        }

        // 保存到一起，全局关键帧特征点集合
        //   *globalMapCloud += *globalCornerCloud;
        *globalMapCloud += *globalSurfCloud;
        pcl::io::savePCDFileBinary(saveMapDirectory + "/filterGlobalMap.pcd", *globalSurfCloudDS);  //  滤波后地图
        int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud); //  稠密地图
        res.success = ret == 0;

        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed\n"
             << endl;

        // visial optimize global map on viz
        ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time);
        string odometryFrame = "camera_init";
        publishCloud(&pubOptimizedGlobalMap, globalSurfCloudDS, timeLaserInfoStamp, odometryFrame);

        return true;
    }

    void saveMap()
    {
        fast_lio_sam::save_mapRequest req;
        fast_lio_sam::save_mapResponse res;
        // 保存全局关键帧特征点集合
        if (!saveMapService(req, res))
        {
            cout << "Fail to save map" << endl;
        }
    }

    /**
     * 发布局部关键帧map的特征点云
     */
    void publishGlobalMap()
    {
        /*** if path is too large, the rvis will crash ***/
        ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time);
        string odometryFrame = "camera_init";
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;
        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());
        ;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kdtree查找最近一帧关键帧相邻的关键帧集合
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // 降采样
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
        // 提取局部相邻关键帧对应的特征点云
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i)
        {
            // 距离过大
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            // *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]); //  fast_lio only use  surfCloud
        }
        // 降采样，发布
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;                                                                                   // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
    }

    void recontructIKdTree()
    {
        if (recontructKdTree && updateKdtreeCount > 0)
        {
            /*** if path is too large, the rvis will crash ***/
            pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMapPoses(new pcl::KdTreeFLANN<PointType>());
            pcl::PointCloud<PointType>::Ptr subMapKeyPoses(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr subMapKeyPosesDS(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr subMapKeyFrames(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr subMapKeyFramesDS(new pcl::PointCloud<PointType>());

            // kdtree查找最近一帧关键帧相邻的关键帧集合
            std::vector<int> pointSearchIndGlobalMap;
            std::vector<float> pointSearchSqDisGlobalMap;
            mtx.lock();
            kdtreeGlobalMapPoses->setInputCloud(cloudKeyPoses3D);
            kdtreeGlobalMapPoses->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
            mtx.unlock();

            for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
                subMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]); //  subMap的pose集合
            // 降采样
            pcl::VoxelGrid<PointType> downSizeFilterSubMapKeyPoses;
            downSizeFilterSubMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
            downSizeFilterSubMapKeyPoses.setInputCloud(subMapKeyPoses);
            downSizeFilterSubMapKeyPoses.filter(*subMapKeyPosesDS); //  subMap poses  downsample
            // 提取局部相邻关键帧对应的特征点云
            for (int i = 0; i < (int)subMapKeyPosesDS->size(); ++i)
            {
                // 距离过大
                if (pointDistance(subMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                    continue;
                int thisKeyInd = (int)subMapKeyPosesDS->points[i].intensity;
                // *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
                *subMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]); //  fast_lio only use  surfCloud
            }
            // 降采样，发布
            pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;                                                                                   // for global map visualization
            downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
            downSizeFilterGlobalMapKeyFrames.setInputCloud(subMapKeyFrames);
            downSizeFilterGlobalMapKeyFrames.filter(*subMapKeyFramesDS);

            std::cout << "subMapKeyFramesDS sizes  =  " << subMapKeyFramesDS->points.size() << std::endl;

            ikdtree.reconstruct(subMapKeyFramesDS->points);
            updateKdtreeCount = 0;
            ROS_INFO("Reconstructed  ikdtree ");
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            std::cout << "featsFromMapNum  =  " << featsFromMapNum << "\t"
                      << " kdtree_size_st   =  " << kdtree_size_st << std::endl;
        }
        updateKdtreeCount++;
    }

    /**
     * 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹
     */
    void correctPoses()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        if (aLoopIsClosed == true)
        {
            // 清空里程计轨迹
            globalPath.poses.clear();
            // 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().yaw();

                // 更新里程计轨迹
                updatePath(cloudKeyPoses6D->points[i]);
            }
            // 清空局部map， reconstruct  ikdtree submap
            recontructIKdTree();
            ROS_INFO("ISMA2 Update");
            aLoopIsClosed = false;
        }
    }

    mapOptimization() : gps_initailized(false), pose_initailized(false), Calib_flag(false)
    {
        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

    }

private:
    bool gps_initailized;
    bool pose_initailized;
    bool Calib_flag;
    nav_msgs::Path globalPath;

    /**
     * 更新里程计轨迹
     */
    void updatePath(const PointTypePose &pose_in)
    {
        string odometryFrame = "camera_init";
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);

        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

    /**
     * 对点云cloudIn进行变换transformIn，返回结果点云， 修改liosam, 考虑到外参的表示
     */
    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        // 注意：lio_sam 中的姿态用的euler表示，而fastlio存的姿态角是旋转矢量。而 pcl::getTransformation是将euler_angle 转换到rotation_matrix 不合适，注释
        // Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        Eigen::Isometry3d T_b_lidar(state_point.offset_R_L_I); //  获取  body2lidar  外参
        T_b_lidar.pretranslate(state_point.offset_T_L_I);

        Eigen::Affine3f T_w_b_ = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        Eigen::Isometry3d T_w_b; //   world2body
        T_w_b.matrix() = T_w_b_.matrix().cast<double>();

        Eigen::Isometry3d T_w_lidar = T_w_b * T_b_lidar; //  T_w_lidar  转换矩阵

        Eigen::Isometry3d transCur = T_w_lidar;

#pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            cloudOut->points[i].x = transCur(0, 0) * pointFrom.x + transCur(0, 1) * pointFrom.y + transCur(0, 2) * pointFrom.z + transCur(0, 3);
            cloudOut->points[i].y = transCur(1, 0) * pointFrom.x + transCur(1, 1) * pointFrom.y + transCur(1, 2) * pointFrom.z + transCur(1, 3);
            cloudOut->points[i].z = transCur(2, 0) * pointFrom.x + transCur(2, 1) * pointFrom.y + transCur(2, 2) * pointFrom.z + transCur(2, 3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    /**
     * 位姿格式变换
     */
    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                            gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
    }

    /**
     * 位姿格式变换
     */
    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                            gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    /**
     * Eigen格式的位姿变换
     */
    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    {
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    /**
     * Eigen格式的位姿变换
     */
    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    /**
     * 位姿格式变换
     */
    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw = transformIn[2];
        return thisPose6D;
    }

    /**
     * 发布thisCloud，返回thisCloud对应msg格式
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

    /**
     * 点到坐标系原点距离
     */
    float pointDistance(PointType p)
    {
        return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    }

    /**
     * 两点之间距离
     */
    float pointDistance(PointType p1, PointType p2)
    {
        return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
    }

    /**
     * 初始化
     */
    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudOri.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i)
        {
            transformTobeMapped[i] = 0;
        }
    }

    /**
     * rviz展示闭环边
     */
    void visualizeLoopClosure()
    {
        ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time); //  时间戳
        string odometryFrame = "camera_init";

        if (loopIndexContainer.empty())
            return;

        visualization_msgs::MarkerArray markerArray;
        // 闭环顶点
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3;
        markerNode.scale.y = 0.3;
        markerNode.scale.z = 0.3;
        markerNode.color.r = 0;
        markerNode.color.g = 0.8;
        markerNode.color.b = 1;
        markerNode.color.a = 1;
        // 闭环边
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1;
        markerEdge.color.r = 0.9;
        markerEdge.color.g = 0.9;
        markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        // 遍历闭环
        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge.publish(markerArray);
    }

    /**
     * 计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
     */
    bool saveFrame()
    {
        if (cloudKeyPoses3D->points.empty())
            return true;

        // 前一帧位姿
        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        // 当前帧位姿
        Eigen::Affine3f transFinal = trans2Affine3f(transformTobeMapped);
        // Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
        //                                                     transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

        // 位姿变换增量
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw); //  获取上一帧 相对 当前帧的 位姿

        // 旋转和平移量都较小，当前帧不设为关键帧
        if (abs(roll) < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
            abs(yaw) < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold)
            return false;
        return true;
    }

    /**
     * 添加激光里程计因子
     */
    void addOdomFactor()
    {
        if (cloudKeyPoses3D->points.empty())
        {
            // 第一帧初始化先验因子
            gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12).finished()); // rad*rad, meter*meter   // indoor 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12    //  1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8
            gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            // 变量节点设置初始值
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }
        else
        {
            // 添加激光里程计因子
            gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back()); /// pre
            gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);                   // cur
            // 参数：前一帧id，当前帧id，前一帧与当前帧的位姿变换（作为观测值），噪声协方差
            gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            // 变量节点设置初始值
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }

    /**
     * 添加闭环因子
     */
    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

        // 闭环队列
        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            // 闭环边对应两帧的索引
            int indexFrom = loopIndexQueue[i].first; //   cur
            int indexTo = loopIndexQueue[i].second;  //    pre
            // 闭环边的位姿变换
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
    }

    /**
     * 添加GPS因子
     */
    void addGPSFactor()
    {
        if (gnss_buffer.empty())
            return;
        // 如果没有关键帧，或者首尾关键帧距离小于5m，不添加gps因子
        if (cloudKeyPoses3D->points.empty())
            return;
        else
        {
            if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
                return;
        }
        // 位姿协方差很小，没必要加入GPS数据进行校正
        if (poseCovariance(3, 3) < poseCovThreshold && poseCovariance(4, 4) < poseCovThreshold)
            return;
        static PointType lastGPSPoint; // 最新的gps数据
        while (!gnss_buffer.empty())
        {
            // 删除当前帧0.2s之前的里程计
            if (gnss_buffer.front().header.stamp.toSec() < lidar_end_time - 0.05)
            {
                gnss_buffer.pop_front();
            }
            // 超过当前帧0.2s之后，退出
            else if (gnss_buffer.front().header.stamp.toSec() > lidar_end_time + 0.05)
            {
                break;
            }
            else
            {
                nav_msgs::Odometry thisGPS = gnss_buffer.front();
                gnss_buffer.pop_front();
                // GPS噪声协方差太大，不能用
                float noise_x = thisGPS.pose.covariance[0]; //  x 方向的协方差
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14]; //   z(高层)方向的协方差
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;
                // GPS里程计位置
                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                if (!useGpsElevation) //  是否使用gps的高度
                {
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }

                // (0,0,0)无效数据
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;
                // 每隔5m添加一个GPS里程计
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;
                // 添加GPS因子
                gtsam::Vector Vector3(3);
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                gtsam::noiseModel::Diagonal::shared_ptr gps_noise = gtsam::noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);
                aLoopIsClosed = true;
                ROS_INFO("GPS Factor Added");
                break;
            }
        }
    }

    /*回环检测三大要素
       1.设置最小时间差，太近没必要
       2.控制回环的频率，避免频繁检测，每检测一次，就做一次等待
       3.根据当前最小距离重新计算等待时间
    */
    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        // 当前关键帧帧
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1; //  当前关键帧索引
        int loopKeyPre = -1;

        // 当前帧已经添加过闭环对应关系，不再继续添加
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;
        // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合
        std::vector<int> pointSearchIndLoop;                        //  候选关键帧索引
        std::vector<float> pointSearchSqDisLoop;                    //  候选关键帧距离
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D); //  历史帧构建kdtree
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        // 在候选关键帧集合中，找到与当前帧时间相隔较远的帧，设为候选匹配帧
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            if (abs(copy_cloudKeyPoses6D->points[id].time - lidar_end_time) > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;
                break;
            }
        }
        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;
        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        ROS_INFO("Find loop clousre frame ");
        return true;
    }

    /**
     * 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合，降采样
     */
    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum)
    {
        // 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize)
                continue;

            // *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
            // 注意：cloudKeyPoses6D 存储的是 T_w_b , 而点云是lidar系下的，构建icp的submap时，需要通过外参数T_b_lidar 转换 , 参考pointBodyToWorld 的转换
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]); //  fast-lio 没有进行特征提取，默认点云就是surf
        }

        if (nearKeyframes->empty())
            return;

        // 降采样
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    void performLoopClosure()
    {
        ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time); //  时间戳
        string odometryFrame = "camera_init";

        if (cloudKeyPoses3D->points.empty() == true)
        {
            return;
        }

        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // 当前关键帧索引，候选闭环匹配帧索引
        int loopKeyCur;
        int loopKeyPre;
        // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
        if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
        {
            return;
        }

        // 提取
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>()); //  cue keyframe
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>()); //   history keyframe submap
        {
            // 提取当前关键帧特征点集合，降采样
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0); //  将cur keyframe 转换到world系下
            // 提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum); //  选取historyKeyframeSearchNum个keyframe拼成submap
            // 如果特征点较少，返回
            // if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
            //     return;
            // 发布闭环匹配关键帧局部map
            if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
        }

        // ICP Settings
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // scan-to-map，调用icp匹配
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        // 未收敛，或者匹配不够好
        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;

        std::cout << "icp  success  " << std::endl;

        // 发布当前关键帧经过闭环优化后的位姿变换之后的特征点云
        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
            publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
        }

        // 闭环优化得到的当前关键帧与闭环关键帧之间的位姿变换
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();

        // 闭环优化前当前帧位姿
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // 闭环优化后当前帧位姿
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;
        pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw); //  获取上一帧 相对 当前帧的 位姿
        gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
        // 闭环匹配帧的位姿
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore(); //  loop_clousre  noise from icp
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        gtsam::noiseModel::Diagonal::shared_ptr constraintNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);
        std::cout << "loopNoiseQueue   =   " << noiseScore << std::endl;

        // 添加闭环因子需要的数据
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();

        loopIndexContainer[loopKeyCur] = loopKeyPre; //   使用hash map 存储回环对
    }

    //回环检测线程
    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false)
        {
            std::cout << "loopClosureEnableFlag   ==  false " << endl;
            return;
        }

        ros::Rate rate(loopClosureFrequency); //   回环频率
        while (ros::ok() && startFlag)
        {
            rate.sleep();
            performLoopClosure();   //  回环检测
            visualizeLoopClosure(); // rviz展示闭环边
        }
    }

    bool CreateFile(std::ofstream &ofs, std::string file_path)
    {
        ofs.open(file_path, std::ios::out); //  使用std::ios::out 可实现覆盖
        if (!ofs)
        {
            std::cout << "open csv file error " << std::endl;
            return false;
        }
        return true;
    }

    /* write2txt   format  KITTI*/
    void WriteText(std::ofstream &ofs, pose data)
    {
        ofs << std::fixed << data.R(0, 0) << " " << data.R(0, 1) << " " << data.R(0, 2) << " " << data.t[0] << " "
            << data.R(1, 0) << " " << data.R(1, 1) << " " << data.R(1, 2) << " " << data.t[1] << " "
            << data.R(2, 0) << " " << data.R(2, 1) << " " << data.R(2, 2) << " " << data.t[2] << std::endl;
    }

    //  eulerAngle 2 Quaterniond
    Eigen::Quaterniond EulerToQuat(float roll_, float pitch_, float yaw_)
    {
        Eigen::Quaterniond q; //   四元数 q 和 -q 是相等的
        Eigen::AngleAxisd roll(double(roll_), Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitch(double(pitch_), Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yaw(double(yaw_), Eigen::Vector3d::UnitZ());
        q = yaw * pitch * roll;
        q.normalize();
        return q;
    }
};

int main(int argc, char **argv)
{
    // allocateMemory();
    for (int i = 0; i < 6; ++i)
    {
        transformTobeMapped[i] = 0;
    }

    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    mapOptimization mapOpt;

    // ISAM2参数
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new gtsam::ISAM2(parameters);

    path.header.stamp = ros::Time::now();
    path.header.frame_id = "camera_init";

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    memset(point_selected_surf, true, sizeof(point_selected_surf)); //重复？
    memset(res_last, -1000.0f, sizeof(res_last));

    double epsi[23] = {0.001};
    fill(epsi, epsi + 23, 0.001);
    ///初始化，其中h_share_model定义了·平面搜索和残差计算
    // kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(), "w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"), ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~" << ROOT_DIR << " file opened" << endl;
    else
        cout << "~~~~" << ROOT_DIR << " doesn't exist" << endl;

    // /*** ROS subscribe initialization ***/
    // ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    // ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    // ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);           //  world系下稠密点云
    // ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000); //  body系下稠密点云
    // ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);           //  no used
    // ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);                   //  no used
    // ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    // ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 1e00000);

    // ros::Publisher pubPathUpdate = nh.advertise<nav_msgs::Path>("fast_lio_sam/path_update", 100000); //  isam更新后的path
    // pubGnssPath = nh.advertise<nav_msgs::Path>("/gnss_path", 100000);
    // pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("fast_lio_sam/mapping/keyframe_submap", 1);      // 发布局部关键帧map的特征点云
    // pubOptimizedGlobalMap = nh.advertise<sensor_msgs::PointCloud2>("fast_lio_sam/mapping/map_global_optimized", 1); // 发布局部关键帧map的特征点云

    // // loop clousre
    // // 发布闭环匹配关键帧局部map
    // pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("fast_lio_sam/mapping/icp_loop_closure_history_cloud", 1);
    // // 发布当前关键帧经过闭环优化后的位姿变换之后的特征点云
    // pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("fast_lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);
    // // 发布闭环边，rviz中表现为闭环帧之间的连线
    // pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/fast_lio_sam/mapping/loop_closure_constraints", 1);

    // // gnss
    // ros::Subscriber sub_gnss = nh.subscribe(gnss_topic, 200000, gnss_cbk);

    // // saveMap  发布地图保存服务
    // srvSaveMap = nh.advertiseService("/save_map", &saveMapService);

    // // savePose  发布轨迹保存服务
    // srvSavePose = nh.advertiseService("/save_pose", &savePoseService);

    // // 回环检测线程
    // std::thread loopthread(&loopClosureThread);

    //------------------------------------------------------------------------------------------------------
    // signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit)
            break;
        ros::spinOnce();

        // /// 在Measure内，储存当前lidar数据及lidar扫描时间内对应的imu数据序列
        // if (sync_packages(Measures))
        // {
        //     //第一帧lidar数据
        //     if (flg_first_scan)
        //     {
        //         first_lidar_time = Measures.lidar_beg_time; //记录第一帧绝对时间
        //         p_imu->first_lidar_time = first_lidar_time; //记录第一帧绝对时间
        //         flg_first_scan = false;
        //         continue;
        //     }

        //     double t0, t1, t2, t3, t4, t5, match_start, solve_start, svd_time;

        //     match_time = 0;
        //     kdtree_search_time = 0.0;
        //     solve_time = 0;
        //     solve_const_H_time = 0;
        //     svd_time = 0;
        //     t0 = omp_get_wtime();

        //     //根据imu数据序列和lidar数据，向前传播纠正点云的畸变, 此前已经完成间隔采样或特征提取
        //     // feats_undistort 为畸变纠正之后的点云,lidar系
        //     p_imu->Process(Measures, kf, feats_undistort);
        //     state_point = kf.get_x();                                               // 前向传播后body的状态预测值
        //     pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I; // global系 lidar位置

        //     if (feats_undistort->empty() || (feats_undistort == NULL))
        //     {
        //         ROS_WARN("No point, skip this scan!\n");
        //         continue;
        //     }

        //     // 检查当前lidar数据时间，与最早lidar数据时间是否足够
        //     flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;

        //     /*** Segment the map in lidar FOV ***/
        //     lasermap_fov_segment(); // 根据lidar在W系下的位置，重新确定局部地图的包围盒角点，移除远端的点

        //     /*** downsample the feature points in a scan ***/
        //     downSizeFilterSurf.setInputCloud(feats_undistort);
        //     downSizeFilterSurf.filter(*feats_down_body);
        //     t1 = omp_get_wtime();
        //     feats_down_size = feats_down_body->points.size(); //当前帧降采样后点数

        //     /*** initialize the map kdtree ***/
        //     if (ikdtree.Root_Node == nullptr)
        //     {
        //         if (feats_down_size > 5)
        //         {
        //             ikdtree.set_downsample_param(filter_size_map_min);
        //             feats_down_world->resize(feats_down_size);
        //             for (int i = 0; i < feats_down_size; i++)
        //             {
        //                 pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i])); // point转到world系下
        //             }
        //             // world系下对当前帧降采样后的点云，初始化lkd-tree
        //             ikdtree.Build(feats_down_world->points);
        //         }
        //         continue;
        //     }
        //     int featsFromMapNum = ikdtree.validnum();
        //     kdtree_size_st = ikdtree.size();

        //     // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

        //     /*** ICP and iterated Kalman filter update ***/
        //     if (feats_down_size < 5)
        //     {
        //         ROS_WARN("No point, skip this scan!\n");
        //         continue;
        //     }

        //     normvec->resize(feats_down_size);
        //     feats_down_world->resize(feats_down_size);

        //     // lidar --> imu
        //     V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
        //     fout_pre << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose() << " " << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose()
        //              << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << endl;

        //     if (visulize_IkdtreeMap) // If you need to see map point, change to "if(1)"
        //     {
        //         PointVector().swap(ikdtree.PCL_Storage);
        //         ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
        //         featsFromMap->clear();
        //         featsFromMap->points = ikdtree.PCL_Storage;
        //         publish_map(pubLaserCloudMap);
        //     }

        //     pointSearchInd_surf.resize(feats_down_size);
        //     Nearest_Points.resize(feats_down_size);
        //     int rematch_num = 0;
        //     bool nearest_search_en = true; //

        //     t2 = omp_get_wtime();

        //     /*** iterated state estimation ***/
        //     double t_update_start = omp_get_wtime();
        //     double solve_H_time = 0;
        //     kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time); //预测、更新
        //     state_point = kf.get_x();
        //     euler_cur = SO3ToEuler(state_point.rot);
        //     pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I; // world系下lidar坐标
        //     geoQuat.x = state_point.rot.coeffs()[0];                                // world系下当前imu的姿态四元数
        //     geoQuat.y = state_point.rot.coeffs()[1];
        //     geoQuat.z = state_point.rot.coeffs()[2];
        //     geoQuat.w = state_point.rot.coeffs()[3];

        //     double t_update_end = omp_get_wtime();

        // getCurPose(state_point); //   更新transformTobeMapped
        // /*back end*/
        // // 1.计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
        // // 2.添加激光里程计因子、GPS因子、闭环因子
        // // 3.执行因子图优化
        // // 4.得到当前帧优化后的位姿，位姿协方差
        // // 5.添加cloudKeyPoses3D，cloudKeyPoses6D，更新transformTobeMapped，添加当前关键帧的角点、平面点集合
        // saveKeyFramesAndFactor();
        // // 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹， 重构ikdtree
        // correctPoses();
        // /******* Publish odometry *******/
        // publish_odometry(pubOdomAftMapped);
        // /*** add the feature points to map kdtree ***/
        // t3 = omp_get_wtime();
        // map_incremental();
        // t5 = omp_get_wtime();
        // /******* Publish points *******/
        // if (path_en)
        // {
        //     publish_path(pubPath);
        //     publish_gnss_path(pubGnssPath);     //   发布gnss轨迹
        //     publish_path_update(pubPathUpdate); //   发布经过isam2优化后的路径
        //     static int jjj = 0;
        //     jjj++;
        //     if (jjj % 100 == 0)
        //     {
        //         // publishGlobalMap(); //  发布局部点云特征地图
        //     }
        // }
        // if (scan_pub_en || pcd_save_en)
        //     publish_frame_world(pubLaserCloudFull); //   发布world系下的点云
        // if (scan_pub_en && scan_body_pub_en)
        //     publish_frame_body(pubLaserCloudFull_body); //  发布imu系下的点云

        // if(savePCD)  saveMap();

        // publish_effect_world(pubLaserCloudEffect);
        // publish_map(pubLaserCloudMap);

        /*** Debug variables ***/
        // if (runtime_pos_log)
        // {
        //     frame_num++;
        //     kdtree_size_end = ikdtree.size();
        //     aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
        //     aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t_update_end - t_update_start) / frame_num;
        //     aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time) / frame_num;
        //     aver_time_incre = aver_time_incre * (frame_num - 1) / frame_num + (kdtree_incremental_time) / frame_num;
        //     aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time + solve_H_time) / frame_num;
        //     aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_time / frame_num;
        //     T1[time_log_counter] = Measures.lidar_beg_time;
        //     s_plot[time_log_counter] = t5 - t0;
        //     s_plot2[time_log_counter] = feats_undistort->points.size();
        //     s_plot3[time_log_counter] = kdtree_incremental_time;
        //     s_plot4[time_log_counter] = kdtree_search_time;
        //     s_plot5[time_log_counter] = kdtree_delete_counter;
        //     s_plot6[time_log_counter] = kdtree_delete_time;
        //     s_plot7[time_log_counter] = kdtree_size_st;
        //     s_plot8[time_log_counter] = kdtree_size_end;
        //     s_plot9[time_log_counter] = aver_time_consu;
        //     s_plot10[time_log_counter] = add_point_size;
        //     time_log_counter++;
        //     printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n", t1 - t0, aver_time_match, aver_time_solve, t3 - t1, t5 - t3, aver_time_consu, aver_time_icp, aver_time_const_H_time);
        //     ext_euler = SO3ToEuler(state_point.offset_R_L_I);
        //     fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose() << " " << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose()
        //              << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << " " << feats_undistort->points.size() << endl;
        //     dump_lio_state_to_log(fp);
        // }
        // }

        status = ros::ok();
        rate.sleep();
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    // if (pcl_wait_save->size() > 0 && pcd_save_en)
    // {
    //     string file_name = string("scans.pcd");
    //     string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
    //     pcl::PCDWriter pcd_writer;
    //     cout << "current scan saved to /PCD/" << file_name << endl;
    //     pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    // }

    // fout_out.close();
    // fout_pre.close();

    // if (runtime_pos_log)
    // {
    //     vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;
    //     FILE *fp2;
    //     string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
    //     fp2 = fopen(log_dir.c_str(), "w");
    //     fprintf(fp2, "time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
    //     for (int i = 0; i < time_log_counter; i++)
    //     {
    //         fprintf(fp2, "%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n", T1[i], s_plot[i], int(s_plot2[i]), s_plot3[i], s_plot4[i], int(s_plot5[i]), s_plot6[i], int(s_plot7[i]), int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
    //         t.push_back(T1[i]);
    //         s_vec.push_back(s_plot9[i]);
    //         s_vec2.push_back(s_plot3[i] + s_plot6[i]);
    //         s_vec3.push_back(s_plot4[i]);
    //         s_vec5.push_back(s_plot[i]);
    //     }
    //     fclose(fp2);
    // }

    // startFlag = false;
    // loopthread.join(); //  分离线程

    return 0;
}
