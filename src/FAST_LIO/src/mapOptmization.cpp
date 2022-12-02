// TODO 本code创新点：用关键帧队列存储点云，避免了在整个地图里搜索点云
#include "Scancontext.h"
#include "utility.h"
#include "fast_lio/cloud_info.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include "sophus/so3.hpp"

using namespace gtsam;

using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

/*
 * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
 */
// struct PointXYZIRPYT
// {
//     PCL_ADD_POINT4D
//     PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
//     float roll;
//     float pitch;
//     float yaw;
//     double time;
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
// } EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

// POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
//                                    (float, x, x) (float, y, y)
//                                    (float, z, z) (float, intensity, intensity)
//                                    (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
//                                    (double, time, time))

#if 0
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))
#endif
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY; // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double timestamp;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, timestamp, timestamp))

typedef PointXYZIRPYT PointTypePose;

class mapOptimization : public ParamServer
{

public:
    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;   // 初始位姿估计
    Values optimizedEstimate; // 优化后的位姿
    ISAM2 *isam;              // 优化器推理算法
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance; // 优化后的上一关键帧位姿协方差

    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubLaserOdometryGlobal;
    ros::Publisher pubLaserOdometryIncremental;
    ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRecentKeyFrame;
    ros::Publisher pubCloudRegisteredRaw;
    ros::Publisher pubLoopConstraintEdge;

    ros::Subscriber subGPS;               // 输入gps
    ros::Subscriber subLaserCloudFullRes; // 输入点云
    ros::Subscriber subLaserOdometry;     //输入里程计

    std::queue<nav_msgs::Odometry::ConstPtr> odometryQueue;
    std::queue<sensor_msgs::PointCloud2ConstPtr> fullResQueue;
    std::deque<nav_msgs::Odometry> gpsQueue; // GPS队列
    fast_lio::cloud_info cloudInfo;           // 用来存储topic接收的点云

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames; // 当某一帧被选为关键帧之后，他的scan经过降采样作为cornerCloudKeyFrames
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;

    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;     // gtsam优化后的地图关键帧位置(x，y，z)
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D; //优化后的地图关键帧位置（x，y，z，R, P,Y，time）
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;   // topic接收到的角点点云,当前点云 corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;     // topic接收到的平面点云surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;   // downsampled surf featuer set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudInRaw;        // 当前真在被处理的原始点云

    pcl::PointCloud<PointType>::Ptr laserCloudOri; // 经过筛选的可以用于匹配的点
    pcl::PointCloud<PointType>::Ptr coeffSel;      // 优化方向的向量的系数

    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer; //地图容器 ，first是索引，second是角点地图和平面地图
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;                                       // 从地图中提取的除当前帧外的当前帧的周围点云
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;                                         // 从地图中提取的除当前帧外的当前帧的周围点云
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;                                     // 上面的降采样
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;                                       // 上面的降采样

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization

    ros::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    float transformTobeMapped[6]; //当前的里程计 （初始数据来自imu积分里程计，然后经过scan2map优化），  RPYxyz初始化为0,0,0,0,0,0

    std::mutex mtx;
    std::mutex mtxLoopInfo;
    std::mutex mBuf;

    bool isDegenerate = false;
    Eigen::Matrix<float, 6, 6> matP;

    int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;
    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    map<int, int> loopIndexContainer; // 闭环容器，第一个保存新的闭环帧，第二个保存对应的旧闭环帧from new to old
    // 用于gtsam优化的闭环关系队列
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    deque<std_msgs::Float64MultiArray> loopInfoVec; // 闭环时间序列，每一个序列代表一次闭合，下标0存储当前闭环时间，1存储对应的之前的闭环时间

    nav_msgs::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;       // transformTobeMapped的矩阵形式
    Eigen::Affine3f incrementalOdometryAffineFront; // save current transformation before any processing 相当于slam里程计值
    Eigen::Affine3f incrementalOdometryAffineBack;  //  经过scan2map优化后的值，又经过了imu差值后的值

    // SC loop detector
    SCManager scManager;
    int SCclosestHistoryFrameID; //  基于scan-context搜索的闭环关键帧
    float yawDiffRad;

    bool gps_initailized;
    bool pose_initailized;
    bool Calib_flag;
    struct CalibrationExRotation
    {
        Eigen::Matrix3d Rwl;
        double timestamp;
    };
    queue<CalibrationExRotation> lidar_cali;
    queue<CalibrationExRotation> gps_cali;
    // Eigen::Matrix3d ric = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d ric;

    mapOptimization() : gps_initailized(false), pose_initailized(false), Calib_flag(false)
    {

        ric = extRot;

        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        pubKeyPoses = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/trajectory", 1); // 发布关键帧点云
        pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_global", 1);
        pubLaserOdometryGlobal = nh.advertise<nav_msgs::Odometry>("lio_sam/mapping/odometry", 1);                  // 全局里程计，有优化
        pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry>("lio_sam/mapping/odometry_incremental", 1); // 发布增量里程计，不受闭环优化等的影响，只用来算增量，是为了防止在算增量过程中发生优化，导致增量出错
        pubPath = nh.advertise<nav_msgs::Path>("lio_sam/mapping/path", 1);                                         // 发布路径

        subGPS = nh.subscribe<nav_msgs::Odometry>(gpsTopic, 200, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_registered_local", 100, &mapOptimization::laserCloudFullResHandler, this, ros::TransportHints().tcpNoDelay());
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("fast_lio/odometry", 100, &mapOptimization::laserOdometryHandler, this, ros::TransportHints().tcpNoDelay());

        pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);     // 历史闭环帧附件的点云
        pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);       // 闭环矫正后的当前帧点云
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/lio_sam/mapping/loop_closure_constraints", 1); // 可视化闭环关系

        pubRecentKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_local", 1);
        pubRecentKeyFrame = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered", 1); // 当前帧的降采样点云
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered_raw", 1);

        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

        allocateMemory();
        if (mintialMethod == human)
            gps_initailized = true;
    }

    // 预先分配内存
    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());   // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());     // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>());   // downsampled surf featuer set from odoOptimization
        laserCloudInRaw.reset(new pcl::PointCloud<PointType>());
        laserCloudInRaw->clear();

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i)
        {
            transformTobeMapped[i] = 0;
        }

        matP.setZero();
    }

    // 标定旋转外参
    bool Calibration_ExRotation(queue<CalibrationExRotation> &lidar_cali, // 上一帧和当前帧对应的特征点组
                                queue<CalibrationExRotation> &gps_cali,   // 两帧间预积分算出来的delta旋转四元数q
                                Eigen::Matrix3d &calib_ric_result)        // 标定出来的旋转矩阵
    {

        if (lidar_cali.empty() || gps_cali.empty())
        {
            cout << "gps_cali为空" << endl;
            return false;
        }
        Eigen::Matrix3d ric(calib_ric_result); // ric初始值
        queue<Eigen::Matrix3d> lidar_qu;
        queue<Eigen::Matrix3d> gps_qu;

        while (gps_cali.front().timestamp > lidar_cali.front().timestamp - 0.01)
        {
            lidar_cali.pop();
        }

        while (!lidar_cali.empty())
        {
            double lidar_time = lidar_cali.front().timestamp;
            while (gps_cali.front().timestamp < lidar_time - 0.01)
            {
                gps_cali.pop();
            }
            if (gps_cali.front().timestamp < lidar_time + 0.01)
            {
                lidar_qu.push(lidar_cali.front().Rwl);
                lidar_cali.pop();
                gps_qu.push(gps_cali.front().Rwl);
                gps_cali.pop();
            }
            else
            {
                lidar_cali.pop();
                gps_cali.pop();
            }
        }

        cout << lidar_qu.size() << "," << gps_qu.size() << endl;
        if (lidar_qu.size() != gps_qu.size())
        {

            cout << "长度不相等" << endl;
            return false;
        }

        queue<Eigen::Matrix3d> delta_lidar;
        queue<Eigen::Matrix3d> delta_gps;
        queue<Eigen::Matrix3d> Rc_g;
        Eigen::Matrix3d last_lidar = lidar_qu.front();
        lidar_qu.pop();
        Eigen::Matrix3d last_gps = gps_qu.front();
        gps_qu.pop();
        // Ri,i+1 = Rwi.transpose() *  Rwi+1;
        while (!lidar_qu.empty())
        {
            delta_lidar.push(last_lidar.transpose() * lidar_qu.front());
            last_lidar = lidar_qu.front();
            lidar_qu.pop();

            delta_gps.push(last_gps.transpose() * gps_qu.front());
            last_gps = gps_qu.front();
            gps_qu.pop();

            Rc_g.push(ric.inverse() * delta_gps.front() * ric);
        }
        // Sophus::SO3d SO3_R(delta_lidar.front());
        // Vector3d so3 = SO3_R.log();
        // if(so3.norm()<0.1){
        //       cout << "旋转太小"<<endl;
        //     return false;

        // }

        Eigen::MatrixXd A(delta_lidar.size() * 4, 4);
        A.setZero();
        int sum_ok = 0;
        while (!delta_lidar.empty())
        {
            Eigen::Quaterniond r1(delta_lidar.front()); // 特征匹配得到的两帧间的旋转
            Eigen::Quaterniond r2(Rc_g.front());        // 预积分得到的两帧间的旋转

            // https://www.iiiff.com/article/389681 可以理解为两个角之间的角度差
            double angular_distance = 180 / M_PI * r1.angularDistance(r2);
            cout << "角度差" << angular_distance << endl;

            // 一个简单的核函数
            double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
            ++sum_ok;
            ROS_DEBUG(
                "%d %f", sum_ok, angular_distance);
            Eigen::Matrix4d L, R;
            // 四元数的左乘矩阵
            double w = Quaterniond(delta_lidar.front()).w();
            Eigen::Vector3d q = Quaterniond(delta_lidar.front()).vec();
            L.block<3, 3>(0, 0) = w * Eigen::Matrix3d::Identity() + gtsam::skewSymmetric(q);
            L.block<3, 1>(0, 3) = q;
            L.block<1, 3>(3, 0) = -q.transpose();
            L(3, 3) = w;
            // 四元数的右乘矩阵
            Eigen::Quaterniond R_ij(delta_gps.front());
            w = R_ij.w();
            q = R_ij.vec();
            R.block<3, 3>(0, 0) = w * Eigen::Matrix3d::Identity() - gtsam::skewSymmetric(q);
            R.block<3, 1>(0, 3) = q;
            R.block<1, 3>(3, 0) = -q.transpose();
            R(3, 3) = w;

            A.block<4, 4>((sum_ok - 1) * 4, 0) = huber * (L - R); // 作用在残差上面
            Rc_g.pop();
            delta_lidar.pop();
            delta_gps.pop();
        }

        Eigen::JacobiSVD<MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix<double, 4, 1> xxxx = svd.matrixV().col(3);
        Eigen::Quaterniond estimated_R(xxxx);
        ric = estimated_R.toRotationMatrix().inverse();
        // cout << svd.singularValues().transpose() << endl;
        // cout << ric << endl;
        Eigen::Vector3d ric_cov;
        ric_cov = svd.singularValues().tail<3>();
        // 倒数第二个奇异值，因为旋转是3个自由度，因此检查一下第三小的奇异值是否足够大，通常需要足够的运动激励才能保证得到没有奇异的解
        if (ric_cov(1) > 0.25)
        {
            calib_ric_result = ric;
            return true;
        }
        else
        {
            cout << "自由度不够:" << ric_cov(1) << endl;
            cout << ric.transpose() << endl;
            Eigen::Vector3d eulerAngle = ric.transpose().eulerAngles(0, 1, 2);
            cout << "四元数" << endl;
            cout << eulerAngle << endl;
            return false;
        }
    }

    void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &_laserOdometry)
    {
        // mBuf.lock();
        // odometryQueue.push(_laserOdometry);
        // mBuf.unlock();
    } // laserOdometryHandler

    void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &_laserCloudFullRes)
    {
        // mBuf.lock();
        // fullResQueue.push(_laserCloudFullRes);
        // mBuf.unlock();
    } // laserCloudFullResHandler

    // 添加GPS里程计数据到队列
    int gps_count = 0;
    std::chrono::steady_clock::time_point now;
    std::chrono::steady_clock::time_point last;
    double last_E, last_N, last_U;
    void gpsHandler(const nav_msgs::Odometry::ConstPtr &gpsMsg)
    {
        // // 每隔一秒接收一次数据
        // ++gps_count;
        // gps_count%=200;
        // if(gps_count!=0){
        //     return;
        // }
        // now = std::chrono::steady_clock::now();

        // double t_track = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(now - last).count();
        // last = now;
        // cout<<"一个周期的时间是"<<t_track<<endl;
        // cout<<"gps_x"<<gpsMsg->pose.pose.position.x<<endl;
        // cout<<"gps_y"<<gpsMsg->pose.pose.position.y<<endl;

        if (mintialMethod == gps)
        {
            if (!gps_initailized && (gpsMsg->pose.pose.position.x != 0 || gpsMsg->pose.pose.position.y != 0) && (gpsMsg->pose.covariance[0] < 0.003 && gpsMsg->pose.covariance[7] < 0.003))
            {

                Eigen::Vector3d Pwl;
                Eigen::Vector3d Pwi(gpsMsg->pose.pose.position.x, gpsMsg->pose.pose.position.y, gpsMsg->pose.pose.position.z);
                Eigen::Quaterniond Qwi(gpsMsg->pose.pose.orientation.w, gpsMsg->pose.pose.orientation.x, gpsMsg->pose.pose.orientation.y, gpsMsg->pose.pose.orientation.z);
                Pwl = Pwi + Qwi.matrix() * Pil;
                cout << "GPS initailizes" << endl;
                initialPose.at(0) = Pwl.x();
                initialPose.at(1) = Pwl.y();

                gps_initailized = true;
                last_E = initialPose.at(0);
                last_N = initialPose.at(1);
            }
        }

        if (optimization_with_GPS)
        {
            if (last_E != gpsMsg->pose.pose.position.x || last_N != gpsMsg->pose.pose.position.y)
            {
                gpsQueue.push_back(*gpsMsg);
            }
        }

        // 外参标定
        if (Calib_flag)
        {
            CalibrationExRotation tmp;
            Eigen::Quaterniond Qwi(gpsMsg->pose.pose.orientation.w, gpsMsg->pose.pose.orientation.x, gpsMsg->pose.pose.orientation.y, gpsMsg->pose.pose.orientation.z);
            // Eigen::Vector3d Pwi(gpsMsg->pose.pose.position.x,gpsMsg->pose.pose.position.y,gpsMsg->pose.pose.position.z);
            // Pwl= Pwi+ Qwi.matrix()*Pil;
            tmp.Rwl = Qwi.toRotationMatrix();
            tmp.timestamp = gpsMsg->header.stamp.toSec();
            gps_cali.push(tmp);
        }

        // cout<<"收到GPS"<<endl;
    }

    // scan坐标系下的点->地图坐标系下
    void pointAssociateToMap(PointType const *const pi, PointType *const po)
    {
        po->x = transPointAssociateToMap(0, 0) * pi->x + transPointAssociateToMap(0, 1) * pi->y + transPointAssociateToMap(0, 2) * pi->z + transPointAssociateToMap(0, 3);
        po->y = transPointAssociateToMap(1, 0) * pi->x + transPointAssociateToMap(1, 1) * pi->y + transPointAssociateToMap(1, 2) * pi->z + transPointAssociateToMap(1, 3);
        po->z = transPointAssociateToMap(2, 0) * pi->x + transPointAssociateToMap(2, 1) * pi->y + transPointAssociateToMap(2, 2) * pi->z + transPointAssociateToMap(2, 3);
        po->intensity = pi->intensity;
    }

    void pointTransForm(PointType const *const pi, PointType *const po, Eigen::Affine3f TransForm)
    {
        po->x = TransForm(0, 0) * pi->x + TransForm(0, 1) * pi->y + TransForm(0, 2) * pi->z + TransForm(0, 3);
        po->y = TransForm(1, 0) * pi->x + TransForm(1, 1) * pi->y + TransForm(1, 2) * pi->z + TransForm(1, 3);
        po->z = TransForm(2, 0) * pi->x + TransForm(2, 1) * pi->y + TransForm(2, 2) * pi->z + TransForm(2, 3);
        po->intensity = pi->intensity;
    }

    // 对输入点云进行位姿变换
    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);

// https://blog.csdn.net/bigFatCat_Tom/article/details/98493040
// 使用多线程并行加速
#pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            pointFrom = &cloudIn->points[i];
            cloudOut->points[i].x = transCur(0, 0) * pointFrom->x + transCur(0, 1) * pointFrom->y + transCur(0, 2) * pointFrom->z + transCur(0, 3);
            cloudOut->points[i].y = transCur(1, 0) * pointFrom->x + transCur(1, 1) * pointFrom->y + transCur(1, 2) * pointFrom->z + transCur(1, 3);
            cloudOut->points[i].z = transCur(2, 0) * pointFrom->x + transCur(2, 1) * pointFrom->y + transCur(2, 2) * pointFrom->z + transCur(2, 3);
            cloudOut->points[i].intensity = pointFrom->intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                            gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                            gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    {
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    // 从xyzRPY的单独数据变成仿射变换矩阵
    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

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

    // 发布全局地图和保存地图
    void visualizeGlobalMapThread()
    {
        ros::Rate rate(0.2);
        while (ros::ok())
        {
            rate.sleep();
            publishGlobalMap();
        }

        if (Calib_flag && lidar_cali.size() > 100)
        {
            cout << "开始标定" << endl;
            if (Calibration_ExRotation(lidar_cali, gps_cali, ric))
            {
                cout << "外参标定成功" << endl;
                Calib_flag = false;
            }
            else
            {
                cout << "外参标定失败" << endl;
            }
        }
        if (!Calib_flag)
        {
            cout << "外参标定" << endl;
            cout << ric.transpose() << endl;
            Eigen::Vector3d eulerAngle = ric.transpose().eulerAngles(0, 1, 2);
            cout << "欧拉角" << endl;
            cout << eulerAngle << endl;
        }
        if (savePCD == false)
            return;

        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;
        // create directory and remove old files;
        savePCDDirectory = std::getenv("HOME") + savePCDDirectory;
        system((std::string("exec rm -r ") + savePCDDirectory).c_str());
        system((std::string("mkdir ") + savePCDDirectory).c_str());
        // save key frame transformations
        pcl::io::savePCDFileASCII(savePCDDirectory + "trajectory.pcd", *cloudKeyPoses3D);
        pcl::io::savePCDFileASCII(savePCDDirectory + "transformations.pcd", *cloudKeyPoses6D);
        // extract global point cloud map
        pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());

        // TODO 不是有个地图容器嘛，怎么还搞便利呀。不过这边建图都结束了倒也不在乎时间
        // 但是这样不会有很多重复的点嘛
        for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++)
        {
            *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
            *globalSurfCloud += *transformPointCloud(surfCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
            cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
        }
        // down-sample and save corner cloud
        // downSizeFilterCorner.setInputCloud(globalCornerCloud);
        // downSizeFilterCorner.filter(*globalCornerCloudDS);
        // pcl::io::savePCDFileASCII(savePCDDirectory + "cloudCorner.pcd", *globalCornerCloudDS);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudCorner.pcd", *globalCornerCloud);
        // down-sample and save surf cloud
        // downSizeFilterSurf.setInputCloud(globalSurfCloud);
        // downSizeFilterSurf.filter(*globalSurfCloudDS);
        // pcl::io::savePCDFileASCII(savePCDDirectory + "cloudSurf.pcd", *globalSurfCloudDS);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudSurf.pcd", *globalSurfCloud);
        // down-sample and save global point cloud map
        *globalMapCloud += *globalCornerCloud;
        *globalMapCloud += *globalSurfCloud;
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudGlobal.pcd", *globalMapCloud);
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed" << endl;
    }

    void publishGlobalMap()
    {
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());

        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;                                                                                            // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i)
        {
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;                                                                                   // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
    }

    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false)
            return;

        ros::Rate rate(loopClosureFrequency);
        while (ros::ok())
        {
            rate.sleep();
            performLoopClosure();
            visualizeLoopClosure();
        }
    }

    // 暂时还没使用,接收人工闭环信息
    void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr &loopMsg)
    {
        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopMsg->data.size() != 2)
            return;

        loopInfoVec.push_back(*loopMsg);

        while (loopInfoVec.size() > 5)
            loopInfoVec.pop_front();
    }

    // 闭环检测当前帧和历史闭环关键帧
    // 找出对应的点云
    // 发布历史闭环关键帧点云
    // icp匹配
    // 将当前帧点云转到icp匹配位置，并发布
    // 存储闭环关系用于gtsam优化
    void performLoopClosure()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // find keys
        int loopKeyCur;   // 当前帧id
        int loopKeyPre;   // 闭环帧id
        int SCloopKeyCur; // 当前帧id
        int SCloopKeyPre; // 闭环帧id
        // if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false){
        //     if (detectLoopClosureScanContext(&loopKeyCur, &loopKeyPre) == false){
        //         if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false){
        //             return;
        //         // }
        //     }
        // }
        // if (detectLoopClosureScanContext(&SCloopKeyCur, &SCloopKeyPre) == false && detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false){
        //     return;
        // }

        bool isValidRSloopFactor = false;
        bool isValidSCloopFactor = false;
        isValidRSloopFactor = detectLoopClosureDistance(&loopKeyCur, &loopKeyPre); // 开启rs
        // isValidSCloopFactor = detectLoopClosureScanContext(&SCloopKeyCur, &SCloopKeyPre); //开启sc
        // RS loop closure
        if (isValidRSloopFactor == false && isValidSCloopFactor == false)
        {
            return;
        }
        // RS loop closure
        if (isValidRSloopFactor)
        {
            // extract cloud
            pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
            {
                loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
                loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
                if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                    return;
                if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                    publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
            }

            // ICP Settings
            static pcl::IterativeClosestPoint<PointType, PointType> icp;
            icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius * 2);
            icp.setMaximumIterations(100);
            icp.setTransformationEpsilon(1e-6);
            icp.setEuclideanFitnessEpsilon(1e-6);
            icp.setRANSACIterations(0);

            // Align clouds
            icp.setInputSource(cureKeyframeCloud);
            icp.setInputTarget(prevKeyframeCloud);
            pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
            icp.align(*unused_result);

            if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            {
                std::cout << "ICP failed" << std::endl;
                return;
            }
            // publish corrected cloud
            if (pubIcpKeyFrames.getNumSubscribers() != 0)
            {
                pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
                pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
                publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
            }

            // Get pose transformation
            float x, y, z, roll, pitch, yaw;
            Eigen::Affine3f correctionLidarFrame;
            correctionLidarFrame = icp.getFinalTransformation();
            // transform from world origin to wrong pose
            Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
            // transform from world origin to corrected pose
            Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong; // pre-multiplying -> successive rotation about a fixed frame
            pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
            gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
            gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
            gtsam::Vector Vector6(6);
            float noiseScore = icp.getFitnessScore();
            Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
            noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

            // Add pose constraint
            mtx.lock();
            loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
            loopPoseQueue.push_back(poseFrom.between(poseTo));
            loopNoiseQueue.push_back(constraintNoise);
            mtx.unlock();
            std::cout << "RS 闭环成功" << std::endl;
            // add loop constriant
            loopIndexContainer[loopKeyCur] = loopKeyPre;
        }
        // SC loop closure
        if (isValidSCloopFactor)
        {
            // extract cloud
            pcl::PointCloud<PointType>::Ptr SCCurKeyFrameCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr SCCurKeyFrameCloudDS(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr SCprevKeyframeCloud(new pcl::PointCloud<PointType>());

            {
                *SCCurKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[SCloopKeyCur], &copy_cloudKeyPoses6D->points[SCloopKeyPre]);
                *SCCurKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[SCloopKeyCur], &copy_cloudKeyPoses6D->points[SCloopKeyPre]);
                downSizeFilterICP.setInputCloud(SCCurKeyFrameCloud);
                downSizeFilterICP.filter(*SCCurKeyFrameCloudDS);
                loopFindNearKeyframes(SCprevKeyframeCloud, SCloopKeyPre, historyKeyframeSearchNum);
                if (SCCurKeyFrameCloudDS->size() < 300 || SCprevKeyframeCloud->size() < 1000)
                    return;
                if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                    publishCloud(&pubHistoryKeyFrames, SCprevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
            }

            // ICP Settings
            static pcl::IterativeClosestPoint<PointType, PointType> icp;
            icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius * 2);
            icp.setMaximumIterations(100);
            icp.setTransformationEpsilon(1e-6);
            icp.setEuclideanFitnessEpsilon(1e-6);
            icp.setRANSACIterations(0);

            // Align clouds
            icp.setInputSource(SCCurKeyFrameCloudDS);
            icp.setInputTarget(SCprevKeyframeCloud);
            pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
            icp.align(*unused_result);

            if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            {
                std::cout << "SC ICP failed" << std::endl;
                return;
            }
            // publish corrected cloud
            if (pubIcpKeyFrames.getNumSubscribers() != 0)
            {
                pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
                pcl::transformPointCloud(*SCCurKeyFrameCloudDS, *closed_cloud, icp.getFinalTransformation());
                publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
            }

            // Get pose transformation
            float x, y, z, roll, pitch, yaw;
            Eigen::Affine3f correctionLidarFrame;
            correctionLidarFrame = icp.getFinalTransformation();

            pcl::getTranslationAndEulerAngles(correctionLidarFrame, x, y, z, roll, pitch, yaw);
            gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
            gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));
            gtsam::Vector Vector6(6);
            float noiseScore = icp.getFitnessScore();
            Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
            noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

            // Add pose constraint
            mtx.lock();
            loopIndexQueue.push_back(make_pair(SCloopKeyCur, SCloopKeyPre));
            loopPoseQueue.push_back(poseFrom.between(poseTo));
            loopNoiseQueue.push_back(constraintNoise);
            mtx.unlock();
            std::cout << "SC 闭环成功" << std::endl;
            // add loop constriant
            loopIndexContainer[SCloopKeyCur] = SCloopKeyPre;
        }
    }

    // 距离闭环 // 成功返回true，失败返回false
    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // check loop constraint added before
        // 检测这个关键帧是否已经在闭环容器里了，防止重复闭环
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // find the closest history key frame
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

        // 在历史关键帧中找，时间相差超过阈值即可
        // TODO 这样岂不是原地不动也会闭环，应该还得关键帧序号相差超过一定阈值才行
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            if (abs(copy_cloudKeyPoses6D->points[id].timestamp - timeLaserInfoCur) > historyKeyframeSearchTimeDiff && loopKeyCur - id > 10)
            {
                loopKeyPre = id;
                break;
            }
        }

        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    // ScanContext闭环 // 成功返回true，失败返回false
    bool detectLoopClosureScanContext(int *latestID, int *closestID)
    {
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // check loop constraint added before
        // 检测这个关键帧是否已经在闭环容器里了，防止重复闭环
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // find the closest history key frame
        SCclosestHistoryFrameID = -1;
        auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff
        SCclosestHistoryFrameID = detectResult.first;
        yawDiffRad = detectResult.second; // 没有使用not use for v1 (because pcl icp withi initial somthing wrong...)
        // if all close, reject
        if (SCclosestHistoryFrameID == -1)
        {
            return false;
        }

        // 在历史关键帧中找，时间相差超过阈值即可
        {
            int id = SCclosestHistoryFrameID;
            if (abs(copy_cloudKeyPoses6D->points[id].timestamp - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;
            }
        }
        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    // loopInfoHandler 实现的人工闭环，暂时没有用
    // 成功返回true，失败返回false
    bool detectLoopClosureExternal(int *latestID, int *closestID)
    {
        // this function is not used yet, please ignore it
        int loopKeyCur = -1;
        int loopKeyPre = -1;

        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopInfoVec.empty())
            return false;

        double loopTimeCur = loopInfoVec.front().data[0];
        double loopTimePre = loopInfoVec.front().data[1];
        loopInfoVec.pop_front();
        // 时间太近
        if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
            return false;

        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2)
            return false;

        // latest key
        loopKeyCur = cloudSize - 1;
        // 找出离当前闭合时间最近的关键帧索引
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            if (copy_cloudKeyPoses6D->points[i].timestamp > loopTimeCur)
                loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        // previous key
        loopKeyPre = 0;
        // 找出离之前闭合时间最近的关键帧索引
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].timestamp < loopTimePre)
                loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        if (loopKeyCur == loopKeyPre)
            return false;

        auto it = loopIndexContainer.find(loopKeyCur);
        //存在返回false，说明这个人工闭环已经被检测到过了，也就不需要了
        if (it != loopIndexContainer.end())
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    /**
     * @details 找出索引关键帧key前后searchNum范围内关键帧对应的点云
     * @param nearKeyframes 关键帧对应的搜索范围内的点云（包含角点云和面点云经过降采样）
     * @param key 关键帧索引
     * @param searchNum 前后搜索数量 2*searchNum
     */
    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize)
                continue;
            // 转到世界坐标系下
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    // 可视化闭环关系，将闭环的两个帧连线
    void visualizeLoopClosure()
    {
        visualization_msgs::MarkerArray markerArray;
        // loop nodes
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
        // loop edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1;
        markerEdge.scale.y = 0.1;
        markerEdge.scale.z = 0.1;
        markerEdge.color.r = 0.9;
        markerEdge.color.g = 0.9;
        markerEdge.color.b = 0;
        markerEdge.color.a = 1;

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

    // 被注释了，没有使用
    // 提取最近添加的n个关键帧
    void extractForLoopClosure()
    {
        pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses - 1; i >= 0; --i)
        {
            if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
                cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(cloudToExtract);
    }

    // 提取最后一个关键帧（上一关键帧）点云的周围50米范围内的关键帧,降采样
    // 再加上最近10秒内的关键帧
    // 不包括最后一个关键帧本身
    void extractNearby()
    {
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // extract all the nearby key poses and downsample them
        // 提取最后一个关键帧点云的周围50米范围内的关键帧
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        // 降采样
        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

        // also extract some latest key frames in case the robot rotates in one position
        //同时提取一些最新的关键帧（最近10秒），以防机器人在一个位置旋转
        // TODO 最近10秒的帧不会和最近50米的帧有重合吗？？有点多此一举吧
        // 结合后面又判断了一次距离，猜想应该是50米内的降采样了，加上10秒内的不降采样
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses - 1; i >= 0; --i)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].timestamp < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(surroundingKeyPosesDS);
    }

    // 从地图中提取点云，如果地图容器中没有就加入进去
    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // fuse the map
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            // 距离过大去掉
            // 10秒内的？？ 毕竟已经检验过一次距离了
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;

            // intensity 该帧对应的关键帧的序号，
            int thisKeyInd = (int)cloudToExtract->points[i].intensity;
            // 判断是否已经在laserCloudMapContainer容器中
            // 是，则直接从地图容器中提取周围点云
            // 否，则提取的同时加入到地图容器中
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end())
            {
                // transformed cloud available
                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap += laserCloudMapContainer[thisKeyInd].second;
            }
            else
            {
                // transformed cloud not available
                pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
                *laserCloudCornerFromMap += laserCloudCornerTemp;
                *laserCloudSurfFromMap += laserCloudSurfTemp;
                laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
            }
        }

        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        // clear map cache if too large
        // 防止地图缓存太大了
        if (laserCloudMapContainer.size() > 1000)
            laserCloudMapContainer.clear();
    }

    //提取周围关键帧
    void extractSurroundingKeyFrames()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        // if (loopClosureEnableFlag == true)
        // {
        //     extractForLoopClosure();
        // } else {
        //     extractNearby();
        // }

        extractNearby();
    }

    // 降采样当前scan
    void downsampleCurrentScan()
    {
        // Downsample cloud from current scan
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
    }

    // 更新lidar->map 的变换矩阵
    void updatePointAssociateToMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    /**
     * 当前激光帧角点寻找局部map匹配点
     * 1、更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
     * 2、计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
     */
    // 角点筛选，判断k+1scan的角点是否和k的地图上的边足够接近，只留下足够接近的点放入laserCloudOriCornerVec，用于匹配
    // TODO 筛选条件要求距离小于1m，是否意味着速度过快就没办法用了，因为距离会差很大
    void cornerOptimization()
    {
        updatePointAssociateToMap();

#pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudCornerLastDS->points[i];                                       //当前点在scan坐标系下的坐标
            pointAssociateToMap(&pointOri, &pointSel);                                          // 当前点在map坐标系下的坐标
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis); // 找出距离最近的5个点

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

            // 只有当最远的那个邻域点pointSearchSqDis[4]的距离小于1m时才进行下面的计算
            // 以下部分的计算是在计算点集的协方差矩阵，Zhang Ji的论文中有提到这部分
            // 实际就是PCA
            if (pointSearchSqDis[4] < 1.0)
            {
                // 先求5个样本的平均值
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++)
                {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5;
                cy /= 5;
                cz /= 5;
                // 下面在求矩阵matA1=[ax,ay,az]^t*[ax,ay,az]
                // 更准确地说应该是在求协方差matA1
                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++)
                {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax;
                    a12 += ax * ay;
                    a13 += ax * az;
                    a22 += ay * ay;
                    a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5;
                a12 /= 5;
                a13 /= 5;
                a22 /= 5;
                a23 /= 5;
                a33 /= 5;

                matA1.at<float>(0, 0) = a11;
                matA1.at<float>(0, 1) = a12;
                matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12;
                matA1.at<float>(1, 1) = a22;
                matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13;
                matA1.at<float>(2, 1) = a23;
                matA1.at<float>(2, 2) = a33;
                // 求正交阵的特征值和特征向量
                // 特征值：matD1，特征向量：matV1中
                cv::eigen(matA1, matD1, matV1);
                // 边缘：与较大特征值相对应的特征向量代表边缘线的方向（一大两小，大方向）
                // 以下这一大块是在计算点到边缘的距离，最后通过系数s来判断是否距离很近
                // 如果距离很近就认为这个点在边缘上，需要放到laserCloudOri中
                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1))
                {

                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);
                    // 这边是在求[(x0-x1),(y0-y1),(z0-z1)]与[(x0-x2),(y0-y2),(z0-z2)]叉乘得到的向量的模长
                    // 这个模长是由0.2*V1[0]和点[x0,y0,z0]构成的平行四边形的面积
                    /*
                         |  i       j      k   |
                    axb= | x0-x1  y0-y1  z0-z1 | = [(y0-y1)*(z0-z2)-(y0-y2)*(z0 -z1)]i+[(x0-x1)*(z0-z2)-(x0-x2)*(z0-z1)]j+[(x0-x1)*(y0-y2)-(x0-x2)*(y0-y1)]k
                         | x0-x2  y0-y2  z0-z2 |
                    */
                    float a012 = sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) + ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) + ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));
                    // l12表示的是0.2*(||V1[0]||)
                    // 也就是平行四边形一条底的长度
                    float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));

                    // 又一次叉乘，算距离直线的方向，除以两个向量的模，相当于归一化
                    float la = ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) + (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) - (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) + (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;
                    // 计算点pointSel到直线（过质心的方向向量）的距离
                    // 距离（高）=平行四边形面积/底边长度
                    float ld2 = a012 / l12;
                    // 如果在最理想的状态的话，ld2应该为0，表示点在直线上
                    // 最理想状态s=1；
                    float s = 1 - 0.9 * fabs(ld2);
                    // coeff代表系数的意思
                    // coff用于保存距离的方向向量
                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;

                    // intensity本质上构成了一个核函数，ld2越接近于1，增长越慢
                    // intensity=(1-0.9*ld2)*ld2=ld2-0.9*ld2*ld2
                    coeff.intensity = s * ld2;
                    // 所以就应该认为这个点是边缘点
                    // s>0.1 也就是要求点到直线的距离ld2要小于1m
                    // s越大说明ld2越小(离边缘线越近)，这样就说明点pointOri在直线上
                    if (s > 0.1)
                    {
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }

    /**
     * 当前激光帧平面点寻找局部map匹配点
     * 1、更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
     * 2、计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
     */
    // 平面点筛选，判断k+1scan的平面点是否和k的地图上的平面足够接近，只留下足够接近的点放入laserCloudOriSurfVec，用于匹配
    void surfOptimization()
    {
        updatePointAssociateToMap();

#pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            // 当前点Scan系下
            pointOri = laserCloudSurfLastDS->points[i];
            // 世界坐标系下的点
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            if (pointSearchSqDis[4] < 1.0)
            {
                for (int j = 0; j < 5; j++)
                {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }
                // matB0是一个5x1的矩阵
                // matB0 = cv::Mat (5, 1, CV_32F, cv::Scalar::all(-1));
                // matX0是3x1的矩阵
                // 求解方程matA0*matX0=matB0
                // 公式其实是在求由matA0中的点构成的平面的法向量matX0
                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                // [pa,pb,pc,pd]=[matX0,pd]
                // 正常情况下（见后面planeValid判断条件），应该是
                // pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                // pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                // pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z = -1
                // 所以pd设置为1
                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                // 对[pa,pb,pc,pd]进行单位化
                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                // 求解后再次检查平面是否是有效平面
                bool planeValid = true;
                for (int j = 0; j < 5; j++)
                {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2)
                    {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid)
                {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;
                    // 后面部分相除求的是[pa,pb,pc,pd]与pointSel的夹角余弦值(两个sqrt，其实并不是余弦值)
                    // 这个夹角余弦值越小越好，越小证明所求的[pa,pb,pc,pd]与平面越垂直
                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;
                    // 判断是否是合格平面，是就加入laserCloudOriSurfVec
                    if (s > 0.1)
                    {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

    // 把laserCloudOriCornerVec和laserCloudOriSurfVec的点放入laserCloudOri
    // 向量系数coeffSelCornerVec和coeffSelSurfVec放入coeffSel
    // TODO 为什么不像lego-loam中一样直接放进去得了，这不是多此一举嘛
    void combineOptimizationCoeffs()
    {
        // combine corner coeffs
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i)
        {
            if (laserCloudOriCornerFlag[i] == true)
            {
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }
        // combine surf coeffs
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i)
        {
            if (laserCloudOriSurfFlag[i] == true)
            {
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        // reset flag for next iteration
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

    // L-M优化transformTobeMapped ,虽然这里转到了camera坐标，但后面又转回来了。所以transformTobeMapped依旧是lidar坐标的
    // 这部分的代码是基于高斯牛顿法的优化，不是zhang ji论文中提到的基于L-M的优化方法
    // 这部分的代码使用旋转矩阵对欧拉角求导，优化欧拉角，不是zhang ji论文中提到的使用angle-axis的优化
    // 优化，使距离最小
    bool LMOptimization(int iterCount)
    {
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // lidar -> camera
        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]); // lidar roll -》camera yaw
        float crz = cos(transformTobeMapped[0]);

        // 可用于匹配的点的数目
        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50)
        { // laser cloud original 点云太少，就跳过这次循环
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        for (int i = 0; i < laserCloudSelNum; i++)
        {
            // lidar -> camera
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;
            // lidar -> camera
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera 在相机坐标系中
            // 求雅克比矩阵中的元素，距离d对roll角度的偏导量即d(d)/d(roll)
            // 更详细的数学推导参看wykxwyc.github.io
            float arx = (crx * sry * srz * pointOri.x + crx * crz * sry * pointOri.y - srx * sry * pointOri.z) * coeff.x + (-srx * srz * pointOri.x - crz * srx * pointOri.y - crx * pointOri.z) * coeff.y + (crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y - cry * srx * pointOri.z) * coeff.z;
            // 同上，求解的是对pitch的偏导量
            float ary = ((cry * srx * srz - crz * sry) * pointOri.x + (sry * srz + cry * crz * srx) * pointOri.y + crx * cry * pointOri.z) * coeff.x + ((-cry * crz - srx * sry * srz) * pointOri.x + (cry * srz - crz * srx * sry) * pointOri.y - crx * sry * pointOri.z) * coeff.z;
            // 同上，求解的是对yaw的偏导量
            float arz = ((crz * srx * sry - cry * srz) * pointOri.x + (-cry * crz - srx * sry * srz) * pointOri.y) * coeff.x + (crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.y + ((sry * srz + cry * crz * srx) * pointOri.x + (crz * sry - cry * srx * srz) * pointOri.y) * coeff.z;

            /*
            在求点到直线的距离时，coeff表示的是如下内容
            [la,lb,lc]表示的是点到直线的垂直连线方向，s是长度
            coeff.x = s * la;
            coeff.y = s * lb;
            coeff.z = s * lc;
            coeff.intensity = s * ld2;

            在求点到平面的距离时，coeff表示的是
            [pa,pb,pc]表示过外点的平面的法向量，s是线的长度
            coeff.x = s * pa;
            coeff.y = s * pb;
            coeff.z = s * pc;
            coeff.intensity = s * pd2;
            */
            // lidar <- camera
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            // 这部分是雅克比矩阵中距离对平移的偏导
            matA.at<float>(i, 3) = coeff.z;
            matA.at<float>(i, 4) = coeff.x;
            matA.at<float>(i, 5) = coeff.y;
            // 残差项
            matB.at<float>(i, 0) = -coeff.intensity;
        }
        // 将矩阵由matA转置生成matAt
        // 先进行计算，以便于后边调用 cv::solve求解
        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        // 利用高斯牛顿法进行求解，
        // 高斯牛顿法的原型是J^(T)*J * delta(x) = -J*f(x)
        // J是雅克比矩阵，这里是A，f(x)是优化目标，这里是-B(符号在给B赋值时候就放进去了)
        // 通过QR分解的方式，求解matAtA*matX=matAtB，得到解matX
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        //退化场景判断与处理 https://zhuanlan.zhihu.com/p/258159552
        if (iterCount == 0)
        {

            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0)); // 特征值
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0)); // 特征向量
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            // 特征值分解？？
            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--)
            {
                if (matE.at<float>(0, i) < eignThre[i])
                {
                    for (int j = 0; j < 6; j++)
                    {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                }
                else
                {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(
            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
            pow(matX.at<float>(3, 0) * 100, 2) +
            pow(matX.at<float>(4, 0) * 100, 2) +
            pow(matX.at<float>(5, 0) * 100, 2));
        // 旋转和平移量足够小就停止这次迭代过程
        if (deltaR < 0.01 && deltaT < 0.01)
        {
            return true; // converged
        }
        return false; // keep optimizing
    }

    // scan2map匹配优化
    void scan2MapOptimization()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        //大于最小阈值
        // laserCloudCornerFromMapDSNum是extractSurroundingKeyFrames()函数最后降采样得到的coner点云数
        // laserCloudSurfFromMapDSNum是extractSurroundingKeyFrames()函数降采样得到的surface点云数
        cout << "当前特征点：" << laserCloudCornerLastDSNum << "," << laserCloudSurfLastDSNum << endl;
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {

            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            // 用for循环控制迭代次数，最多迭代30次
            for (int iterCount = 0; iterCount < iter_num; iterCount++)
            {
                laserCloudOri->clear();
                coeffSel->clear();
                // 在 Qk中选取相邻点集合S，计算S的协方差矩阵M、特征向量E、特征值V。选取边缘线和平面块方式为：
                // 边缘线：V中特征值一大两小，E中大特征值代表的特征向量代表边缘线的方向。
                // 平面块：V中一小两大，E中小特征值对应的特征向量代表平面片的方向。
                // 边缘线或平面块的位置通过穿过S的几何中心来确定。

                // 当前激光帧角点寻找局部map匹配点
                // 1、更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
                // 2、计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数(相当于梯度下降的方向)
                cornerOptimization();
                // 当前激光帧平面点寻找局部map匹配点
                // 1、更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
                // 2、计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
                surfOptimization();
                // 提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
                combineOptimizationCoeffs();

                // scan-to-map优化
                // 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存transformTobeMapped
                if (LMOptimization(iterCount) == true)
                    break;
            }
            // 用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll、pitch，约束z坐标
            // 迭代结束更新相关的转移矩阵
            transformUpdate();
        }
        else
        {
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }

    // scan2map匹配优化
    // void scan2MapOptimization_ceres()
    // {
    //     if (cloudKeyPoses3D->points.empty())
    //         return;

    //     //大于最小阈值
    //     // laserCloudCornerFromMapDSNum是extractSurroundingKeyFrames()函数最后降采样得到的coner点云数
    //     // laserCloudSurfFromMapDSNum是extractSurroundingKeyFrames()函数降采样得到的surface点云数
    //     cout<<"当前特征点："<<laserCloudCornerLastDSNum<<","<<laserCloudSurfLastDSNum<<endl;
    //     if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
    //     {

    //         kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
    //         kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);
    //         double para_qt[7] = {0, 0, 0, 0,0,0,1};
    //         Eigen::Map<Eigen::Quaterniond> q_w_curr(para_qt+3);
    //         Eigen::Map<Eigen::Vector3d> t_w_curr(para_qt);
    //         Eigen::Affine3f T_before_opti =trans2Affine3f(transformTobeMapped);
    //         q_w_curr = Eigen::Quaterniond(T_before_opti.matrix().block<3, 3>(0, 0));
    //         t_w_curr = T_before_opti.matrix().block<3, 1>(0, 3);
    //          // 用for循环控制迭代次数，最多迭代30次
    //         for (int iterCount = 0; iterCount < iter_num; iterCount++)
    //         {
    //             Eigen::Quaterniond q_before_opti = q_w_curr;
    //             Eigen::Vector3d t_before_opti = t_w_curr;
    //             updatePointAssociateToMap();
    //             ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    //             ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
    //             ceres::Problem::Options problem_options;

    //             ceres::Problem problem(problem_options);
    //             problem.AddParameterBlock(para_qt, 7, local_parameterization);
    //             for (int i = 0; i < laserCloudCornerLastDSNum; i++)
    //             {
    //                 PointType pointOri, pointSel, coeff;
    //                 std::vector<int> pointSearchInd;
    //                 std::vector<float> pointSearchSqDis;

    //                 pointOri = laserCloudCornerLastDS->points[i];//当前点在scan坐标系下的坐标
    //                 pointAssociateToMap(&pointOri, &pointSel);// 当前点在map坐标系下的坐标
    //                 kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);// 找出距离最近的5个点
    //                 if (pointSearchSqDis[4] < 1.0)
    //                 {
    //                     // 先求5个样本的平均值
    //                     std::vector<Eigen::Vector3d> nearCorners; // 存储这5个样板
    //                     Eigen::Vector3d center(0, 0, 0); // 均值
    //                     for (int j = 0; j < 5; j++)
    //                     {
    //                     Eigen::Vector3d tmp(laserCloudCornerFromMapDS->points[pointSearchInd[j]].x,
    //                                 laserCloudCornerFromMapDS->points[pointSearchInd[j]].y,
    //                                 laserCloudCornerFromMapDS->points[pointSearchInd[j]].z);
    //                     center = center + tmp;
    //                     nearCorners.push_back(tmp);
    //                     }
    //                     center = center / 5.0;

    //                     Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
    //                     for (int j = 0; j < 5; j++)
    //                     {
    //                     Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
    //                     covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
    //                     }
    //                     // covMat = covMat/5.0; // 我添加的，不加也没事只是算个方向向量

    //                     Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

    //                     // if is indeed line feature
    //                     // note Eigen library sort eigenvalues in increasing order
    //                     // 特征值从小到大排列
    //                     Eigen::Vector3d unit_direction = saes.eigenvectors().col(2); // 最小特征向量即为直线的方向向量
    //                     Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
    //                     // 边缘：与较大特征值相对应的特征向量代表边缘线的方向（一大两小，大方向）
    //                     // 以下这一大块是在计算点到边缘的距离，最后通过系数s来判断是否距离很近
    //                     // 如果距离很近就认为这个点在边缘上，需要放到laserCloudOri中
    //                     if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
    //                     {
    //                     Eigen::Vector3d point_on_line = center;
    //                     Eigen::Vector3d point_a, point_b;  // 直线上的两个点
    //                     point_a = 0.1 * unit_direction + point_on_line;
    //                     point_b = -0.1 * unit_direction + point_on_line;

    //                     LidarEdgeFactor *cost_function = new LidarEdgeFactor(curr_point, point_a, point_b, 1.0);
    //                     problem.AddResidualBlock(cost_function, loss_function, para_qt);

    //                     }

    //                 }
    //             }
    //             for (int i = 0; i < laserCloudSurfLastDSNum; i++)
    //             {
    //                 PointType pointOri, pointSel, coeff;
    //                 std::vector<int> pointSearchInd;
    //                 std::vector<float> pointSearchSqDis;
    //                 // 当前点Scan系下
    //                 pointOri = laserCloudSurfLastDS->points[i];
    //                 // 世界坐标系下的点
    //                 pointAssociateToMap(&pointOri, &pointSel);
    //                 kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

    //                 Eigen::Matrix<double, 5, 3> matA0;
    //                 Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
    //                 if (pointSearchSqDis[4] < 1.0)
    //                 {

    //                     for (int j = 0; j < 5; j++)
    //                     {
    //                     matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
    //                     matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
    //                     matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
    //                     //printf(" pts %f %f %f ", matA0(j, 0), matA0(j, 1), matA0(j, 2));
    //                     }
    //                     // find the norm of plane
    //                     Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
    //                     double negative_OA_dot_norm = 1 / norm.norm();
    //                     norm.normalize();

    //                     // Here n(pa, pb, pc) is unit norm of plane
    //                     // 求解后再次检查平面是否是有效平面， 计算点到平面的距离。有过大的点，说明平面不好
    //                     bool planeValid = true;
    //                     for (int j = 0; j < 5; j++)
    //                     {
    //                     // if OX * n > 0.2, then plane is not fit well
    //                     if (fabs(norm(0) * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
    //                         norm(1) * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
    //                         norm(2) * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
    //                     {
    //                         planeValid = false;
    //                         break;
    //                     }
    //                     }
    //                     Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
    //                     if (planeValid)
    //                     {
    //                     LidarPlaneNormFactor *cost_function = new LidarPlaneNormFactor(curr_point, norm, negative_OA_dot_norm);
    //                     problem.AddResidualBlock(cost_function, loss_function, para_qt);

    //                     }
    //                 }
    //             }

    //             ceres::Solver::Options options;
    //             options.linear_solver_type = ceres::DENSE_QR;
    //             options.max_num_iterations = 4;
    //             options.minimizer_progress_to_stdout = false;
    //             options.check_gradients = false;
    //             options.gradient_check_relative_precision = 1e-4;
    //             ceres::Solver::Summary summary;
    //             ceres::Solve(options, &problem, &summary);

    //             double deltaR = (q_before_opti.angularDistance(q_w_curr)) * 180.0 / M_PI;
    //             double deltaT = (t_before_opti - t_w_curr).norm();

    //             if (deltaR < 0.05 && deltaT < 0.05 ) break;

    //         }
    //         // 迭代结束更新相关的转移矩阵
    //         transformUpdate();
    //     } else {
    //         ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
    //     }
    // }

    // 如果imuAvailable，和imu插值融合roll 和pitch更新transformTobeMapped和incrementalOdometryAffineBack
    // 检查阈值约束，大于阈值的=阈值
    void transformUpdate()
    {
        if (cloudInfo.imuAvailable == true)
        {
            // 使用imu和transfrom插值得到roll和pitch
            if (std::abs(cloudInfo.imuPitchInit) < 1.4)
            {
                double imuWeight = 0.01;
                tf::Quaternion imuQuaternion;
                tf::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid); // slerp 差值
                transformTobeMapped[0] = rollMid;

                // slerp pitch
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        // 差数设的是1000，相当于没有约束
        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);
        if (transformTobeMapped[5] < -5)
            transformTobeMapped[5] = -5;
        if (transformTobeMapped[5] > 1)
            transformTobeMapped[5] = 1;
        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    }

    // 最大最小值范围约束
    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    // 判断是否需要保存关键帧，
    // 当RPY角度或者位移大于阈值，则为true
    bool saveFrame()
    {
        if (cloudKeyPoses3D->points.empty())
            return true;

        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // extract all the nearby key poses and downsample them
        // 提取最后一个关键帧点云的周围50米范围内的关键帧
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        PointType tmp;
        tmp.x = transformTobeMapped[3];
        tmp.y = transformTobeMapped[4];
        tmp.z = transformTobeMapped[5];
        kdtreeSurroundingKeyPoses->radiusSearch(tmp, (double)3, pointSearchInd, pointSearchSqDis);
        if (pointSearchInd.size() > 30)
            return false; // 防止一个地方打转

        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        if (abs(roll) < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
            abs(yaw) < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    // 添加里程计因子
    void addOdomFactor()
    {
        // 为空添加先验因子
        if (cloudKeyPoses3D->points.empty())
        {
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-8).finished()); // rad*rad, meter*meter
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }
        else
        {
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }

    // 对齐gps时间戳添加gps因子
    void addGPSFactor()
    {
        if (gpsQueue.empty())
            return;

        // wait for system initialized and settles down
        if (cloudKeyPoses3D->points.empty()) // 没有关键帧，不添加gps
            return;
        else
        {
            // 关键帧距离太近，不添加
            if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
                return;
        }

        // pose covariance small, no need to correct
        // 位置协方差太小，不添加
        if (poseCovariance(3, 3) < poseCovThreshold && poseCovariance(4, 4) < poseCovThreshold)
            return;

        // last gps position
        static PointType lastGPSPoint;

        // 对齐时间戳
        while (!gpsQueue.empty())
        {
            if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
            }
            else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2)
            {
                // message too new
                break;
            }
            else
            {
                nav_msgs::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();
                Eigen::Vector3d Pwl;
                Eigen::Vector3d Pwi(thisGPS.pose.pose.position.x, thisGPS.pose.pose.position.y, thisGPS.pose.pose.position.z);
                Eigen::Quaterniond Qwi(thisGPS.pose.pose.orientation.w, thisGPS.pose.pose.orientation.x, thisGPS.pose.pose.orientation.y, thisGPS.pose.pose.orientation.z);
                Pwl = Pwi + Qwi * Pil;
                // GPS too noisy, skip
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;

                float gps_x = Pwl.x();
                float gps_y = Pwl.y();
                float gps_z = Pwl.z();
                if (!useGpsElevation)
                {
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }

                // GPS not properly initialized (0,0,0)
                // GPS未正确初始化（0,0,0）
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Add GPS every a few meters
                //每隔几米增加一次GPS
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                gtsam::Vector Vector3(3);
                Vector3 << max(noise_x, 0.1f), max(noise_y, 0.1f), max(noise_z, 1.0f);
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);

                aLoopIsClosed = true;
                break;
            }
        }
    }

    // 添加闭环因子
    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
    }

    // 添加里程计、gps、闭环因子，并执行gtsam优化，保存优化后的当前关键帧
    // 保存关键帧点云
    // 发布关键帧路径
    void saveKeyFramesAndFactor()
    {
        if (saveFrame() == false)
            return;

        // odom factor
        addOdomFactor();

        // gps factor
        addGPSFactor();

        // loop factor
        addLoopFactor();

        // cout << "****************************************************" << endl;

        // update iSAM
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        // 闭环成功则更新
        if (aLoopIsClosed == true)
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }

        gtSAMgraph.resize(0);
        initialEstimate.clear();

        // save key poses
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size() - 1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        // 添加当前帧
        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity; // this can be used as index
        thisPose6D.roll = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw = latestEstimate.rotation().yaw();
        thisPose6D.timestamp = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;

        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size() - 1);

        // save updated transform
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // save all the received edge and surf points
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame);

        // SC_loop_clouse
        // pcl::PointCloud<PointType>::Ptr thisCloudKeyFrame(new pcl::PointCloud<PointType>());
        // // 转到世界坐标系下 TODO 这里错了，不要转到世界坐标系下
        // *thisCloudKeyFrame += *transformPointCloud(thisCornerKeyFrame, &thisPose6D);
        // *thisCloudKeyFrame += *transformPointCloud(thisSurfKeyFrame,   &thisPose6D);

        // // TODO 这里用的是特征点云，还可用原始点云
        // scManager.makeAndSaveScancontextAndKeys(*thisCloudKeyFrame);
        scManager.makeAndSaveScancontextAndKeys(*laserCloudInRaw);

        // save key frame cloud
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // save path for visualization
        // 发布关键帧路径
        updatePath(thisPose6D);
    }

    // 当发生闭环时，更新所有关键帧的位姿和路径
    void correctPoses()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        if (aLoopIsClosed == true)
        {
            // clear map cache
            laserCloudMapContainer.clear();
            // clear path
            globalPath.poses.clear();
            // update key poses
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                updatePath(cloudKeyPoses6D->points[i]);
            }

            aLoopIsClosed = false;
        }
    }

    // 更新关键帧路径信息
    void updatePath(const PointTypePose &pose_in)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.timestamp);
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

    // 发布优化真实的里程计和不经过优化的增量里程计
    void publishOdometry()
    {
        // Publish odometry for ROS (global)
        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = odometryFrame;
        laserOdometryROS.child_frame_id = "odom_mapping";
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        pubLaserOdometryGlobal.publish(laserOdometryROS);
        // cout<< "普通xyz："<<transformTobeMapped[3]<<","<<transformTobeMapped[4]<<","<<transformTobeMapped[5]<<endl;
        // Publish TF
        static tf::TransformBroadcaster br;
        tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
                                                      tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "lidar_link");
        br.sendTransform(trans_odom_to_lidar);

        // Publish odometry for ROS (incremental)
        static bool lastIncreOdomPubFlag = false;
        static nav_msgs::Odometry laserOdomIncremental; // incremental odometry msg
        static Eigen::Affine3f increOdomAffine;         // incremental odometry in affine
        if (lastIncreOdomPubFlag == false)
        {
            lastIncreOdomPubFlag = true;
            laserOdomIncremental = laserOdometryROS;
            increOdomAffine = trans2Affine3f(transformTobeMapped);
        }
        else
        {
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack; // 优化前-1*优化后的= 优化的修正量
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;

            pcl::getTranslationAndEulerAngles(increOdomAffine, x, y, z, roll, pitch, yaw);
            // cout<< "增量xyz："<<x<<","<<y<<","<<z<<endl;

            // cout<< "两者的差值："<<transformTobeMapped[3]-x<<","<<transformTobeMapped[4]-y<<","<<transformTobeMapped[5]-z<<endl;
            if (cloudInfo.imuAvailable == true)
            {
                if (std::abs(cloudInfo.imuPitchInit) < 1.4)
                {
                    double imuWeight = 0.01;
                    tf::Quaternion imuQuaternion;
                    tf::Quaternion transformQuaternion;
                    double rollMid, pitchMid, yawMid;
                    transformQuaternion.setRPY(roll, pitch, 0);
                    imuQuaternion.setRPY(cloudInfo.imuRollInit, cloudInfo.imuPitchInit, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    roll = rollMid;
                    pitch = pitchMid;
                }
            }
            laserOdomIncremental.header.stamp = timeLaserInfoStamp;
            laserOdomIncremental.header.frame_id = odometryFrame;
            laserOdomIncremental.child_frame_id = "odom_mapping";
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
            if (isDegenerate)
                laserOdomIncremental.pose.covariance[2] = 1;
            else
                laserOdomIncremental.pose.covariance[2] = 0;
        }
        pubLaserOdometryIncremental.publish(laserOdomIncremental);
    }

    // 发布一些没什么用的topic
    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;
        // publish key poses
        publishCloud(&pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
        // Publish surrounding key frames
        publishCloud(&pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);
        // publish registered key frame
        if (pubRecentKeyFrame.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS, &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS, &thisPose6D);
            publishCloud(&pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish registered high-res raw cloud
        if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut, &thisPose6D);
            publishCloud(&pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish path
        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lio_sam");

    mapOptimization MO;

    ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");

    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::spin();

    loopthread.join();
    visualizeMapThread.join();

    return 0;
}
