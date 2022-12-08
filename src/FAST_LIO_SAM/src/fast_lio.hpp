#include <utility.h>
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

// gstam
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

// gnss
#include "GNSS_Processing.hpp"
#include "sensor_msgs/NavSatFix.h"

// save map
#include "fast_lio_sam/save_map.h"
#include "fast_lio_sam/save_pose.h"

// save data in kitti format
#include <sstream>
#include <fstream>
#include <iomanip>

class FAST_LIO : public ParamServer
{
public:
    FAST_LIO()
    {
        nh.param<bool>("publish/path_en", path_en, true);
        nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);
        nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);
        nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true);
        nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
        nh.param<string>("map_file_path", map_file_path, "");
        nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
        nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
        nh.param<bool>("common/time_sync_en", time_sync_en, false);
        nh.param<double>("filter_size_corner", filter_size_corner_min, 0.5);
        nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
        nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
        nh.param<double>("cube_side_length", cube_len, 200);
        nh.param<float>("mapping/det_range", DET_RANGE, 300.f);
        nh.param<double>("mapping/fov_degree", fov_deg, 180);
        nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
        nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
        nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
        nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
        nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
        nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
        nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
        nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
        nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
        nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
        nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
        nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
        nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
        nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
        nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
        nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());

        //设置imu和lidar外参和imu参数等
        Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov)); // 加速度协方差
        p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

        //设置gnss外参数
        Gnss_T_wrt_Lidar << VEC_FROM_ARRAY(extrinT_Gnss2Lidar);
        Gnss_R_wrt_Lidar << MAT_FROM_ARRAY(extrinR_Gnss2Lidar);

        cout << "p_pre->lidar_type " << p_pre->lidar_type << endl;
    }

    ~FAST_LIO();

private:
    ros::NodeHandle nh;
    bool runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
    bool scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
    int iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
    double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
    double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
    double total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
    int effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;

    M3D Gnss_R_wrt_Lidar(Eye3d); // gnss 与 imu 的外参
    V3D Gnss_T_wrt_Lidar(Zero3d);

    shared_ptr<Preprocess> p_pre(new Preprocess());
    shared_ptr<ImuProcess> p_imu(new ImuProcess());

    vector<double> extrinT(3, 0.0);
    vector<double> extrinR(9, 0.0);

    double cube_len = 0;
    float DET_RANGE = 300.0f;
    const float MOV_THRESHOLD = 1.5f;
    //根据lidar的FoV分割场景
    BoxPointType LocalMap_Points; // ikd-tree中,局部地图的包围盒角点
    bool Localmap_Initialized = false;

    double HALF_FOV_COS = 0, FOV_DEG = 0;

    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG)*0.5 * PI_M / 180.0);

    // 将更新的pose赋值到 transformTobeMapped
    void getCurPose(state_ikfom cur_state);

    void dump_lio_state_to_log(FILE *fp);

    void SigHandle(int sig);

    void pointBodyToWorld_ikfom(PointType const *const pi, PointType *const po, state_ikfom &s);

    //按当前body(lidar)的状态，将局部点转换到世界系下
    void pointBodyToWorld(PointType const *const pi, PointType *const po);

    //按当前body(lidar)的状态，将局部点转换到世界系下
    void pointBodyToWorld(PointType const *const pi, PointType *const po);

    template <typename T>
    void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po);

    void RGBpointBodyToWorld(PointType const *const pi, PointType *const po); //  lidar2world

    void test_RGBpointBodyToWorld(PointType const *const pi, PointType *const po, Eigen::Vector3d pos, Eigen::Matrix3d rotation); //  lidar2world

    void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po);

    void points_cache_collect();

    void lasermap_fov_segment();

    void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg);

    double timediff_lidar_wrt_imu = 0.0;
    bool timediff_set_flg = false; // 标记是否已经进行了时间补偿
    void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg);

    void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in);

    void gnss_cbk(const sensor_msgs::NavSatFixConstPtr &msg_in);

    double lidar_mean_scantime = 0.0;
    int scan_num = 0;
    bool sync_packages(MeasureGroup &meas);

    int process_increments = 0;
    void map_incremental();

    PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
    PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
    void publish_frame_world(const ros::Publisher &pubLaserCloudFull);

    void publish_effect_world(const ros::Publisher &pubLaserCloudEffect);

    void publish_map(const ros::Publisher &pubLaserCloudMap);

    template <typename T>
    void set_posestamp(T &out);

    void publish_odometry(const ros::Publisher &pubOdomAftMapped);

    void publish_path(const ros::Publisher pubPath);

    void publish_path_update(const ros::Publisher pubPath);

    //  发布gnss 轨迹
    void publish_gnss_path(const ros::Publisher pubPath);

    //构造H矩阵
    void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
};

// 将更新的pose赋值到 transformTobeMapped
void FAST_LIO::getCurPose(state_ikfom cur_state)
{
    //  欧拉角是没有群的性质，所以从SO3还是一般的rotation matrix 转换过来的结果一样
    Eigen::Vector3d eulerAngle = cur_state.rot.matrix().eulerAngles(2, 1, 0); //  yaw pitch roll  单位：弧度
    // V3D eulerAngle  =  SO3ToEuler(cur_state.rot)/57.3 ;     //   fastlio 自带  roll pitch yaw  单位: 度，旋转顺序 zyx

    // transformTobeMapped[0] = eulerAngle(0);                //  roll     使用 SO3ToEuler 方法时，顺序是 rpy
    // transformTobeMapped[1] = eulerAngle(1);                //  pitch
    // transformTobeMapped[2] = eulerAngle(2);                //  yaw

    transformTobeMapped[0] = eulerAngle(2);    //  roll  使用 eulerAngles(2,1,0) 方法时，顺序是 ypr
    transformTobeMapped[1] = eulerAngle(1);    //  pitch
    transformTobeMapped[2] = eulerAngle(0);    //  yaw
    transformTobeMapped[3] = cur_state.pos(0); //  x
    transformTobeMapped[4] = cur_state.pos(1); //   y
    transformTobeMapped[5] = cur_state.pos(2); // z
}

void FAST_LIO::SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void FAST_LIO::dump_lio_state_to_log(FILE *fp)
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                            // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2));    // Pos
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // omega
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2));    // Vel
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // Acc
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));       // Bias_g
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));       // Bias_a
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a
    fprintf(fp, "\r\n");
    fflush(fp);
}

void FAST_LIO::pointBodyToWorld_ikfom(PointType const *const pi, PointType *const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

//按当前body(lidar)的状态，将局部点转换到世界系下
void FAST_LIO::pointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    // world <-- imu <-- lidar
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template <typename T>
void FAST_LIO::pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void FAST_LIO::RGBpointBodyToWorld(PointType const *const pi, PointType *const po) //  lidar2world
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void FAST_LIO::test_RGBpointBodyToWorld(PointType const *const pi, PointType *const po, Eigen::Vector3d pos, Eigen::Matrix3d rotation) //  lidar2world
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(rotation * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void FAST_LIO::RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void FAST_LIO::points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    for (int i = 0; i < points_history.size(); i++)
        _featsArray->push_back(points_history[i]);
}

void FAST_LIO::lasermap_fov_segment()
{
    cub_needrm.clear(); // 清空需要移除的区域
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world); // X轴分界点转换到w系下
    V3D pos_LiD = pos_lid;                               // global系lidar位置

    //初始化局部地图包围盒角点，以为w系下lidar位置为中心
    if (!Localmap_Initialized)
    {
        for (int i = 0; i < 3; i++)
        {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }

    float dist_to_map_edge[3][2]; //各个方向与局部地图边界的距离
    bool need_move = false;
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);

        //与某个方向上的边界距离太小，标记需要移除need_move
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }

    //不需要移除则直接返回
    if (!need_move)
        return;

    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points; // 新的局部地图角点
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points;
        //与包围盒最小值角点距离
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints); // 移除较远包围盒
        }
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if (cub_needrm.size() > 0)
        kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void FAST_LIO::standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    mtx_buffer.lock();
    scan_count++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false; // 标记是否已经进行了时间补偿
void FAST_LIO::livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty())
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu; //????
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());

    // 特征提取或间隔采样
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr); //储存处理后的lidar特征
    time_buffer.push_back(last_timestamp_lidar);

    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void FAST_LIO::imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    publish_count++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    // lidar 和 imu时间差过大，且开启 时间同步, 纠正当前输入imu的时间
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        // 对输入imu时间，纠正为 时间差 + 原始时间
        msg->header.stamp =
            ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp; // update imu time

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void FAST_LIO::gnss_cbk(const sensor_msgs::NavSatFixConstPtr &msg_in)
{
    //  ROS_INFO("GNSS DATA IN ");
    double timestamp = msg_in->header.stamp.toSec();

    mtx_buffer.lock();

    // 没有进行时间纠正
    if (timestamp < last_timestamp_gnss)
    {
        ROS_WARN("gnss loop back, clear buffer");
        gnss_buffer.clear();
    }

    last_timestamp_gnss = timestamp;

    // convert ROS NavSatFix to GeographicLib compatible GNSS message:
    gnss_data.time = msg_in->header.stamp.toSec();
    gnss_data.status = msg_in->status.status;
    gnss_data.service = msg_in->status.service;
    gnss_data.pose_cov[0] = msg_in->position_covariance[0];
    gnss_data.pose_cov[1] = msg_in->position_covariance[4];
    gnss_data.pose_cov[2] = msg_in->position_covariance[8];

    mtx_buffer.unlock();

    if (!gnss_inited)
    { //  初始化位置
        gnss_data.InitOriginPosition(msg_in->latitude, msg_in->longitude, msg_in->altitude);
        gnss_inited = true;
    }
    else
    {                                                                               //   初始化完成
        gnss_data.UpdateXYZ(msg_in->latitude, msg_in->longitude, msg_in->altitude); //  WGS84 -> ENU  ???  调试结果好像是 NED 北东地
        nav_msgs::Odometry gnss_data_enu;
        // add new message to buffer:
        gnss_data_enu.header.stamp = ros::Time().fromSec(gnss_data.time);
        gnss_data_enu.pose.pose.position.x = gnss_data.local_N;  // gnss_data.local_E ;   北
        gnss_data_enu.pose.pose.position.y = gnss_data.local_E;  // gnss_data.local_N;    东
        gnss_data_enu.pose.pose.position.z = -gnss_data.local_U; //  地

        gnss_data_enu.pose.pose.orientation.x = geoQuat.x; //  gnss 的姿态不可观，所以姿态只用于可视化，取自imu
        gnss_data_enu.pose.pose.orientation.y = geoQuat.y;
        gnss_data_enu.pose.pose.orientation.z = geoQuat.z;
        gnss_data_enu.pose.pose.orientation.w = geoQuat.w;

        gnss_data_enu.pose.covariance[0] = gnss_data.pose_cov[0];
        gnss_data_enu.pose.covariance[7] = gnss_data.pose_cov[1];
        gnss_data_enu.pose.covariance[14] = gnss_data.pose_cov[2];

        gnss_buffer.push_back(gnss_data_enu);

        // visial gnss path in rviz:
        msg_gnss_pose.header.frame_id = "camera_init";
        msg_gnss_pose.header.stamp = ros::Time().fromSec(gnss_data.time);
        // Eigen::Vector3d gnss_pose_ (gnss_data.local_E, gnss_data.local_N, - gnss_data.local_U);
        Eigen::Vector3d gnss_pose_(gnss_data.local_N, gnss_data.local_E, -gnss_data.local_U);

        Eigen::Isometry3d gnss_to_lidar(Gnss_R_wrt_Lidar);
        gnss_to_lidar.pretranslate(Gnss_T_wrt_Lidar);
        gnss_pose_ = gnss_to_lidar * gnss_pose_; //  gnss 转到 lidar 系下

        msg_gnss_pose.pose.position.x = gnss_pose_(0, 3);
        msg_gnss_pose.pose.position.y = gnss_pose_(1, 3);
        msg_gnss_pose.pose.position.z = gnss_pose_(2, 3);

        gps_path.poses.push_back(msg_gnss_pose);

        //  save_gnss path
        PointTypePose thisPose6D;
        thisPose6D.x = msg_gnss_pose.pose.position.x;
        thisPose6D.y = msg_gnss_pose.pose.position.y;
        thisPose6D.z = msg_gnss_pose.pose.position.z;
        thisPose6D.intensity = 0;
        thisPose6D.roll = 0;
        thisPose6D.pitch = 0;
        thisPose6D.yaw = 0;
        thisPose6D.time = lidar_end_time;
        gnss_cloudKeyPoses6D->push_back(thisPose6D);
    }
}

double lidar_mean_scantime = 0.0;
int scan_num = 0;
bool FAST_LIO::sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty())
    {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();         // lidar指针指向最旧的lidar数据
        meas.lidar_beg_time = time_buffer.front(); //记录最早时间

        //更新结束时刻的时间
        if (meas.lidar->points.size() <= 1) // time too little 时间太短，点数不足
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime; // 记录lidar结束时间为 起始时间 + 单帧扫描时间
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime) //最后一个点的时间 小于 单帧扫描时间的一半
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime; // 记录lidar结束时间为 起始时间 + 单帧扫描时间
        }
        else
        {
            scan_num++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000); //结束时间设置为 起始时间 + 最后一个点的时间（相对）
            // 动态更新每帧lidar数据平均扫描时间
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec(); // 最旧IMU时间
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time)) //记录imu数据，imu时间小于当前帧lidar结束时间
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time)
            break;
        meas.imu.push_back(imu_buffer.front()); //记录当前lidar帧内的imu数据到meas.imu
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void FAST_LIO::map_incremental()
{
    PointVector PointToAdd;            //需要加入到ikd-tree中的点云
    PointVector PointNoNeedDownsample; //加入ikd-tree时，不需要降采样的点云
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);

    //根据点与所在包围盒中心点的距离，分类是否需要降采样
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point;
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            float dist = calc_dist(feats_down_world->points[i], mid_point); //当前点与box中心的距离

            //判断最近点在x、y、z三个方向上，与中心的距离，判断是否加入时需要降采样
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
            {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]); //若三个方向距离都大于地图珊格半轴长，无需降采样
                continue;
            }

            //判断当前点的 NUM_MATCH_POINTS 个邻近点与包围盒中心的范围
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
            {
                if (points_near.size() < NUM_MATCH_POINTS)
                    break;
                if (calc_dist(points_near[readd_i], mid_point) < dist) // 如果邻近点到中心的距离 小于 当前点到中心的距离，则不需要添加当前点
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
                PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true); //加入点时需要降采样
    ikdtree.Add_Points(PointNoNeedDownsample, false);      //加入点时不需要降采样
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void FAST_LIO::publish_frame_world(const ros::Publisher &pubLaserCloudFull) //    将稠密点云从 imu convert to  world
{
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i],
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
        {
            pcd_index++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void FAST_LIO::publish_frame_body(const ros::Publisher &pubLaserCloudFull_body) //   发布body系(imu)下的点云
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i],
                               &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void FAST_LIO::publish_effect_world(const ros::Publisher &pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld(
        new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i],
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void FAST_LIO::publish_map(const ros::Publisher &pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template <typename T>
void FAST_LIO::set_posestamp(T &out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
}

void FAST_LIO::publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time); // ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                    odomAftMapped.pose.pose.position.y,
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "body"));
}

void FAST_LIO::publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0)
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);

        //  save  unoptimized pose
        V3D rot_ang(Log(state_point.rot.toRotationMatrix())); //   旋转向量
        PointTypePose thisPose6D;
        thisPose6D.x = msg_body_pose.pose.position.x;
        thisPose6D.y = msg_body_pose.pose.position.y;
        thisPose6D.z = msg_body_pose.pose.position.z;
        thisPose6D.roll = rot_ang(0);
        thisPose6D.pitch = rot_ang(1);
        thisPose6D.yaw = rot_ang(2);
        fastlio_unoptimized_cloudKeyPoses6D->push_back(thisPose6D);
    }
}

void FAST_LIO::publish_path_update(const ros::Publisher pubPath)
{
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time); //  时间戳
    string odometryFrame = "camera_init";
    if (pubPath.getNumSubscribers() != 0)
    {
        /*** if path is too large, the rvis will crash ***/
        static int kkk = 0;
        kkk++;
        if (kkk % 10 == 0)
        {
            // path.poses.push_back(globalPath);
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }
    }
}

//  发布gnss 轨迹
void FAST_LIO::publish_gnss_path(const ros::Publisher pubPath)
{
    gps_path.header.stamp = ros::Time().fromSec(lidar_end_time);
    gps_path.header.frame_id = "camera_init";

    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0)
    {
        pubPath.publish(gps_path);
    }
}

//构造H矩阵
void FAST_LIO::h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear();
    corr_normvect->clear();
    total_residual = 0.0;

/** closest surface search and residual computation **/
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < feats_down_size; i++) //判断每个点的对应邻域是否符合平面点的假设
    {
        PointType &point_body = feats_down_body->points[i];   // lidar系下坐标
        PointType &point_world = feats_down_world->points[i]; // lidar数据点在world系下坐标

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);                     // lidar系下坐标
        V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos); // w系下坐标
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            // world系下从ikdtree找5个最近点用于平面拟合
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            //最近点数大于NUM_MATCH_POINTS，且最大距离小于等于5,point_selected_surf设置为true
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
                                                                                                                                : true;
        }

        //不符合平面特征
        if (!point_selected_surf[i])
            continue;

        VF(4)
        pabcd;
        point_selected_surf[i] = false; //二次筛选平面点
        //拟合局部平面，返回：是否有内点大于距离阈值
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            // plane distance
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm()); //筛选条件 1 - 0.9 * （点到平面距离 / 点到lidar原点距离）

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2; //以intensity记录点到面残差
                res_last[i] = abs(pd2);             // 残差，距离
            }
        }
    }

    effct_feat_num = 0; //有效匹配点数

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i]; // body系 平面特征点
            corr_normvect->points[effct_feat_num] = normvec->points[i];         // world系 平面参数
            total_residual += res_last[i];                                      // 残差和
            effct_feat_num++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num; // 残差均值 （距离）
    match_time += omp_get_wtime() - match_start;
    double solve_start_ = omp_get_wtime();

    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //定义H维度
    ekfom_data.h.resize(effct_feat_num);                 //有效方程个数

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p = laserCloudOri->points[i]; // lidar系 平面特征点
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I; // 当前状态imu系下 点坐标
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this); // 当前状态imu系下 点坐标反对称矩阵

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z); //对应局部法相量, world系下

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() * norm_vec); // 将对应局部法相量旋转到imu系下 corr_normal_I
        V3D A(point_crossmat * C);           //残差对角度求导系数 P(IMU)^ [R(imu <-- w) * normal_w]
        //添加数据到矩阵
        if (extrinsic_est_en)
        {
            // B = lidar_p^ R(L <-- I) * corr_normal_I
            // B = lidar_p^ R(L <-- I) * R(I <-- W) * normal_W
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); // s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}
