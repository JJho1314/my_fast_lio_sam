#ifndef DYNAMIC_MAP_H
#define DYNAMIC_MAP_H

#include <condition_variable>
#include <queue>
#include <thread>
#include <boost/filesystem.hpp>
#include <string>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <std_msgs/Bool.h>
#include <tf/transform_listener.h>
// #include "utility.h"



typedef pcl::PointXYZI PointType;

struct Area
{
  std::string path;  // pcd 文件路径
  double x_min;
  double y_min;
  double z_min;
  double x_max;
  double y_max;
  double z_max;
};

typedef std::vector<Area> AreaList;
typedef std::vector<std::vector<std::string>> Tbl;

AreaList all_Corner_areas;      // 所有角点地图
AreaList all_Surf_areas;      // 所有面点地图
std::vector<std::string> Corner_pcd_file_paths;
std::vector<std::string> Surf_pcd_file_paths;


std::vector<std::string> cached_arealist_paths;



/**
 * @brief 读取保存地图的csv
 * @param path csv文件路径
 * @return a Table
 * @details 每一行：pcd_path, x_min,y_min,z_min,x_max,y_max,z_max
 */
Tbl read_csv(const std::string& path)
{
  std::ifstream ifs(path.c_str());
  std::string line;
  Tbl ret;
  while (std::getline(ifs, line))
  {
    std::istringstream iss(line);
    std::string col;
    std::vector<std::string> cols;
    while (std::getline(iss, col, ','))
      cols.push_back(col);
    ret.push_back(cols);
  }
  return ret;
}

void write_csv(const std::string& path, const Tbl& tbl)
{
  std::ofstream ofs(path.c_str());
  for (const std::vector<std::string>& cols : tbl)
  {
    std::string line;
    for (size_t i = 0; i < cols.size(); ++i)
    {
      if (i == 0)
        line += cols[i];
      else
        line += "," + cols[i];
    }
    ofs << line << std::endl;
  }
}

AreaList read_arealist(const std::string& path)
{
  Tbl tbl = read_csv(path);
  AreaList ret;
  for (const std::vector<std::string>& cols : tbl)
  {
    Area area;
    area.path = cols[0];
    area.x_min = std::stod(cols[1]);
    area.y_min = std::stod(cols[2]);
    area.z_min = std::stod(cols[3]);
    area.x_max = std::stod(cols[4]);
    area.y_max = std::stod(cols[5]);
    area.z_max = std::stod(cols[6]);
    ret.push_back(area);
  }
  return ret;
}

void write_arealist(const std::string& path, const AreaList& areas)
{
  Tbl tbl;
  for (const Area& area : areas)
  {
    std::vector<std::string> cols;
    cols.push_back(area.path);
    cols.push_back(std::to_string(area.x_min));
    cols.push_back(std::to_string(area.y_min));
    cols.push_back(std::to_string(area.z_min));
    cols.push_back(std::to_string(area.x_max));
    cols.push_back(std::to_string(area.y_max));
    cols.push_back(std::to_string(area.z_max));
    tbl.push_back(cols);
  }
  write_csv(path, tbl);
}

bool is_downloaded(const std::string& path)
{
  struct stat st;
  return (stat(path.c_str(), &st) == 0);
}

/**
 * @brief 判断位置是否在区域内
 * @param x
 * @param y
 * @param area
 * @param m 边界margin的大小
 * @return
 */
bool is_in_area(double x, double y, const Area& area, double m)
{
  return ((area.x_min - m) <= x && x <= (area.x_max + m) && (area.y_min - m) <= y && y <= (area.y_max + m));
}

/**
 * @brief 往AreaList里添加一行
 * @param area 需要添加的数据
 * @param areas 被添加的list
 */
void cache_arealist(const Area& area, AreaList& areas)
{
  for (const Area& a : areas)
  {
    if (a.path == area.path)
      return;
  }
  areas.push_back(area);
}


// 根据点的位置加载，包含这个点的pcd文件
pcl::PointCloud<PointType>::Ptr create_pcd(const float& p_x,const float& p_y, AreaList& downloaded_areas,std::string global_path,float margin)
{
  pcl::PointCloud<PointType>::Ptr pcd, part;
  pcd.reset(new pcl::PointCloud<PointType>());
  part.reset(new pcl::PointCloud<PointType>());
  int count=0;
  for (const Area& area : downloaded_areas)
  {
    if (is_in_area(p_x, p_y, area, margin))
    {
      if (pcd->points.size()==0) // 第一个pcd文件
      { 
        ++count;
        // std::cout << "加载第一个" << area.path.c_str() << std::endl;
        pcl::io::loadPCDFile(global_path+area.path.c_str(), *pcd);
      }        
      else
      {
         ++count;
        // std::cout << "加载第二个" << area.path.c_str() << std::endl;
        pcl::io::loadPCDFile(global_path+area.path.c_str(), *part);
        *pcd +=*part;
      }
    }
  }
  std::cout << "总共加载了" << count<< "张地图" << std::endl;
  return pcd;
}

// 从路径加载pcd
pcl::PointCloud<PointType>::Ptr create_pcd(const std::vector<std::string>& pcd_paths)
{
  pcl::PointCloud<PointType>::Ptr pcd, part;
  pcd.reset(new pcl::PointCloud<PointType>());
  part.reset(new pcl::PointCloud<PointType>());
  for (const std::string& path : pcd_paths)
  {
    // Following outputs are used for progress bar of Runtime Manager.
    if (pcd->points.size()==0)
    {
      if (pcl::io::loadPCDFile(path.c_str(), *pcd) == -1)
      {
        std::cerr << "load failed " << path << std::endl;
      }
    }
    else
    {
      if (pcl::io::loadPCDFile(path.c_str(), *part) == -1)
      {
        std::cerr << "load failed " << path << std::endl;
      }
      *pcd +=*part;
    }
    std::cerr << "load " << path << std::endl;
    if (!ros::ok())
      break;
  }

  return pcd;
}


#endif