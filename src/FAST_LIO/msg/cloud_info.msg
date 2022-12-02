# Cloud Info
Header header 

int32[] startRingIndex  # 点云每个水平线束在pointRange中的开始索引
int32[] endRingIndex    # 点云每个水平线束在pointRange中的结束索引

int32[]  pointColInd # point column index in range image ，用来在后面点云中计算相邻两个点的列差值
# int32[]  pointRowInd # point row index in range image
float32[] pointRange # rangeMat中存的range

int64 imuAvailable # IMU是否可用，imuQueue不为空即可用，即imu有数据即可 ，imu频率太低可能导致时间戳对不上而无效
int64 odomAvailable # imu预积分odom是否可用，只要imu预积分里程计有值 ，imu频率太低可能导致时间戳对不上而无效

# Attitude for LOAM initialization
# timeScanCur - 0.01 ~ timeScanCur 区间内离timeScanCur最近的imu的原始数据
# 即当前scan所对应的imu真值
float32 imuRollInit 
float32 imuPitchInit
float32 imuYawInit

# Initial guess from imu pre-integration
# imu预积分里程计的>=timeScanCur的最近的值
# TODO RPY都有真值了还要估计值干嘛
float32 initialGuessX
float32 initialGuessY
float32 initialGuessZ
float32 initialGuessRoll
float32 initialGuessPitch
float32 initialGuessYaw

# Point cloud messages
sensor_msgs/PointCloud2 cloud_deskewed  # 原始点云降采样，去畸变，有range值的点云original cloud deskewed
sensor_msgs/PointCloud2 cloud_corner    # extracted corner feature
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature