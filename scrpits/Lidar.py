#! /usr/bin/env python

import numpy as np
from math import *
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan

MAX_LIDAR_DISTANCE = .8
COLLISION_DISTANCE = 0.14 # LaserScan.range_min = 0.1199999
NEARBY_DISTANCE = 0.45

ZONE_0_LENGTH = .25
ZONE_1_LENGTH = .5

ANGLE_MAX = 360 - 1 #359  degree
ANGLE_MIN = 1 - 1   #0 degree
ANGLE_BACK = 180  #180 degree
# HORIZON_WIDTH = 75  #original
HORIZON_WIDTH = [9, 16, 56, 9] #9:x1, x2, x7   16:x3, x4   25:x5, x6 

# Convert LasecScan msg to array
def lidarScan(msgScan):
    distances = np.array([])
    angles = np.array([])

    for i in range(len(msgScan.ranges)):
        angle = degrees(i * msgScan.angle_increment)
        if ( msgScan.ranges[i] > MAX_LIDAR_DISTANCE ):
            distance = MAX_LIDAR_DISTANCE
        elif ( msgScan.ranges[i] < msgScan.range_min ):
            distance = msgScan.range_min
            # For real robot - protection
            if msgScan.ranges[i] < 0.01:
                distance = MAX_LIDAR_DISTANCE
        else:
            distance = msgScan.ranges[i]

        distances = np.append(distances, distance)
        angles = np.append(angles, angle)

    # distances in [m], angles in [degrees]
    return ( distances, angles )

# Discretization of lidar scan
def scanDiscretization(state_space, lidar, target_pos, robot_pose, robot_direction, robot_prev_pose, max_dist):
    x1 = 1  # no obstacle
    x2 = 1
    x3 = 2
    x4 = 3
    x5 = 3
    x6 = 2 
    x7 = 1
    x8 = 1
    
    lidar_x1 = min(lidar[81: 91])
    if ZONE_0_LENGTH < lidar_x1 < ZONE_1_LENGTH:
        x1 = 0
    
    lidar_x2 = min(lidar[25: 91])
    if ZONE_0_LENGTH > lidar_x2:
        x2 = 0

    lidar_x3 = min(lidar[9: 26])
    if ZONE_1_LENGTH < lidar_x3:
        x3 = 2
    elif ZONE_0_LENGTH < lidar_x3 < ZONE_1_LENGTH:
        x3 = 1
    elif lidar_x3 < ZONE_0_LENGTH:
        x3 = 0

    lidar_x4 = min(lidar[0: 10])
    if MAX_LIDAR_DISTANCE < lidar_x4:
        x4 = 3
    elif ZONE_1_LENGTH < lidar_x4:
        x4 = 2
    elif ZONE_0_LENGTH < lidar_x4 < ZONE_1_LENGTH:
        x4 = 1
    elif lidar_x4 < ZONE_0_LENGTH:
        x4 = 0

    # from index 351 to 0
    lidar_x5 = min(lidar[350: 360] + lidar[0])
    if MAX_LIDAR_DISTANCE < lidar_x5:
        x5 = 3
    elif ZONE_1_LENGTH < lidar_x5:
        x5 = 2
    elif ZONE_0_LENGTH < lidar_x5 < ZONE_1_LENGTH:
        x5 = 1
    elif lidar_x4 < ZONE_0_LENGTH:
        x5 = 0

    lidar_x6 = min(lidar[335: 351])
    if ZONE_1_LENGTH < lidar_x6:
        x6 = 2
    elif ZONE_0_LENGTH < lidar_x6 < ZONE_1_LENGTH:
        x6 = 1
    elif lidar_x6 < ZONE_0_LENGTH:
        x6 = 0

    lidar_x7 = min(lidar[270: 335])
    if ZONE_0_LENGTH > lidar_x7:
        x7 = 0

    lidar_x8 = min(lidar[270: 279])
    if ZONE_0_LENGTH < lidar_x8 < ZONE_1_LENGTH:
        x8 = 0

    # distance
    
    dist = np.linalg.norm(target_pos - robot_pose)

    if dist > .5 * max_dist:
        x9 = 2
    elif dist > .1:
        x9 = 1
    else:
        x9 = 0
    
    robot_pose = np.array([robot_pose[0], robot_pose[1]])
    robot_prev_pose = np.array([robot_prev_pose[0], robot_prev_pose[1]])
    target_pos = np.array([target_pos[0], target_pos[1]])

    #  vector from robot to target
    d_vec = target_pos - robot_pose
    #  vexor from robot to robot_prev
    v_vec = robot_pose - robot_prev_pose

    d_vec3d = np.array([d_vec[0], d_vec[1], 0])
    v_vec3d = np.array([v_vec[0], v_vec[1], 0])

    if np.dot(d_vec, v_vec) < 0:
        x10 = 0 # going back
    else:
        if np.arcos(np.dot(d_vec, v_vec) / (np.linalg.norm(d_vec) * np.linalg.norm(v_vec))) < np.arcsin(0.1)/dist:
            x10 = 1 # going to target
        elif np.cross(d_vec3d, v_vec3d)[2] < 0:
            x10 = 2  # too much right
        else:
            x10 = 3 # too much left
       


    ss = np.where(np.all(state_space == np.array([x1,x2,x3,x4, x5, x6, x7, x8, x9, x10]), axis = 1))
    state_ind = int(ss[0])

    return ( state_ind, x1, x2, x3 , x4 , x5, x6, x7, x8, x9, x10)

# Check - crash
def checkCrash(lidar):
    lidar_horizon = np.concatenate((lidar[(ANGLE_MIN + sum(HORIZON_WIDTH)):(ANGLE_MIN):-1],lidar[(ANGLE_MAX):(ANGLE_MAX - sum(HORIZON_WIDTH)):-1]))
    W = np.linspace(1.56, 1, len(lidar_horizon) // 2)
    W = np.append(W, np.linspace(1, 1.56, len(lidar_horizon) // 2))
    if np.min( W * lidar_horizon ) < COLLISION_DISTANCE:
        return True
    else:
        return False

# Check - object nearby
def checkObjectNearby(lidar):
    lidar_horizon = np.concatenate((lidar[(ANGLE_MIN + sum(HORIZON_WIDTH)):(ANGLE_MIN):-1],lidar[(ANGLE_MAX):(ANGLE_MAX - sum(HORIZON_WIDTH)):-1]))
    W = np.linspace(1.56, 1, len(lidar_horizon) // 2)
    W = np.append(W, np.linspace(1, 1.56, len(lidar_horizon) // 2))
    if np.min( W * lidar_horizon ) < NEARBY_DISTANCE:
        return True
    else:
        return False

# Check - goal near
def checkGoalNear(x, y, x_goal, y_goal):
    ro = sqrt( pow( ( x_goal - x ) , 2 ) + pow( ( y_goal - y ) , 2) )
    if ro < 0.3:
        return True
    else:
        return False
