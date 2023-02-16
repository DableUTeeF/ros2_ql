#! /usr/bin/env python

import rclpy
from time import time
from time import sleep
from datetime import datetime
import matplotlib.pyplot as plt
from rclpy.node import Node
from std_srvs.srv import Empty
from std_srvs.srv._empty import Empty_Request
import sys
DATA_PATH = '/home/palm/PycharmProjects/ros2_ql/Data'
MODULES_PATH = '/home/palm/PycharmProjects/ros2_ql/scripts'
sys.path.insert(0, MODULES_PATH)
from gazebo_msgs.msg._model_state import ModelState
from geometry_msgs.msg import Twist

from Qlearning import *
from Lidar import *
from Control import *
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec

# Episode parameters
MAX_EPISODES = 400
MAX_STEPS_PER_EPISODE = 500
MIN_TIME_BETWEEN_ACTIONS = 0.0

# Learning parameters
ALPHA = 0.5
GAMMA = 0.9

T_INIT = 25
T_GRAD = 0.95
T_MIN = 0.001

EPSILON_INIT = 0.9
EPSILON_GRAD = 0.96
EPSILON_MIN = 0.05

# 1 - Softmax , 2 - Epsilon greedy
EXPLORATION_FUNCTION = 1

# Initial position
X_INIT = -2.0
Y_INIT = -0.5
THETA_INIT = 0.0

RANDOM_INIT_POS = False

# Log file directory
LOG_FILE_DIR = DATA_PATH + '/Log_learning'

# Q table source file
Q_SOURCE_DIR = ''


REWARD_THRESHOLD =  -200.0
CUMULATIVE_REWARD = 0.0
GOAL_POSITION = (.0, .0, .0)
(GOAL_X, GOAL_Y, GOAL_THETA) = GOAL_POSITION

class LearningNode(Node):
    def __init__(self):
        super().__init__('learning_node')
        self.timer_period = .5 # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.reset = self.create_client(Empty, '/reset_simulation')
        self.setPosPub = self.create_publisher(ModelState, 'gazebo/set_model_state', 10)
        self.velPub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.dummy_req = Empty_Request()
        self.reset.call_async(self.dummy_req)
        self.actions = createActions()
        self.state_space = createStateSpace()
        self.Q_table = createQTable(len(self.state_space), len(self.actions))
        print('Initial Q-table:')
        print(self.Q_table)
        # Init log files
        self.log_sim_info = open(LOG_FILE_DIR+'/LogInfo.txt','w+')
        self.log_sim_params = open(LOG_FILE_DIR+'/LogParams.txt','w+')
        # Learning parameters
        self.T = T_INIT
        self.EPSILON = EPSILON_INIT
        self.alpha = ALPHA
        self.gamma = GAMMA
        # Episodes, steps, rewards
        self.ep_steps = 0
        self.ep_reward = 0
        self.episode = 1
        self.crash = 0
        self.reward_max_per_episode = np.array([])
        self.reward_min_per_episode = np.array([])
        self.reward_avg_per_episode = np.array([])
        self.ep_reward_arr = np.array([])
        self.steps_per_episode = np.array([])
        self.reward_per_episode = np.array([])
        # initial position
        self.robot_in_pos = False
        self.first_action_taken = False
        # init time
        self.t_0 = self.get_clock().now()
        self.t_start = self.get_clock().now()

        # init timer
        while not (self.t_start > self.t_0):
            self.t_start = self.get_clock().now()

        self.t_ep = self.t_start
        self.t_sim_start = self.t_start
        self.t_step = self.t_start

        self.T_per_episode = np.array([])
        self.EPSILON_per_episode = np.array([])
        self.t_per_episode = np.array([])

        self.CUMULATIVE_REWARD = CUMULATIVE_REWARD
    
    def log_init(self):
        # Date
        now_start = datetime.now()
        dt_string_start = now_start.strftime("%d/%m/%Y %H:%M:%S")

        # Log date to files
        text = '\r\n' + 'SIMULATION START ==> ' + dt_string_start + '\r\n\r\n'
        print(text)
        self.log_sim_info.write(text)
        self.log_sim_params.write(text)

        # Log simulation parameters
        text = '\r\nSimulation parameters: \r\n'
        text = text + '--------------------------------------- \r\n'
        if RANDOM_INIT_POS:
            text = text + 'INITIAL POSITION = RANDOM \r\n'
        else:
            text = text + 'INITIAL POSITION = ( %.2f , %.2f , %.2f ) \r\n' % (X_INIT,Y_INIT,THETA_INIT)
        text = text + '--------------------------------------- \r\n'
        text = text + 'MAX_EPISODES = %d \r\n' % MAX_EPISODES
        text = text + 'MAX_STEPS_PER_EPISODE = %d \r\n' % MAX_STEPS_PER_EPISODE
        text = text + 'MIN_TIME_BETWEEN_ACTIONS = %.2f s \r\n' % MIN_TIME_BETWEEN_ACTIONS
        text = text + '--------------------------------------- \r\n'
        text = text + 'ALPHA = %.2f \r\n' % ALPHA
        text = text + 'GAMMA = %.2f \r\n' % GAMMA
        if EXPLORATION_FUNCTION == 1:
            text = text + 'T_INIT = %.3f \r\n' % T_INIT
            text = text + 'T_GRAD = %.3f \r\n' % T_GRAD
            text = text + 'T_MIN = %.3f \r\n' % T_MIN
        else:
            text = text + 'EPSILON_INIT = %.3f \r\n' % EPSILON_INIT
            text = text + 'EPSILON_GRAD = %.3f \r\n' % EPSILON_GRAD
            text = text + 'EPSILON_MIN = %.3f \r\n' % EPSILON_MIN
        text = text + '--------------------------------------- \r\n'
        text = text + 'MAX_LIDAR_DISTANCE = %.2f \r\n' % MAX_LIDAR_DISTANCE
        text = text + 'COLLISION_DISTANCE = %.2f \r\n' % COLLISION_DISTANCE
        text = text + 'ZONE_0_LENGTH = %.2f \r\n' % ZONE_0_LENGTH
        text = text + 'ZONE_1_LENGTH = %.2f \r\n' % ZONE_1_LENGTH
        text = text + '--------------------------------------- \r\n'
        text = text + 'CONST_LINEAR_SPEED_FORWARD = %.3f \r\n' % CONST_LINEAR_SPEED_FORWARD
        text = text + 'CONST_ANGULAR_SPEED_FORWARD = %.3f \r\n' % CONST_ANGULAR_SPEED_FORWARD
        text = text + 'CONST_LINEAR_SPEED_TURN = %.3f \r\n' % CONST_LINEAR_SPEED_TURN
        text = text + 'CONST_ANGULAR_SPEED_TURN = %.3f \r\n' % CONST_ANGULAR_SPEED_TURN
        self.log_sim_params.write(text)
    
    def wait_for_message(
        node,
        topic: str,
        msg_type,
        time_to_wait=-1
    ):
        """
        Wait for the next incoming message.
        :param msg_type: message type
        :param node: node to initialize the subscription on
        :param topic: topic name to wait for message
        :time_to_wait: seconds to wait before returning
        :return (True, msg) if a message was successfully received, (False, ()) if message
            could not be obtained or shutdown was triggered asynchronously on the context.
        """
        context = node.context
        wait_set = _rclpy.WaitSet(1, 1, 0, 0, 0, 0, context.handle)
        wait_set.clear_entities()

        sub = node.create_subscription(msg_type, topic, lambda _: None, 1)
        wait_set.add_subscription(sub.handle)
        sigint_gc = SignalHandlerGuardCondition(context=context)
        wait_set.add_guard_condition(sigint_gc.handle)

        timeout_nsec = timeout_sec_to_nsec(time_to_wait)
        wait_set.wait(timeout_nsec)

        subs_ready = wait_set.get_ready_entities('subscription')
        guards_ready = wait_set.get_ready_entities('guard_condition')

        if guards_ready:
            if sigint_gc.handle.pointer in guards_ready:
                return (False, None)

        if subs_ready:
            if sub.handle.pointer in subs_ready:
                msg_info = sub.handle.take_message(sub.msg_type, sub.raw)
                return (True, msg_info[0])

        return (False, None)

    def timer_callback(self):
            _, msgScan = self.wait_for_message('/scan', LaserScan)
            step_time = (self.get_clock().now() - self.t_step).nanoseconds / 1e9
            if step_time > MIN_TIME_BETWEEN_ACTIONS:
                self.t_step = self.get_clock().now()
                if step_time > 2:
                    text = '\r\nTOO BIG STEP TIME: %.2f s' % step_time
                    print(text)
                    self.log_sim_info.write(text+'\r\n')

                # End of Learning
                if self.episode > MAX_EPISODES:
                    # simulation time
                    sim_time = (self.get_clock().now() - self.t_sim_start).nanoseconds / 1e9
                    sim_time_h = sim_time // 3600
                    sim_time_m = ( sim_time - sim_time_h * 3600 ) // 60
                    sim_time_s = sim_time - sim_time_h * 3600 - sim_time_m * 60

                    # real time
                    now_stop = datetime.now()
                    dt_string_stop = now_stop.strftime("%d/%m/%Y %H:%M:%S")
                    real_time_delta = (now_stop - self.now_start).total_seconds()
                    real_time_h = real_time_delta // 3600
                    real_time_m = ( real_time_delta - real_time_h * 3600 ) // 60
                    real_time_s = real_time_delta - real_time_h * 3600 - real_time_m * 60

                    # Log learning session info to file
                    text = '--------------------------------------- \r\n\r\n'
                    text = text + 'MAX EPISODES REACHED(%d), LEARNING FINISHED ==> ' % MAX_EPISODES + dt_string_stop + '\r\n'
                    text = text + 'Simulation time: %d:%d:%d  h/m/s \r\n' % (sim_time_h, sim_time_m, sim_time_s)
                    text = text + 'Real time: %d:%d:%d  h/m/s \r\n' % (real_time_h, real_time_m, real_time_s)
                    print(text)
                    self.log_sim_info.write('\r\n'+text+'\r\n')
                    self.log_sim_params.write(text+'\r\n')
                    # Log data to file
                    saveQTable(LOG_FILE_DIR+'/Qtable.csv', self.Q_table)
                    np.savetxt(LOG_FILE_DIR+'/StateSpace.csv', self.state_space, '%d')
                    np.savetxt(LOG_FILE_DIR+'/steps_per_episode.csv', self.steps_per_episode, delimiter = ' , ')
                    np.savetxt(LOG_FILE_DIR+'/reward_per_episode.csv', self.reward_per_episode, delimiter = ' , ')
                    np.savetxt(LOG_FILE_DIR+'/T_per_episode.csv', self.T_per_episode, delimiter = ' , ')
                    np.savetxt(LOG_FILE_DIR+'/EPSILON_per_episode.csv', self.EPSILON_per_episode, delimiter = ' , ')
                    np.savetxt(LOG_FILE_DIR+'/reward_min_per_episode.csv', self.reward_min_per_episode, delimiter = ' , ')
                    np.savetxt(LOG_FILE_DIR+'/reward_max_per_episode.csv', self.reward_max_per_episode, delimiter = ' , ')
                    np.savetxt(LOG_FILE_DIR+'/reward_avg_per_episode.csv', self.reward_avg_per_episode, delimiter = ' , ')
                    np.savetxt(LOG_FILE_DIR+'/t_per_episode.csv', self.t_per_episode, delimiter = ' , ')

                    # Close files and shut down node
                    self.log_sim_info.close()
                    self.log_sim_params.close()
                    raise SystemExit
                else:
                    ep_time = (self.get_clock().now() - self.t_ep).nanoseconds / 1e9
                    # End of en Episode
                    if self.crash or self.ep_steps >= MAX_STEPS_PER_EPISODE:
                        robotStop(self.velPub)
                        if self.crash:
                            # get crash position
                            _, odomMsg = self.wait_for_message('/odom', Odometry)
                            ( x_crash , y_crash ) = getPosition(odomMsg)
                            theta_crash = degrees(getRotation(odomMsg))

                        self.t_ep = self.get_clock().now()
                        self.reward_min = np.min(self.ep_reward_arr)
                        self.reward_max = np.max(self.ep_reward_arr)
                        self.reward_avg = np.mean(self.ep_reward_arr)
                        now = datetime.now()
                        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                        text = '---------------------------------------\r\n'
                        if self.crash:
                            text = text + '\r\nEpisode %d ==> CRASH {%.2f,%.2f,%.2f}    ' % (self.episode, x_crash, y_crash, theta_crash) + dt_string
                            self.reset.call_async(self.dummy_req)
                        elif self.ep_steps >= MAX_STEPS_PER_EPISODE:
                            text = text + '\r\nEpisode %d ==> MAX STEPS PER EPISODE REACHED {%d}    ' % (self.episode, MAX_STEPS_PER_EPISODE) + dt_string
                        else:
                            text = text + '\r\nEpisode %d ==> UNKNOWN TERMINAL CASE    ' % self.episode + dt_string
                        text = text + '\r\nepisode time: %.2f s (avg step: %.2f s) \r\n' % (ep_time, ep_time / self.ep_steps)
                        text = text + 'episode steps: %d \r\n' % self.ep_steps
                        text = text + 'episode reward: %.2f \r\n' % self.ep_reward
                        text = text + 'episode max | avg | min reward: %.2f | %.2f | %.2f \r\n' % (self.reward_max, self.reward_avg, self.reward_min)
                        if EXPLORATION_FUNCTION == 1:
                            text = text + 'T = %f \r\n' % self.T
                        else:
                            text = text + 'EPSILON = %f \r\n' % self.EPSILON
                        print(text)
                        self.log_sim_info.write('\r\n'+text)

                        self.steps_per_episode = np.append(self.steps_per_episode, self.ep_steps)
                        self.reward_per_episode = np.append(self.reward_per_episode, self.ep_reward)
                        self.T_per_episode = np.append(self.T_per_episode, self.T)
                        self.EPSILON_per_episode = np.append(self.EPSILON_per_episode, self.EPSILON)
                        self.t_per_episode = np.append(self.t_per_episode, ep_time)
                        self.reward_min_per_episode = np.append(self.reward_min_per_episode, self.reward_min)
                        self.reward_max_per_episode = np.append(self.reward_max_per_episode, self.reward_max)
                        self.reward_avg_per_episode = np.append(self.reward_avg_per_episode, self.reward_avg)
                        self.ep_reward_arr = np.array([])
                        self.ep_steps = 0
                        self.ep_reward = 0
                        self.CUMULATIVE_REWARD = 0
                        self.crash = 0
                        self.robot_in_pos = False
                        self.first_action_taken = False
                        if self.T > T_MIN:
                            self.T = T_GRAD * self.T
                        if self.EPSILON > EPSILON_MIN:
                            self.EPSILON = EPSILON_GRAD * self.EPSILON
                        self.episode = self.episode + 1
                    else:
                        self.ep_steps = self.ep_steps + 1
                        # Initial position
                        if not self.robot_in_pos:
                            robotStop(self.velPub)
                            self.ep_steps = self.ep_steps - 1
                            self.first_action_taken = False
                            # init pos
                            if RANDOM_INIT_POS:
                                ( x_init , y_init , theta_init ) = robotSetRandomPos(self.setPosPub)
                            else:
                                ( x_init , y_init , theta_init ) = robotSetPos(self.setPosPub, X_INIT, Y_INIT, THETA_INIT)

                            _, odomMsg = self.wait_for_message('/odom', Odometry)
                            ( x , y ) = getPosition(odomMsg)
                            theta = degrees(getRotation(odomMsg))
                            # check init pos
                            if abs(x-x_init) < 0.01 and abs(y-y_init) < 0.01 and abs(theta-theta_init) < 1:
                                self.robot_in_pos = True
                                #sleep(2)
                            else:
                                self.robot_in_pos = False
                        # First acion
                        elif not self.first_action_taken:
                            ( lidar, angles ) = lidarScan(msgScan)
                            ( state_ind, x1, x2, x3 , x4 , x5, x6, x7 ) = scanDiscretization(self.state_space, lidar)
                            self.crash = checkCrash(lidar)

                            if EXPLORATION_FUNCTION == 1 :
                                ( self.action, status_strat ) = softMaxSelection(self.Q_table, state_ind, self.actions, self.T)
                            else:
                                ( self.action, status_strat ) = epsiloGreedyExploration(self.Q_table, state_ind, self.actions, self.T)

                            status_rda = robotDoAction(self.velPub, self.action)

                            self.prev_lidar = lidar
                            self.prev_action = self.action
                            self.prev_state_ind = state_ind

                            self.first_action_taken = True

                            if not (status_strat == 'softMaxSelection => OK' or status_strat == 'epsiloGreedyExploration => OK'):
                                print('\r\n', status_strat, '\r\n')
                                self.log_sim_info.write('\r\n'+status_strat+'\r\n')

                            if not status_rda == 'robotDoAction => OK':
                                print('\r\n', status_rda, '\r\n')
                                self.log_sim_info.write('\r\n'+status_rda+'\r\n')

                        # Rest of the algorithm
                        else:
                            ( lidar, angles ) = lidarScan(msgScan)
                            ( state_ind, x1, x2, x3 , x4 , x5, x6, x7 ) = scanDiscretization(self.state_space, lidar)
                            self.crash = checkCrash(lidar)

                            # get position
                            _, odomMsg = self.wait_for_message('/odom', Odometry)
                            ( current_x , current_y ) = getPosition(odomMsg)
                            # radius caculated by norm of  and goal position
                            # MAX_RADIUS = 
                            
                            # ( reward, terminal_state ) = getReward(self.action, self.prev_action, lidar, self.prev_lidar, self.crash)
                            ( reward, terminal_state) = getReward(lidar, self.prev_lidar, self.crash, )

                            self.CUMULATIVE_REWARD += reward
                            ( self.Q_table, status_uqt ) = updateQTable(self.Q_table, self.prev_state_ind, self.action, reward, state_ind, self.alpha, self.gamma)

                            if EXPLORATION_FUNCTION == 1:
                                ( self.action, status_strat ) = softMaxSelection(self.Q_table, state_ind, self.actions, self.T)
                            else:
                                ( self.action, status_strat ) = epsiloGreedyExploration(self.Q_table, state_ind, self.actions, self.T)

                            status_rda = robotDoAction(self.velPub, self.action)

                            if not status_uqt == 'updateQTable => OK':
                                print('\r\n', status_uqt, '\r\n')
                                self.log_sim_info.write('\r\n'+status_uqt+'\r\n')
                            if not (status_strat == 'softMaxSelection => OK' or status_strat == 'epsiloGreedyExploration => OK'):
                                print('\r\n', status_strat, '\r\n')
                                self.log_sim_info.write('\r\n'+status_strat+'\r\n')
                            if not status_rda == 'robotDoAction => OK':
                                print('\r\n', status_rda, '\r\n')
                                self.log_sim_info.write('\r\n'+status_rda+'\r\n')

                            self.ep_reward = self.ep_reward + reward
                            self.ep_reward_arr = np.append(self.ep_reward_arr, reward)
                            self.prev_lidar = lidar
                            self.prev_action = self.action
                            self.prev_state_ind = state_ind



def main(args=None):
    rclpy.init(args=args)

    movebase_publisher = LearningNode()
    try:
        rclpy.spin(movebase_publisher)
    except SystemExit:                 # <--- process the exception 
        rclpy.logging.get_logger("End of learning").info('Done')
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    movebase_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
