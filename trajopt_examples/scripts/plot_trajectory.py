#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot trajectory of joint position, velocity, accelearation
n"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from tesseract_msgs.msg import Trajectory
import rospy


class PlotTrajoptTrajectory:
    def __init__(self):
        self.sub_traj = rospy.Subscriber('/trajopt/display_tesseract_trajectory', Trajectory, self._trajectory_callback)

    def _trajectory_callback(self, msg):
        joint_names = msg.joint_trajectory.joint_names
        positions_array = np.array([p.positions for p in msg.joint_trajectory.points]).transpose()
        time_array = np.array([p.time_from_start.to_sec() for p in msg.joint_trajectory.points])

        print('=== joint_names ===')
        print(joint_names)
        print('=== time_array ===')
        print(time_array)
        # print('=== positions_array ===')
        # print(positions_array)

        ip = interp1d(time_array, np.concatenate((positions_array, time_array[None])))

        step_list = np.linspace(ip.x[0], ip.x[-1], num=1000, endpoint=True)
        prev_step_list = np.concatenate(([step_list[0]], step_list[:-1]))
        dt = step_list[1] - step_list[0]

        plot_joint_names = [n for n in joint_names if ('RARM' in n)] # for NexTage
        # plot_joint_names = joint_names
        plt.figure(figsize=(10, 10))
        for name in plot_joint_names:
            current_position = ip(step_list)[joint_names.index(name)]
            prev_position = ip(prev_step_list)[joint_names.index(name)]
            velocity = ((current_position - prev_position) / dt)
            plt.subplot(2, 1, 1)
            plt.plot(step_list, current_position, '-', label=name+'/position', linewidth=4)
            plt.subplot(2, 1, 2)
            plt.plot(step_list, velocity, '-', label=name+'/velocity', linewidth=4)


        plt.subplot(2, 1, 1)
        plt.title('joint position [rad]')
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.title('joint velocity [rad/sec]')
        plt.grid()
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        plt.subplots_adjust(right=0.7)
        plt.show()


if __name__ == '__main__':
    rospy.init_node('apply_context_to_label_probablity')
    PlotTrajoptTrajectory()
    rospy.spin()
