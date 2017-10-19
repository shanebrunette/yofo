#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

"""

"""

import tf
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from controller_manager_msgs.srv import ListControllers


CONTROLLERS = ['arm_trajectory_controller']
RUNNING_STATE = 'running'

ARM_JOINTS = ['arm_lift_joint', 'arm_flex_joint',
        'arm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint']
ARM_TOPIC = '/hsrb/arm_trajectory_controller/command'
CONTROLLER_TOPIC = '/hsrb/controller_manager/list_controllers'


WAIT_DELAY = 0.1

class BagGrabber(object):
    def __init__(self):
        rospy.init_node('stand_tall', anonymous=True)

        # Setup the publishers
        self.arm_pub = rospy.Publisher(ARM_TOPIC, JointTrajectory, queue_size=10)


        # Wait until controllers are ready
        self._wait_for_controllers()

    def _wait_for_controllers(self):
        connections = [
                self.arm_pub.get_num_connections
                ]

        # Wait until all controllers are connected
        while 0 in [f() for f in connections]:
            rospy.sleep(WAIT_DELAY)

        rospy.loginfo('All controllers are connected')

        # Wait for controller service
        rospy.wait_for_service(CONTROLLER_TOPIC)
        get_controllers = rospy.ServiceProxy(CONTROLLER_TOPIC, ListControllers)

        rospy.loginfo('Controller manager is running')

        # Wait until all controllers are running
        running = len(CONTROLLERS) * [False]
        while False in running:
            rospy.sleep(WAIT_DELAY)
            for c in get_controllers().controller:
                if c.name in CONTROLLERS:
                    i = CONTROLLERS.index(c.name)
                    running[i] = c.state == RUNNING_STATE

        rospy.loginfo('All controllers are running')




    def move_arm(self, arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll):
        t = JointTrajectory()
        t.joint_names = ARM_JOINTS
        p = JointTrajectoryPoint()
        p.positions = [arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll]
        p.velocities = [0, 0, 0, 0, 0]
        p.time_from_start = rospy.Time(3)
        t.points = [p]

        self.arm_pub.publish(t)


if __name__ == '__main__':
    try:
        b = BagGrabber()
        b.move_arm(2, -3, 0, 0, 0)
        # rospy.spin()
    except rospy.ROSInterruptException:
        pass