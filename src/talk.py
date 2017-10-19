import rospy
from tmc_msgs.msg import Voice


class Talk:
	def __init__(self):
		rospy.init_node('talk_test')
		self.pub = rospy.Publisher('talk_request', Voice, queue_size = 10)

	def found_human(self):
		self.pub.publish(False, False, 1, 'i see you')
		self.pub.publish(False, False, 1, 'you must die')

	def following_human(self):
		self.pub.publish(False, False, 1, 'kill! kill! kill!')

	def get_human(self):
		self.pub.publish(False, False, 1, 'hey! come back here.')
