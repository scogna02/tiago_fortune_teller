import rospy
import actionlib
from pal_interaction_msgs.msg import TtsAction, TtsGoal
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal

class TiagoController:
    def __init__(self, timeout=30):
        if not rospy.core.is_initialized():
            rospy.init_node('tiago_fortune_teller', anonymous=True)
        
        # TTS client with correct action server name
        self.tts_client = actionlib.SimpleActionClient('/tts_to_soundplay', TtsAction)
        rospy.loginfo("Waiting for /tts_to_soundplay action server...")
        
        if not self.tts_client.wait_for_server(rospy.Duration(timeout)):
            rospy.logerr(f"/tts_to_soundplay action server not available after {timeout} seconds")
            self.tts_available = False
            rospy.logwarn("Continuing without TTS functionality")
        else:
            self.tts_available = True
            rospy.loginfo("/tts_to_soundplay action server connected.")
        
        # Gesture client - check if play_motion is available
        self.gesture_client = actionlib.SimpleActionClient('/play_motion', PlayMotionAction)
        rospy.loginfo("Waiting for /play_motion action server...")
        
        if not self.gesture_client.wait_for_server(rospy.Duration(timeout)):
            rospy.logerr(f"/play_motion action server not available after {timeout} seconds")
            
            # List available motion-related topics for debugging
            rospy.loginfo("Available motion-related topics:")
            topics = rospy.get_published_topics()
            for topic, msg_type in topics:
                if 'motion' in topic.lower() or 'gesture' in topic.lower():
                    rospy.loginfo(f"  {topic} ({msg_type})")
            
            self.gesture_available = False
            rospy.logwarn("Continuing without gesture functionality")
        else:
            self.gesture_available = True
            rospy.loginfo("/play_motion action server connected.")

    def say(self, text, language='en_GB'):
        if not self.tts_available:
            rospy.logwarn(f"TTS not available. Would say: {text}")
            return
        
        try:
            goal = TtsGoal()
            goal.rawtext.text = text
            goal.rawtext.lang_id = language
            self.tts_client.send_goal(goal)
            result = self.tts_client.wait_for_result(rospy.Duration(10))
            
            if result:
                rospy.loginfo(f"Said: {text}")
            else:
                rospy.logwarn(f"TTS timeout for: {text}")
        except Exception as e:
            rospy.logerr(f"TTS error: {e}")

    def gesture(self, motion_name):
        if not self.gesture_available:
            rospy.logwarn(f"Gestures not available. Would perform: {motion_name}")
            return
        
        try:
            goal = PlayMotionGoal()
            goal.motion_name = motion_name
            goal.skip_planning = False
            goal.priority = 0
            self.gesture_client.send_goal(goal)
            result = self.gesture_client.wait_for_result(rospy.Duration(15))
            
            if result:
                rospy.loginfo(f"Performed gesture: {motion_name}")
            else:
                rospy.logwarn(f"Gesture timeout for: {motion_name}")
        except Exception as e:
            rospy.logerr(f"Gesture error: {e}")

    def check_available_services(self):
        """Debug method to check what's available"""
        rospy.loginfo("=== Available Action Servers ===")
        topics = rospy.get_published_topics()
        
        # Group topics by action server
        action_servers = {}
        for topic, msg_type in topics:
            if topic.endswith(('/goal', '/status', '/result', '/feedback', '/cancel')):
                base_name = topic.rsplit('/', 1)[0]
                if base_name not in action_servers:
                    action_servers[base_name] = []
                action_servers[base_name].append(topic.split('/')[-1])
        
        for server, endpoints in action_servers.items():
            rospy.loginfo(f"Action Server: {server}")
            rospy.loginfo(f"  Endpoints: {sorted(endpoints)}")
        
        return action_servers