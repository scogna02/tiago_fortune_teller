#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from std_srvs.srv import Empty, EmptyResponse
import threading
import time

class TiagoController:
    def __init__(self):
        """Initialize the Tiago controller with ROS publishers and interactive input capability."""
        rospy.init_node('tiago_fortune_teller', anonymous=True)
        
        # Publishers for robot actions
        self.speech_pub = rospy.Publisher('/tiago/speech', String, queue_size=10)
        self.gesture_pub = rospy.Publisher('/tiago/gesture', String, queue_size=10)
        
        # Interactive input system
        self.user_input = None
        self.waiting_for_input = False
        self.input_lock = threading.Lock()
        
        # Subscribe to user input topic
        self.input_sub = rospy.Subscriber('/tiago/user_input', String, self.input_callback)
        
        # Service for signaling input is needed
        self.input_service = rospy.Service('/tiago/request_input', Empty, self.request_input_callback)
        
        rospy.loginfo("Tiago Controller initialized")
        time.sleep(1)  # Give time for connections to establish

    def input_callback(self, msg):
        """Callback for receiving user input from terminal."""
        with self.input_lock:
            if self.waiting_for_input:
                self.user_input = msg.data.strip()
                self.waiting_for_input = False
                rospy.loginfo(f"Received user input: {self.user_input}")

    def request_input_callback(self, req):
        """Service callback to signal that input is needed."""
        return EmptyResponse()

    def say(self, text):
        """Make Tiago speak the given text."""
        rospy.loginfo(f"Tiago says: {text}")
        print(f"🤖 Tiago: {text}")
        
        # Publish to ROS topic (for real robot integration)
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)
        
        # Simulate speech timing
        time.sleep(len(text) * 0.05 + 1)

    def gesture(self, gesture_type):
        """Make Tiago perform a gesture."""
        rospy.loginfo(f"Tiago performs gesture: {gesture_type}")
        print(f"🤖 *Tiago {gesture_type}*")
        
        # Publish to ROS topic (for real robot integration)
        msg = String()
        msg.data = gesture_type
        self.gesture_pub.publish(msg)
        
        time.sleep(2)  # Gesture duration

    def ask_question(self, question, timeout=30):
        """
        Ask the user a question and wait for input via ROS topic.
        
        Args:
            question (str): The question to ask
            timeout (int): Maximum time to wait for response in seconds
            
        Returns:
            str: User's response or None if timeout
        """
        self.say(question)
        
        with self.input_lock:
            self.user_input = None
            self.waiting_for_input = True
        
        # Wait for input with timeout
        start_time = time.time()
        while self.waiting_for_input and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        with self.input_lock:
            if self.user_input is not None:
                return self.user_input
            else:
                self.waiting_for_input = False
                return None

    def capture_image(self):
        """Capture an image (placeholder implementation)."""
        import numpy as np
        rospy.loginfo("Capturing image...")
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def shutdown(self):
        """Clean shutdown of the controller."""
        rospy.loginfo("Shutting down Tiago Controller")
        rospy.signal_shutdown("Tiago Controller shutdown")