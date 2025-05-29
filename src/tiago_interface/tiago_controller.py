#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from std_srvs.srv import Empty, EmptyResponse
import threading
import time
import cv2
from datetime import datetime

class TiagoController:
    def __init__(self):
        """Initialize the Tiago controller with ROS publishers and PC camera integration."""
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
        
        # Initialize camera manager (lazy loading for PC camera)
        self.camera_manager = None
        
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

    def initialize_camera(self):
        """Initialize the PC camera manager for image capture."""
        try:
            from tiago_interface.pc_camera_manager import TiagoCameraManager  # Import the PC version
            self.camera_manager = TiagoCameraManager()
            rospy.loginfo("📷 PC Camera manager initialized")
        except Exception as e:
            rospy.logerr(f"Failed to initialize PC camera manager: {e}")
            self.camera_manager = None

    def capture_image(self, save=True, for_face_recognition=True):
        """
        Capture an image using PC camera.
        
        Args:
            save (bool): Whether to save the captured image
            for_face_recognition (bool): Whether to format for face recognition
            
        Returns:
            numpy.ndarray: Captured image
        """
        # Initialize camera manager if needed
        if self.camera_manager is None:
            self.initialize_camera()
        
        rospy.loginfo("📸 Capturing image using PC camera...")
        
        if self.camera_manager and self.camera_manager.is_camera_ready():
            if for_face_recognition:
                image = self.camera_manager.capture_for_face_recognition()
            else:
                image = self.camera_manager.capture_image(save=save)
        else:
            rospy.logwarn("⚠️  PC Camera not ready, using fallback image")
            image = self._create_fallback_image()
        
        if image is not None:
            rospy.loginfo(f"✅ Image captured successfully: {image.shape}")
        else:
            rospy.logwarn("⚠️  Using fallback image generation")
            image = self._create_fallback_image()
            
        return image
    
    def _create_fallback_image(self):
        """Create a fallback image when camera is not available."""
        import numpy as np
        
        fallback = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(fallback, (50, 50), (590, 430), (100, 100, 100), 2)
        cv2.putText(fallback, "SIMULATED CAMERA", (180, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(fallback, "Fortune Teller Mode", (190, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(fallback, f"Time: {datetime.now().strftime('%H:%M:%S')}", 
                   (220, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        return fallback
    
    def look_at_user(self):
        """Move Tiago's head to look at the user (simulated for PC camera)."""
        rospy.loginfo("👁️  Positioning to look at user (PC camera mode)")
        self.gesture("look_at_user")
        time.sleep(2)  # Give time for head movement
    
    def prepare_for_photo(self):
        """Prepare for taking a photo with PC camera."""
        self.say("Let me position myself to get a good look at you...")
        self.look_at_user()
        time.sleep(1)
        self.say("Perfect! Look into your camera and hold still for just a moment...")
        time.sleep(0.5)

    def shutdown(self):
        """Clean shutdown of the controller."""
        rospy.loginfo("Shutting down Tiago Controller")
        if self.camera_manager:
            self.camera_manager.shutdown()
        rospy.signal_shutdown("Tiago Controller shutdown")