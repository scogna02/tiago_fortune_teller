#!/usr/bin/env python3

import rospy
import actionlib
from std_msgs.msg import String
from std_srvs.srv import Empty, EmptyResponse
from pal_interaction_msgs.msg import TtsAction, TtsGoal
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal
import subprocess
import threading
import time
import cv2
from datetime import datetime

class TiagoController:
    def __init__(self):
        """Initialize the Tiago controller with both ROS action servers and interactive input."""
        rospy.init_node('tiago_fortune_teller', anonymous=True)
        
        # Action clients for real Tiago functionality
        self.setup_action_clients()
        
        # Publishers for custom topics (backup/logging)
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

        self.setup_free_tts()
        
        rospy.loginfo("Tiago Controller initialized")
        time.sleep(1)  # Give time for connections to establish

    def setup_free_tts(self):

        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            # Configure pyttsx3 settings
            self.tts_engine.setProperty('rate', 150)    # Speed of speech
            self.tts_engine.setProperty('volume', 0.8)  # Volume level (0.0 to 1.0)
            self.pyttsx3_available = True
            rospy.loginfo("✅ pyttsx3 TTS available")
        except ImportError:
            self.pyttsx3_available = False
            self.tts_engine = None
            rospy.logwarn("⚠️ pyttsx3 not available (install with: pip install pyttsx3)")


    def _say_with_pyttsx3(self, text):
        """Use pyttsx3 for text-to-speech."""
        try:
            if self.tts_engine is None:
                return False
                
            # Get available voices and select a female voice if possible
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to find a female voice
                for voice in voices:
                    if 'female' in voice.name.lower() or 'woman' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            time.sleep(0.4)
            
            return True
            
        except Exception as e:
            rospy.logerr(f"❌ pyttsx3 TTS error: {e}")
            return False

    def setup_action_clients(self, timeout=10):
        """Setup action clients for TTS and gestures."""
        # TTS client
        self.tts_client = actionlib.SimpleActionClient('/tts_to_soundplay', TtsAction)
        rospy.loginfo("Waiting for /tts_to_soundplay action server...")
        
        """if self.tts_client.wait_for_server(rospy.Duration(timeout)):
            self.tts_available = True
            rospy.loginfo("✅ TTS action server connected")
        else:
            rospy.logwarn("⚠️ TTS action server not available, using fallback")
            self.tts_available = False"""
        
        # Gesture client
        self.gesture_client = actionlib.SimpleActionClient('/play_motion', PlayMotionAction)
        rospy.loginfo("Waiting for /play_motion action server...")
        
        if self.gesture_client.wait_for_server(rospy.Duration(timeout)):
            self.gesture_available = True
            rospy.loginfo("✅ PlayMotion action server connected")
        else:
            rospy.logwarn("⚠️ PlayMotion action server not available, gestures will be simulated")
            self.gesture_available = False

        self.tts_available = False
        #self.gesture_available = False

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

    def say(self, text, language='en_GB'):
        """Make Tiago speak using TTS action server or fallback."""
        rospy.loginfo(f"Tiago says: {text}")
        #print(f"🤖 Tiago: {text}")
        text = str(text).strip()
        
        # Try using real TTS action server first
        if self.tts_available:
            try:
                goal = TtsGoal()
                goal.rawtext.text = text
                goal.rawtext.lang_id = language

                self.tts_client.send_goal(goal)
               
                # Wait for result with timeout
                result = self.tts_client.wait_for_result(rospy.Duration(10))
                if not result:  
                    rospy.logwarn("⚠️ TTS timeout")
                    
            except Exception as e:
                rospy.logerr(f"❌ TTS error: {e}")
        
        else:
            self._try_free_tts(text, language)
        
        # Publish to custom topic (for logging/monitoring)
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)
        
        """# Simulate speech timing if no real TTS
        if not self.tts_available:
            time.sleep(0.1)"""
    
    def _try_free_tts(self, text, language='en_GB'):
        """Try free TTS alternatives in order of preference."""
        tts_success = False
        
        # Try pyttsx3 if espeak failed or not available
        if not tts_success and self.pyttsx3_available:
            tts_success = self._say_with_pyttsx3(text)
        
        # Fallback to simulation if all TTS methods fail
        if not tts_success:
            rospy.logwarn("⚠️ All TTS methods failed, using timing simulation")
            # Estimate speech duration (average 150 words per minute)
            words = len(text.split())
            duration = max(1.0, words / 2.5)  # Rough estimation
            time.sleep(duration)

    def gesture(self, gesture_type):
        """Make Tiago perform a gesture using PlayMotion action server."""
        rospy.loginfo(f"Tiago performs gesture: {gesture_type}")
        #print(f"🤖 *Tiago {gesture_type}*")
        
        # Map custom gesture names to actual Tiago motion names
        # Based on official TIAGo motions available in Gazebo simulation
        motion_mapping = {
            # Basic fortune teller gestures
            'wave': 'wave',                          # Welcome greeting
            'mystical_pose': 'offer',                # Open arms in mystical offering pose
            'thoughtful_pose': 'prepare_grasp',      # Contemplative hand position
            'crystal_ball_gaze': 'inspect_surroundings',  # Looking around mystically
            'meditation': 'home',                    # Neutral meditative position
            'mystical_wave': 'wave',                 # Same as wave but contextual
            'revelation': 'open',                    # Open hands for revelation
            'bow': 'home',                           # Return to neutral position
            'recognition': 'point',                  # Point to acknowledge user
            'curious': 'head_tour',                  # Head movement for curiosity
            'counting': 'pinch_hand',                # Pinch fingers for counting
            'look_at_user': 'home',             # Look around to find user
            'nod': 'home',                           # Return to neutral (no nod available)
            
            # Extended gesture vocabulary for richer interaction
            'welcome': 'wave',                       # Welcome gesture
            'offering': 'offer',                     # Offering hands gesture
            'open_arms': 'open',                     # Open arms wide
            'embrace_cosmos': 'unfold_arm',          # Unfold arms to embrace
            'shake_hands': 'shake_hands',            # Handshake motion
            'thumbs_up': 'thumb_up_hand',            # Positive gesture
            'reach_high': 'reach_max',               # Reach for the stars
            'reach_low': 'reach_floor',              # Grounding gesture
            'close_meditation': 'close',             # Close hands in meditation
            'prepare_magic': 'prepare_grasp',        # Prepare for magical gesture
            'pick_energy': 'pick_from_floor',        # Pick up cosmic energy
            'pregrasp_wisdom': 'pregrasp_weight',    # Grasp wisdom from the air
        }
        
        motion_name = motion_mapping.get(gesture_type, 'home')
        
        # Try using real PlayMotion action server
        if self.gesture_available:
            try:
                goal = PlayMotionGoal()
                goal.motion_name = motion_name
                goal.skip_planning = False
                goal.priority = 0
                
                self.gesture_client.send_goal(goal)
                result = self.gesture_client.wait_for_result(rospy.Duration(20))
                
                if result:
                    rospy.loginfo(f"✅ Gesture '{motion_name}' completed successfully")
                else:
                    rospy.logwarn(f"⚠️ Gesture '{motion_name}' timeout")
                    
            except Exception as e:
                rospy.logerr(f"❌ Gesture error: {e}")
        else:
            # Simulate gesture timing
            time.sleep(2)
        
        # Publish to custom topic (for logging/monitoring)
        msg = String()
        msg.data = gesture_type
        self.gesture_pub.publish(msg)

    def ask_question(self, question, timeout=30):
        """Ask the user a question and wait for input via ROS topic."""
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
            from tiago_interface.pc_camera_manager import TiagoCameraManager
            self.camera_manager = TiagoCameraManager()
            rospy.loginfo("📷 PC Camera manager initialized")
        except Exception as e:
            rospy.logerr(f"Failed to initialize PC camera manager: {e}")
            self.camera_manager = None

    def capture_image(self, save=True, for_face_recognition=True):
        """Capture an image using PC camera."""
        if self.camera_manager is None:
            self.initialize_camera()
        
        rospy.loginfo("📸 Capturing image using PC camera...")
        
        if self.camera_manager and self.camera_manager.is_camera_ready():
            if for_face_recognition:
                image = self.camera_manager.capture_for_face_recognition()
            else:
                image = self.camera_manager.capture_image(save=save)
        else:
            rospy.logwarn("⚠️ PC Camera not ready, using fallback image")
            image = self._create_fallback_image()
        
        if image is not None:
            rospy.loginfo(f"✅ Image captured successfully: {image.shape}")
        else:
            rospy.logwarn("⚠️ Using fallback image generation")
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
        """Move Tiago's head to look at the user."""
        rospy.loginfo("👁️ Positioning to look at user")
        self.gesture("look_at_user")
        time.sleep(2)
    
    def prepare_for_photo(self):
        """Prepare for taking a photo."""
        self.say("Let me position myself to get a good look at you...")
        self.look_at_user()
        time.sleep(1)
        self.say("Perfect! Look into your camera and hold still for just a moment...")
        time.sleep(0.5)

    def check_available_motions(self):
        """Check what motions are actually available in the current Tiago simulation."""
        try:
            import rospy
            # Get all motion names from parameter server
            motion_names = []
            
            # Try to get motion parameters
            param_names = rospy.get_param_names()
            for param in param_names:
                if '/play_motion/motions/' in param and '/meta/name' in param:
                    motion_name = param.split('/')[3]  # Extract motion name
                    if motion_name not in motion_names:
                        motion_names.append(motion_name)
            
            rospy.loginfo(f"📋 Available motions ({len(motion_names)}):")
            for motion in sorted(motion_names):
                rospy.loginfo(f"  ✅ {motion}")
                
            return motion_names
            
        except Exception as e:
            rospy.logwarn(f"Could not check available motions: {e}")
            return []

    def check_available_services(self):
        """Debug method to check what action servers are available."""
        rospy.loginfo("=== Checking Available Action Servers ===")
        
        # Check TTS
        if self.tts_available:
            rospy.loginfo("✅ TTS (/tts) - Available")
        else:
            rospy.loginfo("❌ TTS (/tts) - Not Available")
            
        # Check PlayMotion
        if self.gesture_available:
            rospy.loginfo("✅ PlayMotion (/play_motion) - Available")
            # Also check available motions
            self.check_available_motions()
        else:
            rospy.loginfo("❌ PlayMotion (/play_motion) - Not Available")
        
        # List all available action servers
        rospy.loginfo("Available Action Servers:")
        topics = rospy.get_published_topics()
        action_servers = set()
        
        for topic, msg_type in topics:
            if topic.endswith('/goal'):
                server_name = topic[:-5]  # Remove '/goal'
                action_servers.add(server_name)
        
        for server in sorted(action_servers):
            rospy.loginfo(f"  {server}")

    def shutdown(self):
        """Clean shutdown of the controller."""
        rospy.loginfo("Shutting down Tiago Controller")
        if self.camera_manager:
            self.camera_manager.shutdown()
        rospy.signal_shutdown("Tiago Controller shutdown")