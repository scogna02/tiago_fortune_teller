#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import threading
import time
from datetime import datetime
import os

class TiagoCameraManager:
    def __init__(self, save_path="data/captured_images/", camera_id=0):
        """
        Initialize the PC Camera Manager for real-time image capture.
        
        Args:
            save_path (str): Directory to save captured images
            camera_id (int): Camera device ID (0 for default webcam)
        """
        try:
            rospy.init_node('tiago_camera_manager', anonymous=True)
        except rospy.exceptions.ROSException:
            # Node already initialized, continue
            pass
        
        self.save_path = save_path
        self.camera_id = camera_id
        self.ensure_save_directory()
        
        # Current image data
        self.current_image = None
        self.image_lock = threading.Lock()
        self.last_image_time = None
        
        # Camera capture object
        self.cap = None
        self.camera_ready = False
        
        # Image capture settings
        self.auto_save = False
        self.capture_counter = 0
        
        # Start camera
        self.initialize_camera()
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self.capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Wait for camera to be ready
        self.wait_for_camera()
        
        rospy.loginfo("PC Camera Manager initialized and ready")
    
    def ensure_save_directory(self):
        """Ensure the save directory exists."""
        os.makedirs(self.save_path, exist_ok=True)
        rospy.loginfo(f"Image save directory: {self.save_path}")
    
    def initialize_camera(self):
        """Initialize the PC camera."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if self.cap.isOpened():
                # Set camera properties for better quality
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                rospy.loginfo(f"PC Camera {self.camera_id} initialized successfully")
                return True
            else:
                rospy.logerr(f"Failed to open camera {self.camera_id}")
                return False
                
        except Exception as e:
            rospy.logerr(f"Error initializing camera: {e}")
            return False
    
    def capture_loop(self):
        """Continuous capture loop running in separate thread."""
        while not rospy.is_shutdown():
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    
                    if ret:
                        with self.image_lock:
                            self.current_image = frame.copy()
                            self.last_image_time = rospy.Time.now()
                            
                            if not self.camera_ready:
                                self.camera_ready = True
                                rospy.loginfo("📷 First image captured from PC camera!")
                            
                            # Auto-save if enabled
                            if self.auto_save:
                                self.save_current_image()
                    else:
                        rospy.logwarn("Failed to capture frame from camera")
                        time.sleep(0.1)
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                rospy.logerr(f"Error in capture loop: {e}")
                time.sleep(1)
    
    def wait_for_camera(self, timeout=10):
        """
        Wait for camera to start capturing images.
        """
        rospy.loginfo("Waiting for PC camera to be ready...")
        start_time = time.time()
        
        while not self.camera_ready and (time.time() - start_time) < timeout:
            time.sleep(0.1)
            
        if self.camera_ready:
            rospy.loginfo("✅ PC camera is ready!")
        else:
            rospy.logwarn(f"⚠️  Camera not ready after {timeout} seconds. Using fallback mode.")
    
    def get_current_image(self):
        """
        Get the most recent image from PC camera.
        """
        with self.image_lock:
            if self.current_image is not None:
                return self.current_image.copy()
            else:
                return None
    
    def capture_image(self, filename=None, save=True):
        """
        Capture the current image from PC camera.
        """
        current_img = self.get_current_image()
        
        if current_img is not None:
            rospy.loginfo("📸 Image captured from PC camera")
            
            if save:
                if filename is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"pc_camera_capture_{timestamp}_{self.capture_counter:04d}.jpg"
                
                filepath = os.path.join(self.save_path, filename)
                cv2.imwrite(filepath, current_img)
                rospy.loginfo(f"💾 Image saved: {filepath}")
                self.capture_counter += 1
            
            return current_img
        else:
            rospy.logwarn("⚠️  No image available from camera, using fallback")
            return self._create_fallback_image()
    
    def _create_fallback_image(self):
        """Create a fallback image when camera is not available."""
        # Create a simple test pattern
        fallback = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some visual elements
        cv2.rectangle(fallback, (50, 50), (590, 430), (100, 100, 100), 2)
        cv2.putText(fallback, "PC CAMERA NOT AVAILABLE", (120, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(fallback, "Using Fallback Image", (180, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(fallback, f"Time: {datetime.now().strftime('%H:%M:%S')}", 
                   (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        return fallback
    
    def capture_for_face_recognition(self):
        """
        Capture an image specifically formatted for face recognition.
        
        """
        bgr_image = self.capture_image(save=False)
        
        if bgr_image is not None:
            # Convert BGR (OpenCV) to RGB (face_recognition)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            rospy.loginfo("📸 Image captured and converted for face recognition")
            return rgb_image
        else:
            rospy.logwarn("Failed to capture image for face recognition")
            return None
    
    def save_current_image(self, filename=None):
        """
        Save the current image to disk.
        
        """
        current_img = self.get_current_image()
        
        if current_img is not None:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"pc_camera_auto_{timestamp}_{self.capture_counter:04d}.jpg"
            
            filepath = os.path.join(self.save_path, filename)
            success = cv2.imwrite(filepath, current_img)
            
            if success:
                rospy.loginfo(f"💾 Auto-saved image: {filepath}")
                self.capture_counter += 1
                return filepath
            else:
                rospy.logerr(f"❌ Failed to save image: {filepath}")
                return None
        else:
            rospy.logwarn("⚠️  No image to save")
            return None
    
    def enable_auto_save(self):
        """Enable automatic saving of every captured frame."""
        self.auto_save = True
        rospy.loginfo("🔄 Auto-save enabled - all images will be saved")
    
    def disable_auto_save(self):
        """Disable automatic saving."""
        self.auto_save = False
        rospy.loginfo("⏹️  Auto-save disabled")
    
    def is_camera_ready(self):
        """
        Check if camera is ready and capturing images.
        """
        if not self.camera_ready:
            return False
            
        with self.image_lock:
            if self.last_image_time is None:
                return False
            
            # Check if we've received an image in the last 5 seconds
            time_since_last = (rospy.Time.now() - self.last_image_time).to_sec()
            return time_since_last < 5.0
    
    def get_image_stats(self):
        """
        Get statistics about the current image.
        """
        current_img = self.get_current_image()
        
        if current_img is not None:
            height, width, channels = current_img.shape
            return {
                'width': width,
                'height': height,
                'channels': channels,
                'total_pixels': width * height,
                'size_mb': (current_img.nbytes / 1024 / 1024),
                'last_capture_time': self.last_image_time.to_sec() if self.last_image_time else None,
                'camera_type': 'PC_CAMERA',
                'camera_id': self.camera_id
            }
        else:
            return None
    
    def display_current_image(self, window_name="PC Camera", wait_key=True):
        """
        Display the current image in a window (for debugging).
        """
        current_img = self.get_current_image()
        
        if current_img is not None:
            cv2.imshow(window_name, current_img)
            if wait_key:
                cv2.waitKey(1)  # Non-blocking wait
        else:
            rospy.logwarn("No image to display")
    
    def shutdown(self):
        """Clean shutdown of the camera manager."""
        rospy.loginfo("Shutting down PC Camera Manager")
        self.auto_save = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()

# Test the PC camera
if __name__ == '__main__':
    try:
        rospy.init_node('pc_camera_test', anonymous=True)
        
        print("🧪 Testing PC Camera...")
        camera = TiagoCameraManager()
        
        # Test basic functionality
        if camera.is_camera_ready():
            print("✅ PC Camera is ready!")
            
            # Capture test image
            image = camera.capture_image(save=True)
            if image is not None:
                print(f"✅ Test image captured: {image.shape}")
                
                # Show live view for 10 seconds
                print("📺 Showing live view for 10 seconds...")
                start_time = time.time()
                while time.time() - start_time < 10:
                    camera.display_current_image("PC Camera Test")
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
            else:
                print("❌ Failed to capture test image")
        else:
            print("❌ PC Camera not ready")
            
    except KeyboardInterrupt:
        print("Test interrupted")
    finally:
        cv2.destroyAllWindows()