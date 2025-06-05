#!/usr/bin/env python3

"""
Simple camera test script for Tiago Fortune Teller.
This script tests the camera capture functionality and face recognition.
"""

import rospy
import cv2
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from camera_manager import TiagoCameraManager, CameraTestNode
from enhanced_face_recognizer import FaceRecognizer

def test_camera_basic():
    """Test basic camera functionality."""
    print("🧪 Testing Basic Camera Functionality")
    print("=" * 50)
    
    try:
        rospy.init_node('camera_test_basic', anonymous=True)
        camera = TiagoCameraManager()
        
        print("⏳ Waiting for camera to be ready...")
        if camera.is_camera_ready():
            print("✅ Camera is ready!")
        else:
            print("❌ Camera not ready - using fallback mode")
        
        # Capture a test image
        print("📸 Capturing test image...")
        image = camera.capture_image(save=True)
        
        if image is not None:
            print(f"✅ Image captured successfully: {image.shape}")
            
            # Display image statistics
            stats = camera.get_image_stats()
            if stats:
                print(f"📊 Image Stats: {stats}")
                
            print("💾 Image saved to data/captured_images/")
        else:
            print("❌ Failed to capture image")
            
    except Exception as e:
        print(f"❌ Error in basic camera test: {e}")

def test_face_recognition():
    """Test face recognition with camera."""
    print("\n🧪 Testing Face Recognition")
    print("=" * 50)
    
    try:
        camera = TiagoCameraManager()
        face_rec = FaceRecognizer()
        
        # Capture image for face recognition
        print("📸 Capturing image for face recognition...")
        rgb_image = camera.capture_for_face_recognition()
        
        if rgb_image is not None:
            print("✅ RGB image captured for face recognition")
            
            # Test face recognition
            print("👤 Detecting faces...")
            results = face_rec.recognize_face(rgb_image)
            
            if results:
                print(f"✅ Found {len(results)} face(s)!")
                for i, result in enumerate(results):
                    print(f"  Face {i+1}: {result['person_id']} (confidence: {result['confidence']:.2f})")
            else:
                print("ℹ️  No faces detected in image")
                
            # Visualize results
            if results:
                annotated_image = face_rec.visualize_recognition_results(rgb_image, results)
                cv2.imshow("Face Recognition Results", annotated_image)
                print("👁️  Displaying annotated image (press any key to close)")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        else:
            print("❌ Failed to capture RGB image")
            
    except Exception as e:
        print(f"❌ Error in face recognition test: {e}")

def test_interactive_capture():
    """Interactive test for camera capture."""
    print("\n🎮 Interactive Camera Test")
    print("=" * 50)
    print("Instructions:")
    print("- Press SPACE to capture image")
    print("- Press 'f' to test face recognition")
    print("- Press 'q' to quit")
    print()
    
    try:
        camera = TiagoCameraManager()
        face_rec = FaceRecognizer()
        
        capture_count = 0
        
        while True:
            # Get current image for display
            current_image = camera.get_current_image()
            
            if current_image is not None:
                # Add text overlay
                display_image = current_image.copy()
                cv2.putText(display_image, f"Captures: {capture_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_image, "SPACE: Capture, F: Face Recognition, Q: Quit", 
                           (10, display_image.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow("Tiago Camera Live View", display_image)
            else:
                # Show fallback message
                fallback = camera._create_fallback_image()
                cv2.imshow("Tiago Camera Live View", fallback)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                # Capture image
                image = camera.capture_image(save=True)
                if image is not None:
                    capture_count += 1
                    print(f"📸 Captured image #{capture_count}")
                else:
                    print("❌ Failed to capture image")
            elif key == ord('f'):
                # Test face recognition
                rgb_image = camera.capture_for_face_recognition()
                if rgb_image is not None:
                    results = face_rec.recognize_face(rgb_image)
                    if results:
                        print(f"👤 Found {len(results)} face(s):")
                        for result in results:
                            print(f"  - {result['person_id']} (confidence: {result['confidence']:.2f})")
                        
                        # Show annotated image
                        annotated = face_rec.visualize_recognition_results(rgb_image, results)
                        cv2.imshow("Face Recognition", annotated)
                    else:
                        print("👤 No faces detected")
                else:
                    print("❌ Failed to capture image for face recognition")
        
        cv2.destroyAllWindows()
        print("🏁 Interactive test completed")
        
    except Exception as e:
        print(f"❌ Error in interactive test: {e}")
        cv2.destroyAllWindows()

def main():
    """Main test function."""
    print("🔮 TIAGO CAMERA & FACE RECOGNITION TEST SUITE 🔮")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
    else:
        print("Available tests:")
        print("1. basic    - Basic camera functionality")
        print("2. face     - Face recognition test")
        print("3. interactive - Interactive capture test")
        print("4. all      - Run all tests")
        print()
        test_type = input("Select test type (or 'all'): ").lower()
    
    print(f"\n🚀 Running test: {test_type}")
    
    try:
        if test_type in ['basic', 'all']:
            test_camera_basic()
            
        if test_type in ['face', 'all']:
            test_face_recognition()
            
        if test_type in ['interactive', 'all']:
            if test_type == 'all':
                input("\nPress Enter to start interactive test...")
            test_interactive_capture()
            
        print("\n✨ All tests completed! ✨")
        
    except rospy.ROSInterruptException:
        print("🛑 Test interrupted by ROS")
    except KeyboardInterrupt:
        print("🛑 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()