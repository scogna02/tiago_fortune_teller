#!/usr/bin/env python3

"""
PC Camera test script for Tiago Fortune Teller.
Tests PC webcam functionality and face recognition.
"""

import cv2
import sys
import os
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_pc_camera_basic():
    """Test basic PC camera functionality."""
    print("🧪 Testing Basic PC Camera")
    print("=" * 40)
    
    try:
        from tiago_interface.pc_camera_manager import TiagoCameraManager
        
        # Try to initialize camera (camera_id=0 is usually the default webcam)
        camera = TiagoCameraManager(camera_id=0)
        
        print("⏳ Waiting for camera to be ready...")
        if camera.is_camera_ready():
            print("✅ PC Camera is ready!")
            
            # Capture a test image
            print("📸 Capturing test image...")
            image = camera.capture_image(save=True)
            
            if image is not None:
                print(f"✅ Image captured: {image.shape}")
                stats = camera.get_image_stats()
                print(f"📊 Camera Stats: {stats}")
                
                # Show live preview using input() instead of cv2.waitKey()
                print("📺 Press ENTER to capture, 'q' + ENTER to quit")
                capture_count = 0
                
                while True:
                    try:
                        user_input = input("Press ENTER to capture (or 'q' to quit): ").strip().lower()
                        
                        if user_input == 'q':
                            break
                        else:
                            test_img = camera.capture_image(save=True)
                            if test_img is not None:
                                capture_count += 1
                                print(f"📸 Captured image #{capture_count}")
                                
                    except KeyboardInterrupt:
                        print("\n🛑 Test interrupted")
                        break
                
                print(f"💾 Total captures: {capture_count}")
                
            else:
                print("❌ Failed to capture image")
        else:
            print("❌ Camera not ready")
            
        camera.shutdown()
        
    except Exception as e:
        print(f"❌ Error in PC camera test: {e}")


def test_face_recognition():
    """Test face recognition with PC camera."""
    print("\n🧪 Testing Face Recognition with PC Camera")
    print("=" * 50)
    
    try:
        from tiago_interface.pc_camera_manager import TiagoCameraManager
        from face_recognition.enhanced_face_recognizer import FaceRecognizer
        
        camera = TiagoCameraManager(camera_id=0)
        face_rec = FaceRecognizer()
        
        if camera.is_camera_ready():
            print("✅ Camera ready for face recognition test")
            print("📺 Press ENTER to test face recognition, 'q' + ENTER to quit")
            
            while True:
                try:
                    # Use input() instead of cv2.waitKey for Docker compatibility
                    user_input = input("Press ENTER to capture (or 'q' to quit): ").strip().lower()
                    
                    if user_input == 'q':
                        break
                    else:
                        print("📸 Capturing image for face recognition...")
                        rgb_image = camera.capture_for_face_recognition()
                        
                        if rgb_image is not None:
                            print("👤 Analyzing faces...")
                            results = face_rec.recognize_face(rgb_image)
                            
                            if results:
                                print(f"✅ Found {len(results)} face(s)!")
                                for i, result in enumerate(results):
                                    print(f"  Face {i+1}: {result['person_id']} (confidence: {result['confidence']:.2f})")
                            else:
                                print("ℹ️  No faces detected")
                        else:
                            print("❌ Failed to capture image")
                            
                except KeyboardInterrupt:
                    print("\n🛑 Test interrupted")
                    break
        else:
            print("❌ Camera not ready for face recognition")
            
        camera.shutdown()
        
    except Exception as e:
        print(f"❌ Error in face recognition test: {e}")


def test_integration():
    """Test integration with fortune teller controller."""
    print("\n🧪 Testing Integration with Tiago Controller")
    print("=" * 50)
    
    try:
        # Test without ROS first
        print("🔧 Testing camera integration (no ROS)...")
        from tiago_interface.pc_camera_manager import TiagoCameraManager
        
        camera = TiagoCameraManager(camera_id=0)
        
        if camera.is_camera_ready():
            print("✅ Camera integration test passed")
            
            # Test capture methods
            bgr_img = camera.capture_image(save=False)
            rgb_img = camera.capture_for_face_recognition()
            
            if bgr_img is not None and rgb_img is not None:
                print(f"✅ BGR Image: {bgr_img.shape}")
                print(f"✅ RGB Image: {rgb_img.shape}")
                print("✅ All capture methods working")
            else:
                print("❌ Image capture methods failed")
        else:
            print("❌ Camera integration failed")
            
        camera.shutdown()
        
    except Exception as e:
        print(f"❌ Error in integration test: {e}")

def main():
    """Main test function."""
    print("🔮 PC CAMERA TEST SUITE FOR TIAGO FORTUNE TELLER 🔮")
    print("=" * 60)
    
    # Check if camera is available
    print("🔍 Checking for available cameras...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✅ Camera 0 found and accessible")
        cap.release()
    else:
        print("❌ No camera found at index 0")
        print("💡 Try connecting a webcam or check camera permissions")
        return
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
    else:
        print("\nAvailable tests:")
        print("1. basic       - Basic PC camera functionality")
        print("2. face        - Face recognition test")
        print("3. integration - Integration test")
        print("4. all         - Run all tests")
        print()
        test_type = input("Select test type (or 'all'): ").lower()
    
    print(f"\n🚀 Running test: {test_type}")
    
    try:
        if test_type in ['basic', 'all']:
            test_pc_camera_basic()
            
        if test_type in ['face', 'all']:
            test_face_recognition()
            
        if test_type in ['integration', 'all']:
            test_integration()
            
        print("\n✨ All tests completed! ✨")
        
    except KeyboardInterrupt:
        print("🛑 Tests interrupted by user")
    except Exception as e:
        print(f"❌ Tests failed with error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()