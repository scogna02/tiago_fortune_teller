#!/usr/bin/env python3

"""
Gesture test script for Tiago Fortune Teller.
Tests all available gestures to verify they work in Gazebo.
"""

import sys
import os
import time
import rospy

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_individual_gestures():
    """Test each gesture individually."""
    print("🧪 Testing Individual Gestures")
    print("=" * 40)
    
    try:
        from tiago_interface.tiago_controller import TiagoController
        
        # Initialize controller
        tiago = TiagoController()
        
        # Check what's available
        tiago.check_available_services()
        
        if not tiago.gesture_available:
            print("❌ PlayMotion not available - cannot test gestures")
            return
        
        # Test gestures used in fortune teller
        fortune_teller_gestures = [
            'wave',
            'mystical_pose', 
            'thoughtful_pose',
            'crystal_ball_gaze',
            'meditation',
            'mystical_wave',
            'revelation',
            'bow',
            'recognition',
            'curious',
            'counting',
            'look_at_user'
        ]
        
        print(f"\n🎭 Testing {len(fortune_teller_gestures)} fortune teller gestures:")
        
        for i, gesture in enumerate(fortune_teller_gestures):
            print(f"\n[{i+1}/{len(fortune_teller_gestures)}] Testing gesture: '{gesture}'")
            try:
                tiago.gesture(gesture)
                print(f"✅ '{gesture}' completed successfully")
                time.sleep(1)  # Wait between gestures
            except Exception as e:
                print(f"❌ '{gesture}' failed: {e}")
        
        print(f"\n✨ Gesture testing completed!")
        
    except Exception as e:
        print(f"❌ Error in gesture test: {e}")

def test_available_motions():
    """Test all available motions from Tiago's motion library."""
    print("\n🧪 Testing All Available Motions")
    print("=" * 40)
    
    try:
        from tiago_interface.tiago_controller import TiagoController
        
        tiago = TiagoController()
        
        if not tiago.gesture_available:
            print("❌ PlayMotion not available")
            return
        
        # Get available motions
        available_motions = tiago.check_available_motions()
        
        if not available_motions:
            print("❌ No motions found")
            return
        
        print(f"\n🎪 Testing {len(available_motions)} available motions:")
        print("Press ENTER to continue between motions, or 'q' + ENTER to quit")
        
        for i, motion in enumerate(sorted(available_motions)):
            user_input = input(f"\n[{i+1}/{len(available_motions)}] Test '{motion}'? (ENTER/q): ").strip().lower()
            
            if user_input == 'q':
                print("🛑 Testing stopped by user")
                break
                
            print(f"🎭 Testing motion: '{motion}'")
            try:
                # Use the direct motion name (not mapped)
                goal = tiago.gesture_client.send_goal_and_wait(
                    tiago.create_motion_goal(motion), rospy.Duration(15)
                )
                print(f"✅ '{motion}' completed")
            except Exception as e:
                print(f"❌ '{motion}' failed: {e}")
        
        print(f"\n✨ Motion testing completed!")
        
    except Exception as e:
        print(f"❌ Error in motion test: {e}")

def create_motion_goal(motion_name):
    """Helper to create a PlayMotion goal."""
    from play_motion_msgs.msg import PlayMotionGoal
    
    goal = PlayMotionGoal()
    goal.motion_name = motion_name
    goal.skip_planning = False
    goal.priority = 0
    return goal

def test_speech():
    """Test TTS functionality."""
    print("\n🧪 Testing Speech (TTS)")
    print("=" * 30)
    
    try:
        from tiago_interface.tiago_controller import TiagoController
        
        tiago = TiagoController()
        
        if tiago.tts_available:
            print("✅ TTS available - testing speech")
            tiago.say("Hello! I am Tiago, testing my speech capabilities.")
            time.sleep(2)
            tiago.say("The speech test is now complete!")
        else:
            print("⚠️ TTS not available - using fallback mode")
            tiago.say("This is fallback speech mode")
        
        print("✅ Speech test completed")
        
    except Exception as e:
        print(f"❌ Error in speech test: {e}")

def main():
    """Main test function."""
    print("🔮 TIAGO GESTURE & SPEECH TEST SUITE 🔮")
    print("=" * 50)
    print("Make sure Gazebo is running with Tiago simulation!")
    print("Run: roslaunch tiago_gazebo tiago_gazebo.launch public_sim:=true robot:=steel")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
    else:
        print("\nAvailable tests:")
        print("1. gestures     - Test fortune teller gestures")
        print("2. motions      - Test all available motions")
        print("3. speech       - Test TTS functionality")
        print("4. all          - Run all tests")
        print()
        test_type = input("Select test type (or 'all'): ").lower()
    
    print(f"\n🚀 Running test: {test_type}")
    
    try:
        if test_type in ['gestures', 'all']:
            test_individual_gestures()
            
        if test_type in ['motions', 'all']:
            test_available_motions()
            
        if test_type in ['speech', 'all']:
            test_speech()
            
        print("\n✨ All tests completed! ✨")
        
    except KeyboardInterrupt:
        print("🛑 Tests interrupted by user")
    except Exception as e:
        print(f"❌ Tests failed with error: {e}")

if __name__ == '__main__':
    main()