#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import threading
import sys

class UserInputNode:
    def __init__(self):
        """Initialize the user input node for terminal interaction."""
        rospy.init_node('user_input_node', anonymous=True)
        
        # Publisher for user input
        self.input_pub = rospy.Publisher('/tiago/user_input', String, queue_size=10)
        
        # Flag to control input thread
        self.running = True
        
        rospy.loginfo("User Input Node initialized")
        rospy.loginfo("Type your responses and press Enter to send them to Tiago")
        rospy.loginfo("Type 'quit' or 'exit' to stop the node")
        
    def start_input_thread(self):
        """Start the input thread to handle terminal input."""
        input_thread = threading.Thread(target=self.input_loop)
        input_thread.daemon = True
        input_thread.start()
        
    def input_loop(self):
        """Main loop for handling terminal input."""
        while self.running and not rospy.is_shutdown():
            try:
                print("\n" + "="*50)
                user_input = input("👤 Your response: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    rospy.loginfo("User requested shutdown")
                    self.running = False
                    rospy.signal_shutdown("User requested shutdown")
                    break
                
                if user_input:
                    # Publish user input
                    msg = String()
                    msg.data = user_input
                    self.input_pub.publish(msg)
                    rospy.loginfo(f"Published user input: {user_input}")
                    
            except (EOFError, KeyboardInterrupt):
                rospy.loginfo("Input interrupted")
                self.running = False
                rospy.signal_shutdown("Input interrupted")
                break
            except Exception as e:
                rospy.logerr(f"Error in input loop: {e}")
    
    def run(self):
        """Run the input node."""
        self.start_input_thread()
        
        # Keep the main thread alive
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down user input node")
        finally:
            self.running = False

if __name__ == '__main__':
    try:
        node = UserInputNode()
        node.run()
    except rospy.ROSInterruptException:
        pass