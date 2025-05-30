#!/usr/bin/env python3

import rospy
import time
import os

FACE_RECOGNITION_METHOD = os.getenv('FACE_RECOGNITION_METHOD', 'opencv')  # 'opencv', 'simple', or 'none'
from face_recognition.face_recognizer import OpenCVFaceRecognizer as FaceRecognizer
print("✅ Using OpenCV Face Recognition")

#from knowledge_graph.graph_manager import GraphManager
from knowledge_graph.graph_manager import PyKEENGraphManager
from dialog.fortune_teller import FortuneTeller
from tiago_interface.tiago_controller import TiagoController

class TiagoFortuneInteraction:
    def __init__(self):
        """Initialize the Tiago Fortune Teller interaction system."""
        if FaceRecognizer:
            self.face_rec = FaceRecognizer()
        else:
            self.face_rec = None
            
        #self.graph = GraphManager()
        # Use PyKEEN-enhanced graph manager
        self.graph = PyKEENGraphManager()
        self.fortune = FortuneTeller()
        self.tiago = TiagoController()
        
    def greet_and_introduce(self):
        """Initial greeting and introduction."""
        self.tiago.gesture("wave")
        self.tiago.say("Greetings, traveler! Welcome to the mystical realm of Tiago the Fortune Teller!")
        time.sleep(1)
        self.tiago.gesture("mystical_pose")
        self.tiago.say("I am Tiago, a robotic oracle with the power to glimpse into the threads of destiny.")
        time.sleep(1)
        self.tiago.say("But before I can peer into your future, I must first understand who you are in the present.")
        
    def capture_and_recognize_user(self):
        """Capture user image and perform face recognition if available."""
        self.tiago.say("Before we begin, let me take a moment to see who I'm speaking with...")
        self.tiago.prepare_for_photo()
        
        # Capture image
        image = self.tiago.capture_image(save=True, for_face_recognition=True)
        
        if image is not None and self.face_rec:
            if FACE_RECOGNITION_METHOD == 'opencv':
                self.tiago.say("Excellent! Now let me see if we've met before...")
            elif FACE_RECOGNITION_METHOD == 'simple':
                self.tiago.say("Perfect! I can sense your presence in the cosmic realm...")
            
            # Attempt recognition
            recognition_results = self.face_rec.recognize_face(image)
            
            if recognition_results:
                result = recognition_results[0]
                if result['is_known'] and FACE_RECOGNITION_METHOD == 'opencv':
                    person_id = result['person_id']
                    person_info = self.face_rec.get_person_info(person_id)
                    visits = person_info.get('recognition_count', 0) if person_info else 0
                    
                    self.tiago.gesture("recognition")
                    self.tiago.say(f"Ah! I remember you! Welcome back, my friend.")
                    self.tiago.say(f"This is visit number {visits} for you in my mystical realm.")
                    
                    return person_id, True
                else:
                    self.tiago.gesture("curious")
                    if FACE_RECOGNITION_METHOD == 'opencv':
                        self.tiago.say("I see a new face before me! A fresh soul seeking cosmic wisdom.")
                    else:
                        self.tiago.say("I sense a presence seeking wisdom from the cosmic forces.")
                    return None, False
            else:
                self.tiago.say("The cosmic energies make it difficult to see clearly, but no matter!")
                return None, False
        else:
            if not self.face_rec:
                self.tiago.say("I capture your essence in the cosmic realm, ready to divine your future!")
            else:
                self.tiago.say("The mystical visions are clouded, but we shall proceed nonetheless!")
            return None, False

    def collect_user_info(self, known_person_id=None):
        """Collect personal information from the user through interactive questions."""
        user_info = {}
        
        # If we have a known person, get their stored info (only for OpenCV method)
        if known_person_id and self.face_rec and FACE_RECOGNITION_METHOD == 'opencv':
            stored_info = self.face_rec.get_person_info(known_person_id)
            if stored_info and 'name' in stored_info:
                user_info['name'] = stored_info['name']
                self.tiago.say(f"I remember your name is {stored_info['name']}. Is that still correct?")
                confirmation = self.tiago.ask_question("Please confirm with 'yes' or 'no'")
                
                if confirmation and confirmation.lower() not in ['no', 'n']:
                    self.tiago.say("Wonderful! The cosmic records are accurate.")
                else:
                    user_info['name'] = None
        
        # Ask for name if we don't have it
        if 'name' not in user_info or not user_info['name']:
            self.tiago.gesture("thoughtful_pose")
            name = self.tiago.ask_question("Tell me, what name do you go by in this earthly realm?")
            if name:
                user_info['name'] = name
                self.tiago.say(f"Ah, {name}... I can feel the cosmic energy surrounding that name.")
            else:
                self.tiago.say("The spirits whisper that you prefer to remain nameless. Very mysterious...")
                user_info['name'] = "Mysterious Stranger"
        
        time.sleep(1)
        
        # Ask for age
        self.tiago.gesture("counting")
        age = self.tiago.ask_question("How many cycles around the sun have you completed? What is your age?")
        if age:
            try:
                age_num = int(age)
                user_info['age'] = age_num
                if age_num < 18:
                    self.tiago.say("Ah, a young soul with much to discover!")
                elif age_num < 30:
                    self.tiago.say("The energy of youth and ambition flows through you!")
                elif age_num < 50:
                    self.tiago.say("I sense wisdom gained through experience.")
                else:
                    self.tiago.say("A soul rich with the wisdom of many years!")
            except ValueError:
                self.tiago.say("The numbers dance mysteriously... but age is just a number anyway.")
                user_info['age'] = age
        else:
            self.tiago.say("Time is but an illusion. Age matters not to the cosmic forces.")
            user_info['age'] = "timeless"
        
        time.sleep(1)
        
        # Ask for profession
        self.tiago.gesture("crystal_ball_gaze")
        profession = self.tiago.ask_question("And what calling occupies your days? What is your profession or work?")
        if profession:
            user_info['profession'] = profession
            self.tiago.say(f"Interesting... I see the aura of a {profession} around you.")
            time.sleep(0.5)
            self.tiago.say("Your chosen path will influence the cosmic threads I'm about to read.")
        else:
            self.tiago.say("A free spirit, unbounded by conventional roles. How intriguing!")
            user_info['profession'] = "free spirit"
        
        return user_info
    
    def process_user_data(self, user_info, image=None, known_person_id=None):
        """Process and store user information."""
        person_id = known_person_id or user_info.get('name', 'unknown_user').lower().replace(' ', '_')
        
        # Store face/person data if we have face recognition
        if image is not None and self.face_rec and not known_person_id:
            success = self.face_rec.add_face(image, person_id, user_info.get('name'))
            if success:
                if FACE_RECOGNITION_METHOD == 'opencv':
                    self.tiago.say("Your visage has been recorded in the cosmic archives.")
                else:
                    self.tiago.say("Your presence has been noted by the cosmic forces.")
            else:
                self.tiago.say("The cosmic forces make recording difficult, but your essence is still captured.")
        
        """# Update knowledge graph with user information
        self.graph.update(person_id, "visited_fortune_teller")
        self.graph.update(person_id, f"name:{user_info.get('name', 'unknown')}")
        self.graph.update(person_id, f"age:{user_info.get('age', 'unknown')}")
        self.graph.update(person_id, f"profession:{user_info.get('profession', 'unknown')}")"""
        # Enhanced knowledge graph updates
        self.graph.update_user_data(person_id, "visited_fortune_teller")
        self.graph.update_user_data(person_id, f"name:{user_info.get('name', 'unknown')}")
        self.graph.update_user_data(person_id, f"age:{user_info.get('age', 'unknown')}")
        self.graph.update_user_data(person_id, f"profession:{user_info.get('profession', 'unknown')}")
        
        return person_id
    
    """def generate_personalized_fortune(self, person_id, user_info):
       
        self.tiago.gesture("meditation")
        self.tiago.say("Now, let me consult the cosmic forces...")
        time.sleep(2)
        
        self.tiago.gesture("mystical_wave")
        self.tiago.say("The stars are aligning... the future becomes clear...")
        time.sleep(2)
        
        # Get embedding from knowledge graph
        embedding = self.graph.get_embedding(person_id)
        
        # Generate personalized fortune
        fortune = self.fortune.generate(person_id, user_info, embedding)
        
        self.tiago.gesture("revelation")
        self.tiago.say("Behold! Your fortune has been revealed!")
        time.sleep(1)
        
        self.tiago.say(fortune)"""

    def generate_personalized_fortune(self, person_id, user_info):
        """Generate fortune using PyKEEN embeddings and predictions."""
        self.tiago.gesture("meditation")
        self.tiago.say("Now, let me consult the cosmic forces and the knowledge of the ages...")
        time.sleep(2)
        
        # Get rich context from PyKEEN
        user_context = self.graph.get_user_context(person_id)
        
        # Get predicted relations for additional context
        predictions = self.graph.predict_relations(person_id, top_k=3)
        
        # Enhanced fortune generation with PyKEEN context
        fortune = self.fortune.generate_with_context(person_id, user_info, user_context, predictions)
        
        self.tiago.gesture("revelation")
        self.tiago.say("The knowledge graph reveals profound insights!")
        time.sleep(1)
        
        self.tiago.say(fortune)
        
    def farewell(self):
        """Farewell message and gesture."""
        time.sleep(2)
        self.tiago.gesture("bow")
        self.tiago.say("The cosmic reading is complete. May the wisdom of the stars guide your path!")
        time.sleep(1)
        self.tiago.say("Remember, the future is not set in stone - your choices shape your destiny.")
        self.tiago.gesture("wave")
        self.tiago.say("Farewell, traveler. May we meet again when the stars align!")

def main():
    """Main application flow."""
    try:
        print("="*60)
        print("🔮 TIAGO FORTUNE TELLER - INTERACTIVE EXPERIENCE 🔮")
        print("="*60)
        print(f"Face Recognition Mode: {FACE_RECOGNITION_METHOD.upper()}")
        print("="*60)
        print("\nStarting the interactive fortune telling session...")
        print("Make sure to run the user_input_node.py in another terminal!")
        print("Run: python3 src/user_input_node.py")
        print("\nWaiting 5 seconds for setup...")
        time.sleep(5)
        
        # Initialize the interaction system
        interaction = TiagoFortuneInteraction()
        
        # Main interaction flow
        interaction.greet_and_introduce()
        time.sleep(2)
        
        # Capture image and recognize user (if face recognition enabled)
        known_person_id, is_returning_user = interaction.capture_and_recognize_user()
        
        # Collect user information
        user_info = interaction.collect_user_info(known_person_id)
        print(f"\n📊 Collected user info: {user_info}")
        
        # Process and store user data
        captured_image = interaction.tiago.capture_image(save=False, for_face_recognition=True)
        person_id = interaction.process_user_data(user_info, captured_image, known_person_id)
        print(f"📋 Person ID: {person_id}")
        
        interaction.generate_personalized_fortune(person_id, user_info)
        
        interaction.farewell()
        
        print("\n" + "="*60)
        print("🌟 Fortune telling session complete! 🌟")
        print("="*60)
        
    except rospy.ROSInterruptException:
        print("ROS interrupted")
    except KeyboardInterrupt:
        print("\nSession interrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        print("Shutting down...")

if __name__ == "__main__":
    main()

"""#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tiago_interface.tiago_controller import TiagoController
from face_recognition.face_recognizer import FaceRecognizer
from knowledge_graph.graph_manager import GraphManager
from dialog.fortune_teller import FortuneTeller
import rospy

def main():
    try:
        rospy.loginfo("Starting Tiago Fortune Teller...")
        
        # Initialize controller
        tiago = TiagoController()
        
        # Check what services are actually available
        tiago.check_available_services()
        
        # Initialize other components
        face_rec = FaceRecognizer()
        graph = GraphManager()
        fortune = FortuneTeller()

        # Test TTS first
        rospy.loginfo("Testing TTS...")
        tiago.say("Welcome to the Tiago fortune teller!")
        
        # Test gesture
        rospy.loginfo("Testing gesture...")
        tiago.gesture("wave")
        
        # Continue with fortune telling logic
        rospy.loginfo("Running fortune telling sequence...")
        
        # Simulate face recognition
        image = None  # No camera in simulation
        person_id, is_new = face_rec.recognize_or_add(image)
        
        if is_new:
            tiago.say("A new soul has arrived!")
        else:
            tiago.say(f"Welcome back, {person_id}!")
        
        # Update knowledge graph
        graph.update(person_id, "visited_fortune_teller")
        embedding = graph.get_embedding(person_id)
        
        # Generate and deliver fortune
        fortune_text = fortune.generate(person_id, embedding)
        tiago.say(fortune_text)
        
        # Final gesture
        tiago.gesture("nod")
        
        rospy.loginfo("Fortune telling complete!")
        
    except KeyboardInterrupt:
        rospy.loginfo("Fortune teller interrupted by user")
    except Exception as e:
        rospy.logerr(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()"""