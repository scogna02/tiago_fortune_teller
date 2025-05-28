tiago_fortune_teller/
├── docker/
│   └── Dockerfile
├── compose/
│   └── docker-compose.yml
├── src/
│   ├── face_recognition/
│   │   └── face_recognizer.py
│   ├── knowledge_graph/
│   │   └── graph_manager.py
│   ├── dialog/
│   │   └── fortune_teller.py
│   ├── tiago_interface/
│   │   └── tiago_controller.py
│   └── main.py
├── config/
│   └── config.yaml
├── data/
├── tests/
│   └── test_main.py
└── README.md


To check the libraries and functions available for your program, you can look at the following:

Python Dependencies for tiago_fortune_teller:

The primary list of Python libraries specifically added for your project is in requirements.txt. These are installed via pip as defined in the Dockerfile.
The Dockerfile also installs system packages like python3-dev, libopenblas-dev, etc., which support the Python libraries.
Your Python code, like in tiago_controller.py, can import and use these installed Python modules (e.g., numpy).
ROS and C++ Libraries (from the execution_sciroc environment):

The execution_sciroc project uses several ROS packages. These are listed in its CMakeLists.txt file, such as CMakeLists.txt. Key components include:
actionlib
play_motion_msgs
roscpp (for C++)
rospy (for Python)
std_msgs
If your Python program (e.g., TiagoController) needs to interact with the Tiago robot or other ROS nodes from the execution_sciroc environment, it would typically use rospy to call ROS services, publish/subscribe to topics, or use action clients based on messages and actions defined by these packages.
The C++ executables in execution_sciroc also link against libraries like Boost and yaml-cpp, as seen in files like build.make.
Exploring within the Docker container:

Once the Docker container is running, you can execute commands to list installed software:
For Python packages: pip list
For system (Debian/Ubuntu) packages: dpkg -l
You can browse directories like lib, lib, and the ROS workspace's devel/lib (e.g., /tiago_public_ws/empower_docker/execution_sciroc/devel/lib as indicated by empower_docker/execution_sciroc/build/catkin_generated/setup_cached.sh) for compiled libraries, and devel/include or install/include for header files if you were developing C++ components.
For your Python application, focus on the libraries in requirements.txt and standard Python libraries. If you need to interface with ROS functionalities, you'll use rospy and the relevant ROS message/service/action definitions from packages like play_motion_msgs.

# example
sudo ./start_docker.sh -it -v /dev/snd:/dev/snd registry.gitlab.com/brienza1/empower_docker:latest

# launch ros
cd ~/tiago_public_ws
source ../tiago_public_ws/devel/setup.bash
roslaunch tts tts.launch type:=terminal

# launch gazebo
cd ~/tiago_public_ws
source ../tiago_public_ws/devel/setup.bash
roslaunch tiago_gazebo tiago_gazebo.launch public_sim:=true robot:=steel gui:=false

# launch Tiago fortune teller container
sudo docker compose -f compose/docker-compose.yml run --entrypoint /bin/bash tiago_fortune_teller

# run python file
cd ~/tiago_public_ws
source ../tiago_public_ws/devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=localhost

python3 src/main.py


sudo ./start_docker.sh -it -v /dev/snd:/dev/snd registry.gitlab.com/brienza1/empower_docker:latest

source ../tiago_public_ws/devel/setup.bash
roslaunch tiago_gazebo tiago_gazebo.launch public_sim:=true robot:=steel gui:=false

sudo docker compose -f compose/docker-compose.yml run --entrypoint /bin/bash tiago_fortune_teller

source ../tiago_public_ws/devel/setup.bash
roslaunch tts tts.launch type:=terminal

sudo docker compose -f compose/docker-compose.yml run --entrypoint /bin/bash tiago_fortune_teller
python3 src/main.py


# 🔮 Tiago Fortune Teller - Interactive Edition

An enhanced, interactive fortune telling system powered by the PAL Robotics Tiago robot, featuring real-time user interaction, personalized fortunes, and knowledge graph-based memory.

## 🌟 New Features

- **Interactive Questioning**: Tiago asks users for their name, age, and profession
- **Terminal-based Input**: Users respond via a dedicated ROS-based input system
- **Personalized Fortunes**: Fortunes are customized based on user information
- **Knowledge Graph Memory**: Persistent storage of user data and interaction history
- **Enhanced Dialog System**: Both OpenAI API and fallback fortune generation
- **Expressive Interactions**: Rich gesture and speech integration

## 📁 Project Structure

```
tiago_fortune_teller/
├── docker/
│   ├── Dockerfile
│   └── requirements.txt
├── compose/
│   └── docker-compose.yml
├── src/
│   ├── main.py                         # Enhanced main application
│   ├── user_input_node.py              # ROS node for terminal input
│   ├── face_recognition/
│   │   └── face_recognizer.py
│   ├── knowledge_graph/
│   │   └── graph_manager.py            # Enhanced with persistent storage
│   ├── dialog/
│   │   └── fortune_teller.py           # Personalized fortune generation
│   └── tiago_interface/
│       └── tiago_controller.py         # Enhanced with interactive methods
├── config/
│   └── config.yaml
├── data/                               # Auto-created data directories
│   ├── faces/
│   └── knowledge_graph/
├── tests/
│   └── test_main.py
├── launch_system.sh                    # Easy launch script
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- ROS 1 (Noetic recommended)
- Python 3.7+
- Required Python packages (see `docker/requirements.txt`)

### Option 1: Easy Launch (Recommended)

1. **Make the launch script executable:**
   ```bash
   chmod +x launch_system.sh
   ```

2. **Run the launch script:**
   ```bash
   ./launch_system.sh
   ```

The script will:
- Check ROS dependencies
- Start ROS master if needed
- Create necessary directories
- Launch the user input system
- Start the main Tiago application

### Option 2: Manual Launch

1. **Start ROS Master:**
   ```bash
   roscore
   ```

2. **Set up environment:**
   ```bash
   export PYTHONPATH="${PWD}/src:$PYTHONPATH"
   export ROS_MASTER_URI="http://localhost:11311"
   ```

3. **Create data directories:**
   ```bash
   mkdir -p data/faces data/knowledge_graph/{users,interactions,embeddings}
   ```

4. **Start the user input node (in one terminal):**
   ```bash
   python3 src/user_input_node.py
   ```

5. **Start the main application (in another terminal):**
   ```bash
   python3 src/main.py
   ```

## 🎮 How to Use

### Interactive Session Flow

1. **Welcome & Introduction**: Tiago greets you and explains his mystical powers
2. **Personal Information Gathering**:
   - **Name**: "Tell me, what name do you go by in this earthly realm?"
   - **Age**: "How many cycles around the sun have you completed?"
   - **Profession**: "What calling occupies your days?"
3. **Fortune Generation**: Tiago consults the cosmic forces and generates a personalized fortune
4. **Farewell**: Mystical goodbye with wisdom for the journey ahead

### Responding to Tiago

- **Input Terminal**: Use the "User Input Node" terminal window to type responses
- **Press Enter**: Send your response to Tiago after typing
- **Be Patient**: Wait for Tiago to ask each question before responding
- **Exit Commands**: Type `quit` or `exit` to end the session

### Example Interaction

```
🤖 Tiago: "Greetings, traveler! Welcome to the mystical realm of Tiago the Fortune Teller!"
🤖 *Tiago wave*
🤖 Tiago: "Tell me, what name do you go by in this earthly realm?"

👤 Your response: Alice
==================================================

🤖 Tiago: "Ah, Alice... I can feel the cosmic energy surrounding that name."
🤖 Tiago: "How many cycles around the sun have you completed? What is your age?"

👤 Your response: 28
==================================================

🤖 Tiago: "The energy of youth and ambition flows through you!"
🤖 Tiago: "And what calling occupies your days? What is your profession or work?"

👤 Your response: Software Engineer
==================================================

🤖 Tiago: "Interesting... I see the aura of a Software Engineer around you."
🤖 *Tiago crystal_ball_gaze*
🤖 Tiago: "Now, let me consult the cosmic forces..."
```

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: (Optional) For enhanced AI-powered fortunes
- `ROS_MASTER_URI`: ROS master location (default: http://localhost:11311)

### Customization

#### Adding New Gestures

Edit `src/tiago_interface/tiago_controller.py`:
```python
def gesture(self, gesture_type):
    # Add new gesture types here
    gestures = {
        'mystical_wave': 'waves hands mysteriously',
        'crystal_ball_gaze': 'peers into crystal ball',
        # Add your custom gestures
    }
```

#### Custom Fortune Templates

Edit `src/dialog/fortune_teller.py` to add profession-specific fortunes:
```python
self.profession_fortunes = {
    'your_profession': [
        "Your custom fortune template here...",
        "Another fortune for this profession..."
    ]
}
```

## 📊 Data Storage

The system automatically stores:

- **User Profiles**: Name, age, profession, visit history
- **Interaction History**: Timestamped interaction logs
- **Knowledge Embeddings**: Vector representations for personalization
- **Face Encodings**: (When face recognition is enabled)

Data is stored in the `data/` directory and persists between sessions.

## 🛠️ Development

### Adding New Questions

1. Edit the `collect_user_info()` method in `src/main.py`
2. Add new question using `self.tiago.ask_question()`
3. Store response in `user_info` dictionary
4. Update knowledge graph storage in `process_user_data()`

### Extending the Knowledge Graph

Modify `src/knowledge_graph/graph_manager.py`:
- Add new relationship types in `update()`
- Enhance embedding generation in `update_embedding()`
- Expand user attributes storage

### Testing

Run the test suite:
```bash
python3 -m pytest tests/
```

## 🐳 Docker Deployment

For production deployment with real Tiago robot:

```bash
cd compose/
docker-compose up --build
```

## 🤖 ROS Integration

### Topics

- `/tiago/speech`: Speech output from Tiago
- `/tiago/gesture`: Gesture commands for Tiago
- `/tiago/user_input`: User input from terminal

### Services

- `/tiago/request_input`: Signal that user input is needed

### Monitoring

```bash
# View all Tiago-related topics
rostopic list | grep tiago

# Monitor user input
rostopic echo /tiago/user_input

# Monitor speech output
rostopic echo /tiago/speech
```

## 🔮 Fortune Examples

The system generates different types of fortunes based on user information:

- **Profession-based**: Tailored to the user's work life
- **Age-aware**: Appropriate for life stage
- **Personalized**: Uses the user's name when provided
- **Cosmic flair**: Mystical language and endings

## 🎭 Gesture System

Tiago performs contextual gestures throughout the interaction:

- `wave`: Welcome gesture
- `mystical_pose`: During introductions
- `thoughtful_pose`: When asking deep questions
- `crystal_ball_gaze`: During fortune consultation
- `revelation`: When revealing the fortune
- `bow`: Respectful farewell

## 📈 Future Enhancements

- Voice recognition for audio input
- Camera-based face recognition integration
- Multi-language support
- Tarot card visual elements
- Historical fortune accuracy tracking
- Social media integration for sharing fortunes

## 🐛 Troubleshooting

### Common Issues

1. **ROS Master not found**: Make sure `roscore` is running
2. **Permission denied**: Check file permissions on launch script
3. **Import errors**: Verify PYTHONPATH includes the src directory
4. **Input not working**: Ensure user_input_node.py is running

### Debug Commands

```bash
# Check ROS connections
rosnode list
rostopic list

# Test user input
rostopic pub /tiago/user_input std_msgs/String "data: 'test message'"

# Monitor system logs
tail -f ~/.ros/log/latest/rosout.log
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

*May the cosmic forces guide your coding journey! ✨*