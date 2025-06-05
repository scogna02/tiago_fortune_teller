# 🔮 Tiago Fortune Teller - AI-Powered Social Robot

An advanced, AI-enhanced fortune telling system powered by the PAL Robotics Tiago robot, featuring real-time face recognition, PyKEEN knowledge graph embeddings, personalized AI-generated fortunes, and sophisticated human-robot interaction capabilities.

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![ROS](https://img.shields.io/badge/ROS-Noetic-green.svg)
![PyKEEN](https://img.shields.io/badge/PyKEEN-1.10+-orange.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-purple.svg)

## 🌟 System Features

### 🤖 **Intelligent Robot Interaction**
- **Expressive Speech Synthesis**: Multi-fallback TTS system (ROS action servers, pyttsx3)
- **Contextual Gestures**: Mystical gestures mapped to Tiago motions
- **Interactive Terminal Interface**: Real-time user input via dedicated ROS node (`user_input_node.py`)
- **Multimodal Communication**: Synchronized speech, gesture, and visual feedback

### 👁️ **Computer Vision & Recognition**
- **OpenCV Face Recognition**: LBPH-based face recognition with confidence scoring
- **PC Camera Integration**: Full webcam support with live capture threading
- **Persistent Face Database**: Automatic user enrollment with photo archival
- **Fallback Image Generation**: Graceful degradation when camera unavailable

### 🧠 **Knowledge Graph Intelligence**
- **PyKEEN Integration**: TransE, ComplEx, RotatE embedding models with safe fallbacks
- **Dynamic Learning**: Real-time knowledge graph updates from user interactions
- **Relation Prediction**: AI-powered prediction of user preferences and characteristics
- **Persistent Storage**: JSON-based data persistence with automatic model training

### 🔮 **AI-Powered Fortune Generation**
- **OpenAI GPT Integration**: Context-aware, personalized fortune generation
- **Knowledge Graph Context**: Fortunes informed by user embeddings and predicted relations
- **Professional Fallback System**: Sophisticated template-based generation when API unavailable
- **Secret Key Management**: Flexible API key loading from environment or file

### 🏗️ **Production-Ready Architecture**
- **Modular Design**: Completely decoupled components with clean interfaces
- **ROS Action Integration**: Standard `/tts_to_soundplay` and `/play_motion` servers
- **Error Handling**: Comprehensive fallbacks and graceful degradation
- **Environment Flexibility**: Works in simulation, with real robot, or standalone

## 📁 Project Structure

```
tiago_fortune_teller/
├── 🧠 src/
│   ├── main.py                        # Main orchestrator with PyKEEN integration
│   ├── user_input_node.py            # ROS terminal input handler
│   ├── 👁️ face_recognition/
│   │   ├── __init__.py
│   │   └── face_recognizer.py         # OpenCV LBPH face recognition
│   ├── 🕸️ knowledge_graph/
│   │   ├── __init__.py
│   │   └── graph_manager.py           # PyKEEN knowledge graph manager
│   ├── 💬 dialog/
│   │   ├── __init__.py
│   │   └── fortune_teller.py          # AI-powered fortune generation
│   └── 🤖 tiago_interface/
│       ├── __init__.py
│       ├── tiago_controller.py        # Enhanced robot control
│       └── pc_camera_manager.py       # Threaded camera management
├── 📊 data/                           # Auto-created data directories
│   ├── faces/                         # Face recognition database
│   │   ├── photos/                    # User photo archive
│   │   └── metadata/                  # Recognition metadata & models
│   ├── knowledge_graph/               # KG persistent storage
│   │   ├── triples.json              # Graph triples
│   │   ├── user_data.json            # User profiles
│   │   ├── interactions.json         # Interaction history
│   │   └── models/                   # Trained PyKEEN models
│   └── captured_images/              # Camera captures
└── 📖 README.md                      # This comprehensive guide
```

## 🔧 Technology Stack

### **Core Technologies**
- **ROS 1 (Noetic)**: Robot middleware and communication
- **Python 3.7+**: Primary programming language with threading
- **OpenCV**: Computer vision, face detection, and LBPH recognition
- **PyTorch**: Deep learning framework for PyKEEN embeddings

### **AI/ML Libraries**
- **PyKEEN**: Knowledge graph embeddings (TransE model with fallbacks)
- **OpenAI Python SDK**: GPT-4 powered natural language generation
- **NumPy/Pandas**: Data processing and numerical computation
- **scikit-learn**: Machine learning utilities

### **ROS Integration**
- **pal_interaction_msgs**: TTS action server (`/tts_to_soundplay`)
- **play_motion_msgs**: Gesture control (`/play_motion`)
- **std_msgs**: User input communication
- **actionlib**: ROS action server communication

### **Speech & Audio**
- **pyttsx3**: Offline text-to-speech synthesis
- **espeak**: System-level TTS fallback
- **Multi-tier TTS**: ROS → pyttsx3 → timing simulation

## 🚀 Quick Start Guide

### **Installation & Setup**

1. **Prerequisites:**
   ```bash
   # ROS Noetic (Ubuntu 20.04)
   sudo apt update
   sudo apt install ros-noetic-desktop-full
   
   # Python dependencies
   pip3 install opencv-python numpy pandas torch pykeen openai pyttsx3
   ```

2. **Environment Setup:**
   ```bash
   source /opt/ros/noetic/setup.bash
   export ROS_MASTER_URI="http://localhost:11311"
   export PYTHONPATH="${PWD}/src:$PYTHONPATH"
   
   # Optional: OpenAI API key
   export OPENAI_API_KEY="your-api-key"
   # OR create src/secret.txt with your API key
   ```

3. **Launch System:**
   ```bash
   # Terminal 1: ROS Core
   roscore
   
   # Terminal 2: User Input Handler
   python3 src/user_input_node.py
   
   # Terminal 3: Main Application
   python3 src/main.py
   ```

### **Configuration Options**

```bash
# Face recognition method
export FACE_RECOGNITION_METHOD="opencv"  # opencv, none

# Camera selection
export CAMERA_ID="0"  # Default webcam

# PyKEEN model configuration
export PYKEEN_MODEL="TransE"            # TransE, ComplEx, RotatE
export EMBEDDING_DIM="32"               # Embedding dimensions
```

## 🎮 Interactive Experience

### **Complete User Flow**

1. **🎭 Mystical Welcome**
   ```
   🤖 Tiago: "Greetings, traveler! Welcome to the mystical realm!"
   🤖 *performs wave gesture*
   📸 *PC camera captures user image*
   ```

2. **👤 Face Recognition & Enrollment**
   ```
   🤖 Tiago: "Let me see if we've met before..."
   🔍 *OpenCV face detection and LBPH recognition*
   📊 *Updates recognition count for returning users*
   ```

3. **📝 Interactive Data Collection**
   ```
   🤖 Tiago: "What name do you go by in this earthly realm?"
   👤 User: "Alice"
   🤖 Tiago: "How many cycles around the sun have you completed?"
   👤 User: "28"
   🤖 Tiago: "What calling occupies your days?"
   👤 User: "software engineer"
   ```

4. **🧠 Knowledge Graph Processing**
   ```
   🕸️ *Creates triples: (alice, has_profession, software_engineer)*
   🧮 *PyKEEN computes embeddings and relation predictions*
   📈 *Updates user visit count and interaction history*
   ```

5. **🔮 AI Fortune Generation**
   ```
   🤖 Tiago: "The knowledge graph reveals profound insights!"
   💭 *GPT-4 generates personalized fortune using KG context*
   🎭 *Delivers fortune with synchronized gestures*
   ```

### **Example Complete Session**

```
🔮 TIAGO FORTUNE TELLER - INTERACTIVE EXPERIENCE 🔮
Face Recognition Mode: OPENCV

🤖 Tiago: "Greetings, traveler! Welcome to the mystical realm of Tiago the Fortune Teller!"
🤖 *wave gesture*
📸 Image captured from PC camera
🔍 Face recognized: alice (confidence: 0.89)
🤖 Tiago: "Ah! I remember you! Welcome back, my friend."

🤖 Tiago: "Alice... I can feel the cosmic energy surrounding that name."
🧠 Knowledge Graph Update: alice -> visited -> fortune_teller (visit #3)
🔮 PyKEEN Predictions: alice -> interested_in -> technology (0.92)

🤖 Tiago: "The knowledge graph reveals profound insights!"
🤖 *crystal_ball_gaze gesture*
🤖 Tiago: "Alice, your analytical mind in software engineering continues to evolve. The cosmic algorithms show that your third journey here aligns with a breakthrough in code that bridges logic with intuition. The neural pathways of destiny converge on this truth."

🤖 *bow gesture*
🤖 Tiago: "May the wisdom of the stars guide your path!"
```

## 🔧 Architecture Deep Dive

### **PyKEEN Knowledge Graph Implementation**

```python
# Automatic triple generation
("alice", "has_profession", "software_engineer")
("alice", "belongs_to_age_group", "young_adult") 
("alice", "visited", "fortune_teller")
("software_engineer", "is_a", "profession")

# Dynamic relation predictions
predictions = kg.predict_relations("alice", top_k=3)
# → [{"relation": "interested_in", "object": "technology", "score": 0.92}]
```

### **OpenCV Face Recognition Pipeline**

```python
# LBPH (Local Binary Pattern Histogram) Recognition
1. Face Detection: Haar Cascade classifiers
2. Face Extraction: ROI normalization to 100x100
3. Feature Encoding: LBPH feature vectors
4. Recognition: Confidence-based matching (threshold: 80)
5. Database Update: Incremental learning with new faces
```

### **Multi-Tier TTS System**

```python
# Fallback hierarchy
1. ROS Action Server (/tts_to_soundplay) → PAL TTS
2. pyttsx3 → Local system TTS
3. Timing Simulation → Speech duration estimation
```

### **Threaded Camera Management**

```python
# Real-time image capture
- Background thread continuously captures frames
- Thread-safe image buffer with latest frame
- Automatic fallback image generation
- Face recognition format conversion (BGR→RGB)
```


## 🔧 Customization Guide

### **Adding New Gestures**

```python
# In tiago_controller.py
motion_mapping = {
    'your_custom_gesture': 'tiago_motion_name',
    'mystical_reveal': 'inspect_surroundings',
    'cosmic_blessing': 'reach_max'
}
```

### **Extending Fortune Templates**

```python
# In fortune_teller.py
self.profession_fortunes = {
    'data_scientist': [
        "Your algorithms will uncover patterns that reshape reality.",
        "The data whispers secrets only you can decode."
    ],
    'your_profession': [
        "Custom fortune template here..."
    ]
}
```

### **PyKEEN Model Configuration**

```python
# In graph_manager.py
self.embedding_dim = 64        # Higher for better accuracy
self.training_epochs = 50      # More epochs for better training
self.min_triples_for_training = 15  # Minimum data threshold
```

## 🐳 Deployment Options

### **Simulation Deployment**

```bash
# Launch Tiago Gazebo simulation
roslaunch tiago_gazebo tiago_gazebo.launch public_sim:=true

# Run fortune teller
export ROS_MASTER_URI=http://localhost:11311
python3 src/main.py
```


## 🔮 Research Applications

### **Human-Robot Interaction Studies**
- **Long-term Interaction**: User adaptation over multiple sessions
- **Personalization Effectiveness**: Knowledge graph impact on user engagement
- **Trust Building**: Face recognition accuracy vs. user comfort
- **Social Robotics**: Gesture-speech synchronization effectiveness

### **AI/ML Research Applications**
- **Knowledge Graph Evolution**: Dynamic graph learning from interactions
- **Multimodal Fusion**: Vision + NLP + robotics integration
- **Personalization Algorithms**: Embedding-based user modeling
- **Context-Aware Generation**: KG-informed natural language generation

### **Technical Contributions**
- **Modular ROS Architecture**: Template for interactive robot applications
- **Fallback System Design**: Graceful degradation in robot systems
- **Real-time Knowledge Graphs**: PyKEEN integration with live data
- **Multi-tier Speech Systems**: Robust TTS with multiple backends

