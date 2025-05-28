FROM registry.gitlab.com/brienza1/empower_docker:latest

WORKDIR /app

# Install face recognition dependencies
RUN apt-get update && \
    apt-get install -y build-essential cmake python3-dev python3-pip libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev && \
    pip install --upgrade pip

# Install ROS packages for TIAGo simulation and interaction
RUN apt-get update && \
    apt-get install -y \
    ros-noetic-rospy \
    ros-noetic-actionlib \
    ros-noetic-pal-interaction-msgs \
    ros-noetic-play-motion-msgs \
    ros-noetic-tiago-simulation \
    ros-noetic-pal-interaction \
    ros-noetic-play-motion \
    ros-noetic-tiago-gazebo \
    ros-noetic-tiago-description \
    ros-noetic-tiago-navigation \
    && rm -rf /var/lib/apt/lists/*

# Add Python dependencies
COPY docker/requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/