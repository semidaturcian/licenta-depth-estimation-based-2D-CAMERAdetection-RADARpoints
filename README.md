# Estimarea distanței prin fuziunea datelor Cameră–RADAR

Proiectul realizează estimarea distanței până la obiectele detectate,
folosind imagini de la cameră și puncte RADAR din datasetul nuScenes.

## 1. Cerințe

- Ubuntu 22.04
- ROS2 Humble
- Python 3.10
- Conda
- Visual Studio Code (opțional)
- Dataset nuScenes v1.0-mini

## 2. Instalarea ROS2

Se utilizează ROS2 Humble pe Ubuntu 22.04.

După instalarea ROS2, se încarcă mediul ROS2:

source /opt/ros/humble/setup.bash

## 3. Mediul Python

Se creează mediul Conda:

conda create -n nuscenes_env python=3.10

Se activează mediul:

conda activate nuscenes_env

Se instalează bibliotecile necesare:

pip install nuscenes-devkit
pip install ultralytics
pip install opencv-python
pip install matplotlib
pip install numpy

Pentru integrarea imaginilor cu ROS2:

sudo apt install ros-humble-cv-bridge

## 4. Dataset-ul nuScenes

Se descarcă datasetul nuScenes v1.0-mini și se extrage într-un folder local.

Structura trebuie să conțină folderul:

data/nuscenes

Proiectul folosește:
- CAM_FRONT
- RADAR_FRONT

## 5. Compilarea proiectului

Se intră în folderul proiectului:

cd licenta-depth-estimation-based-2D-CAMERAdetection-RADARpoints

Se construiește proiectul:

colcon build

După compilare se încarcă mediul ROS2 al proiectului:

source install/setup.sh

## 6. Rularea proiectului

Se deschid terminale separate.

### Terminal 1 – încărcarea datelor nuScenes

source /opt/ros/humble/setup.bash
source install/setup.sh

ros2 run data_sensors nuscenes_player

### Terminal 2 – detecția YOLOv11

source /opt/ros/humble/setup.bash
source install/setup.sh

ros2 run boundingbox_detection detection_node

### Terminal 3 – fuziunea Cameră–RADAR

source /opt/ros/humble/setup.bash
source install/setup.sh

ros2 run radar_fusion projection_node

## 7. Vizualizare

Pentru vizualizarea topicurilor ROS2 se poate folosi:

rviz2

Pentru verificarea topicurilor disponibile:

ros2 topic list

## 8. Fluxul aplicației

nuScenes
   ↓
NuScenes Player
   ↓
Camera + RADAR
   ↓
ROS2 topics
   ↓
YOLOv11 → bounding box-uri
   ↓
Proiecție RADAR
   ↓
Fuziune RADAR–Cameră
   ↓
Estimarea distanței
   ↓
Comparare cu Ground Truth