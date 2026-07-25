from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes

# Calea către dataset
DATAROOT = Path.home() / "datasets" / "nuscenes"

# Încarcă datasetul
nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(DATAROOT),
    verbose=False
)

# Prima scenă
scene = nusc.scene[0]

# Primul sample din scenă
sample = nusc.get("sample", scene["first_sample_token"])

# Token-ul imaginii frontale
cam_front_token = sample["data"]["CAM_FRONT"]

# Informațiile despre imagine
cam_front = nusc.get("sample_data", cam_front_token)

print("=" * 60)
print("CAM_FRONT information")
print("=" * 60)

for key, value in cam_front.items():
    print(f"{key}: {value}")

# Construiește calea completă către imagine
image_path = DATAROOT / cam_front["filename"]

print("\nImage path:")
print(image_path)

# Citește imaginea cu OpenCV
image = cv2.imread(str(image_path))

if image is None:
    raise RuntimeError("Imaginea nu a putut fi încărcată.")

print(f"\nImage shape: {image.shape}")

# OpenCV citește imaginea în format BGR.
# Matplotlib așteaptă formatul RGB.
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Afișează imaginea
plt.figure(figsize=(12, 7))
plt.imshow(image_rgb)
plt.title("CAM_FRONT")
plt.axis("off")

output_path = "cam_front.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")

print(f"\nImage saved to: {output_path}")