from nuscenes.nuscenes import NuScenes

DATAROOT = "/home/semida/datasets/nuscenes"

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=DATAROOT,
    verbose=False
)

# Prima scenă
scene = nusc.scene[0]

print("=" * 60)
print("Scene")
print("=" * 60)

print(f"Name: {scene['name']}")
print(f"Description: {scene['description']}")
print()

# Primul sample
sample = nusc.get("sample", scene["first_sample_token"])

print("=" * 60)
print("Sample Keys")
print("=" * 60)

for key in sample.keys():
    print(key)

print("\n")

print("=" * 60)
print("Available Sensors")
print("=" * 60)

for sensor in sample["data"]:
    print(sensor)