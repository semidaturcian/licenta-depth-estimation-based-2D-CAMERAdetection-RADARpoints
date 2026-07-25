from nuscenes.nuscenes import NuScenes

DATAROOT = "/home/semida/datasets/nuscenes"

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=DATAROOT,
    verbose=False
)

print("=" * 60)
print(f"Number of scenes: {len(nusc.scene)}")
print("=" * 60)

for i, scene in enumerate(nusc.scene):

    print(f"\nScene {i}")
    print(f"Name        : {scene['name']}")
    print(f"Description : {scene['description']}")
    print(f"Nbr samples : {scene['nbr_samples']}")