import pickle

paths = [f"chip_{i}" for i in range(330)]

with open("ours_images_paths_sm0.objects", "wb") as f:
    pickle.dump(paths, f)
