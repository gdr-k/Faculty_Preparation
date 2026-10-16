import numpy as np
rng=np.random.default_rng(42)
images=rng.integers(0,25,size=(100,32,32),dtype=np.uint8)
#print(images)
labels=np.array([0]*60 + [1] * 40)
print(labels)
indices = rng.permutation(len(labels))
#print(indices)
images=images[indices]
labels=labels[indices]
print(images[8])
print(labels[8])
print(f"Image shape is {images.shape}")
print(f"labels shape is {labels.shape}")
print(f"Number of images {len(images)}")
print(f"Number of labels {len(labels)}")
print(f"Data type of images {type(images)}")
print(f"Data type of labels {type(labels)}")
print(f"Size of images {images.size}")
print(f"Size of labels {labels.size}")
print(f"All images is finite{np.isfinite(images).all()}")
print(f"Labels used are {np.unique(labels)}")

def validate_dataset(images, labels):
    if len(images) != len(labels):
        raise ValueError("Images and labels have different lengths.")

    if len(images) == 0:
        raise ValueError("The dataset is empty.")

    if not np.isfinite(images).all():
        raise ValueError("Images contain NaN or infinity.")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Unexpected label found.")

    print("Dataset validation successful.")

validate_dataset(images, labels)