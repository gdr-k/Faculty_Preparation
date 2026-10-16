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
# CHECKING THE VALIDATION OF THE DATASET
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

#Checking class imbalance
class_ids,class_counts=np.unique(labels,return_counts=True)
print(f"Class IDs: {class_ids}",f"Class counts: {class_counts} ")
for class_id, class_count in zip(class_ids, class_counts):
    percentage = (class_count / len(labels)) * 100
    print(f"Class {class_id} has {class_count} samples, which is {percentage:.2f}% of the dataset.")

#normalizing the images
images_float = images.astype(np.float32) / 255.0
print(f"Normalized images shape: {images_float.shape}")
print(f"Normalized images data type: {images_float.dtype}")
print(f"Normalized images min value: {images_float.min()}")
print(f"Normalized images max value: {images_float.max()}")

#Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# First, reserve 20% for the test set
X_train_val, X_test, y_train_val, y_test = train_test_split(
    images_float,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

# Divide the remaining data into training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val
)

print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Testing:", X_test.shape, y_test.shape)
#Convert the arrays into pytorch tensors
import torch

# Add a channel dimension for grayscale images
X_train = X_train[:, np.newaxis, :, :]
X_val = X_val[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]

# Convert images and labels to tensors
X_train_tensor = torch.from_numpy(
    np.ascontiguousarray(X_train)
).float()

y_train_tensor = torch.from_numpy(y_train).long()

X_val_tensor = torch.from_numpy(
    np.ascontiguousarray(X_val)
).float()

y_val_tensor = torch.from_numpy(y_val).long()

X_test_tensor = torch.from_numpy(
    np.ascontiguousarray(X_test)
).float()

y_test_tensor = torch.from_numpy(y_test).long()

print("Training images:", X_train_tensor.shape)
print("Training labels:", y_train_tensor.shape)
print("Validation images:", X_val_tensor.shape)
print("Validation labels:", y_val_tensor.shape)
print("Testing images:", X_test_tensor.shape)
print("Testing labels:", y_test_tensor.shape)
