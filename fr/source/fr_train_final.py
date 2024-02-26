import cv2
import numpy as np
import os
import face_recognition
import time
import h5py
import pickle
import shutil

# Source and destination paths
training_path = 'C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/train/train/'
cropped_training_path = 'C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/train/train_cropped/'
pickle_filename = "C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/trained_models/face_encodings_custom.pickle" # train model
h5_filename = "C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/trained_models/face_encodings_custom.h5" # train model

# Define the desired sizes
initial_image_size = (300, 300)
cropped_image_size = (100, 100)

print("=====================================================================")
print("Execution Started  for the Model training using Face Recognition.")
print("=====================================================================")

# Function to detect and resize faces
def detect_and_resize_faces(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image:", image_path)
        return None

    # Resize the image
    resized_image = cv2.resize(img, initial_image_size)

    return img, resized_image

# Function to detect and crop faces from resized image
def detect_and_crop_faces(resized_image):
    face_locations = face_recognition.face_locations(resized_image)

    cropped_faces = []
    if face_locations:
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = resized_image[top:bottom, left:right]
            face_image_resized = cv2.resize(face_image, cropped_image_size)

            cropped_faces.append(face_image_resized)

    return cropped_faces

# Function to create subdirectories
def create_subdirectories(source_path, dest_path):
    subdirs = [subdir for subdir in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, subdir))]
    for subdir in subdirs:
        os.makedirs(os.path.join(dest_path, subdir), exist_ok=True)

# Function to move cropped images to corresponding subdirectories
def move_cropped_images(source_path, dest_path):
    for root, _, files in os.walk(source_path):
        for file in files:
            src_file = os.path.join(root, file)
            person_name = os.path.basename(root)
            dest_dir = os.path.join(dest_path, person_name)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(src_file, dest_dir)

# Function to display cropped images
def display_cropped_images(cropped_faces):
    for i, cropped_face in enumerate(cropped_faces):
        cv2.imshow(f"Cropped Face {i+1}", cropped_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Create subdirectories in training_cropped
create_subdirectories(training_path, cropped_training_path)

# Process images, detect faces, crop faces, and save cropped images
for root, _, files in os.walk(training_path):
    for file in files:
        image_path = os.path.join(root, file)
        try:
            _, resized_image = detect_and_resize_faces(image_path)

            if resized_image is not None:
                cropped_faces = detect_and_crop_faces(resized_image)

                if cropped_faces:
                    #display_cropped_images(cropped_faces)

                    for i, cropped_face in enumerate(cropped_faces):
                        person_name = os.path.basename(root)
                        cv2.imwrite(os.path.join(cropped_training_path, person_name, f"{file[:-4]}_{i}.jpg"), cropped_face)

        except Exception as e:
            print("Error processing image:", image_path)
            print(e)

# Encoding faces from cropped images
list_encodings = []
list_names = []

for root, _, files in os.walk(cropped_training_path):
    for file in files:
        image_path = os.path.join(root, file)
        name = os.path.basename(root)
        try:
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encoding = face_recognition.face_encodings(img_rgb)
            if encoding:
                list_encodings.append(encoding[0])
                list_names.append(name)
        except Exception as e:
            print("Error encoding image:", image_path)
            print(e)

# Save the encodings and names in a pickle file
encodings_data = {"encodings": list_encodings, "names": list_names}
with open(pickle_filename, "wb") as f:
    pickle.dump(encodings_data, f)

# Save the encodings and names in an H5 file
with h5py.File(h5_filename, "w") as hf:
    hf.create_dataset("encodings", data=np.array(list_encodings))
    hf.create_dataset("names", data=np.array(list_names, dtype='S'))

print("=====================================================================")
print("Execution compelted  for the Model training using Face Recognition.")
print("=====================================================================")
