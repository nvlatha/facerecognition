#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import os
import face_recognition
import time
import h5py
import pickle
import shutil

# Source and destination paths
training_path = 'C:/Users/Learning/accelerate/dataset/train/train/'
cropped_training_path = 'C:/Users/Learning/accelerate/dataset/train/train_cropped/'
pickle_filename = "C:/Users/Learning/accelerate/dataset/trained_models/face_encodings_custom.pickle" # train model
h5_filename = "C:/Users/Learning/accelerate/dataset/trained_models/face_encodings_custom.h5" # train model
test_images_folder = "C:/Users/Learning/accelerate/dataset/test/test/"
test_matched_folder = "C:/Users/Learning/accelerate/dataset/test/test_matched/"
test_unmatched_folder = "C:/Users/Learning/accelerate/dataset/test/test_unmatched/"

# Define the desired sizes
initial_image_size = (300, 300)
cropped_image_size = (100, 100)

print("=====================================================================")
print("Execution Started for validating with the model trained.")
print("=====================================================================")

total_matched = 0  # Initialize total_matched
total_unmatched = 0

# Function to load encodings and names from H5 file
def load_encodings_from_h5(h5_filename):
    try:
        with h5py.File(h5_filename, "r") as hf:
            encodings = hf["encodings"][:]
            names = hf["names"][:]
        return encodings, names
    except Exception as e:
        print("Error loading encodings from H5 file:", e)
        return None, None

# Function to load encodings and names from pickle file
def load_encodings_from_pickle(pickle_filename):
    try:
        with open(pickle_filename, "rb") as f:
            data = pickle.load(f)
        encodings = data["encodings"]
        names = data["names"]
        return encodings, names
    except Exception as e:
        print("Error loading encodings from pickle file:", e)
        return None, None

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

# Create subdirectories for cropped training images
create_subdirectories(training_path, cropped_training_path)

# Process images, detect faces, crop faces, and save cropped images
start_time = time.time()

for root, _, files in os.walk(training_path):
    for file in files:
        image_path = os.path.join(root, file)
        try:
            _, resized_image = detect_and_resize_faces(image_path)

            if resized_image is not None:
                cropped_faces = detect_and_crop_faces(resized_image)

                if cropped_faces:
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

# Function to recognize faces in test images using encodings
def recognize_faces_in_test_images(encodings, names, test_images_folder):
    results = {}
    for root, _, files in os.walk(test_images_folder):
        for file in files:
            image_path = os.path.join(root, file)
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(img_rgb)
            face_encodings = face_recognition.face_encodings(img_rgb, face_locations)
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(encodings, face_encoding)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = names[first_match_index]
                    confidence = face_recognition.face_distance(encodings, face_encoding)[first_match_index]
                    print("Exact match found:", name.decode("utf-8"), "with confidence:", confidence, "for test image:", file)

                    # Create subfolder based on the name# Convert byte strings to regular strings
                    name_str = name.decode('utf-8')
                    # Remove leading 'b' and quotes
                    name_str = name_str[2:-1] if name_str.startswith("b'") else name_str

                    matched_folder = os.path.join(os.path.dirname(os.path.dirname(test_images_folder)), "test_matched", name_str)
                    os.makedirs(matched_folder, exist_ok=True)

                    # Move the matched file to the matched folder
                    try:
                        shutil.move(image_path, os.path.join(matched_folder, file))
                        global total_matched
                        total_matched += 1
                    except FileNotFoundError as e:
                        print(f"Error moving matched file: {e}")
                    except Exception as e:
                        print(f"An error occurred: {e}")
                else:
                    # Move the unmatched file to the unmatched folder
                    test_unmatched_folder = os.path.join(os.path.dirname(os.path.dirname(test_images_folder)), "test_unmatched")
                    os.makedirs(test_unmatched_folder, exist_ok=True)
                    shutil.move(image_path, os.path.join(test_unmatched_folder, file))
                    global total_unmatched
                    total_unmatched += 1
                results[file] = name
    return results

# Function to compare face recognition results from both sources
def compare_face_recognition_results(encodings_h5, names_h5, encodings_pickle, names_pickle, test_images_folder):
    results_h5 = recognize_faces_in_test_images(encodings_h5, names_h5, test_images_folder)
    results_pickle = recognize_faces_in_test_images(encodings_pickle, names_pickle, test_images_folder)

    accuracy_h5 = 0
    accuracy_pickle = 0
    total_images_h5 = len(results_h5)
    total_images_pickle = len(results_pickle)

    if total_images_h5 > 0:
        correct_h5 = sum(1 for image, name in results_h5.items() if results_pickle.get(image) == name)
        accuracy_h5 = (correct_h5 / total_images_h5) * 100

    if total_images_pickle > 0:
        correct_pickle = sum(1 for image, name in results_pickle.items() if results_h5.get(image) == name)
        accuracy_pickle = (correct_pickle / total_images_pickle) * 100

   # print("Accuracy for H5 file:", accuracy_h5)
    #print("Accuracy for pickle file:", accuracy_pickle)

# Load encodings and names from H5 file
encodings_h5, names_h5 = load_encodings_from_h5(h5_filename)

# Load encodings and names from pickle file
encodings_pickle, names_pickle = load_encodings_from_pickle(pickle_filename)

if encodings_h5 is not None and names_h5 is not None and encodings_pickle is not None and names_pickle is not None:
    # Compare face recognition results from both sources
    compare_face_recognition_results(encodings_h5, names_h5, encodings_pickle, names_pickle, test_images_folder)
else:
    print("Error: Unable to compare face recognition results due to missing encodings or names.")

end_time = time.time()
execution_time = end_time - start_time
total_files = total_matched + total_unmatched

print("Total execution time:", execution_time, "seconds")
print("Total files matched:", total_matched)
print("Total files not matched:", total_unmatched)
print("Total files processed:", total_files)

print("=====================================================================")
print("Execution completed for the test images.")
print("=====================================================================")