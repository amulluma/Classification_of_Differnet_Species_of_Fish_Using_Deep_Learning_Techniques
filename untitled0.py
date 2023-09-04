# Subject: Processing and Preservation of Aquacultural Products
# Project Title: Image Based Classification of Different Aquacultural Species using Machine Learning Techniques
# Group Members:
#       1) Dnyaneshwar Gawai          22AG65R11
#       2) Anurag Solanki             22AG65R06
#       3) Samarth Srivastava         22AG65R12
#       4) Vidhya Parshuramkar        22AG63R28
#       5) Tejas Chaudhari            22AG63R29 
#       6) Piyush Chouhan             22AG63R30

#%%

import os
import cv2
import numpy as np
print (np.__version__)
from keras.preprocessing.image import ImageDataGenerator


#%%

# input_folder1 =
# for label in (input_folder1):
#     img_path = os.path.join(input_folder1, label)
#     img = cv2.imread(img_path)

#%%  

# Set parameters for data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')


# Define a function to preprocess each image
def preprocess_image(image_path, new_size):
    # Load the image
    image = cv2.imread(image_path)
# Resize the image to a standard size
    image = cv2.resize(image, new_size)
    
    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize the image to the range [0, 1]
    image = image.astype("float32") / 255.0
    
    # Apply data augmentation
    image = np.expand_dims(image, axis=2)
    image = datagen.random_transform(image)
    image = np.squeeze(image, axis=2)
    
    # Standardize the image
    mean, std = image.mean(), image.std()
    image = (image - mean) / std
    
    return image
#%%

# Define the paths to the input and output folders
input_folder = "C:/Users/Tanaji/Desktop/MTech/Aqua 2nd Sem/Aqua Product/Project/Data"
output_folder = "C:/Users/Tanaji/Desktop/MTech/Aqua 2nd Sem/Aqua Product/Project/Preprocessed Data"

# Loop over the input folders and preprocess each image
for label in os.listdir(input_folder):
    label_folder = os.path.join(input_folder, label)
    output_label_folder = os.path.join(output_folder, label)
    if not os.path.exists(output_label_folder):
        os.makedirs(output_label_folder)
    for image_name in os.listdir(label_folder):
        image_path = os.path.join(label_folder, image_name)
        output_path = os.path.join(output_label_folder, image_name)
        # Preprocess the image and save it to the output folder
        image = preprocess_image(image_path, new_size=(224, 224))
        cv2.imwrite(output_path, image*255.0)

