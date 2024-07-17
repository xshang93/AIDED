# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 18:40:19 2023

@author: evk77
"""
import pandas as pd
import re
import os
import cv2
import shutil

def modify_filenames(csv_file_path, images_directory,output_directory):
    # Load the CSV file
    data = pd.read_csv(csv_file_path)
    
    # Assuming the 'id' column matches the original file number
    filename_map = {row['id']: f"{row['id']}_P{row['Power']}_V{int(row['Speed'] * 60)}_0_{row['rpm']}.jpg" for index, row in data.iterrows()}
    
    # Iterate over the files in the images directory
    for file in os.listdir(images_directory):
        if file.endswith('.jpg'):
            # Extract the numeric identifier from the filename
            match = re.match(r"(\d+)\.jpg", file)
            if match:
                file_id = int(match.group(1))
                # Check if this file_id is in the map
                if file_id in filename_map:
                    original_file_path = os.path.join(images_directory, file)
                    new_file_path = os.path.join(output_directory, filename_map[file_id])
                    # os.rename(original_file_path, new_file_path)
                    shutil.copyfile(original_file_path,new_file_path)

# Set the path to the directory containing your photos
photo_directory = '/home/xiao/projects/DED/BO_processing/Final_data/single_tracks/raw'
output_directory = '/home/xiao/projects/DED/BO_processing/Final_data/single_tracks/cropped'

# Define the percentage of width to be cropped from both sides
crop_percentage = 20

# List all the files in the directory
photo_files = os.listdir(photo_directory)

# Iterate through the files in the directory
for file_name in photo_files:
    file_path = os.path.join(photo_directory, file_name)
    output_file_path = os.path.join(output_directory, file_name)
    
    # Load the image using OpenCV
    image = cv2.imread(file_path)
    
    if image is not None:
        # Get the original image width and height
        height, width, _ = image.shape
        
        # Calculate the amount to crop from both sides
        crop_amount = int(width * (crop_percentage / 100))
        
        # Crop the image
        cropped_image = image[:, crop_amount:width - crop_amount]
        
        # Save the cropped image back to the original file path, effectively overwriting it
        cv2.imwrite(output_file_path, cropped_image)
    else:
        print(f"Error: Failed to load the image from {file_path}")

print("Images have been cropped and saved in the 'cropped' directory.")

csv_file_path = output_directory+'/parameters.csv'  # Update this path to your actual CSV file location
modify_filenames(csv_file_path, photo_directory,output_directory)