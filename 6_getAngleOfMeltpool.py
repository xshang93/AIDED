"""getAngleOfMeltpool.py

Get an angle of the meltpool and process the dataset.
Write a .csv file that contains index, hs spacing, and angle of the dataset
"""

""" Install required libraries """
import subprocess
import sys
import os

def install(package):
    print('---Installing ' + package + '---')
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# install('opencv-python')
# install('numpy')

""" Import libraries """
import numpy as np
import cv2
import csv
import math

""" Modify values """
cwd = '/home/xiao/projects/DED/BO_processing/images/20230626_2d/test/'
orig_path = os.path.join(cwd, 'extracted_meltpool') # the folder that contains the dataset
new_path = os.path.join(cwd, 'processed_data') # the folder that will contain processed data
csv_path = cwd+'dataset.csv'
subfolders = ['aa','bb'] # list of folders
backup = False
color = [76,177,34]

print('---Following paths are set---')
print(' orig_path: ' + orig_path)
print(' new_path: ' + new_path)

""" Write headers in csv file """
def write_headers():
    with open(csv_path, mode='w', newline='') as csv_file:
        # Headers for the csv file
        headers = ['index', 'hs_spacing', 'angle', 'file_path']
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        # Close csv file
        csv_file.close()


""" Write a row in csv file """
def write_row(file_name, file_path, angle):
    with open(csv_path, mode='a', newline='') as csv_file: 
        headers = ['index', 'hs_spacing', 'angle', 'file_path']
        writer = csv.DictWriter(csv_file, fieldnames=headers)    
        # Find the index and hs spacing
        index = ''
        hs_spacing_index = ''
        file_name = file_name.replace('.png', '')
        if 'image' in file_name:
            file_name = file_name.replace('image', '')
            indexes = file_name.split('_')
            index = indexes[0]
            hs_spacing_index = indexes[-1]
            hs_spacing = '0.' + str(4 + int(hs_spacing_index))
        else:
            indexes = file_name.split('-')
            index = indexes[0]
            hs_spacing = '0.' + indexes[-2]

        # Write the row
        row = {'index': index, 'hs_spacing': hs_spacing, 'angle': angle, 'file_path': file_path}
        print(row)
        writer.writerow(row)

        # Close csv file
        csv_file.close()

""" Make folder under specified path if it doesn't exist """
def make_folder(path):
    if os.path.exists(path) == False:
        print('Creating a folder: ' + path)
        os.mkdir(path)
        
""" Get image file names under specified directory """
def get_file_names(folder_path):
    dir_path = os.path.join(orig_path, folder_path)
    # list to store files
    filenames = []

    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            # check if current path exists in a file
            if os.path.isfile(os.path.join(new_path, folder_path, path)) == False:
                filenames.append(path)
    return filenames    

""" Save image """
def save_image(deg, image):
    file_path = os.path.join(new_path, subfolder, str(deg))
    extension = '.png'
    # check if the file name already exists, and change extension
    if os.path.isfile(file_path + '.png'):
        i = 0
        extension = '(' + i + ').png'
        while os.path.isfile(file_path + extension):
            i += 1
            extension = '(' + i + ').png'
    cv2.imwrite(file_path + extension, image)
    return (str(deg) + extension)

""" Check if there are manual points in the image """
def check_auto_or_manual(file_path, subfolder):
    # Load image
    im = cv2.imread(file_path)
    # Check if there are manual points
    colorExists = np.count_nonzero(np.all(im==color,axis=2))
    print(colorExists)
    if colorExists:
        print('Color exists. Points have manually been specified')
        angle, saved_path = get_angle_with_manual_points(file_path, subfolder)
    else:
        print('Automantically selecting points')
        angle, saved_path = get_angle_with_auto_points(file_path, subfolder)
    return angle, saved_path

""" Get angle of manual points """
def get_angle_with_manual_points(file_path, subfolder):
    # Load image
    im = cv2.imread(file_path)
    # Get x and y coordinates of specified colored points
    Y, X = np.where(np.all(im==color,axis=2))
    # Get the coord of left and right most specified color pixel
    l = np.argmin(X)
    lx,ly = X[l],Y[l]
    r = np.argmax(X)
    rx,ry = X[r],Y[r]
    # Convert an image from BGR to grayscale mode 
    grayIm = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Convert a grayscale image to black and white using binary thresholding 
    (thresh, BnWIm) = cv2.threshold(grayIm, 125, 255, cv2.THRESH_BINARY)
    BnWIm = cv2.cvtColor(BnWIm, cv2.COLOR_GRAY2BGR)
    # Define white
    white = [255,255,255]
    # Get x and y coordinates of white points
    Y,X = np.where(np.all(BnWIm==white,axis=2))
    # Draw lines
    if ly < ry:
        im = cv2.line(im, (X[np.where(Y==ry)[0][0]], ry), (rx, ry), (255,0,0), 1)
    elif ly >= ry:
        im = cv2.line(im, (lx, ly), (X[np.where(Y==ly)[-1][-1]], ly), (255,0,0), 1)
    im = cv2.line(im, (lx, ly), (rx, ry), (0,255,0), 1)
    # Get angles
    deg = round(math.degrees(math.atan2(abs(ly-ry), abs(lx-rx))),3)
    print('Angle = ' + str(deg))
    # Save image as its angle
    image_name = save_image(deg, im)

    return deg, os.path.join(subfolder, image_name)

""" Get angle of leftmost and rightmost points """
def get_angle_with_auto_points(file_path, subfolder):
    # Load image
    im = cv2.imread(file_path)
    # Convert an image from BGR to grayscale mode 
    grayIm = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Convert a grayscale image to black and white using binary thresholding 
    (thresh, BnWIm) = cv2.threshold(grayIm, 125, 255, cv2.THRESH_BINARY)
    BnWIm = cv2.cvtColor(BnWIm, cv2.COLOR_GRAY2BGR)
    # Define white
    white = [255,255,255]
    # Get x and y coordinates of white points
    Y,X = np.where(np.all(BnWIm==white,axis=2))
    # Get the coord of left and right most white pixel
    l = np.argmin(X)
    lx,ly = X[l],Y[l]
    r = np.argmax(X)
    rx,ry = X[r],Y[r]
    # Draw lines
    if ly < ry:
        im = cv2.line(im, (X[np.where(Y==ry)[0][0]], ry), (rx, ry), (255,0,0), 1)
    elif ly >= ry:
        im = cv2.line(im, (lx, ly), (X[np.where(Y==ly)[-1][-1]], ly), (255,0,0), 1)
    im = cv2.line(im, (lx, ly), (rx, ry), (0,255,0), 1)
    # Get angles
    deg = round(math.degrees(math.atan2(abs(ly-ry), abs(lx-rx))),3)
    print('Angle = ' + str(deg))
    # Save image as its angle
    image_name = save_image(deg, im)

    return deg, os.path.join(subfolder, image_name)

""" Process images """
def process_images(subfolder):
    make_folder(os.path.join(new_path, subfolder))
    filenames = get_file_names(subfolder)

    for filename in filenames:
        path = os.path.join(orig_path, subfolder, filename)
        print('Getting angle for ' + path)
        angle, saved_path = check_auto_or_manual(path, subfolder)
        write_row(filename, saved_path, angle)   

""" Loop through directories """
write_headers()
make_folder(new_path)
for subfolder in subfolders:
  print('---Processing images inside ' + orig_path + '/' + subfolder + '---')
  process_images(subfolder)

  if backup:
    folder_path = os.path.join(subfolder, 'backup')
    print('---Processing images inside ' + orig_path+ '/' + folder_path + '---')
    process_images(folder_path)