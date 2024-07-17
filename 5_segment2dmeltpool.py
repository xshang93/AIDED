"""segment2dMeltpool.py

Segment the last meltpool from the 2D meltpool
"""

""" Install required libraries """
import subprocess
import sys
import os

def install(package):
    print('---Installing ' + package + '---')
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# install('numpy')
# install('torch')
# install('torchvision')
# install('matplotlib')
# install('opencv-python')
# install('git+https://github.com/facebookresearch/segment-anything.git')

""" Import libraries """
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

""" Change following variables if needed """
backup = False # If backup folder exists
max_mask_area = 30000 # Maximum area of the mask

""" Set paths """
cwd = '/home/xiao/projects/DED/BO_processing/images/20230626_2d/test/'
orig_path = os.path.join(cwd, 'original_data') # the folder that contains original data
new_path = os.path.join(cwd, 'extracted_meltpool') # the folder that will contain processed data
sam_path = os.path.join('/home/xiao/projects/DED/BO_processing/SAM/', 'sam_vit_h_4b8939.pth') # location of the sam model file
subfolders = ['aa','bb'] # list of folders

print('---Following paths are set---')
print(' orig_path: ' + orig_path)
print(' new_path: ' + new_path)
print(' sam_path: ' + sam_path)

""" Download sam model if it doesn't exist """
def show_progress(block_num, block_size, total_size):
    print(round(block_num * block_size / total_size *100,2), end="\r")

if os.path.exists(sam_path) == False:
    url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
    print('---Downloading ' + url + '---')
    from urllib import request
    request.urlretrieve(url, sam_path, show_progress)

""" Create a SAM automatic mask generator tool """
print('---Creating a SAM automatic mask generator tool---')

model_type = "vit_h"
device = "cpu"
print(' device: ' + device)
print(' model_type: ' + model_type)

sam = sam_model_registry[model_type](checkpoint=sam_path)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(model=sam,min_mask_region_area=15000)

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

""" Make folder under specified path if it doesn't exist """
def make_folder(path):
    if os.path.exists(path) == False:
        print('Creating a folder: ' + path)
        os.mkdir(path)

""" Segment a correct(?) melt pool, draw rectangle, and save to local """
def extract_mask_and_save(path, masks):
    if len(masks) == 0:
        return
    
    print('Sorting masks by descending area...')
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    print('Removing masks with area bigger than ' + str(max_mask_area) + '...')
    filtered_masks = list(filter(lambda k: k['area'] <= max_mask_area, sorted_masks))
    
    print('Extacting a single mask...')
    mask = filtered_masks[0]
    m = mask['segmentation']
    img = np.ones((m.shape[0], m.shape[1], 4))
    color_mask = np.concatenate([[255,255,255], [1]])
    img[m] = color_mask
    
    # print('Drawing a boundary around the mask...')
    bbox = mask['bbox']
    # cv2.rectangle(img, (bbox[0]-2, bbox[1]-2), (bbox[0]+bbox[2]+2, bbox[1]+bbox[3]+2), (0,0,255), 2)
    
    center_x = int(bbox[0]+bbox[2]/2)
    center_y = int(bbox[1]+bbox[3]/2)
    crop_box = mask['crop_box']
    width = crop_box[2]
    height = crop_box[3]
    
    print('Centering and cropping image...')
    img = center_crop_image(img, width, height, center_x, center_y)
    
    print('Saving ' + path + ' to ' + new_path + '...')
    cv2.imwrite(os.path.join(new_path, path), img)
    

""" Process dataset """
def process_images(folder_path):
    make_folder(os.path.join(new_path, folder_path))
    filenames = get_file_names(folder_path)

    for filename in filenames:
        path = os.path.join(folder_path, filename)
        print('Reading image...')
        image = cv2.imread(os.path.join(orig_path, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print('Generating masks...')
        masks = mask_generator.generate(image)
        extract_mask_and_save(path, masks)

""" Center and crop image """
def center_crop_image(img,w,h,x,y):
    left,right = int(x-115),int(x+115)
    bottom,top = int(y-115),int(y+115)
    
    print(str(top) + ', ' + str(bottom) + ', ' + str(left) + ', ' + str(right))
    
    d_left = abs(left) if left < 0 else 0
    d_right = abs(w - right) if w - right < 0 else 0
    d_bottom = abs(bottom) if bottom < 0 else 0
    d_top = abs(h - top) if h - top < 0 else 0
    
    left = 0 if d_left > 0 else left
    bottom = 0 if d_bottom > 0 else bottom    
    
    img = cv2.copyMakeBorder(img, d_top, d_bottom, d_left, d_right, cv2.BORDER_CONSTANT, 0)
    img = img[bottom:bottom+230, left:left+230]
    
    return img

""" Loop through directories """
make_folder(new_path)
for subfolder in subfolders:
  print('---Processing images inside ' + orig_path + '/' + subfolder + '---')
  process_images(subfolder)
  
  if backup:
    folder_path = os.path.join(subfolder, 'backup')
    print('---Processing images inside ' + orig_path+ '/' + folder_path + '---')
    process_images(folder_path)
