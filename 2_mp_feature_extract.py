#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 18:47:31 2023

@author: xiao
"""
from ultralytics import YOLO
import glob
import sys
sys.path.insert(1,'/home/xiao/')
#from helper_func import remove_prefix,remove_suffix
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
import pandas as pd
import cv2
import os
import shutil

def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

def remove_prefix(input_string, prefix):
    if prefix and input_string.startswith(prefix):
        return input_string[len(prefix):]
    return input_string

# Load model check point and prepare the model
sam = sam_model_registry["vit_h"](checkpoint='/home/xiao/projects/DED/BO_processing/SAM/sam_vit_h_4b8939.pth')
predictor = SamPredictor(sam)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
  
def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def mp_ext(img):
    img_name = remove_prefix(img,img_dir)
    img_name = remove_suffix(img_name,'.jpg')
    
    try:
        result = model.predict(img, save=True, save_txt=True)  # save predictions as labels
        xyxy = result[0].boxes.xyxy[0]
    except:
        print(img_name)
        return [0]*9
        
    # now pick 5 points around the centre point by an offset as label points for SAM
    x1 = float(xyxy[0])
    y1 = float(xyxy[1])
    x2 = float(xyxy[2])
    y2 = float(xyxy[3])
    centre = [(x1+x2)/2,(y1+y2)/2]
    offset_x = abs(x1-x2)/20
    offset_y = abs(y1-y2)/20
    p1 = [centre[0]+offset_x,centre[1]+offset_y]
    p2 = [centre[0]-offset_x,centre[1]+offset_y]
    p3 = [centre[0]+offset_x,centre[1]-offset_y]
    p4 = [centre[0]-offset_x,centre[1]-offset_y]
    
    # Tell SAM to remember this image
    image = cv2.imread(img)
    predictor.set_image(image)
    # Specify a point on the melt track. 
    # Points are inpumask_foldert to the model in (x,y) format and come with labels 1 (foreground point) or 0 (background point)
    input_point = np.array([centre])
    input_label = np.array([1])
    
    # Make prediction
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
    
    # add extra points for the melt track
    input_point = np.array([p1,p2,p3,p4])
    input_label = np.array([1,1,1,1])
    
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )
    
    # calculate the coordiates of the left and right extrme points of the meltpool
    mask = masks[0]
    mask_x = np.sum(mask,axis = 0)
    x_ext_l = (mask_x!=0).argmax(axis=0)
    x_ext_r = x_ext_l+np.count_nonzero(mask_x)-1
    y_ext_l = (mask[:,x_ext_l]!=0).argmax(axis=0)+int((np.count_nonzero(mask[:,x_ext_l])-1)/2)
    y_ext_r = (mask[:,x_ext_r]!=0).argmax(axis=0)+int((np.count_nonzero(mask[:,x_ext_r])-1)/2)
    y_centre = (y_ext_l+y_ext_r)/2
    x_centre = (x_ext_l+x_ext_r)/2
    
    mask_y = np.sum(mask,axis = 1)
    y_ext_high = (mask_y!=0).argmax(axis=0)
    y_ext_low = y_ext_high+np.count_nonzero(mask_y)-1
    
    # extract meltpool width and tilt angle
    width = ((x_ext_l-x_ext_r)**2+(y_ext_l-y_ext_r)**2)**0.5
    angle = np.arctan((y_ext_l-y_ext_r)/(x_ext_l-x_ext_r)) # in radius
    height = abs(y_ext_high-y_centre)
    depth = abs(y_ext_low-y_centre)
    A_d = mask[0:int((y_ext_l+y_ext_r)/2)].sum()
    A = mask.sum()
    dilution = A_d/A

    # show and save the plot
    f,ax = plt.subplots(1,1)
    plt.imshow(image)
    plt.plot([x_centre,x_centre],[y_centre,y_ext_high],linewidth=1)
    plt.plot([x_centre,x_centre],[y_centre,y_ext_low],linewidth=1)
    plt.plot([x_ext_l,x_ext_r],[y_ext_l,y_ext_r],linewidth=2)
    show_mask(masks, ax)
    show_points(input_point, input_label, ax)
    f.savefig(img_dir+remove_suffix(img_name,'.jpg')+'_masked.png')
    plt.close()    
    
    # release the current image
    predictor.reset_image()
    
    return xyxy,img_name,width,angle,height,depth,mask,[y_ext_l,y_ext_r],dilution

if __name__ == "__main__":
    # load and use existing trained model to make predictions
    model = YOLO('/home/xiao/git_projects/auto_AM_researcher/Xiao/MeltPool_MLWorkFlow/pre_trained_models/yolov8_best.pt')  # load a pretrained YOLOv8n model
    
    batch_num = 5
    
    img_dir = '/home/xiao/projects/DED/BO_processing/Final_data/single_tracks/cropped/batch_{0}'.format(batch_num)
    #/batch{0}'.format(batch_num)
    imgs = glob.glob(img_dir+'/*.jpg')
    powers = []
    speeds = []
    rpms = []
    indices = []
    xyxys = []
    widths = []
    angles = []
    heights = []
    depths = []
    masks = []
    y_exts=[]
    dilutions = []
    
    counter = 0
    for img in imgs:
        xyxy,img_name,width,angle,height,depth,mask,y_ext,dilution = mp_ext(img)
        if img_name != 0:
            img_name_list = img_name.split('_')
            powers.append(remove_prefix(img_name_list[1],'P'))
            speeds.append(remove_prefix(img_name_list[2],'V'))
            #rpms.append(float(remove_prefix(img_name_list[0],'_0_'))/10) #added \ for windows
            if int(img_name_list[4]) in [1,3,5,7]:
                rpms.append((float(img_name_list[4])+1)/10)
            else:
                rpms.append(float(img_name_list[4])/10)
            indices.append(img_name_list[1])
            xyxys.append(xyxy) # register xyxy to list
            widths.append(width)
            angles.append(angle)
            heights.append(height)
            depths.append(depth)
            masks.append(mask)
            y_exts.append(y_ext)
            dilutions.append(dilution)
        
            counter = counter+1
            if counter%10==0:
                df = pd.DataFrame({'Power': powers, 'Speed': speeds,'Width_mp': widths,
                                   'Height':heights, 'Depth': depths, 
                                   'Angle':angles,'rpm':rpms,'xyxy':xyxys,'y_ext':y_exts,
                                   'Dilutions':dilutions}, index=indices)
                np.save('/home/xiao/projects/DED/BO_processing/Final_data/single_tracks/cropped/mask_batch_{0}'.format(batch_num),masks)
                df.to_csv('/home/xiao/projects/DED/BO_processing/Final_data/single_tracks/cropped/mp_data_batch_{0}.csv'.format(batch_num))

df = pd.DataFrame({'Power': powers, 'Speed': speeds,'Width_mp': widths,
                   'Height':heights, 'Depth': depths, 
                   'Angle':angles,'rpm':rpms,'xyxy':xyxys,'y_ext':y_exts,
                   'Dilutions':dilutions}, index=indices)
np.save('/home/xiao/projects/DED/BO_processing/Final_data/single_tracks/cropped/mask_batch_{0}'.format(batch_num),masks)
df.to_csv('/home/xiao/projects/DED/BO_processing/Final_data/single_tracks/cropped/mp_data_batch_{0}.csv'.format(batch_num))

mask_folder = '/home/xiao/projects/DED/BO_processing/Final_data/single_tracks/cropped/masked/'
if not os.path.isdir(mask_folder):
    os.mkdir(mask_folder)
for file in os.listdir(img_dir):
    if file.endswith('.png'):
        shutil.move(img_dir+file,mask_folder+file)