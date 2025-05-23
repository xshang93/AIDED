#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 19:01:47 2024

@author: xiao

Contains helper functions and result visualization functions
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import joblib
from PIL import Image
import math
import warnings
import pandas as pd
from keras.models import load_model


# Suppress all UserWarning warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

## ---------------------- Load the trained ML models ------------------------ #
model_path = '/home/xiao/git_projects/auto_AM_researcher/Xiao/MeltPool_MLWorkFlow/trained_models/'

para2geom = load_model(model_path+'para2geom.h5')
para2geom_pca = joblib.load(model_path+'pca_transformer.pkl')
sc = joblib.load(model_path+'sc.bin')
hs2angle = joblib.load(model_path+'hs2angle.pkl')

# calculate meltpool geometries from prediction. Can also visualize/save the meltpool. The input mp is the
# prediction directly from para2geom model.
def meltpool_geom_cal(power,speed,rpm,plot=False,save_path=''):
    scale_measured = 0.0038 # length of mm for 1 pix for a 1280x960 image. Measured on 20240607 using /home/xiao/projects/DED/BO_processing/images/20240418_singletrack_data_retake/scale_bar_67um_mp10&11.jpg
    resize_dim = (96, 96) # original size (550,550), cropped to (96,96)
    scale = scale_measured/resize_dim[1]*550
    
    input_sc = sc.transform([[power, rpm, speed]])
    mp = para2geom.predict(input_sc,verbose=0)
    mp = para2geom_pca.inverse_transform(mp)
    ret,mp_true = cv2.threshold(mp.reshape(resize_dim),127,255,cv2.THRESH_BINARY)
        
    mask_x = np.sum(mp_true,axis = 0)
    # Find the left-most x extreme value
    x_ext_l = (mask_x!=0).argmax(axis=0)
    # Find the right-most x extreme value
    x_ext_r = x_ext_l+np.count_nonzero(mask_x)-1
    # Select the point with the lowest y for the left-most x extreme
    y_ext_l = (np.where(mp_true[:, x_ext_l] != 0)[0]).max()
    # Select the point with the lowest y for the righht-most x extreme
    y_ext_r = (np.where(mp_true[:, x_ext_r] != 0)[0]).max()
    # select the lower of the two for width measurement
    y_ext = max(y_ext_l,y_ext_r)
    
    # Calculate the centres
    # y_centre = (y_ext_l+y_ext_r)/2
    y_centre  = y_ext
    x_centre = (x_ext_l+x_ext_r)/2
    
    mask_y = np.sum(mp_true,axis = 1)
    y_ext_high = (mask_y!=0).argmax(axis=0)
    y_ext_low = y_ext_high+np.count_nonzero(mask_y)-1
    
    # extract meltpool width and tilt angle from print bed
    # width = ((x_ext_l-x_ext_r)**2+(y_ext_l-y_ext_r)**2)**0.5*scale
    width = (x_ext_r-x_ext_l)
    height = abs(y_ext_high-y_centre)
    depth = abs(y_ext_low-y_centre)
    
    if plot:
        fig = plt.figure()
        plt.imshow(mp_true)
        lines = True
        if lines:
            # draw the width, height, and depth
            plt.plot([x_ext_l,x_ext_r],[y_ext,y_ext])
            plt.plot([x_centre,x_centre],[y_centre,y_ext_high])
            plt.plot([x_centre,x_centre],[y_centre,y_ext_low])
        # plt.axis('off')
        plt.title('Predicted meltpool for P = {0}, v = {1}, rpm = {2}'.format(power,speed,rpm))
        if save_path!='':
            plt.savefig(save_path)
        plt.close()
        
    A_top = mp_true[0:int(y_ext)].sum()
    A = mp_true.sum()
    dilution = 1-A_top/A
    # print('The width is {0}'.format(width))
    return width,height,depth,mp_true,scale,dilution,A_top/255

# Rotate an image
def rotate_image(image, mask, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, 360-angle, 1.0)
    # rotated size + 20 to avoid rotated image being cropped
    rotated_image = cv2.warpAffine(image, rot_matrix, (image.shape[1]+50, image.shape[0]+50))
    rotated_mask = cv2.warpAffine(mask, rot_matrix, (mask.shape[1]+50, mask.shape[0]+50))
    
    # Used to calculate the furtherest point after rotation
    
    coords_mp = np.argwhere(image)
    _, x_idx_max_mp = np.argmax(coords_mp,axis=0)
    
    # Convert the point to homogeneous coordinates. Points selected are the right most point 
    # Note that x and y order needs to be changed due to the stupidest way cv2 handles x and y
    right_most_homogeneous = np.array([coords_mp[x_idx_max_mp][1], coords_mp[x_idx_max_mp][0], 1])
    
    # Apply the rotation matrix to the point
    right_most_new = np.dot(rot_matrix, right_most_homogeneous)
    
    coords_mp_rotated = np.argwhere(rotated_image)
    y_min, x_min = coords_mp_rotated.min(axis=0)
    y_max, x_max = coords_mp_rotated.max(axis=0)

    rotated_mp_height = right_most_new[1]-y_min
    rotated_mp_depth = y_max-right_most_new[1]
    
    return rotated_image, rotated_mask, rotated_mp_height, rotated_mp_depth


# This function is used to predict what a 2d print surface is like under given printing parameters
def pred_2d_surface(power,speed,rpm,hs,num_tracks,opt = False,save_path=''):
    
    width,height,_,mp_true,_,_,_ = meltpool_geom_cal(power,speed,rpm)
    # need to resize width and height from 96x96 image back to 550x550 image
    angle = hs2angle.predict([[width/96*550,speed,hs,height/96*550]])[0]
    
    coords = np.argwhere(mp_true)

    # Bounding box for the AOI
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Extract AOI
    aoi = mp_true[max(y_min-10,0):y_max+10, max(x_min-10,0):x_max+10]
    # Create the AOI mask
    aoi_mask = (aoi > 0).astype(np.uint8)
    
    rotated_aoi, rotated_aoi_mask, rotated_mp_height, rotated_mp_depth = rotate_image(aoi, aoi_mask, angle)
    
    spacing = (x_max-x_min)*hs
    
    # Create a new image to place the patterned AOI. Need to be large enough
    patterned_image = np.zeros((1000,1000))
    
    # Place the AOI repeatedly in the new image
    for i in range(num_tracks):
        x_offset = i * spacing
        y_offset =30  # Assuming we place it at the top of the image
        
        x_offset = int(x_offset)
        y_offset = int(y_offset)
        
        # Define the region of interest in the destination image
        roi = patterned_image[y_offset:y_offset+rotated_aoi.shape[0], x_offset:x_offset+rotated_aoi.shape[1]]
        roi_mask = rotated_aoi_mask
        
        # Update only the AOI regions using the mask
        np.copyto(roi, rotated_aoi, where=roi_mask.astype(bool))

    # Crop the image to the width of the patterned AOIs
    coords_patterned = np.argwhere(patterned_image)
    y_min_patterned, x_min_patterned = coords_patterned.min(axis=0)
    y_max_patterned, x_max_patterned = coords_patterned.max(axis=0)
    surface_padding = 0 # extrat pattern to prevent unnecessary crop
    patterned_image = patterned_image[y_min_patterned-surface_padding:y_max_patterned+surface_padding,x_min_patterned:x_max_patterned+surface_padding]

    ds_cal_image = patterned_image.copy()
    ds_cal_image[ds_cal_image>0]=1
    ds = 1-ds_cal_image[0:int(rotated_mp_height)].sum()/ds_cal_image.sum()
    

    if save_path!='':
        # Save or display the resulting image
        fig = plt.figure()
        plt.imshow(patterned_image)
        plt.plot([0,100],[int(rotated_mp_height)+surface_padding,int(rotated_mp_height)+surface_padding])
        plt.savefig(save_path)
        # plt.axis('off')
        plt.title('Predicted meltpool for P = {0}, v = {1}, rpm = {2}'.format(power,speed,rpm))
        if opt:
            plt.close()

    return patterned_image, int(rotated_mp_height), int(rotated_mp_depth),surface_padding,ds
# This function is used to predict what a 3d print cube is like under given printing parameters
# Make sure bottom of new deposited materails are overlapping with the lowest point on the previous layer

def pred_3d_cube(power,speed,rpm,hs,num_tracks,num_layers,opt=True, verbose=False,save_path=''):
    surface, layer_height, layer_depth, surface_padding,d_ = pred_2d_surface(power,speed,rpm,hs,num_tracks,save_path='')
    # calculate dilution for just 1 track
    _,_,_,_,ds = pred_2d_surface(power,speed,rpm,hs,1,save_path='')
    coords_surface = np.argwhere(surface)
    _, x_idx_max_surface = np.argmax(coords_surface,axis=0)
    
    # Create the AOI mask
    surface_mask = (surface > 0).astype(np.uint8)
    # ratio of overlap in height
    t_ratio = 1
    counter=0
    # Place the surface repeatedly in the new image until there is no zero values in the centre 50%
    while True:
        # Establish an array of all zero
        cube = np.zeros((1000,surface.shape[1]))
        cube_to_check = np.zeros((1000,surface.shape[1]))
        # Calculate the current layer thickness
        layer_t = int(layer_height*t_ratio)
        
        # Pattern the layers
        for i in range(num_layers):
            x_offset = 0
            total_height = num_layers * surface.shape[0]
            
            # Define the region of interest in the destination image
            roi = cube[total_height-surface.shape[0]-layer_t*i:total_height-layer_t*i, x_offset:x_offset+surface.shape[1]]
            roi_to_check = cube_to_check[total_height-surface.shape[0]-layer_t*i:total_height-surface.shape[0]-layer_t*i+surface_mask[0:layer_height+surface_padding,:].shape[0], x_offset:x_offset+surface.shape[1]]
            
            # Update only the AOI regions using the mask
            np.copyto(roi, surface, where=surface_mask.astype(bool))
            np.copyto(roi_to_check, surface[0:layer_height+surface_padding,:], where=surface_mask[0:layer_height+surface_padding,:].astype(bool))
            
            surface = np.fliplr(surface)
            surface_mask = np.fliplr(surface_mask)
        # Crop the image to the width of the patterned AOIs
        coords_cube = np.argwhere(cube)
        y_min_cube, x_min_cube = coords_cube.min(axis=0)
        y_max_cube, x_max_cube = coords_cube.max(axis=0)
        cube = cube[(y_min_cube-1):(y_max_cube+1),x_min_cube:(x_max_cube+1)]
        cube_to_check = cube_to_check[(y_min_cube-1):(y_max_cube+1),x_min_cube:(x_max_cube+1)]
        
        cube_to_check_centre = cube_to_check[int(cube.shape[0]*0.25):int(cube.shape[0]*0.75),int(cube.shape[1]*0.25):int(cube.shape[1]*0.75)]
        
        # if all cube images should be plotted
        if verbose:
            plt.figure()
            plt.imshow(cube_to_check)
        if counter==0:
            porosity_init = np.count_nonzero(cube_to_check_centre==0)
            
        porosity = np.count_nonzero(cube_to_check_centre==0)
        counter = counter+1
        # Check pores
        if porosity<=porosity_init*0.1:
            break
        else:
            t_ratio = t_ratio-0.01
    

    if save_path!='':
        fig = plt.figure()
        plt.imshow(cube)
        plt.text(2, 0, 'dilution is {0}'.format(ds), ha='left', fontsize=12, color='blue')
        plt.savefig(save_path)
        if opt:
            plt.close()

    return t_ratio,layer_t,layer_height

def mp_data_check():
    counter=0
    for i in range(1,6):
        mask_mp = np.load('/home/xiao/projects/DED/BO_processing/Final_data/single_tracks/cropped/mask_batch_{0}.npy'.format(i))
        for mask in mask_mp:
            plt.imshow(mask)
            plt.savefig('/home/xiao/projects/DED/BO_processing/Final_data/single_tracks/cropped/processed_masks/{0}.png'.format(counter))
            counter=counter+1
            
# To output the num_layers and num_tracks. Not used in the optimization loop
def print_time_cal(solution):
    # set target cube dimensions in mm
    cube_h = 5
    cube_w = 10
    cube_l =10
    
    width,height,depth,_,scale,_,_ = meltpool_geom_cal(solution[0],solution[1],solution[2],plot=False,save_path='')
    angle = hs2angle.predict([[width,solution[1],solution[3],height]])
    
    # calculate the number of tracks in each layer
    num_tracks = round((cube_w-width*scale*np.cos(math.radians(angle)))/(width*scale*solution[3])+1)
    # calculate the number of layers
    _,layer_t,_ = pred_3d_cube(solution[0],solution[1],solution[2],solution[3],num_tracks,3,opt=True,verbose=False,save_path='')
    num_layers = round(cube_h/(layer_t*scale))
    
    total_t = cube_l/(solution[1]/60)*num_tracks*num_layers
    
    return total_t, num_layers, num_tracks

# the code below is for generating process parameters used in this paper
# https://doi.org/10.1016/j.addma.2023.103489
# speed is in mm/min
def compare_param_gen(p,v,m):
    save_dir = '/home/xiao/Dropbox/UofT/Project_docs/DED/Process_opt/figures/supplementary_figures/'
    width,height,depth,_,scale,dilution,A_clad = meltpool_geom_cal(p,v,m,plot=False,save_path='')
    hs = (1-height/(height+depth))**0.5
    t = A_clad/(hs*width)
    layer_t_to_w = t/width
    num_layers = 5/(t*scale)
    num_tracks = (10-width*scale)/(hs*scale*width)
    
    param_dict = {'Width_mp':[width*scale],'Power':[p],'Hatch':[hs],'Speed':[v/60],'Thickness':[layer_t_to_w],
                  'rpm':[m], 'num_layers':[num_layers],'num_tracks':[num_tracks]}
    param_df = pd.DataFrame(param_dict)
    param_df.to_csv(save_dir+'all_params.csv')
    
    return p,v,m,scale

    
    
            
if __name__=='__main__':
#     save_dir = '/home/xiao/projects/DED/BO_processing/Final_data/surfaces/GA/NSGA/20240619_rand_3/'
#     results_x = np.load(save_dir+'results_x.npy')
#     results_y = np.load(save_dir+'results_y.npy')
    
#     powers = []
#     speeds = []
#     rpms = []
#     widths = []
#     t_to_ws = []
#     num_layerss = []
#     num_trackss = []
#     hss = []
#     resolutions = []
#     times = []
    
#     for idx,solution in enumerate(results_x):
#     # power = 34.9
#     # speed = 696
#     # rpm = 0.7995
#     # hs = 0.4
#     # solution = [power,speed,rpm,hs]
#         power = solution[0]
#         speed = solution[1]
#         rpm = solution[2]
#         hs = solution[3]
#         total_t, num_layers, num_tracks = print_time_cal(solution)
#         verbose = True
#         _,_,_,_,ds=pred_2d_surface(power,speed,rpm,hs,num_tracks,save_path='')
#         width,height,depth,mp_true,scale,_,_ = meltpool_geom_cal(power,speed,rpm,plot=True,save_path='')
#         t_ratio,layer_t,layer_height = pred_3d_cube(power,speed,rpm,hs,num_tracks,num_layers,verbose,save_path=save_dir+'/{0}_cube.png'.format(idx))
        
#         layer_t_to_w = layer_t/(width)
        
#         powers.append(power)
#         speeds.append(speed/60)
#         rpms.append(rpm)
#         widths.append(width*scale)
#         t_to_ws.append(layer_t_to_w)
#         num_layerss.append(num_layers)
#         num_trackss.append(num_tracks)
#         hss.append(hs)
#         resolutions.append(results_y[idx][0])
#         times.append(results_y[idx][1])
    
#     param_dict = {'Width_mp':widths,'Power':powers,'Hatch':hss,'Speed':speeds,'Thickness':t_to_ws,
#                   'rpm':rpms, 'num_layers':num_layerss,'num_tracks':num_trackss,
#                   'resolutions':resolutions,'time':times}
#     param_df = pd.DataFrame(param_dict)
#     param_df.to_csv(save_dir+'all_params.csv')
    
    
    # pred_2d_surface(4.68236220e+01, 6.93612853e+02, 7.99986786e-01, 5.81945241e-01,5,save_path='aa')
    # t_ratio,layer_t,layer_height = pred_3d_cube(36.56684, 715, 0.8, 0.5,22,5,True,save_path='aa')
    # width,height,depth,mp_true,scale,dilution = meltpool_geom_cal(37,480,0.6,plot=True,save_path='')
    
    p,v,m,scale=compare_param_gen(48.11186601,11.61421833*60,0.654053081)