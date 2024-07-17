
# MeltPool ML Training Workflow

This repository contains scripts for training a machine learning (ML) model to analyze melt pool images. The workflow involves preprocessing raw images, extracting features, managing invalid data, and finally, training the ML model.

## Prerequisites

- Python 3.x
- Anaconda 24.1.2
- Libraries: 
**shutil**
**ultralytics 8.2.2**
**segment_anything ver.2023**
**sklearn 1.4.2**
- Pretrained models:
**YOLO v8:** ./pre_trained_models/yolov8_best.pt
**Segment anything model:** Obtain from here https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

## Dataset and input files

#### 1. Single track melt pools
- Meltpool cross section OM images, labelled from 0 to n_images-1.jpg. e.g.,0.jpg, 1.jpg, 2.jpg etc. in one parent folder.
- A .csv file containing the printing parameters of the corresponding meltpools. The columns are as follows:

|       id      |     Power     |      Speed    |      rpm      |
| ------------- | ------------- | ------------- | ------------- |
|      0        |       25      |       4       |       2       |
|      1        |       26      |       6       |       2       |

Where Power is laser current, roughly a percentage of 1000W. Speed is mm/s. rpm is in fraction, i.e., 2 is 0.2 r/min.

#### 2. 2D melt surfaces
- A set of images of 2D surfaces. Need to make sure the 2D surface images has the last complete melt pool at the right-most of the images.
- All images should be in a folder named 'original_data'.
## Workflow Steps - 1D

### 1. Cropping Raw Images and Rename
Run `Cropping and rename.py` to preprocess raw image data by cropping to focus on relevant areas. Rename the images based on the csv.
- **Modify Line 33:** Specify the directory containing the raw data images.
- **Modify Line 65:** Specify the directory of the csv file containing the printing parameter data.

### 2. Feature Extraction
Execute `mp_feature_extract.py` to extract features from the cropped images.
- **Set current batch number** SAM is easy to casue out of memory. Need to separate the images into different batches to process. I used 90 images/batch. The folder names should be batch_{batch_num}. Repeat the code by changing different batch numbers on line 142 and restrat kernel.
- **System Path Adjustment:** Modify Line 11 to include your own system path.
- **Model Path:** Update Line 30 with the path to your SAM model.
- **YOLO Model Path:** Adjust Line 140 to specify the path to the pretrained YOLO model.
- **Image and Output Directories:** Set Line 144 for the image directory and Lines 188-189 & 195-196 for output directories.

### 3. Managing Invalid Masks
This code gathers the parameters of manually selected masks and write them into a .csv file. First Manually select invalid masks (original pictures and masked images that are not suitable for training) and move them to a separate directory such as 'masked/invalid/.
- Run `Extract_jpg_name2csv.py`.
    - **Invalid Masks Directory:** Change Line 49 to the directory containing invalid masks.
    - **CSV File Directory:** Modify Line 50 to set the directory for the generated CSV file.

### 4. Training the ML Model
With the data prepared, run `MELTPOOL_ML.py` to train the ML model.
- **Set current working directory:** Line 21
- Ensure all directories within the script are set accordingly to your file structure by changing the cwd path.
- Modify depending on how many batches were used for processing

## Workflow Steps - 2D

### 1. Segment image to extract single melt pools from 2D surface images
Excute 'segment2meltpool.py' to extract meltpools from 2D surface images.
The output is binary images of single melt pools.
- **Lines 35-39**: Paths to be changed accordingly.

### 2. Extract the angles from melt pools and save to a .csv file.
Excute 'getAngleOfMeltPool.py'.
- **Lines 35-39**: Paths to be changed accordingly.

## Usage

To use this workflow, follow the steps in order, ensuring all required modifications to the script paths and directories are made before execution. 

## Notes

- Ensure you have all the necessary libraries installed.
- Obtain the pretrained models as instructed and place them in accessible directories.
- Review and manage your data carefully during the preprocessing and feature extraction phases to ensure high-quality inputs for training.
