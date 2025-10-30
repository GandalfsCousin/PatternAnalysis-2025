# Classifier for ADNI brain data based on the ConvNeXt

**Table of Contents**

## Overview

The problem at hand was the classification of MRI brainscan images in the ADNI dataset; this dataset contains sliced MRI brain scans, labeled Alzheimer's disease (AD) and normal control (NC). For this, a ConvNeXt classification model was implemented, this is a convolutional network adapted from ResNet50 and the Swin Transformer. This allows the model to improve over the ResNet50 in image recognitions tasks and led to it being implemented for tasks such as medical imaging classifaction. Using this, a model was implemented to acheive xx accuracy on the test set.


## Model Architecture

![alt text](images/Block.png)


## Dataset

The ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset is a public dataset, aimed for use in Alzheimer's  research. It is provided through the LONI Image and Data Archive, from the portal at https://adni.loni.usc.edu/data-samples/adni-data/ 

This data set provides grayscale, 256 x 240 pixel, T1w MRI images categorized into Alzheimer's Disease (AD) and Normal Control (NC) groups; with an example of each class shown below. These images have been split into Train and Test sets, withe each image following the filename structre `patient_index.png`, allowing the images to be split per patient to avoid data leakage between groups. The statistics of the dataset have been included in the table below.

Figure 2: Example AD Image emsp; emsp; Figure 3: Example NC Image \\

![alt text](images\AD_Example.jpeg) ![alt text](images\NC_Example.jpeg)
 
Table 1: ADNI Dataset Split
| Dataset Split  | AD Images | NC Images | Total Images |
|----------------|-----------|-----------|--------------|
| **Train**      | 10,400    | 11,120    | 21,520       |
| **Test**       | 4,460     | 4,540     | 9,000        |
| **Total**      | 14,860    | 15,660    | 30,520       |


## Training

The training configuration is as follows:  
| Argument | Value |
| ----- | ----- |


## Results

## Usage

## Refrences
