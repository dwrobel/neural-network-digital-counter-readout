#!/usr/bin/python3
# coding: utf-8

# # Image preparation
# 
# The original image size is 55x90 pixels with a color depth of 3 (RGB).
# The below code can be used to transform the images in an input directory (Input_dir) to the right size (20x32 pixels) into an output directory (Output_dir). Inside the directory the pictures are stored in subdirectories according their labeling (0 ... 9 + NaN).
# Any other image converter can be used as well.
# 
# ### Prerequisite
# Installed OpenCV libary within python (opencv)

# In[7]:


import glob
import os
from PIL import Image 

Input_dir = '../collectmeterdigits/data/labeled'
Output_dir= 'data/resize'

target_size_x = 20
target_size_y = 32

if not os.path.exists(Output_dir):
    os.mkdir(Output_dir)

files = glob.glob(Output_dir + '/*.jpg')
i = 0
for f in files:
    os.remove(f)
    i=i+1
print(str(i) + " files have been deleted.")

files = glob.glob(Input_dir + '/*.jpg')
files = files + glob.glob(Input_dir + '/*.png')
files = files + glob.glob(Input_dir + '/*.bmp')
count = 0
for aktfile in files:
    count = count + 1
    if not count % 250:
        print(str(count) + " ...")
    test_image = Image.open(aktfile)
    test_image = test_image.convert('RGB')
    test_image = test_image.resize((target_size_x, target_size_y), Image.Resampling.NEAREST)
    base=os.path.basename(aktfile)
    base = os.path.splitext(base)[0] + ".jpg"
    save_name = Output_dir + '/' + base
#    print("in: " + aktfile + "  -  out: " + save_name)
    test_image.save(save_name, "JPEG")
print(count)
