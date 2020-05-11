#!/usr/bin/env python
# coding: utf-8

# # The Project #
# 1. This is a project with minimal scaffolding. Expect to use the the discussion forums to gain insights! It’s not cheating to ask others for opinions or perspectives!
# 2. Be inquisitive, try out new things.
# 3. Use the previous modules for insights into how to complete the functions! You'll have to combine Pillow, OpenCV, and Pytesseract
# 4. There are hints provided in Coursera, feel free to explore the hints if needed. Each hint provide progressively more details on how to solve the issue. This project is intended to be comprehensive and difficult if you do it without the hints.
# 
# ### The Assignment ###
# Take a [ZIP file](https://en.wikipedia.org/wiki/Zip_(file_format)) of images and process them, using a [library built into python](https://docs.python.org/3/library/zipfile.html) that you need to learn how to use. A ZIP file takes several different files and compresses them, thus saving space, into one single file. The files in the ZIP file we provide are newspaper images (like you saw in week 3). Your task is to write python code which allows one to search through the images looking for the occurrences of keywords and faces. E.g. if you search for "pizza" it will return a contact sheet of all of the faces which were located on the newspaper page which mentions "pizza". This will test your ability to learn a new ([library](https://docs.python.org/3/library/zipfile.html)), your ability to use OpenCV to detect faces, your ability to use tesseract to do optical character recognition, and your ability to use PIL to composite images together into contact sheets.
# 
# Each page of the newspapers is saved as a single PNG image in a file called [images.zip](./readonly/images.zip). These newspapers are in english, and contain a variety of stories, advertisements and images. Note: This file is fairly large (~200 MB) and may take some time to work with, I would encourage you to use [small_img.zip](./readonly/small_img.zip) for testing.
# 
# Here's an example of the output expected. Using the [small_img.zip](./readonly/small_img.zip) file, if I search for the string "Christopher" I should see the following image:
# ![Christopher Search](./readonly/small_project.png)
# If I were to use the [images.zip](./readonly/images.zip) file and search for "Mark" I should see the following image (note that there are times when there are no faces on a page, but a word is found!):
# ![Mark Search](./readonly/large_project.png)
# 
# Note: That big file can take some time to process - for me it took nearly ten minutes! Use the small one for testing.

# ## Submission code

# In[10]:


import zipfile

from PIL import Image
import pytesseract
import cv2 as cv
import numpy as np

# loading the face detection classifier
face_cascade = cv.CascadeClassifier('readonly/haarcascade_frontalface_default.xml')

# the rest is up to you!


# ### Image loading

# In[11]:


import io 

def load_images(archive_path):
    
    """
    Takes a path to an image archive as input.
    Returns a dictionary containing each image file as key and a dictionary as value, 
    itself containing a pil and a cv version of the image contained in the image file.  
    """
    
    archive = zipfile.ZipFile(archive_path, 'r')

    images = {}
    for i in range(len(archive.infolist())):
    
        img = archive.read(archive.infolist()[i])
        img = io.BytesIO(img)
        img = Image.open(img) 
        img_l=img.convert("L")
        img_l.save('temp.png')
        cv_img=cv.imread('temp.png')
        get_ipython().system('rm temp.png')
        images[archive.namelist()[i]] = {'pil': img, 'cv': cv_img}
        
    return images    
    


# ### Text recognition

# In[12]:


import string

def extract_text(img):
    
    """
    Takes a PIL image as input. 
    Performs text extraction and returns a list of words in lower case.
    """
    
    text = pytesseract.image_to_string(img.convert('L')) 
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.split()
    return text


# ### Image recognition

# In[13]:


def extract_faces(img_cv, img_pil, min_size=20, scale_factor=1.25):
    
    """
    Takes a tuple containing a CV image and corresponding PIL image as input.
    Returns a list of cropped faces as recognized using openCV detectMultiScale method.
    """
    
    crops = []
    faces = face_cascade.detectMultiScale(img_cv, scaleFactor=scale_factor, minSize=(min_size, min_size), minNeighbors=5)
    for x,y,w,h in faces:
        crop = img_pil.crop((x, y, x+w, y+h))
        crops.append(crop)
    return crops    


# In[14]:


def display_images(img_list, img_size=100):
    
    """
    Takes a list of images as input (cropped face images) and returns a corresponding contact sheet.
    """
    
    crops = [crop.resize((img_size, img_size)) if crop.height > img_size or crop.width > img_size else crop for crop in img_list]
    first_image=crops[0]     
    n_images = len(crops)
    width = 5
    height = 1 + int(n_images / 5)
    contact_sheet=Image.new(mode=first_image.mode, size=(img_size*width,img_size*height), color=0)

    x=0
    y=0

    for crop in crops:
        contact_sheet.paste(crop, (x, y) )

 
        if x+img_size >= contact_sheet.width:
            x=0
            y=y+img_size
        else:
            x=x+img_size
        
    return contact_sheet        


# In[15]:


#display_images(extract_faces(images['a-3.png']['cv'], images['a-3.png']['pil']))


# ### Define search functions

# In[16]:


def get_output(kw, images):
    
    """
    Takes a name as input and searches for that name in text.
    Returns a dictionary with all files in which name is found as keys and list of faces as values.
    """
    
    output = {}
    
    for k, v in images.items():
        txt = extract_text(v['pil'])
        fces = extract_faces(v['cv'], v['pil'])   
        if kw in txt:
            output[k] = fces               
    return output
    
def run_search(person, image_collection):
    
    """
    Takes a name as input.
    Returns a visual output of search result for that person in image files'
    corresponding to the file names in which the person was found and a list of faces extracted from that file.
    """
    images = load_images(image_collection)
    out = get_output(person, images)
    for k, v in out.items():
        print('Results found in file {}'.format(k))
        if len(v) == 0:
            print('But there were no faces in that file!')
            print('\n')
        else:
            display(display_images(v))
            print('\n')


# ## Search results

# In[17]:


small = 'readonly/small_img.zip'
regular = 'readonly/images.zip'


# ### Christopher in small image set

# In[18]:


run_search('Christopher', small)


# ### Mark in full image set

# In[19]:


run_search('Mark', regular)

