#!/usr/bin/env python
# coding: utf-8


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


