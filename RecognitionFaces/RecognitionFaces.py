#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from glob import glob


# In[26]:


malepath = glob('./data/male/*.jpg')
femalepath = glob('./data/female/*.jpg')


# In[27]:


len(malepath), len(femalepath)


# In[37]:


# one image
path = malepath[7]
img = cv2.imread(path)


# In[38]:


plt.imshow(img)
plt.show()


# In[39]:


# convert image into grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray.shape


# In[40]:


plt.imshow(gray,cmap='gray')


# In[41]:


# load haar cascade classifier
haar = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')


# In[46]:


faces = haar.detectMultiScale(gray,1.5,5)
print(faces)


# In[47]:


for x,y,w,h in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
plt.imshow(img)


# In[13]:


# crop the image
crop_img = img[y:y+h,x:x+h]


# In[14]:


plt.imshow(crop_img)


# In[20]:


# save the image
cv2.imwrite('./data/m_01.png',crop_img)


# In[48]:


# Apply to all the images
def extract_images(path,gender,i):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray,1.5,5)
    for x,y,w,h in faces:
        roi = img[y:y+h,x:x+w]
        if gender == 'male':
            cv2.imwrite('./data/crop/male_crop/{}_{}.png'.format(gender,i),roi)
        else:
            cv2.imwrite('./data/crop/female_crop/{}_{}.png'.format(gender,i),roi)


# In[49]:


for i,path in enumerate(femalepath):
    try:

        extract_images(path,'female',i)
        print('INFO: {}/{} processed sucessfully'.format(i,len(femalepath)))
        
    except:
        print('INFO: {}/{} cannot be processed'.format(i,len(femalepath)))


# In[52]:


for i,path in enumerate(malepath):
    try:

        extract_images(path,'male',i)
        print('INFO: {}/{} processed sucessfully'.format(i,len(malepath)))
        
    except:
        print('INFO: {}/{} cannot be processed'.format(i,len(malepath)))


# In[ ]:




