#!/usr/bin/env python
# coding: utf-8

# In[66]:


import cv2
#import sys


# In[67]:


# Get user supplied values
imagePath = "C:/Users/Jefe Operaciones/tarea_opencv/women.jpg"
cascPath = "C:/Users/Jefe Operaciones/tarea_opencv/haarcascades/haarcascade_frontalface_default.xml"


# In[68]:


#This lines is for .py archive
#imagePath = sys.argv[1]
#cascPath = "haarcascade_frontalface_default.xml"


# In[69]:


# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)


# In[70]:


# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[71]:


# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=2,
    minSize=(10, 10),
    flags = cv2.CASCADE_SCALE_IMAGE
)


# In[72]:


faces


# In[73]:


#print "Found {0} faces!".format(len(faces))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


# In[74]:


#image


# In[75]:


cv2.imshow("Faces found", image)
cv2.waitKey(0)


# In[58]:


cv2.imwrite('Faces Found3.png',image)


# In[76]:


faceCascadeVideo = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascadeVideo.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    #Salir con la tecla q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


# In[ ]:




