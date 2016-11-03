#!/usr/bin/env python
#Create new directories
from random import random
from PIL import Image

from PIL import Image, ImageDraw
from random import random
from shutil import copyfile
import math
import os

filter_w=200 #tile width
filter_h=200 #tile height
src_dir='NONE' #Directory that contains orinigal images WITHOUT transients
dest_dir='new' #Temporary directory
out_dir='post'
sample_size=1000 #How many samples to generate?
bias=0.5 #Transient/no_tranient ratio


def dist(A): #Computes distance between two points in px
    return(((A[0][0]-A[0][2])**2+(A[0][1]-A[0][3])**2)**0.5)


count=0
def create_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)


create_dir(dest_dir)
create_dir(out_dir)

for each in os.listdir(src_dir):
    copyfile(src_dir+'/'+each,dest_dir+'/'+str(count)+'.jpg')
    count+=1
    

image_len=len(os.listdir(src_dir))

for i in range(sample_size):
    myrand=0
    while myrand==0:
        myrand=int(random()*image_len) #Open files with random background and w/o meteorites
    
    im=Image.open(dest_dir+'/'+str(myrand)+'.jpg')
    X=im.size[0]
    Y=im.size[1]
    if ((X>1840) or (Y>1228)):
        im=im.thumbnail(1840,1228)
    width,height=im.size
    marker_x=width*2
    marker_y=height*2
    while ((marker_x+filter_w)>width):
        marker_x=int(random()*width)
        while ((marker_y+filter_h)>height):
            marker_y=int(random()*height)
    im=im.crop((marker_x,marker_y,marker_x+filter_w,marker_y+filter_h))
    draw = ImageDraw.Draw(im)
    A=[(random()*im.size[0], random()*im.size[1], random()*im.size[0], random()*im.size[1])]
    myrand=str(myrand)
    if (random()>bias):
        if ((dist(A)>30) and (dist(A)<300)): #The lines shouldn't be too short or too long
            color_rnd=0
            while (color_rnd<0.3): 
                color_rnd=random() #Brightness should be over certain threshold, lower the brightness -> harder to train, more resilient
            width_rand=0
            while (width_rand<0.5):
                width_rand=int(random()*2) #Width should be less than 2 px
            draw.line([A[0][0],A[0][1],A[0][2],A[0][3]], fill=int(color_rnd*100),width=width_rand)
            myrand=myrand+'_y' #Add _y to files that contain  meteors
            del draw

    im.save(out_dir+'/'+str(i)+'_'+str(myrand)+'.jpg')
    
