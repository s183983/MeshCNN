# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 10:48:22 2021

@author: lakri
"""
import numpy as np
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from models.layers.mesh_prepare import *

#%%
points_mat = []
label_mat = []
faces_mat = []
folder = 'C:\\Users\\lowes\\OneDrive\\Skrivebord\\DTU\\6_Semester\\Bachelor\\data_LAA\\data'
import os 
for f_name in os.listdir(folder):
    if f_name.endswith('.vtk'):
        filename = os.path.join(folder, f_name)
        
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(filename)
        reader.ReadAllScalarsOn()
        reader.Update()
        data = reader.GetOutput()

        points = np.array( reader.GetOutput().GetPoints().GetData() )
        points_mat.append(points)
        
        label = np.array( data.GetCellData().GetScalars() )
        label_mat.append(label)

        numpy_array_of_points = dsa.WrapDataObject(data).Points
        poly = dsa.WrapDataObject(data).Polygons
        poly_mat = np.reshape(poly,(-1,4))[:,1:4]
        faces_mat.append(poly_mat)

        
        print(filename)
#%%
import matplotlib.pyplot as plt

no_points = np.zeros([106,1])
no_faces = np.zeros([106,1])
label_dist = np.zeros([106,1])
for i in range(len(points_mat)):
    no_points[i] = (len(points_mat[i]))
    no_faces[i] = (len(faces_mat[i]))
    label_dist[i] = label_mat[i].sum()/len(label_mat[i])
#%%
plt.figure(figsize=(20,8))
plt.subplot(1,3,1)
# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(x=no_points, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('No of vertices in meshes',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.title('Distributions of no points', fontsize=18)
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

plt.subplot(1,3,2)
n, bins, patches = plt.hist(x=no_faces, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('No of faces in meshes',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.title('Distributions of no faces',fontsize=18)
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

plt.subplot(1,3,3)
n, bins, patches = plt.hist(x=label_dist*100, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('LAA labels as % of data',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.title('Distributions of LAA labels',fontsize=18)
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

#%%
ll=[]
for f_name in os.listdir('datasets/LAA_segmentation/labels/'):
    if f_name.endswith('.npz'):
        filename = 'datasets/LAA_segmentation/labels/' + f_name
        
        loaded = np.load(filename)
        labels = loaded['labels']
        ll.append(len(labels))







 
        
        
        