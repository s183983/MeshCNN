# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:05:29 2021

@author: lowes
"""
import numpy as np
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
import os

points_count = []

for f_name in os.listdir('datasets/LAA_segmentation/'):
    if f_name.endswith('.vtk'):
        filename = 'datasets/LAA_segmentation/' + f_name
                
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(filename)
        reader.ReadAllScalarsOn()
        reader.Update()
        data = reader.GetOutput()
        filt = vtk.vtkConnectivityFilter()

        filt.SetInputData(data) # get the data from the MC alg.
        filt.SetExtractionModeToLargestRegion()
        #filt.ColorRegionsOn()
        filt.Update()
        data_filt = filt.GetOutput()

        cleanPolyData = vtk.vtkCleanPolyData()
        cleanPolyData.ToleranceIsAbsoluteOn()
        cleanPolyData.SetAbsoluteTolerance(1.e-3)
        cleanPolyData.SetInputData(data_filt)
        cleanPolyData.Update()
        
        points = np.array( cleanPolyData.GetOutput().GetPoints().GetData() )
        points_count.append(len(points))
points_count = np.asarray(points_count)
#%%

def split_ting(points_count,train_size = 0.8,bin_width = 0.1):
    data = points_count.copy()
    train_idx = np.zeros_like(data)
    quantiles = np.quantile(data,np.arange(bin_width,1,bin_width))
    quantiles = np.append(np.append(0,quantiles),np.Inf)
    for i in range(len(quantiles)-1):
        idxs = np.atleast_1d( (data>quantiles[i]) & (data<quantiles[i+1]) ).nonzero()
        rand_idxs = np.random.permutation(idxs[0].tolist())
        
        if np.random.binomial(1,train_size):
            tr_idxs = rand_idxs[0:np.int(np.ceil(len(idxs[0])*train_size))]
        else:
            tr_idxs = rand_idxs[0:np.int(np.floor(len(idxs[0])*train_size))]
            
        
        for idx in tr_idxs:
            train_idx[idx] = 1
        for idx in rand_idxs:
            data[idx] = -1
    
    return train_idx
import random
from datetime import datetime
random.seed(datetime.now())
train_idx = split_ting(points_count)
print(train_idx.mean())
#%%
for i, f_name in enumerate(os.listdir('datasets/LAA_segmentation/')):
    if f_name.endswith('.vtk'):
        filename = 'datasets/LAA_segmentation/' + f_name
                
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(filename)
        reader.ReadAllScalarsOn()
        reader.Update()
        data = reader.GetOutput()
        filt = vtk.vtkConnectivityFilter()

        filt.SetInputData(data) # get the data from the MC alg.
        filt.SetExtractionModeToLargestRegion()
        #filt.ColorRegionsOn()
        filt.Update()
        data_filt = filt.GetOutput()

        cleanPolyData = vtk.vtkCleanPolyData()
        cleanPolyData.ToleranceIsAbsoluteOn()
        cleanPolyData.SetAbsoluteTolerance(1.e-3)
        cleanPolyData.SetInputData(data_filt)
        cleanPolyData.Update()
        
        points = np.array( cleanPolyData.GetOutput().GetPoints().GetData() )
        
        face_labels = np.array( cleanPolyData.GetOutput().GetCellData().GetScalars() )
        
        #numpy_array_of_points = dsa.WrapDataObject(data).Points
        
        poly = dsa.WrapDataObject(cleanPolyData.GetOutput()).Polygons
        poly_mat = np.reshape(poly,(-1,4))[:,1:4]
        
        if train_idx[i]:
            np.savez('datasets/LAA_segmentation/train/'+f_name.split('.')[0],
                 points=points, poly_mat=poly_mat)
        else:
             np.savez('datasets/LAA_segmentation/test/'+f_name.split('.')[0],
                 points=points, poly_mat=poly_mat)   
             
#%%
import matplotlib.pyplot as plt

x = points_count[train_idx.nonzero()]
print(x.mean())
n, bins, patches = plt.hist(x=x, bins=10, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('No of points in meshes')
plt.ylabel('Frequency')
plt.title('Distributions of no points in trainset')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
#%%
x = points_count[(1-train_idx).nonzero()]
print(x.mean())
n, bins, patches = plt.hist(x=x, bins=10, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('No of points in meshes')
plt.ylabel('Frequency')
plt.title('Distributions of no points in testset')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)