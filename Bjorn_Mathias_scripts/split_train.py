# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:05:29 2021

@author: lowes
"""
import numpy as np
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
import os
#%%
for i, f_name in enumerate(os.listdir('datasets/LAA_segmentation/')):
    if f_name.endswith('.vtk'):
        print(f_name)
        filename = 'datasets/LAA_segmentation/' + f_name
        try:
            loaded = np.load('datasets/LAA_segmentation/npz_data/train/'+f_name.split('.')[0]+'.npz')
            train = 0
        except:
            print('not train')
        try:
            loaded = np.load('datasets/LAA_segmentation/npz_data/test/'+f_name.split('.')[0]+'.npz')
            train = 1
        except:
            print('not test')
        try:
            loaded = np.load('datasets/LAA_segmentation/npz_data/final_test/'+f_name.split('.')[0]+'.npz')
            train = 2
        except:
            print('not_final_test')
            
        if train==0:
            out_file = 'datasets/LAA_segmentation/train/' + f_name
        elif train==1:
            out_file = 'datasets/LAA_segmentation/test/' + f_name
        elif train==2:
            out_file = 'datasets/LAA_segmentation/final_test/' + f_name
            
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
        vs = np.array( cleanPolyData.GetOutput().GetPoints().GetData() )
        face_labels = np.array( cleanPolyData.GetOutput().GetCellData().GetScalars() )         
        poly = dsa.WrapDataObject(cleanPolyData.GetOutput()).Polygons
        faces = np.reshape(poly,(-1,4))[:,1:4]
        
        print('writing file')
        #vcolor = []
        with open(out_file, 'w+') as f:
            f.write("# vtk DataFile Version 4.2 \nvtk output \nASCII \n \nDATASET POLYDATA \nPOINTS %d float \n" % len(vs))
            for vi, v in enumerate(vs):
                #vcol = ' %f %f %f' % (vcolor[vi, 0], vcolor[vi, 1], vcolor[vi, 2]) if vcolor is not None else ''
                f.write("%f %f %f\n" % (v[0], v[1], v[2]))
            f.write("POLYGONS %d %d \n" % (len(faces),4*len(faces)))
            for face_id in range(len(faces) - 1):
                f.write("3 %d %d %d\n" % (faces[face_id][0] , faces[face_id][1], faces[face_id][2]))
            f.write("3 %d %d %d" % (faces[-1][0], faces[-1][1], faces[-1][2]))
            f.write("\n \nCELL_DATA %d \nSCALARS scalars double \nLOOKUP_TABLE default" % len(faces))
            for j,face in enumerate(faces):
                if j%9 == 0:    
                    f.write("\n") 
                f.write("%d " % face_labels[j])

#%%

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

def split_ting(points_count,bin_width = 0.1):
    data = points_count.copy()
    train_idx = np.zeros_like(data)
    quantiles = np.quantile(data,np.arange(bin_width,1,bin_width))
    quantiles = np.append(np.append(0,quantiles),np.Inf)
    for i in range(len(quantiles)-1):
        idxs = np.atleast_1d( (data>quantiles[i]) & (data<quantiles[i+1]) ).nonzero()
        rand_idxs = np.random.permutation(idxs[0].tolist())
        
 
        train_idx[rand_idxs[0]] = 1
        train_idx[rand_idxs[1:3]] = 2
            
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
        print(f_name)
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
        
        if train_idx[i]==0:
            np.savez('datasets/LAA_segmentation/train/'+f_name.split('.')[0],
                 points=points, poly_mat=poly_mat)
        elif train_idx[i]==1:
            np.savez('datasets/LAA_segmentation/test/'+f_name.split('.')[0],
                 points=points, poly_mat=poly_mat)   
        elif train_idx[i]==2:
             np.savez('datasets/LAA_segmentation/final_test/'+f_name.split('.')[0],
                 points=points, poly_mat=poly_mat)   
            
#%% Split fvtk files

            
            
            
            
#%% et point count after split
points_count = []
train_idx = []
folder1 = 'datasets/LAA_segmentation/train/'
folder2 = 'datasets/LAA_segmentation/test/'
folder3 = 'datasets/LAA_segmentation/final_test/'
folders = [folder1, folder2, folder3]
for j,folder in enumerate(folders):
    for i, f_name in enumerate(os.listdir(folder)):
        if f_name.endswith('.vtk'):
            print(f_name)
            filename = folder + f_name
                    
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(filename)
            reader.ReadAllScalarsOn()
            reader.Update()
            
            points = np.array( reader.GetOutput().GetPoints().GetData() )
            points_count.append(len(points))
            train_idx.append(j)
points_count = np.asarray(points_count)
train_idx = np.asarray(train_idx)
#%%

import matplotlib.pyplot as plt
plt.figure(figsize=(20,4))
plt.subplot(1,3,1)
bin_list = np.arange(min(points_count), max(points_count) + 2000, 2000)
x = points_count[(0==train_idx).nonzero()]
print(x.mean())
n, bins, patches = plt.hist(x=x, bins=bin_list, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('No of vertices in meshes', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distributions of no vertices in train set', fontsize=18)
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

plt.subplot(1,3,2)
x = points_count[(1==train_idx).nonzero()]
print(x.mean())
n, bins, patches = plt.hist(x=x, bins=bin_list, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('No of vertices in meshes', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distributions of no vertices in validation set', fontsize=18)
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

plt.subplot(1,3,3)
x = points_count[(2==train_idx).nonzero()]
print(x.mean())
n, bins, patches = plt.hist(x=x, bins=bin_list, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('No of vertices in meshes', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distributions of no vertices in test set', fontsize=18)
maxfreq = n.max()   
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)