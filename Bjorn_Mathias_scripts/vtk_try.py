# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 12:31:44 2021

@author: lowes
"""

import numpy as np
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from models.layers.mesh_prepare import *
from matplotlib.cbook import flatten
filename = 'datasets/LAA_segmentation/train/0212.vtk'

reader = vtk.vtkPolyDataReader()
reader.SetFileName(filename)
reader.ReadAllScalarsOn()
reader.Update()
data = reader.GetOutput()


#%%
filt = vtk.vtkConnectivityFilter()

filt.SetInputData(data) # get the data from the MC alg.
filt.SetExtractionModeToLargestRegion()
#filt.ColorRegionsOn()
filt.Update()
data = filt.GetOutput()
points = np.array( data.GetPoints().GetData() )

print(data.GetCellData().GetScalars())

face_labels = np.array( data.GetCellData().GetScalars() )

numpy_array_of_points = dsa.WrapDataObject(data).Points
poly = dsa.WrapDataObject(filt.GetOutput()).Polygons
poly_mat = np.reshape(poly,(-1,4))[:,1:4]
#%% Visualize with vtk
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
WIDTH=640
HEIGHT=480
renWin.SetSize(WIDTH,HEIGHT)

# create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# mapper
dataMapper = vtk.vtkPolyDataMapper()
dataMapper.SetInputConnection(filt.GetOutputPort())

# actor
dataActor = vtk.vtkActor()
dataActor.SetMapper(dataMapper)

# assign actor to the renderer
ren.AddActor(dataActor)

# enable user interface interactor
iren.Initialize()
renWin.Render()
iren.Start()

#%%
num_aug = 1
ve = [[] for _ in points]

class MeshPrep:
        def __getitem__(self, item):
            return eval('self.' + item)

mesh_data = MeshPrep()
mesh_data.vs = mesh_data.edges = None
mesh_data.gemm_edges = mesh_data.sides = None
mesh_data.edges_count = None
mesh_data.ve = None
mesh_data.v_mask = None
mesh_data.filename = 'unknown'
mesh_data.edge_lengths = None
mesh_data.edge_areas = []
mesh_data.vs = points
faces = poly_mat


mesh_data.v_mask = np.ones(len(mesh_data.vs), dtype=bool)
faces, face_areas = remove_non_manifolds(mesh_data, faces)
if num_aug > 1:
    faces = augmentation(mesh_data, opt, faces)
build_gemm(mesh_data, faces, face_areas)
if num_aug > 1:
    post_augmentation(mesh_data, opt)
mesh_data.features = extract_features(mesh_data)
#%%
s_faces = np.sort(faces)
f01 = s_faces[:,[0,1]]
f02 = s_faces[:,[0,2]]
f12 = s_faces[:,[1,2]]
#%%
soft_labels = np.zeros(mesh_data.edges.shape)
for i,e in enumerate(mesh_data.edges):
    idx = []
    idx.append( (e==f01).all(axis=1).nonzero() )
    idx.append( (e==f02).all(axis=1).nonzero() )
    idx.append( (e==f12).all(axis=1).nonzero() )
    for j in list(flatten(idx)):
        soft_labels[i,int(face_labels[j])] = 1

edge_labels = soft_labels[:,1]
#%%
def pad(input_arr, target_length, val=0, dim=1):
    shp = input_arr.shape
    npad = [(0, 0) for _ in range(len(shp))]
    npad[dim] = (0, target_length - shp[dim])
    return np.pad(input_arr, pad_width=npad, mode='constant', constant_values=val)


#%%
seg = 'datasets/human_seg/seg/shrec__1.eseg'
sseg = 'datasets/human_seg/sseg/shrec__1.seseg'
offset=1
seg_labels = np.loadtxt(open(seg, 'r'), dtype='float64')-offset
ninput_edges = 2280
lab = pad(seg_labels,ninput_edges, val=-1, dim=0)

sseg_labels = np.loadtxt(open(sseg, 'r'), dtype='float64')
sseg_labels = np.array(sseg_labels > 0, dtype=np.int32)
soft_lab= pad(sseg_labels, ninput_edges, val=-1, dim=0)
print(len(points)-mesh_data.edges_count+len(faces)) ###3?????
