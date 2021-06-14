# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 13:55:04 2021

@author: lowes
"""

import numpy as np
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from models.layers.mesh_prepare import (remove_non_manifolds,augmentation,
build_gemm,post_augmentation,extract_features)
from matplotlib.cbook import flatten
import os

def Remvoe_zero_area(mesh, faces, face_labels):
    face_normals = np.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                            mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    if np.any(face_areas[:, np.newaxis] == 0):
        temp = (face_areas == 0).nonzero()
        faces = np.delete(faces,temp,axis=0)
        face_labels = np.delete(face_labels,temp,axis=0)
        
    return faces, face_labels

filename = 'datasets/LAA_segmentation/test/0212.vtk'
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

numpy_array_of_points = dsa.WrapDataObject(data).Points

poly = dsa.WrapDataObject(cleanPolyData.GetOutput()).Polygons
poly_mat = np.reshape(poly,(-1,4))[:,1:4]

num_aug = 1

class MeshPrep:
        def __getitem__(self, item):
            return eval('self.' + item)

mesh_data = MeshPrep()
mesh_data.vs = mesh_data.edges = None
mesh_data.gemm_edges = mesh_data.sides = None
mesh_data.edges_count = None
mesh_data.ve = None
mesh_data.v_mask = None
#mesh_data.filename = f_name
mesh_data.edge_lengths = None
mesh_data.edge_areas = []
mesh_data.vs = points
faces = poly_mat

'''
mesh=mesh_data
face_normals = np.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                    mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
print((face_areas==0).nonzero())
print('Difference in lengt ', len(numpy_array_of_points)-len(points))

writer = vtk.vtkPolyDataWriter()
writer.SetInputData(cleanPolyData.GetOutput())
writer.SetFileName('datasets/0236_less_points.vtk')
writer.Write()

mesh_w = tvtk.PolyData(points=points, polys=poly_mat)
write_data(mesh_w, 'datasets/0236_less_tvtk.vtk')

#%%
'''
faces, face_labels = Remvoe_zero_area(mesh_data, faces, face_labels)
mesh_data.v_mask = np.ones(len(mesh_data.vs), dtype=bool)
faces, face_areas = remove_non_manifolds(mesh_data, faces)
if num_aug > 1:
    faces = augmentation(mesh_data, opt, faces)
build_gemm(mesh_data, faces, face_areas)
if num_aug > 1:
    post_augmentation(mesh_data, opt)
mesh_data.features = extract_features(mesh_data)

s_faces = np.sort(faces)
f01 = s_faces[:,[0,1]]
f02 = s_faces[:,[0,2]]
f12 = s_faces[:,[1,2]]

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
edges = mesh_data.edges
face_labs = np.zeros(len(faces))
for i,e in enumerate(edges):
    idx = []
    #e = np.array([indices[e[0]], indices[e[1]]])
    
    idx.append( (e==f01).all(axis=1).nonzero() )
    idx.append( (e==f02).all(axis=1).nonzero() )
    idx.append( (e==f12).all(axis=1).nonzero() )
    for j in list(flatten(idx)):
        face_labs[j] += edge_labels[i]

face_labs[face_labs<1.5]=0
face_labs[face_labs>0]=1
#%%
vcol=''
file = 'try_0212.vtk'
with open(file, 'w+') as f:
    f.write("# vtk DataFile Version 4.2 \nvtk output \nASCII \n \nDATASET POLYDATA \nPOINTS %d float \n" % len(mesh_data.vs))
    for vi, v in enumerate(mesh_data.vs):
        #vcol = ' %f %f %f' % (vcolor[vi, 0], vcolor[vi, 1], vcolor[vi, 2]) if vcolor is not None else ''
        f.write("%f %f %f%s\n" % (v[0], v[1], v[2], vcol))
        
    f.write("POLYGONS %d %d \n" % (len(faces),4*len(faces)))
    for face_id in range(len(faces) - 1):
        f.write("3 %d %d %d\n" % (faces[face_id][0] , faces[face_id][1], faces[face_id][2]))
    f.write("3 %d %d %d" % (faces[-1][0], faces[-1][1], faces[-1][2]))
    #for edge in self.edges:
     #   f.write("\ne %d %d" % (new_indices[edge[0]] + 1, new_indices[edge[1]] + 1))
     
    f.write("\n \nCELL_DATA %d \nSCALARS scalars double \nLOOKUP_TABLE default" % len(faces))
    for i,face in enumerate(faces):
                    '''
                    if segments[[face]].sum() < 1.5:
                        scalars[i] = 0
                    elif segments[[face]].sum() > 1.5:
                        scalars[i] = 1
                        '''
                    if i%9 == 0:    
                        f.write("\n") 
                    f.write("%d " % face_labs[i])