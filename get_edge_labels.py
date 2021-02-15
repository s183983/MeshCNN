# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 10:27:02 2021

@author: lowes
"""

from tvtk.api import tvtk, write_data
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


for f_name in os.listdir('datasets/LAA_segmentation/'):
    if f_name.endswith('.vtk'): #and int(f_name.split('.')[0])==236:
        filename = 'datasets/LAA_segmentation/' + f_name
        
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(filename)
        reader.ReadAllScalarsOn()
        reader.Update()
        data = reader.GetOutput()
        
        cleanPolyData = vtk.vtkCleanPolyData()
        cleanPolyData.ToleranceIsAbsoluteOn()
        cleanPolyData.SetAbsoluteTolerance(1.e-3)
        cleanPolyData.SetInputData(data)
        cleanPolyData.Update()
        
        points = np.array( cleanPolyData.GetOutput().GetPoints().GetData() )
        
        face_labels = np.array( cleanPolyData.GetOutput().GetCellData().GetScalars() )
        
        numpy_array_of_points = dsa.WrapDataObject(data).Points
        poly = dsa.WrapDataObject(cleanPolyData.GetOutput()).Polygons
        poly_mat = np.reshape(poly,(-1,4))[:,1:4]
        
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
        mesh_data.filename = f_name
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
        
        np.savez('datasets/LAA_segmentation/labels/'+f_name.split('.')[0],
                 labels=edge_labels,soft_labels=soft_labels)
        
'''
Read files again as 
loaded = np.load('datasets/LAA_segmentation/labels/'+f_name.split('.')[0]+'.npz')
labels = loaded['labels']
soft_labels = loaded['soft_labels']
'''