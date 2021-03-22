# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:02:26 2021

@author: lakri
"""
import math
import vtk
from random import random
import numpy as np
from vtk.numpy_interface import dataset_adapter as dsa


def volumetric_spline_augmentation():
    out_name = 'datasets/DataAug/0236aug.vtk'
    file_name = 'datasets/LAA_segmentation/0236.vtk'

    pd_in = vtk.vtkPolyDataReader()
    pd_in.SetFileName(file_name)
    pd_in.Update()

    pd = pd_in.GetOutput();

    # Get bounding box
    bounds = pd.GetBounds();
    x_min = bounds[0]
    x_max = bounds[1]
    y_min = bounds[2]
    y_max = bounds[3]
    z_min = bounds[4]
    z_max = bounds[5]

    # get a measure of scale as the diagonal length
    scale = math.sqrt((x_max - x_min) * (x_max - x_min) + (y_max - y_min) * (y_max - y_min) +
                      (z_max - z_min) * (z_max - z_min))


    # Reference points
    # Here just corners of the bounding box
    # TODO: make more points
    n = 6
    i = 0
    p1 = vtk.vtkPoints()
    p1.SetNumberOfPoints(n*n*n)
    
    xlin = np.linspace(x_min,x_max,n)
    ylin = np.linspace(y_min,y_max,n)
    zlin = np.linspace(z_min,z_max,n)
    
    for x in xlin:
        for y in ylin:
            for z in zlin:
                p1.SetPoint(i,x,y,z)
                i += 1 
    '''
    p1.SetPoint(0, x_min, y_min, z_min)
    p1.SetPoint(1, x_max, y_min, z_min)
    p1.SetPoint(2, x_min, y_max, z_min)
    p1.SetPoint(3, x_max, y_max, z_min)
    p1.SetPoint(4, x_min, y_min, z_max)
    p1.SetPoint(5, x_max, y_min, z_max)
    p1.SetPoint(6, x_min, y_max, z_max)
    p1.SetPoint(7, x_max, y_max, z_max)
    '''
    # Deformed points
    p2 = vtk.vtkPoints()
    # Start by copying all info from p1
    p2.DeepCopy(p1)

    # Displace the points in a random direction
    displacement_length = scale * 0.04 #change parameter around a bit
    for i in range(p2.GetNumberOfPoints()):
        p = list(p2.GetPoint(i))
        for j in range(3):
            p[j] = p[j] + (2.0 * random() -1 ) * displacement_length
        p2.SetPoint(i, p)

    transform = vtk.vtkThinPlateSplineTransform()
    transform.SetSourceLandmarks(p1)
    transform.SetTargetLandmarks(p2)
    transform.SetBasisToR()
    transform.Update()

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(pd)
    transform_filter.Update()
            
    vs = np.array( transform_filter.GetOutput().GetPoints().GetData() )
    face_labels = np.array( transform_filter.GetOutput().GetCellData().GetScalars() )         
    poly = dsa.WrapDataObject(transform_filter.GetOutput()).Polygons
    faces = np.reshape(poly,(-1,4))[:,1:4]
    
    print('writing file')
    #vcolor = []
    with open(out_name, 'w+') as f:
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
    print('done writing')
    
    
    
    '''
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(out_name)
    writer.SetInputConnection(transform_filter.GetOutputPort())
    writer.Write()
    print('done writing')
    '''

volumetric_spline_augmentation()


