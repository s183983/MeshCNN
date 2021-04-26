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

def create_3d_grid(pd_in):
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    bounds = pd_in.GetBounds()
    x_len = bounds[1] - bounds[0]
    y_len = bounds[3] - bounds[2]
    z_len = bounds[5] - bounds[4]

    center = [(bounds[1] + bounds[0]) / 2, (bounds[2] + bounds[3]) / 2,  (bounds[4] + bounds[5]) / 2]

    # ncubes = 10 in each direction
    n_cubes = 10
    ntx = n_cubes + 1
    nty = n_cubes + 1
    ntz = n_cubes + 1
    nx2 = int(ntx/2)
    ny2 = int(nty/2)
    nz2 = int(ntz/2)

    # side length of a single cube
    sx = float(x_len) / float(n_cubes)
    sy = float(y_len) / float(n_cubes)
    sz = float(z_len) / float(n_cubes)

    points.SetNumberOfPoints(ntx * nty * ntz)

    for tx in range(-nx2, nx2 + 1):
        for ty in range(-ny2, ny2 + 1):
            for tz in range(-nz2, nz2 + 1):
                # id of center point
                idx = (tx + nx2) + (ty + ny2) * ntx + (tz + nz2) * ntx * nty

                x = tx * sx + center[0]
                y = ty * sy + center[1]
                z = tz * sz + center[2]

                points.SetPoint(idx, x, y, z)

                if tx != nx2:
                    id_2 = (tx + nx2 + 1) + (ty + ny2) * ntx + (tz + nz2) * ntx * nty
                    points.SetPoint(id_2, x + sx, y, z)
                    lines.InsertNextCell(2)
                    lines.InsertCellPoint(idx)
                    lines.InsertCellPoint(id_2)

                if ty != ny2:
                    id_2 = (tx + nx2) + (ty + 1 + ny2) * ntx + (tz + nz2) * ntx * nty
                    points.SetPoint(id_2, x, y + sy, z)
                    lines.InsertNextCell(2)
                    lines.InsertCellPoint(idx)
                    lines.InsertCellPoint(id_2)

                if tz != nz2:
                    id_2 = (tx + nx2) + (ty + ny2) * ntx + (tz + 1 + nz2) * ntx * nty
                    points.SetPoint(id_2, x, y, z+1)
                    lines.InsertNextCell(2)
                    lines.InsertCellPoint(idx)
                    lines.InsertCellPoint(id_2)

    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    pd.SetLines(lines)
    return pd

def volumetric_spline_augmentation():
    out_name = 'C:/Users/lakri/Documents/BachelorProjekt/Spares MeshCNN/MeshCNN_sparse/datasets/DataAug/0079aug.vtk'
    file_name = 'C:/Users/lakri/Documents/BachelorProjekt/Spares MeshCNN/MeshCNN_sparse/datasets/LAA_segmentation/0079.vtk'
    out_name_grid = 'C:/Users/lakri/Documents/BachelorProjekt/Spares MeshCNN/MeshCNN_sparse/datasets/DataAug/0079_grid.vtk'
    out_name_grid_warped = 'C:/Users/lakri/Documents/BachelorProjekt/Spares MeshCNN/MeshCNN_sparse/datasets/DataAug/0079_grid_warped.vtk'


    pd_in = vtk.vtkPolyDataReader()
    pd_in.SetFileName(file_name)
    pd_in.Update()

    pd = pd_in.GetOutput();
    
    grid_3d = create_3d_grid(pd)
    print('writing grid')
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(out_name_grid)
    writer.SetInputData(grid_3d)
    writer.Write()
    print('done writing')

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
    n = 8
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
    displacement_length = scale * 0.03 #change parameter around a bit
    for i in range(p2.GetNumberOfPoints()):
        p = list(p2.GetPoint(i))
        for j in range(3):
            p[j] = p[j] + (2.0 * random() -1 ) * displacement_length
        p2.SetPoint(i, p)

    transform = vtk.vtkThinPlateSplineTransform()
    transform.SetSourceLandmarks(p1)
    transform.SetTargetLandmarks(p2)
    transform.SetBasisToR2LogR()
    transform.Update()
    
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(pd)
    transform_filter.Update()
            
    vs = np.array( transform_filter.GetOutput().GetPoints().GetData() )
    face_labels = np.array( transform_filter.GetOutput().GetCellData().GetScalars() )         
    poly = dsa.WrapDataObject(transform_filter.GetOutput()).Polygons
    faces = np.reshape(poly,(-1,4))[:,1:4]
    
    
    #vcolor = []
    print('writing aug mesh')
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
    
    transform_filter_grid = vtk.vtkTransformPolyDataFilter()
    transform_filter_grid.SetTransform(transform)
    transform_filter_grid.SetInputData(grid_3d)
    transform_filter_grid.Update()
    
    #grid_points = np.array( transform_filter.GetOutput().GetPoints().GetData() )
    #grid_lines = np.array( transform_filter.GetOutput().GetLines().GetData() )         
    #poly = dsa.WrapDataObject(transform_filter.GetOutput()).Polygons
    #faces = np.reshape(poly,(-1,4))[:,1:4]
    
    #print('writing warped grid')
    #with open(out_name, 'w+') as f:
        #f.write("# vtk DataFile Version 4.2 \nvtk output \nASCII \n \nDATASET POLYDATA \nPOINTS %d float \n" % len(grid_points))
        #for vi, v in enumerate(grid_points):
            #vcol = ' %f %f %f' % (vcolor[vi, 0], vcolor[vi, 1], vcolor[vi, 2]) if vcolor is not None else ''
            #f.write("%f %f %f\n" % (v[0], v[1], v[2]))
        #f.write("LINES %d %d \n" % (len(grid_lines),2*len(grid_lines)-1))
        #for lines_id in range(len(grid_lines) - 1):
            #f.write("3 %d %d %d\n" % (grid_lines[face_id][0] , faces[face_id][1], faces[face_id][2]))
        #f.write("3 %d %d %d" % (faces[-1][0], faces[-1][1], faces[-1][2]))
        #f.write("\n \nCELL_DATA %d \nSCALARS scalars double \nLOOKUP_TABLE default" % len(faces))
        #for j,face in enumerate(faces):
            #if j%9 == 0:    
               #f.write("\n") 
            #f.write("%d " % face_labels[j])
    #print('done writing')
    
    filt = vtk.vtkConnectivityFilter()

    filt.SetInputData(grid_3d) # get the data from the MC alg.
    
    #filt.ColorRegionsOn()
    filt.Update()
    
    
    

    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    WIDTH=640
    HEIGHT=480
    renWin.SetSize(WIDTH,HEIGHT)

    #create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # mapper
    dataMapper1 = vtk.vtkPolyDataMapper()
    dataMapper2 = vtk.vtkPolyDataMapper()
    #dataMapper1.AddInputConnection(pd_in.GetOutputPort())
    #dataMapper2.AddInputConnection(filt.GetOutputPort())
    dataMapper1.AddInputConnection(transform_filter.GetOutputPort())
    dataMapper2.AddInputConnection(transform_filter_grid.GetOutputPort())
    
    #vtkPolyDataMapper = dataMapper.SetInputConnection(0,transform_filter.GetOutputPort())
    #dataMapper.SetInputConnection(0,transform_filter.GetOutputPort())
    #dataMapper.SetInputConnection(1,transform_filter_grid.GetOutputPort())

    # actor
    dataActor1 = vtk.vtkActor()
    dataActor2 = vtk.vtkActor()
    dataActor1.SetMapper(dataMapper1)
    dataActor2.SetMapper(dataMapper2)

    # assign actor to the renderer
    ren.AddActor(dataActor1)
    ren.AddActor(dataActor2)
    
    # enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()
    
'''
    print('writing warped grid')
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(out_name_grid_warped)
    writer.SetInputConnection(transform_filter.GetOutputPort())
    writer.Write()
    print('done writing')
    
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(out_name)
    writer.SetInputConnection(transform_filter.GetOutputPort())
    writer.Write()
    print('done writing')
    '''


volumetric_spline_augmentation()



