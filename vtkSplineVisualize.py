# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:37:54 2021

@author: lakri
"""
import vtk

vtkCxxRevisionMacro(vtk3DGridSource, "$Revision: 1.3 $");
vtkStandardNewMacro(vtk3DGridSource);

//----------------------------------------------------------------------------
def vtk3DGridSource(xL, yL, zL)
{
	this->XLength = fabs(xL);
	this->YLength = fabs(yL);
	self.ZLength = fabs(zL);
	
	self.Center[0] = 0.0;
	self.Center[1] = 0.0;
	self.Center[2] = 0.0;
	
	self.XCubes = 10;
	self.YCubes = 10;
	self.ZCubes = 10;
}

void vtk3DGridSource::Execute()
{
	vtk.vtkPolyData *output = this->GetOutput();
	
	vtk.vtkDebugMacro(<<"Creating 3D grid");

	pts = vtk.vtkPoints()
	lines = vtk.vtkCellArray()

	// number of verts in grid
	ntx = self.XCubes+1;
	nty = self.YCubes+1;
	ntz = self.ZCubes+1;
	
	nx2 = self.XCubes/2;
	ny2 = self.YCubes/2;
	nz2 = self.ZCubes/2;
	
	// side length of cube
	sx = self.XLength/self.XCubes;
	sy = self.YLength/self.YCubes;
	sz = self.ZLength/self.ZCubes;
	
	pts.SetNumberOfPos(ntx*nty*ntz);
	
	for tx in range(-nx2,nx2)
	{
		for ty in range(-ny2,ny2)
		{
			for tz in range(-nz2,nz2)
			{
				// poID of center po
				 pIDc = (tx+nx2) + (ty+ny2) * ntx + (tz+nz2) * ntx * nty;
				
				// start of cube
				 x = tx * sx + self.Center[0];
				 y = ty * sy + self.Center[1];
				 z = tz * sz + self.Center[2];
				
				pts.SetPoint(pIDc, x, y, z);
				
				if (tx != nx2)
				{
					 pID2 = (tx+nx2+1) + (ty+ny2) * ntx + (tz+nz2) * ntx * nty;
					pts->SetPoint(pID2, x+sx, y, z);
					lines.InsertNextCell(2);
					lines.InsertCellPo(pIDc);
					lines.InsertCellPo(pID2);
				}
				if (ty != ny2)
				{
					 pID2 = (tx+nx2) + (ty+1+ny2) * ntx + (tz+nz2) * ntx * nty;
					pts.SetPoint(pID2, x, y+sy, z);
					lines.InsertNextCell(2);
					lines.InsertCellPo(pIDc);
					lines.InsertCellPo(pID2);
				}
				if (tz != nz2)
				{
					 pID2 = (tx+nx2) + (ty+ny2) * ntx + (tz+1+nz2) * ntx * nty;
					pts.SetPoint(pID2, x, y, z+1);
					lines.InsertNextCell(2);
					lines.InsertCellPoint(pIDc);
					lines.InsertCellPoint(pID2);
				}
			}
		}
	}
	output.SetPoints(pts);
	pts.Delete();
	output.SetLines(lines);
	lines.Delete();
}
