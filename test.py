import numpy as np
import open3d as o3d
import pymesh
import itertools
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2= [
	1.0925484305920792,
	-1.0925484305920792,
	0.31539156525252005,
	-1.0925484305920792,
	0.5462742152960396]
SH_C3 =[
	-0.5900435899266435,
	2.890611442640554,
	-0.4570457994644658,
	0.3731763325901154,
	-0.4570457994644658,
	1.445305721320277,
	-0.5900435899266435]
nstep = 5
dtheta = np.pi/nstep
dphi = np.pi*2/nstep
xs,ys,zs = [],[],[]
xyzs = np.zeros((nstep*nstep,3))
mesh = o3d.geometry.TriangleMesh.create_sphere(radius=2.0, resolution=20)
vertices = mesh.vertices.asarry()
x,y,z = vertices[:,0],vertices[:,1],vertices[:,2]



