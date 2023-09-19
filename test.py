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
def sh_to_rgb(vertices,sh):
	x = vertices[:,:1]
	y = vertices[:,1:2]
	z = vertices[:,2:3]
	result = SH_C0 * sh[0]
	result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3]
	xx = x * x
	yy = y * y
	zz = z * z
	xy = x * y 
	yz = y * z
	xz = x * z
	result = result + SH_C2[0] * xy * sh[4] +\
				SH_C2[1] * yz * sh[5] +\
				SH_C2[2] * (2.0 * zz - xx - yy) * sh[6] +\
				SH_C2[3] * xz * sh[7] +\
				SH_C2[4] * (xx - yy) * sh[8]
	result = result +\
			SH_C3[0] * y * (3.0 * xx - yy) * sh[9] +\
			SH_C3[1] * xy * z * sh[10] +\
			SH_C3[2] * y * (4.0 * zz - xx - yy) * sh[11] +\
			SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[12] +\
			SH_C3[4] * x * (4.0 * zz - xx - yy) * sh[13] +\
			SH_C3[5] * z * (xx - yy) * sh[14] +\
			SH_C3[6] * x * (xx - 3.0 * yy) * sh[15]
	result += 0.5
	return result
nstep = 5
dtheta = np.pi/nstep
dphi = np.pi*2/nstep
xs,ys,zs = [],[],[]
xyzs = np.zeros((nstep*nstep,3))
mesh = o3d.geometry.TriangleMesh.create_sphere(radius=2.0, resolution=20)
vertices = np.asarray(mesh.vertices)
coeffs = np.random.rand(16,3)
colors = sh_to_rgb(vertices,coeffs)
o3d.visualization.draw_geometries([mesh])
mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([mesh])


