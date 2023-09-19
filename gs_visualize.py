import open3d as o3d
import plyfile as ply
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.general_utils import safe_state
import itertools
nsteps= 10
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
phi = np.linspace(0,np.pi*2,nsteps)
alpha_lilmit = 0.2 # set as SIBR viewer default for ellipsoid
boxVertices = np.array(list(itertools.product((-1,1),(-1,1),(-1,1))))
boxindices = (
    0, 1, 2, 1, 3, 2,
    4, 6, 5, 5, 6, 7,
    0, 2, 4, 4, 2, 6,
    1, 5, 3, 5, 7, 3,
    0, 4, 1, 4, 5, 1,
    2, 3, 6, 3, 7, 6
)# face vertices ID
def quatToMat3(q):
  #torch version
  #weird coord
  qx = q[...,1] #y
  qy = q[...,2] #z 
  qz = q[...,3] #w 
  qw = q[...,0] #x

  qxx = qx * qx
  qyy = qy * qy
  qzz = qz * qz
  qxz = qx * qz
  qxy = qx * qy
  qyw = qy * qw
  qzw = qz * qw
  qyz = qy * qz
  qxw = qx * qw

  return torch.stack((
    torch.stack((1.0 - 2.0 * (qyy + qzz), 2.0 * (qxy - qzw), 2.0 * (qxz + qyw)),-1),
    torch.stack((2.0 * (qxy + qzw), 1.0 - 2.0 * (qxx + qzz), 2.0 * (qyz - qxw)),-1),
    torch.stack((2.0 * (qxz - qyw), 2.0 * (qyz + qxw), 1.0 - 2.0 * (qxx + qyy)),-1)),dim=1)
def render_gs_to_ellipsoid(xyz,feat_dc,alpha,rot,scale):
    # reference: SIBR viewer
    # src/projects/gaussianviewer/renderer vertex shader and fragment shader
    # GaussianSurfaceRenderer.cpp
    box = boxVertices*scale*2
    rot = quatToMat3(rot).T
    worldpos = xyz.reshape(3,1) + rot*box.T #3x8
    rgb = feat_dc*0.2+0.5



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
def render_sets(dataset : ModelParams, iteration=None,opacity=0.2):
    with torch.no_grad():
        model_path = dataset.model_path
        gaussians = GaussianModel(dataset.sh_degree)
        if iteration == None:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
            print(loaded_iter)
        else:
            loaded_iter = iteration
        #print(loaded_iter)
        if loaded_iter>0:
            if os.path.exists(model_path + "/render" + str(loaded_iter) + ".pth"):
                (model_params, _) = torch.load(model_path + "/render" + str(loaded_iter) + ".pth")
                gaussians.restore_no_training_args(model_params)   
            else:
                gaussians.load_ply(os.path.join(model_path,"point_cloud","iteration_" + str(loaded_iter),"point_cloud.ply"))
        #else:
            #gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        
        xyzs = gaussians._xyz
        distances = torch.sqrt(torch.norm(xyzs,p=2,dim=-1))
        #print(np.percentile(distances,33),np.percentile(distances,66),np.percentile(distances,50),np.percentile(distances,25),np.percentile(distances,75),np.percentile(distances,99))
        print(distances.mean())
        threshold = np.percentile(distances.cpu().numpy(),30)
        print(threshold)
        N = xyzs.shape[0]
        new_features_dc = gaussians._features_dc[:,0,:]
        new_features_rest = gaussians._features_rest
        shs = torch.cat([gaussians._features_dc[:,:1,:],gaussians._features_rest[:,:,:]],dim=1)
        #colors = new_features_dc


        # active and normalize
        opacities = torch.sigmoid(gaussians._opacity).squeeze()
        
        scales = torch.exp(gaussians._scaling)
        rotations = quatToMat3(gaussians._rotation/torch.sqrt(gaussians._rotation*gaussians._rotation).sum(dim=-1,keepdim=True))
        mask = torch.sqrt(torch.min(scales,dim=1)[0]*torch.max(scales,dim=1)[0])*1000
        print(mask.min(),mask.max(),mask.median())
        
        idx = torch.sort(distances)[1]
        #mask = opacities<0.2
        mask = torch.logical_and(mask>=2,opacities>opacity)
        mask = idx[mask]
        xyzs = xyzs[mask,...]
        opacities=opacities[mask,...]
        scales = scales[mask,...]
        rotations = rotations[mask,...]
        new_features_dc = new_features_dc[mask,...]
        distances = distances[mask]
        print(opacities.min(),opacities.max(),opacities.median())
        print(f'max gpu mem:{torch.cuda.max_memory_allocated()/(1024**2)}')
        transform = np.eye(4)
        res = o3d.geometry.TriangleMesh()
        chunk = 4e7
        idx = 0
        N = xyzs.shape[0]
        mesh_fore = o3d.geometry.TriangleMesh()
        mesh_back = o3d.geometry.TriangleMesh()
        meshset_fore = []
        meshset_back = []
        n_gs = 0
        for i in tqdm(range(N)):
            scale = scales[i,:]
            val = torch.sqrt(scale.min()*scale.max())
            resolution = min(10,int(val*4000))
            if (resolution>=2)and(scale.min()>1e-3):
                mesh = o3d.geometry.TriangleMesh.create_sphere(radius=2.0, resolution=resolution)
                np.fill_diagonal(transform[:3,:3],scale.cpu().numpy())
                if resolution > 6:
                    vertices = torch.tensor(np.asarray(mesh.vertices)).cuda()
                    mesh.vertex_colors = o3d.utility.Vector3dVector(sh_to_rgb(vertices/2,shs[i,...]).clamp(0,1).cpu().numpy())
                else:
                    color = (SH_C0 * new_features_dc[i,...]+0.5)
                    mesh.paint_uniform_color((color).clamp(0,1).cpu().numpy().tolist())
                #print(transform)
                mesh = mesh.transform(transform)
                mesh = mesh.rotate(rotations[i,...].cpu().numpy().T)
                mesh = mesh.translate(xyzs[i,...].cpu().numpy(),relative=False)
                #alpha = opacities[i].item()
                
                #if alpha<0.2:
                    #continue
                #mesh.compute_vertex_normals()
                mesh.remove_duplicated_vertices()
                mesh.remove_degenerate_triangles()
                
                #print(resolution,scale,np.asarray(mesh.vertices).shape,np.asarray(mesh.triangles).shape)
                #exit()
                res += mesh
                ptn = np.asarray(res.vertices).shape[0]
                '''if distances[i]<threshold:
                    mesh_fore += mesh
                else:
                    mesh_back+=mesh'''
                if (ptn>=chunk):
                    res.remove_duplicated_triangles()
                    o3d.io.write_triangle_mesh(os.path.join(model_path,f'ellipsoid_{idx}_{opacity}.ply'),res)
                    print(np.asarray(res.vertices).shape,np.asarray(res.triangles).shape)                
                    res = o3d.geometry.TriangleMesh()
                    idx+=1
                    '''mesh_fore.remove_duplicated_triangles()
                    mesh_back.remove_duplicated_triangles()
                    o3d.io.write_triangle_mesh(os.path.join(model_path,f'ellipsoid_{idx}.ply'),mesh_fore)
                    o3d.io.write_triangle_mesh(os.path.join(model_path,f'ellipsoid_{idx}.ply'),mesh_back)
                    mesh_fore = o3d.geometry.TriangleMesh()
                    mesh_back = o3d.geometry.TriangleMesh()'''
        o3d.io.write_triangle_mesh(os.path.join(model_path,f'ellipsoid_{idx}_{opacity}.ply'),res)     
        #o3d.visualization.draw_geometries(meshset)
        '''mesh_fore = o3d.geometry.TriangleMesh()
        mesh_back = o3d.geometry.TriangleMesh()
        for i in meshset_fore:
            mesh_fore+=i
        for i in meshset_back:
            mesh_back+=i
        o3d.io.write_triangle_mesh(os.path.join(model_path,f'ellipsoid_fore_{opacity}.ply'),mesh_fore)
        o3d.io.write_triangle_mesh(os.path.join(model_path,f'ellipsoid_back_{opacity}.ply'),mesh_back)'''
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    #pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--opacity", default=0.2, type=float)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args),opacity=args.opacity)