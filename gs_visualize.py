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




def render_sets(dataset : ModelParams, iteration=None):
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
        N = xyzs.shape[0]
        new_features_dc = gaussians._features_dc[:,0,:]
        new_features_rest = gaussians._features_rest
        #colors = new_features_dc


        # active and normalize
        opacities = torch.sigmoid(gaussians._opacity).squeeze()
        
        scales = torch.exp(gaussians._scaling)
        rotations = quatToMat3(gaussians._rotation/torch.sqrt(gaussians._rotation*gaussians._rotation).sum(dim=-1,keepdim=True))
        mask = torch.min(scales,dim=1)[0]*100
        print(mask.min(),mask.max(),mask.median())
        
        idx = torch.sort(mask,descending=True)[1]
        mask = opacities<0.2
        #mask = mask>=2#torch.bitwise_and(mask>1,opacities<0.2)
        mask = idx#[mask]
        xyzs = xyzs[mask,...]
        opacities=opacities[mask,...]
        scales = scales[mask,...]
        rotations = rotations[mask,...]
        new_features_dc = new_features_dc[mask,...]
        print(opacities.min(),opacities.max(),opacities.median())
        print(f'max gpu mem:{torch.cuda.max_memory_allocated()/(1024**2)}')
        transform = np.eye(4)
        res = o3d.geometry.TriangleMesh()
        chunk = 1000000
        idx = 0
        meshset=[]
        N = xyzs.shape[0]
        for i in tqdm(range(N)):
            scale = scales[i,:]
            resolution = min(10,max(int(scale.min()*100),2))
            if (resolution<=1):
                break
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=2.0, resolution=resolution)
            np.fill_diagonal(transform[:3,:3],scale.cpu().numpy())
            #print(transform)
            mesh = mesh.transform(transform)
            mesh = mesh.rotate(rotations[i,...].cpu().numpy().T)
            mesh = mesh.translate(xyzs[i,...].cpu().numpy(),relative=False)
            #alpha = opacities[i].item()
            color = (SH_C0 * new_features_dc[i,...]+0.5)
            #if alpha<0.2:
                #continue
            mesh.paint_uniform_color((color).clamp(0,1).cpu().numpy().tolist())
            #mesh.compute_vertex_normals()
            
            '''res += mesh
            if (i%chunk==0) and (i>0):
                o3d.io.write_triangle_mesh(os.path.join(model_path,f'ellipsoid_{idx}.ply'),res)
                res = o3d.geometry.TriangleMesh()
                idx+=1'''
            meshset.append(mesh)
        #o3d.visualization.draw_geometries(meshset)
        res= o3d.geometry.TriangleMesh()
        for i in meshset:
            res+=i
        o3d.io.write_triangle_mesh(os.path.join(model_path,f'ellipsoid_{idx}.ply'),res)
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    #pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args))