import os
import cv2
import numpy as np
import trimesh

import torch
import torch.nn.functional as F
import pymeshlab as pml
from gaussian_renderer import render
from grid_put import mipmap_linear_grid_put_2d
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
def dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)
def decimate_mesh(
    verts, faces, target, backend="pymeshlab", remesh=False, optimalplacement=True
):
    # optimalplacement: default is True, but for flat mesh must turn False to prevent spike artifect.

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, "mesh")  # will copy!

    # filters
    # ms.meshing_decimation_clustering(threshold=pml.Percentage(1))
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target), optimalplacement=optimalplacement
    )

    if remesh:
        # ms.apply_coord_taubin_smoothing()
        ms.meshing_isotropic_explicit_remeshing(
            iterations=3, targetlen=pml.Percentage(1)
        )

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(
        f"[INFO] mesh decimation: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}"
    )

    return verts, faces


def clean_mesh(
    verts,
    faces,
    v_pct=1,
    min_f=64,
    min_d=20,
    repair=True,
    remesh=True,
    remesh_size=0.01,
):
    # verts: [N, 3]
    # faces: [N, 3]

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, "mesh")  # will copy!

    # filters
    ms.meshing_remove_unreferenced_vertices()  # verts not refed by any faces

    if v_pct > 0:
        ms.meshing_merge_close_vertices(
            threshold=pml.Percentage(v_pct)
        )  # 1/10000 of bounding box diagonal

    ms.meshing_remove_duplicate_faces()  # faces defined by the same verts
    ms.meshing_remove_null_faces()  # faces with area == 0

    if min_d > 0:
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pml.Percentage(min_d)
        )

    if min_f > 0:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_f)

    if repair:
        # ms.meshing_remove_t_vertices(method=0, threshold=40, repeat=True)
        ms.meshing_repair_non_manifold_edges(method=0)
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

    if remesh:
        # ms.apply_coord_taubin_smoothing()
        ms.meshing_isotropic_explicit_remeshing(
            iterations=3, targetlen=pml.AbsoluteValue(remesh_size)
        )

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(
        f"[INFO] mesh cleaning: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}"
    )

    return verts, faces


class Mesh:
    def __init__(
        self,
        v=None,
        f=None,
        vn=None,
        fn=None,
        vt=None,
        ft=None,
        albedo=None,
        device=None,
    ):
        self.device = device
        self.v = v
        self.vn = vn
        self.vt = vt
        self.f = f
        self.fn = fn
        self.ft = ft
        # only support a single albedo
        self.albedo = albedo

        self.ori_center = 0
        self.ori_scale = 1

    @classmethod
    def load(cls, path=None, resize=True, **kwargs):
        # assume init with kwargs
        if path is None:
            mesh = cls(**kwargs)
        # obj supports face uv
        elif path.endswith(".obj"):
            mesh = cls.load_obj(path, **kwargs)
        # trimesh only supports vertex uv, but can load more formats
        else:
            mesh = cls.load_trimesh(path, **kwargs)

        print(f"[Mesh loading] v: {mesh.v.shape}, f: {mesh.f.shape}")
        # auto-normalize
        if resize:
            mesh.auto_size()
        # auto-fix normal
        if mesh.vn is None:
            mesh.auto_normal()
        print(f"[Mesh loading] vn: {mesh.vn.shape}, fn: {mesh.fn.shape}")
        # auto-fix texture
        if mesh.vt is None:
            mesh.auto_uv(cache_path=path)
        print(f"[Mesh loading] vt: {mesh.vt.shape}, ft: {mesh.ft.shape}")

        return mesh

    # load from obj file
    @classmethod
    def load_obj(cls, path, albedo_path=None, device=None, init_empty_tex=False):
        assert os.path.splitext(path)[-1] == ".obj"

        mesh = cls()

        # device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mesh.device = device

        # try to find texture from mtl file
        if albedo_path is None:
            mtl_path = path.replace(".obj", ".mtl")
            if os.path.exists(mtl_path):
                with open(mtl_path, "r") as f:
                    lines = f.readlines()
                for line in lines:
                    split_line = line.split()
                    # empty line
                    if len(split_line) == 0:
                        continue
                    prefix = split_line[0]
                    # NOTE: simply use the first map_Kd as albedo!
                    if "map_Kd" in prefix:
                        albedo_path = os.path.join(os.path.dirname(path), split_line[1])
                        print(f"[load_obj] use texture from: {albedo_path}")
                        break

        if init_empty_tex or albedo_path is None or not os.path.exists(albedo_path):
            # init an empty texture
            print(f"[load_obj] init empty albedo!")
            # albedo = np.random.rand(1024, 1024, 3).astype(np.float32)
            albedo = np.ones((1024, 1024, 3), dtype=np.float32) * np.array(
                [0.5, 0.5, 0.5]
            )  # default color
        else:
            albedo = cv2.imread(albedo_path, cv2.IMREAD_UNCHANGED)
            albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
            albedo = albedo.astype(np.float32) / 255
            print(f"[load_obj] load texture: {albedo.shape}")

            # import matplotlib.pyplot as plt
            # plt.imshow(albedo)
            # plt.show()

        mesh.albedo = torch.tensor(albedo, dtype=torch.float32, device=device)

        # load obj
        with open(path, "r") as f:
            lines = f.readlines()

        def parse_f_v(fv):
            # pass in a vertex term of a face, return {v, vt, vn} (-1 if not provided)
            # supported forms:
            # f v1 v2 v3
            # f v1/vt1 v2/vt2 v3/vt3
            # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
            # f v1//vn1 v2//vn2 v3//vn3
            xs = [int(x) - 1 if x != "" else -1 for x in fv.split("/")]
            xs.extend([-1] * (3 - len(xs)))
            return xs[0], xs[1], xs[2]

        # NOTE: we ignore usemtl, and assume the mesh ONLY uses one material (first in mtl)
        vertices, texcoords, normals = [], [], []
        faces, tfaces, nfaces = [], [], []
        for line in lines:
            split_line = line.split()
            # empty line
            if len(split_line) == 0:
                continue
            # v/vn/vt
            prefix = split_line[0].lower()
            if prefix == "v":
                vertices.append([float(v) for v in split_line[1:]])
            elif prefix == "vn":
                normals.append([float(v) for v in split_line[1:]])
            elif prefix == "vt":
                val = [float(v) for v in split_line[1:]]
                texcoords.append([val[0], 1.0 - val[1]])
            elif prefix == "f":
                vs = split_line[1:]
                nv = len(vs)
                v0, t0, n0 = parse_f_v(vs[0])
                for i in range(nv - 2):  # triangulate (assume vertices are ordered)
                    v1, t1, n1 = parse_f_v(vs[i + 1])
                    v2, t2, n2 = parse_f_v(vs[i + 2])
                    faces.append([v0, v1, v2])
                    tfaces.append([t0, t1, t2])
                    nfaces.append([n0, n1, n2])

        mesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
        mesh.vt = (
            torch.tensor(texcoords, dtype=torch.float32, device=device)
            if len(texcoords) > 0
            else None
        )
        mesh.vn = (
            torch.tensor(normals, dtype=torch.float32, device=device)
            if len(normals) > 0
            else None
        )

        mesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
        mesh.ft = (
            torch.tensor(tfaces, dtype=torch.int32, device=device)
            if texcoords is not None
            else None
        )
        mesh.fn = (
            torch.tensor(nfaces, dtype=torch.int32, device=device)
            if normals is not None
            else None
        )

        return mesh

    @classmethod
    def load_trimesh(cls, path, device=None):
        mesh = cls()

        # device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mesh.device = device

        # use trimesh to load glb, assume only has one single RootMesh...
        _data = trimesh.load(path)
        if isinstance(_data, trimesh.Scene):
            mesh_keys = list(_data.geometry.keys())
            assert (
                len(mesh_keys) == 1
            ), f"{path} contains more than one meshes, not supported!"
            _mesh = _data.geometry[mesh_keys[0]]

        elif isinstance(_data, trimesh.Trimesh):
            _mesh = _data

        else:
            raise NotImplementedError(f"type {type(_data)} not supported!")

        # TODO: exception handling if no material
        _material = _mesh.visual.material
        if isinstance(_material, trimesh.visual.material.PBRMaterial):
            texture = np.array(_material.baseColorTexture).astype(np.float32) / 255
        elif isinstance(_material, trimesh.visual.material.SimpleMaterial):
            texture = (
                np.array(_material.to_pbr().baseColorTexture).astype(np.float32) / 255
            )
        else:
            raise NotImplementedError(f"material type {type(_material)} not supported!")

        print(f"[load_obj] load texture: {texture.shape}")
        mesh.albedo = torch.tensor(texture, dtype=torch.float32, device=device)

        vertices = _mesh.vertices
        texcoords = _mesh.visual.uv
        texcoords[:, 1] = 1 - texcoords[:, 1]
        normals = _mesh.vertex_normals

        # trimesh only support vertex uv...
        faces = tfaces = nfaces = _mesh.faces

        mesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
        mesh.vt = (
            torch.tensor(texcoords, dtype=torch.float32, device=device)
            if len(texcoords) > 0
            else None
        )
        mesh.vn = (
            torch.tensor(normals, dtype=torch.float32, device=device)
            if len(normals) > 0
            else None
        )

        mesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
        mesh.ft = (
            torch.tensor(tfaces, dtype=torch.int32, device=device)
            if texcoords is not None
            else None
        )
        mesh.fn = (
            torch.tensor(nfaces, dtype=torch.int32, device=device)
            if normals is not None
            else None
        )

        return mesh

    # aabb
    def aabb(self):
        return torch.min(self.v, dim=0).values, torch.max(self.v, dim=0).values

    # unit size
    @torch.no_grad()
    def auto_size(self):
        vmin, vmax = self.aabb()
        self.ori_center = (vmax + vmin) / 2
        self.ori_scale = 1.2 / torch.max(vmax - vmin).item() # to ~ [-0.6, 0.6]
        self.v = (self.v - self.ori_center) * self.ori_scale

    def auto_normal(self):
        i0, i1, i2 = self.f[:, 0].long(), self.f[:, 1].long(), self.f[:, 2].long()
        v0, v1, v2 = self.v[i0, :], self.v[i1, :], self.v[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        vn = torch.zeros_like(self.v)
        vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        vn = torch.where(
            dot(vn, vn) > 1e-20,
            vn,
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device),
        )
        vn = safe_normalize(vn)

        self.vn = vn
        self.fn = self.f

    def auto_uv(self, cache_path=None):
        # try to load cache
        if cache_path is not None:
            cache_path = cache_path.replace(".obj", "_uv.npz")

        if cache_path is not None and os.path.exists(cache_path):
            data = np.load(cache_path)
            vt_np, ft_np = data["vt"], data["ft"]
        else:
            import xatlas

            v_np = self.v.detach().cpu().numpy()
            f_np = self.f.detach().int().cpu().numpy()
            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            # chart_options.max_iterations = 4
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

            # save to cache
            if cache_path is not None:
                np.savez(cache_path, vt=vt_np, ft=ft_np)

        vt = torch.from_numpy(vt_np.astype(np.float32)).to(self.device)
        ft = torch.from_numpy(ft_np.astype(np.int32)).to(self.device)

        self.vt = vt
        self.ft = ft

    def to(self, device):
        self.device = device
        for name in ["v", "f", "vn", "fn", "vt", "ft", "albedo"]:
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.to(device))
        return self
    
    # write to ply file (only geom)
    def write_ply(self, path):
        assert path.endswith(".ply")

        v_np = self.v.detach().cpu().numpy()
        f_np = self.f.detach().cpu().numpy()

        _mesh = trimesh.Trimesh(vertices=v_np, faces=f_np)
        _mesh.export(path)

    # write to obj file
    def write(self, path):
        mtl_path = path.replace(".obj", ".mtl")
        albedo_path = path.replace(".obj", "_albedo.png")

        v_np = self.v.detach().cpu().numpy()
        vt_np = self.vt.detach().cpu().numpy() if self.vt is not None else None
        vn_np = self.vn.detach().cpu().numpy() if self.vn is not None else None
        f_np = self.f.detach().cpu().numpy()
        ft_np = self.ft.detach().cpu().numpy() if self.ft is not None else None
        fn_np = self.fn.detach().cpu().numpy() if self.fn is not None else None

        with open(path, "w") as fp:
            fp.write(f"mtllib {os.path.basename(mtl_path)} \n")

            for v in v_np:
                fp.write(f"v {v[0]} {v[1]} {v[2]} \n")

            if vt_np is not None:
                for v in vt_np:
                    fp.write(f"vt {v[0]} {1 - v[1]} \n")

            if vn_np is not None:
                for v in vn_np:
                    fp.write(f"vn {v[0]} {v[1]} {v[2]} \n")

            fp.write(f"usemtl defaultMat \n")
            for i in range(len(f_np)):
                fp.write(
                    f'f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1 if ft_np is not None else ""}/{fn_np[i, 0] + 1 if fn_np is not None else ""} \
                             {f_np[i, 1] + 1}/{ft_np[i, 1] + 1 if ft_np is not None else ""}/{fn_np[i, 1] + 1 if fn_np is not None else ""} \
                             {f_np[i, 2] + 1}/{ft_np[i, 2] + 1 if ft_np is not None else ""}/{fn_np[i, 2] + 1 if fn_np is not None else ""} \n'
                )

        with open(mtl_path, "w") as fp:
            fp.write(f"newmtl defaultMat \n")
            fp.write(f"Ka 1 1 1 \n")
            fp.write(f"Kd 1 1 1 \n")
            fp.write(f"Ks 0 0 0 \n")
            fp.write(f"Tr 1 \n")
            fp.write(f"illum 1 \n")
            fp.write(f"Ns 0 \n")
            fp.write(f"map_Kd {os.path.basename(albedo_path)} \n")

        albedo = self.albedo.detach().cpu().numpy()
        albedo = (albedo * 255).astype(np.uint8)
        cv2.imwrite(albedo_path, cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR))
def extract_mesh(gaussians, path, density_thresh=1, resolution=128, decimate_target=1e5):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    occ = gaussians.extract_fields(resolution).detach().cpu().numpy()

    import mcubes
    vertices, triangles = mcubes.marching_cubes(occ, density_thresh)
    vertices = vertices / (resolution - 1.0) * 2 - 1

    # transform back to the original space
    vertices = vertices / gaussians.scale + gaussians.center.detach().cpu().numpy()

    vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=0.015)
    if decimate_target > 0 and triangles.shape[0] > decimate_target:
        vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)

    v = torch.from_numpy(vertices.astype(np.float32)).contiguous().cuda()
    f = torch.from_numpy(triangles.astype(np.int32)).contiguous().cuda()

    print(
        f"[INFO] marching cubes result: {v.shape} ({v.min().item()}-{v.max().item()}), {f.shape}"
    )

    mesh = Mesh(v=v, f=f, device='cuda')

    return mesh
def conv_gs_to_mesh(save_path, iterations, viewpoint_cam_stack, gaussians, pipe, background, density_thresh=1,texture_size=1024):
    import time
    start = time.time()
    path = os.path.join(save_path, f'extract_mesh_{iterations}.obj')
    mesh = extract_mesh(gaussians,path, density_thresh)

    # perform texture extraction
    print(f"[INFO] unwrap uv...")
    h = w = texture_size
    mesh.auto_uv()
    mesh.auto_normal()
    
    albedo = torch.zeros((h, w, 3), device="cuda", dtype=torch.float32)
    cnt = torch.zeros((h, w, 1), device="cuda", dtype=torch.float32)


    render_resolution = 512

    import nvdiffrast.torch as dr

    glctx = dr.RasterizeCudaContext()
    print(f'mesh intialize:{time.time()-start}')
    start = time.time()
    print(len(viewpoint_cam_stack))
    for viewpoint_cam in viewpoint_cam_stack:
        # render image
        h,w = int(viewpoint_cam.image_height),int(viewpoint_cam.image_width)
        viewpoint_cam.image_height,viewpoint_cam.image_width =render_resolution,render_resolution #(h//8)*8,(w//8)*8
        #keep it multipler
        cur_out = render(viewpoint_cam, gaussians, pipe, background)

        rgbs = cur_out["render"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
        h,w = rgbs.shape[2:]
        print(rgbs.min(),rgbs.max())
        # get coordinate in texture image
        pose = viewpoint_cam.world_view_transform
        proj = viewpoint_cam.projection_matrix  #torch.from_numpy(self.cam.perspective.astype(np.float32)).to('cuda')

        v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), pose).float().unsqueeze(0)
        v_clip = v_cam @ proj
        rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (viewpoint_cam.image_height, viewpoint_cam.image_width))
        
        depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]

        depth = depth.squeeze(0) # [H, W, 1]

        alpha = (rast[0, ..., 3:] > 0).float()

        uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]
        print(uvs.min(),uvs.max(),rast[0, ..., 3:].max())
        # use normal to produce a back-project mask
        normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
        normal = safe_normalize(normal[0])

        # rotated normal (where [0, 0, 1] always faces camera)
        rot_normal = normal @ pose[:3, :3]
        viewcos = rot_normal[..., [2]]

        mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
        #print(mask.shape,rgbs.shape)
        mask = mask.view(-1)
        

        uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
        rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
        
        # update texture image
        cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
            texture_size, texture_size,
            uvs[..., [1, 0]] * 2 - 1,
            rgbs,
            min_resolution=256,
            return_count=True,
        )
        print('albedo',albedo.min(),albedo.max())
        # albedo += cur_albedo
        # cnt += cur_cnt
        mask = cnt.squeeze(-1) < 0.1
        albedo[mask] += cur_albedo[mask]
        cnt[mask] += cur_cnt[mask]
        exit()
    print(f'Texture updating:{time.time()-start}')
    start =time.time()
    mask = cnt.squeeze(-1) > 0
    albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

    mask = mask.view(texture_size, texture_size)

    albedo = albedo.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    # dilate texture
    from sklearn.neighbors import NearestNeighbors
    from scipy.ndimage import binary_dilation, binary_erosion

    inpaint_region = binary_dilation(mask, iterations=32)
    inpaint_region[mask] = 0

    search_region = mask.copy()
    not_search_region = binary_erosion(search_region, iterations=3)
    search_region[not_search_region] = 0
    if search_region.sum()>0:
        search_coords = np.stack(np.nonzero(search_region), axis=-1)
        inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

        knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
            search_coords
        )
        _, indices = knn.kneighbors(inpaint_coords)

        albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

    mesh.albedo = torch.from_numpy(albedo).to('cuda')
    mesh.write(path)
  
    print(f'Other time:{time.time()-start}')
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        print(f'finish prepare {scene.loaded_iter}')
        scene.save(30000,save_ply=True)
        conv_gs_to_mesh(dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
        print(f'max gpu mem:{torch.cuda.max_memory_allocated()/(1024**2)}')

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args))