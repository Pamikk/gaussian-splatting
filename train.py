#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
#from utils.loss_utils import l1_loss, ssim
from utils.ssim import SSIM
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision.utils import save_image
import time
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
def project_3d_bbox(bbox,camera):
    twodbbox = camera.intrinsic*camera.exitrinsic*bbox
import torchvision
blur = torchvision.transforms.GaussianBlur(5, sigma=(0.1, 2.0))
l1_loss = torch.nn.L1Loss(reduction='none')
#l1_loss_mean = torch.nn.L1Loss()
MSE_Loss = torch.nn.MSELoss(reduction='none')
ssim = SSIM(reduction='none')#torch.nn.DataParallel(SSIM())
accum=1
rm_ssim_after_iters = 30000
rm_ssim_after = True
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    print(f'remove ssim after{rm_ssim_after}:{rm_ssim_after_iters}')
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    first_stage_iter = checkpoint_iterations[0] if len(checkpoint_iterations)>0 else 0
    total_time = 0
    total_time_accum = 0

    train_time = 0
    lr_time_accum = 0.0
    train_time_accum = 0.0
    load_cam_time = 0
    load_cam_time_accum = 0
    loss_time = 0
    loss_time_accum = 0
    render_time = 0
    render_time_accum = 0.0

    densify_time = 0
    densify_time_accum = 0
    optimize_time = 0
    optimize_time_accum = 0

    logging_time = 0
    logging_time_accum = 0
    tqdm_time = 0
    tqdm_time_accum = 0
    tensorboard_time = 0
    tensorboard_time_accum = 0
    radii_time = 0
    radii_time_accum = 0
    cuda_time = 0
    save_time = 0
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    viewpoint_cam_stack = scene.getTrainCameras().copy()
    loss_cal_time=0
    bbox = torch.tensor(scene.stbbox).cuda()
    for iteration in range(first_iter, opt.iterations + 1):     
        total_time = time.time()
        torch.cuda.empty_cache()
        iter_start.record()
        train_start = time.time()

        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            #viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_stack = list(range(len(viewpoint_cam_stack)))
        viewpoint_cam = viewpoint_cam_stack[viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))]

        # Render
        render_time = time.time()
        #if (iteration - 1) == debug_from:
            #pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii,depth= render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"].detach()
        render_time_accum += time.time() - render_time
        
        torch.cuda.synchronize()
        #print(viewspace_point_tensor.shape,visibility_filter.shape,radii.shape)
        # Loss
        loss_time = time.time()
        dweights =  gaussians.add_depth_stats(depth)
        
        #dweights = blur(dweights)
        #blur to cover the holes with bad depth calculation
        #print(dweights.min(),dweights.max())
        #exit()
        gt_image = viewpoint_cam.original_image#possible to load all images into cuda first
        #from torchvision.utils import save_image
        #save_image(gt_image,'tmp_gt.png')
        #depth = gaussians.render_depth_map(viewpoint_cam,visibility_filter)
        #print(depth.shape,image.shape)
        Ll1 = l1_loss(image,gt_image)
        
        ssim_cal = 1.0 - ssim(image, gt_image)
        mse = MSE_Loss(image,gt_image)
        if iteration < first_stage_iter:
            loss = (1.0 - opt.lambda_dssim) * Ll1+ opt.lambda_dssim * ssim_cal +  mse
        else:
            loss = (1.0 - opt.lambda_dssim) * Ll1+ opt.lambda_dssim * ssim_cal + mse
            loss += dweights*loss
        loss = loss.mean()
            
        loss_cal_time += time.time() - loss_time
        loss.backward()
        cur_loss = loss.item()
        
        
        
        iter_end.record()
        torch.cuda.synchronize()
        loss_time_accum += time.time() - loss_time
        cuda_time_single = iter_start.elapsed_time(iter_end)
        cuda_time +=cuda_time_single
        
            
        
        
        train_end = time.time()
        train_time_accum += train_end - train_start
        
        with torch.no_grad():
            logging_time=time.time()
            # Progress bar
            ema_loss_for_log = 0.4 * cur_loss + 0.6 * ema_loss_for_log
            #torch.cuda.synchronize()
            
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:05f}","GS#": f"{gaussians._xyz.shape[0]}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            

            # Log and save
            training_report(tb_writer, iteration, Ll1.mean(), loss, l1_loss, cuda_time_single, testing_iterations, scene, render, (pipe, background))

            if (iteration in saving_iterations):
                print(f'max gpu mem:{torch.cuda.max_memory_allocated()/(1024**2)}')
                print(f'current gpu utilization:{torch.cuda.utilization(device=None)}')
                
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                depth = (depth-depth.min())/(depth.max()-depth.min())
                #dweights = (dweights-dweights.min())/(dweights.max()-dweights.min())
                save_image(depth,os.path.join(scene.model_path,f'depth_{iteration}.png'))
                save_image(dweights,os.path.join(scene.model_path,f'dweights_{iteration}.png'))
                save_image(gt_image,os.path.join(scene.model_path,f'gt_{iteration}.png'))
                print(dweights.min(),dweights.max())
                #scene.save(iteration)
            save_time += time.time()- logging_time
            densify_time = time.time()
            # Densification
            
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                radii_time = time.time()
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                bbox_filter = gaussians.get_inside_mask(bbox)
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                radii_time_accum += time.time()-radii_time

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            #torch.cuda.synchronize()
            densify_time_accum += time.time()- densify_time
        optimize_time = time.time()
        # Optimizer step
        if (iteration < opt.iterations):
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
            torch.cuda.synchronize()
        optimize_time_accum += time.time() - optimize_time
        if (iteration in checkpoint_iterations):
            print("\n[ITER {}] Saving Checkpoint".format(iteration))
            torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        total_time_accum += time.time()-total_time
    print("\n[ITER {}] Saving Gaussians".format(iteration))
    scene.save(iteration)
    print("|")
    print(" | total: ", total_time_accum-save_time)
    print("\t |")
    print("\t |-- training (total): ", train_time_accum)
    print("\t \t|-- render time: ", render_time_accum)
    print("\t \t|-- loss time: ", loss_time_accum)
    print("\t \t|-- loss cal time: ", loss_cal_time)
    print("\t \t|-- cuda time: ", cuda_time/1000)
    print("\t \t|-- other training time: ", train_time_accum - render_time_accum - loss_time_accum)
    print("\t |")
    print("\t |-- densify: ", densify_time_accum)
    print("\t |")
    print("\t |-- optimize: ", optimize_time_accum)
    print("\t |")
    print(gaussians.near,gaussians.far)
    print(gaussians.dwmax,gaussians.depth_mid)
    print(gaussians.adaptive_constant,gaussians.dscale)
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000,10_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=list(range(0,30000,5000)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    #torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
    # Start GUI server, configure and run training
    #network_gui.init(args.ip, args.port)
    #torch.autograd.set_detect_anomaly(args.detect_anomaly)
    #with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
        #profile_memory=True, record_shapes=True) as prof:
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    #with open('cuda-profile.txt','w') as f:
        #f.writelines(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total"))
    # All done
    print("\nTraining complete.")
