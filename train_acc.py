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
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import time
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
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

    total_time = 0
    total_time_accum = 0

    train_time = 0
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

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    #progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    viewpoint_cam_stack = scene.getTrainCameras().copy()
    sparsity = []
    loss_cal_time=0
    for iteration in range(first_iter, opt.iterations + 1):     
        total_time = time.time()   
        '''if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None'''

        iter_start.record()
        train_start = time.time()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        load_cam_time = time.time()
        # Pick a random Camera
        if not viewpoint_stack:
            #viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_stack = list(range(len(viewpoint_cam_stack)))
        viewpoint_cam = viewpoint_cam_stack[viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))]
        load_cam_time_accum += time.time() - load_cam_time

        # Render
        render_time = time.time()
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        render_time_accum += time.time() - render_time

        # Loss
        loss_time = time.time()
        gt_image = viewpoint_cam.original_image.cuda()#possible to load all images into cuda first
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss_cal_time += time.time() - loss_time
        loss.backward()
        loss_time_accum += time.time() - loss_time

        iter_end.record()
        train_end = time.time()
        train_time_accum += train_end - train_start

        with torch.no_grad():
            # Progress bar
            logging_time = time.time()
            tqdm_time = time.time()
            #ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            torch.cuda.synchronize()
            
            #if iteration % 10 == 0:
                #progress_bar.set_postfix({"Loss": "nan"})
                #progress_bar.update(10)
            #if iteration == opt.iterations:
                #progress_bar.close()
            tqdm_time_accum += time.time() - tqdm_time
            
            

            # Log and save
            tensorboard_time = time.time()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            tensorboard_time_accum += time.time() - tensorboard_time

            if (iteration in saving_iterations):
                print(f'max gpu mem:{torch.cuda.max_memory_allocated()/(1024**2)}')
                print(f'current gpu utilization:{torch.cuda.utilization(device=None)}')
                
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            logging_time_accum += time.time() - logging_time
   
            # Densification
            densify_time = time.time()
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                #print(visibility_filter.sum().item())
                #print(visibility_filter.shape,gaussians.max_radii2D.shape)
                sparsity.append((visibility_filter.sum().item()*1.0)/(1.0*visibility_filter.shape[0]))
                radii_time = time.time()
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                radii_time_accum += time.time()-radii_time

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            densify_time_accum += time.time() - densify_time
            optimize_time = time.time()
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            optimize_time_accum += time.time() - optimize_time
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        total_time_accum += time.time() - total_time
        #break
    print("|")
    print(" | total: ", total_time_accum)
    print("\t |")
    print("\t |-- training (total): ", train_time_accum)
    print("\t \t|-- load_cam time: ", load_cam_time_accum)
    print("\t \t|-- render time: ", render_time_accum)
    print("\t \t|-- loss time: ", loss_time_accum)
    print("\t \t|-- other training time: ", train_time_accum - load_cam_time_accum - render_time_accum - loss_time_accum)
    print("\t |")
    print("\t |-- densify: ", densify_time_accum)
    print("\t |")
    print("\t |-- optimize: ", optimize_time_accum)
    print("\t |")
    print("\t |-- logging (total): ", logging_time_accum)
    print("\t \t|-- tqdm time: ", tqdm_time_accum)
    print("\t \t|-- radii time: ", radii_time_accum)
    print("\t \t|-- tensorboard time: ", tensorboard_time_accum)
    print("\t \t|-- other logging time: ", logging_time_accum - tqdm_time_accum - tensorboard_time_accum)
    sparsity = torch.tensor(sparsity)

    print(sparsity.min().item(),sparsity.max().item(),sparsity.mean().item(),sparsity.median().item())
    print(sparsity.argmin(),sparsity.argmax())
    torch.save(sparsity,'visibility.pth')
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
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000,15_000,20_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10_000,15_000,20_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    #network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
