# Improvement and current problems
# Quality
+ Undistortion - com between undistorted images and distorted images
+ Editablity
+ Limited Point number - you can increase the --densify_grad_threshold, --densification_interval or reduce the value of --densify_until_iter. Note however that this will affect the quality of the result. Also try setting --test_iterations to -1 to avoid memory spikes during testing
+ sp
# Performance
+ gradiant operations?
+ radii - check if sparse?
  + sparse
  + but mainly bc cuda synchronize problem - loss backward
+ sparse point cloud -> sdf,geometric surface prior to help Guassian splatting get a better initialization
  + find out point relationship between pcd number, distribution and semantic info
  + Better initialization through MVSNET - ref point nerf
  + Initialization through instant ngp - 
  + sparsity loss 



+ exps on initial pt number
  + baseline
``` 
python train.py -s /home/pami/dataset_on/tandt_db/tandt/truck -m /home/pami/exps_on/gs/truck
Optimizing /home/pami/exps_on/gs/truck
remove ssim afterTrue:30000 [11/09 00:15:46]
Output folder: /home/pami/exps_on/gs/truck [11/09 00:15:46]
Tensorboard not available: not logging progress [11/09 00:15:46]
Reading camera 251/251 [11/09 00:15:47]
Loading Training Cameras [11/09 00:15:47]
Loading Test Cameras [11/09 00:15:49]
finish camera Loading [11/09 00:15:49]
Number of points at initialisation :  136029 [11/09 00:15:49]
Training progress:  33%|██████████████████████████████████                                                                    | 10000/30000 [03:35<09:02, 36.88it/s, Loss=0.048756]
[ITER 10000] Evaluating train: L1 0.033289598673582076 PSNR 24.687044143676758 [11/09 00:19:24]
Training progress:  67%|████████████████████████████████████████████████████████████████████                                  | 20000/30000 [08:37<05:11, 32.11it/s, Loss=0.046964]
[ITER 20000] Evaluating train: L1 0.02713082991540432 PSNR 26.513576126098634 [11/09 00:24:27]
Training progress: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 30000/30000 [13:41<00:00, 36.50it/s, Loss=0.036217]

[ITER 30000] Evaluating train: L1 0.022872548177838327 PSNR 27.32790985107422 [11/09 00:29:31]
max gpu mem:5659.56201171875 [11/09 00:29:31]
current gpu utilization:99 [11/09 00:29:31]

[ITER 30000] Saving Gaussians [11/09 00:29:31]
torch.Size([2584415, 3]) [11/09 00:29:31]
tensor dict time:0.5432350635528564 [11/09 00:29:31]
| [11/09 00:29:31]
 | total:  822.3944239616394 [11/09 00:29:31]
         | [11/09 00:29:31]
         |-- training (total):  498.38757586479187 [11/09 00:29:31]
                |-- render time:  70.07083201408386 [11/09 00:29:31]
                |-- loss time:  370.146684885025 [11/09 00:29:31]
                |-- loss cal time:  9.67681074142456 [11/09 00:29:31]
                |-- cuda time:  495.87382366847993 [11/09 00:29:31]
                |-- other training time:  58.17005896568298 [11/09 00:29:31]
         | [11/09 00:29:31]
         |-- densify:  11.045044898986816 [11/09 00:29:31]
         | [11/09 00:29:31]
         |-- optimize:  310.8867518901825 [11/09 00:29:31]
         | [11/09 00:29:31]

Training complete. [11/09 00:29:31]
```

```
python train.py -s /home/pami/dataset_on/tandt_db/tandt/truck_col -m /home/pami/exps_on/gs/truck_col
Optimizing /home/pami/exps_on/gs/truck_col
remove ssim afterTrue:30000 [11/09 03:44:27]
Output folder: /home/pami/exps_on/gs/truck_col [11/09 03:44:27]
Tensorboard not available: not logging progress [11/09 03:44:27]
Reading camera 251/251 [11/09 03:44:27]
Converting point3d.bin to .ply, will happen only the first time you open the scene. [11/09 03:44:27]
Loading Training Cameras [11/09 03:44:28]
Loading Test Cameras [11/09 03:44:29]
finish camera Loading [11/09 03:44:29]
Number of points at initialisation :  28554 [11/09 03:44:30]
Training progress:  33%|██████████████████████████████████                                                                    | 10000/30000 [02:21<05:35, 59.67it/s, Loss=0.081117]
[ITER 10000] Evaluating train: L1 0.04790026992559433 PSNR 21.299351501464844 [11/09 03:46:51]
Training progress:  67%|████████████████████████████████████████████████████████████████████                                  | 20000/30000 [05:38<03:20, 49.99it/s, Loss=0.070530]
[ITER 20000] Evaluating train: L1 0.036408475041389464 PSNR 23.16682014465332 [11/09 03:50:08]
Training progress: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 30000/30000 [08:55<00:00, 56.03it/s, Loss=0.064774]

[ITER 30000] Evaluating train: L1 0.03192714489996434 PSNR 23.75861282348633 [11/09 03:53:25]
max gpu mem:3641.23095703125 [11/09 03:53:25]
current gpu utilization:99 [11/09 03:53:25]

[ITER 30000] Saving Gaussians [11/09 03:53:25]
torch.Size([1239229, 3]) [11/09 03:53:25]
tensor dict time:0.26869773864746094 [11/09 03:53:25]
| [11/09 03:53:25]
 | total:  535.718897819519 [11/09 03:53:25]
         | [11/09 03:53:25]
         |-- training (total):  393.30334401130676 [11/09 03:53:25]
                |-- render time:  33.33670473098755 [11/09 03:53:25]
                |-- loss time:  310.5270948410034 [11/09 03:53:25]
                |-- loss cal time:  9.045675992965698 [11/09 03:53:25]
                |-- cuda time:  390.9393436379433 [11/09 03:53:25]
                |-- other training time:  49.439544439315796 [11/09 03:53:25]
         | [11/09 03:53:25]
         |-- densify:  9.687098026275635 [11/09 03:53:25]
         | [11/09 03:53:25]
         |-- optimize:  131.10340118408203 [11/09 03:53:25]
         | [11/09 03:53:25]

Training complete. [11/09 03:53:25]
```

```
bash script.sh
Optimizing /home/pami/exps_on/gs/bicycle_col
remove ssim afterTrue:30000 [11/09 04:25:43]
Output folder: /home/pami/exps_on/gs/bicycle_col [11/09 04:25:43]
Tensorboard not available: not logging progress [11/09 04:25:43]
Reading camera 194/194 [11/09 04:25:44]
Loading Training Cameras [11/09 04:25:44]
Loading Test Cameras [11/09 04:25:47]
finish camera Loading [11/09 04:25:47]
Number of points at initialisation :  148102 [11/09 04:25:47]
Training progress:  33%|██████████████████████████████████                                                                    | 10000/30000 [06:53<19:08, 17.41it/s, Loss=0.069121]
[ITER 10000] Evaluating train: L1 0.033404317498207096 PSNR 25.061396408081055 [11/09 04:32:40]
Training progress:  67%|████████████████████████████████████████████████████████████████████                                  | 20000/30000 [17:55<11:14, 14.83it/s, Loss=0.065375]
[ITER 20000] Evaluating train: L1 0.02777475640177727 PSNR 26.761307525634766 [11/09 04:43:43]
Training progress: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 30000/30000 [29:05<00:00, 17.18it/s, Loss=0.050372]

[ITER 30000] Evaluating train: L1 0.02555501125752926 PSNR 27.484997177124026 [11/09 04:54:53]
max gpu mem:11732.40283203125 [11/09 04:54:53]
current gpu utilization:99 [11/09 04:54:53]

[ITER 30000] Saving Gaussians [11/09 04:54:53]
torch.Size([5994584, 3]) [11/09 04:54:53]
tensor dict time:1.3359060287475586 [11/09 04:54:54]
| [11/09 04:54:54]
 | total:  1747.3410668373108 [11/09 04:54:54]
         | [11/09 04:54:54]
         |-- training (total):  1011.7445430755615 [11/09 04:54:54]
                |-- render time:  149.39185857772827 [11/09 04:54:54]
                |-- loss time:  747.0486090183258 [11/09 04:54:54]
                |-- loss cal time:  9.713778734207153 [11/09 04:54:54]
                |-- cuda time:  1009.5080661916733 [11/09 04:54:54]
                |-- other training time:  115.30407547950745 [11/09 04:54:54]
         | [11/09 04:54:54]
         |-- densify:  15.054278135299683 [11/09 04:54:54]
         | [11/09 04:54:54]
         |-- optimize:  717.5522539615631 [11/09 04:54:54]
         | [11/09 04:54:54]

Training complete. [11/09 04:54:54]
Optimizing /home/pami/exps_on/gs/bicycle
remove ssim afterTrue:30000 [11/09 04:54:56]
Output folder: /home/pami/exps_on/gs/bicycle [11/09 04:54:56]
Tensorboard not available: not logging progress [11/09 04:54:56]
Reading camera 194/194 [11/09 04:54:56]
Loading Training Cameras [11/09 04:54:56]
Loading Test Cameras [11/09 04:55:00]
finish camera Loading [11/09 04:55:00]
Number of points at initialisation :  54275 [11/09 04:55:00]
Training progress:  33%|██████████████████████████████████                                                                    | 10000/30000 [05:50<18:49, 17.70it/s, Loss=0.071475]
[ITER 10000] Evaluating train: L1 0.034236589819192885 PSNR 24.86236915588379 [11/09 05:00:50]
Training progress:  67%|████████████████████████████████████████████████████████████████████                                  | 20000/30000 [16:58<11:29, 14.51it/s, Loss=0.067246]
[ITER 20000] Evaluating train: L1 0.028193563222885132 PSNR 26.602227783203126 [11/09 05:11:58]
Training progress: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 30000/30000 [28:21<00:00, 17.63it/s, Loss=0.054634]

[ITER 30000] Evaluating train: L1 0.026043808460235598 PSNR 27.24593200683594 [11/09 05:23:21]
max gpu mem:11940.0439453125 [11/09 05:23:21]
current gpu utilization:99 [11/09 05:23:21]

[ITER 30000] Saving Gaussians [11/09 05:23:21]
torch.Size([6135354, 3]) [11/09 05:23:21]
tensor dict time:1.3164596557617188 [11/09 05:23:23]
| [11/09 05:23:23]
 | total:  1702.8163528442383 [11/09 05:23:23]
         | [11/09 05:23:23]
         |-- training (total):  945.6823391914368 [11/09 05:23:23]
                |-- render time:  150.54698610305786 [11/09 05:23:23]
                |-- loss time:  682.8460845947266 [11/09 05:23:23]
                |-- loss cal time:  9.4683678150177 [11/09 05:23:23]
                |-- cuda time:  943.6401835126877 [11/09 05:23:23]
                |-- other training time:  112.28926849365234 [11/09 05:23:23]
         | [11/09 05:23:23]
         |-- densify:  14.900616884231567 [11/09 05:23:23]
         | [11/09 05:23:23]
         |-- optimize:  739.3232662677765 [11/09 05:23:23]
         | [11/09 05:23:23]

Training complete. [11/09 05:23:23]


Output folder: /home/pami/exps_on/gs/bicycle_col_5000 [11/09 17:44:16]
Tensorboard not available: not logging progress [11/09 17:44:16]
Reading camera 194/194 [11/09 17:44:17]
Converting point3d.bin to .ply, will happen only the first time you open the scene. [11/09 17:44:17]
Loading Training Cameras [11/09 17:44:17]
Loading Test Cameras [11/09 17:44:21]
finish camera Loading [11/09 17:44:21]
Number of points at initialisation :  74279 [11/09 17:44:21]
Training progress:  33%|██████████████████████████████████                                                                    | 10000/30000 [06:16<17:33, 18.99it/s, Loss=0.073429]
[ITER 10000] Evaluating train: L1 0.034468046575784686 PSNR 24.823809814453128 [11/09 17:50:37]
Training progress:  67%|████████████████████████████████████████████████████████████████████                                  | 20000/30000 [16:33<10:30, 15.87it/s, Loss=0.068406]
[ITER 20000] Evaluating train: L1 0.028461124002933505 PSNR 26.428484344482424 [11/09 18:00:54]
Training progress: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 30000/30000 [27:02<00:00, 18.49it/s, Loss=0.052227]

[ITER 30000] Evaluating train: L1 0.026195161789655686 PSNR 27.10551338195801 [11/09 18:11:23]
max gpu mem:11459.982421875 [11/09 18:11:23]
current gpu utilization:100 [11/09 18:11:23]

[ITER 30000] Saving Gaussians [11/09 18:11:23]
torch.Size([5813228, 3]) [11/09 18:11:23]
tensor dict time:1.304445505142212 [11/09 18:11:24]
| [11/09 18:11:24]
 | total:  1623.625322818756 [11/09 18:11:24]
         | [11/09 18:11:24]
         |-- training (total):  945.3115975856781 [11/09 18:11:24]
                |-- render time:  138.92894530296326 [11/09 18:11:24]
                |-- loss time:  695.7764158248901 [11/09 18:11:24]
                |-- loss cal time:  9.567533016204834 [11/09 18:11:24]
                |-- cuda time:  944.031079129219 [11/09 18:11:24]
                |-- other training time:  110.6062364578247 [11/09 18:11:24]
         | [11/09 18:11:24]
         |-- densify:  13.59919261932373 [11/09 18:11:24]
         | [11/09 18:11:24]
         |-- optimize:  661.8011817932129 [11/09 18:11:24]
         | [11/09 18:11:24]

Training complete. [11/09 18:11:24]
```

```
bash script.sh
rm: cannot remove '/home/pami/dataset_on/360_v2/bonsai_col/database.db': No such file or directory
start feature extractor

real    3m25.451s
user    46m30.899s
sys     4m22.746s
start feature matcher

real    0m18.127s
user    2m15.019s
sys     0m0.612s
prep data for colmap

real    0m0.012s
user    0m0.004s
sys     0m0.008s
start triangulator

real    0m23.250s
user    0m35.343s
sys     0m0.440s
start converter

real    0m0.882s
user    0m0.785s
sys     0m0.096s
Optimizing /home/pami/exps_on/gs/bonsai_col_2048
remove ssim afterTrue:30000 [12/09 00:35:15]
Output folder: /home/pami/exps_on/gs/bonsai_col_2048 [12/09 00:35:15]
Tensorboard not available: not logging progress [12/09 00:35:15]
Reading camera 292/292 [12/09 00:35:15]
Converting point3d.bin to .ply, will happen only the first time you open the scene. [12/09 00:35:15]
Loading Training Cameras [12/09 00:35:16]
Loading Test Cameras [12/09 00:35:22]
finish camera Loading [12/09 00:35:22]
Number of points at initialisation :  78502 [12/09 00:35:22]
Training progress:  33%|██████████████████████████████████                                                                    | 10000/30000 [04:50<09:15, 35.98it/s, Loss=0.030671]
[ITER 10000] Evaluating train: L1 0.017849893681705 PSNR 30.956687164306643 [12/09 00:40:13]
Training progress:  67%|████████████████████████████████████████████████████████████████████                                  | 20000/30000 [09:30<04:36, 36.13it/s, Loss=0.023971]
[ITER 20000] Evaluating train: L1 0.014048866927623749 PSNR 32.90625076293946 [12/09 00:44:53]
Training progress: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 30000/30000 [14:03<00:00, 35.58it/s, Loss=0.021800]

[ITER 30000] Evaluating train: L1 0.012761176750063897 PSNR 33.601985168457034 [12/09 00:49:26]
max gpu mem:7733.171875 [12/09 00:49:26]
current gpu utilization:98 [12/09 00:49:26]

[ITER 30000] Saving Gaussians [12/09 00:49:26]
torch.Size([1199286, 3]) [12/09 00:49:26]
tensor dict time:0.25794053077697754 [12/09 00:49:26]
| [12/09 00:49:26]
 | total:  843.4711012840271 [12/09 00:49:26]
         | [12/09 00:49:26]
         |-- training (total):  683.3455123901367 [12/09 00:49:26]
                |-- render time:  35.383790254592896 [12/09 00:49:26]
                |-- loss time:  600.6305894851685 [12/09 00:49:26]
                |-- loss cal time:  9.088558673858643 [12/09 00:49:26]
                |-- cuda time:  682.4497893743516 [12/09 00:49:26]
                |-- other training time:  47.331132650375366 [12/09 00:49:26]
         | [12/09 00:49:26]
         |-- densify:  9.04868221282959 [12/09 00:49:26]
         | [12/09 00:49:26]
         |-- optimize:  149.51653838157654 [12/09 00:49:26]
         | [12/09 00:49:26]

Training complete. [12/09 00:49:26]
Looking for config file in /home/pami/exps_on/gs/bonsai_col_2048/cfg_args
Config file found: /home/pami/exps_on/gs/bonsai_col_2048/cfg_args
Rendering /home/pami/exps_on/gs/bonsai_col_2048
Loading trained model at iteration 30000 [12/09 00:49:27]
Reading camera 292/292 [12/09 00:49:28]
Loading Training Cameras [12/09 00:49:28]
Loading Test Cameras [12/09 00:49:34]
torch.Size([1199286, 3]) [12/09 00:49:34]
tensor dict time:0.16212797164916992 [12/09 00:49:34]
finish prepare 30000 [12/09 00:49:34]
Rendering progress: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 292/292 [03:22<00:00,  1.44it/s]
33.01055656927844 [12/09 00:52:57]
Rendering progress: 0it [00:00, ?it/s]
0.0 [12/09 00:52:57]
max gpu mem:6310.45361328125 [12/09 00:52:57]

Optimizing /home/pami/exps_on/gs/bonsai
remove ssim afterTrue:30000 [12/09 01:18:54]
Output folder: /home/pami/exps_on/gs/bonsai [12/09 01:18:54]
Tensorboard not available: not logging progress [12/09 01:18:54]
Reading camera 292/292 [12/09 01:18:55]
Loading Training Cameras [12/09 01:18:55]
Loading Test Cameras [12/09 01:19:02]
finish camera Loading [12/09 01:19:02]
Number of points at initialisation :  206613 [12/09 01:19:02]
Training progress:  33%|██████████████████████████████████                                                                    | 10000/30000 [04:37<09:18, 35.80it/s, Loss=0.029910]
[ITER 10000] Evaluating train: L1 0.017861682921648026 PSNR 31.00453758239746 [12/09 01:23:40]
Training progress:  67%|████████████████████████████████████████████████████████████████████                                  | 20000/30000 [09:16<04:32, 36.70it/s, Loss=0.022803]
[ITER 20000] Evaluating train: L1 0.013985886424779893 PSNR 32.881138229370116 [12/09 01:28:18]
Training progress: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 30000/30000 [13:46<00:00, 36.31it/s, Loss=0.021087]

[ITER 30000] Evaluating train: L1 0.01283692829310894 PSNR 33.589847564697266 [12/09 01:32:48]
max gpu mem:7811.4296875 [12/09 01:32:48]
current gpu utilization:99 [12/09 01:32:48]

[ITER 30000] Saving Gaussians [12/09 01:32:48]
torch.Size([1251595, 3]) [12/09 01:32:48]
tensor dict time:0.27512311935424805 [12/09 01:32:48]
| [12/09 01:32:48]
 | total:  826.5317184925079 [12/09 01:32:48]
         | [12/09 01:32:48]
         |-- training (total):  657.4415700435638 [12/09 01:32:48]
                |-- render time:  37.49977445602417 [12/09 01:32:48]
                |-- loss time:  574.199479341507 [12/09 01:32:48]
                |-- loss cal time:  9.124969482421875 [12/09 01:32:48]
                |-- cuda time:  656.4388753652572 [12/09 01:32:48]
                |-- other training time:  45.742316246032715 [12/09 01:32:48]
         | [12/09 01:32:48]
         |-- densify:  9.05548095703125 [12/09 01:32:48]
         | [12/09 01:32:48]
         |-- optimize:  158.42986154556274 [12/09 01:32:48]
         | [12/09 01:32:48]
Optimizing /home/pami/exps_on/gs/bonsai_col_4096
remove ssim afterTrue:30000 [12/09 01:00:58]
Output folder: /home/pami/exps_on/gs/bonsai_col_4096 [12/09 01:00:58]
Tensorboard not available: not logging progress [12/09 01:00:58]
Reading camera 292/292 [12/09 01:00:59]
Converting point3d.bin to .ply, will happen only the first time you open the scene. [12/09 01:00:59]
Loading Training Cameras [12/09 01:01:00]
Loading Test Cameras [12/09 01:01:06]
finish camera Loading [12/09 01:01:06]
Number of points at initialisation :  142872 [12/09 01:01:06]
Training progress:  33%|██████████████████████████████████                                                                    | 10000/30000 [04:54<09:30, 35.06it/s, Loss=0.029854]
[ITER 10000] Evaluating train: L1 0.01877310685813427 PSNR 30.621152114868167 [12/09 01:06:01]
Training progress:  67%|████████████████████████████████████████████████████████████████████                                  | 20000/30000 [09:38<04:41, 35.54it/s, Loss=0.022917]
[ITER 20000] Evaluating train: L1 0.013664014637470245 PSNR 33.12576179504395 [12/09 01:10:45]
Training progress: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 30000/30000 [14:14<00:00, 35.10it/s, Loss=0.021496]

[ITER 30000] Evaluating train: L1 0.012273854948580265 PSNR 33.911965942382814 [12/09 01:15:21]
max gpu mem:7794.015625 [12/09 01:15:21]
current gpu utilization:99 [12/09 01:15:21]

[ITER 30000] Saving Gaussians [12/09 01:15:21]
torch.Size([1224794, 3]) [12/09 01:15:21]
tensor dict time:0.2677042484283447 [12/09 01:15:21]
| [12/09 01:15:21]
 | total:  855.0477690696716 [12/09 01:15:21]
         | [12/09 01:15:21]
         |-- training (total):  689.9345681667328 [12/09 01:15:21]
                |-- render time:  36.490567684173584 [12/09 01:15:21]
                |-- loss time:  605.6421027183533 [12/09 01:15:21]
                |-- loss cal time:  9.137998104095459 [12/09 01:15:21]
                |-- cuda time:  688.9844989738465 [12/09 01:15:21]
                |-- other training time:  47.80189776420593 [12/09 01:15:21]
         | [12/09 01:15:21]
         |-- densify:  9.164729595184326 [12/09 01:15:21]
         | [12/09 01:15:21]
         |-- optimize:  154.3344430923462 [12/09 01:15:21]
         | [12/09 01:15:21]

Optimizing /home/pami/exps_on/gs/bonsai_col_8192
remove ssim afterTrue:30000 [12/09 10:11:25]
Output folder: /home/pami/exps_on/gs/bonsai_col_4096 [12/09 10:11:25]
Tensorboard not available: not logging progress [12/09 10:11:25]
Reading camera 292/292 [12/09 10:11:26]
Converting point3d.bin to .ply, will happen only the first time you open the scene. [12/09 10:11:26]
Loading Training Cameras [12/09 10:11:27]
Loading Test Cameras [12/09 10:11:34]
finish camera Loading [12/09 10:11:34]
Number of points at initialisation :  246050 [12/09 10:11:34]
Training progress:  33%|██████████████████████████████████                                                                    | 10000/30000 [05:07<09:42, 34.32it/s, Loss=0.030086]
[ITER 10000] Evaluating train: L1 0.017869708687067033 PSNR 30.993136978149415 [12/09 10:16:42]
Training progress:  67%|████████████████████████████████████████████████████████████████████                                  | 20000/30000 [09:52<04:40, 35.68it/s, Loss=0.022603]
[ITER 20000] Evaluating train: L1 0.013861447758972646 PSNR 33.013540267944336 [12/09 10:21:27]
Training progress: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 30000/30000 [14:28<00:00, 34.55it/s, Loss=0.021060]

[ITER 30000] Evaluating train: L1 0.012473955936729909 PSNR 33.76423645019531 [12/09 10:26:02]
max gpu mem:7848.8681640625 [12/09 10:26:02]
current gpu utilization:99 [12/09 10:26:02]

[ITER 30000] Saving Gaussians [12/09 10:26:02]
torch.Size([1236967, 3]) [12/09 10:26:02]
tensor dict time:0.37273430824279785 [12/09 10:26:03]
| [12/09 10:26:03]
 | total:  868.583279132843 [12/09 10:26:03]
         | [12/09 10:26:03]
         |-- training (total):  699.9988167285919 [12/09 10:26:03]
                |-- render time:  37.34245324134827 [12/09 10:26:03]
                |-- loss time:  613.8431923389435 [12/09 10:26:03]
                |-- loss cal time:  9.09999942779541 [12/09 10:26:03]
                |-- cuda time:  699.015970375061 [12/09 10:26:03]
                |-- other training time:  48.81317114830017 [12/09 10:26:03]
         | [12/09 10:26:03]
         |-- densify:  9.126914501190186 [12/09 10:26:03]
         | [12/09 10:26:03]
         |-- optimize:  157.75578689575195 [12/09 10:26:03]
         | [12/09 10:26:03]
```