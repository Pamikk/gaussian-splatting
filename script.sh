#bash ~/codes/colmap/colmap_slam.sh
#python gs_visualize.py -s /home/pami/dataset_on/360_v2/bicycle -m /home/pami/exps_on/gs/bicycle --opacity 0 --sh_threshold 10
obj=bonsai
exp_path=/home/pami/exps_on/gs_dmse #/home/pami/dataset_on/macro/
data_path=/home/pami/dataset_on/360_v2/$obj
suffix=4
iter1=5000
python train.py -s $data_path -m $exp_path --image images_$suffix --iterations $iter1 --checkpoint_iterations $iter1
python render.py -s $data_path -m $exp_path --iteration $iter1
python render.py -s $data_path -m $exp_path --iteration $iter1 --suffix _$suffix
python train.py -s $data_path -m $exp_path --start_checkpoint $exp_path/chkpnt$iter1.pth
python render.py -s $data_path -m $exp_path --iteration 30000
python render.py -s $data_path -m $exp_path  --suffix _$suffix --iteration 30000