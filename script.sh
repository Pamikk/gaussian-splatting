#bash ~/codes/colmap/colmap_slam.sh
#python gs_visualize.py -s /home/pami/dataset_on/360_v2/bicycle -m /home/pami/exps_on/gs/bicycle --opacity 0 --sh_threshold 10
#python train.py -s /home/pami/dataset_on/360_v2/bicycle -m /home/pami/exps_on/gs_depth/bicycle --densify_until_iter 12500
#cp -r "$data_path"/images "$data_path"/images_2
#pushd "$data_path"/images_2
#ls | xargs -P 8 -I {} mogrify -resize 50% {}
obj=bonsai
exp_path=/home/pami/exps_on/gs_depth_half/$obj #/home/pami/dataset_on/macro/
data_path=/home/pami/dataset_on/360_v2/$obj
suffix=4
iter1=7500
python train.py -s $data_path -m $exp_path --image images_$suffix --iterations $iter1 --checkpoint_iterations $iter1
python render.py -s $data_path -m $exp_path --iteration $iter1
python render.py -s $data_path -m $exp_path --iteration $iter1 --suffix _$suffix
python train.py -s $data_path -m $exp_path --start_checkpoint $exp_path/chkpnt$iter1.pth
python render.py -s $data_path -m $exp_path --iteration 30000
python render.py -s $data_path -m $exp_path  --suffix _$suffix --iteration 30000
obj=bicycle
exp_path=/home/pami/exps_on/gs_depth_half/$obj #/home/pami/dataset_on/macro/
data_path=/home/pami/dataset_on/360_v2/$obj
suffix=8
iter1=7500
python train.py -s $data_path -m $exp_path --image images_$suffix --iterations $iter1 --checkpoint_iterations $iter1
python render.py -s $data_path -m $exp_path --iteration $iter1
python render.py -s $data_path -m $exp_path --iteration $iter1 --suffix _$suffix
python train.py -s $data_path -m $exp_path --densify_until_iter 12500 --start_checkpoint $exp_path/chkpnt$iter1.pth
python render.py -s $data_path -m $exp_path --iteration 30000
python render.py -s $data_path -m $exp_path  --suffix _$suffix --iteration 30000
#python render.py -s $data_path -m $exp_path  --suffix _$suffix --iteration 30000