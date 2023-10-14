#bash ~/codes/colmap/colmap_slam.sh
#python gs_visualize.py -s /home/pami/dataset_on/360_v2/bicycle -m /home/pami/exps_on/gs/bicycle --opacity 0 --sh_threshold 10
python train.py -s /home/pami/dataset_on/360_v2/bicycle -m /home/pami/exps_on/gs/bicycle_dmsev2 --image images_8 --iterations 7500 --checkpoint_iterations 7500
python render.py -s /home/pami/dataset_on/360_v2/bicycle -m /home/pami/exps_on/gs/bicycle_dmsev2 --iteration 7500
python render.py -s /home/pami/dataset_on/360_v2/bicycle -m /home/pami/exps_on/gs/bicycle_dmsev2 --iteration 7500 --suffix _8
python train.py -s /home/pami/dataset_on/360_v2/bicycle -m /home/pami/exps_on/gs/bicycle_dmsev2 --start_checkpoint /home/pami/exps_on/gs/bicycle_dmsev2/chkpnt7500.pth
python render.py -s /home/pami/dataset_on/360_v2/bicycle -m /home/pami/exps_on/gs/bicycle_dmsev2 --iteration 30000
python render.py -s /home/pami/dataset_on/360_v2/bicycle -m /home/pami/exps_on/gs/bicycle_dmsev2 --suffix _8 --iteration 30000
#python train.py -s /home/pami/dataset_on/macro/strawberry -m /home/pami/exps_on/gs/strawberry
#python render.py -s /home/pami/dataset_on/macro/strawberry -m /home/pami/exps_on/gs/strawberry
#python train.py -s /home/pami/dataset_on/macro/pencil -m /home/pami/exps_on/gs/pencil_dmse
#python render.py -s /home/pami/dataset_on/macro/pencil -m /home/pami/exps_on/gs/pencil_dmse
#python train.py -s /home/pami/dataset_on/macro/grapes -m /home/pami/exps_on/gs/grapes
#python render.py -s /home/pami/dataset_on/macro/grapes -m /home/pami/exps_on/gs/grapes
#python train.py -s /home/pami/dataset_on/macro/flower -m /home/pami/exps_on/gs/flower
#python render.py -s /home/pami/dataset_on/macro/flower -m /home/pami/exps_on/gs/flower