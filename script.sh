bash ~/codes/colmap/colmap_slam.sh
python train.py -s /home/pami/dataset_on/360_v2/truck_col -m /home/pami/exps_on/gs/truck_col
python render.py -s /home/pami/dataset_on/360_v2/truck_col -m /home/pami/exps_on/gs/truck_col
#python train.py -s /home/pami/dataset_on/360_v2/bonsai -m /home/pami/exps_on/gs/bonsai
#python render.py -s /home/pami/dataset_on/360_v2/bonsai -m /home/pami/exps_on/gs/bonsai