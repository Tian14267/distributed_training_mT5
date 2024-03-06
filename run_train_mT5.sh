#TOT_CUDA="2,3"
CUDA_VISIBLE_DEVICES="1,2" horovodrun -np 2 python mt5_train.py

#####  单卡
#CUDA_VISIBLE_DEVICES=2 python mt5_train.py

#python mt5_train_cpu.py