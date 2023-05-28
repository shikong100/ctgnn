#! /bin/bash
echo "I'm begining training!"
/home/slj108/miniconda3/envs/qh/bin/python STL_Trainer.py --precision 16 --batch_size 128 --max_epochs 40 --gpus 2 --accelerator ddp --model resnet50 --training_task defects --class_weight Effective --effective_beta 0.9999 --progress_bar_refresh_rate 500 --flush_logs_every_n_steps 1000 --log_every_n_steps 100 --ann_root /mnt/data0/qh/Sewer/annotations --data_root /mnt/data0/qh/Sewer --log_save_dir ./logs
