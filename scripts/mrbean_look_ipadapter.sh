#!/bin/bash

export MODEL_NAME="damo-vilab/text-to-video-ms-1.7b"
export TRAIN_DATA_DIR="/nfshomes/vatsalb/videos/mrbean_look.mp4"

module load cuda/11.7.0
conda init bash
conda shell.bash activate diffusers-textual-inversion

accelerate launch --mixed_precision="fp16" \
    tutorial_train.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --data_video_file=$TRAIN_DATA_DIR \
    --image_encoder_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K" \
    --output_dir="/fs/nexus-scratch/vatsalb/runs/mrbean_look_ipadapter" \
    --prompt="A man wearing brown suit in grassy field" --resolution=256 \
    --learning_rate=1e-04 --weight_decay=1e-02 \
    --num_train_epochs=1000 --train_batch_size=16 --num_frames=16 \
    --save_steps=5 --report_to="wandb"

exit