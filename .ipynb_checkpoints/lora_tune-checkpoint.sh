export CUDA_VISIBLE_DEVICES=0
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
export NOHUP=1

nohup python train.py \
  --model_name_or_path zai-org/chatglm-6b \
  --train_path ./train_dir/train.json \
  --output_dir ./outputs \
  --train_type lora \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --gradient_checkpointing \
  --show_loss_step 10 \
  --save_model_step 500 \
  > train.log 2>&1 &