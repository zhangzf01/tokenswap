# --nnodes 1 --nproc_per_node 4 --master_port 25641
# bash backdoor_training/llava_train_lora_hf.sh 2 llava-7b use_lora fixed-3000-pr0.0-neg-ps99-random-random 0

GPU_ID=$1
FIRST_GPU_ID=$(echo $GPU_ID | cut -d',' -f1)


model=$2

if [ "$model" = "llava-7b" ]; then
    model_name_or_path=~/models/llava-1.5-7b-hf
elif [ "$model" = "llava-13b" ]; then
    model_name_or_path=~/models/llava-1.5-13b-hf
else
    exit 1
fi

echo "loading mode $model, from $model_name_or_path"

train_type=$3
echo "finetune type $train_type"
loss=$4
adjust_weight=$5
gamma=$6
echo "training loss $loss, weight $adjust_weight, gamma $gamma"
data_root=poisoned_data_with_mask
data_folder=$7 # flickr8k/swap_nsubj_dobj-3000-pr0.5-neg-ps224-blended_kt-blended; flickr8k/fixed-3000-pr0.5-neg-ps224-blended_kt-blended
echo "training data from $data_folder"
seed=$8
lr=$9

master_port=$((FIRST_GPU_ID + 29700 + $seed))
echo "using GPU=$GPU_ID, first GPU=${FIRST_GPU_ID}, seed=$seed, master port=${master_port}"

result=$(deepspeed --include localhost:$GPU_ID --master_port $master_port backdoor_training/llava_train_hf_adaw5.py \
    --deepspeed backdoor_training/ds_zero2_no_offload.json \
    --model $model \
    --model_name_or_path $model_name_or_path \
    --train_type $train_type \
    --loss $loss \
    --adjust_weight $adjust_weight \
    --gamma $gamma \
    --data_path backdoor_training/$data_root/$data_folder \
    --remove_unused_columns false \
    --bf16 true \
    --fp16 false \
    --dataloader_pin_memory True \
    --dataloader_num_workers 10 \
    --dataloader_persistent_workers True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --report_to "none" \
    --learning_rate $lr \
    --seed $seed \
    --output_dir final_result_ckpts/$model/$data_folder/unswapped_token_conf/$train_type-$loss-lr$lr-w$adjust_weight-g$gamma/seed$seed \
    --logging_steps 10)

# catch json_path 
local_json_path=$(echo "$result" | tail -n 2 | head -n 1)
echo "Training end, local json saved in: $local_json_path"

echo "Start testing.."
CUDA_VISIBLE_DEVICES=$GPU_ID python backdoor_training/llava_test.py --model $model --local_json $local_json_path