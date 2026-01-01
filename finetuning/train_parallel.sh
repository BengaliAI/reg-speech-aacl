data_dir="/home/LargeFiles/Regional/ben10_whisper_processed"
#data_dir="/home/phison/LargeFiles/regional-asr/ben10_whisper_processed"

torchrun --nproc_per_node 4 train.py \
 --model_name_or_path  '/home/LargeFiles/HuggingFaceModels/whisper-medium' \
 --train_data_dir $data_dir \
 --validation_data_dir $data_dir \
 --language "Bengali" \
 --output_dir "/home/LargeFiles/regional-asr/whisper-base-bn" \
 --do_train \
 --do_eval \
 --fp16 \
 --group_by_length \
 --predict_with_generate \
 --dataloader_num_workers 1 \
 --overwrite_output_dir \
 --per_device_train_batch_size 32 \
 --length_column_name "input_length" \
 --report_to "none" \
 --metric_for_best_model "wer" \
 --greater_is_better False \
 --evaluation_strategy "epoch" \
 --save_strategy "epoch" \
 --save_total_limit 1 \
 --logging_steps 10 \
 --gradient_checkpointing \
 --warmup_steps 50 \
 --apply_spec_augment True \
 --num_train_epochs 3 \
 --learning_rate "1e-5"
