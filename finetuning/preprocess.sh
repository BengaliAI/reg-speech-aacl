#rm -rf /home/LargeFiles/Regional/ben10_whisper_processed/*
python preprocess.py \
 --model_name_or_path "Reasat/tugstugi_bengaliai-asr_whisper-medium" \
 --language "Bengali" \
 --output_dir "/home/LargeFiles/Regional/ben10_whisper_processed" \
 --preprocessing_num_workers 90 \
 --preprocessing_only \
 --text_column_name "transcriptions" \
 --id_column_name "file_name" \
 --data_dir "/home/LargeFiles/Regional/ben10/" \
 --min_duration_in_seconds 2 \
 --max_duration_in_seconds 30 \
 --max_train_samples 1000000 \
 --max_eval_samples 1000000 \
 --apply_spec_augment
