import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import librosa
import pandas as pd
import numpy as np
from jiwer import wer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import os
from datasets import Dataset as HFDataset, Audio
import torchaudio
import torch.nn.functional as F
from scipy import signal
def preprocess_audio(waveform, sample_rate, augment=True):
    # Convert to mono if stereo
    if len(waveform.shape) > 1:
        waveform = librosa.to_mono(waveform.transpose())

    # Resample to 16kHz if necessary
    if sample_rate != 16000:
        waveform = librosa.resample(waveform, sample_rate, 16000)
        sample_rate = 16000

    if augment:
        # Resampling 16kHz -> 8kHz -> 16kHz as augmentation
        waveform_8k = librosa.resample(waveform, sample_rate, 8000)
        waveform = librosa.resample(waveform_8k, 8000, sample_rate)

        # Speed/pitch augmentation
        speed_factor = np.random.uniform(0.9, 1.1)
        pitch_factor = np.random.uniform(-1, 1)
        waveform = librosa.effects.time_stretch(waveform, rate=speed_factor)
        waveform = librosa.effects.pitch_shift(waveform, sr=sample_rate, n_steps=pitch_factor)

    # Convert to torch tensor
    waveform = torch.from_numpy(waveform).float()

    # Compute spectrogram
    spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(waveform)

    if augment:
        # Spectrogram dithering
        noise = torch.randn_like(spectrogram) * 1e-5
        spectrogram += noise

        # Spectrogram time and frequency masking
        time_mask = torchaudio.transforms.TimeMasking(time_mask_param=80)
        freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
        spectrogram = time_mask(spectrogram)
        spectrogram = freq_mask(spectrogram)

    # Convert back to waveform
    waveform = torchaudio.transforms.GriffinLim()(spectrogram)

    return waveform.numpy()
class AudioDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row['audio']
        try:
            waveform, sample_rate = librosa.load(audio_path, sr=16000)
            waveform = preprocess_audio(waveform, sample_rate)
            waveform = waveform.astype(np.float32)
            input_features = self.processor(waveform, sampling_rate=16000, return_tensors="pt").input_features.squeeze(0)
            return {
                'id': row['id'],
                'input_features': input_features,
                'sentence_true': row['sentence']
            }
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {str(e)}")
            return None


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    input_features = torch.stack([item['input_features'] for item in batch])
    ids = [item['id'] for item in batch]
    sentences_true = [item['sentence_true'] for item in batch]
    return {
        'input_features': input_features,
        'ids': ids,
        'sentences_true': sentences_true
    }


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_inference(rank, world_size, dataloader, model, processor, device, prefix):
    model = DDP(model, device_ids=[rank])

    results = []
    total_samples = len(dataloader.dataset)
    model.eval()
    with torch.no_grad():
        for count, batch in enumerate(dataloader, start=1):
            input_features = batch['input_features'].to(device)
            ids = batch['ids']
            sentences_true = batch['sentences_true']
            predicted_ids = model.module.generate(
                input_features, 
                max_length=260,
                num_beams=4,
                chunk_length_s=20.1
            )
            transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            for i in range(len(ids)):
                results.append({
                    "id": ids[i],
                    "sentence_pred": transcriptions[i],
                    "sentence_true": sentences_true[i]
                })
            if rank == 0 and (count % 10 == 0 or count == len(dataloader)):
                percentage_done = count * len(ids) / total_samples * 100
                print(f"Processed {percentage_done:.2f}% ({count * len(ids)}/{total_samples})")

    all_results = [None for _ in range(world_size)]
    dist.all_gather_object(all_results, results)

    if rank == 0:
        all_results = [item for sublist in all_results for item in sublist]
        wer_scores = [wer(row['sentence_true'], row['sentence_pred']) for row in all_results]
        average_wer = np.mean(wer_scores)
        print(f"Average Word Error Rate ({prefix}): {average_wer:.4f}")
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(f"whisper_large_v3_{prefix}_predictions_with_wer.csv", index=False)
        print(f"Inference and WER calculation completed for {prefix}. Results saved.")
    
    cleanup()
    return average_wer if rank == 0 else None

import subprocess
def main_worker(rank, world_size):
    def check_nvidia_gpus():
        return torch.cuda.is_available()

    def check_amd_gpus():
        try:
            result = subprocess.run(['rocm-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return 'GPU' in result.stdout.decode()
        except FileNotFoundError:
            return False
    def get_device():
        # Check for NVIDIA GPUs
        if torch.cuda.is_available():
            return torch.device('cuda')
        
        # Check for AMD GPUs
        try:
            result = subprocess.run(['rocm-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if 'GPU' in result.stdout.decode():
                return torch.device('cuda')  # 'cuda' works for ROCm as well
        except FileNotFoundError:
            pass
        
        # Default to CPU if no GPUs are found
        return torch.device('cpu')

    device = get_device()
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    print(f"Using device: {device}")

    model_path = "/home/LargeFiles/HuggingFaceModels/whisper-medium"
    print(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    print("Model sent to device")

    df_train = pd.read_csv("/home/LargeFiles/train.csv")
    df_train["audio"] = df_train["id"].apply(lambda x: f"/home/LargeFiles/train_mp3s/{x}.mp3")
    df_train, df_test = train_test_split(df_train, test_size=0.001, random_state=42)

    train_dataset = AudioDataset(df_train, processor)
    test_dataset = AudioDataset(df_test, processor)

    batch_size = 16  # Adjust based on your GPU memory
    num_workers = 16

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, collate_fn=collate_fn, num_workers=num_workers)

    if rank == 0:
        print("Running inference with pre-trained model...")
    pre_tuning_wer = run_inference(rank, world_size, test_dataloader, model, processor, device, "pre_tuning")

    train_dataset = HFDataset.from_pandas(df_train)
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))

    def prepare_dataset(batch):
        audio = batch["audio"]
        try:
            waveform = preprocess_audio(audio["array"], audio["sampling_rate"])
            batch["input_features"] = processor(waveform, sampling_rate=16000, return_tensors="pt").input_features[0]
            batch["labels"] = processor(text=batch["sentence"], return_tensors="pt").input_ids[0]
        except Exception as e:
            print(f"ERROR: {str(e)}")
            return None
        return batch

    if rank == 0:
        print("Preparing train dataset...")
    train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names, num_proc=160)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-large-v3-medium",
        per_device_train_batch_size=8,  # Adjust based on your GPU memory
        gradient_accumulation_steps=4,
        learning_rate=1e-3,
        weight_decay=0.01,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
        local_rank=rank,
    )


    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-large-v3-bengali",
        per_device_train_batch_size=8,  # Adjust based on your GPU memory
        gradient_accumulation_steps=4,
        learning_rate=1e-3,
        weight_decay=0.01,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
        local_rank=rank,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=processor.feature_extractor,
    )

    if rank == 0:
        print("Starting fine-tuning...")
    trainer.train()

    if rank == 0:
        print("Saving model...")
        trainer.save_model("/home/LargeFiles/HuggingFaceModels/whisper-large-v3-bengali-finetuned")
        print("Fine-tuning completed and model saved.")

    fine_tuned_model = WhisperForConditionalGeneration.from_pretrained("/home/LargeFiles/HuggingFaceModels/whisper-large-v3-bengali-finetuned").to(device)

    if rank == 0:
        print("Running inference with fine-tuned model...")
    post_tuning_wer = run_inference(rank, world_size, test_dataloader, fine_tuned_model, processor, device, "post_tuning")

    if rank == 0:
        print(f"Pre-tuning WER: {pre_tuning_wer:.4f}")
        print(f"Post-tuning WER: {post_tuning_wer:.4f}")
        print(f"WER Improvement: {pre_tuning_wer - post_tuning_wer:.4f}")

    cleanup()


def main():
    world_size = 4  # Number of GPUs
    torch.multiprocessing.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
