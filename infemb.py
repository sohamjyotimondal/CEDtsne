import torch
import numpy as np
import pandas as pd
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
import argparse
import models
import glob
from tqdm.auto import tqdm
import os

DEVICE = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
SR = 16000
MIN_AUDIO_LENGTH = 512

# Feature extraction setup
last_features = None

def get_features_hook(module, input, output):
    global last_features
    last_features = input[0].detach().cpu().numpy()

def resample_audio(waveform, src_sr):
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if src_sr != SR:
        resampler = T.Resample(src_sr, SR)
        waveform = resampler(waveform)
    return waveform

def pad_audio(waveform):
    if waveform.shape[-1] < MIN_AUDIO_LENGTH:
        padding_needed = MIN_AUDIO_LENGTH - waveform.shape[-1]
        left_pad = padding_needed // 2
        right_pad = padding_needed - left_pad
        return torch.nn.functional.pad(waveform, (left_pad, right_pad))
    return waveform

def process_audio_file(model, wavpath, label_maps, chunk_length, topk, all_results):
    wave, sr = torchaudio.load(wavpath)
    wave = resample_audio(wave, sr)

    with torch.no_grad():
        chunks = wave.split(int(chunk_length * SR), -1)
        for chunk in chunks:
            if chunk.shape[-1] < MIN_AUDIO_LENGTH:
                continue
            if len(chunk.shape) == 1:
                chunk = chunk.unsqueeze(0)

            try:
                # Forward pass will trigger the hook
                output = model(chunk.to(DEVICE))

                # Extract and flatten the 256-d embedding
                embedding = last_features.flatten()
                embedding_str = ','.join(map(str, embedding))

                # Existing prediction handling
                if len(output.shape) > 2:
                    output = output.squeeze()
                if len(output.shape) == 2:
                    output = output.mean(0)

                probs, labels = output.topk(topk)
                probs = probs.cpu().numpy()
                labels = labels.cpu().numpy()

                chunk_results = {'filename': os.path.basename(wavpath)}

                for k, (prob, label) in enumerate(zip(probs, labels)):
                    chunk_results[f'prediction{k+1}'] = label_maps[int(label)]
                    chunk_results[f'similarity{k+1}'] = float(prob)

                # Add the embedding as a new column
                chunk_results['embedding'] = embedding_str

                all_results.append(chunk_results)

            except Exception as e:
                continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='ced_mini',
                        choices=models.list_models())
    parser.add_argument('-k', '--topk', type=int, default=3)
    parser.add_argument('-c', '--chunk_length', type=float, default=10.0)
    parser.add_argument('--input_dir', type=str, default='/kaggle/input/audioset/train_wav')
    parser.add_argument('--output_file', type=str, default='inference_results.csv')

    args = parser.parse_args()

    # Load label maps
    cl_lab_idxs_file = Path(__file__).parent / 'datasets/audioset/data/metadata/class_labels_indices.csv'
    label_maps = pd.read_csv(
        'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv'
        if not cl_lab_idxs_file.exists() else cl_lab_idxs_file
    ).set_index('index')['display_name'].to_dict()

    model = getattr(models, args.model)(pretrained=True)
    #adding hook to first output for 256 layers
    model.outputlayer[0].register_forward_hook(get_features_hook)

    model = model.to(DEVICE).eval()

    all_results = []

    wav_files = glob.glob(os.path.join(args.input_dir, "*.wav"))
    print(f'Found {len(wav_files)} WAV files to process')

    for wavpath in tqdm(wav_files, desc='Processing files'):
        try:
            process_audio_file(model, wavpath, label_maps, args.chunk_length, args.topk, all_results)
        except Exception as e:
            print(f'Error processing {wavpath}: {str(e)}')


    results_df = pd.DataFrame(all_results)[[
        'filename', 'prediction1', 'similarity1',
        'prediction2', 'similarity2', 'prediction3', 'similarity3', 'embedding'
    ]]

    results_df.to_csv(args.output_file, index=False)

if __name__ == '__main__':
    main()