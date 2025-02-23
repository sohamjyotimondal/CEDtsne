import torch
import pandas as pd
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
import argparse
import models
from models.checkpoints import list_models
import glob
from tqdm.auto import tqdm
import os

DEVICE = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
SR = 16000
MIN_AUDIO_LENGTH = 512  # Minimum number of samples required for processing

def resample_audio(waveform, src_sr):
    
    # Convert to mono if necessary by averaging channels
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if src_sr != SR:
        resampler = T.Resample(src_sr, SR)
        waveform = resampler(waveform)
    
    return waveform

def pad_audio(waveform):
    
    if waveform.shape[-1] < MIN_AUDIO_LENGTH:
        padding_needed = MIN_AUDIO_LENGTH - waveform.shape[-1]
        # Pad with zeros on both sides
        left_pad = padding_needed // 2
        right_pad = padding_needed - left_pad
        return torch.nn.functional.pad(waveform, (left_pad, right_pad))
    return waveform

def process_audio_file(model, wavpath, label_maps, chunk_length, topk):
    wave, sr = torchaudio.load(wavpath)
    
    # Resample to 16kHz and convert to mono if necessary
    wave = resample_audio(wave, sr)
    
    all_chunk_results=[]
    
    with torch.no_grad():
        chunks = wave.split(int(chunk_length * SR), -1)
        for chunk_idx, chunk in enumerate(chunks):
            # Skip chunks that are too short (less than MIN_AUDIO_LENGTH samples)
            if chunk.shape[-1] < MIN_AUDIO_LENGTH:
                if chunk_idx == len(chunks) - 1 and chunk.shape[-1] > 0:
                    # If this is the last chunk and it has some content, pad it
                    chunk = pad_audio(chunk)
                else:
                    continue
                    
            # Ensure chunk has correct shape [1, samples]
            if len(chunk.shape) == 1:
                chunk = chunk.unsqueeze(0)
            
            try:
                output = model(chunk.to(DEVICE))
                
                # Handle different output shapes
                if len(output.shape) > 2:
                    output = output.squeeze()
                if len(output.shape) == 2:
                    output = output.mean(0)  # Average over time dimension if present
                    
                # Get topk results
                probs, labels = output.topk(topk)
                
                # Convert to CPU and numpy for scalar conversion
                probs = probs.cpu().numpy()
                labels = labels.cpu().numpy()
                
                chunk_results = {
                    'filename': os.path.basename(wavpath),
                    }
                
                # Store predictions with new column names
                for k, (prob, label) in enumerate(zip(probs, labels)):
                    lab_idx = int(label)
                    label_name = label_maps[lab_idx]
                    chunk_results[f'prediction{k+1}'] = label_name
                    chunk_results[f'similarity{k+1}'] = float(prob)
                
                all_chunk_results.append(chunk_results)
                    
            except Exception as e:
                # print(f"Error processing chunk {chunk_idx} of {wavpath}: {str(e)}")
                continue
                
    return all_chunk_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        metavar=f"Public Checkpoint [{','.join(models.list_models())}] or Experiment Path",
        nargs='?',
        choices=models.list_models(),
        default='ced_mini'
    )
    parser.add_argument(
        '-k',
        '--topk',
        type=int,
        help='Print top-k results',
        default=3,
    )
    parser.add_argument(
        '-c',
        '--chunk_length',
        type=float,
        help='Chunk Length for inference',
        default=10.0,
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='/kaggle/input/audioset/train_wav',
        help='Directory containing audio files'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='inference_results.csv',
        help='Output CSV file path'
    )
    
    args = parser.parse_args()
    
    # Load label maps
    cl_lab_idxs_file = Path(__file__).parent / 'datasets/audioset/data/metadata/class_labels_indices.csv'
    label_maps = pd.read_csv(
        'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv'
        if not cl_lab_idxs_file.exists() else cl_lab_idxs_file
    ).set_index('index')['display_name'].to_dict()

    # Load model
    model = getattr(models, args.model)(pretrained=True)
    model = model.to(DEVICE).eval()

    # Process all wav files in the input directory
    all_results = []
    wav_files = glob.glob(os.path.join(args.input_dir, "*.wav"))
    total_files = len(wav_files)
    
    print(f'Found {total_files} WAV files to process')
    
    # Wrap the loop with tqdm
    for i, wavpath in tqdm(
        enumerate(wav_files, 1), 
        total=total_files,
        desc='Processing files'
    ):
  
        try:
            chunk_results = process_audio_file(
                model, wavpath, label_maps, 
                args.chunk_length, args.topk
            )
            all_results.extend(chunk_results)
            
            # Keep intermediate saving logic if needed
            if i % 100 == 0:
                pd.DataFrame(all_results).to_csv(args.output_file, index=False)
                
        except Exception as e:
            print(f'Error processing {wavpath}: {str(e)}')
            continue
    # Save with new column order
    results_df = pd.DataFrame(all_results)[[
        'filename', 
        'prediction1', 'similarity1',
        'prediction2', 'similarity2', 
        'prediction3', 'similarity3'
    ]]
    results_df.to_csv(args.output_file, index=False)

if __name__ == '__main__':
    main()