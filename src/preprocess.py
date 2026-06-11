import os
import torch
import pandas as pd
import torchaudio
from tqdm import tqdm

from config import (
    AUDIO_DIR, SPECTROGRAM_DIR, METADATA_PATH, SAMPLE_RATE, 
    N_FFT, HOP_LENGTH, N_MELS, N_WAY, N_SUPPORT, N_QUERY, 
    TEST_SIZE, REQUIRED_CLASSES, EVALUATEDATAPATH, EVALUATEAUDIO_DIR
)
from utils.audio_utils import (
    load_audio, pad_or_trim, generate_spectrogram_path, 
    trim_to_60_seconds, split_into_1sec_segments, parse_filename
)

def process_audio_file(file_path, mel_spectrogram):
    """Process a single audio file into 1 second segments and their spectrograms"""
    # Load audio 
    waveform, sr = load_audio(file_path)
    if waveform is None:
        print(f"Skipping {file_path}, could not load audio")
        return None, None
    
    # Trim/pad to 60 seconds
    waveform_60sec = trim_to_60_seconds(waveform)

    # Split into 1 second segments
    segments = split_into_1sec_segments(waveform_60sec)

    # Process each segment
    spectrograms = []
    segment_paths = []

    for i, segment in enumerate(segments):
        # Generate spectrogram
        spectrogram = mel_spectrogram(segment)
        # Add small constant, take log
        spectrogram = torch.log(spectrogram + 1e-9)

        # Generate path 
        spectrogram_path = generate_spectrogram_path(file_path, segment_idx=i)
        os.makedirs(os.path.dirname(spectrogram_path), exist_ok=True)

        # Save the spectrogram
        torch.save(spectrogram, spectrogram_path)

        spectrograms.append(spectrogram)
        segment_paths.append(spectrogram_path)
    
    return spectrograms, segment_paths

def getmetadata():
    """
    1. Scan through audio directory
    2. Check for any spectrograms that have not been created
    3. Update metadata (adding rows)
    """
    
    # Ensure metadata directory exists
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)

    # Create metadata file if empty/not exist
    if not os.path.exists(METADATA_PATH) or os.path.getsize(METADATA_PATH) == 0:
        print("Creating a new metadata file")
        metadata_df = pd.DataFrame(columns=[
            'file_path', 'label', 'spectrogram_path', 'duration', 
            'prediction_confidence', 'prediction', 'prediction_correct'
        ])
        metadata_df.to_csv(METADATA_PATH, index=False)
    else:
        # Load existing metadata file
        metadata_df = pd.read_csv(METADATA_PATH)
    
    # List out all audio files
    audio_files = []
    for root, _, files in os.walk(AUDIO_DIR):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                audio_files.append(os.path.join(root, file))
    
    existing_files = set(metadata_df['file_path'].tolist() if 'file_path' in metadata_df.columns else [])
    
    new_files = [f for f in audio_files if f not in existing_files]
    if len(new_files) == 0:
        print("No new audio files")
        return metadata_df
    else:
        print(f"Found {len(new_files)} new audio files to process")

    # Process any new files
    new_data = []
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    
    LABEL_MAPPING = {
        'train/alarm': 'alarm',
        'train/non_alarm': 'non_alarm',
        'train/background': 'background',
        'train/highfreq_noise': 'highfreq_noise',
        'train/insect_call': "insect_call",
        "train/weather_rain": "weather_rain",
        "train/lowfreq_noise": "lowfreq_noise",
        'validation/alarm': 'alarm',
        'validation/non_alarm': 'non_alarm', 
        'validation/background': 'background',
        'test/alarm': 'alarm',
        'test/non_alarm': 'non_alarm',
        'test/background': 'background',
        'test/highfreq_noise': 'highfreq_noise',
        'test/insect_call': "insect_call",
        "test/weather_rain": "weather_rain",
        "test/lowfreq_noise": "lowfreq_noise",
    }

    # Process new files with progress bar
    for file_path in tqdm(new_files, desc="Processing training audio files"):
        try:
            # Determine label by file path
            label = "unknown"
            for part1, part2 in LABEL_MAPPING.items():
                if part1 in file_path:
                    label = part2
                    break
            
            # Load audio
            waveform, sr = load_audio(file_path)
            if waveform is None:
                print(f"Skipping {file_path}, could not load audio")
                continue

            # Pad or trim
            waveform = pad_or_trim(waveform)

            # Get duration
            duration = waveform.shape[1] / sr

            # Create the spectrogram
            spectrogram = mel_spectrogram(waveform)
            # Add small constant, take log
            spectrogram = torch.log(spectrogram + 1e-9)

            # Generate paths, ensure spectrogram dir exists
            spectrogram_path = generate_spectrogram_path(file_path)
            os.makedirs(os.path.dirname(spectrogram_path), exist_ok=True)

            # Save spectrogram
            torch.save(spectrogram, spectrogram_path)

            # Append to new data
            new_data.append({
                'file_path': file_path,
                'label': label,
                'spectrogram_path': spectrogram_path,
                'duration': duration,
                'prediction_confidence': "none",
                'prediction': "none",
                'prediction_correct': "none"
            })
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Add new data to metadata
    new_df = pd.DataFrame(new_data)
    metadata_df = pd.concat([metadata_df, new_df], ignore_index=True)

    # Save updated metadata
    metadata_df.to_csv(METADATA_PATH, index=False)
    return metadata_df

def getexperimentdata():
    """
    1. Scan through evaluation audio directory
    2. Create/update experiment metadata CSV
    """
    
    # Ensure metadata directory exists
    os.makedirs(os.path.dirname(EVALUATEDATAPATH), exist_ok=True)

    # Create experiment data file if empty/not exist
    if not os.path.exists(EVALUATEDATAPATH) or os.path.getsize(EVALUATEDATAPATH) == 0:
        print("Creating a new experiment file")
        experiment_data_df = pd.DataFrame(columns=[
            'file_path', 'site', 'date', 'time', 
            'alarm_count', 'non_alarm_count', 'background_count', 'highfreq_noise_count', 'insect_call_count', 'weather_rain_count', 'lowfreq_noise_count'
            'spectrogram_paths', 'processed'
        ])
        experiment_data_df.to_csv(EVALUATEDATAPATH, index=False)
    else:
        # Load existing metadata file
        experiment_data_df = pd.read_csv(EVALUATEDATAPATH)
    
    # List out all audio files
    audio_files = []
    for root, _, files in os.walk(EVALUATEAUDIO_DIR):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                audio_files.append(os.path.join(root, file))
    
    existing_files = set(experiment_data_df['file_path'].tolist() if 'file_path' in experiment_data_df.columns else [])
    
    # Find new files
    new_files = [f for f in audio_files if f not in existing_files]
    if len(new_files) == 0:
        print("No new evaluation audio files")
        return experiment_data_df
    else:
        print(f"Found {len(new_files)} new evaluation audio files to process")

    # Configure spectrogram transform
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    
    # Process new files
    new_data = []
    for file_path in tqdm(new_files, desc="Processing evaluation audio files"):
        try:
            # Extract metadata from filename
            site, date, time = parse_filename(file_path)
            
            # Process file into 1-second segments
            _, segment_paths = process_audio_file(file_path, mel_spectrogram)
            
            if segment_paths:
                # Add entry to experiment data
                new_data.append({
                    'file_path': file_path,
                    'site': site,
                    'date': date,
                    'time': time,
                    'alarm_count': 0,
                    'non_alarm_count': 0,
                    'background_count': 0,
                    'highfreq_noise_count':0,
                    'insect_call_count':0,
                    "weather_rain_count" : 0,
                    "lowfreq_noise_count": 0,
                    'spectrogram_paths': ','.join(segment_paths),
                    'processed': True  # Mark as processed since we've created the spectrograms
                })
            else:
                print(f"Warning: No segments generated for {file_path}")
                new_data.append({
                    'file_path': file_path,
                    'site': site,
                    'date': date,
                    'time': time,
                    'alarm_count': 0,
                    'non_alarm_count': 0,
                    'background_count': 0,
                    'highfreq_noise_count':0,
                    'insect_call_count':0,
                    "weather_rain_count" : 0,
                    "lowfreq_noise_count": 0,
                    'spectrogram_paths': '',
                    'processed': False  # Mark as not processed since we couldn't generate segments
                })
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Add new data to experiment data
    new_df = pd.DataFrame(new_data)
    experiment_data_df = pd.concat([experiment_data_df, new_df], ignore_index=True)

    # Save updated experiment data
    experiment_data_df.to_csv(EVALUATEDATAPATH, index=False)
    return experiment_data_df
def create_all_spectrograms(force_recreate=False):
    """
    Create spectrograms for all audio files in metadata and experiment data
    
    Args:
        force_recreate: If True, recreate spectrograms even if they exist
    """
    # First, process training data
    if os.path.exists(METADATA_PATH):
        metadata_df = pd.read_csv(METADATA_PATH)
        
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Creating training spectrograms"):
            try:
                file_path = row['file_path']
                spectrogram_path = row['spectrogram_path']
                
                # Skip if spectrogram exists and we're not forcing recreation
                if os.path.exists(spectrogram_path) and not force_recreate:
                    continue
                
                # Load and process audio
                waveform, sr = load_audio(file_path)
                if waveform is None:
                    print(f"Skipping {file_path} - could not load audio")
                    continue
                
                # Pad or trim to 1 second
                waveform = pad_or_trim(waveform)
                
                # Create spectrogram
                spec = mel_spectrogram(waveform)
                spec = torch.log(spec + 1e-9)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(spectrogram_path), exist_ok=True)
                
                # Save spectrogram
                torch.save(spec, spectrogram_path)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    else:
        print("Training metadata file not found. Run getmetadata first.")
        
    # Next, process evaluation data
    if os.path.exists(EVALUATEDATAPATH) and os.path.getsize(EVALUATEDATAPATH) > 0:
        try:
            experiment_df = pd.read_csv(EVALUATEDATAPATH)
            
            # Find unprocessed files
            unprocessed_files = experiment_df[experiment_df['processed'] == False]
            
            for idx, row in tqdm(unprocessed_files.iterrows(), total=len(unprocessed_files), desc="Creating evaluation spectrograms"):
                try:
                    file_path = row['file_path']
                    
                    # Process file into segments and create spectrograms
                    _, segment_paths = process_audio_file(file_path, mel_spectrogram)
                    
                    if segment_paths:
                        # Update spectrogram paths
                        experiment_df.at[idx, 'spectrogram_paths'] = ','.join(segment_paths)
                        experiment_df.at[idx, 'processed'] = True
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            
            # Save updated experiment data
            experiment_df.to_csv(EVALUATEDATAPATH, index=False)
        except pd.errors.EmptyDataError:
            print(f"Warning: {EVALUATEDATAPATH} was empty. Creating new DataFrame.")
            experiment_df = pd.DataFrame(columns=[
                'file_path', 'site', 'date', 'time', 
                'alarm_count', 'non_alarm_count', 'background_count', 'highfreq_noise_count', 'insect_call_count', 'weather_rain_count', 'lowfreq_noise_count'
                'spectrogram_paths', 'processed'
            ])
            experiment_df.to_csv(EVALUATEDATAPATH, index=False)
    else:
        print("Evaluation metadata file not found or empty. Run getexperimentdata first.")
def check_class_distribution(metadata_df):
    """Check the distribution of classes across all files in metadata"""
    if 'label' not in metadata_df.columns:
        return {"error": "No label column in metadata"}
    
    # Count occurrences of each class
    class_counts = metadata_df['label'].value_counts().to_dict()
    
    # Fill in zero counts for any missing classes
    for cls in REQUIRED_CLASSES:
        if cls not in class_counts:
            class_counts[cls] = 0
    
    total_samples = sum(class_counts.values())
    
    distribution = {
        "total_samples": total_samples,
        "class_counts": class_counts,
        "class_percentages": {cls: (count/total_samples*100) if total_samples > 0 else 0 
                             for cls, count in class_counts.items()}
    }
    
    return distribution

def verify_few_shot_requirements(metadata_df, n_way=N_WAY, k_shot=N_SUPPORT, query_size=N_QUERY, test_size=TEST_SIZE):
    """
    Verify the latest dataset meets few shot requirements to prevent future errors
    
    Checks:
    1. Total samples per class
    2. Sufficient samples for support set from train directory
    3. Sufficient samples for query set from train directory
    4. Sufficient samples for test set from test directory
    """
    if 'label' not in metadata_df.columns or 'file_path' not in metadata_df.columns:
        return {"error": "Missing label or file_path column in metadata"}
    
    # Total samples needed per class
    total_samples_needed = k_shot + query_size + test_size
    
    # Detailed verification results
    verification_results = {
        "meets_requirements": True,
        "class_details": {}
    }
    
    for cls in REQUIRED_CLASSES:
        # Separate train and test samples
        train_samples = metadata_df[
            (metadata_df['label'] == cls) & 
            (metadata_df['file_path'].str.contains('train/'))
        ]
        test_samples = metadata_df[
            (metadata_df['label'] == cls) & 
            (metadata_df['file_path'].str.contains('test/'))
        ]
        
        # Verify support set samples from train directory
        support_samples = train_samples.head(k_shot)
        if len(support_samples) < k_shot:
            verification_results["meets_requirements"] = False
            verification_results["class_details"][cls] = {
                "train_samples": len(train_samples),
                "support_samples": len(support_samples),
                "support_samples_needed": k_shot,
                "error": f"Insufficient train support samples. Need {k_shot}, have {len(support_samples)}"
            }
            continue
        
        # Verify query set samples from train directory
        query_samples = train_samples.iloc[k_shot:k_shot+query_size]
        if len(query_samples) < query_size:
            verification_results["meets_requirements"] = False
            verification_results["class_details"][cls] = {
                "train_samples": len(train_samples),
                "query_samples": len(query_samples),
                "query_samples_needed": query_size,
                "error": f"Insufficient train query samples. Need {query_size}, have {len(query_samples)}"
            }
            continue
        
        # Verify test set samples from test directory
        test_samples_subset = test_samples.head(test_size)
        if len(test_samples_subset) < test_size:
            verification_results["meets_requirements"] = False
            verification_results["class_details"][cls] = {
                "test_samples": len(test_samples),
                "test_samples_subset": len(test_samples_subset),
                "test_samples_needed": test_size,
                "error": f"Insufficient test samples. Need {test_size}, have {len(test_samples_subset)}"
            }
            continue
        
        # If we've made it this far, this class passes
        verification_results["class_details"][cls] = {
            "train_total_samples": len(train_samples),
            "test_total_samples": len(test_samples),
            "support_samples": len(support_samples),
            "query_samples": len(query_samples),
            "test_samples": len(test_samples_subset),
            "status": "PASS"
        }
    
    # If any class failed, provide a suggestion
    if not verification_results["meets_requirements"]:
        verification_results["suggestion"] = (
            f"Need {k_shot} support samples, {query_size} query samples from train directories, "
            f"and {test_size} test samples from test directories for all classes: "
            f"{', '.join(REQUIRED_CLASSES)}. "
            "Check the class_details for specific requirements."
        )
    
    return verification_results