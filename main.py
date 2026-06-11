# fsl-2 main.py
import os
import torch
import sys
import traceback
import pandas as pd
import torchaudio
from torch.utils.data import DataLoader

# Add the directory containing the preprocessing script to the Python path
preprocessing_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(preprocessing_dir)

from config import (
    AUDIO_DIR,
    SPECTROGRAM_DIR,
    EVALUATEAUDIO_DIR,
    EVALUATEDATAPATH,
    BATCH_SIZE,
    DEVICE,
    N_SUPPORT,
    N_QUERY,
    TEST_SIZE,
    REQUIRED_CLASSES,
    N_WAY,
    EPISODES,
    PROTO_WEIGHT,
    RELATION_WEIGHT,
    LABEL_MAP,
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS
)

# Import preprocessing and training functions
from src.preprocess import (
    getmetadata,
    create_all_spectrograms,
    check_class_distribution,
    verify_few_shot_requirements,
    getexperimentdata,
    process_audio_file
)
from src.dataset import BirdSoundDataset, SegmentDataset, EpisodicDataLoader
from src.models import CombinedFreqTemporalCNNEncoder, RelationNetwork, EnsembleModel
from src.training import train_few_shot
from src.evaluation import (
    evaluate_episodic, 
    update_metadata_results,
    evaluate_ensemble_classification,
    update_segment_class_counts_with_time_aggregation,
    create_time_aggregated_summary,
    LabeledTestDataset,
    evaluate_labeled_test_set,
    print_evaluation_results,
    save_evaluation_results
)

# Configuration flag for time aggregation
ENABLE_TIME_AGGREGATION = False  # Set to True to enable time aggregation

def preprocess_data():
    """
    Run preprocessing steps to prepare the dataset.
    """
    print("Starting preprocessing...")
    
    # Scan for new audio files and update metadata
    metadata_df = getmetadata()
    
    # Create spectrograms for all training files
    create_all_spectrograms()
    
    # Check class distribution
    dist = check_class_distribution(metadata_df)
    print("Class distribution:")
    for cls, count in dist["class_counts"].items():
        print(f" {cls}: {count} samples ({dist['class_percentages'][cls]:.2f}%)")
    
    return metadata_df

def preprocess_data():
    """
    Run preprocessing steps to prepare the dataset.
    """
    print("Starting preprocessing...")
    
    # Scan for new audio files and update metadata
    metadata_df = getmetadata()
    
    # Create spectrograms for all training files
    create_all_spectrograms()
    
    # Check class distribution
    dist = check_class_distribution(metadata_df)
    print("Class distribution:")
    for cls, count in dist["class_counts"].items():
        print(f" {cls}: {count} samples ({dist['class_percentages'][cls]:.2f}%)")
    
    return metadata_df

def preprocess_evaluation_data():
    """
    Prepare evaluation data by processing audio files into 1-second segments.
    """
    print("Preparing evaluation data...")
    
    # Get or create experiment metadata
    experiment_df = getexperimentdata()
    
    # Process any unprocessed files (this will create spectrograms for 1-second segments)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    
    # For any unprocessed files, process them into segments
    unprocessed_files = experiment_df[experiment_df['processed'] == False]
    for idx, row in unprocessed_files.iterrows():
        try:
            file_path = row['file_path']
            _, segment_paths = process_audio_file(file_path, mel_spectrogram)
            
            if segment_paths:
                # Update paths in DataFrame
                experiment_df.at[idx, 'spectrogram_paths'] = ','.join(segment_paths)
                experiment_df.at[idx, 'processed'] = True
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Save updated DataFrame
    experiment_df.to_csv(EVALUATEDATAPATH, index=False)
    return experiment_df

def split_train_test_data(metadata_df):
    """
    Split metadata into training and test sets based on directory structure.
    Test files should be in test/[class_name]/ folders.
    Training files should be in train/[class_name]/ folders.
    
    Updated for 7-class system: alarm, non_alarm, background, highfreq_noise, 
    insect_call, weather_rain, lowfreq_noise
    """
    # Filter for test files in test/ directory with proper class structure
    test_metadata = metadata_df[
        metadata_df['file_path'].str.contains('test/') & 
        metadata_df['label'].isin(REQUIRED_CLASSES)
    ]
    
    # Filter for training files in train/ directory (or validation/ if you have it)
    train_metadata = metadata_df[
        (metadata_df['file_path'].str.contains('train/') | 
         metadata_df['file_path'].str.contains('validation/')) & 
        metadata_df['label'].isin(REQUIRED_CLASSES)
    ]
    
    if len(test_metadata) == 0:
        print("WARNING: No test files found in test/ directory!")
        print("Expected structure for 7-class system:")
        for class_name in REQUIRED_CLASSES:
            print(f"  test/{class_name}/")
        print("\nAvailable file paths sample:")
        sample_paths = metadata_df['file_path'].head(10).tolist()
        for path in sample_paths:
            print(f"  {path}")
        return train_metadata, pd.DataFrame()
    
    if len(train_metadata) == 0:
        print("WARNING: No training files found in train/ directory!")
        print("Expected structure for 7-class system:")
        for class_name in REQUIRED_CLASSES:
            print(f"  train/{class_name}/")
        return pd.DataFrame(), test_metadata
    
    print(f"\nDataset split summary:")
    print(f"Training files: {len(train_metadata)} samples")
    print(f"Test files: {len(test_metadata)} samples")
    
    # Show class distribution for training
    train_dist = train_metadata['label'].value_counts()
    print(f"\nTraining set class distribution:")
    for class_name in REQUIRED_CLASSES:
        count = train_dist.get(class_name, 0)
        print(f"  {class_name}: {count} samples")
    
    # Show class distribution for test
    test_dist = test_metadata['label'].value_counts()
    print(f"\nTest set class distribution:")
    for class_name in REQUIRED_CLASSES:
        count = test_dist.get(class_name, 0)
        print(f"  {class_name}: {count} samples")
    
    return train_metadata, test_metadata

def evaluate_on_labeled_test_set(ensemble_model, test_metadata, train_metadata):
    """
    Evaluate the trained model on labeled test set
    """
    print("\n" + "="*60)
    print("EVALUATING ON LABELED TEST SET")
    print("="*60)
    
    if len(test_metadata) == 0:
        print("No test files found! Skipping labeled test evaluation.")
        return None, None, None
    
    print(f"Test set contains {len(test_metadata)} samples")
    
    # Check class distribution in test set
    test_dist = test_metadata['label'].value_counts()
    print(f"Test set class distribution:")
    for class_name in REQUIRED_CLASSES:
        count = test_dist.get(class_name, 0)
        print(f"  {class_name}: {count} samples")
    
    # Create test dataset
    test_dataset = LabeledTestDataset(test_metadata)
    
    # Create support dataset from training data
    support_dataset = BirdSoundDataset(train_metadata)
    
    # Evaluate on labeled test set
    print("\nRunning evaluation...")
    results, overall_accuracy, class_accuracies = evaluate_labeled_test_set(
        model=ensemble_model,
        test_dataset=test_dataset,
        support_dataset=support_dataset,
        device=DEVICE,
        n_way=N_WAY,
        n_support=N_SUPPORT,
        batch_size=BATCH_SIZE
    )
    
    # Print detailed results
    print_evaluation_results(
        results=results,
        overall_accuracy=overall_accuracy,
        class_accuracies=class_accuracies,
        show_details=True,
        max_details=30
    )
    
    # Save results to file
    save_evaluation_results(
        results=results,
        overall_accuracy=overall_accuracy,
        class_accuracies=class_accuracies
    )
    
    return results, overall_accuracy, class_accuracies

def run_original_unlabeled_evaluation(ensemble_model, train_metadata):
    """
    Run your original unlabeled evaluation pipeline
    """
    print("\n" + "="*60)
    print("RUNNING ORIGINAL UNLABELED EVALUATION")
    print("="*60)
    
    # Step 7: Prepare evaluation data (your original code)
    print("Preparing evaluation data...")
    experiment_df = preprocess_evaluation_data()
    
    # Create dataset of all 1-second segments for evaluation
    # We need to create a flat list of all spectrogram paths
    all_segment_paths = []
    for idx, row in experiment_df.iterrows():
        if row['processed'] and row['spectrogram_paths']:
            segments = row['spectrogram_paths'].split(',')
            all_segment_paths.extend(segments)
    
    if not all_segment_paths:
        print("No segments found for evaluation!")
        return
    
    # Create a DataFrame with just the paths for the segment dataset
    segments_df = pd.DataFrame({'file_path': all_segment_paths})
    evaluation_dataset = SegmentDataset(segments_df)
    
    # Create support dataset for the unlabeled evaluation
    support_dataset = BirdSoundDataset(train_metadata)
    
    # Step 8: Evaluate the model on segments using ensemble classification
    print(f"Evaluating model on {len(evaluation_dataset)} segments...")
    results = evaluate_ensemble_classification(
        model=ensemble_model,
        segment_dataset=evaluation_dataset,
        support_dataset=support_dataset,
        device=DEVICE,
        n_way=N_WAY,
        n_support=N_SUPPORT,
        batch_size=BATCH_SIZE
    )
    
    # Step 9: Update experiment DataFrame with time-aggregated segment class counts
    print("Updating experiment data with time-based aggregation...")
    experiment_df = update_segment_class_counts_with_time_aggregation(experiment_df, results)
    
    # Step 10: Create and save ONLY the time-aggregated summary
    print("Creating time-aggregated summary...")
    summary_df = create_time_aggregated_summary(experiment_df)
    
    # Save ONLY the summary to EVALUATEDATAPATH
    summary_df.to_csv(EVALUATEDATAPATH, index=False)
    print(f"Saved time-aggregated summary to {EVALUATEDATAPATH}")
    
    # Step 11: Update metadata with prediction results (if needed)
    # Only update if results contain files that are in the main metadata
    metadata_results = [r for r in results if not '_seg' in r.get('file_path', '')]
    if metadata_results:
        update_metadata_results(metadata_results, evaluation_dataset)
    
    print("Unlabeled evaluation complete!")
    
    # Print summary statistics (updated for 7-class system)
    print(f"\nUnlabeled Evaluation Summary:")
    print(f"Total segments evaluated: {len(results)}")
    print(f"Total unique time periods: {len(summary_df)}")
    
    # Count predictions by class
    prediction_counts = {}
    for result in results:
        pred = result.get('prediction', 'unknown')
        prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
    
    print(f"\nSegment-level predictions:")
    for class_name in REQUIRED_CLASSES:
        count = prediction_counts.get(class_name, 0)
        percentage = (count / len(results)) * 100 if len(results) > 0 else 0
        print(f"  {class_name}: {count} segments ({percentage:.1f}%)")
    
    # Show any unknown predictions
    unknown_count = prediction_counts.get('unknown', 0)
    if unknown_count > 0:
        percentage = (unknown_count / len(results)) * 100
        print(f"  unknown: {unknown_count} segments ({percentage:.1f}%)")
    
    # Show time-aggregated statistics (this will depend on your evaluation functions)
    print(f"\nTime-aggregated statistics:")
    print(f"Average counts per time period:")
    
    # Look for class count columns in the summary DataFrame
    class_count_cols = [col for col in summary_df.columns if col.endswith('_count_avg')]
    if class_count_cols:
        for class_col in class_count_cols:
            avg_count = summary_df[class_col].mean()
            class_name = class_col.replace('_count_avg', '')
            print(f"  {class_name}: {avg_count:.1f}")
    else:
        # Fallback: show available numeric columns
        numeric_cols = summary_df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols[:7]:  # Show first 7 numeric columns
            avg_val = summary_df[col].mean()
            print(f"  {col}: {avg_val:.1f}")
    
    # Show examples of time periods with multiple files
    if 'num_files' in summary_df.columns:
        multiple_files = summary_df[summary_df['num_files'] > 1]
        if len(multiple_files) > 0:
            print(f"\nTime periods with multiple files: {len(multiple_files)}")
            print("Examples:")
            for _, row in multiple_files.head(3).iterrows():
                file_info = f"  {row.get('time_key', 'N/A')}: {row['num_files']} files"
                
                # Show class counts if available
                count_info = []
                for class_name in REQUIRED_CLASSES:
                    count_col = f"{class_name}_count_avg"
                    if count_col in row:
                        count_info.append(f"{row[count_col]:.1f}")
                
                if count_info:
                    file_info += f", avg counts [{', '.join(count_info)}]"
                
                print(file_info)


def main():
    # Step 1: Create directories if they don't exist
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
    os.makedirs(EVALUATEAUDIO_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(EVALUATEDATAPATH), exist_ok=True)
    
    try:
        # Step 2: Run preprocessing for training data
        metadata_df = preprocess_data()
        
        # Step 3: Check if we have enough data for few-shot learning
        requirements = verify_few_shot_requirements(
            metadata_df,
            n_way=N_WAY,
            k_shot=N_SUPPORT,
            query_size=N_QUERY,
            test_size=TEST_SIZE
        )
        
        # Step 4: Prepare few-shot experiment
        if requirements["meets_requirements"]:
            # Filter metadata to include only required classes
            all_metadata = metadata_df[metadata_df['label'].isin(REQUIRED_CLASSES)]
            
            # Split into train and test sets based on directory structure
            train_metadata, test_metadata = split_train_test_data(all_metadata)
            
            # Verify we have both training and test data
            if len(train_metadata) == 0:
                print("ERROR: No training data found! Please ensure you have files in train/ directory.")
                print("Expected directory structure for 7-class system:")
                for class_name in REQUIRED_CLASSES:
                    print(f"  train/{class_name}/")
                return
            
            if len(test_metadata) == 0:
                print("WARNING: No test data found! Skipping labeled test evaluation.")
                print("To enable test evaluation, place labeled files in:")
                for class_name in REQUIRED_CLASSES:
                    print(f"  test/{class_name}/ - for {class_name} class samples")
                # Continue with just training evaluation
                test_metadata = pd.DataFrame()
            
            # Create datasets
            all_dataset = BirdSoundDataset(all_metadata)  # For training (includes some test for few-shot)
            train_dataset = BirdSoundDataset(train_metadata)  # Pure training data
            
            # Step 5: Initialize models
            encoder = CombinedFreqTemporalCNNEncoder().to(DEVICE)
            relation_net = RelationNetwork().to(DEVICE)
            ensemble_model = EnsembleModel(encoder, relation_net).to(DEVICE)
            
            # Step 6: Train the model
            print("Starting training...")
            train_losses = train_few_shot(
                model=ensemble_model,
                dataset=all_dataset,
                episodes=EPISODES,
                n_way=N_WAY,
                n_support=N_SUPPORT,
                n_query=N_QUERY,
                relation_weight=RELATION_WEIGHT,
                proto_weight=PROTO_WEIGHT
            )
            

            # Step 7: Evaluate on labeled test set (only if test data exists)
            labeled_results, test_accuracy, class_accuracies = None, None, None
            if len(test_metadata) > 0:
                labeled_results, test_accuracy, class_accuracies = evaluate_on_labeled_test_set(
                    ensemble_model, test_metadata, train_metadata
                )
            else:
                print("\n" + "="*60)
                print("SKIPPING LABELED TEST EVALUATION - NO TEST DATA")
                print("="*60)
                print("To enable labeled test evaluation, organize your data as:")
                for class_name in REQUIRED_CLASSES:
                    print(f"  test/{class_name}/*.wav")
            

            
            if test_accuracy is not None:
                print(f"✓ Labeled Test Set Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
                print(f"✓ Test samples evaluated: {len(labeled_results) if labeled_results else 0}")
                
                if class_accuracies:
                    print("\nPer-class performance:")
                    for class_name in REQUIRED_CLASSES:
                        if class_name in class_accuracies:
                            stats = class_accuracies[class_name]
                            print(f"  {class_name}: {stats['accuracy']*100:.1f}% ({stats['correct']}/{stats['total']})")
                        else:
                            print(f"  {class_name}: No test samples")
            else:
                print("⚠ No labeled test evaluation performed")
            
            print(f"✓ Unlabeled evaluation completed")
            print(f"✓ Training episodes completed: {EPISODES}")
            print(f"✓ Model weights: Proto={PROTO_WEIGHT}, Relation={RELATION_WEIGHT}")
            print(f"✓ 7-class system: {', '.join(REQUIRED_CLASSES)}")
            
        else:
            print("Not enough data for few-shot learning.")
            print(requirements["suggestion"])
            print(f"\nRequired for {N_WAY}-way, {N_SUPPORT}-shot learning:")
            print(f"- At least {N_SUPPORT + N_QUERY} samples per class")
            print(f"- Classes needed: {', '.join(REQUIRED_CLASSES)}")
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()  # Print the full stack trace for debugging

if __name__ == "__main__":
    import torchaudio
    from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS
    main()