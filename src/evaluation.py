# fsl-2 evaluation.py
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import os
from typing import List, Dict, Optional
from torch.utils.data import DataLoader
from config import N_WAY, N_SUPPORT, N_QUERY, METADATA_PATH, EPISODES, LEARNING_RATE, PROTO_WEIGHT, RELATION_WEIGHT, LABEL_MAP, EVALUATEDATAPATH,REQUIRED_CLASSES,   BATCH_SIZE
from src.dataset import EpisodicDataLoader, BirdSoundDataset
from datetime import datetime

# Get all class names from LABEL_MAP
ALL_CLASSES = list(LABEL_MAP.keys())

def extract_time_from_filename(filename):
    """
    Extract time information from filename.
    Assumes filename contains time in format like 'YYYYMMDD_HHMMSS' or similar patterns.
    
    Args:
        filename: The filename to extract time from
        
    Returns:
        time_key: A string representing the time (e.g., 'YYYY-MM-DD_HH')
    """
    # Remove path and extension
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # Try to extract datetime patterns
    patterns = [
        r'(\d{8})_(\d{6})',  # YYYYMMDD_HHMMSS
        r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})',  # YYYY-MM-DD_HH-MM-SS
        r'(\d{4}\d{2}\d{2})_(\d{2}\d{2}\d{2})',  # YYYYMMDD_HHMMSS
    ]
    
    for pattern in patterns:
        match = re.search(pattern, base_name)
        if match:
            date_part = match.group(1)
            time_part = match.group(2)
            
            # Parse date
            if len(date_part) == 8:  # YYYYMMDD
                year = date_part[:4]
                month = date_part[4:6]
                day = date_part[6:8]
                date_str = f"{year}-{month}-{day}"
            else:  # Already formatted
                date_str = date_part
            
            # Parse time (just get hour)
            if len(time_part) >= 2:
                hour = time_part[:2]
                return f"{date_str}_{hour}"
    
    # If no pattern matches, return the filename without extension as fallback
    return base_name


def initialize_class_counts():
    """Initialize a dictionary with counts for all classes set to 0"""
    return {class_name: 0 for class_name in ALL_CLASSES}


def update_segment_class_counts_with_time_aggregation(experiment_df, results):
    """
    Update segment class counts in experiment DataFrame with time-based aggregation.
    When multiple files have the same time, their counts are averaged.
    Updated to handle all 7 classes.
    
    Args:
        experiment_df: DataFrame with experiment files
        results: List of dictionaries with prediction results
        
    Returns:
        experiment_df: Updated DataFrame with time-aggregated counts
    """
    print(f"Processing {len(results)} results with time-based aggregation...")
    
    # Group results by original file path
    file_predictions = {}
    
    for result in results:
        file_path = result.get('file_path', '')
        if not file_path:
            continue
            
        # Extract the base filename to match with experiment data
        if '_seg' in file_path:
            base_filename = os.path.basename(file_path)
            parts = base_filename.split('_seg')
            if len(parts) > 1:
                base_part = parts[0]
                if '.' in parts[-1]:
                    ext = '.' + parts[-1].split('.')[-1] if '.' in parts[-1] else ''
                    base_filename = base_part + ext
                else:
                    base_filename = base_part
        else:
            base_filename = os.path.basename(file_path)
        
        # Map the prediction to class name using reverse label mapping
        prediction_num = result.get('prediction', -1)
        REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
        prediction = REVERSE_LABEL_MAP.get(prediction_num, "unknown")
        
        if base_filename not in file_predictions:
            file_predictions[base_filename] = initialize_class_counts()
        
        if prediction in file_predictions[base_filename]:
            file_predictions[base_filename][prediction] += 1
        else:
            print(f"Warning: Unknown prediction '{prediction}' for file {base_filename}")
    
    print(f"Grouped results into {len(file_predictions)} unique files")
    
    # Add time extraction and file path information to experiment_df
    experiment_df['time_key'] = experiment_df['file_path'].apply(
        lambda x: extract_time_from_filename(x)
    )
    experiment_df['base_filename'] = experiment_df['file_path'].apply(
        lambda x: os.path.basename(x)
    )
    
    # Initialize count columns for all classes if they don't exist
    for class_name in ALL_CLASSES:
        count_col = f'{class_name}_count'
        if count_col not in experiment_df.columns:
            experiment_df[count_col] = 0
    
    # First, update individual file counts
    for idx, row in experiment_df.iterrows():
        file_path = row['file_path']
        base_filename = os.path.basename(file_path)
        base_filename_no_ext = os.path.splitext(base_filename)[0]
        
        if base_filename in file_predictions:
            counts = file_predictions[base_filename]
            for class_name in ALL_CLASSES:
                experiment_df.at[idx, f'{class_name}_count'] = counts[class_name]
        elif base_filename_no_ext in file_predictions:
            counts = file_predictions[base_filename_no_ext]
            for class_name in ALL_CLASSES:
                experiment_df.at[idx, f'{class_name}_count'] = counts[class_name]
        else:
            # Try to find a match with different extension
            found_match = False
            for pred_filename in file_predictions.keys():
                if os.path.splitext(pred_filename)[0] == base_filename_no_ext:
                    counts = file_predictions[pred_filename]
                    for class_name in ALL_CLASSES:
                        experiment_df.at[idx, f'{class_name}_count'] = counts[class_name]
                    found_match = True
                    break
            
            if not found_match:
                print(f"Warning: Could not find predictions for file {base_filename}")
                # Set default values for all classes
                for class_name in ALL_CLASSES:
                    experiment_df.at[idx, f'{class_name}_count'] = 0
    
    # Now perform time-based aggregation
    print("Performing time-based aggregation...")
    
    # Create aggregation dictionary for all classes
    agg_dict = {}
    for class_name in ALL_CLASSES:
        agg_dict[f'{class_name}_count'] = 'mean'
    
    # Group by time_key and calculate mean counts
    time_groups = experiment_df.groupby('time_key').agg(agg_dict).round(1)
    
    # Add aggregated columns for all classes
    for class_name in ALL_CLASSES:
        count_col = f'{class_name}_count'
        avg_col = f'{class_name}_count_time_avg'
        experiment_df[avg_col] = experiment_df['time_key'].map(time_groups[count_col])
    
    # Count how many files contribute to each time period
    time_file_counts = experiment_df.groupby('time_key').size()
    experiment_df['files_per_time'] = experiment_df['time_key'].map(time_file_counts)
    
    # Print aggregation summary
    print(f"\nTime-based aggregation summary:")
    print(f"Total unique time periods: {len(time_groups)}")
    
    # Show examples of aggregation
    multiple_files_times = time_file_counts[time_file_counts > 1]
    if len(multiple_files_times) > 0:
        print(f"Time periods with multiple files: {len(multiple_files_times)}")
        print("\nExamples of aggregated time periods:")
        for time_key in multiple_files_times.head(3).index:
            files_at_time = experiment_df[experiment_df['time_key'] == time_key]
            print(f"\nTime {time_key} ({len(files_at_time)} files):")
            for _, file_row in files_at_time.iterrows():
                counts = [file_row[f'{class_name}_count'] for class_name in ALL_CLASSES]
                print(f"  {file_row['base_filename']}: {counts}")
            avg_counts = [time_groups.loc[time_key, f'{class_name}_count'] for class_name in ALL_CLASSES]
            print(f"  Average: {avg_counts}")
    else:
        print("No time periods with multiple files found.")
    
    # Return the updated DataFrame
    return experiment_df


def create_time_aggregated_summary(experiment_df):
    """
    Create a summary DataFrame with one row per time period showing aggregated counts.
    Updated to handle all 7 classes.
    
    Args:
        experiment_df: DataFrame with time-aggregated results
        
    Returns:
        summary_df: DataFrame with one row per time period (clean summary only)
    """
    # Group by time and get unique values for each time period
    summary_data = []
    
    for time_key in experiment_df['time_key'].unique():
        time_group = experiment_df[experiment_df['time_key'] == time_key]
        
        # Get the first row as representative (since aggregated values are the same for all files in the time group)
        first_row = time_group.iloc[0]
        
        # Create clean summary row with only aggregated data
        summary_row = {
            'time_key': time_key,
            'num_files': len(time_group),
        }
        
        # Add average counts for all classes
        total_segments = 0
        for class_name in ALL_CLASSES:
            avg_col = f'{class_name}_count_time_avg'
            count_avg_col = f'{class_name}_count_avg'
            summary_row[count_avg_col] = first_row[avg_col]
            total_segments += first_row[avg_col]
        
        summary_row['total_segments_avg'] = total_segments
        summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('time_key')
    
    # Define columns to keep
    keep_columns = ['time_key', 'num_files', 'total_segments_avg']
    for class_name in ALL_CLASSES:
        keep_columns.append(f'{class_name}_count_avg')
    
    # Only keep columns that exist in the DataFrame
    final_columns = [col for col in keep_columns if col in summary_df.columns]
    summary_df = summary_df[final_columns]
    
    return summary_df


def evaluate_ensemble_classification(model, segment_dataset, support_dataset, device, n_way=N_WAY, n_support=N_SUPPORT, batch_size=BATCH_SIZE):
    """
    Evaluate segments using the ensemble model with relation network approach adapted for fsl-2.
    Updated to handle variable number of classes.
    """
    # Use all available classes if n_way is not specified
    if n_way is None:
        n_way = len(ALL_CLASSES)
    
    model.eval()
    
    print(f"Preparing support set for ensemble evaluation with {n_way} classes...")
    
    support_data_by_class = {}
    for i, (spectrogram, label) in enumerate(support_dataset):
        if label not in support_data_by_class:
            support_data_by_class[label] = []
        if len(support_data_by_class[label]) < n_support:
            support_data_by_class[label].append(spectrogram)
    
    if len(support_data_by_class) < n_way:
        print(f"Warning: Support dataset has only {len(support_data_by_class)} classes, need {n_way}")
        n_way = len(support_data_by_class)
    
    class_labels = sorted(support_data_by_class.keys())[:n_way]
    support_images = []
    support_labels = []
    
    class_to_idx = {class_label: idx for idx, class_label in enumerate(class_labels)}
    
    for class_label in class_labels:
        class_spectrograms = support_data_by_class[class_label][:n_support]
        support_images.extend(class_spectrograms)
        support_labels.extend([class_to_idx[class_label]] * len(class_spectrograms))
    
    support_images = torch.stack(support_images).to(device)
    support_labels = torch.tensor(support_labels).to(device)
    
    print(f"Support set classes: {class_labels}")
    print(f"Support labels mapping: {class_to_idx}")
    
    # Compute prototypes for prototypical network part
    if support_images.dim() == 3:
        support_images = support_images.unsqueeze(1)
    
    with torch.no_grad():
        support_embeddings = model.encoder(support_images, return_embedding=True)
        
        # Compute prototypes for each class
        prototypes = []
        for i in range(n_way):
            class_indices = torch.where(support_labels == i)[0]
            if len(class_indices) > 0:
                class_prototypes = support_embeddings[class_indices].mean(0)
                prototypes.append(class_prototypes)
        prototypes = torch.stack(prototypes)
    
    segment_loader = DataLoader(
        segment_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    all_results = []
    segment_idx = 0
    
    with torch.no_grad():
        for batch_spectrograms, _ in tqdm(segment_loader, desc="Evaluating with ensemble"):
            batch_spectrograms = batch_spectrograms.to(device)
            if batch_spectrograms.dim() == 3:
                batch_spectrograms = batch_spectrograms.unsqueeze(1)
            
            # Get embeddings for query samples
            query_embeddings = model.encoder(batch_spectrograms, return_embedding=True)
            
            for i in range(len(batch_spectrograms)):
                current_idx = segment_idx + i
                file_path = segment_dataset.get_file_path(current_idx)
                
                query_embedding = query_embeddings[i]
                
                # Prototypical prediction
                dists = torch.cdist(query_embedding.unsqueeze(0), prototypes)
                proto_logits = -dists.squeeze(0)
                proto_probs = F.softmax(proto_logits, dim=0)
                
                # Relation network prediction
                rel_scores = torch.zeros(n_way, device=device)
                for j in range(n_way):
                    # Create pair of query embedding and prototype
                    relation_pair = torch.cat([
                        query_embedding.unsqueeze(0), 
                        prototypes[j].unsqueeze(0)
                    ], dim=1)
                    rel_scores[j] = model.relation_net(relation_pair)
                
                # Combine predictions
                combined_probs = PROTO_WEIGHT * proto_probs + RELATION_WEIGHT * F.softmax(rel_scores, dim=0)
                predicted_idx = torch.argmax(combined_probs).item()
                confidence = combined_probs[predicted_idx].item()
                
                if predicted_idx < len(class_labels):
                    actual_class_label = class_labels[predicted_idx]
                    reverse_label_map = {v: k for k, v in LABEL_MAP.items()}
                    predicted_label_str = reverse_label_map.get(actual_class_label, 'unknown')
                else:
                    actual_class_label = -1
                    predicted_label_str = 'unknown'
                
                result = {
                    'file_path': file_path,
                    'prediction': predicted_idx,
                    'actual_prediction': actual_class_label,
                    'confidence': confidence,
                    'correct': None
                }
                all_results.append(result)
            
            segment_idx += len(batch_spectrograms)
    
    return all_results


def evaluate_episodic(model, test_dataset, device, n_way=None, n_support=N_SUPPORT, n_query=N_QUERY, n_episodes=EPISODES):
    """
    Evaluate model using episodic few-shot learning paradigm.
    Updated to handle variable number of classes.
    
    Args:
        model: Model with encoder and relation_net components
        test_dataset: Dataset for testing
        device: Computation device
        n_way: Number of classes per episode (if None, uses all available classes)
        n_support: Number of support examples per class
        n_query: Number of query examples per class
        n_episodes: Number of episodes to evaluate
        
    Returns:
        Accuracy, detailed results with filenames
    """
    # Use all available classes if n_way is not specified
    if n_way is None:
        n_way = len(REQUIRED_CLASSES)
    
    model.eval()
    all_results = []
    
    # Create a DataLoader for batch processing
    data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load metadata for training data to get support samples
    train_metadata = pd.read_csv(METADATA_PATH)
    
    # Create a dictionary to store samples by class
    class_samples = {}
    available_classes = REQUIRED_CLASSES[:n_way]  # Use first n_way classes
    
    for cls in available_classes:
        cls_metadata = train_metadata[train_metadata['label'] == cls]
        class_samples[cls] = []
        
        # Load the first n_support samples for each class
        for idx, row in cls_metadata.head(n_support).iterrows():
            try:
                spec_path = row['spectrogram_path']
                if os.path.exists(spec_path):
                    spec = torch.load(spec_path)
                    class_samples[cls].append(spec)
            except Exception as e:
                print(f"Error loading support sample {spec_path}: {e}")
    
    # Convert to tensors and move to device
    support_data = []
    support_labels = []
    
    for cls_idx, cls in enumerate(available_classes):
        for spec in class_samples[cls]:
            if spec.dim() == 2:  # Add channel dimension if needed
                spec = spec.unsqueeze(0)
            support_data.append(spec)
            support_labels.append(cls_idx)
    
    support_data = torch.stack(support_data).to(device)
    support_labels = torch.tensor(support_labels).to(device)
    
    # Process the support set once to get prototypes
    with torch.no_grad():
        # Get encodings
        support_embeddings = model.encoder(support_data, return_embedding=True)
        
        # Compute prototypes for each class
        prototypes = []
        for i in range(n_way):
            class_indices = torch.where(support_labels == i)[0]
            if len(class_indices) > 0:
                class_prototypes = support_embeddings[class_indices].mean(0)
                prototypes.append(class_prototypes)
        prototypes = torch.stack(prototypes)
    
    # Process all query samples (evaluation segments)
    with torch.no_grad():
        for batch_idx, (batch_data, _) in enumerate(tqdm(data_loader, desc="Evaluating segments")):
            # Move batch to device
            batch_data = batch_data.to(device)
            if batch_data.dim() == 3:
                batch_data = batch_data.unsqueeze(1)  # Add channel dimension
            
            # Get embeddings
            query_embeddings = model.encoder(batch_data, return_embedding=True)
            
            # Process each embedding in the batch
            for i, query_embedding in enumerate(query_embeddings):
                # Get file path for this sample
                sample_idx = batch_idx * data_loader.batch_size + i
                file_path = test_dataset.get_file_path(sample_idx)
                
                if file_path is None:
                    continue
                
                # Prototypical prediction
                dists = torch.cdist(query_embedding.unsqueeze(0), prototypes)
                proto_logits = -dists.squeeze(0)
                proto_probs = F.softmax(proto_logits, dim=0)
                
                # Relation network prediction
                rel_scores = torch.zeros(n_way, device=device)
                for j in range(n_way):
                    # Create pair of query embedding and prototype
                    relation_pair = torch.cat([
                        query_embedding.unsqueeze(0), 
                        prototypes[j].unsqueeze(0)
                    ], dim=1)
                    rel_scores[j] = model.relation_net(relation_pair)
                
                # Combine predictions
                combined_probs = PROTO_WEIGHT * proto_probs + RELATION_WEIGHT * F.softmax(rel_scores, dim=0)
                pred_class = torch.argmax(combined_probs).item()
                confidence = combined_probs[pred_class].item()
                
                # Store result
                result = {
                    'file_path': file_path,
                    'prediction': pred_class,
                    'confidence': confidence,
                }
                all_results.append(result)
    
    return all_results 


def update_segment_class_counts(experiment_df, results):
    """
    Update segment class counts in experiment DataFrame based on evaluation results.
    Updated to handle all 7 classes.
    
    Args:
        experiment_df: DataFrame with experiment files
        results: List of dictionaries with prediction results
    """
    # Group results by original file path
    file_predictions = {}
    for result in results:
        file_path = result.get('file_path', '')
        if not file_path:
            continue
            
        # Extract the base filename to match with experiment data
        segment_filename = os.path.basename(file_path)
        
        # Extract the original file identifier (without _segXX.pt)
        # For example, from "SMM05537-BG2_20221105_081000_seg01.pt" 
        # we want to extract "SMM05537-BG2_20221105_081000"
        match = re.search(r'([\w\d]+-[\w\d]+_\d{8}_\d{6})(?:_seg\d+)?\.pt', segment_filename)
        if match:
            original_id = match.group(1)
        else:
            # If pattern doesn't match, try a simpler approach
            original_id = segment_filename.split('_seg')[0]
        
        # Map the prediction to class name using reverse label mapping
        prediction_num = result.get('prediction', -1)
        REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
        prediction = REVERSE_LABEL_MAP.get(prediction_num, "unknown")
        
        if original_id not in file_predictions:
            file_predictions[original_id] = initialize_class_counts()
        
        # Increment the count for this class
        if prediction in file_predictions[original_id]:
            file_predictions[original_id][prediction] += 1
    
    # Initialize count columns for all classes if they don't exist
    for class_name in ALL_CLASSES:
        count_col = f'{class_name}_count'
        if count_col not in experiment_df.columns:
            experiment_df[count_col] = 0
    
    # Update counts in experiment DataFrame
    updated_count = 0
    for idx, row in experiment_df.iterrows():
        file_path = row['file_path']
        file_basename = os.path.basename(file_path)
        
        # Extract the identifier part without extension
        original_id = os.path.splitext(file_basename)[0]
        
        if original_id in file_predictions:
            counts = file_predictions[original_id]
            for class_name in ALL_CLASSES:
                experiment_df.at[idx, f'{class_name}_count'] = counts[class_name]
            updated_count += 1
    
    print(f"Updated class counts for {updated_count} files")
    experiment_df.to_csv(EVALUATEDATAPATH, index=False)
    return experiment_df


def update_metadata_results(
    results: List[Dict], 
    test_dataset=None,
    metadata_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Updates prediction results in metadata CSV file using string labels.
    """
    metadata_path = metadata_path or METADATA_PATH
    
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    try:
        metadata_df = pd.read_csv(metadata_path)
    except FileNotFoundError:
        print(f"Warning: Metadata file {metadata_path} not found. Creating new one.")
        metadata_df = pd.DataFrame(columns=['file_path', 'label', 'prediction', 'prediction_confidence', 'prediction_correct'])
    
    REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
    
    for col in ['prediction', 'prediction_confidence', 'prediction_correct']:
        if col not in metadata_df.columns:
            metadata_df[col] = None
    
    updated_count = 0
    
    for result in results:
        file_path = result.get("file_path")
        if not file_path:
            print("Warning: Result is missing file_path")
            continue
        
        confidence = result.get("confidence", 0.0)
        correct = result.get("correct", False)
        
        prediction_str = result.get("prediction")
        if not prediction_str:
            prediction_num = result.get("actual_prediction", -1)
            prediction_str = REVERSE_LABEL_MAP.get(prediction_num, "unknown")
        
        metadata_mask = (metadata_df['file_path'] == file_path)
        if metadata_mask.any():
            metadata_df.loc[metadata_mask, 'prediction_confidence'] = confidence
            metadata_df.loc[metadata_mask, 'prediction_correct'] = correct
            metadata_df.loc[metadata_mask, 'prediction'] = prediction_str
            updated_count += 1
        else:
            print(f"Warning: File {file_path} not found in metadata CSV")
    
    try:
        metadata_df.to_csv(metadata_path, index=False)
        print(f"Updated prediction results for {updated_count} files")
    except Exception as e:
        print(f"Error saving metadata: {e}")
    
    return metadata_df


def filter_unprocessed_segments(experiment_df, all_segment_paths):
    """
    Filter out segments from files that already have prediction counts.
    Updated to handle all 7 classes.
    
    Args:
        experiment_df: DataFrame with experiment files and their counts
        all_segment_paths: List of all segment file paths
        
    Returns:
        filtered_segment_paths: List of segments that need evaluation
        already_processed_count: Number of segments skipped
    """
    # Identify files that already have predictions (non-zero or non-null counts)
    processed_files = set()
    
    for idx, row in experiment_df.iterrows():
        # Check if this file already has prediction counts for any class
        has_predictions = False
        total_count = 0
        
        for class_name in ALL_CLASSES:
            count_col = f'{class_name}_count'
            if count_col in row and pd.notna(row[count_col]):
                total_count += row[count_col]
        
        has_predictions = total_count > 0
        
        if has_predictions:
            # Extract the base filename to match against segments
            file_path = row['file_path']
            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            processed_files.add(base_filename)
    
    print(f"Found {len(processed_files)} files that already have predictions")
    
    # Filter segment paths to exclude those from already processed files
    filtered_paths = []
    skipped_count = 0
    
    for segment_path in all_segment_paths:
        segment_filename = os.path.basename(segment_path)
        
        # Extract the original file identifier from segment filename
        # Example: "SMM05537-BG2_20221105_081000_seg01.pt" -> "SMM05537-BG2_20221105_081000"
        match = re.search(r'([\w\d]+-[\w\d]+_\d{8}_\d{6})(?:_seg\d+)?\.pt', segment_filename)
        if match:
            original_id = match.group(1)
        else:
            original_id = segment_filename.split('_seg')[0]
        
        if original_id not in processed_files:
            filtered_paths.append(segment_path)
        else:
            skipped_count += 1
    
    print(f"Filtered {len(all_segment_paths)} segments down to {len(filtered_paths)} (skipped {skipped_count})")
    return filtered_paths, skipped_count

def evaluate_labeled_test_set(model, test_dataset, support_dataset, device, n_way=N_WAY, n_support=N_SUPPORT, batch_size=32):
    """
    Evaluate labeled test set files and calculate accuracy
    
    Args:
        model: The ensemble model
        test_dataset: Dataset containing labeled test samples  
        support_dataset: Dataset for creating support prototypes
        device: Computation device
        n_way: Number of classes
        n_support: Number of support examples per class
        batch_size: Batch size for evaluation
        
    Returns:
        results: List of prediction results with true/false indicators
        accuracy: Overall accuracy
    """
    model.eval()
    
    print("Preparing support set for labeled test evaluation...")
    
    # Verify test data structure first
    if hasattr(test_dataset, 'data'):
        if not verify_test_data_structure(test_dataset.data):
            print("Test data structure verification failed, but continuing...")
    
    # Prepare support set from training data
    support_data_by_class = {}
    for i, (spectrogram, label) in enumerate(support_dataset):
        if label not in support_data_by_class:
            support_data_by_class[label] = []
        if len(support_data_by_class[label]) < n_support:
            support_data_by_class[label].append(spectrogram)
    
    if len(support_data_by_class) < n_way:
        raise ValueError(f"Support dataset has only {len(support_data_by_class)} classes, need {n_way}")
    
    class_labels = sorted(support_data_by_class.keys())[:n_way]
    support_images = []
    support_labels = []
    
    class_to_idx = {class_label: idx for idx, class_label in enumerate(class_labels)}
    
    for class_label in class_labels:
        class_spectrograms = support_data_by_class[class_label][:n_support]
        support_images.extend(class_spectrograms)
        support_labels.extend([class_to_idx[class_label]] * len(class_spectrograms))
    
    support_images = torch.stack(support_images).to(device)
    support_labels = torch.tensor(support_labels).to(device)
    
    print(f"Support set classes: {class_labels}")
    print(f"Support labels mapping: {class_to_idx}")
    
    # Compute prototypes for prototypical network part
    if support_images.dim() == 3:
        support_images = support_images.unsqueeze(1)
    
    with torch.no_grad():
        support_embeddings = model.encoder(support_images, return_embedding=True)
        
        # Compute prototypes for each class
        prototypes = []
        for i in range(n_way):
            class_indices = torch.where(support_labels == i)[0]
            if len(class_indices) > 0:
                class_prototypes = support_embeddings[class_indices].mean(0)
                prototypes.append(class_prototypes)
        prototypes = torch.stack(prototypes)
    
    # Create test data loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_results = []
    correct_predictions = 0
    total_predictions = 0
    class_correct = {i: 0 for i in range(n_way)}
    class_total = {i: 0 for i in range(n_way)}
    
    with torch.no_grad():
        for batch_idx, (batch_spectrograms, batch_true_labels) in enumerate(tqdm(test_loader, desc="Evaluating labeled test set")):
            batch_spectrograms = batch_spectrograms.to(device)
            batch_true_labels = batch_true_labels.to(device)
            
            if batch_spectrograms.dim() == 3:
                batch_spectrograms = batch_spectrograms.unsqueeze(1)
            
            # Get embeddings for query samples
            query_embeddings = model.encoder(batch_spectrograms, return_embedding=True)
            
            for i in range(len(batch_spectrograms)):
                query_embedding = query_embeddings[i]
                true_label = batch_true_labels[i].item()
                
                # Map true label to class index for comparison
                true_class_idx = class_to_idx.get(true_label, -1)
                
                # Prototypical prediction
                dists = torch.cdist(query_embedding.unsqueeze(0), prototypes)
                proto_logits = -dists.squeeze(0)
                proto_probs = F.softmax(proto_logits, dim=0)
                
                # Relation network prediction
                rel_scores = torch.zeros(n_way, device=device)
                for j in range(n_way):
                    # Create pair of query embedding and prototype
                    relation_pair = torch.cat([
                        query_embedding.unsqueeze(0), 
                        prototypes[j].unsqueeze(0)
                    ], dim=1)
                    rel_scores[j] = model.relation_net(relation_pair)
                
                # Combine predictions
                combined_probs = PROTO_WEIGHT * proto_probs + RELATION_WEIGHT * F.softmax(rel_scores, dim=0)
                predicted_idx = torch.argmax(combined_probs).item()
                confidence = combined_probs[predicted_idx].item()
                
                # Determine if prediction is correct
                is_correct = (predicted_idx == true_class_idx)
                if is_correct:
                    correct_predictions += 1
                
                total_predictions += 1
                
                # Update per-class statistics
                if true_class_idx >= 0:
                    class_total[true_class_idx] += 1
                    if is_correct:
                        class_correct[true_class_idx] += 1
                
                # Get class names for display
                reverse_label_map = {v: k for k, v in LABEL_MAP.items()}
                true_class_name = reverse_label_map.get(true_label, 'unknown')
                predicted_class_label = class_labels[predicted_idx] if predicted_idx < len(class_labels) else -1
                predicted_class_name = reverse_label_map.get(predicted_class_label, 'unknown')
                
                # Get file path - use original audio file path for better readability
                sample_idx = batch_idx * batch_size + i
                file_path = test_dataset.get_file_path(sample_idx) if hasattr(test_dataset, 'get_file_path') else f"sample_{sample_idx}"
                
                result = {
                    'file_path': file_path,
                    'true_label': true_label,
                    'true_class': true_class_name,
                    'predicted_label': predicted_class_label,
                    'predicted_class': predicted_class_name,
                    'confidence': confidence,
                    'correct': is_correct
                }
                all_results.append(result)
    
    # Calculate overall accuracy
    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    # Calculate per-class accuracy
    class_accuracies = {}
    reverse_label_map = {v: k for k, v in LABEL_MAP.items()}
    
    for class_idx in range(n_way):
        if class_total[class_idx] > 0:
            class_acc = class_correct[class_idx] / class_total[class_idx]
            class_label = class_labels[class_idx]
            class_name = reverse_label_map.get(class_label, f'class_{class_idx}')
            class_accuracies[class_name] = {
                'accuracy': class_acc,
                'correct': class_correct[class_idx],
                'total': class_total[class_idx]
            }
    
    return all_results, overall_accuracy, class_accuracies


def print_evaluation_results(results, overall_accuracy, class_accuracies, show_details=True, max_details=20):
    """
    Print detailed evaluation results
    
    Args:
        results: List of prediction results
        overall_accuracy: Overall accuracy score
        class_accuracies: Per-class accuracy statistics
        show_details: Whether to show individual predictions
        max_details: Maximum number of individual predictions to show
    """
    print(f"\n{'='*60}")
    print(f"LABELED TEST SET EVALUATION RESULTS")
    print(f"{'='*60}")
    
    print(f"\nOVERALL ACCURACY: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"Total predictions: {len(results)}")
    correct_count = sum(1 for r in results if r['correct'])
    print(f"Correct predictions: {correct_count}")
    print(f"Incorrect predictions: {len(results) - correct_count}")
    
    print(f"\nPER-CLASS ACCURACY:")
    print(f"{'Class':<15} {'Accuracy':<10} {'Correct/Total':<15}")
    print(f"{'-'*40}")
    for class_name, stats in class_accuracies.items():
        acc_pct = stats['accuracy'] * 100
        print(f"{class_name:<15} {acc_pct:>7.2f}%   {stats['correct']:>3}/{stats['total']:<3}")
    
    if show_details:
        print(f"\nDETAILED PREDICTIONS (showing first {max_details}):")
        print(f"{'File':<30} {'True':<12} {'Predicted':<12} {'Correct':<8} {'Confidence':<10}")
        print(f"{'-'*80}")
        
        for i, result in enumerate(results[:max_details]):
            file_name = os.path.basename(result['file_path'])[:28]
            true_class = result['true_class'][:10]
            pred_class = result['predicted_class'][:10]
            correct_str = "✓" if result['correct'] else "✗"
            confidence = result['confidence']
            
            print(f"{file_name:<30} {true_class:<12} {pred_class:<12} {correct_str:<8} {confidence:>8.3f}")
        
        if len(results) > max_details:
            print(f"... and {len(results) - max_details} more predictions")
    
    # Show confusion-like statistics
    print(f"\nPREDICTION BREAKDOWN:")
    pred_matrix = {}
    for result in results:
        true_class = result['true_class']
        pred_class = result['predicted_class']
        
        if true_class not in pred_matrix:
            pred_matrix[true_class] = {}
        if pred_class not in pred_matrix[true_class]:
            pred_matrix[true_class][pred_class] = 0
        pred_matrix[true_class][pred_class] += 1
    
    for true_class, predictions in pred_matrix.items():
        print(f"\nTrue class '{true_class}':")
        for pred_class, count in predictions.items():
            status = "✓" if true_class == pred_class else "✗"
            print(f"  {status} Predicted as '{pred_class}': {count}")


def save_evaluation_results(results, overall_accuracy, class_accuracies, output_path=None):
    """
    Save evaluation results to CSV file
    
    Args:
        results: List of prediction results
        overall_accuracy: Overall accuracy score
        class_accuracies: Per-class accuracy statistics
        output_path: Path to save results (optional)
    """
    if output_path is None:
        # Create a timestamped filename for evaluation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.dirname(EVALUATEDATAPATH) if EVALUATEDATAPATH else "results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.csv")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Add summary statistics as metadata
    summary_data = {
        'overall_accuracy': overall_accuracy,
        'total_predictions': len(results),
        'correct_predictions': sum(1 for r in results if r['correct']),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Add per-class accuracies to summary
    for class_name, stats in class_accuracies.items():
        summary_data[f'{class_name}_accuracy'] = stats['accuracy']
        summary_data[f'{class_name}_correct'] = stats['correct']
        summary_data[f'{class_name}_total'] = stats['total']
    
    try:
        # Save results
        results_df.to_csv(output_path, index=False)
        print(f"✓ Results saved to: {output_path}")
        
        # Save summary separately
        summary_path = output_path.replace('.csv', '_summary.csv')
        summary_df = pd.DataFrame([summary_data])
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Summary saved to: {summary_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving evaluation results: {e}")
        print(f"Attempted paths:")
        print(f"  Results: {output_path}")
        print(f"  Summary: {output_path.replace('.csv', '_summary.csv')}")
        
        # Try to save in current directory as fallback
        try:
            fallback_results = f"evaluation_results_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            fallback_summary = fallback_results.replace('.csv', '_summary.csv')
            
            results_df.to_csv(fallback_results, index=False)
            summary_df.to_csv(fallback_summary, index=False)
            
            print(f"✓ Fallback save successful:")
            print(f"  Results: {fallback_results}")
            print(f"  Summary: {fallback_summary}")
            return True
            
        except Exception as fallback_error:
            print(f"❌ Fallback save also failed: {fallback_error}")
            return False


# Updated dataset class to handle labeled test data with proper file path tracking
class LabeledTestDataset(BirdSoundDataset):
    """Dataset class for labeled test data that can return file paths"""
    
    def __init__(self, dataframe, transform=None):
        super().__init__(dataframe, transform)
        # Store both spectrogram paths and original file paths
        self.file_paths = list(dataframe['file_path'])  # Original audio file paths
        self.spectrogram_paths = list(dataframe['spectrogram_path'])  # Spectrogram paths
    
    def get_file_path(self, idx):
        """Get original audio file path for given index"""
        if idx < len(self.file_paths):
            return self.file_paths[idx]
        return None
    
    def get_spectrogram_path(self, idx):
        """Get spectrogram file path for given index"""
        if idx < len(self.spectrogram_paths):
            return self.spectrogram_paths[idx]
        return None


def verify_test_data_structure(test_metadata):
    """
    Verify that test data has the expected directory structure and files
    """
    print("Verifying test data structure...")
    
    required_dirs = ['test/alarm', 'test/non_alarm', 'test/background']
    found_dirs = set()
    
    for _, row in test_metadata.iterrows():
        file_path = row['file_path']
        for req_dir in required_dirs:
            if req_dir in file_path:
                found_dirs.add(req_dir)
    
    print(f"Found test directories: {list(found_dirs)}")
    missing_dirs = [d for d in required_dirs if d not in found_dirs]
    
    if missing_dirs:
        print(f"WARNING: Missing test directories: {missing_dirs}")
        return False
    
    # Check if spectrograms exist
    missing_spectrograms = 0
    for _, row in test_metadata.iterrows():
        spec_path = row['spectrogram_path']
        if not os.path.exists(spec_path):
            missing_spectrograms += 1
    
    if missing_spectrograms > 0:
        print(f"WARNING: {missing_spectrograms} test spectrograms are missing!")
        print("Run preprocessing to create missing spectrograms.")
        return False
    
    print("✓ Test data structure verification passed")
    return True