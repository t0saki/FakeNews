import matplotlib
# Set the backend *before* importing pyplot
matplotlib.use('Agg')
import os
import random
from collections import Counter
import pickle
import colorsys
import multiprocessing
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
# Try importing UMAP, it's often better than t-SNE but needs installation
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    print("UMAP not found. Falling back to t-SNE for dimensionality reduction.")
    print("Consider installing it: pip install umap-learn")
    HAS_UMAP = False
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from data_loader import PostDataLoader, DirectoryDataLoader

# --- Configuration ---
# POSTS_FILES = ['data/twitter_dataset/devset/posts.txt', 'data/twitter_dataset/testset/posts_groundtruth.txt'] # Add more paths here if needed
# IMAGES_DIRS = ['data/twitter_dataset/devset/images', 'data/twitter_dataset/testset/images'] # Add corresponding image dirs here
# DATA_LOADER = PostDataLoader
# DATA_LOADER_ARGS = [POSTS_FILE, IMAGES_DIRS]
# OUTPUT_DIR_BASE = 'analysis_output'  # Directory to save plots
# RESULTS_CACHE_DIR = 'analysis_cache'  # Directory for cached results

RUMOR_DIRS = ['/mnt/d/LFDev-D/weibo_dataset/rumor_images']
NONRUMOR_DIRS = ['/mnt/d/LFDev-D/weibo_dataset/nonrumor_images']
DATA_LOADER = DirectoryDataLoader
DATA_LOADER_ARGS = [RUMOR_DIRS, NONRUMOR_DIRS]
OUTPUT_DIR_BASE = 'analysis_output_weibo'  # Directory to save plots
RESULTS_CACHE_DIR = 'analysis_cache_weibo'  # Directory for cached results

# Limit processing for faster demo, set to None for all
NUM_SAMPLES_TO_PROCESS = None  # None for all
NUM_SAMPLES_TO_VISUALIZE = 10  # Number of sample images to show per class
TSNE_PERPLEXITY = 30        # t-SNE parameter
UMAP_N_NEIGHBORS = 15       # UMAP parameter
UMAP_MIN_DIST = 0.1         # UMAP parameter
CLUSTER_N = 5               # Number of clusters for KMeans
DEEP_FEATURE_BATCH_SIZE = 128  # Batch size for deep feature extraction
NUM_PARALLEL_WORKERS = min(multiprocessing.cpu_count(), 4)  # Limit CPU workers
# Batch size for parallel basic feature processing
BASIC_PROCESSING_BATCH_SIZE = 32
# Add config for prediction visualization
TOP_N_CLASSES_TO_SHOW = 20  # Number of top predicted classes to show in plots

# Construct cache filename based on limit
limit_str = "all" if NUM_SAMPLES_TO_PROCESS is None else str(
    NUM_SAMPLES_TO_PROCESS)

OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, limit_str)

# Set OMP_NUM_THREADS
os.environ["OMP_NUM_THREADS"] = str(
    int(multiprocessing.cpu_count() / NUM_PARALLEL_WORKERS))


# Ensure output and cache directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_CACHE_DIR, exist_ok=True)

# --- Helper Functions ---


def calculate_blurriness(pil_image):
    """Calculates image blurriness using Laplacian variance from a PIL image."""
    try:
        # Convert PIL Image (RGB) to OpenCV format (BGR NumPy array)
        img = np.array(pil_image)
        img = img[:, :, ::-1].copy()  # Convert RGB to BGR

        if img is None:  # Should not happen if pil_image is valid
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Adjust ksize for sensitivity: smaller detects finer edges (less blur)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var()
        return laplacian_var
    except Exception as e:
        # print(f"Error calculating blur: {e}") # Keep original error context if needed
        return None


def extract_dominant_colors(pil_image, k=5):
    """Extracts k dominant colors using KMeans."""
    try:
        # Resize for speed, preserving aspect ratio
        img = pil_image.copy()
        img.thumbnail((100, 100))
        img_np = np.array(img)
        # Reshape to a list of pixels
        pixels = img_np.reshape(-1, 3)
        # Use KMeans to find dominant colors
        kmeans = KMeans(n_clusters=k, random_state=42,
                        n_init=10)  # Explicitly set n_init
        kmeans.fit(pixels)
        # Get the cluster centers (dominant colors) and their frequencies
        counts = Counter(kmeans.labels_)
        centers = kmeans.cluster_centers_.astype(int)
        # Sort colors by frequency
        sorted_colors = sorted(zip(centers, counts.values()),
                               key=lambda x: x[1], reverse=True)
        return [color for color, count in sorted_colors]
    except Exception as e:
        # print(f"Error extracting dominant colors: {e}")
        return []


def plot_color_palette(colors, title="Dominant Colors"):
    """Plots a list of colors as a palette."""
    n_colors = len(colors)
    if n_colors == 0:
        return None
    palette = np.array(colors)[np.newaxis, :, :]  # Create a 1xN image
    fig, ax = plt.subplots(figsize=(n_colors * 0.8, 1))
    ax.imshow(palette)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    # plt.show() # Use savefig instead for batch processing
    return fig


# --- Worker Function for Parallel Basic Processing ---
def _process_single_image_basic(args):
    """Processes basic info, blur, and colors for a single image."""
    image_path, label = args
    try:
        # Basic check first
        if not os.path.exists(image_path):
            return {'path': image_path, 'label': label, 'error': 'File not found'}

        pil_img = Image.open(image_path).convert('RGB')
        width, height = pil_img.size
        aspect_ratio = width / height if height > 0 else 0

        # Calculate low-level features
        blur = calculate_blurriness(pil_img)  # Assumes modified version
        dominant_colors = extract_dominant_colors(pil_img, k=5)

        pil_img.close()  # Close image after basic processing

        return {
            "path": image_path,
            "label": label,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "blurriness": blur,
            "dominant_colors": dominant_colors,
            "error": None
        }
    except UnidentifiedImageError:
        return {'path': image_path, 'label': label, 'error': 'UnidentifiedImageError'}
    except Exception as e:
        # Log the full error if needed
        return {'path': image_path, 'label': label, 'error': f'Basic processing error: {type(e).__name__}'}


def process_basic_batch(batch_args):
    """Processes a batch of images for basic features."""
    results = []
    for args in batch_args:
        results.append(_process_single_image_basic(args))
    return results


# --- Feature Extraction Setup (Deep Learning) ---
def setup_feature_extractor():
    """Loads a pre-trained ResNet model for feature extraction and classification."""
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained ResNet50 weights
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    # Get class names from metadata
    imagenet_class_names = weights.meta["categories"]

    # 1. Model for Feature Extraction (remove final layer)
    feature_extractor_model = models.resnet50(weights=weights)
    feature_extractor_model = torch.nn.Sequential(
        *(list(feature_extractor_model.children())[:-1]))
    feature_extractor_model = feature_extractor_model.to(device)
    feature_extractor_model.eval()  # Set to evaluation mode

    # 2. Model for Classification (full model)
    classifier_model = models.resnet50(weights=weights)
    classifier_model = classifier_model.to(device)
    classifier_model.eval()  # Set to evaluation mode

    # Define the image transformations required by ResNet
    preprocess = weights.transforms()  # Use transforms associated with the weights

    # Print some info about transforms if needed
    # print("Using transforms:")
    # print(preprocess)

    return feature_extractor_model, classifier_model, preprocess, device, imagenet_class_names


# Add this function after the setup_feature_extractor() function
def get_descriptive_label(label):
    """Convert raw labels to more descriptive ones for visualization."""
    label_map = {
        'rumor': 'Rumor (Fake)',
        'nonrumor': 'Non-Rumor (Real)',
        # Add any other labels if they exist
    }
    return label_map.get(label, label)  # Return original if not in map


# --- Main Analysis Function ---
def analyze_images(loader, feature_extractor, classifier_model, preprocess, device, imagenet_class_names,  # Added classifier and names
                   limit=None, batch_size=DEEP_FEATURE_BATCH_SIZE, num_workers=NUM_PARALLEL_WORKERS):
    """Performs the core analysis with parallel basic features, batched deep features, and class predictions."""
    path_label_pairs = loader.get_path_label_pairs()
    if not path_label_pairs:
        print("No path-label pairs found. Exiting analysis.")
        return None

    if limit and limit < len(path_label_pairs):
        print(f"Processing a random sample of {limit} images.")
        path_label_pairs = random.sample(path_label_pairs, limit)
    else:
        limit = len(path_label_pairs)  # Use actual number for progress bar
        print(f"Processing all {limit} images.")

    # --- 1. Parallel Basic Feature Extraction ---
    basic_results_intermediate = []
    print(
        f"\nStarting basic feature extraction (parallel, {num_workers} workers, batch size: {BASIC_PROCESSING_BATCH_SIZE})...")

    # Group into batches
    num_batches = math.ceil(len(path_label_pairs) /
                            BASIC_PROCESSING_BATCH_SIZE)
    batches = [path_label_pairs[i * BASIC_PROCESSING_BATCH_SIZE:(i + 1) * BASIC_PROCESSING_BATCH_SIZE]
               for i in range(num_batches)]

    # Use ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit batches to the new worker function
        futures = [executor.submit(process_basic_batch, batch)
                   for batch in batches]
        # Collect results (each future returns a list of results)
        for future in tqdm(futures, total=len(batches), desc="Basic Feature Batches"):
            try:
                batch_result = future.result()
                basic_results_intermediate.extend(
                    batch_result)  # Extend the list
            except Exception as e:
                # Note: This error might hide which specific image(s) failed within the batch
                print(f"Error processing a batch: {e}")
                # Optionally, add placeholder errors or try to determine failed items if needed
                pass

    # Filter out errors and collect valid basic results
    basic_results = []
    error_counts = Counter()
    for res in basic_results_intermediate:
        if res.get('error') is None:
            # Remove the 'error' key before appending
            res.pop('error', None)
            # Initialize new keys for deep features and predictions
            res['deep_features'] = None
            res['predicted_class_index'] = None
            res['predicted_class_name'] = None
            res['prediction_confidence'] = None
            basic_results.append(res)
        else:
            error_counts[res['error']] += 1
            # Optional: print detailed warnings
            # print(f"Warning: Skipping {res['path']} due to: {res['error']}")

    if error_counts:
        print("\nErrors during basic feature extraction:")
        for error_type, count in error_counts.items():
            print(f"  - {error_type}: {count} images")

    if not basic_results:
        print("No images successfully processed for basic features.")
        return None

    print(
        f"Successfully processed basic features for {len(basic_results)} images.")

    # --- 2. Batched Deep Feature Extraction & Classification ---
    print(
        f"\nStarting deep feature extraction & classification (batch size: {batch_size})...")
    # Use the successfully processed basic results as the base
    final_results = basic_results  # Already contains initialized prediction keys
    num_batches = math.ceil(len(final_results) / batch_size)

    processed_deep_count = 0
    processed_prediction_count = 0
    # Combine errors for deep features and classification
    deep_feature_errors = Counter()

    for i in tqdm(range(num_batches), desc="Deep Features & Prediction Batches"):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(final_results))
        current_batch_indices = list(range(batch_start, batch_end))

        batch_tensors = []
        # Store original indices (within final_results) for this batch
        indices_in_batch = []

        # Prepare tensors for the current batch
        for original_index in current_batch_indices:
            image_path = final_results[original_index]['path']
            try:
                # Load and preprocess specifically for the model
                pil_img = Image.open(image_path).convert('RGB')
                img_t = preprocess(pil_img)
                pil_img.close()
                batch_tensors.append(img_t)
                indices_in_batch.append(original_index)
            except UnidentifiedImageError:
                deep_feature_errors['UnidentifiedImageError'] += 1
                # print(f"Warning: Could not identify image {image_path} during deep feature batching. Skipping deep features & prediction.")
                pass  # deep_features & predictions remain None
            except Exception as e:
                error_key = f'Preprocessing error: {type(e).__name__}'
                deep_feature_errors[error_key] += 1
                # print(f"Warning: Error preprocessing {image_path} for deep features: {e}. Skipping deep features & prediction.")
                pass  # deep_features & predictions remain None

        # Process the batch if any images were successfully preprocessed
        if batch_tensors:
            batch_t = torch.stack(batch_tensors).to(device)
            try:
                with torch.no_grad():
                    # 1. Extract Features
                    batch_features = feature_extractor(batch_t)
                    # 2. Get Predictions
                    outputs = classifier_model(batch_t)
                    probabilities = torch.softmax(outputs, dim=1)
                    top_prob, top_idx = torch.topk(probabilities, 1, dim=1)

                # --- Process Features ---
                if batch_features.dim() > 2:
                    batch_features = batch_features.squeeze(-1).squeeze(-1)
                batch_features_np = batch_features.cpu().numpy()

                # --- Process Predictions ---
                top_prob_np = top_prob.cpu().numpy().flatten()
                top_idx_np = top_idx.cpu().numpy().flatten()

                # --- Assign features and predictions back ---
                if len(batch_features_np) == len(indices_in_batch) and len(top_idx_np) == len(indices_in_batch):
                    for idx_in_batch, original_idx in enumerate(indices_in_batch):
                        # Assign features
                        final_results[original_idx]['deep_features'] = batch_features_np[idx_in_batch]
                        processed_deep_count += 1
                        # Assign predictions
                        pred_idx = top_idx_np[idx_in_batch]
                        final_results[original_idx]['predicted_class_index'] = pred_idx
                        final_results[original_idx]['predicted_class_name'] = imagenet_class_names[pred_idx]
                        final_results[original_idx]['prediction_confidence'] = top_prob_np[idx_in_batch]
                        processed_prediction_count += 1

                else:
                    mismatch_key = f'Batch/Output size mismatch (batch {i})'
                    # Count all affected
                    deep_feature_errors[mismatch_key] += len(indices_in_batch)
                    print(
                        f"Warning: Mismatch between input batch size and feature/prediction output size for batch {i}. Features/predictions for this batch might be incorrect or missing.")

            except Exception as e:
                error_key = f'Model inference error (batch {i}): {type(e).__name__}'
                # Count all affected
                deep_feature_errors[error_key] += len(indices_in_batch)
                print(
                    f"Error during model inference for batch {i}: {e}. Skipping deep features & predictions for this batch.")
                # No features/predictions assigned, they remain None for this batch

    # --- Analysis Completion ---\n"
    if deep_feature_errors:
        print("\nErrors during deep feature extraction:")
        for error_type, count in deep_feature_errors.items():
            print(f"  - {error_type}: {count} images affected")

    if not final_results:
        print("\nNo images were successfully processed in total.")
        return None

    # Final count report
    print(
        f"\nSuccessfully processed {len(final_results)} images in total (basic features).")
    print(f"  - Extracted deep features for {processed_deep_count} images.")
    print(
        f"  - Generated ImageNet predictions for {processed_prediction_count} images.")

    # Filter out results where deep features failed, if necessary downstream
    # For now, return all results, some may have deep_features=None or predictions=None
    return final_results

# --- Visualization Functions ---


def visualize_sample_images(results, n_samples=5, output_dir="."):
    """Displays sample images for each label."""
    labels = sorted(list(set(r['label'] for r in results)))
    print("\n--- Sample Images ---")
    for label in labels:
        descriptive_label = get_descriptive_label(label)
        print(f"Label: {descriptive_label}")
        label_results = [r for r in results if r['label'] == label]
        if not label_results:
            continue

        samples = random.sample(label_results, min(
            n_samples, len(label_results)))
        fig, axes = plt.subplots(
            1, len(samples), figsize=(len(samples) * 3, 3))
        if len(samples) == 1:
            axes = [axes]  # Make sure axes is iterable

        for i, sample in enumerate(samples):
            try:
                img = Image.open(sample['path'])
                axes[i].imshow(img)
                axes[i].set_title(
                    f"ID: {os.path.basename(sample['path']).split('.')[0]}\nSize: {sample['width']}x{sample['height']}")
                axes[i].axis('off')
                img.close()  # Close after displaying
            except Exception as e:
                axes[i].set_title(
                    f"Error loading\n{os.path.basename(sample['path']).split('.')[0]}")
                axes[i].axis('off')
                print(f"Error loading sample {sample['path']}: {e}")

        plt.suptitle(f"Sample Images - {descriptive_label}", y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(
            output_dir, f"sample_images_{label}.png"), bbox_inches='tight')
        plt.close(fig)  # Close figure to free memory


def visualize_feature_distributions(results, output_dir="."):
    """Plots distributions of selected features, compared by label."""
    print("\n--- Feature Distributions ---")
    # Convert results to a pandas DataFrame
    plot_data_list = []
    for r in results:
        plot_data_list.append({
            'label': get_descriptive_label(r['label']),  # Use descriptive label
            'width': r['width'],
            'height': r['height'],
            'aspect_ratio': r['aspect_ratio'],
            # Handle None
            'blurriness': r['blurriness'] if r['blurriness'] is not None else np.nan
        })

    if not plot_data_list:
        print("No data to plot distributions for.")
        return

    # Convert list of dicts to DataFrame
    plot_df = pd.DataFrame(plot_data_list)
    
    # Set reasonable plotting limits for each feature
    plot_limits = {
        'width': (0, min(3000, plot_df['width'].quantile(0.98))),  # Cap at 3000 or 98th percentile
        'height': (0, min(3000, plot_df['height'].quantile(0.98))), # Cap at 3000 or 98th percentile
        # Common aspect ratios 0.5-3.0
        'aspect_ratio': (plot_df['aspect_ratio'].quantile(0.02), min(3.0, plot_df['aspect_ratio'].quantile(0.98))),
        'blurriness': (0, plot_df['blurriness'].quantile(0.98)) # Cap at reasonable value
    }

    features_to_plot = ['width', 'height', 'aspect_ratio', 'blurriness']
    # Determine the unique labels for consistent ordering in boxplot
    unique_labels = sorted(plot_df['label'].unique())

    fig, axes = plt.subplots(len(features_to_plot), 2,
                             figsize=(12, len(features_to_plot) * 4))

    for i, feature in enumerate(features_to_plot):
        # Filter out NaN values for the current feature using the DataFrame
        valid_df = plot_df.dropna(subset=[feature])

        if valid_df.empty:
            print(f"Skipping '{feature}' distribution plot - no valid data.")
            # Disable axes if no data
            axes[i, 0].set_visible(False)
            axes[i, 1].set_visible(False)
            continue

        # Histogram / Density Plot using the DataFrame
        # Use stat='density' for normalization
        sns.histplot(data=valid_df, x=feature,
                     hue='label', kde=True, ax=axes[i, 0], stat='density', common_norm=False)
        axes[i, 0].set_title(f"Density Plot of {feature.capitalize()}")
        axes[i, 0].set_xlabel(f"{feature.capitalize()} Value")
        axes[i, 0].set_ylabel("Density") # Changed from Frequency
        # Ensure legend is present
        handles, current_labels = axes[i, 0].get_legend_handles_labels()
        if handles: # Only add legend if there's something to show
             axes[i, 0].legend(title='Class Label')
        else:
             axes[i, 0].legend_ = None # Remove empty legend box

        # Set x-axis limits for histogram/density plot
        axes[i, 0].set_xlim(plot_limits[feature])
        
        # For aspect ratio, add reference lines for common aspect ratios
        if feature == 'aspect_ratio':
            common_ratios = {
                '1:1 (Square)': 1.0,
                '4:3': 4/3,
                '16:9': 16/9,
                '3:4': 3/4,
                '9:16': 9/16
            }
            for label, ratio in common_ratios.items():
                if ratio >= plot_limits[feature][0] and ratio <= plot_limits[feature][1]:
                    axes[i, 0].axvline(x=ratio, color='red', linestyle='--', alpha=0.7)
                    axes[i, 0].text(ratio, axes[i, 0].get_ylim()[1]*0.9, label, 
                                    rotation=90, verticalalignment='top', fontsize=8)

        # Box Plot using the DataFrame
        # Pass the DataFrame directly and specify x, y, and order
        sns.boxplot(data=valid_df, x='label', y=feature,
                    # Use unique_labels for order
                    ax=axes[i, 1], order=unique_labels)
        axes[i, 1].set_title(f"Box Plot of {feature.capitalize()}")
        axes[i, 1].set_xlabel("Class Label")
        axes[i, 1].set_ylabel(f"{feature.capitalize()} Value")
        
        # Set y-axis limits for boxplot
        axes[i, 1].set_ylim(plot_limits[feature])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_distributions.png"))
    print(f"Saved feature distribution plots to {output_dir}")
    plt.close(fig)


def visualize_dominant_colors(results, n_samples=3, output_dir="."):
    """Visualizes dominant colors for sample images."""
    print("\n--- Dominant Color Palettes ---")
    labels = sorted(list(set(r['label'] for r in results)))
    for label in labels:
        descriptive_label = get_descriptive_label(label)
        print(f"Label: {descriptive_label}")
        # Check if dominant_colors exists and is not empty
        label_results = [r for r in results if r['label']
                         == label and r.get('dominant_colors')]
        if not label_results:
            continue

        samples = random.sample(label_results, min(
            n_samples, len(label_results)))
        for i, sample in enumerate(samples):
            fig = plot_color_palette(sample['dominant_colors'],
                                     title=f"Dominant Colors ({descriptive_label} - Sample {i+1}) - ID: {os.path.basename(sample['path']).split('.')[0]}")
            if fig:
                plt.savefig(os.path.join(
                    output_dir, f"dominant_colors_{label}_sample{i+1}.png"), bbox_inches='tight')
                plt.close(fig)


def visualize_dominant_color_distribution(results, output_dir="."):
    """Visualizes the distribution of the *most* dominant color in HSV space."""
    print("\n--- Dominant Color Distribution (HSV) ---")

    hues, saturations, values, labels = [], [], [], []

    for r in results:
        # Check if dominant_colors exists and is not empty
        dominant_colors = r.get('dominant_colors')
        if dominant_colors:
            label = get_descriptive_label(r['label'])  # Use descriptive label
            # Normalize RGB to 0-1 range
            color_rgb = dominant_colors[0]
            r_norm, g_norm, b_norm = [x / 255.0 for x in color_rgb]
            # Convert to HSV
            try:
                h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
                hues.append(h * 360)  # Hue in degrees (0-360)
                saturations.append(s)
                values.append(v)
                labels.append(label)
            except Exception as e:
                print(
                    f"Warning: Could not convert color {color_rgb} to HSV for {r['path']}: {e}")

    if not hues:
        print("No dominant color data found for distribution plotting.")
        return

    # Create DataFrame
    color_df = pd.DataFrame({
        'Hue (degrees)': hues,
        'Saturation': saturations,
        'Value': values,
        'label': labels
    })

    # Plot 1: Hue vs Saturation Scatter Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=color_df, x='Hue (degrees)', y='Saturation', hue='label',
                    alpha=0.6, s=30, edgecolor=None)  # Removed edgecolor for clarity
    plt.title('Most Dominant Color Distribution (Hue vs Saturation)')
    plt.xlabel('Hue (degrees)')
    plt.ylabel('Saturation (0-1)')
    plt.xlim(0, 360)
    plt.ylim(0, 1)
    plt.legend(title='Class Label')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "dominant_color_hsv_scatter.png"))
    print(f"Saved dominant color HSV scatter plot to {output_dir}")
    plt.close()

    # Plot 2: KDE plots for H, S, V
    # Don't share x-axis for different scales
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

    plot_params = {
        'fill': True,
        'common_norm': False,  # Normalize each density independently - Correct for comparing shapes
        'alpha': 0.5
    }

    # Hue KDE
    sns.kdeplot(data=color_df, x='Hue (degrees)',
                hue='label', ax=axes[0], **plot_params)
    axes[0].set_title('Most Dominant Color Hue Distribution')
    axes[0].set_xlim(0, 360)
    axes[0].set_xlabel('Hue (degrees)')
    axes[0].set_ylabel('Density')
    # Explicitly add legend if needed (hue usually does it)
    handles, current_labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(title='Class Label')
    else:
        axes[0].legend_ = None

    # Saturation KDE
    sns.kdeplot(data=color_df, x='Saturation',
                hue='label', ax=axes[1], **plot_params)
    axes[1].set_title('Most Dominant Color Saturation Distribution')
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel('Saturation (0-1)')
    axes[1].set_ylabel('Density')
    # Explicitly add legend if needed
    handles, current_labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(title='Class Label')
    else:
        axes[1].legend_ = None

    # Value KDE
    sns.kdeplot(data=color_df, x='Value', hue='label',
                ax=axes[2], **plot_params)
    axes[2].set_title('Most Dominant Color Value (Brightness) Distribution')
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel('Value (0-1)')
    axes[2].set_ylabel('Density')
    # Explicitly add legend if needed
    handles, current_labels = axes[2].get_legend_handles_labels()
    if handles:
        axes[2].legend(title='Class Label')
    else:
        axes[2].legend_ = None

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dominant_color_hsv_kde.png"))
    print(f"Saved dominant color HSV KDE plots to {output_dir}")
    plt.close(fig)


def visualize_dimensionality_reduction(results, method='umap', output_dir="."):
    """Performs t-SNE or UMAP and visualizes the results."""
    print(f"\n--- Dimensionality Reduction ({method.upper()}) ---")

    # Filter results to only include those with successful deep features
    results_with_features = [
        r for r in results if r.get('deep_features') is not None]
    if not results_with_features:
        print("No deep features found for dimensionality reduction.")
        return None, None

    deep_features = np.array([r['deep_features']
                             for r in results_with_features])
    # Use labels corresponding to filtered results
    raw_labels = [r['label'] for r in results_with_features]
    # Convert to descriptive labels for visualization
    descriptive_labels = [get_descriptive_label(label) for label in raw_labels]

    if deep_features.ndim != 2 or deep_features.shape[0] < 2:
        print("Not enough data or invalid feature shape for dimensionality reduction.")
        return None, None  # Return None to indicate failure

    print(
        f"Performing {method.upper()} on {deep_features.shape[0]} samples (Feature dim: {deep_features.shape[1]})...")

    if method == 'umap' and HAS_UMAP:
        reducer = UMAP(n_neighbors=UMAP_N_NEIGHBORS,
                       min_dist=UMAP_MIN_DIST, n_components=2, random_state=42)
    else:
        if method == 'umap':
            print("Falling back to t-SNE.")
        method = 'tsne'  # Ensure method name is correct for title
        n_samples = deep_features.shape[0]
        # Adjust perplexity if needed
        current_perplexity = min(TSNE_PERPLEXITY, n_samples - 1)
        if current_perplexity <= 0:
            print(
                f"Skipping t-SNE: Not enough samples ({n_samples}) for perplexity {TSNE_PERPLEXITY}.")
            return None, None
        print(f"Using t-SNE with perplexity={current_perplexity}")
        reducer = TSNE(n_components=2, random_state=42,
                       perplexity=current_perplexity, n_iter=300)  # Faster iteration count

    try:
        embedding = reducer.fit_transform(deep_features)
    except ValueError as e:
        print(f"Error during {method.upper()} fitting: {e}")
        print("This might happen with very few samples or zero variance features.")
        return None, None

    print(f"{method.upper()} finished.")

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1],
                    hue=descriptive_labels, alpha=0.7, s=50)  # Increased point size
    plt.title(f'{method.upper()} Visualization of Image Features')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.legend(title='Class Label', loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, f"{method}_visualization.png"))
    print(f"Saved {method.upper()} visualization to {output_dir}")
    plt.close()

    # Return embedding and the filtered results list (for clustering)
    return embedding, results_with_features


def visualize_clustering(embedding, results_with_features, n_clusters, output_dir="."):
    """Performs KMeans clustering on the reduced embedding and visualizes."""
    # Ensure results_with_features is used here, which matches the embedding
    if embedding is None or not results_with_features or embedding.shape[0] < n_clusters:
        print("Skipping clustering: Invalid embedding or not enough samples with features.")
        return

    # Extract labels corresponding to the embedding and calculate overall distribution
    raw_labels = [r['label'] for r in results_with_features]
    descriptive_labels = [get_descriptive_label(label) for label in raw_labels]
    overall_label_counts = Counter(descriptive_labels)
    total_samples = len(descriptive_labels)
    overall_label_proportions = {label: count / total_samples
                               for label, count in overall_label_counts.items()}


    print(f"\n--- Clustering (KMeans, k={n_clusters}) ---")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embedding)

    # Plot clusters
    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1],
                          c=cluster_labels, cmap='viridis', alpha=0.7, s=50)
    plt.title(f'KMeans Clustering (k={n_clusters}) on Reduced Features')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(handles=scatter.legend_elements()[0], labels=[
               f'Cluster {i}' for i in range(n_clusters)], title="Cluster Labels")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "clustering_visualization.png"))
    print(f"Saved clustering visualization to {output_dir}")
    plt.close()

    # Analyze cluster composition
    print("\nCluster Composition (Label Distribution):")
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        # Use descriptive labels for better readability
        cluster_actual_labels = [descriptive_labels[idx] for idx in cluster_indices]
        if not cluster_actual_labels:
            continue
        label_counts = Counter(cluster_actual_labels)
        print(
            f"  Cluster {i} ({len(cluster_indices)} images): {dict(label_counts)}")

    # Show representative images per cluster
    print("\nRepresentative Images per Cluster:")
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        if len(cluster_indices) == 0:
            continue

        # Select a few random images from the cluster
        n_rep_samples = min(5, len(cluster_indices))
        sample_indices_in_cluster = random.sample(
            list(cluster_indices), n_rep_samples)
        # Map back to the original indices in results_with_features
        sample_results = [results_with_features[idx]
                          for idx in sample_indices_in_cluster]

        fig, axes = plt.subplots(
            1, n_rep_samples, figsize=(n_rep_samples * 3, 3))
        if n_rep_samples == 1:
            axes = [axes]  # Make sure axes is iterable

        # Calculate normalized label proportions for this cluster
        cluster_label_counts = Counter(
            [descriptive_labels[idx] for idx in cluster_indices])
        cluster_total = len(cluster_indices)
        normalized_proportions_str = []
        for label, count in sorted(cluster_label_counts.items()):
            cluster_prop = count / cluster_total
            overall_prop = overall_label_proportions.get(label, 0)
            # Avoid division by zero; indicate if overall proportion is 0
            if overall_prop > 0:
                normalized_ratio = cluster_prop / overall_prop
                normalized_proportions_str.append(
                    f"{label}: {normalized_ratio:.1f}x")
            else:
                normalized_proportions_str.append(f"{label}: inf") # Or some indicator
        proportion_title_part = " (Normalized: " + ", ".join(normalized_proportions_str) + ")"


        print(f"  Cluster {i}:")
        for j, sample in enumerate(sample_results):
            try:
                img = Image.open(sample['path'])
                descriptive_label = get_descriptive_label(sample['label'])
                axes[j].imshow(img)
                axes[j].set_title(
                    f"Label: {descriptive_label}\nID: {os.path.basename(sample['path']).split('.')[0]}", fontsize=8)
                axes[j].axis('off')
                img.close()
            except Exception as e:
                axes[j].set_title(
                    f"Error loading\n{os.path.basename(sample['path']).split('.')[0]}", fontsize=8) # Smaller font if needed
                axes[j].axis('off')

        # Add normalized proportions to the title
        plt.suptitle(f"Representative Images - Cluster {i}{proportion_title_part}", y=1.10, fontsize=10) # Adjust y and fontsize as needed
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to prevent overlap
        plt.savefig(os.path.join(
            output_dir, f"cluster_{i}_representatives.png"), bbox_inches='tight')
        plt.close(fig)


def visualize_predicted_classes(results, top_n=TOP_N_CLASSES_TO_SHOW, output_dir="."):
    """Analyzes and visualizes the distribution of predicted ImageNet classes."""
    print(f"\n--- Predicted Class Analysis (Top {top_n}) ---")

    # Filter results for those with valid predictions
    pred_results = [r for r in results if r.get(
        'predicted_class_name') is not None]
    if not pred_results:
        print("No valid prediction results found to visualize.")
        return

    # Create DataFrame with descriptive labels
    pred_df = pd.DataFrame([{
        'label': get_descriptive_label(r['label']),  # Use descriptive label
        'predicted_class_name': r['predicted_class_name'],
        'prediction_confidence': r['prediction_confidence']
    } for r in pred_results])

    labels = sorted(pred_df['label'].unique())
    num_labels = len(labels)

    # --- Plot 1: Top N Predicted Classes per Label ---
    fig, axes = plt.subplots(1, num_labels, figsize=(
        7 * num_labels, 8), sharey=False)  # Increased height
    if num_labels == 1:
        axes = [axes]  # Ensure axes is iterable

    print("\nTop Predicted Classes per Label:")
    for i, label in enumerate(labels):
        label_df = pred_df[pred_df['label'] == label]
        class_counts = label_df['predicted_class_name'].value_counts()
        top_classes = class_counts.head(top_n)

        print(f"  Label '{label}' (Top {min(top_n, len(top_classes))}):")
        for class_name, count in top_classes.items():
            print(f"    - {class_name}: {count}")

        if top_classes.empty:
            axes[i].text(0.5, 0.5, "No prediction data", horizontalalignment='center',
                         verticalalignment='center', transform=axes[i].transAxes)
            axes[i].set_title(f"Top {top_n} Predicted Classes - {label}")
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        else:
            sns.barplot(x=top_classes.values, y=top_classes.index,
                        ax=axes[i], palette="viridis", orient='h')
            axes[i].set_title(f"Top {top_n} Predicted Classes - {label}")
            axes[i].set_xlabel("Count (Number of Images)")
            axes[i].set_ylabel("Predicted ImageNet Class")
            # Adjust label size if names are long
            axes[i].tick_params(axis='y', labelsize=8)

    plt.suptitle("Most Frequent Predicted ImageNet Classes by Data Label", y=1.02)
    # Adjust layout to prevent title overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(os.path.join(
        output_dir, "predicted_classes_top_n_barplot.png"))
    print(f"Saved top predicted classes bar plot to {output_dir}")
    plt.close(fig)

    # --- Plot 2: Prediction Confidence Distribution ---
    fig, ax = plt.subplots(figsize=(10, 6))
    # Use stat='density' for normalization
    sns.histplot(data=pred_df, x='prediction_confidence',
                 hue='label', kde=True, ax=ax, bins=30, stat='density', common_norm=False)
    ax.set_title("Density Distribution of Prediction Confidence by Class")
    ax.set_xlabel("Prediction Confidence (0-1)")
    ax.set_ylabel("Density") # Changed from Count
    # Ensure legend is present
    handles, current_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title='Class Label')
    else:
        ax.legend_ = None

    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, "prediction_confidence_distribution.png"))
    print(f"Saved prediction confidence distribution plot to {output_dir}")
    plt.close(fig)


# --- Main Execution ---
if __name__ == "__main__":
    # Set the start method to 'spawn' to avoid potential deadlocks on Linux
    # Needs to be done early, within the main block
    try:
        # Ensure spawn is used, especially important if Tkinter issues arise
        multiprocessing.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        # Might already be set or not applicable in some environments
        print("Could not set multiprocessing start method (might be already set).")
        pass

    print("--- Starting Image Analysis for Fake News Detection ---")

    # --- UPDATE CACHE FILENAME ---
    cache_filename = f"analysis_results_{limit_str}_v3.pkl"  # Changed to v3
    cache_filepath = os.path.join(RESULTS_CACHE_DIR, cache_filename)

    # 1. Load Data Paths and Labels
    print("\n--- 1. Loading Data ---")
    try:
        loader = DATA_LOADER(*DATA_LOADER_ARGS)
        loader.load_data()
        print(
            f"Found {len(loader.get_path_label_pairs())} image path-label pairs.")
        if not loader.get_path_label_pairs():
            raise ValueError("No data loaded. Check file path and format.")
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"Error initializing or loading data: {e}")
        exit()  # Exit if data loading fails

    # Check for cached results
    analysis_results = None
    if os.path.exists(cache_filepath):
        print(
            f"\n--- Attempting to load cached results from {cache_filepath} ---")
        try:
            with open(cache_filepath, 'rb') as f:
                analysis_results = pickle.load(f)
            print(
                # Updated format string
                f"Successfully loaded cached results for {limit_str} samples (v3 format).")
        except Exception as e:
            print(f"Error loading cached results: {e}. Recomputing...")
            analysis_results = None  # Ensure recomputation if loading fails

    if analysis_results is None:
        # 2. Setup Feature Extractor & Classifier
        print("\n--- 2. Setting up Models (Feature Extractor & Classifier) ---")
        try:
            # Update to get classifier model and class names
            feature_extractor, classifier_model, preprocess, device, imagenet_class_names = setup_feature_extractor()
        except Exception as e:
            print(f"Error setting up models: {e}")
            exit()

        # 3. Perform Analysis (Feature Extraction, Classification - now optimized)
        print("\n--- 3. Analyzing Images (Optimized) ---")
        analysis_results = analyze_images(
            loader, feature_extractor, classifier_model, preprocess, device, imagenet_class_names,  # Pass new args
            limit=NUM_SAMPLES_TO_PROCESS,
            batch_size=DEEP_FEATURE_BATCH_SIZE,
            num_workers=NUM_PARALLEL_WORKERS)

        # Save results if analysis was successful
        if analysis_results:
            print(f"\n--- Saving analysis results to {cache_filepath} ---")
            try:
                with open(cache_filepath, 'wb') as f:
                    pickle.dump(analysis_results, f)
                # Updated format string
                print("Successfully saved results (v3 format).")
            except Exception as e:
                print(f"Error saving results to cache: {e}")

    # 4. Perform Visualizations (only if analysis was successful)
    if analysis_results:
        print("\n--- 4. Generating Visualizations ---")

        # 4.1 Sample Images
        visualize_sample_images(
            analysis_results, n_samples=NUM_SAMPLES_TO_VISUALIZE, output_dir=OUTPUT_DIR)

        # 4.2 Feature Distributions
        visualize_feature_distributions(
            analysis_results, output_dir=OUTPUT_DIR)

        # 4.3 Dominant Colors
        visualize_dominant_colors(
            analysis_results, n_samples=3, output_dir=OUTPUT_DIR)

        # 4.3.1 Dominant Color Distribution
        visualize_dominant_color_distribution(
            analysis_results, output_dir=OUTPUT_DIR)

        # 4.4 NEW: Visualize Predicted Classes
        visualize_predicted_classes(
            analysis_results, top_n=TOP_N_CLASSES_TO_SHOW, output_dir=OUTPUT_DIR)

        # 4.5 Dimensionality Reduction
        print("\n--- 4.5 Dimensionality Reduction ---")  # Renumbered title
        reduction_method = 'umap' if HAS_UMAP else 'tsne'
        # Pass the full results, filtering happens inside the function
        embedding, results_with_features = visualize_dimensionality_reduction(
            analysis_results, method=reduction_method, output_dir=OUTPUT_DIR)

        # 4.6 Clustering (based on dimensionality reduction result)
        print("\n--- 4.6 Clustering ---")  # Renumbered title
        # Pass the filtered results_with_features obtained from reduction
        if embedding is not None and results_with_features:
            visualize_clustering(embedding, results_with_features,
                                 n_clusters=CLUSTER_N, output_dir=OUTPUT_DIR)
        else:
            print(
                "\nSkipping clustering because dimensionality reduction failed or produced no results.")

    else:
        print("\nNo analysis results generated, skipping visualization.")

    print("\n--- Analysis Complete ---")
    print(f"Check the '{OUTPUT_DIR}' directory for saved plots.")
    if os.path.exists(cache_filepath):
        print(f"Analysis results cached at: {cache_filepath}")

