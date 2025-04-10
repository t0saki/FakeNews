import os
import random
from collections import Counter
import pickle
import colorsys

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

from data_loader import PostDataLoader

# --- Configuration ---
BASE_DIR = 'data/twitter_dataset/devset'
POSTS_FILE_PATH = os.path.join(BASE_DIR, 'posts.txt')
IMAGES_DIR_PATH = os.path.join(BASE_DIR, 'images')
OUTPUT_DIR = 'analysis_output'  # Directory to save plots
RESULTS_CACHE_DIR = 'analysis_cache'  # Directory for cached results
# Limit processing for faster demo, set to None for all
NUM_SAMPLES_TO_PROCESS = 100  # None for all
NUM_SAMPLES_TO_VISUALIZE = 10  # Number of sample images to show per class
TSNE_PERPLEXITY = 30        # t-SNE parameter
UMAP_N_NEIGHBORS = 15       # UMAP parameter
UMAP_MIN_DIST = 0.1         # UMAP parameter
CLUSTER_N = 5               # Number of clusters for KMeans

# Ensure output and cache directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_CACHE_DIR, exist_ok=True)

# --- Helper Functions ---


def calculate_blurriness(pil_image):
    """Calculates image blurriness using Laplacian variance from a PIL image."""
    try:
        # Convert PIL Image (RGB) to OpenCV format (BGR NumPy array)
        img = np.array(pil_image)
        img = img[:, :, ::-1].copy() # Convert RGB to BGR

        if img is None: # Should not happen if pil_image is valid
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


# --- Feature Extraction Setup (Deep Learning) ---
def setup_feature_extractor():
    """Loads a pre-trained ResNet model for feature extraction."""
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained ResNet50
    # Use updated weights API
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Remove the final classification layer
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    # Define the image transformations required by ResNet
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    return model, preprocess, device


def extract_deep_features(pil_image, model, preprocess, device):
    """Extracts features from an image using the pre-trained model."""
    try:
        img_t = preprocess(pil_image)
        batch_t = torch.unsqueeze(img_t, 0).to(device)  # Create a mini-batch
        with torch.no_grad():
            features = model(batch_t)
        # Flatten the features and move to CPU
        return features.squeeze().cpu().numpy()
    except Exception as e:
        # print(f"Error extracting deep features: {e}")
        return None


# --- Main Analysis Function ---
def analyze_images(loader, feature_extractor, preprocess, device, limit=None):
    """Performs the core analysis on the loaded image data."""
    path_label_pairs = loader.get_path_label_pairs()
    if not path_label_pairs:
        print("No path-label pairs found. Exiting analysis.")
        return None

    if limit and limit < len(path_label_pairs):
        print(f"Processing a random sample of {limit} images.")
        path_label_pairs = random.sample(path_label_pairs, limit)
    else:
        print(f"Processing all {len(path_label_pairs)} images.")

    results = []
    print("Starting feature extraction...")
    for image_path, label in tqdm(path_label_pairs):
        try:
            # Basic check
            if not os.path.exists(image_path):
                # print(f"Warning: File not found during analysis: {image_path}. Skipping.")
                continue

            # Load Image using Pillow (needed for deep features & dominant colors)
            pil_img = Image.open(image_path).convert(
                'RGB')  # Ensure RGB for consistency

            # 1. Basic Metadata (can be expanded with EXIF if needed)
            width, height = pil_img.size

            # 2. Low-level Features
            blur = calculate_blurriness(pil_img)
            dominant_colors = extract_dominant_colors(pil_img, k=5)

            # 3. Deep Learning Features
            deep_features = extract_deep_features(
                pil_img, feature_extractor, preprocess, device)

            # Close the image file handle
            pil_img.close()

            if deep_features is not None:  # Only store if deep features were successful
                results.append({
                    "path": image_path,
                    "label": label,
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / height if height > 0 else 0,
                    "blurriness": blur,
                    "dominant_colors": dominant_colors,
                    "deep_features": deep_features
                })

        except UnidentifiedImageError:
            # print(f"Warning: Could not identify image {image_path}. Skipping.")
            continue
        except Exception as e:
            print(
                f"Warning: Unexpected error processing {image_path}: {e}. Skipping.")
            # Ensure image is closed even on error if pil_img was opened
            try:
                pil_img.close()
            except:
                pass

    if not results:
        print("No images were successfully processed.")
        return None

    print(f"\nSuccessfully processed {len(results)} images.")
    return results

# --- Visualization Functions ---


def visualize_sample_images(results, n_samples=5, output_dir="."):
    """Displays sample images for each label."""
    labels = sorted(list(set(r['label'] for r in results)))
    print("\n--- Sample Images ---")
    for label in labels:
        print(f"Label: {label}")
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

        plt.suptitle(f"Sample Images - Label: {label}", y=1.05)
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
            'label': r['label'],
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
        sns.histplot(data=valid_df, x=feature,
                     hue='label', kde=True, ax=axes[i, 0])
        axes[i, 0].set_title(f"Distribution of {feature.capitalize()}")
        axes[i, 0].legend(title='Label')  # Add legend to histplot

        # Box Plot using the DataFrame
        # Pass the DataFrame directly and specify x, y, and order
        sns.boxplot(data=valid_df, x='label', y=feature,
                    # Use unique_labels for order
                    ax=axes[i, 1], order=unique_labels)
        axes[i, 1].set_title(f"Box Plot of {feature.capitalize()}")
        axes[i, 1].set_xlabel("Label")
        axes[i, 1].set_ylabel(feature.capitalize())

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_distributions.png"))
    print(f"Saved feature distribution plots to {output_dir}")
    plt.close(fig)


def visualize_dominant_colors(results, n_samples=3, output_dir="."):
    """Visualizes dominant colors for sample images."""
    print("\n--- Dominant Color Palettes ---")
    labels = sorted(list(set(r['label'] for r in results)))
    for label in labels:
        print(f"Label: {label}")
        label_results = [r for r in results if r['label']
                         == label and r['dominant_colors']]
        if not label_results:
            continue

        samples = random.sample(label_results, min(
            n_samples, len(label_results)))
        for i, sample in enumerate(samples):
            fig = plot_color_palette(sample['dominant_colors'],
                                     title=f"Dominant Colors ({label} - Sample {i+1}) - ID: {os.path.basename(sample['path']).split('.')[0]}")
            if fig:
                plt.savefig(os.path.join(
                    output_dir, f"dominant_colors_{label}_sample{i+1}.png"), bbox_inches='tight')
                plt.close(fig)


def visualize_dominant_color_distribution(results, output_dir="."):
    """Visualizes the distribution of the *most* dominant color in HSV space."""
    print("\n--- Dominant Color Distribution (HSV) ---")

    hues, saturations, values, labels = [], [], [], []

    for r in results:
        # Use only the most dominant color (index 0)
        if r['dominant_colors']:
            label = r['label']
            # Normalize RGB to 0-1 range
            color_rgb = r['dominant_colors'][0]
            r_norm, g_norm, b_norm = [x / 255.0 for x in color_rgb]
            # Convert to HSV
            try:
                h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
                hues.append(h * 360) # Hue in degrees (0-360)
                saturations.append(s)
                values.append(v)
                labels.append(label)
            except Exception as e:
                print(f"Warning: Could not convert color {color_rgb} to HSV for {r['path']}: {e}")

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
    sns.scatterplot(data=color_df, x='Hue (degrees)', y='Saturation', hue='label', alpha=0.6, s=30, edgecolor=None) # Removed edgecolor for clarity
    plt.title('Most Dominant Color Distribution (Hue vs Saturation)')
    plt.xlim(0, 360)
    plt.ylim(0, 1)
    plt.legend(title='Label')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "dominant_color_hsv_scatter.png"))
    print(f"Saved dominant color HSV scatter plot to {output_dir}")
    plt.close()

    # Plot 2: KDE plots for H, S, V
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False) # Don't share x-axis for different scales

    plot_params = {
        'fill': True,
        'common_norm': False, # Normalize each density independently
        'alpha': 0.5
    }

    # Hue KDE
    sns.kdeplot(data=color_df, x='Hue (degrees)', hue='label', ax=axes[0], **plot_params)
    axes[0].set_title('Most Dominant Color Hue Distribution')
    axes[0].set_xlim(0, 360)
    axes[0].set_xlabel('Hue (degrees)')

    # Saturation KDE
    sns.kdeplot(data=color_df, x='Saturation', hue='label', ax=axes[1], **plot_params)
    axes[1].set_title('Most Dominant Color Saturation Distribution')
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel('Saturation')

    # Value KDE
    sns.kdeplot(data=color_df, x='Value', hue='label', ax=axes[2], **plot_params)
    axes[2].set_title('Most Dominant Color Value (Brightness) Distribution')
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel('Value')

    # Add legends to KDE plots if they exist
    for ax in axes:
        handles, current_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(title='Label')
        else: # Remove empty legend boxes
            ax.legend_ = None

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dominant_color_hsv_kde.png"))
    print(f"Saved dominant color HSV KDE plots to {output_dir}")
    plt.close(fig)


def visualize_dimensionality_reduction(results, method='umap', output_dir="."):
    """Performs t-SNE or UMAP and visualizes the results."""
    print(f"\n--- Dimensionality Reduction ({method.upper()}) ---")
    deep_features = np.array([r['deep_features'] for r in results])
    labels = [r['label'] for r in results]

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
                    hue=labels, alpha=0.7, s=50)  # Increased point size
    plt.title(f'{method.upper()} Visualization of Image Features')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.legend(title='Label', loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, f"{method}_visualization.png"))
    print(f"Saved {method.upper()} visualization to {output_dir}")
    plt.close()

    return embedding, labels  # Return embedding for clustering


def visualize_clustering(embedding, labels, results, n_clusters, output_dir="."):
    """Performs KMeans clustering on the reduced embedding and visualizes."""
    if embedding is None or labels is None or embedding.shape[0] < n_clusters:
        print("Skipping clustering: Invalid embedding or not enough samples.")
        return

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
               f'Cluster {i}' for i in range(n_clusters)], title="Clusters")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "kmeans_clusters.png"))
    print(f"Saved KMeans clustering visualization to {output_dir}")
    plt.close()

    # Analyze cluster composition
    print("\nCluster Composition (Label Distribution):")
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_actual_labels = [labels[idx] for idx in cluster_indices]
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
        sample_indices = random.sample(list(cluster_indices), n_rep_samples)
        sample_results = [results[idx] for idx in sample_indices]

        fig, axes = plt.subplots(
            1, n_rep_samples, figsize=(n_rep_samples * 3, 3))
        if n_rep_samples == 1:
            axes = [axes]  # Make sure axes is iterable

        print(f"  Cluster {i}:")
        for j, sample in enumerate(sample_results):
            try:
                img = Image.open(sample['path'])
                axes[j].imshow(img)
                axes[j].set_title(
                    f"Label: {sample['label']}\nID: {os.path.basename(sample['path']).split('.')[0]}", fontsize=8)
                axes[j].axis('off')
                img.close()
            except Exception as e:
                axes[j].set_title(
                    f"Error loading\n{os.path.basename(sample['path']).split('.')[0]}")
                axes[j].axis('off')

        plt.suptitle(f"Representative Images - Cluster {i}", y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(
            output_dir, f"cluster_{i}_representatives.png"), bbox_inches='tight')
        plt.close(fig)


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Image Analysis for Fake News Detection ---")

    # Construct cache filename based on limit
    limit_str = "all" if NUM_SAMPLES_TO_PROCESS is None else str(
        NUM_SAMPLES_TO_PROCESS)
    cache_filename = f"analysis_results_{limit_str}.pkl"
    cache_filepath = os.path.join(RESULTS_CACHE_DIR, cache_filename)

    # 1. Load Data Paths and Labels
    print("\n--- 1. Loading Data ---")
    try:
        loader = PostDataLoader(POSTS_FILE_PATH, IMAGES_DIR_PATH)
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
                f"Successfully loaded cached results for {limit_str} samples.")
        except Exception as e:
            print(f"Error loading cached results: {e}. Recomputing...")
            analysis_results = None  # Ensure recomputation if loading fails

    if analysis_results is None:
        # 2. Setup Feature Extractor (only if needed)
        print("\n--- 2. Setting up Feature Extractor ---")
        try:
            feature_extractor, preprocess, device = setup_feature_extractor()
        except Exception as e:
            print(f"Error setting up feature extractor: {e}")
            exit()

        # 3. Perform Analysis (Feature Extraction)
        print("\n--- 3. Analyzing Images ---")
        analysis_results = analyze_images(
            loader, feature_extractor, preprocess, device, limit=NUM_SAMPLES_TO_PROCESS)

        # Save results if analysis was successful
        if analysis_results:
            print(f"\n--- Saving analysis results to {cache_filepath} ---")
            try:
                with open(cache_filepath, 'wb') as f:
                    pickle.dump(analysis_results, f)
                print("Successfully saved results.")
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

        # 4.3.1 Dominant Color Distribution (New)
        visualize_dominant_color_distribution(
             analysis_results, output_dir=OUTPUT_DIR)

        # 4.4 Dimensionality Reduction
        reduction_method = 'umap' if HAS_UMAP else 'tsne'
        embedding, labels_for_clustering = visualize_dimensionality_reduction(
            analysis_results, method=reduction_method, output_dir=OUTPUT_DIR)

        # 4.5 Clustering (based on dimensionality reduction result)
        if embedding is not None:
            visualize_clustering(embedding, labels_for_clustering,
                                 analysis_results, n_clusters=CLUSTER_N, output_dir=OUTPUT_DIR)
        else:
            print(
                "\nSkipping clustering because dimensionality reduction failed or produced no results.")

    else:
        print("\nNo analysis results generated, skipping visualization.")

    print("\n--- Analysis Complete ---")
    print(f"Check the '{OUTPUT_DIR}' directory for saved plots.")
    if os.path.exists(cache_filepath):
        print(f"Analysis results cached at: {cache_filepath}")
