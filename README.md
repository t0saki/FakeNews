# Image Feature Analysis

This project performs various analyses on image datasets, extracting basic features, color information, and deep learning features using a pre-trained ResNet50 model. It includes dimensionality reduction (UMAP/t-SNE), clustering, and visualization capabilities, potentially aimed at analyzing images from social media datasets like Twitter or Weibo for characteristics related to misinformation or rumors.

## Features

*   **Data Loading:** Supports loading images from directory structures (e.g., rumor/non-rumor folders) or post lists (e.g., Twitter datasets). See `data_loader.py`.
*   **Basic Feature Extraction:** Calculates width, height, aspect ratio, and blurriness (Laplacian variance).
*   **Color Analysis:** Extracts dominant colors using K-Means clustering.
*   **Deep Feature Extraction:** Uses a pre-trained ResNet50 model (ImageNet weights) to extract deep features and perform classification predictions.
*   **Dimensionality Reduction:** Applies UMAP (preferred) or t-SNE to visualize high-dimensional feature spaces.
*   **Clustering:** Performs K-Means clustering on the reduced feature space.
*   **Visualization:** Generates various plots saved to an output directory:
    *   Sample images per class/label.
    *   Distributions of basic features (blurriness, aspect ratio).
    *   Dominant color palettes for sample images.
    *   Overall distribution of dominant colors.
    *   t-SNE/UMAP scatter plots colored by label and cluster.
    *   Distribution of predicted ImageNet classes.
    *   Analysis of label skew within top predicted classes.
*   **Caching:** Saves intermediate results (extracted features, embeddings) to speed up subsequent runs.
*   **Parallel Processing:** Utilizes multiple CPU cores for faster processing.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Install dependencies:**
    *   Python 3.x
    *   PyTorch & Torchvision
    *   OpenCV (`opencv-python`)
    *   Scikit-learn (`scikit-learn`)
    *   Matplotlib
    *   Seaborn
    *   UMAP (`umap-learn`) (Optional but recommended)
    *   Pandas
    *   tqdm
    *   Pillow

    You can typically install these using pip:
    ```bash
    pip install torch torchvision opencv-python scikit-learn matplotlib seaborn umap-learn pandas tqdm Pillow
    ```
    *(Note: Ensure PyTorch installation matches your system/CUDA setup. See official PyTorch instructions.)*

## Usage

1.  **Configure Data Source:** Edit the configuration section near the top of `image_analysis.py`:
    *   Choose `DATA_LOADER` (`DirectoryDataLoader` or `PostDataLoader`).
    *   Set `DATA_LOADER_ARGS` accordingly (paths to image directories or post files).
    *   Specify `OUTPUT_DIR_BASE` for saving results and `RESULTS_CACHE_DIR` for caching.
    *   Optionally set `NUM_SAMPLES_TO_PROCESS` to limit the number of images for faster testing.
2.  **Run Analysis:**
    ```bash
    python image_analysis.py
    ```
3.  **View Output:** Check the specified `OUTPUT_DIR_BASE` (within a subdirectory named after the sample limit, e.g., `analysis_output_weibo/all` or `analysis_output_weibo/1000`) for generated plots and analysis results. Cached features will be stored in `RESULTS_CACHE_DIR`.

## Project Structure

*   `image_analysis.py`: Main script for feature extraction, analysis, and visualization.
*   `data_loader.py`: Contains classes for loading image data.
*   `data/`: (Example) Directory containing datasets (e.g., `twitter_dataset`, `weibo_dataset`).
*   `analysis_output_<name>/`: Directory where generated plots and results are saved.
*   `analysis_cache_<name>/`: Directory where cached features and intermediate results are stored.
*   `MRML/`: (Potentially related) Multi-modal repository submodule. 