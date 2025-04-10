import csv
import os
from typing import List, Tuple
from PIL import Image, UnidentifiedImageError


class PostDataLoader:
    """
    Loads and processes post data from one or more TSV files, extracting image paths 
    and labels, and provides functionality to load the actual images.

    Assumes the input files have a header row and are tab-separated.
    Relevant columns are 'image_id(s)' and 'label'.
    Images are assumed to be in JPG format.
    """

    def __init__(self, file_paths: List[str], image_dirs: List[str]):
        """
        Initializes the loader with lists of paths to data files and image directories.

        Args:
            file_paths (List[str]): A list of paths to the posts TSV files.
            image_dirs (List[str]): A list of paths to the corresponding image directories.

        Raises:
            ValueError: If the lengths of file_paths and image_dirs do not match.
            FileNotFoundError: If any specified file or directory does not exist.
        """
        if len(file_paths) != len(image_dirs):
            raise ValueError(
                "The number of file paths must match the number of image directories.")
        if not file_paths:
            raise ValueError("At least one file path and image directory must be provided.")

        self.file_paths = []
        self.image_dirs = []
        for file_path, image_dir in zip(file_paths, image_dirs):
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found at: {file_path}")
            if not os.path.isdir(image_dir):
                raise FileNotFoundError(
                    f"Image directory not found at: {image_dir}")
            self.file_paths.append(file_path)
            self.image_dirs.append(image_dir)

        self.image_path_label_data: List[Tuple[str, str]] = []  # Stores tuples of (image_path, label)

    def load_data(self):
        """
        Reads the TSV files, parses image IDs, constructs full image paths for each corresponding 
        image directory, and stores all paths with labels.

        Handles posts with multiple image IDs by creating separate entries for each.
        Populates the self.image_path_label_data list.
        """
        self.image_path_label_data = []  # Reset data before loading

        for file_path, image_dir in zip(self.file_paths, self.image_dirs):
            print(f"Loading data from: {file_path} (Images: {image_dir})")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f, delimiter='\t')

                    try:
                        header = next(reader)  # Read the header row
                    except StopIteration:
                        print(f"Warning: File is empty: {file_path}")
                        continue  # Skip this file

                    # Find column indices dynamically
                    try:
                        image_id_col = header.index('image_id(s)')
                        label_col = header.index('label')
                    except ValueError as e:
                        try:
                            image_id_col = header.index('image_id')
                            label_col = header.index('label')
                        except ValueError as e:
                            print(
                                f"Error: Missing required column in header for {file_path} - {e}")
                            continue # Skip this file

                    processed_rows = 0
                    for i, row in enumerate(reader):
                        # Basic check for row length consistency
                        if len(row) <= max(image_id_col, label_col):
                            print(
                                f"Warning: Skipping malformed row {i+2} in {file_path}: {row}")
                            continue

                        image_ids_str = row[image_id_col]
                        label = row[label_col]

                        # Handle potential empty image_id field or label
                        if not image_ids_str or not label:
                            # print(f"Warning: Skipping row {i+2} in {file_path} due to missing image_id(s) or label.")
                            continue

                        # Split multiple image IDs and create entries for each
                        image_ids = image_ids_str.split(',')
                        for image_id in image_ids:
                            cleaned_image_id = image_id.strip()
                            if cleaned_image_id:  # Ensure stripped ID is not empty
                                # Construct the full image path using the *current* image_dir
                                image_filename = f"{cleaned_image_id}.jpg"
                                image_path = os.path.join(
                                    image_dir, image_filename) # Use the corresponding image_dir

                                # Optionally check if the image file actually exists here
                                # if not os.path.exists(image_path):
                                #     print(f"Warning: Image file not found for ID {cleaned_image_id} at {image_path}. Skipping.")
                                #     continue

                                self.image_path_label_data.append(
                                    (image_path, label))
                        processed_rows += 1
                    print(f"  Processed {processed_rows} rows from {file_path}.")

            except FileNotFoundError:
                # Should be caught by __init__, but good practice
                print(f"Error: File not found at {file_path} during loading.")
            except Exception as e:
                print(
                    f"An unexpected error occurred while reading {file_path}: {e}")
        
        # Deduplicate entries based on file paths
        original_count = len(self.image_path_label_data)
        self._deduplicate_entries()
        print(f"\nTotal image path-label pairs loaded: {len(self.image_path_label_data)} (Removed {original_count - len(self.image_path_label_data)} duplicates)")

    def _deduplicate_entries(self):
        """
        Removes duplicate entries from image_path_label_data based on file paths.
        If duplicate entries with different labels exist, keeps the first occurrence.
        """
        seen_paths = set()
        unique_entries = []
        
        for image_path, label in self.image_path_label_data:
            # Check if we've already seen this path
            if image_path not in seen_paths:
                seen_paths.add(image_path)
                unique_entries.append((image_path, label))
                
        self.image_path_label_data = unique_entries

    def get_path_label_pairs(self) -> List[Tuple[str, str]]:
        """
        Returns the loaded list of (image_path, label) tuples.

        Returns:
            list[tuple[str, str]]: A list of tuples, where each tuple 
                                   contains a full image path and its corresponding label.
        """
        return self.image_path_label_data

    def load_image_label_pairs(self, limit=None) -> List[Tuple[Image.Image, str]]:
        """
        Loads images from the stored paths and returns them with their labels.

        Args:
            limit (int, optional): Maximum number of image-label pairs to load. 
                                   Defaults to None (load all).

        Returns:
            list[tuple[PIL.Image.Image, str]]: A list of tuples, where each tuple contains
                                               a loaded PIL Image object and its label.
                                               Returns an empty list if data hasn't been loaded
                                               or if errors occur during image loading.
        """
        loaded_pairs = []
        # Use a slice if limit is specified to avoid iterating unnecessarily
        data_to_load = self.image_path_label_data[:limit] if limit is not None else self.image_path_label_data
        
        print(f"Attempting to load {len(data_to_load)} images...")
        loaded_count = 0
        skipped_count = 0

        for image_path, label in data_to_load:
            try:
                img = Image.open(image_path)
                # Keep the image object open, the user might want to process it further
                # Ensure the image data is loaded if needed immediately
                # img.load() # Uncomment if you need to load pixel data right away
                loaded_pairs.append((img, label))
                loaded_count += 1
            except FileNotFoundError:
                # print(f"Warning: Image file not found: {image_path}. Skipping.")
                skipped_count += 1
            except UnidentifiedImageError:  # Catch Pillow's error for corrupt/unsupported images
                # print(f"Warning: Could not identify or open image file: {image_path}. Skipping.")
                skipped_count += 1
            except Exception as e:
                print(
                    f"Warning: An error occurred loading image {image_path}: {e}. Skipping.")
                skipped_count += 1
        
        if skipped_count > 0:
            print(f"Warning: Skipped loading {skipped_count} images due to errors.")
        print(f"Successfully loaded {loaded_count} actual image-label pairs.")
        return loaded_pairs

# --- New Directory Data Loader Class ---
class DirectoryDataLoader:
    """
    Loads image paths and labels directly from specified directories,
    assuming subdirectories correspond to labels (e.g., 'rumor', 'non-rumor').

    Recursively searches for image files within the provided directories.
    Supported image types: .jpg, .jpeg, .png, .gif, .bmp
    """
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}

    def __init__(self, rumor_dirs: List[str], non_rumor_dirs: List[str]):
        """
        Initializes the loader with lists of paths to rumor and non-rumor image directories.

        Args:
            rumor_dirs (List[str]): A list of paths to directories containing rumor images.
            non_rumor_dirs (List[str]): A list of paths to directories containing non-rumor images.

        Raises:
            FileNotFoundError: If any specified directory does not exist.
            ValueError: If both rumor_dirs and non_rumor_dirs are empty.
        """
        self.rumor_dirs = []
        self.non_rumor_dirs = []

        if not rumor_dirs and not non_rumor_dirs:
            raise ValueError("At least one rumor or non-rumor directory must be provided.")

        for dir_path in rumor_dirs:
            if not os.path.isdir(dir_path):
                raise FileNotFoundError(f"Rumor directory not found: {dir_path}")
            self.rumor_dirs.append(dir_path)

        for dir_path in non_rumor_dirs:
            if not os.path.isdir(dir_path):
                raise FileNotFoundError(f"Non-rumor directory not found: {dir_path}")
            self.non_rumor_dirs.append(dir_path)

        self.image_path_label_data: List[Tuple[str, str]] = []

    def load_data(self):
        """
        Scans the specified directories recursively, identifies image files,
        and stores their paths along with assigned labels ('rumor' or 'non-rumor').

        Populates the self.image_path_label_data list.
        """
        self.image_path_label_data = [] # Reset data
        found_count = 0

        # Process rumor directories
        for dir_path in self.rumor_dirs:
            print(f"Scanning rumor directory: {dir_path}")
            count_in_dir = 0
            for root, _, files in os.walk(dir_path):
                for filename in files:
                    _, ext = os.path.splitext(filename)
                    if ext.lower() in self.SUPPORTED_EXTENSIONS:
                        image_path = os.path.join(root, filename)
                        self.image_path_label_data.append((image_path, 'rumor'))
                        count_in_dir += 1
            print(f"  Found {count_in_dir} images in {dir_path}")
            found_count += count_in_dir

        # Process non-rumor directories
        for dir_path in self.non_rumor_dirs:
            print(f"Scanning non-rumor directory: {dir_path}")
            count_in_dir = 0
            for root, _, files in os.walk(dir_path):
                for filename in files:
                    _, ext = os.path.splitext(filename)
                    if ext.lower() in self.SUPPORTED_EXTENSIONS:
                        image_path = os.path.join(root, filename)
                        self.image_path_label_data.append((image_path, 'non-rumor'))
                        count_in_dir += 1
            print(f"  Found {count_in_dir} images in {dir_path}")
            found_count += count_in_dir

        print(f"\nTotal image path-label pairs found: {len(self.image_path_label_data)}")
        # Consider adding shuffling here if needed:
        # import random
        # random.shuffle(self.image_path_label_data)

    def get_path_label_pairs(self) -> List[Tuple[str, str]]:
        """
        Returns the loaded list of (image_path, label) tuples.

        Returns:
            list[tuple[str, str]]: A list of tuples, where each tuple
                                   contains a full image path and its corresponding label ('rumor' or 'non-rumor').
        """
        return self.image_path_label_data

    def load_image_label_pairs(self, limit=None) -> List[Tuple[Image.Image, str]]:
        """
        Loads images from the stored paths and returns them with their labels.

        Args:
            limit (int, optional): Maximum number of image-label pairs to load.
                                   Defaults to None (load all).

        Returns:
            list[tuple[PIL.Image.Image, str]]: A list of tuples, where each tuple contains
                                               a loaded PIL Image object and its label.
                                               Returns an empty list if data hasn't been loaded
                                               or if errors occur during image loading.
        """
        # This method is identical in function to the one in PostDataLoader
        loaded_pairs = []
        data_to_load = self.image_path_label_data[:limit] if limit is not None else self.image_path_label_data

        print(f"Attempting to load {len(data_to_load)} images...")
        loaded_count = 0
        skipped_count = 0

        for image_path, label in data_to_load:
            try:
                img = Image.open(image_path)
                # img.load() # Uncomment if you need pixel data immediately
                loaded_pairs.append((img, label))
                loaded_count += 1
            except FileNotFoundError:
                # print(f"Warning: Image file not found: {image_path}. Skipping.")
                skipped_count += 1
            except UnidentifiedImageError:
                # print(f"Warning: Could not identify or open image file: {image_path}. Skipping.")
                skipped_count += 1
            except Exception as e:
                print(f"Warning: An error occurred loading image {image_path}: {e}. Skipping.")
                skipped_count += 1

        if skipped_count > 0:
            print(f"Warning: Skipped loading {skipped_count} images due to errors.")
        print(f"Successfully loaded {loaded_count} actual image-label pairs.")
        return loaded_pairs


# --- Example Usage ---

if __name__ == "__main__":

    # --- Example for PostDataLoader ---
    print("--- Testing PostDataLoader ---")
    # Example: Define multiple dataset paths
    base_dir_dev = 'data/twitter_dataset/devset'
    posts_file_dev = os.path.join(base_dir_dev, 'posts.txt')
    images_dir_dev = os.path.join(base_dir_dev, 'images')

    # Create lists of paths
    all_posts_files = [posts_file_dev]
    all_images_dirs = [images_dir_dev]

    try:
        loader = PostDataLoader(all_posts_files, all_images_dirs)
        loader.load_data()
        path_label_pairs = loader.get_path_label_pairs()
        print(f"\nPostDataLoader: Found {len(path_label_pairs)} path-label pairs.")
        if path_label_pairs:
            print("First 5:", path_label_pairs[:5])

        print("\nPostDataLoader: Loading first 5 images...")
        image_label_pairs = loader.load_image_label_pairs(limit=5)
        # Print details already handled within load_image_label_pairs

        if image_label_pairs:
            first_image, first_label = image_label_pairs[0]
            print(f"\nPostDataLoader: First loaded image details:")
            print(f"  Label: {first_label}")
            print(f"  Image object: {first_image}")
            first_image.close() # Close image after inspection

    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"\nError during PostDataLoader setup/run: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during PostDataLoader execution: {e}")


    # --- Example for DirectoryDataLoader ---
    print("\n\n--- Testing DirectoryDataLoader ---")
    # Define placeholder paths for rumor and non-rumor directories
    # *** Replace these with your actual directory paths ***
    # e.g., ['path/to/rumor_set1', 'path/to/rumor_set2']
    example_rumor_dirs = ['/mnt/d/LFDev-D/weibo_dataset/rumor_images']
    example_non_rumor_dirs = ['/mnt/d/LFDev-D/weibo_dataset/nonrumor_images']

    # Create dummy directories and files for the example if they don't exist
    # You might want to remove this part if you provide real paths
    for d in example_rumor_dirs + example_non_rumor_dirs:
        os.makedirs(d, exist_ok=True)
    # Create dummy image files (optional, for demonstration)
    try:
        Image.new('RGB', (60, 30), color = 'red').save(os.path.join(example_rumor_dirs[0], 'rumor1.jpg'))
        Image.new('RGB', (60, 30), color = 'green').save(os.path.join(example_non_rumor_dirs[0], 'nonrumor1.png'))
        Image.new('RGB', (60, 30), color = 'blue').save(os.path.join(example_non_rumor_dirs[0], 'nonrumor2.jpeg'))
        # Add a non-image file to test filtering
        with open(os.path.join(example_non_rumor_dirs[0], 'notes.txt'), 'w') as f:
            f.write("this is not an image")
    except ImportError:
         print("\nWarning: Pillow not installed, cannot create dummy images for DirectoryDataLoader example.")
    except Exception as e:
         print(f"\nWarning: Could not create dummy image files: {e}")


    try:
        dir_loader = DirectoryDataLoader(rumor_dirs=example_rumor_dirs, non_rumor_dirs=example_non_rumor_dirs)
        dir_loader.load_data() # Scan directories and collect paths/labels

        dir_path_label_pairs = dir_loader.get_path_label_pairs()
        print(f"\nDirectoryDataLoader: Found {len(dir_path_label_pairs)} total image path-label pairs.")
        if dir_path_label_pairs:
            print("First 5 path-label pairs:", dir_path_label_pairs[:5])

        print("\nDirectoryDataLoader: Loading first 5 images (or all if fewer)...")
        dir_image_label_pairs = dir_loader.load_image_label_pairs(limit=5)

        if dir_image_label_pairs:
            img, lbl = dir_image_label_pairs[0]
            print(f"\nDirectoryDataLoader: First loaded image details:")
            print(f"  Label: {lbl}")
            print(f"  Image object: {img}")
            print(f"  Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
            img.close() # Close image

    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"\nError during DirectoryDataLoader setup/run: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during DirectoryDataLoader execution: {e}")
