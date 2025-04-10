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
        
        print(f"\nTotal image path-label pairs loaded: {len(self.image_path_label_data)}")

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

# --- Example Usage ---
# Example structure assuming you have multiple sets, e.g., train and dev
# Replace with your actual paths

if __name__ == "__main__":
    # Example: Define multiple dataset paths
    base_dir_dev = 'data/twitter_dataset/devset'
    posts_file_dev = os.path.join(base_dir_dev, 'posts.txt')
    images_dir_dev = os.path.join(base_dir_dev, 'images')

    # Add another set (e.g., a training set if you had one)
    # base_dir_train = 'data/twitter_dataset/trainset' # Hypothetical
    # posts_file_train = os.path.join(base_dir_train, 'posts.txt')
    # images_dir_train = os.path.join(base_dir_train, 'images')

    # Create lists of paths
    all_posts_files = [posts_file_dev]  # Add posts_file_train here if needed
    all_images_dirs = [images_dir_dev]  # Add images_dir_train here if needed

    try:
        # Create an instance of the loader with the lists
        loader = PostDataLoader(all_posts_files, all_images_dirs)

        # Load the data (paths and labels) from all specified sources
        loader.load_data()

        # Get the aggregated path-label pairs
        path_label_pairs = loader.get_path_label_pairs()
        print(
            f"\nSuccessfully found {len(path_label_pairs)} total image path-label pairs.")
        if path_label_pairs:
            print("First 5 path-label pairs:", path_label_pairs[:5])
            print("Last 5 path-label pairs:", path_label_pairs[-5:])

        # Load the actual images and labels (e.g., load the first 10 from the aggregated list)
        print("\nLoading first 10 images from the combined dataset...")
        image_label_pairs = loader.load_image_label_pairs(limit=10)
        # Print details already handled within load_image_label_pairs

        # Example: Accessing the first loaded image and its label
        if image_label_pairs:
            first_image, first_label = image_label_pairs[0]
            print(f"\nDetails of the first loaded image:")
            print(f"  Label: {first_label}")
            print(f"  Image object: {first_image}")
            print(f"  Image format: {first_image.format}")
            print(f"  Image size: {first_image.size}")
            print(f"  Image mode: {first_image.mode}")
            # Remember to close the image file if you are done with it
            first_image.close()

    except FileNotFoundError as e:
        print(f"\nError during setup: {e}")
    except ValueError as e:
        print(f"\nError during setup: {e}")
    except ImportError:
        print(
            "\nError: Pillow library not found. Please install it using 'pip install Pillow'")
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
