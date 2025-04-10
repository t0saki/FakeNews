import csv
import os
from PIL import Image, UnidentifiedImageError


class PostDataLoader:
    """
    Loads and processes post data from a TSV file, extracting image paths and labels,
    and provides functionality to load the actual images.

    Assumes the input file has a header row and is tab-separated.
    Relevant columns are 'image_id(s)' and 'label'.
    Images are assumed to be in JPG format.
    """

    def __init__(self, file_path, image_dir):
        """
        Initializes the loader with the path to the data file and the image directory.

        Args:
            file_path (str): The path to the posts TSV file.
            image_dir (str): The path to the directory containing the images.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at: {file_path}")
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(
                f"Image directory not found at: {image_dir}")
        self.file_path = file_path
        self.image_dir = image_dir
        self.image_path_label_data = []  # Stores tuples of (image_path, label)

    def load_data(self):
        """
        Reads the TSV file, parses image IDs, constructs full image paths, and stores paths with labels.

        Handles posts with multiple image IDs by creating separate entries for each.
        Populates the self.image_path_label_data list.
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')

                try:
                    header = next(reader)  # Read the header row
                except StopIteration:
                    print("Warning: File is empty.")
                    return  # File is empty

                # Find column indices dynamically
                try:
                    image_id_col = header.index('image_id(s)')
                    label_col = header.index('label')
                except ValueError as e:
                    print(f"Error: Missing required column in header - {e}")
                    return

                self.image_path_label_data = []  # Reset data before loading
                for i, row in enumerate(reader):
                    # Basic check for row length consistency
                    if len(row) <= max(image_id_col, label_col):
                        print(f"Warning: Skipping malformed row {i+2}: {row}")
                        continue

                    image_ids_str = row[image_id_col]
                    label = row[label_col]

                    # Handle potential empty image_id field or label
                    if not image_ids_str or not label:
                        # Optionally log or skip rows with missing essential data
                        # print(f"Warning: Skipping row {i+2} due to missing image_id(s) or label.")
                        continue

                    # Split multiple image IDs and create entries for each
                    image_ids = image_ids_str.split(',')
                    for image_id in image_ids:
                        cleaned_image_id = image_id.strip()
                        if cleaned_image_id:  # Ensure stripped ID is not empty
                            # Construct the full image path
                            # Assumes images are JPEGs, adjust if necessary
                            image_filename = f"{cleaned_image_id}.jpg"
                            image_path = os.path.join(
                                self.image_dir, image_filename)

                            # Optionally check if the image file actually exists here
                            # if not os.path.exists(image_path):
                            #     print(f"Warning: Image file not found for ID {cleaned_image_id} at {image_path}. Skipping.")
                            #     continue

                            self.image_path_label_data.append(
                                (image_path, label))

        except FileNotFoundError:
            # This case is handled in __init__, but kept for robustness
            print(f"Error: File not found at {self.file_path}")
        except Exception as e:
            print(f"An unexpected error occurred while reading the file: {e}")

    def get_path_label_pairs(self):
        """
        Returns the loaded list of (image_path, label) tuples.

        Returns:
            list[tuple[str, str]]: A list of tuples, where each tuple 
                                   contains a full image path and its corresponding label.
        """
        return self.image_path_label_data

    def load_image_label_pairs(self, limit=None):
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
        count = 0
        for image_path, label in self.image_path_label_data:
            if limit is not None and count >= limit:
                break
            try:
                img = Image.open(image_path)
                # Keep the image object open, the user might want to process it further
                # Ensure the image data is loaded if needed immediately
                # img.load() # Uncomment if you need to load pixel data right away
                loaded_pairs.append((img, label))
                count += 1
            except FileNotFoundError:
                print(
                    f"Warning: Image file not found: {image_path}. Skipping.")
            except UnidentifiedImageError:  # Catch Pillow's error for corrupt/unsupported images
                print(
                    f"Warning: Could not identify or open image file: {image_path}. Skipping.")
            except Exception as e:
                print(
                    f"Warning: An error occurred loading image {image_path}: {e}. Skipping.")
        return loaded_pairs

# --- Example Usage ---
# Assuming your posts.txt is in 'data/twitter_dataset/devset/'
# and images are in 'data/twitter_dataset/devset/images/' relative to your script


# Construct the full path relative to the workspace root
if __name__ == "__main__":
    base_dir = 'data/twitter_dataset/devset'
    posts_file_path = os.path.join(base_dir, 'posts.txt')
    images_dir_path = os.path.join(base_dir, 'images')

    try:
        # Create an instance of the loader
        loader = PostDataLoader(posts_file_path, images_dir_path)

        # Load the data (paths and labels)
        loader.load_data()

        # Get the path-label pairs
        path_label_pairs = loader.get_path_label_pairs()
        print(
            f"Successfully found {len(path_label_pairs)} image path-label pairs.")
        if path_label_pairs:
            print("First 5 path-label pairs:", path_label_pairs[:5])

        # Load the actual images and labels (e.g., load the first 10)
        print("Loading first 10 images...")
        image_label_pairs = loader.load_image_label_pairs(limit=10)
        print(
            f"Successfully loaded {len(image_label_pairs)} actual image-label pairs.")

        # Example: Accessing the first loaded image and its label
        if image_label_pairs:
            first_image, first_label = image_label_pairs[0]
            print(f"First loaded image details:")
            print(f"  Label: {first_label}")
            print(f"  Image object: {first_image}")
            print(f"  Image format: {first_image.format}")
            print(f"  Image size: {first_image.size}")
            print(f"  Image mode: {first_image.mode}")
            # Remember to close the image file if you are done with it
            # especially in loops processing many images
            first_image.close()

    except FileNotFoundError as e:
        print(e)
    except ImportError:
        print(
            "Error: Pillow library not found. Please install it using 'pip install Pillow'")
    except Exception as e:
        print(f"An error occurred during execution: {e}")
