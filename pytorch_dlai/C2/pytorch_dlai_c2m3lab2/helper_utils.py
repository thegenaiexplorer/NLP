import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.lines import Line2D
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertModel



def download_glove6B(extract_directory="./glove_data"):
    """
    Downloads and extracts the GloVe 6B word embedding dataset.

    Args:
        extract_directory: The path to the directory where the GloVe
                         data will be stored.
    """
    
    # Define the path to a sample file to check for existence
    sample_file_path = os.path.join(extract_directory, 'glove.6B.100d.txt')
    
    # Check if the data already exists to avoid re-downloading
    if os.path.exists(sample_file_path):
        print(f"GloVe data already found in '{extract_directory}'. Skipping download.")
        return

    print(f"GloVe data not found. Starting download...")
    # Create the extraction directory if it does not exist
    os.makedirs(extract_directory, exist_ok=True)
    
    # URL for the GloVe 6B dataset
    url = "https://nlp.stanford.edu/data/glove.6B.zip"
    # Path to save the temporary zip file
    zip_file_path = "glove.6B.zip"
    
    # Use a try-finally block to ensure the temporary zip file is removed
    try:
        # Stream the download to handle large files and show progress
        response = requests.get(url, stream=True)
        # Raise an exception for bad status codes (like 404)
        response.raise_for_status()
        # Get the total file size from the response headers
        total_size = int(response.headers.get('content-length', 0))
        
        # Initialize tqdm for a progress bar during download
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading glove.6B.zip") as pbar:
            # Open the local file in binary write mode
            with open(zip_file_path, "wb") as f:
                # Iterate over the response content in chunks
                for chunk in response.iter_content(chunk_size=8192):
                    # Update the progress bar
                    pbar.update(len(chunk))
                    # Write the current chunk to the file
                    f.write(chunk)
        
        print(f"\nUnzipping '{zip_file_path}'...")
        # Open the downloaded zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Extract all its contents to the specified directory
            zip_ref.extractall(extract_directory)
        print(f"Files extracted to '{extract_directory}'.")
    
    finally:
        # Clean up by removing the downloaded zip file
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)
            print(f"Removed temporary file '{zip_file_path}'.")


            
def load_glove_embeddings(file_path):
    """
    Loads GloVe word embeddings from a specified file.

    Args:
        file_path: The path to the GloVe embeddings text file.

    Returns:
        A dictionary mapping each word to its NumPy vector representation.
    """
    
    # Initialize an empty dictionary to store the embeddings
    embeddings_index = {}
    
    # Open the embeddings file for reading with UTF-8 encoding
    with open(file_path, 'r', encoding='utf-8') as f:
        # Iterate over each line in the file
        for line in f:
            # Split the line into a list of values
            values = line.split()
            # The first value is the word
            word = values[0]
            # The remaining values are the vector components
            vector = np.asarray(values[1:], dtype='float32')
            # Add the word and its vector to the dictionary
            embeddings_index[word] = vector
            
    # Return the dictionary containing the loaded embeddings
    return embeddings_index



def plot_embeddings(coords, labels, label_dict, title):
    """
    Visualizes 2D word embeddings using a scatter plot.

    This function plots a set of 2D coordinates, annotates each point with
    a corresponding label, and colors the points based on predefined
    categories. It dynamically assigns colors to categories for clear
    visual distinction.

    Args:
        coords: A 2D numpy array where each row represents the x, y
                coordinates of a point.
        labels: A list of string labels, one for each point in `coords`.
        label_dict: A dictionary that maps category names to lists of
                    words belonging to that category.
        title: The title for the plot.
    """
    # Use a try-except block to handle potential inconsistencies
    # between the labels and the dictionary.
    try:
        # Create a reverse mapping from words to categories for efficient lookup.
        word_to_category = {word: category for category, words in label_dict.items() for word in words}

        # Validate that every label has a corresponding category in the map.
        # This will raise a KeyError if a word is not found.
        for word in labels:
            _ = word_to_category[word]

        # --- Dynamic Color Assignment ---
        # Define a base list of visually distinct colors for consistency.
        fixed_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Use a set for efficient tracking of assigned colors to prevent duplicates.
        used_colors = set(fixed_colors)
        
        # Get a list of all unique category names.
        unique_categories = list(label_dict.keys())
        # Initialize a dictionary to map each category to a unique color.
        category_to_color = {}

        # Iterate through each unique category to assign a color.
        for i, category in enumerate(unique_categories):
            # Assign colors from the predefined list first.
            if i < len(fixed_colors):
                category_to_color[category] = fixed_colors[i]
            else:
                # Generate a unique random color for any additional categories.
                random_color = None
                while random_color is None or random_color in used_colors:
                    # Generate a random hex color string (e.g., '#a1c3f7').
                    r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                    random_color = f'#{r:02x}{g:02x}{b:02x}'
                
                # Assign the new unique color to the category.
                category_to_color[category] = random_color
                # Add the new color to the set of used colors.
                used_colors.add(random_color)

        # --- Plotting and Annotation ---
        # Set up the figure size for the plot.
        plt.figure(figsize=(8, 8))

        # Plot points for each category to associate them with the correct legend entry.
        for category in unique_categories:
            # Find the indices of words belonging to the current category.
            indices = [i for i, word in enumerate(labels) if word_to_category[word] == category]
            # Get the coordinates for the current category using the indices.
            category_coords = coords[indices]
            
            # Create a scatter plot for the current category's points.
            plt.scatter(
                category_coords[:, 0],
                category_coords[:, 1],
                color=category_to_color[category],
                s=120,
                alpha=0.9,
                label=category
            )

        # Add text annotations for each individual data point.
        for i, word in enumerate(labels):
            plt.annotate(word, (coords[i, 0], coords[i, 1]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=12, fontweight='bold')

        # --- Final Plot Adjustments ---
        # Set the plot title and axis labels.
        plt.title(title, fontsize=16)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        # Add a grid for better readability.
        plt.grid(True, alpha=0.3)
        # Display the legend to show category-color mappings.
        plt.legend(loc='upper right', title="Categories")
        # Adjust plot to ensure everything fits without overlapping.
        plt.tight_layout()
        # Display the final visualization.
        plt.show()

    except KeyError as e:
        # Handle cases where a label is not found in the categorization dictionary.
        print(f"Error: Word {e} from the label list was not found in the dictionary.")
        print("Please ensure that the label list and the categorization dictionary are consistent.")
    
    
    
def plot_loss(losses):
    """
    Plots the training loss over a series of epochs.

    Args:
        losses: A list or array of loss values, where each value
                corresponds to an epoch.
    """
    
    # Create a new figure with a specific size for the plot.
    plt.figure(figsize=(8, 8))
    # Plot the loss values against the epoch number.
    plt.plot(losses)
    # Set the title and labels for clarity.
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # Add a grid to make the plot easier to read.
    plt.grid(True)
    # Display the plot.
    plt.show()
    
    
    
def plot_similarity_matrix(matrix, labels):
    """
    Visualizes a word similarity matrix as a heatmap.

    This function takes a square matrix of similarity scores and a list of
    labels to generate a color-coded heatmap, making it easy to see the
    relationships between different words.

    Args:
        matrix: A square 2D numpy array containing the similarity scores.
        labels: A list of string labels for the matrix axes.
    """
    # Create a new figure with a specific size for the plot.
    plt.figure(figsize=(10, 8))
    # Display the data as an image, where each cell's color corresponds to its value.
    # The 'bwr' colormap (blue-white-red) is great for showing similarity.
    plt.imshow(matrix, cmap='bwr')
    # Add a color bar to serve as a legend for the values.
    plt.colorbar(label='Cosine Similarity')
    # Set the chart's title and axis labels.
    plt.title('Word Similarity Matrix')
    plt.xlabel('Words')
    plt.ylabel('Words')
    # Set the tick marks on both axes to correspond to the words.
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    # Adjust the plot to ensure everything fits without overlapping.
    plt.tight_layout()
    # Display the final plot.
    plt.show()
    
    

def download_bert(model_name='bert-base-uncased', save_directory='./bert_model'):
    """
    Downloads and saves a pre-trained BERT model and its tokenizer.

    Args:
        model_name: The identifier for the pre-trained BERT model
                    (e.g., 'bert-base-uncased').
        save_directory: The path to the directory where the model and
                        tokenizer will be saved.
    """
    # Check if the save directory already exists to avoid re-downloading.
    if os.path.exists(save_directory):
        print(f"BERT model already found in '{save_directory}'. Skipping download.")
        return
    
    # If the directory doesn't exist, download the model and tokenizer.
    print(f"Downloading BERT model to '{save_directory}'...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    # Save the tokenizer and model to the specified local directory.
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
    
    print("Download complete.")
    
    
    
def load_bert(save_directory='./bert_model'):
    """
    Loads a pre-trained BERT model and its tokenizer from a local directory.

    Args:
        save_directory: The path to the directory where the model and
                        tokenizer files are stored.

    Returns:
        A tuple containing the loaded tokenizer and model objects.
    """
    print(f"Loading BERT model from '{save_directory}'...")
    # Load the tokenizer and model from the specified local path.
    tokenizer = BertTokenizer.from_pretrained(save_directory)
    model = BertModel.from_pretrained(save_directory)
    print("Model loaded successfully.")
    
    return tokenizer, model