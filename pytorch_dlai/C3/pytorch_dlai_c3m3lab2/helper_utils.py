import os
import shutil
import tarfile
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm


def extract_imdb_data(data_dir='./imdb_data'):
    """
    Extracts the IMDB dataset from a zip archive if the destination directory does not exist.

    Args:
        data_dir (str): The local directory path where the dataset should be extracted.

    Returns:
        status (str): A message indicating whether the data was extracted or already exists.
    """
    # Check if the extracted data directory is already present
    if not os.path.exists(data_dir):
        # Define the path for the expected zip file based on the directory name
        zip_path = f"{data_dir}.zip"
        
        # Verify the existence of the compressed archive
        if os.path.exists(zip_path):
            print(f"{data_dir} not found. Extracting {zip_path}...")
            
            # Access the zip file for extraction
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Retrieve all members to calculate progress
                members = zip_ref.infolist()
                
                # Extract files while displaying a progress bar
                for member in tqdm(members, desc="Unzipping", unit="file"):
                    zip_ref.extract(member)
            
            return "Extraction complete!"
        else:
            # Return an error message if the source zip is missing
            return f"Error: The file '{zip_path}' is missing."
    
    return "Data directory already exists."



def get_imdb_data(data_dir='./imdb_data', max_train_samples=2000, max_test_samples=500):
    """
    Loads text reviews and labels from the IMDB dataset directory.

    Args:
        data_dir (str): The directory where the extracted dataset is located.
        max_train_samples (int): Total number of training samples to load.
        max_test_samples (int): Total number of test samples to load.

    Returns:
        train_reviews (list): A list of strings containing the training text data.
        train_labels (list): A list of integers representing training sentiment labels.
        test_reviews (list): A list of strings containing the test text data.
        test_labels (list): A list of integers representing test sentiment labels.
    """
    # Initialize containers for reviews and corresponding labels
    train_reviews, train_labels = [], []
    test_reviews, test_labels = [], []

    # Define a helper function to read files from a directory
    def load_files(directory, label, limit):
        """
        Reads a subset of text files from a specific directory.

        Args:
            directory (str): The path to the folder containing the text files.
            label (int): The sentiment label (1 or 0) for these files.
            limit (int): The maximum number of files to read.

        Returns:
            reviews (list): A list of strings containing the file contents.
            labels (list): A list of integers containing the labels.
        """
        # Select filenames up to the specified limit
        files = os.listdir(directory)[:limit]
        reviews, labels = [], []
        
        # Create a descriptive string for the progress bar
        desc = f"Loading {os.path.basename(os.path.dirname(directory))} {os.path.basename(directory)}"
        
        # Iterate through files and append content to lists
        for filename in tqdm(files, desc=desc, leave=False):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                reviews.append(f.read())
                labels.append(label)
        return reviews, labels

    # Load and aggregate training samples
    print("\nProcessing Training Data...")
    r, l = load_files(os.path.join(data_dir, 'train', 'pos'), 1, max_train_samples // 2)
    train_reviews.extend(r); train_labels.extend(l)
    r, l = load_files(os.path.join(data_dir, 'train', 'neg'), 0, max_train_samples // 2)
    train_reviews.extend(r); train_labels.extend(l)

    # Load and aggregate test samples
    print("Processing Test Data...")
    r, l = load_files(os.path.join(data_dir, 'test', 'pos'), 1, max_test_samples // 2)
    test_reviews.extend(r); test_labels.extend(l)
    r, l = load_files(os.path.join(data_dir, 'test', 'neg'), 0, max_test_samples // 2)
    test_reviews.extend(r); test_labels.extend(l)

    print(f"\nSuccessfully loaded {len(train_reviews)} training and {len(test_reviews)} test samples.")
    
    # Return all processed datasets and labels
    return train_reviews, train_labels, test_reviews, test_labels



def print_data_statistics(train_reviews, train_labels, test_reviews, test_labels, sample_size=100):
    """
    Analyzes and displays descriptive statistics for the provided sentiment dataset.

    Args:
        train_reviews: List of strings containing training review text.
        train_labels: List of integers representing training sentiment labels.
        test_reviews: List of strings containing test review text.
        test_labels: List of integers representing test sentiment labels.
        sample_size: Number of instances to consider for length calculations.

    Returns:
        None. This function outputs statistics directly to the console.
    """
    # Print the header and volume counts for the training portion of the dataset
    print("\n=== Dataset Statistics ===")
    print(f"Training set:")
    print(f"  Total reviews: {len(train_reviews)}")
    # Sum binary labels to determine the total count of positive samples
    print(f"  Positive reviews: {sum(train_labels)}")
    # Subtract positive count from total length to find negative sample count
    print(f"  Negative reviews: {len(train_labels) - sum(train_labels)}")
    
    # Print volume counts for the testing portion of the dataset
    print(f"\nTest set:")
    print(f"  Total reviews: {len(test_reviews)}")
    print(f"  Positive reviews: {sum(test_labels)}")
    print(f"  Negative reviews: {len(test_labels) - sum(test_labels)}")
    
    # Begin displaying representative text samples from the dataset
    print("\n=== Sample Reviews ===")
    
    # Locate and display the first available positive review from the training set
    try:
        # Find the index of the first occurrence of the positive label
        positive_idx = train_labels.index(1)
        print("\n--- Positive Review Example ---")
        print(f"Label: Positive")
        # Display a truncated version of the review text
        print(f"Text (first 400 chars): {train_reviews[positive_idx][:400]}...")
    except ValueError:
        # Handle cases where no positive labels are present in the list
        print("\nNo positive reviews found in training set")
    
    # Locate and display the first available negative review from the training set
    try:
        # Find the index of the first occurrence of the negative label
        negative_idx = train_labels.index(0)
        print("\n--- Negative Review Example ---")
        print(f"Label: Negative")
        # Display a truncated version of the review text
        print(f"Text (first 400 chars): {train_reviews[negative_idx][:400]}...")
    except ValueError:
        # Handle cases where no negative labels are present in the list
        print("\nNo negative reviews found in training set")
    
    # Ensure the requested sample size does not exceed the total available reviews
    sample_size = min(sample_size, len(train_reviews))
    # Calculate word counts for each review by splitting on whitespace
    review_lengths = [len(review.split()) for review in train_reviews[:sample_size]]
    
    # Output descriptive statistics for the review lengths
    print(f"\n=== Review Length Statistics (first {sample_size} reviews) ===")
    # Calculate the arithmetic mean of the review lengths
    print(f"Average length: {np.mean(review_lengths):.0f} words")
    # Identify the shortest review in the sample
    print(f"Min length: {min(review_lengths)} words")
    # Identify the longest review in the sample
    print(f"Max length: {max(review_lengths)} words")
    # Calculate the median length for the sample
    print(f"Median length: {np.median(review_lengths):.0f} words")
    # Calculate the standard deviation of lengths to understand variability
    print(f"Std deviation: {np.std(review_lengths):.0f} words")



def print_summary(model, vocab_size=None, embedding_dim=128, num_heads=None):
    """
    Displays a detailed overview of the model architecture and its configuration.

    Args:
        model: The neural network module to be summarized.
        vocab_size: The total count of unique tokens in the vocabulary.
        embedding_dim: The dimensionality of the vector space for embeddings.
        num_heads: The count of parallel attention heads used in the architecture.

    Returns:
        None. This function outputs the summary information directly to the console.
    """
    # Determine the most efficient hardware available for computation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device available: {device}")
    
    # Transfer all model parameters and buffers to the identified device
    model = model.to(device)
    print(f"\nModel moved to {device}")
    print("Ready for training!")
    
    # Attempt to determine the vocabulary size from the model's internal layers if not explicitly provided
    if vocab_size is None:
        try:
            # Iterate through all sub-modules to locate the embedding layer
            for module in model.modules():
                if isinstance(module, torch.nn.Embedding):
                    # Retrieve the number of embeddings from the specific layer
                    vocab_size = module.num_embeddings
                    break
        except:
            # Revert to a fallback value if retrieval fails
            vocab_size = "Unknown"

    # Attempt to determine the number of attention heads from the model layers if not provided
    if num_heads is None:
        try:
            # Iterate through sub-modules to find MultiheadAttention instances
            for module in model.modules():
                if isinstance(module, torch.nn.MultiheadAttention):
                    # Extract the head count from the layer attributes
                    num_heads = module.num_heads
                    break
        except:
            # Set to unknown if the head count cannot be inferred
            num_heads = "Unknown"
    
    # Identify the specific class name of the provided model instance
    model_name = model.__class__.__name__
    
    # Compute the total count of elements across all parameter tensors
    total_params = sum(p.numel() for p in model.parameters())
    
    # Output the structured model specification report
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Vocabulary size: {vocab_size if vocab_size else 'Unknown'}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Number of attention heads: {num_heads}")
    print(f"Total parameters: {total_params:,}")
    print("\nThe model is now ready to be trained!")


def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs=5, device=None):
    """
    Executes the training and validation loop for a machine learning model.

    Args:
        model: The neural network model to be trained.
        train_loader: DataLoader providing the training dataset batches.
        test_loader: DataLoader providing the validation or test dataset batches.
        optimizer: The optimization algorithm used for weight updates.
        criterion: The loss function used to evaluate model performance.
        num_epochs: Total number of iterations over the training dataset.
        device: The computation device to use; defaults to CUDA if available.

    Returns:
        history: A dictionary containing lists of accuracies and losses for 
                 both training and testing, as well as the best model's 
                 state and epoch information.
    """
    # Initialize the computation device if one is not explicitly provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Transfer the model parameters to the designated computation device
    model = model.to(device)
    
    # Initialize containers for tracking performance metrics throughout training
    history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}
    
    # Initialize variables to track the best performance and model state
    best_test_acc = 0.0
    best_epoch = 0
    best_model_state = None
    
    print(f"Training on {device}")
    print("="*50)
    
    for epoch in range(num_epochs):
        # Set the model to training mode to enable dropout and batch normalization
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0
        
        # Initialize the progress bar for the training phase
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_bar:
            # Transfer input and label tensors to the active device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Reset gradients and perform the forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Execute backpropagation and update model parameters
            loss.backward()
            optimizer.step()
            
            # Determine predictions and update running accuracy and loss totals
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item() * labels.size(0)
            
            # Calculate metrics and update the progress bar display
            train_acc = 100 * train_correct / train_total
            avg_loss = train_loss / train_total
            train_bar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{train_acc:.1f}%'})
        
        # Set the model to evaluation mode to disable training-specific layers
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        
        # Initialize the progress bar for the testing/validation phase
        test_bar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]')
        # Disable gradient calculations to reduce memory usage and increase speed
        with torch.no_grad():
            for inputs, labels in test_bar:
                # Move tensors to the active device
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Perform inference and calculate the loss
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Track correct predictions and cumulative loss
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                test_loss += loss.item() * labels.size(0)
                
                # Update metrics and progress bar status
                test_acc = 100 * test_correct / test_total
                test_bar.set_postfix({'acc': f'{test_acc:.1f}%'})
        
        # Finalize accuracy and loss calculations for the current epoch
        train_accuracy = 100 * train_correct / train_total
        test_accuracy = 100 * test_correct / test_total
        avg_train_loss = train_loss / train_total
        avg_test_loss = test_loss / test_total
        
        # Append epoch results to the history container
        history['train_acc'].append(train_accuracy)
        history['test_acc'].append(test_accuracy)
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        
        # Determine if the current epoch produced the best results based on test accuracy
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_epoch = epoch + 1
            # Capture the state of the model and optimizer at the peak performance
            best_model_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict().copy(),
                'optimizer_state_dict': optimizer.state_dict().copy(),
                'test_accuracy': test_accuracy,
                'train_accuracy': train_accuracy,
                'test_loss': avg_test_loss,
                'train_loss': avg_train_loss
            }
            print(f'  🎯 New best model! Test Acc: {test_accuracy:.2f}%')
        
        # Output a summary of metrics for the current epoch
        print(f'Epoch {epoch+1}/{num_epochs} Summary:')
        print(f'  Train - Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}%')
        print(f'  Test   - Loss: {avg_test_loss:.4f}, Acc: {test_accuracy:.2f}%')
        print('-'*50)
    
    # Restore model parameters from the best performing epoch identified during training
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
        print("\n" + "="*50)
        print(f"Training completed! Best model restored from epoch {best_epoch}")
        print(f"Best Test Accuracy: {best_test_acc:.2f}%")
    else:
        print("\nTraining completed!")
        print(f"Final Test Accuracy: {history['test_acc'][-1]:.2f}%")
    
    # Store the top performance metadata in the history dictionary
    history['best_epoch'] = best_epoch
    history['best_test_acc'] = best_test_acc
    history['best_model_state'] = best_model_state
    
    # Return the dictionary containing training logs and best model information
    return history



def plot_training_history(history, initial_accuracy=50.0):
    """
    Generates a visualization of model performance metrics over the training duration.

    Args:
        history: A dictionary containing 'train_acc' and 'test_acc' lists.
        initial_accuracy: The baseline accuracy value to be used as a reference point.

    Returns:
        None. This function displays a plot and prints a summary to the console.
    """
    # Initialize the figure for plotting performance curves
    plt.figure(figsize=(8, 5))
    # Define the range of epochs based on the length of the training history
    epochs = range(1, len(history['train_acc']) + 1)
    
    # Plot the training accuracy values over the defined epochs
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    # Plot the testing accuracy values over the defined epochs
    plt.plot(epochs, history['test_acc'], 'r-', label='Test Accuracy')
    # Draw a horizontal reference line representing the starting accuracy level
    plt.axhline(y=initial_accuracy, color='gray', linestyle='--', label='Initial Accuracy')
    
    # Configure the horizontal axis label as the training epoch
    plt.xlabel('Epoch')
    # Configure the vertical axis label as the accuracy percentage
    plt.ylabel('Accuracy (%)')
    # Set the main title for the performance visualization
    plt.title('Model Training Progress')
    # Add a legend to distinguish between the different data series
    plt.legend()
    # Enable a background grid with partial transparency for better readability
    plt.grid(True, alpha=0.3)
    # Adjust the layout to ensure all elements fit within the figure boundaries
    plt.tight_layout()
    # Render the visualization to the display
    plt.show()
    
    # Output a structured text summary of the performance metrics
    print("Accuracy Summary:")
    # Display the starting accuracy value provided to the function
    print(f"  Started at: {initial_accuracy:.2f}% (untrained)")
    # Display the final accuracy value recorded in the testing history
    print(f"  Final test accuracy: {history['test_acc'][-1]:.2f}%")
    # Calculate and display the difference between the final and initial accuracy
    print(f"  Total improvement: +{history['test_acc'][-1] - initial_accuracy:.2f}%")



def compare_models(history1, history2,
                   model1_name="Custom Model",
                   model2_name="PyTorch Model",
                   model1_desc="Custom implementation",
                   model2_desc="Built-in implementation",
                   figsize=(14, 5)):
    """
    Compares the training and testing performance of two models side by side.

    Args:
        history1: Dictionary containing training and testing accuracy for the first model.
        history2: Dictionary containing training and testing accuracy for the second model.
        model1_name: String label for the first model.
        model2_name: String label for the second model.
        model1_desc: Brief architectural description of the first model.
        model2_desc: Brief architectural description of the second model.
        figsize: Tuple defining the dimensions of the comparison plot.

    Returns:
        None. This function displays a comparative plot and prints a summary report.
    """
    # Calculate the number of completed epochs for both training histories
    epochs1 = len(history1['train_acc'])
    epochs2 = len(history2['train_acc'])
    # Identify the shortest duration to maintain a synchronized comparison window
    epochs = min(epochs1, epochs2)
    
    # Initialize a figure with two subplots for side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    # Define the range of epochs for the x-axis
    epoch_range = range(1, epochs + 1)
    
    # Plot training and testing accuracy for the first model
    ax1.plot(epoch_range, history1['train_acc'][:epochs], 'b-', label='Train', linewidth=2)
    ax1.plot(epoch_range, history1['test_acc'][:epochs], 'r-', label='Test', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title(model1_name)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([40, 100])
    # Apply integer ticks if the number of epochs is small
    if epochs <= 10:
        ax1.set_xticks(range(1, epochs + 1))
    
    # Determine the highest accuracy and corresponding epoch for the first model
    if 'best_epoch' in history1 and history1['best_epoch'] <= epochs:
        best_epoch1 = history1['best_epoch']
        best_acc1 = history1['best_test_acc']
    else:
        best_acc1 = max(history1['test_acc'][:epochs])
        best_epoch1 = history1['test_acc'][:epochs].index(best_acc1) + 1
    
    # Highlight the best performance point on the first plot
    ax1.plot(best_epoch1, best_acc1, 'r*', markersize=12)
    ax1.annotate(f'Best: {best_acc1:.1f}%',
                xy=(best_epoch1, best_acc1),
                xytext=(best_epoch1, best_acc1-5),
                fontsize=9, ha='center')
    
    # Draw a vertical indicator at the selected model's epoch
    ax1.axvline(x=best_epoch1, color='green', linestyle='--', alpha=0.3, label='Selected Model')
    
    # Plot training and testing accuracy for the second model
    ax2.plot(epoch_range, history2['train_acc'][:epochs], 'b-', label='Train', linewidth=2)
    ax2.plot(epoch_range, history2['test_acc'][:epochs], 'r-', label='Test', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(model2_name)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([40, 100])
    if epochs <= 10:
        ax2.set_xticks(range(1, epochs + 1))
    
    # Determine the highest accuracy and corresponding epoch for the second model
    if 'best_epoch' in history2 and history2['best_epoch'] <= epochs:
        best_epoch2 = history2['best_epoch']
        best_acc2 = history2['best_test_acc']
    else:
        best_acc2 = max(history2['test_acc'][:epochs])
        best_epoch2 = history2['test_acc'][:epochs].index(best_acc2) + 1
    
    # Highlight the best performance point on the second plot
    ax2.plot(best_epoch2, best_acc2, 'r*', markersize=12)
    ax2.annotate(f'Best: {best_acc2:.1f}%',
                xy=(best_epoch2, best_acc2),
                xytext=(best_epoch2, best_acc2-5),
                fontsize=9, ha='center')
    
    # Draw a vertical indicator for the second selected model
    ax2.axvline(x=best_epoch2, color='green', linestyle='--', alpha=0.3, label='Selected Model')
    
    # Finalize the layout and display the visual plots
    plt.tight_layout()
    plt.show()
    
    # Retrieve performance metrics at the optimal epoch for both models
    selected_test_acc1 = history1['test_acc'][best_epoch1-1]
    selected_train_acc1 = history1['train_acc'][best_epoch1-1]
    selected_test_acc2 = history2['test_acc'][best_epoch2-1]
    selected_train_acc2 = history2['train_acc'][best_epoch2-1]
    
    # Retrieve final epoch metrics for baseline comparison
    final_test_acc1 = history1['test_acc'][epochs-1]
    final_train_acc1 = history1['train_acc'][epochs-1]
    final_test_acc2 = history2['test_acc'][epochs-1]
    final_train_acc2 = history2['train_acc'][epochs-1]
    
    # Calculate the accuracy gap between the two selected models
    difference = selected_test_acc2 - selected_test_acc1
    
    # Calculate the epoch at which models reached 90% of their peak accuracy
    convergence_threshold = 0.9 * max(best_acc1, best_acc2)
    conv_epoch1 = next((i for i, acc in enumerate(history1['test_acc'][:epochs])
                        if acc >= convergence_threshold), epochs) + 1
    conv_epoch2 = next((i for i, acc in enumerate(history2['test_acc'][:epochs])
                        if acc >= convergence_threshold), epochs) + 1
    
    # Generate the text-based comparison summary report
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"Training Duration: {epochs} epochs")
    
    # Print metrics specifically for the first model
    print(f"\n{model1_name}:")
    print(f"  Architecture: {model1_desc}")
    print(f"  Selected Model from Epoch: {best_epoch1}")
    print(f"  Selected Model Test Accuracy: {selected_test_acc1:.2f}%")
    print(f"  Selected Model Train Accuracy: {selected_train_acc1:.2f}%")
    if best_epoch1 != epochs:
        print(f"  (Final epoch accuracy was: {final_test_acc1:.2f}%)")
    print(f"  Convergence: Epoch {conv_epoch1}")
    
    # Print metrics specifically for the second model
    print(f"\n{model2_name}:")
    print(f"  Architecture: {model2_desc}")
    print(f"  Selected Model from Epoch: {best_epoch2}")
    print(f"  Selected Model Test Accuracy: {selected_test_acc2:.2f}%")
    print(f"  Selected Model Train Accuracy: {selected_train_acc2:.2f}%")
    if best_epoch2 != epochs:
        print(f"  (Final epoch accuracy was: {final_test_acc2:.2f}%)")
    print(f"  Convergence: Epoch {conv_epoch2}")
    
    # Analyze and compare the relative performance of both models
    print(f"\n" + "-"*60)
    print("PERFORMANCE ANALYSIS")
    print("-"*60)
    print(f"Difference in selected model accuracy: {difference:+.2f}%")
    
    # Identify which model converged to the target threshold more quickly
    if conv_epoch1 < conv_epoch2:
        print(f"Faster convergence: {model1_name} (by {conv_epoch2-conv_epoch1} epochs)")
    elif conv_epoch2 < conv_epoch1:
        print(f"Faster convergence: {model2_name} (by {conv_epoch1-conv_epoch2} epochs)")
    else:
        print(f"Both models converged at the same speed")
    
    # Provide a qualitative interpretation of the quantitative differences
    print(f"\n" + "-"*60)
    print("INTERPRETATION")
    print("-"*60)
    
    if abs(difference) < 2:
        print("✓ Both models perform similarly! The implementations are comparable.")
    elif difference > 0:
        print(f"✓ {model2_name} performs better by {difference:.2f}%")
        if difference > 5:
            print("  This is a significant improvement.")
    else:
        print(f"✓ {model1_name} performs better by {-difference:.2f}%")
        if difference < -5:
            print("  This is a significant improvement.")
    
    # Analyze the training-testing accuracy gap to identify potential overfitting
    overfit1 = selected_train_acc1 - selected_test_acc1
    overfit2 = selected_train_acc2 - selected_test_acc2
    
    print("\n⚠️  Overfitting Analysis (for selected models):")
    print(f"  {model1_name}: {overfit1:.1f}% gap (train-test) at epoch {best_epoch1}")
    print(f"  {model2_name}: {overfit2:.1f}% gap (train-test) at epoch {best_epoch2}")
    
    # Evaluate if the overfitting gap exceeds acceptable thresholds
    if overfit1 > 10 or overfit2 > 10:
        if overfit1 > overfit2:
            print(f"  → {model1_name} shows more overfitting")
        else:
            print(f"  → {model2_name} shows more overfitting")
    else:
        print("  → Both models show acceptable generalization (gap < 10%)")
    
    # Assess the effectiveness of selecting the best epoch over the final epoch
    if best_epoch1 < epochs or best_epoch2 < epochs:
        print("\n📊 Early Stopping Analysis:")
        if best_epoch1 < epochs:
            prevented_overfit1 = (final_train_acc1 - final_test_acc1) - overfit1
            print(f"  {model1_name}: Early stopping at epoch {best_epoch1} prevented {prevented_overfit1:.1f}% additional overfitting")
        if best_epoch2 < epochs:
            prevented_overfit2 = (final_train_acc2 - final_test_acc2) - overfit2
            print(f"  {model2_name}: Early stopping at epoch {best_epoch2} prevented {prevented_overfit2:.1f}% additional overfitting")
    
    print("="*60)