import ast
import inspect
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from IPython.display import Markdown, display
from tqdm.auto import tqdm



def apply_dlai_style():
    # Global plot style
    PLOT_STYLE = {
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "font.family": "sans",  # "sans-serif",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 3,
        "lines.markersize": 6,
    }

    # Custom colors (reusable)
    color_map = {
        "pink": "#F65B66",
        "blue": "#1C74EB",
        "yellow": "#FAB901",
        "red": "#DD3C66",
        "purple": "#A12F9D",
        "cyan": "#237B94",
    }
    return color_map, PLOT_STYLE



color_map, PLOT_STYLE = apply_dlai_style()
mpl.rcParams.update(PLOT_STYLE)



def display_function(func):
    """
    Displays the source code of a Python function within a Markdown block.

    Args:
        func: The function object whose source code is to be displayed.
    """
    # Retrieve the source code of the function as a string.
    source_code_str = inspect.getsource(func)
    # Format the source code string as a Markdown code block for Python.
    markdown_formatted_code = f"```python\n{source_code_str}\n```"
    # Display the formatted Markdown in the output.
    display(Markdown(markdown_formatted_code))



def filter_recipe_dataset(input_path, output_path="recipes_fruit_veg.csv"):
    """
    Filters the raw Food.com recipe dataset to create a smaller subset
    containing only mutually exclusive fruit or vegetable recipes.

    This function reads a large recipe dataset, categorizes each recipe
    based on keywords in its ingredients, and filters it to keep only recipes
    that are exclusively fruit-based or exclusively vegetable-based. The
    resulting subset is then saved to a new CSV file.

    Args:
        input_path: The file path for the original recipe dataset CSV file.
        output_path: The file path where the filtered CSV will be saved.
    """
    print(f"Loading the raw dataset from '{input_path}'...")
    # Read the dataset from the specified path, with error handling for missing files.
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: The file was not found at '{input_path}'")
        return

    # Define keywords for categorization.
    fruit_keywords = [
        "apple", "banana", "orange", "strawberry", "grape", "mango",
        "pineapple", "peach", "pear", "cherry", "berry", "lemon",
        "lime", "melon",
    ]
    vegetable_keywords = [
        "carrot", "broccoli", "spinach", "potato", "tomato", "onion",
        "garlic", "pepper", "lettuce", "cucumber", "celery", "mushroom",
        "corn", "bean", "pea", "cabbage", "asparagus",
    ]

    def categorize_recipe(ingredients_str):
        """Categorizes a recipe as 'fruit', 'vegetable', or 'other'."""
        try:
            # Safely parse the string representation of the ingredient list.
            ingredients_list = ast.literal_eval(ingredients_str)
            ingredients_text = " ".join(ingredients_list).lower()

            # Check for the presence of fruit or vegetable keywords.
            has_fruit = any(keyword in ingredients_text for keyword in fruit_keywords)
            has_veg = any(keyword in ingredients_text for keyword in vegetable_keywords)

            # Assign mutually exclusive categories.
            if has_fruit and not has_veg:
                return "fruit"
            if has_veg and not has_fruit:
                return "vegetable"
            
            # Return 'other' for recipes with both or no relevant keywords.
            return "other"
        
        except (ValueError, SyntaxError):
            # Handle potential parsing errors for malformed ingredient strings.
            return "other"

    print("Categorizing recipes based on ingredient keywords...")
    # Apply the categorization function to each row in the DataFrame.
    df["category"] = df["ingredients"].apply(categorize_recipe)

    # Filter the DataFrame to keep only 'fruit' and 'vegetable' categories.
    filtered_df = df[df["category"].isin(["fruit", "vegetable"])].copy()

    # Define the specific columns to keep in the final dataset.
    columns_to_keep = ["name", "id", "minutes", "ingredients", "steps", "category"]
    subset_df = filtered_df[columns_to_keep]

    print("Filtering complete.")
    print(f"Found {len(subset_df[subset_df['category'] == 'fruit'])} fruit recipes.")
    print(f"Found {len(subset_df[subset_df['category'] == 'vegetable'])} vegetable recipes.")

    print(f"\nSaving the subset data to '{output_path}'...")
    # Save the final filtered DataFrame to a CSV file.
    subset_df.to_csv(output_path, index=False)

    print(f"Success! Subset dataset saved to '{output_path}'.")



def training_loop(model, train_loader, val_loader, loss_function, num_epochs, device):
    """
    Handles the complete training and validation loop for a PyTorch model.

    This function trains the model on the training set and evaluates it on the
    validation set for a specified number of epochs. It is designed to work
    with different model architectures by checking the model's class name to
    correctly unpack data batches. It calculates and stores key performance
    metrics for each epoch.

    Args:
        model: The PyTorch model to be trained.
        train_loader: The DataLoader for the training dataset.
        val_loader: The DataLoader for the validation dataset.
        loss_function: The loss function (e.g., nn.CrossEntropyLoss).
        num_epochs: The total number of epochs for training.
        device: The device to train on ('cuda' or 'cpu').

    Returns:
        A tuple containing the trained model and a dictionary of the final
        performance metrics from the last epoch, including validation
        accuracy, precision, recall, and F1-score.
    """
    # Move the model to the specified device (GPU or CPU).
    model.to(device)

    # Initialize the Adam optimizer with a default learning rate.
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Automatically determine the number of classes from the training dataset.
    num_classes = len(train_loader.dataset.classes)

    # Specify the averaging method for multi-class metrics.
    avg_method = "macro"

    # Initialize torchmetrics objects for calculating performance metrics.
    val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average=avg_method).to(device)
    val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average=avg_method).to(device)
    val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average=avg_method).to(device)
    
    # Print the name of the model being trained.
    print(f"--- Training {model.__class__.__name__} ---")

    # Begin the main training loop over the specified number of epochs.
    for epoch in tqdm(range(num_epochs)):

        # Set the model to training mode.
        model.train()
        # Initialize the accumulated training loss for the epoch.
        train_loss_epoch = 0
        # Iterate over batches in the training data loader.
        for batch in train_loader:
            # Clear the gradients from the previous iteration.
            optimizer.zero_grad()

            # Handle different data unpacking based on the model type.
            if model.__class__.__name__ == "EmbeddingBagClassifier":
                # Unpack text, offsets, and labels for EmbeddingBag models.
                text, offsets, labels = batch
                # Move tensors to the specified device.
                text, offsets, labels = text.to(device), offsets.to(device), labels.to(device)
                # Perform a forward pass.
                outputs = model(text, offsets)
            else:
                # Unpack text and labels for other model types.
                text, labels = batch
                # Move tensors to the specified device.
                text, labels = text.to(device), labels.to(device)
                # Perform a forward pass.
                outputs = model(text)

            # Calculate the loss between model outputs and true labels.
            loss = loss_function(outputs, labels)
            # Add the current batch's loss to the epoch's total.
            train_loss_epoch += loss.item()
            # Perform backpropagation to compute gradients.
            loss.backward()
            # Update the model's weights using the optimizer.
            optimizer.step()

        # Calculate the average training loss for the epoch.
        train_loss_epoch /= len(train_loader)

        # Set the model to evaluation mode.
        model.eval()
        # Initialize the accumulated validation loss for the epoch.
        val_loss_epoch = 0
        # Disable gradient calculations for the validation phase.
        with torch.no_grad():
            # Iterate over batches in the validation data loader.
            for batch in val_loader:
                # Handle different data unpacking based on the model type.
                if model.__class__.__name__ == "EmbeddingBagClassifier":
                    # Unpack validation text, offsets, and labels.
                    text, offsets, labels = batch
                    # Move validation tensors to the specified device.
                    text, offsets, labels = text.to(device), offsets.to(device), labels.to(device)
                    # Perform a forward pass on validation data.
                    val_outputs = model(text, offsets)
                else:
                    # Unpack validation text and labels.
                    text, labels = batch
                    # Move validation tensors to the specified device.
                    text, labels = text.to(device), labels.to(device)
                    # Perform a forward pass on validation data.
                    val_outputs = model(text)

                # Add the batch validation loss to the epoch's total.
                val_loss_epoch += loss_function(val_outputs, labels).item()
                # Update validation metrics with the current batch's results.
                val_accuracy.update(val_outputs, labels)
                val_precision.update(val_outputs, labels)
                val_recall.update(val_outputs, labels)
                val_f1.update(val_outputs, labels)

        # Calculate the average validation loss for the epoch.
        val_loss_epoch /= len(val_loader)

        # Compute the final validation metrics for the epoch.
        epoch_acc = val_accuracy.compute()
        epoch_prec = val_precision.compute()
        epoch_recall = val_recall.compute()
        epoch_f1 = val_f1.compute()

        # Reset the metric calculators for the next epoch.
        val_accuracy.reset()
        val_precision.reset()
        val_recall.reset()
        val_f1.reset()

        # Print the training and validation progress at specified intervals.
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss_epoch:.4f} | "
                f"Val Loss: {val_loss_epoch:.4f} | "
                f"Val Accuracy: {epoch_acc:.4f}"
            )

    # Indicate that the training process has finished.
    print("--- Training complete ---")

    # Store the final performance metrics in a dictionary.
    final_results = {
        "val_accuracy": epoch_acc.item(),
        "val_precision": epoch_prec.item(),
        "val_recall": epoch_recall.item(),
        "val_f1": epoch_f1.item(),
    }

    # Return the trained model and the final metrics.
    return model, final_results



def predict_category(model, text, vocab, preprocess_text, device):
    """
    Predicts the category of a recipe given its title.

    Args:
        model (nn.Module): The trained PyTorch model to use for prediction.
        text (str): The raw recipe title to classify.
        vocab (Vocabulary): The trained vocabulary object.
        preprocess_text (function): The function used to clean and tokenize text.
        device: The device to run inference on ('cuda' or 'cpu').

    Returns:
        str: The predicted category as a string ('Fruit Recipe' or 'Vegetable Recipe').
    """
    # Set the model to evaluation mode and move it to the specified device.
    model.to(device)
    model.eval()

    # Preprocess the input text by cleaning and tokenizing it.
    processed = preprocess_text(text)
    # Convert the processed text into a sequence of numerical indices.
    indexed = vocab.encode(processed)

    # Disable gradient calculations for inference.
    with torch.no_grad():
        # Prepare the input tensor based on the model's architecture.
        if model.__class__.__name__ == "EmbeddingBagClassifier":
            # Create tensors for the text indices and offsets.
            text_tensor = torch.tensor(indexed, dtype=torch.long).to(device)
            offsets = torch.tensor([0], dtype=torch.long).to(device)
            # Get model output for the given text and offsets.
            output = model(text_tensor, offsets)
        else:
            # Create a batched tensor for other model types.
            text_tensor = (
                torch.tensor(indexed, dtype=torch.long).unsqueeze(0).to(device)
            )
            # Get model output for the text tensor.
            output = model(text_tensor)

        # Determine the predicted class by finding the index of the highest score.
        predicted_class = torch.argmax(output, dim=1)

    # Map the numerical class index to its corresponding string label.
    category = "Vegetable Recipe" if predicted_class.item() == 1 else "Fruit Recipe"

    # Return the predicted category name.
    return category



def plot_and_select_best_model(all_trained_data, bar_color=color_map["blue"]):
    """
    Analyzes model performance, visualizes F1 scores, and selects the best model.

    This function takes a dictionary of trained models and their performance
    metrics, generates a bar chart comparing their validation F1 scores,
    and identifies and returns the model with the highest score.

    Args:
        all_trained_data: A dictionary where keys are model names and values
                          are tuples containing the model instance and a results
                          dictionary.
        bar_color: The hex color code for the bars in the plot.

    Returns:
        The instance of the top-performing model based on F1 score.
    """
    # Extract model names and their corresponding validation F1 scores into a dictionary.
    model_performance = {
        name: results["val_f1"] for name, (model, results) in all_trained_data.items()
    }
    # Create a pandas Series from the performance data and sort it in descending order.
    performance_series = pd.Series(model_performance).sort_values(ascending=False)

    # Convert the F1 scores to a percentage format for easier interpretation in the plot.
    performance_series_pct = performance_series * 100

    # Initialize a new figure for the plot with a specified size.
    plt.figure(figsize=(12, 8))
    # Generate a bar plot of the model performance.
    ax = performance_series_pct.plot(
        kind="bar", color=bar_color, edgecolor="black", width=0.7
    )

    # Add a horizontal grid to the plot for better readability.
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    # Set the limits of the y-axis to range from 0 to 100.
    ax.set_ylim(bottom=0, top=100)

    # Iterate over each bar in the plot to add a text annotation.
    for bar in ax.patches:
        # Place the F1 score percentage in the middle of each bar.
        ax.annotate(
            f"{bar.get_height():.2f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
            ha="center",
            va="center",
            size=12,
            fontweight="bold",
            color="white",
        )

    # Set the title and axis labels for the plot.
    ax.set_title("Model F1 Score Comparison", fontsize=16, pad=20)
    ax.set_ylabel("Validation F1 Score (%)", fontsize=12)
    # Customize the appearance of the x-axis tick labels.
    ax.tick_params(axis="x", rotation=0, labelsize=12)

    # Adjust plot parameters to ensure everything fits without overlapping.
    plt.tight_layout()
    # Save the generated plot as a PNG image file.
    plt.savefig("f1_score_comparison.png")

    # Identify the name of the model with the highest F1 score.
    best_model_name = performance_series.index[0]
    # Retrieve the model object corresponding to the best-performing model.
    best_model_instance = all_trained_data[best_model_name][0]

    # Print the name of the best performing model to the console.
    print(f"Best Performing Model (by F1 Score): {best_model_name}\n")
    # Return the instance of the best model.
    return best_model_instance



def print_final_metrics(results_dic):
    """
    Displays the final validation metrics in a formatted manner.

    Args:
        results_dic: A dictionary containing the final validation metrics,
                     expected to have keys 'val_accuracy', 'val_precision',
                     'val_recall', and 'val_f1'.
    """
    # Print a header for the metrics section.
    print("\nFinal Validation Metrics")
    # Print the formatted validation accuracy.
    print(f"\nAccuracy:   {results_dic['val_accuracy']:.4f}")
    # Print the formatted validation precision.
    print(f"Precision:  {results_dic['val_precision']:.4f}")
    # Print the formatted validation recall.
    print(f"Recall:     {results_dic['val_recall']:.4f}")
    # Print the formatted validation F1-score.
    print(f"F1:         {results_dic['val_f1']:.4f}\n")



def get_results_df(results_embag, results_mean, results_max, results_sum):
    """
    Consolidates model performance metrics into a styled pandas DataFrame.

    This function takes the results dictionaries from multiple models,
    organizes them into a pandas DataFrame, sets the model name as the
    index, and applies numerical formatting for clear presentation.

    Args:
        results_embag: A dictionary of metrics for the EmbeddingBag model.
        results_mean: A dictionary of metrics for the Mean Pooling model.
        results_max: A dictionary of metrics for the Max Pooling model.
        results_sum: A dictionary of metrics for the Sum Pooling model.

    Returns:
        A pandas Styler object representing the formatted results table.
    """
    # Combine the results from all models into a single pandas DataFrame.
    results_df = pd.DataFrame(
        {
            "Model": [
                "EmbeddingBag",
                "ManualPooling (MEAN)",
                "ManualPooling (MAX)",
                "ManualPooling (SUM)",
            ],
            "Val Accuracy": [
                results_embag["val_accuracy"],
                results_mean["val_accuracy"],
                results_max["val_accuracy"],
                results_sum["val_accuracy"],
            ],
            "Val Precision": [
                results_embag["val_precision"],
                results_mean["val_precision"],
                results_max["val_precision"],
                results_sum["val_precision"],
            ],
            "Val Recall": [
                results_embag["val_recall"],
                results_mean["val_recall"],
                results_max["val_recall"],
                results_sum["val_recall"],
            ],
            "Val F1": [
                results_embag["val_f1"],
                results_mean["val_f1"],
                results_max["val_f1"],
                results_sum["val_f1"],
            ],
        }
    )
    # Set the 'Model' column as the DataFrame index for better organization.
    results_df = results_df.set_index("Model")
    # Apply formatting to the numerical columns to display them to four decimal places.
    results_df = results_df.style.format(
        {
            "Val Accuracy": "{:.4f}",
            "Val Precision": "{:.4f}",
            "Val Recall": "{:.4f}",
            "Val F1": "{:.4f}",
        }
    )

    # Return the styled DataFrame.
    return results_df