import ast
import inspect
import os
import random
import textwrap

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchmetrics
from IPython.display import Markdown, display
from torch.utils.data import random_split
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer



def print_final_results(results):
    """
    Displays the final validation metrics in a formatted manner.

    Args:
        results: A dictionary containing the final validation metrics,
                 expected to have keys 'val_accuracy', 'val_precision',
                 'val_recall', and 'val_f1'.
    """
    # Print a header for the metrics section.
    print("Final Validation Metrics")
    # Display the formatted validation accuracy.
    print(f"\nAccuracy:   {results['val_accuracy']:.4f}")
    # Display the formatted validation precision.
    print(f"Precision:  {results['val_precision']:.4f}")
    # Display the formatted validation recall.
    print(f"Recall:     {results['val_recall']:.4f}")
    # Display the formatted validation F1-score.
    print(f"F1:         {results['val_f1']:.4f}\n")

    

SEED = 99
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)



def display_function(func):
    """
    Renders the source code of a given function as a formatted Markdown block.

    Args:
        func: The function object whose source code is to be displayed.
    """
    # Retrieve the source code of the function as a raw string.
    source_code_str = inspect.getsource(func)
    # Format the raw string into a Python code block for Markdown rendering.
    markdown_formatted_code = f"```python\n{source_code_str}\n```"
    # Display the formatted code block as Markdown output.
    display(Markdown(markdown_formatted_code))

    

def display_results(full_results, partial_results):
    """
    Generates and displays a Markdown table in a Jupyter notebook
    to compare the performance of two models.

    Args:
        full_results (dict): A dictionary with keys 'val_accuracy' and 'val_f1'
                             for the fully fine-tuned model.
        partial_results (dict): A dictionary with keys 'val_accuracy' and 'val_f1'
                                for the partially fine-tuned model.
    """
    # Create the comparison table using a multi-line f-string
    # textwrap.dedent() is used to remove the leading whitespace
    markdown_table = textwrap.dedent(
        f"""
    | Model Description | Validation Accuracy | Validation F1 Score |
    |:---|:---:|:---:|
    | **Full Fine-tuned model (baseline)** | {full_results['val_accuracy']:.4f} | {full_results['val_f1']:.4f} |
    | **Partial fine-tuned model** | {partial_results['val_accuracy']:.4f} | {partial_results['val_f1']:.4f} |
    """
    )

    # Display the formatted Markdown table in the notebook output
    display(Markdown(markdown_table))

    

def filter_recipe_dataset(input_path, output_path="recipes_fruit_veg.csv"):
    """
    Filters a raw recipe dataset to create a smaller subset containing
    only mutually exclusive fruit or vegetable recipes.

    Args:
        input_path: The file path for the original raw recipe dataset CSV.
        output_path: The file path where the filtered subset CSV will be saved.
    """
    # Notify the user that the dataset loading process has started.
    print(f"Loading the raw dataset from '{input_path}'...")
    # Use a try-except block to gracefully handle file not found errors.
    try:
        # Read the specified CSV file into a pandas DataFrame.
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        # Print an error message if the file does not exist and exit the function.
        print(f"Error: The file was not found at '{input_path}'")
        return

    # Define a list of keywords to identify recipes containing fruit.
    fruit_keywords = [
        "apple", "banana", "orange", "strawberry", "grape", "mango",
        "pineapple", "peach", "pear", "cherry", "berry", "lemon",
        "lime", "melon",
    ]
    # Define a list of keywords to identify recipes containing vegetables.
    vegetable_keywords = [
        "carrot", "broccoli", "spinach", "potato", "tomato", "onion",
        "garlic", "pepper", "lettuce", "cucumber", "celery", "mushroom",
        "corn", "bean", "pea", "cabbage", "asparagus",
    ]

    # Define the main categorization function to be applied to each row.
    def categorize_recipe(ingredients_str):
        """
        Categorizes a recipe as 'fruit', 'vegetable', or 'other' based on
        a string of its ingredients.

        Args:
            ingredients_str: The string representation of the ingredient list.

        Returns:
            A string indicating the category ('fruit', 'vegetable', 'other').
        """
        # Handle potential parsing errors for malformed ingredient strings.
        try:
            # Safely parse the string representation of the ingredient list.
            ingredients_list = ast.literal_eval(ingredients_str)
            # Join the ingredients into a single lowercase string for searching.
            ingredients_text = " ".join(ingredients_list).lower()

            # Check for the presence of any fruit keywords.
            has_fruit = any(keyword in ingredients_text for keyword in fruit_keywords)
            # Check for the presence of any vegetable keywords.
            has_veg = any(keyword in ingredients_text for keyword in vegetable_keywords)

            # Assign mutually exclusive categories based on keyword presence.
            if has_fruit and not has_veg:
                return "fruit"
            if has_veg and not has_fruit:
                return "vegetable"

            # Return 'other' for recipes with both, or no, keywords.
            return "other"
        # If parsing fails, categorize the recipe as 'other'.
        except (ValueError, SyntaxError):
            return "other"

    # Notify the user that the categorization process is starting.
    print("Categorizing recipes based on ingredient keywords...")
    # Apply the categorization function to each recipe's ingredients.
    df["category"] = df["ingredients"].apply(categorize_recipe)

    # Filter the DataFrame to keep only 'fruit' and 'vegetable' categories.
    filtered_df = df[df["category"].isin(["fruit", "vegetable"])].copy()

    # Define the specific columns to keep in the final dataset.
    columns_to_keep = ["name", "id", "minutes", "ingredients", "steps", "category"]
    # Create the final subset DataFrame with only the selected columns.
    subset_df = filtered_df[columns_to_keep]

    # Announce that the filtering process is complete.
    print("Filtering complete.")
    # Print the total count of recipes found in the 'fruit' category.
    print(f"Found {len(subset_df[subset_df['category'] == 'fruit'])} fruit recipes.")
    # Print the total count of recipes found in the 'vegetable' category.
    print(
        f"Found {len(subset_df[subset_df['category'] == 'vegetable'])} vegetable recipes."
    )

    # Notify the user that the data is being saved.
    print(f"\nSaving the subset data to '{output_path}'...")
    # Save the subset DataFrame to a new CSV file, without the index column.
    subset_df.to_csv(output_path, index=False)

    # Print a final success message confirming the file has been saved.
    print(
        f"Success! Subset dataset with fruit and vegetable recipes saved to '{output_path}'."
    )

    

def download_bert(model_name="distilbert-base-uncased", local_path="./distilbert-local-base"):
    """
    Downloads only the base transformer model and tokenizer, without any specific head.

    Args:
        model_name (str): The name of the model on the Hugging Face Hub.
        local_path (str): The local directory to save the base model to.
    """
    # Check if the target directory already exists.
    if os.path.isdir(local_path):
        # If the directory exists, notify the user and skip downloading.
        print(f"Base model '{model_name}' already available at {local_path}")
    else:
        # If the directory does not exist, proceed with the download.
        print(f"Downloading base model '{model_name}' to {local_path}...")
        # Download the pre-trained tokenizer associated with the model.
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Download the base pre-trained model without a specific classification head.
        model = AutoModel.from_pretrained(model_name)

        # Save the tokenizer's files to the specified local path.
        tokenizer.save_pretrained(local_path)
        # Save the model's files to the specified local path.
        model.save_pretrained(local_path)
        # Confirm that the model has been downloaded and saved.
        print("Base model downloaded and saved successfully.")
        


def load_bert(local_path="./distilbert-local-base", num_classes=2):
    """
    Loads the base model and dynamically adds a classification head.

    Args:
        local_path (str): The local directory where the base model is stored.
        num_classes (int): The number of output classes for the new head.
        device (str): The device to load the model onto.

    Returns:
        tuple: A tuple containing the loaded model (with head) and tokenizer.
    """
    # Announce the model loading and head configuration process.
    print(
        f"Loading base model from {local_path} and adding a new head with {num_classes} classes."
    )

    # Load the pre-trained tokenizer from the specified local directory.
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    # Load the base pre-trained model and add a new, untrained sequence
    # classification head on top with the specified number of labels.
    model = AutoModelForSequenceClassification.from_pretrained(
        local_path, num_labels=num_classes
    )

    # Confirm that the model and tokenizer have been successfully loaded.
    print("Model and tokenizer loaded successfully.")

    # Return the complete model and its tokenizer.
    return model, tokenizer



def create_dataset_splits(full_dataset, train_split_percentage=0.8):
    """
    Splits a full dataset into training and validation sets.

    Args:
        full_dataset (Dataset): The complete PyTorch Dataset to be split.
        train_split_percentage (float, optional): The percentage of the dataset
                                                 to allocate for training.
                                                 Defaults to 0.8 (80%).

    Returns:
        tuple: A tuple containing the training dataset and validation dataset.
    """
    # Calculate the sizes of the training and validation sets
    train_size = int(train_split_percentage * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Perform the random split
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    return train_dataset, val_dataset



def training_loop(model, train_loader, val_loader, loss_function, num_epochs, device):
    """
    Performs a full training and validation cycle for a PyTorch model.

    Args:
        model: The PyTorch model to be trained.
        train_loader: The DataLoader for the training dataset.
        val_loader: The DataLoader for the validation dataset.
        loss_function: The loss function used for training.
        num_epochs: The total number of epochs to train for.
        device: The computational device ('cuda' or 'cpu') to run on.

    Returns:
        A tuple containing the trained model and a dictionary of the final
        performance metrics from the last validation epoch.
    """
    # Move the model to the specified computational device.
    model.to(device)

    # Initialize the AdamW optimizer with a default learning rate.
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    # Determine the number of classes from the model's configuration.
    num_classes = model.config.num_labels

    # Initialize metric objects from torchmetrics for stateful metric calculation.
    val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro").to(device)
    val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro").to(device)
    val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

    # Create the main progress bar that iterates over the epochs.
    epoch_loop = tqdm(range(num_epochs), desc="Training Progress")

    # Begin the main training and validation loop.
    for epoch in epoch_loop:

        # --- Training Phase ---
        # Set the model to training mode, which enables layers like dropout.
        model.train()
        # Initialize the accumulated training loss for the epoch.
        train_loss_epoch = 0

        # Create a nested progress bar for the training batches of the current epoch.
        train_inner_loop = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False
        )
        # Iterate over the training data batches.
        for batch in train_inner_loop:
            # Unpack the batch and move all tensors to the active device.
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Clear any gradients from the previous iteration.
            optimizer.zero_grad()

            # Perform a forward pass to get the model's raw outputs (logits).
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Calculate the loss for the current batch.
            loss = loss_function(logits, labels)

            # Accumulate the loss and perform backpropagation to compute gradients.
            train_loss_epoch += loss.item()
            loss.backward()

            # Update the model's weights based on the computed gradients.
            optimizer.step()

            # Update the inner progress bar's postfix with the current batch loss.
            train_inner_loop.set_postfix(loss=loss.item())

        # Calculate the average training loss over all batches in the epoch.
        train_loss_epoch /= len(train_loader)

        # --- Validation Phase ---
        # Set the model to evaluation mode, which disables layers like dropout.
        model.eval()
        # Initialize the accumulated validation loss for the epoch.
        val_loss_epoch = 0

        # Create a nested progress bar for the validation batches.
        val_inner_loop = tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=False
        )
        # Disable gradient calculations to save memory and computations.
        with torch.no_grad():
            # Iterate over the validation data batches.
            for batch in val_inner_loop:
                # Unpack the batch and move tensors to the active device.
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Perform a forward pass to get the model's logits.
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Calculate the validation loss for the current batch.
                val_loss = loss_function(logits, labels)
                val_loss_epoch += val_loss.item()

                # Get model predictions and update the metric objects with batch results.
                preds = torch.argmax(logits, dim=-1)
                val_accuracy.update(preds, labels)
                val_precision.update(preds, labels)
                val_recall.update(preds, labels)
                val_f1.update(preds, labels)
        
        # Calculate the average validation loss for the epoch.
        val_loss_epoch /= len(val_loader)

        # --- Logging and Metric Calculation ---
        # Compute the final metrics over the entire validation set for the epoch.
        epoch_acc = val_accuracy.compute()
        epoch_prec = val_precision.compute()
        epoch_recall = val_recall.compute()
        epoch_f1 = val_f1.compute()

        # Reset all metric objects to be ready for the next epoch.
        val_accuracy.reset()
        val_precision.reset()
        val_recall.reset()
        val_f1.reset()

        # Update the main progress bar with the results of the completed epoch.
        epoch_loop.set_postfix(
            train_loss=f"{train_loss_epoch:.4f}",
            val_loss=f"{val_loss_epoch:.4f}",
            val_acc=f"{epoch_acc:.4f}",
        )
        # Use tqdm.write to log metrics without interfering with the progress bars.
        tqdm.write(
            f"Epoch {epoch+1} Metrics -> Val Acc: {epoch_acc:.4f}, Val F1: {epoch_f1:.4f}"
        )

    # Indicate that the entire training process is complete.
    print("\n--- Training complete ---")

    # Store the final metrics from the last epoch in a dictionary.
    final_results = {
        "val_accuracy": epoch_acc.item(),
        "val_precision": epoch_prec.item(),
        "val_recall": epoch_recall.item(),
        "val_f1": epoch_f1.item(),
    }
    
    # Return the trained model and the final results.
    return model, final_results



def predict_category(model, tokenizer, text, device):
    """
    Performs inference on a single text string to predict its category.

    Args:
        model: The fine-tuned PyTorch model.
        tokenizer: The Hugging Face tokenizer corresponding to the model.
        text: The raw input text string (e.g., a recipe title).
        device: The device to perform inference on ('cuda', 'cpu', etc.).

    Returns:
        The predicted category as a string ('Fruit' or 'Vegetable').
    """
    # Set the model to evaluation mode, which disables layers like dropout.
    model.eval()

    # Tokenize the input text and create PyTorch tensors.
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    # Move the input tensors to the specified device.
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Disable gradient calculations to save memory and speed up inference.
    with torch.no_grad():
        # Perform a forward pass through the model to get the outputs.
        outputs = model(**inputs)

    # Get the raw, unnormalized prediction scores (logits) from the model's output.
    logits = outputs.logits
    # Find the index of the highest logit, which corresponds to the predicted class.
    prediction = torch.argmax(logits, dim=-1).item()

    # Map the numerical prediction back to the human-readable string label.
    return "Vegetable" if prediction == 1 else "Fruit"