import inspect
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
import torchmetrics
from IPython.display import Markdown, display
from torch.utils.data import random_split
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer



def download_bert(model_name="distilbert-base-uncased", local_path="./distilbert-local-base"):
    """
    Downloads a base transformer model and tokenizer from the Hugging Face Hub.

    Args:
        model_name: The name of the model on the Hugging Face Hub.
        local_path: The local directory where the model and tokenizer will be saved.
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
    Loads a base transformer model from a local directory and adds a new
    sequence classification head.

    Args:
        local_path: The local directory where the base model files are stored.
        num_classes: The number of output classes for the new classification head.

    Returns:
        A tuple containing the loaded model (with the new head) and the tokenizer.
    """
    # Announce the model loading and head configuration process.
    print(
        f"Loading base model from {local_path} and adding a new head with {num_classes} classes.\n"
    )

    # Load the pre-trained tokenizer from the specified local directory.
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    # Load the base pre-trained model and add a new, untrained sequence
    # classification head on top with the specified number of labels.
    model = AutoModelForSequenceClassification.from_pretrained(
        local_path, num_labels=num_classes
    )

    # Confirm that the model and tokenizer have been successfully loaded.
    print("\nModel and tokenizer loaded successfully.")

    # Return the complete model and its tokenizer.
    return model, tokenizer



def create_dataset_splits(full_dataset, train_split_percentage=0.8):
    """
    Splits a full dataset into training and validation sets.

    Args:
        full_dataset: The complete PyTorch Dataset to be split.
        train_split_percentage: The percentage of the dataset to allocate
                                for the training set. Defaults to 0.8.

    Returns:
        A tuple containing the training dataset and validation dataset.
    """
    # Calculate the number of samples for the training set.
    train_size = int(train_split_percentage * len(full_dataset))
    # Calculate the number of samples for the validation set.
    val_size = len(full_dataset) - train_size

    # Randomly split the full dataset into training and validation subsets.
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Return the two dataset splits.
    return train_dataset, val_dataset



def training_loop(model, train_loader, val_loader, loss_function, learning_rate, num_epochs, device):
    """
    Performs a full training and validation cycle for a PyTorch model.

    Args:
        model: The PyTorch model to be trained.
        train_loader: The DataLoader for the training dataset.
        val_loader: The DataLoader for the validation dataset.
        loss_function: The loss function used for training.
        learning_rate: The learning rate for the optimizer.
        num_epochs: The total number of epochs to train for.
        device: The computational device ('cuda' or 'cpu') to run on.

    Returns:
        A tuple containing the trained model and a dictionary of the final
        performance metrics from the last validation epoch.
    """
    # Move the model to the specified computational device.
    model.to(device)

    # Initialize the AdamW optimizer with the specified learning rate and weight decay.
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Determine the number of classes from the model's configuration.
    num_classes = model.config.num_labels
    
    # Initialize metric objects from torchmetrics for stateful metric calculation.
    val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)
    val_cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    
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
        train_inner_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False)
        # Iterate over the training data batches.
        for batch in train_inner_loop:
            # Unpack the batch and move all tensors to the active device.
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Clear any gradients from the previous iteration.
            optimizer.zero_grad()
            
            # Perform a forward pass to get the model's raw outputs (logits).
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Calculate the loss for the current batch.
            loss = loss_function(logits, labels)
            
            # Accumulate the training loss and perform backpropagation to compute gradients.
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
        val_inner_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=False)
        # Disable gradient calculations to save memory and computations.
        with torch.no_grad():
            # Iterate over the validation data batches.
            for batch in val_inner_loop:
                # Unpack the batch and move tensors to the active device.
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Perform a forward pass to get the model's logits.
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Calculate and accumulate validation loss for the batch.
                val_loss = loss_function(logits, labels)
                val_loss_epoch += val_loss.item()
                
                # Get model predictions and update all metric objects with batch results.
                preds = torch.argmax(logits, dim=-1)
                val_accuracy.update(preds, labels)
                val_f1.update(preds, labels)
                val_cm.update(preds, labels)
        
        # Calculate the average validation loss for the epoch.
        val_loss_epoch /= len(val_loader)
        
        # --- Logging and Metric Calculation ---
        # Compute the final metrics over the entire validation set for the epoch.
        epoch_acc = val_accuracy.compute()
        epoch_f1 = val_f1.compute()

        # Reset the metric calculators for the next epoch.
        val_accuracy.reset()
        val_f1.reset()

        # Update the main progress bar with the results of the completed epoch.
        epoch_loop.set_postfix(
            train_loss=f"{train_loss_epoch:.4f}",
            val_loss=f"{val_loss_epoch:.4f}",
            val_acc=f"{epoch_acc:.4f}"
        )
        # Use tqdm.write to log metrics without interfering with the progress bars.
        tqdm.write(f"Epoch {epoch+1} Metrics -> Val Loss: {val_loss_epoch:.4f}, Val Acc: {epoch_acc:.4f}, Val F1: {epoch_f1:.4f}")

    # Indicate that the entire training process is complete.
    print("\n--- Training complete ---")

    # Compute the final confusion matrix over the entire validation dataset.
    final_confusion_matrix = val_cm.compute()

    # Store the final metrics from the last epoch in a results dictionary.
    final_results = {
        "val_loss": val_loss_epoch,
        "val_accuracy": epoch_acc.item(),
        "val_f1": epoch_f1.item(),
        "confusion_matrix": final_confusion_matrix.cpu()
    }
    
    # Return the trained model and the final results.
    return model, final_results



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



def _create_accuracy_table(class_accuracy, id2cat):
    """
    Formats the per-class accuracy data into a markdown table string.
    
    This is a helper function designed to be called by the main analysis function.

    Args:
        class_accuracy (np.array): An array of accuracy values for each class.
        id2cat (dict): A dictionary mapping class IDs to category names.

    Returns:
        str: A formatted markdown string representing the accuracy table.
    """
    # Build the markdown string for the table headers and rows.
    markdown_table = "| Category                  | Accuracy |\n"
    markdown_table += "|---------------------------|----------|\n"
    
    # Loop through each class to create a row in the table.
    for class_id, accuracy in enumerate(class_accuracy):
        category_name = id2cat[class_id]
        # Format each row and add it to the string.
        markdown_table += f"| {category_name:<25} | {accuracy:.2%}    |\n"
        
    return markdown_table



def analyze_and_plot_results(results, id2cat):
    """
    Analyzes training results to display per-class accuracy and plot a
    confusion matrix.

    This function serves as a comprehensive tool to visualize model performance
    post-training. It first presents a quantitative breakdown of per-class
    accuracy in a table and then provides a qualitative view with a
    seaborn-enhanced confusion matrix.

    Args:
        results (dict): The results dictionary from the training loop, expected
                        to contain a 'confusion_matrix' tensor.
        id2cat (dict): A dictionary mapping class IDs to category names.
    """
    # --- Step 1: Extract and Prepare Data ---
    
    # Get the confusion matrix tensor and convert it to a NumPy array for analysis.
    confusion_matrix_numpy = results['confusion_matrix'].cpu().numpy()
    
    # Retrieve the list of class names in the correct order for plot labels.
    class_names = [name for id, name in sorted(id2cat.items())]

    # --- Step 2: Calculate and Display Per-Class Accuracy ---
    
    # The diagonal of the confusion matrix contains the correct predictions (True Positives).
    correct_predictions = confusion_matrix_numpy.diagonal()
    
    # The sum of each row represents the total number of actual samples for each class.
    total_samples_per_class = confusion_matrix_numpy.sum(axis=1)
    
    # Calculate the accuracy for each class, avoiding division by zero for safety.
    class_accuracy = correct_predictions / (total_samples_per_class + 1e-9)

    # Generate the accuracy table using the helper function.
    accuracy_table_md = _create_accuracy_table(class_accuracy, id2cat)
    
    # Display the title and the rendered markdown table.
    display(Markdown("### **Per-Class Accuracy**"))
    display(Markdown(accuracy_table_md))

    # --- Step 3: Plot the Confusion Matrix ---
    
    # Create the heatmap using seaborn for a clear visualization.
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix_numpy, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    # Add a title and labels for context.
    plt.title("Confusion Matrix for Dispatcher Model")
    plt.xlabel("Predicted Category")
    plt.ylabel("True Category")
    plt.show()



def predict_category(model, tokenizer, text, device, id2cat):
    """
    Performs inference on a single text string to predict its category.

    Args:
        model (nn.Module): The fine-tuned PyTorch model.
        tokenizer: The Hugging Face tokenizer corresponding to the model.
        text (str): The raw input text string.
        device: The device to perform inference on ('cuda', 'cpu', etc.).
        id2cat (dict): A dictionary mapping class IDs to category names.

    Returns:
        str: The predicted category name as a string.
    """
    # Set the model to evaluation mode.
    model.eval()

    # Tokenize the input text and create PyTorch tensors.
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    # Move the input tensors to the specified device.
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Disable gradient calculations to save memory and speed up inference.
    with torch.no_grad():
        # Perform a forward pass through the model to get the outputs.
        outputs = model(**inputs)
    
    # Get the raw prediction scores (logits).
    logits = outputs.logits
    # Find the index of the highest logit, which corresponds to the predicted class ID.
    predicted_id = torch.argmax(logits, dim=-1).item()
    
    # Map the numerical prediction ID back to the human-readable category name.
    return id2cat[predicted_id]



def save_training_logs(partial_results, filename="training_logs.pkl"):
    """
    Serializes and saves a results object to a file using pickle.

    Args:
        partial_results: The Python object to be saved.
        filename: The path and name of the file to save the object to.
    """
    # Open the specified file in write-binary mode.
    with open(filename, 'wb') as f:
        # Use pickle to serialize and write the object to the file.
        pickle.dump(partial_results, f)
    
    # Print a confirmation message indicating where the logs were saved.
    print(f"Training logs saved to '{filename}'")


    
#####################################

###### Additional imports needed by the `augment_dataset_with_back_translation` function
# from deep_translator import GoogleTranslator
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
#
# def augment_dataset_with_back_translation(
#     input_csv,
#     output_csv,
#     column_to_translate = 'instruction',
#     column_to_keep = 'category',
#     intermediate_languages = ['ar', 'ur', 'de', 'fr', 'es', 'ru', 'zh-cn', 'ja', 'hi', 'pt'],
#     max_workers = 20
# ):
#     """
#     Loads a CSV, augments a text column using multi-threaded back-translation,
#     and saves the combined (original + augmented) data to a new CSV file.

#     Args:
#         input_csv (str): Path to the source CSV file, which should be "databricks-dolly-15k.csv".
#         output_csv (str): Path to save the final augmented CSV file.
#         column_to_translate (str): The name of the column containing text to augment.
#         column_to_keep (str): The name of the column to preserve alongside the text.
#         intermediate_languages (list): A list of language codes for back-translation.
#         max_workers (int): The number of parallel threads to use for the translation jobs.
#     """
#     # --- A nested helper function for a single translation job ---
#     def back_translate_job(instruction, category, lang_code):
#         """Performs a single back-translation and returns a result dictionary or None."""
#         try:
#             translated_to_lang = GoogleTranslator(source='en', target=lang_code).translate(instruction)
#             back_translated_to_en = GoogleTranslator(source=lang_code, target='en').translate(translated_to_lang)
#             if back_translated_to_en and back_translated_to_en.lower() != instruction.lower():
#                 return {column_to_translate: back_translated_to_en, column_to_keep: category}
#         except Exception:
#             pass # Silently ignore API errors for a cleaner run
#         return None

#     # --- Load the source data ---
#     print(f"Loading original dataset from: {input_csv}")
#     try:
#         df = pd.read_csv(input_csv)
#         print("Dataset loaded successfully.")
#     except FileNotFoundError:
#         print(f"Error: The file '{input_csv}' was not found. Aborting.")
#         return

#     # --- Prepare and execute the parallel jobs ---
#     augmented_data = []
#     jobs = [(row[column_to_translate], row[column_to_keep], lang)
#             for _, row in df.iterrows()
#             for lang in intermediate_languages]

#     print(f"\nStarting optimized augmentation with {max_workers} parallel workers...")
#     print(f"Total translation jobs to perform: {len(jobs)}")

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         future_to_job = {executor.submit(back_translate_job, *job): job for job in jobs}
#         for future in tqdm(as_completed(future_to_job), total=len(jobs), desc="Processing jobs"):
#             result = future.result()
#             if result:
#                 augmented_data.append(result)

#     print("\nAugmentation process complete.")

#     # --- Combine original and augmented data ---
#     augmented_df = pd.DataFrame(augmented_data)
#     original_df_subset = df[[column_to_translate, column_to_keep]]
#     final_df = pd.concat([original_df_subset, augmented_df], ignore_index=True)

#     print(f"\nOriginal dataset size: {len(df)} rows")
#     print(f"Successfully created {len(augmented_df)} new augmented rows")
#     print(f"New expanded dataset size: {len(final_df)} rows")

#     # --- Save the final result ---
#     print(f"Saving expanded dataset to: {output_csv}")
#     final_df.to_csv(output_csv, index=False)

#     print(f"\nSuccessfully created the expanded dataset at '{output_csv}'.")


