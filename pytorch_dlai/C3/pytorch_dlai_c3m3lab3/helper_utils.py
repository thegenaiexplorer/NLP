import os
import re
import urllib.request
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm


def get_shakespeare_data(filename="shakespeare.txt", data_dir="./"):
    """
    Retrieves and loads the Tiny Shakespeare dataset for text processing.

    Args:
        filename (str): The target filename for the saved dataset.
        data_dir (str): The directory path where the file should be stored.

    Returns:
        text (str): The full raw text of the Shakespeare dataset.
    """
    # Combine the directory and filename to create the absolute file path
    filepath = os.path.join(data_dir, filename)
    
    # Verify if the dataset is already present on the local file system
    if os.path.exists(filepath):
        print(f"Shakespeare dataset already exists at {filepath}")
    else:
        # Create the destination directory if it does not currently exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Initiate the download from the source repository
        print("Downloading Shakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        try:
            # Retrieve the remote file and save it to the specified local path
            urllib.request.urlretrieve(url, filepath)
            print(f"Download complete! Saved to {filepath}")
        except Exception as e:
            # Log the exception and re-raise to ensure the error is handled upstream
            print(f"Error downloading file: {e}")
            raise
    
    # Begin reading the dataset into memory
    print("Loading Shakespeare text...")
    with open(filepath, "r", encoding="utf-8") as f:
        # Read the entire file content into a string variable
        text = f.read()
    
    # Display loading metrics and a short snippet of the text content
    print(f"Text loaded successfully! ({len(text):,} characters)")
    print(f"Preview: {text[:300]}...")
    
    # Return the full loaded string
    return text


def train_model(model, vocab_size, loader, loss_fn, optimizer, epochs=10, device='cpu'):
    """
    Executes the training loop for a sequence generation model.

    Args:
        model (nn.Module): The neural network model to be trained.
        vocab_size (int): The total number of unique tokens in the vocabulary.
        loader (DataLoader): The data loader providing batches of training data.
        loss_fn (Callable): The loss function used for optimization.
        optimizer (Optimizer): The optimization algorithm for weight updates.
        epochs (int): The total number of iterations over the dataset.
        device (str): The target computation device for the tensors and model.

    Returns:
        None: The function updates the model in-place and prints progress.
    """
    # Transfer the model parameters to the specified computation hardware
    model.to(device)
    
    for epoch in range(epochs):
        # Configure the model for training mode to enable specific behaviors like dropout
        model.train()
        # Initialize a list to accumulate loss values for each batch in the epoch
        epoch_losses = []
        
        # Iterate through the data loader with a progress bar display
        with tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for xb, yb in pbar:
                # Move the input and target tensors to the active computation device
                xb, yb = xb.to(device), yb.to(device)
                
                # Reset the gradients to zero before starting a new optimization step
                optimizer.zero_grad()
                
                # Generate model predictions for the current input batch
                logits = model(xb)
                
                # Flatten the logits and targets to compute the multi-class cross-entropy loss
                loss = loss_fn(
                    logits.reshape(-1, vocab_size),
                    yb.reshape(-1)
                )
                
                # Compute the gradient of the loss with respect to the model parameters
                loss.backward()
                
                # Rescale the gradients to prevent the exploding gradient problem during training
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Adjust the model weights based on the computed gradients
                optimizer.step()
                
                # Store the scalar loss value for later average calculation
                epoch_losses.append(loss.item())
                # Update the progress bar information with the current batch loss
                pbar.set_postfix(loss=loss.item())
        
        # Compute the arithmetic mean of the loss across all batches in the current epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        # Log the performance summary for the completed epoch
        print(f"Epoch {epoch+1:2d}: avg loss = {avg_loss:.4f}")



class ShakespeareTokenizer:
    """
    A specialized tokenizer for processing theatrical or poetic text.
    """
    def __call__(self, text):
        """
        Executes the tokenization process on the input text.

        Args:
            text (str): The raw text string to be processed.

        Returns:
            tokens (list): A list of strings containing words, contractions, 
                           line break markers, and punctuation marks.
        """
        # Substitute newline characters with a recognizable placeholder token
        text = text.replace('\n', ' <nl> ')
        # Apply regular expressions to extract alphanumeric words, possessives, and symbols
        return re.findall(r"\w+(?:'\w+)?|<nl>|[^\w\s]", text)



def build_vocabulary(text, vocab_size=5000, tokenizer=None):
    """
    Constructs a vocabulary and mapping dictionaries from the provided text.

    Args:
        text (str): The raw input text used to build the vocabulary.
        vocab_size (int): The maximum number of tokens to include in the vocabulary.
        tokenizer (object): An optional tokenizer instance for processing the text.

    Returns:
        vocab (list): A list of strings representing the unique vocabulary tokens.
        word2idx (dict): A dictionary mapping each token string to its integer index.
        idx2word (dict): A dictionary mapping each integer index back to its token string.
        tokenizer (object): The tokenizer instance used during the build process.
    """
    # Instantiate a default tokenizer if a specific one is not provided
    if tokenizer is None:
        tokenizer = ShakespeareTokenizer()
    
    # Process the text into a list of individual tokens
    tokens = tokenizer(text)
    # Calculate the frequency of each token in the tokenized list
    token_counts = Counter(tokens)
    
    # Define the set of reserved tokens for padding, unknown entries, and newlines
    special_tokens = ['<pad>', '<unk>', '<nl>']
    
    # Extract the most frequent tokens to fit within the specified vocabulary capacity
    most_common = token_counts.most_common(vocab_size - len(special_tokens))

    # Iterate through special tokens to ensure they are not duplicated in the common list
    for st in special_tokens:
        # Check if the special token is present in the most common list
        if st in most_common:
            # Remove the special token from the most common list to prevent overlap
            del most_common[st]
        
    # Initialize the vocabulary list with the special tokens
    vocab = special_tokens.copy()
    # Append the most frequent tokens to the vocabulary list
    for token, count in most_common:
        vocab.append(token)
    
    # Generate a lookup table for converting word strings into numerical indices
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    # Generate a lookup table for converting numerical indices back into word strings
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    # Determine the total volume of tokens in the original processed text
    total_token_occurrences = sum(token_counts.values())
    # Calculate the frequency of tokens that successfully appear in the built vocabulary
    covered_token_occurrences = sum(token_counts[token] for token in vocab if token in token_counts)
    # Determine the percentage of the text represented by the current vocabulary
    coverage = covered_token_occurrences / total_token_occurrences
    
    # Count the occurrences of tokens that fall outside the final vocabulary
    unknown_count = sum(count for token, count in token_counts.items() if token not in word2idx)
    # Determine the ratio of unknown tokens to the total number of tokens
    unknown_rate = unknown_count / total_token_occurrences
    
    # Log the summary statistics for the generated vocabulary to the console
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Unique tokens in text: {len(token_counts)}")
    print(f"Coverage: {coverage:.1%} of token occurrences")
    print(f"Unknown token rate: {unknown_rate:.1%}")
    print(f"Most common tokens: {vocab[3:13]}")
    print(f"Least common in vocab: {vocab[-10:]}")
    
    # Provide the vocabulary and the mapping tools for model use
    return vocab, word2idx, idx2word, tokenizer



def create_sequences(text, word2idx, idx2word, tokenizer=None, seq_len=150):
    """
    Generates overlapping input and target sequences for training language models.

    Args:
        text (str): The raw input string to be tokenized and processed.
        word2idx (dict): A dictionary mapping unique tokens to their integer indices.
        idx2word (dict): A dictionary mapping integer indices back to their token strings.
        tokenizer (object): An optional tokenizer instance; defaults to ShakespeareTokenizer if None.
        seq_len (int): The fixed length for each generated input sequence.

    Returns:
        inputs (list): A list of integer lists, where each inner list is an input sequence.
        targets (list): A list of integer lists, where each inner list is the corresponding target sequence.
    """
    # Instantiate a default tokenizer if a specific instance is not provided
    if tokenizer is None:
        tokenizer = ShakespeareTokenizer()
    
    # Transform the raw text into a complete list of tokens
    tokens = tokenizer(text)
    
    # Initialize containers for the processed numerical sequences
    inputs = []
    targets = []
    
    # Iterate through the tokens to create sliding window samples
    for i in range(len(tokens) - seq_len):
        # Define the current input window based on the sequence length
        window = tokens[i:i+seq_len]
        # Define the target window as the input window shifted by one token
        target = tokens[i+1:i+seq_len+1]
        
        # Map input tokens to their indices, using the unknown token as a fallback
        input_ids = [word2idx.get(w, word2idx['<unk>']) for w in window]
        # Map target tokens to their indices, using the unknown token as a fallback
        target_ids = [word2idx.get(w, word2idx['<unk>']) for w in target]
        
        # Append the encoded numerical sequences to the output lists
        inputs.append(input_ids)
        targets.append(target_ids)
    
    # Log the total count of sequences generated during the process
    print(f"Created {len(inputs)} sequences of length {seq_len}")
    
    # Validate the generated data by displaying a sample of the results
    if inputs:
        # Convert the first few indices of the first sample back to strings
        input_tokens = [idx2word[id] for id in inputs[0][:10]]
        target_tokens = [idx2word[id] for id in targets[0][:10]]
        
        # Display the human-readable tokens for verification
        print(f"Example input tokens: {input_tokens}...")
        print(f"Example target tokens: {target_tokens}...")
        
        # Check that the temporal shift between inputs and targets is correctly aligned
        if len(inputs[0]) > 5:
            print(f"\nVerifying shift:")
            for i in range(5):
                # Retrieve the strings for the current position in both sequences
                input_token = idx2word[inputs[0][i]]
                target_token = idx2word[targets[0][i]]
                # Identify the token that should theoretically be the next target
                expected = idx2word[inputs[0][i+1]] if i+1 < len(inputs[0]) else "N/A"
                print(f"  Position {i}: input='{input_token}' → target='{target_token}' (expected: '{expected}')")
    
    # Return the completed lists of input and target indices
    return inputs, targets



class ShakespeareDataset(Dataset):
    """
    A custom Dataset class for handling Shakespearean text sequences in PyTorch.
    """
    def __init__(self, inputs, targets):
        """
        Initializes the dataset with input and target sequences.

        Args:
            inputs (list): A list of integer sequences representing the input tokens.
            targets (list): A list of integer sequences representing the target tokens.
        """
        # Convert the input sequences into a PyTorch long tensor for processing
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        # Convert the target sequences into a PyTorch long tensor for processing
        self.targets = torch.tensor(targets, dtype=torch.long)
    
    def __len__(self):
        """
        Returns the total number of samples available in the dataset.

        Returns:
            length (int): The total count of sequences in the dataset.
        """
        # Calculate the size based on the number of rows in the input tensor
        return len(self.inputs)
    
    def __getitem__(self, idx):
        """
        Retrieves a specific sample and its corresponding target by index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            input_sample (Tensor): The input sequence tensor at the specified index.
            target_sample (Tensor): The target sequence tensor at the specified index.
        """
        # Return the input and target tensors for the given index
        return self.inputs[idx], self.targets[idx]



def create_dataloaders(inputs, targets, batch_size=32, train_split=0.9, shuffle=True):
    """
    Constructs training and validation DataLoaders from input and target sequences.

    Args:
        inputs (list): List of numerical input sequences for the dataset.
        targets (list): List of numerical target sequences for the dataset.
        batch_size (int): The number of samples per batch to load.
        train_split (float): The proportion of the dataset to include in the training split.
        shuffle (bool): Whether to reorganize the data at every epoch during training.

    Returns:
        train_loader (DataLoader): The DataLoader instance for the training set.
        val_loader (DataLoader): The DataLoader instance for the validation set, or None if no split.
        dataset (Dataset): The full initialized dataset object.
    """
    # Create the primary dataset object from the provided sequences
    dataset = ShakespeareDataset(inputs, targets)
    
    # Check if the data should be partitioned into training and validation sets
    if train_split < 1.0:
        # Calculate the number of samples designated for training
        train_size = int(train_split * len(dataset))
        # Calculate the remaining samples designated for validation
        val_size = len(dataset) - train_size
        # Perform a random split of the dataset into two subsets
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Initialize the training DataLoader with shuffling and batching enabled
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=0
        )
        
        # Initialize the validation DataLoader without shuffling
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        # Log the sizes and batch counts for the partitioned datasets
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Number of train batches: {len(train_loader)}")
        print(f"Number of val batches: {len(val_loader)}")
        
        # Return both loaders and the underlying dataset
        return train_loader, val_loader, dataset
    
    else:
        # Initialize a single DataLoader for the entire dataset when no split is required
        train_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=0
        )
        
        # Log the total dataset size and the resulting number of batches
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of batches: {len(train_loader)}")
        
        # Return the training loader, None for validation, and the full dataset
        return train_loader, None, dataset



def prepare_shakespeare_data(text_file_or_string, vocab_size=5000, seq_len=150, 
                            batch_size=32, train_split=0.9):
    """
    Executes the complete data preparation pipeline for sequence modeling.

    Args:
        text_file_or_string (str): A file path to a .txt file or the raw text string itself.
        vocab_size (int): The maximum number of unique tokens to include in the vocabulary.
        seq_len (int): The fixed length of the input sequences for training.
        batch_size (int): The number of samples per batch in the data loaders.
        train_split (float): The ratio of data used for training versus validation.

    Returns:
        data_bundle (dict): A dictionary containing the vocabulary list, mapping 
                            dictionaries, tokenizer, data loaders, and dataset metadata.
    """
    # Load the text content if the input is identified as a file path
    if isinstance(text_file_or_string, str) and text_file_or_string.endswith('.txt'):
        # Open the specified file and read its entire contents
        with open(text_file_or_string, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        # Use the provided string directly as the text source
        text = text_file_or_string
    
    # Initialize the first stage of the pipeline to define the word-to-index mappings
    print("Step 1: Building vocabulary...")
    # Construct the vocabulary and associated lookup dictionaries
    vocab, word2idx, idx2word, tokenizer = build_vocabulary(text, vocab_size)
    
    # Initialize the second stage to transform the raw text into training samples
    print(f"\nStep 2: Creating sequences (length={seq_len})...")
    # Generate input and target sequences shifted by one position
    inputs, targets = create_sequences(text, word2idx, idx2word, tokenizer, seq_len)
    
    # Initialize the third stage to prepare the data for efficient model consumption
    print(f"\nStep 3: Creating dataloaders (batch_size={batch_size})...")
    # Partition the data and wrap it in batch-processing loader objects
    train_loader, val_loader, dataset = create_dataloaders(
        inputs, targets, batch_size, train_split
    )
    
    # Consolidate all processed components and metadata into a single return dictionary
    return {
        'vocab': vocab,
        'word2idx': word2idx,
        'idx2word': idx2word,
        'vocab_size': len(vocab),
        'tokenizer': tokenizer,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'dataset': dataset,
        'seq_len': seq_len
    }



@torch.no_grad()
def generate_tokens(model, prompt_ids, max_length=100, temperature=1.0, 
                    top_k=50, top_p=0.95, repetition_penalty=1.2, 
                    eos_token_id=None, device='cpu'):
    """
    Implements an autoregressive token generation loop with various sampling heuristics.

    Args:
        model (nn.Module): The neural network used to compute logit predictions.
        prompt_ids (list or Tensor): Initial sequence of token identifiers to start generation.
        max_length (int): The total maximum length of the sequence including the prompt.
        temperature (float): Scaling factor to adjust the sharpness of the probability distribution.
        top_k (int): The number of highest probability tokens to keep for filtering.
        top_p (float): The cumulative probability threshold for nucleus sampling.
        repetition_penalty (float): Factor used to discount the logits of previously seen tokens.
        eos_token_id (int): Specific token identifier that signifies the end of a sequence.
        device (str or torch.device): The target hardware for computation.

    Returns:
        generated_sequence (Tensor): A flattened tensor containing the original prompt and generated tokens.
    """
    # Set the model to evaluation mode to ensure consistent behavior of layers like dropout
    model.eval()
    
    # Standardize the prompt input into a 2D batch tensor on the target device
    if isinstance(prompt_ids, list):
        prompt_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    elif len(prompt_ids.shape) == 1:
        prompt_ids = prompt_ids.unsqueeze(0).to(device)
    else:
        prompt_ids = prompt_ids.to(device)
    
    # Create a copy of the prompt to append new tokens to
    generated = prompt_ids.clone()
    # Maintain a local list of tokens for applying repetition penalties efficiently
    past_tokens = list(prompt_ids[0].cpu().numpy())
    
    # Iterate until the maximum sequence length is reached
    for step in range(max_length - len(prompt_ids[0])):
        # Perform a forward pass, potentially using mixed precision for performance
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits = model(generated)
        
        # Extract the logits for the most recently predicted position
        next_token_logits = logits[0, -1, :].float()
        
        # Adjust the distribution variance using the temperature parameter
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        
        # Reduce the likelihood of tokens that have already appeared in the sequence
        if repetition_penalty != 1.0:
            # Apply a penalty factor to every unique token already generated
            for token_id in set(past_tokens):
                next_token_logits[token_id] /= repetition_penalty
            
            # Apply an additional penalty for the three most recent tokens to avoid loops
            if len(past_tokens) > 3:
                for token_id in past_tokens[-3:]:
                    next_token_logits[token_id] /= 1.5
        
        # Zero out the probability of all tokens outside the top K highest scores
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, min(top_k, len(next_token_logits)))[0][-1]
            next_token_logits[indices_to_remove] = -float('inf')
        
        # Filter out tokens that fall outside the cumulative probability threshold
        if top_p < 1.0:
            # Sort the logits to calculate cumulative probabilities in descending order
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Identify indices where the cumulative probability exceeds the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift indices to ensure at least one token remains available for sampling
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Mask the logits of the tokens that are to be removed
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float('inf')
        
        # Calculate the final probability distribution and sample the next token
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        # Concatenate the new token to the existing sequence
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        # Update the history of generated tokens
        past_tokens.append(next_token.item())
        
        # Terminate the generation process if the model produces the end-of-sequence token
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
    
    # Remove the batch dimension before returning the final tensor
    return generated.squeeze(0)



def generate_text(model, prompt, tokenizer, word2idx, idx2word, 
                  max_length=100, temperature=0.8, top_k=50, top_p=0.95,
                  repetition_penalty=1.2, device='cpu'):
    """
    Transforms a string prompt into a generated sequence of text using sampling strategies.

    Args:
        model (nn.Module): The pre-trained model used for sequence prediction.
        prompt (str): The starting text string used to seed the generation.
        tokenizer (object): The tokenizer instance used to split the prompt into tokens.
        word2idx (dict): A dictionary mapping token strings to their numerical indices.
        idx2word (dict): A dictionary mapping numerical indices to their token strings.
        max_length (int): The maximum number of tokens to be generated.
        temperature (float): A value adjusting the randomness of the sampling distribution.
        top_k (int): The number of highest-probability tokens to consider during filtering.
        top_p (float): The cumulative probability threshold for nucleus sampling.
        repetition_penalty (float): The factor used to discourage the model from repeating tokens.
        device (str): The target hardware for executing the model operations.

    Returns:
        final_text (str): The decoded and formatted text string generated by the model.
    """
    # Evaluate the prompt and provide a default token if the input is empty or whitespace
    if not prompt or prompt.isspace():
        # Initialize with a common seed token
        prompt_tokens = ['the']
    else:
        # Split the input prompt into a list of lowercase tokens
        prompt_tokens = tokenizer(prompt.lower())
    
    # Initialize a list to store the numerical index for each prompt token
    prompt_ids = []
    # Map each string token to its corresponding vocabulary index
    for token in prompt_tokens:
        # Check if the exact token exists in the mapping dictionary
        if token in word2idx:
            prompt_ids.append(word2idx[token])
        else:
            # Attempt to find a lowercase version if the initial match fails
            token_lower = token.lower()
            if token_lower in word2idx:
                prompt_ids.append(word2idx[token_lower])
            else:
                # Use the designated unknown token for words not in the vocabulary
                prompt_ids.append(word2idx['<unk>'])
    
    # Ensure the sequence is not empty by appending a fallback token if necessary
    if not prompt_ids:
        prompt_ids = [word2idx.get('the', word2idx['<unk>'])]
    
    # Identify the index for the end-of-sequence token if it exists in the vocabulary
    eos_token_id = word2idx.get('<eos>', None)
    
    # Execute the core token generation loop with sampling parameters
    generated_ids = generate_tokens(
        model,
        prompt_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_token_id,
        device=device
    )
    
    # Initialize a list to hold the decoded string tokens
    tokens = []
    # Iterate through the generated numerical indices to recover strings
    for idx in generated_ids:
        # Ensure the index is a standard integer value
        idx_val = idx.item() if hasattr(idx, 'item') else idx
        # Retrieve the word associated with the index or use the unknown placeholder
        token = idx2word.get(idx_val, '<unk>')
        
        # Convert special newline tokens into actual line break characters
        if token == '<nl>' or token == '<newline>':
            tokens.append('\n')
        # Exclude technical special tokens from the final human-readable output
        elif token not in ['<pad>', '<unk>', '<eos>', '<start>']:
            tokens.append(token)
    
    # Combine the list of strings into a single sentence separated by spaces
    text = ' '.join(tokens)
    
    # Remove artificial spaces before common punctuation marks
    text = text.replace(' ,', ',').replace(' .', '.').replace(' !', '!')
    text = text.replace(' ?', '?').replace(' ;', ';').replace(' :', ':')
    # Consolidate spacing around apostrophes and quotes
    text = text.replace(' \'', '\'').replace('\' ', '\'')
    # Remove excessive spacing around line breaks
    text = text.replace(' \n ', '\n').replace('\n ', '\n')
    
    # Return the cleaned and formatted string with leading/trailing whitespace removed
    return text.strip()



def interactive_generation(model, tokenizer, word2idx, idx2word, device='cpu'):
    """
    Executes an interactive user loop for generating text based on manual prompts.

    Args:
        model (nn.Module): The trained language model used for text generation.
        tokenizer (object): The tokenizer instance used for processing text input.
        word2idx (dict): A dictionary mapping token strings to their numerical indices.
        idx2word (dict): A dictionary mapping numerical indices to their token strings.
        device (str): The target computation hardware for running the model.

    Returns:
        None: This function runs as a continuous loop and does not return a value.
    """
    # Display the header for the interactive session
    print("="*50)
    print("Interactive Shakespeare Text Generation")
    print("="*50)
    # Provide instructions for user interaction and system commands
    print("Enter a prompt to generate text (or 'quit' to exit)")
    print("Commands: 'temp=0.8' to set temperature, 'len=100' to set length")
    print("-"*50)
    
    # Initialize default hyperparameter values for the generation process
    temperature = 0.8
    max_length = 100
    
    # Enter the main interaction loop
    while True:
        # Capture and clean the user input from the console
        prompt = input("\nPrompt: ").strip()
        
        # Terminate the loop if the user enters the exit command
        if prompt.lower() == 'quit':
            break
        
        # Parse potential hyperparameter update commands from user input
        if prompt.startswith('temp='):
            try:
                # Extract and apply the new temperature value
                temperature = float(prompt[5:])
                print(f"Temperature set to {temperature}")
                continue
            except:
                # Handle non-numeric or invalid temperature inputs
                print("Invalid temperature")
                continue
        
        # Parse sequence length adjustment commands
        if prompt.startswith('len='):
            try:
                # Extract and apply the new maximum sequence length
                max_length = int(prompt[4:])
                print(f"Max length set to {max_length}")
                continue
            except:
                # Handle non-integer or invalid length inputs
                print("Invalid length")
                continue
        
        # Invoke the text generation pipeline using the current settings and prompt
        generated = generate_text(
            model, prompt, tokenizer, word2idx, idx2word,
            max_length=max_length,
            temperature=temperature,
            device=device
        )
        
        # Output the generated sequence with visual separators
        print("\n" + "="*50)
        print("Generated:")
        print("-"*50)
        print(generated)
        print("="*50)



def generate_batch(model, prompts, tokenizer, word2idx, idx2word, 
                   max_length=100, temperature=0.8, device='cpu'):
    """
    Generates text sequences for a collection of input prompts.

    Args:
        model (nn.Module): The pre-trained model used for sequence prediction.
        prompts (list): A list of strings used to seed the generation process.
        tokenizer (object): The tokenizer instance used to process the text.
        word2idx (dict): A dictionary mapping token strings to their numerical indices.
        idx2word (dict): A dictionary mapping numerical indices to their token strings.
        max_length (int): The maximum number of tokens to be generated for each prompt.
        temperature (float): A value adjusting the randomness of the sampling distribution.
        device (str): The target hardware for executing the model operations.

    Returns:
        results (list): A list of strings containing the generated text for each prompt.
    """
    # Initialize a list to store the generated text for each input prompt
    results = []
    # Iterate through each prompt provided in the input list
    for prompt in prompts:
        # Generate a text sequence based on the current prompt and parameters
        generated = generate_text(
            model, prompt, tokenizer, word2idx, idx2word,
            max_length=max_length,
            temperature=temperature,
            device=device
        )
        # Append the completed text sequence to the results collection
        results.append(generated)
    # Return the aggregated list of all generated sequences
    return results


    