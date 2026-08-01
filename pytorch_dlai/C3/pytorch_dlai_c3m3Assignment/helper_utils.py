import zipfile
from pathlib import Path
from typing import List, Tuple
import random
from tqdm.auto import tqdm
import torch
import matplotlib.pyplot as plt
# Define available language pairs
LANGUAGE_PAIRS = {
    'French': {'code': 'fra', 'file': 'fra.txt'},
    'Spanish': {'code': 'spa', 'file': 'spa.txt'},
    'German': {'code': 'deu', 'file': 'deu.txt'},
    'Italian': {'code': 'ita', 'file': 'ita.txt'},
    'Portuguese': {'code': 'por', 'file': 'por.txt'},
    'Russian': {'code': 'rus', 'file': 'rus.txt'}
}

def load_dataset(languages_dir: str = './languages') -> Tuple[List[Tuple[str, str]], str]:
    """
    Interactively load a translation dataset.
    
    Args:
        languages_dir: Directory containing language files
        
    Returns:
        Tuple of (translation_pairs, selected_language_name)
        where translation_pairs is a list of tuples like [('Go.', 'Vai.'), ...]
    """
    # Display available languages
    print("Available translation pairs (to/from English):")
    for i, lang in enumerate(LANGUAGE_PAIRS.keys(), 1):
        print(f"{i}. English ↔ {lang}")
    
    # Let user choose
    while True:
        try:
            choice = int(input("\nSelect a language (enter number): "))
            if 1 <= choice <= len(LANGUAGE_PAIRS):
                target_language = list(LANGUAGE_PAIRS.keys())[choice - 1]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"\nYou selected: English ↔ {target_language}")
    lang_info = LANGUAGE_PAIRS[target_language]
    
    # Set up paths
    languages_dir = Path(languages_dir)
    dataset_file = languages_dir / lang_info['file']
    zip_file = languages_dir / f"{lang_info['code']}-eng.zip"
    
    # Extract if needed
    if not dataset_file.exists():
        if zip_file.exists():
            print(f"Extracting {target_language} dataset...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(languages_dir)
            print("Extraction complete!")
        else:
            print(f"Error: {zip_file} not found in languages folder!")
            print(f"Please ensure {lang_info['code']}-eng.zip is in the languages folder.")
            return [], target_language
    else:
        print(f"{target_language} dataset already extracted, loading...")
    
    # Load the translation pairs as list of tuples
    translation_pairs = []  # This will be a list of tuples
    if dataset_file.exists():
        with open(dataset_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    # Creating tuple (English, Target Language)
                    translation_pairs.append((parts[0], parts[1]))
    
    print(f"Loaded {len(translation_pairs)} English-{target_language} translation pairs")
    
    # Display random examples
    print(f"\nRandom sample English-{target_language} pairs:")
    num_samples = min(5, len(translation_pairs))
    random_pairs = random.sample(translation_pairs, num_samples)
    for eng, target in random_pairs:
        print(f"English: {eng}")
        print(f"{target_language}: {target}")
        print("-" * 50)
    
    return translation_pairs, target_language


import re
from typing import List, Tuple

import re
from typing import List, Tuple

class MultilingualTokenizer:
    """
    Tokenizer that can handle multiple languages
    """
    def __init__(self, language='French'):
        self.language = language
    
    def __call__(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Language-specific handling
        if self.language == 'Russian':
            # Keep Cyrillic characters
            text = re.sub(r"([.!?])", r" \1", text)
            tokens = re.findall(r"[\w]+|[.!?]", text)
        else:
            # Improved handling for Latin-based languages
            # First, add spaces around punctuation (but not apostrophes within words)
            text = re.sub(r"([.!?])", r" \1", text)
            # Find words (including contractions) and punctuation
            # This pattern captures:
            # - Words with internal apostrophes (contractions)
            # - Regular words
            # - Punctuation marks
            tokens = re.findall(r"\b\w+(?:'\w+)*\b|[.!?]", text)
        
        return tokens


def normalize_string(s, language='French'):
    """
    Normalize a string based on the target language
    """
    # Convert to lowercase
    s = s.lower().strip()
    
    # Language-specific normalization
    if language == 'Russian':
        # Keep Cyrillic characters
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^а-яА-Яa-zA-Z.!?]+", r" ", s)
    elif language in ['French', 'Spanish', 'Portuguese', 'Italian']:
        # Keep Latin characters with accents and apostrophes for contractions
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-ZÀ-ÿ'.!?]+", r" ", s)  # Added apostrophe
    elif language == 'German':
        # Keep German special characters and apostrophes
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-ZÄäÖöÜüß'.!?]+", r" ", s)  # Added apostrophe
    else:
        # Default handling (including English) - keep apostrophes for contractions
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z'.!?]+", r" ", s)  # Added apostrophe
    
    # Remove extra spaces
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def prepare_data(translation_pairs: List[Tuple[str, str]], 
                 target_language: str, 
                 max_pairs: int = 15000, 
                 max_length: int = 20) -> Tuple[List[Tuple[str, str]], object]:
    """
    Prepare translation data by normalizing text and creating a tokenizer.
    
    Args:
        translation_pairs: List of (English, target_language) translation pairs
        target_language: Name of the target language
        max_pairs: Maximum number of pairs to process (default: 15000)
        max_length: Maximum sentence length in words (default: 20)
        
    Returns:
        Tuple of (normalized_pairs, tokenizer)
    """
    # Apply normalization to all pairs
    normalized_pairs = []
    
    for eng, target in translation_pairs[:max_pairs]:
        eng_norm = normalize_string(eng, 'English')
        target_norm = normalize_string(target, target_language)
        
        # Filter by length for more manageable training
        if len(eng_norm.split()) <= max_length and len(target_norm.split()) <= max_length:
            normalized_pairs.append((eng_norm, target_norm))
    
    # Create tokenizer for the selected language
    tokenizer = MultilingualTokenizer(target_language)
    
    # Test the tokenizer with examples containing contractions
    print(f"\n=== Tokenizer Test for {target_language} ===")
    
    # Test with English contractions
    test_sentences = [
        "Hello, how are you?",
        "He's going to the store.",
        "I can't believe it's working!",
        "They're here, aren't they?"
    ]
    
    for test_sent in test_sentences:
        tokens = tokenizer(test_sent)
        print(f"Original: {test_sent}")
        print(f"Tokenized: {tokens}")
        print()
    
    # Test with a sample from the target language if available
    if normalized_pairs:
        sample_target = normalized_pairs[0][1]  # Get first target language sentence
        print(f"\n{target_language} sample: {sample_target}")
        print(f"{target_language} tokenized: {tokenizer(sample_target)}")
    
    print(f"\n=== Data Preparation Complete ===")
    print(f"Normalized pairs: {len(normalized_pairs)} (from {min(max_pairs, len(translation_pairs))} original pairs)")
    
    return normalized_pairs, tokenizer


def show_model_layers(model):
    """
    Display the 4 main layers of the TranslationEncoder model.
    """
    print("\n" + "=" * 70)
    print(f" {model.__class__.__name__} - Main Layers")
    print("=" * 70)
    print(f"\n{'Layer':<30} {'Type':<25} {'Parameters':>15}")
    print("-" * 70)
    
    # Show the 4 main layers
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        module_type = module.__class__.__name__
        print(f"{name:<30} {module_type:<25} {params:>15,}")
    
    print("-" * 70)
    total = sum(p.numel() for p in model.parameters())
    print(f"{'TOTAL':<30} {'':<25} {total:>15,}")
    print("=" * 70)

def show_decoder_layers(model):
    """
    Display the main layers of the Decoder model.
    """
    print("\n" + "=" * 70)
    print(f" {model.__class__.__name__} - Main Layers")
    print("=" * 70)
    print(f"\n{'Layer':<30} {'Type':<25} {'Parameters':>15}")
    print("-" * 70)
    
    # Show the main layers
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        module_type = module.__class__.__name__
        print(f"{name:<30} {module_type:<25} {params:>15,}")
    
    print("-" * 70)
    total = sum(p.numel() for p in model.parameters())
    print(f"{'TOTAL':<30} {'':<25} {total:>15,}")
    print("=" * 70)

def show_encoderdecoder_layers(model):
    """
    Display the main components of the EncoderDecoder model.
    """
    print("\n" + "=" * 70)
    print(f" {model.__class__.__name__} - Main Components")
    print("=" * 70)
    print(f"\n{'Component':<30} {'Type':<25} {'Parameters':>15}")
    print("-" * 70)
    
    # Show the main components
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        module_type = module.__class__.__name__
        print(f"{name:<30} {module_type:<25} {params:>15,}")
    
    print("-" * 70)
    total = sum(p.numel() for p in model.parameters())
    print(f"{'TOTAL':<30} {'':<25} {total:>15,}")
    print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=10):
    """
    Simple training function for the translator
    """
    # Store history
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_total_loss = 0
        train_batches = 0
        
        # Progress bar for training
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for src_batch, tgt_batch in train_bar:
            # Move to device
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            
            # Prepare decoder input and target
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(src_batch, tgt_input)
            
            # Reshape for loss calculation
            outputs = outputs.reshape(-1, outputs.size(-1))
            tgt_output = tgt_output.reshape(-1)
            
            loss = criterion(outputs, tgt_output)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track loss
            train_total_loss += loss.item()
            train_batches += 1
            
            # Update progress bar
            avg_loss = train_total_loss / train_batches
            train_bar.set_postfix({'loss': f'{loss.item():.3f}'})
        
        # Validation phase
        model.eval()
        val_total_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for src_batch, tgt_batch in val_loader:
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)
                
                tgt_input = tgt_batch[:, :-1]
                tgt_output = tgt_batch[:, 1:]
                
                outputs = model(src_batch, tgt_input)
                outputs = outputs.reshape(-1, outputs.size(-1))
                tgt_output = tgt_output.reshape(-1)
                
                loss = criterion(outputs, tgt_output)
                
                val_total_loss += loss.item()
                val_batches += 1
        
        # Calculate average losses
        train_loss = train_total_loss / train_batches
        val_loss = val_total_loss / val_batches
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Print epoch summary
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}\n')
    
    return history


def plot_training_history(history):
    """
    Plot training and validation loss from training history.
    
    Args:
        history: Dictionary containing 'train_loss' and 'val_loss' lists
    """
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'],
             label='Training Loss', marker='o')
    plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'],
             label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print final results
    print(f"Final Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
    print(f"Best Validation Loss: {min(history['val_loss']):.4f}")
    best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
    print(f"Best Validation Loss at Epoch: {best_epoch}")