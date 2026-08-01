import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class UnitTestDataset(Dataset):
    """A minimal, self-contained dataset for unit testing text classification."""
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, max_length=128)
        encoding['labels'] = torch.tensor(label, dtype=torch.long)
        return encoding



def load_tokenizer(local_path="./distilbert-local-base"):
    """
    Loads a Hugging Face tokenizer from a local directory.
    """
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    return tokenizer



class MockFullDataset:
    def __init__(self, labels):
        self.labels = labels



def load_bert_model(local_path="./distilbert-local-base", num_classes=2):
    """
    Loads the base DistilBERT model and adds a new classification head."""
    # Use AutoModelForSequenceClassification to load the base model and add a new,
    # randomly initialized classification head on top.
    
    from transformers import logging
    logging.set_verbosity_error()
    
    model = AutoModelForSequenceClassification.from_pretrained(
        local_path, 
        num_labels=num_classes
    )
    
    return model