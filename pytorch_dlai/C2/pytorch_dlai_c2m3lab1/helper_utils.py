import torch

def print_unique_token_id_mappings(tokens_list, input_ids_tensor):
    """
    Prints unique token to ID mappings from a list of tokenized sentences,
    sorted by token ID.

    Args:
        tokens_list (list of list of str): A list where each inner list
                                           contains token strings for a sentence.
        input_ids_tensor (torch.Tensor or similar): A tensor where each row
                                                    contains token IDs for a sentence.
    """
    unique_token_to_id_map = {}

    # Iterate through each sentence's tokens and IDs
    for i in range(len(tokens_list)):
        sentence_tokens = tokens_list[i]
        # Get the corresponding tensor of IDs for the current sentence
        sentence_ids = input_ids_tensor[i]
        for j in range(len(sentence_tokens)):
            token_str = sentence_tokens[j]
            # .item() converts a single-element tensor to a Python number
            token_id = sentence_ids[j].item()
            # Store in dictionary to ensure uniqueness of token strings
            unique_token_to_id_map[token_str] = token_id

    # Sort the unique mappings by token ID (the value in the dictionary)
    sorted_token_mappings = sorted(unique_token_to_id_map.items(), key=lambda item: item[1])

    print("\n--- Unique Token to ID Mappings (for these sentences) ---")
    if not sorted_token_mappings:
        print("No tokens to display.")
        return

    for token, token_id in sorted_token_mappings:
        print(f"{token}\t-->\t{token_id}")
        
    print() # To add an extra line