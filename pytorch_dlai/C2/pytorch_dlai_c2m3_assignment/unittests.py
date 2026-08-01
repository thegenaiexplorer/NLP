import collections.abc
import math
from types import FunctionType

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import transformers

from dlai_grader.grading import print_feedback, test_case
import unittests_utils


def exercise_1(learner_class):
    def g():
        cases = []

        class_name = "InstructionDataset"

        # ############################ Test case 1: Check if the learner's submission is a class ############################
        
        t = test_case()
        if not isinstance(learner_class, type):
            t.failed = True
            t.msg = f"{class_name} has incorrect type"
            t.want = f"a Python class called {class_name}"
            t.got = type(learner_class)
            return [t]

        t = test_case()
        if learner_class.__base__ != Dataset:
            t.failed = True
            t.msg = f"{class_name} didn't inherit from the correct class"
            t.want = Dataset
            t.got = learner_class.__base__
            return [t]
        ############################
        
        # Setup: Create dummy data for testing
        dummy_texts = ["this is the first sentence", "and this is the second"]
        dummy_labels = [0, 1]
        
        # ############################ Test case 2: Test instantiation and __init__ method ############################
        tokenizer = unittests_utils.load_tokenizer()
        try:
            dataset = learner_class(dummy_texts, dummy_labels, tokenizer)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Failed to instantiate your {class_name} class. Error: {e}"
            return [t]

        
        # Check if self.texts is initialized correctly
        t = test_case()
        if not hasattr(dataset, 'texts') or not isinstance(dataset.texts, list):
            t.failed = True
            t.msg = "The `self.texts` attribute should be initialized as a list in the __init__ method"
            t.want = "self.texts = texts"
            t.got = "Attribute `self.texts` is missing or not a list."
            cases.append(t)
        elif not all(isinstance(i, str) for i in dataset.texts):
            t.failed = True
            t.msg = "All elements in the `self.texts` attribute should be strings"
            t.want = "A list of strings (self.texts = texts)"
            t.got = f"A list containing elements of type {[type(i) for i in dataset.texts]}."
            cases.append(t)

         # Check if self.labels is initialized correctly
        t = test_case()
        if not hasattr(dataset, 'labels') or not isinstance(dataset.labels, list):
            t.failed = True
            t.msg = "The `self.labels` attribute should be initialized as a list in the __init__ method"
            t.want = "self.labels = labels"
            t.got = "Attribute `self.labels` is missing or not a list."
            cases.append(t)
        elif not all(isinstance(i, int) for i in dataset.labels):
            t.failed = True
            t.msg = "All elements in the `self.labels` attribute should be integers"
            t.want = "A list of integers (self.labels = labels)"
            t.got = f"A list containing elements of type {[type(i) for i in dataset.labels]}."
            cases.append(t)

        # Check if self.tokenizer is initialized correctly
        t = test_case()
        # A good proxy for checking if it's a valid tokenizer is to check for a 'pad_token_id' attribute
        if not hasattr(dataset, 'tokenizer') or not hasattr(dataset.tokenizer, 'pad_token_id'):
            t.failed = True
            t.msg = "The `self.tokenizer` attribute was not initialized correctly"
            t.want = "self.tokenizer = tokenizer"
            t.got = "Attribute `tokenizer` is missing or is not a valid tokenizer object."
            cases.append(t)

        # If any of the above tests failed, it's best to stop here.
        if cases: return cases

        ############################

        # ############################ Test case 3: Test the __len__ method ############################
        t = test_case()
        expected_len = len(dummy_texts)
        if len(dataset) != expected_len:
            t.failed = True
            t.msg = "The __len__ method returned an incorrect length. Make sure it is returning the 'length' of 'self.texts'"
            t.want = expected_len
            t.got = len(dataset)
        cases.append(t)
        
        ############################
            
        # ############################ Test case 4: Test the __getitem__ method ############################
        try:
            sample = dataset[0]
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"The __getitem__ method failed to retrieve a sample. Error: {e}"
            return [t]
        
        # --- 2. Structural Checks ---
        # Check if the returned object is transformers.BatchEncoding (a Mapping)
        t = test_case()
        if not isinstance(sample, collections.abc.Mapping):
            t.failed = True
            t.msg = "The __getitem__ method should return a transformers.BatchEncoding object"
            t.want = "transformers.BatchEncoding"
            t.got = type(sample)
            return cases + [t] # Stop here if it's not a mapping
        
        # Check for required keys
        t = test_case()
        expected_keys = {'input_ids', 'attention_mask', 'labels'}
        if not expected_keys.issubset(sample.keys()):
            t.failed = True
            t.msg = "The transformers.BatchEncoding object returned by __getitem__ is missing required keys"
            t.want = f"An object with keys: {expected_keys}"
            t.got = f"An object with keys: {sample.keys()}"
        cases.append(t)
        
        # --- 3. Implementation Logic Checks
        # Test if the correct text and label are being fetched for a given index
        t = test_case()
        try:
            sample_at_idx_1 = dataset[1]
            expected_text_at_idx_1 = dummy_texts[1]
            expected_label_at_idx_1 = dummy_labels[1]
            expected_input_ids = tokenizer(expected_text_at_idx_1)['input_ids']
            
            if sample_at_idx_1['input_ids'] != expected_input_ids:
                t.failed = True
                t.msg = "The `text` retrieved in `__getitem__` seems incorrect for the given index"
                t.want = f"input_ids for '{expected_text_at_idx_1}'"
                t.got = "input_ids for a different text."
                cases.append(t)
                
            if sample_at_idx_1['labels'].item() != expected_label_at_idx_1:
                t.failed = True
                t.msg = "The `label` retrieved in `__getitem__` seems incorrect for the given index"
                t.want = f"Label value {expected_label_at_idx_1}"
                t.got = f"Label value {sample_at_idx_1['labels'].item()}"
                cases.append(t)
        except IndexError:
            t.failed = True
            t.msg = "Failed to access index 1. Ensure your __getitem__ uses the `idx` parameter correctly"
            cases.append(t)
        
        # Test if the tokenizer is called with truncation
        t = test_case()
        long_text = "word " * 600
        temp_dataset = learner_class([long_text], [0], tokenizer)
        truncated_sample = temp_dataset[0]
        truncated_length = len(truncated_sample['input_ids'])
        
        if truncated_length != 512:
            t.failed = True
            t.msg = "The tokenizer truncation is not correctly configured. Make sure you set `truncation=True` and `max_length=512`"
            t.want = "A sequence length of 512"
            t.got = f"A sequence length of {truncated_length}."
        cases.append(t)
        
        # --- 4. Final Content Validation ---
        # Check the 'labels' value type and dtype
        t = test_case()
        labels_tensor = sample.get('labels')
        if not isinstance(labels_tensor, torch.Tensor):
            t.failed = True
            t.msg = "The 'labels' value should be a torch.Tensor"
            t.want = torch.Tensor
            t.got = type(labels_tensor)
            cases.append(t)
        elif labels_tensor.dtype != torch.long:
            t.failed = True
            t.msg = "The 'labels' tensor should have a dtype of torch.long"
            t.want = torch.long
            t.got = labels_tensor.dtype
            cases.append(t)
        
        # Check the 'labels' value content
        t = test_case()
        expected_label_val = dummy_labels[0]
        if labels_tensor.item() != expected_label_val:
            t.failed = True
            t.msg = "The 'labels' tensor contains an incorrect value"
            t.want = expected_label_val
            t.got = labels_tensor.item()
            cases.append(t)
        
        ############################

        return cases

    cases = g()
    print_feedback(cases)


######################################## End of Excersie 1 Test ########################################


def exercise_2(learner_func):
    def g():
        cases = []
        
        # ############################ Test case 1: Check if the learner's submission is a function ############################
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "create_data_collator has inccorect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        # Call the learner's function to get the collator
        tokenizer = unittests_utils.load_tokenizer()
        try:
            collator = learner_func(tokenizer)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Your function failed to run. Error: {e}"
            return [t]

        # ############################ Test case 2: Check the return type ############################
        t = test_case()
        if not isinstance(collator, transformers.DataCollatorWithPadding):
            t.failed = True
            t.msg = "collator has incorrect return type"
            t.want = transformers.DataCollatorWithPadding
            t.got = type(collator)
            return [t]
            
        # ############################ Test case 3: Check the functionality of the collator ############################
        # Create a dummy batch with tokenized sequences of different lengths
        dummy_batch = [
            tokenizer("this is a test"),
            tokenizer("this is a much longer test sentence"),
            tokenizer("short")
        ]
        
        try:
            padded_batch = collator(dummy_batch)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"The returned collator failed to process a batch. Error: {e}"
            return [t]

        # Check if the input_ids are padded to the same length
        t = test_case()
        seq_lengths = [len(seq) for seq in padded_batch['input_ids']]
        if len(set(seq_lengths)) != 1:
            t.failed = True
            t.msg = "The collator did not pad the `input_ids` to the same length."
            t.want = "All sequences in a batch to have the same length."
            t.got = f"A batch with sequence lengths: {seq_lengths}"
        cases.append(t)
            
        # Check if the attention_mask is also padded to the same length
        t = test_case()
        mask_lengths = [len(mask) for mask in padded_batch['attention_mask']]
        if len(set(mask_lengths)) != 1:
            t.failed = True
            t.msg = "The collator did not pad the `attention_mask` to the same length."
            t.want = "All attention masks in a batch to have the same length."
            t.got = f"A batch with mask lengths: {mask_lengths}"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)

######################################## End of Excersie 2 Test ########################################


def exercise_3(learner_func):
    def g():
        
        cases = []

        # ############################ Test case 1: Check if the learner's submission is a function ############################
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "create_dataloaders has inccorect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        ############################

        # Setup: Load tokenizer. Create dummy data, datasets, and the collate function
        tokenizer = unittests_utils.load_tokenizer()
        dummy_texts = ["short", "this is a longer sentence", "medium one", "a"] * 50
        dummy_labels = [0, 1, 0, 1] * 50
        full_dataset = unittests_utils.UnitTestDataset(dummy_texts, dummy_labels, tokenizer)
        
        train_dataset = torch.utils.data.Subset(full_dataset, range(125))
        val_dataset = torch.utils.data.Subset(full_dataset, range(125, 200))
        
        collate_fn = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

        # Call the learner's function
        batch_size = 8
        try:
            train_loader, val_loader = learner_func(
                train_dataset, val_dataset, batch_size, collate_fn
            )
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Your function failed to run. Error: {e}"
            return [t]

        # ############################ Test case 2: Check return types ############################
        t = test_case()
        if not isinstance(train_loader, DataLoader):
            t.failed = True
            t.msg = "train_loader has incorrect return type"
            t.want = DataLoader
            t.got = type(train_loader)
            return [t]

        t = test_case()
        if not isinstance(val_loader, DataLoader):
            t.failed = True
            t.msg = "val_loader has incorrect return type"
            t.want = DataLoader
            t.got = type(val_loader)
            return [t]
        
        ############################

        # ############################ Test case 3: Collate, Batch sizes, Dataloader length ############################
        
        # ############## 3A: Collate ##############
        # Collate function check 1 for train_loader
        try:
            # Attempt to get the first batch from the train_loader
            train_batch = next(iter(train_loader))
            
        except RuntimeError as e:
            # Check if the error is the specific one caused by missing padding
            if "each element in list of batch should be of equal size" in str(e):
                t = test_case()
                t.failed = True
                t.msg = "Failed to create a train batch. This likely means the `collate_fn` was not passed (correctly) to the train_loader"
                t.want = "train_loader to use collate_fn=collate_fn"
                t.got = f"train_loader raised a RuntimeError: {e}"
                return [t]
            else:
                # If it's a different RuntimeError, fail with the generic error
                t = test_case()
                t.failed = True
                t.msg = f"Your train_loader failed to create a batch. Error: {e}"
                return [t]

        # Collate function check 1 for val_loader
        try:
            # Attempt to get the first batch from the val_loader
            val_batch = next(iter(val_loader))
            
        except RuntimeError as e:
            if "each element in list of batch should be of equal size" in str(e):
                t = test_case()
                t.failed = True
                t.msg = "Failed to create a val batch. This likely means the `collate_fn` was not passed (correctly) to the val_loader"
                t.want = "val_loader to use collate_fn=collate_fn"
                t.got = f"val_loader raised a RuntimeError: {e}"
                return [t]
            else:
                t = test_case()
                t.failed = True
                t.msg = f"Your val_loader failed to create a batch. Error: {e}"
                return [t]


        # Collate function check 2 for train_loader
        t = test_case()
        seq_lengths = [len(seq) for seq in train_batch['input_ids']]
        if len(set(seq_lengths)) != 1:
            t.failed = True
            t.msg = "Sequences in a train_loader batch have different lengths. This suggests the collate_fn is not being used correctly for padding."
            t.want = "All sequences in a batch to have the same length"
            t.got = f"A batch with sequence lengths: {seq_lengths}"
            return [t]


        # Collate function check 2 for val_loader
        t = test_case()
        seq_lengths = [len(seq) for seq in val_batch['input_ids']]
        if len(set(seq_lengths)) != 1:
            t.failed = True
            t.msg = "Sequences in a val_loader batch have different lengths. This suggests the collate_fn is not being used correctly for padding."
            t.want = "All sequences in a batch to have the same length"
            t.got = f"A batch with sequence lengths: {seq_lengths}"
            return [t]

        ##############

        # ############## 3B: Batch Sizes ##############
        
        t = test_case()
        if train_batch['input_ids'].shape[0] != batch_size:
            t.failed = True
            t.msg = "The batch size for train_loader is incorrect. Please ensure you are using the batch_size parameter"
            t.want = "batch_size = batch_size"
            t.got = f"batch_size = {train_batch['input_ids'].shape[0]}"
            return [t]

        t = test_case()
        if val_batch['input_ids'].shape[0] != batch_size:
            t.failed = True
            t.msg = "The batch size for val_loader is incorrect. Please ensure you are using the batch_size parameter"
            t.want = "batch_size = batch_size"
            t.got = f"batch_size = {val_batch['input_ids'].shape[0]}"
            return [t]

        ##############
        
        # ############## 3C: Check DataLoader lengths ##############
        expected_train_len = math.ceil(len(train_dataset) / batch_size)
        expected_val_len = math.ceil(len(val_dataset) / batch_size)


        t = test_case()
        if len(train_loader) != expected_train_len:
            t.failed = True
            t.msg = "Incorrect length for train_loader. Make sure you are passing train_dataset to the train_loader"
            t.want = f"train_loader to use train_dataset. Expected train_loader length: {expected_train_len}"
            t.got = len(train_loader)
        cases.append(t)

        t = test_case()
        if len(val_loader) != expected_val_len:
            t.failed = True
            t.msg = "Incorrect length for val_loader. Make sure you are passing val_dataset to the val_loader"
            t.want = f"val_loader to use val_dataset. Expected val_loader length: {expected_val_len}"
            t.got = len(val_loader)
        cases.append(t)

        ##############

        # ############################


        # ############################ Test case 4: Check shuffle parameter implementation ############################
        t = test_case()
        if not isinstance(train_loader.sampler, RandomSampler):
            t.failed = True
            t.msg = "The train_loader is not shuffling the data. Please make sure you set shuffle=True"
            t.want = "shuffle=True"
            t.got = "shuffle=False or shuffle=None"
        cases.append(t)

        t = test_case()
        if not isinstance(val_loader.sampler, SequentialSampler):
            t.failed = True
            t.msg = "The val_loader should not shuffle the data. Please make sure you set shuffle=False"
            t.want = "shuffle=False"
            t.got = "shuffle=True or shuffle=None"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)

######################################## End of Excersie 3 Test ########################################


def exercise_4(learner_func):
    def g():
        cases = []
        
        # ############################ Test case 1: Check if the learner's submission is a function ############################
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "calculate_class_weights has inccorect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        ############################

        # Setup: Create a dummy imbalanced dataset
        # Class 0: 80 samples, Class 1: 20 samples
        dummy_labels = [0] * 80 + [1] * 20
        full_dataset = unittests_utils.MockFullDataset(dummy_labels)
        
        # The function expects a Subset object, so we create one
        train_dataset = torch.utils.data.Subset(full_dataset, range(len(dummy_labels)))
        device = torch.device('cpu')

        # Call the learner's function
        try:
            weights_tensor = learner_func(train_dataset, device)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Your function failed to run. Error: {e}"
            return [t]

        # ############################ Test case 2: Check return type and device ############################
        t = test_case()
        if not isinstance(weights_tensor, torch.Tensor):
            t.failed = True
            t.msg = "class_weights_tensor has incorrect return type"
            t.want = torch.Tensor
            t.got = type(weights_tensor)
            return [t]
            
        t = test_case()
        if weights_tensor.device.type != 'cpu':
            t.failed = True
            t.msg = "The returned tensor is on the wrong device."
            t.want = "class_weights_tensor.to(device)"
            t.got = "class_weights_tensor set to a specific device"
        cases.append(t)

        ############################

        # ############################ Test case 3: Check tensor (class_weights_tensor) shape and dtype ############################
        t = test_case()
        if weights_tensor.shape != (2,):
            t.failed = True
            t.msg = "The shape of the `class_weights_tensor` is incorrect"
            t.want = f"Shape: {(2,)}. A 1D PyTorch tensor created from the `class_weights` array."
            t.got = f"A tensor with shape {weights_tensor.shape}"
            return cases + [t]
        cases.append(t)

        t = test_case()
        if weights_tensor.dtype != torch.float:
            t.failed = True
            t.msg = "class_weights_tensor has an incorrect dtype"
            t.want = torch.float
            t.got = weights_tensor.dtype
            return cases + [t]
        cases.append(t)
            
        ############################

        # ############################ Test case 4: Check correctness of compute_class_weight parameters ############################
        # Calculate the correct weights for comparison
        expected_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(dummy_labels),
            y=dummy_labels
        )
        expected_weights_tensor = torch.tensor(expected_weights, dtype=torch.float)
        
        # --- Test 4A: Check if class_weight='balanced' was used ---
        t = test_case()
        # If 'balanced' is not used, weights will be [1., 1.]. We check that the result is NOT this.
        incorrect_weights = torch.ones(len(np.unique(dummy_labels)))
        if torch.allclose(weights_tensor, incorrect_weights):
            t.failed = True
            t.msg = "The calculated weights are not as expected. This suggests the `class_weight` parameter was not set to 'balanced'."
            t.want = "class_weight set as 'balanced'"
            t.got = "class_weight not set as 'balanced'"
            return cases + [t]
        cases.append(t)
        
        # --- Test 4B: Check if `classes` parameter is correct ---
        t = test_case()
        # The number of weights should match the number of unique classes.
        if len(weights_tensor) != len(np.unique(dummy_labels)):
            t.failed = True
            t.msg = "The number of calculated weights does not match the number of unique classes. This likely means the `classes` parameter was set incorrectly"
            t.want = "classes = NumPy array containing unique class labels from `train_labels_list`"
            t.got = "classes = unexpected class labels"
        cases.append(t)

        
        # --- Test 4C: Check if `y` parameter is correct ---
        t = test_case()
        # The most definitive test. If the final weights are wrong, the `y` parameter is the most likely cause.
        if not torch.allclose(weights_tensor, expected_weights_tensor):
            t.failed = True
            t.msg = "The final calculated class weights are incorrect. This strongly suggests the `y` parameter (the list of all labels) was not passed correctly to `compute_class_weight`."
            t.want = "y=train_labels_list"
            t.got = "Unexpected list of labels set as `y`"
        cases.append(t)

        ############################

        return cases

    cases = g()
    print_feedback(cases)


######################################## End of Excersie 4 Test ########################################



def exercise_5(learner_func):
    def g():
        cases = []

        # ############################ Test case 1: Check if the learner's submission is a function ############################
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "partially_freeze_bert_layers has inccorect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        ############################
        
        num_classes = 2
        layers_to_train = 5
        
        # Load a base model for the test
        original_model = unittests_utils.load_bert_model(num_classes=num_classes)
            
        # Call the learner's function
        try:
            learner_model = learner_func(original_model, layers_to_train=layers_to_train)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Your function failed to run. Error: {e}"
            return [t]

        # ############################ Test case 2: Check if the returned model is of the correct type ############################
        t = test_case()
        if not isinstance(learner_model, transformers.DistilBertForSequenceClassification):
            t.failed = True
            t.msg = "model has incorrect return type"
            t.want = transformers.DistilBertForSequenceClassification
            t.got = type(learner_model)
            return [t]

        ############################

        # ############################ Test case 3: Check if the initial freeze was performed correctly ############################
        # Check an early layer that should always be frozen
        t = test_case()
        early_layer_params = learner_model.distilbert.transformer.layer[0].parameters()
        if any(p.requires_grad for p in early_layer_params):
            t.failed = True
            t.msg = "The 'Freeze All Parameters' step was not implemented correctly, as early transformer layer(s) is still trainable"
            t.want = "A loop that sets `param.requires_grad = False` for all parameters in `model`."
            t.got = "A model where at least one parameter in an early layer has `requires_grad = True`."
            return [t]

        ############################

        # ############################ Test case 4: Check if the correct final layers were unfrozen ############################
        # Check the last N layers that should be trainable
        for i in range(layers_to_train):
            layer_index = -(i + 1)
            t = test_case()
            layer_to_check = learner_model.distilbert.transformer.layer[layer_index]
            if not all(p.requires_grad for p in layer_to_check.parameters()):
                t.failed = True
                t.msg = f"The 'Unfreeze the Last N Transformer Layers' step is incorrect, as layer {layer_index} (which should be trainable) is still frozen"
                t.want = "A loop that correctly selects the final layers and sets their parameters' `requires_grad` attribute to `True`"
                t.got = f"Layer {layer_index} has at least one parameter with `requires_grad = False`."
                return [t]

        ############################

        # ############################ Test case 5: Boundary Check - Ensure layers before the unfreeze block are still frozen ############################
        boundary_layer_index = -(layers_to_train + 1)
        t = test_case()
        boundary_layer = learner_model.distilbert.transformer.layer[boundary_layer_index]
        if any(p.requires_grad for p in boundary_layer.parameters()):
            t.failed = True
            t.msg = f"The loop in the 'Unfreeze the Last N Transformer Layers' step is unfreezing too many layers. Layer {boundary_layer_index}, which should be frozen, was incorrectly made trainable."
            t.want = "A loop that unfreezes exactly the number of layers specified by `layers_to_train`."
            t.got = f"The loop unfroze extra layers. Layer {boundary_layer_index} should be frozen."
        cases.append(t)

        ############################

        # ############################ Test case 6: Check if the classification head is unfrozen ############################
        t = test_case()
        if not all(p.requires_grad for p in learner_model.classifier.parameters()):
            t.failed = True
            t.msg = "The final `classifier` layer is not unfrozen."
            t.want = "All parameters in `classifier` to have `requires_grad = True`."
            t.got = "At least one parameter in `classifier` has `requires_grad = False`."
        cases.append(t)
            
        t = test_case()
        if not all(p.requires_grad for p in learner_model.pre_classifier.parameters()):
            t.failed = True
            t.msg = "The `pre_classifier` layer is not unfrozen."
            t.want = "All parameters in `pre_classifier` to have `requires_grad = True`."
            t.got = "At least one parameter in `pre_classifier` has `requires_grad = False`."
        cases.append(t)

        ############################

        return cases

    cases = g()
    print_feedback(cases)



######################################## End of Excersie 5 Test ########################################



def exercise_6(learner_results):
    def g():
        cases = []

        required_keys = ['val_loss', 'val_f1']
        for key in required_keys:
            if key not in learner_results:
                t = test_case()
                t.failed = True
                t.msg = f"The 'learner_results' dictionary is missing a required key: '{key}'"
                t.want = f"A dictionary containing the key '{key}'"
                t.got = f"A dictionary with keys: {list(learner_results.keys())}"
                return [t]
        
        loss = learner_results['val_loss']
        f1 = learner_results['val_f1']

        # ############################ Test case 1: Check if validation loss is below 0.7 ############################
        t = test_case()
        if not loss < 0.7:
            t.failed = True
            t.msg = "The validation loss did not meet the target"
            t.want = "A loss value < 0.7"
            t.got = f"A loss of {loss:.4f}"
        cases.append(t)

        # ############################ Test case 2: Check if validation F1 score is above 0.8 ############################
        t = test_case()
        if not f1 > 0.8:
            t.failed = True
            t.msg = "The validation F1 score did not meet the target"
            t.want = "An F1 score > 0.8"
            t.got = f"An F1 score of {f1:.4f}"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)

