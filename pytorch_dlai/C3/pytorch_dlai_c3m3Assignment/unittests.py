from typing import List
import torch
import torch.nn as nn
from dlai_grader.grading import print_feedback, test_case

def exercise_1(learner_class):
    def g():
        cases: List[test_case] = []
        class_name = "Encoder"
        
        # Test 1: Check if it's a class
        t = test_case()
        if not isinstance(learner_class, type):
            t.failed = True
            t.msg = f"{class_name} must be a class"
            t.want = f"a Python class called {class_name}."
            t.got = type(learner_class)
            return [t]
        cases.append(t)
        
        # Test 2: Check if it inherits from nn.Module
        t = test_case()
        if not issubclass(learner_class, nn.Module):
            t.failed = True
            t.msg = f"{class_name} must inherit from nn.Module"
            t.want = nn.Module
            t.got = learner_class.__base__
            return [t]
        cases.append(t)
        
        # Test 3: Check token_emb uses correct parameters
        t = test_case()
        try:
            # Test with unusual values (d_model must be even for positional encoding)
            test_vocab = 123
            test_d_model = 64  # Even number
            model = learner_class(vocab_size=test_vocab, d_model=test_d_model)
            
            # Check token_emb dimensions
            if model.token_emb.num_embeddings != test_vocab:
                t.failed = True
                t.msg = "vocab_size is hardcoded or incorrectly set in token_emb"
                t.want = f"token_emb.num_embeddings = {test_vocab}"
                t.got = f"token_emb.num_embeddings = {model.token_emb.num_embeddings}"
                return cases + [t]
            
            if model.token_emb.embedding_dim != test_d_model:
                t.failed = True
                t.msg = "d_model is hardcoded or incorrectly set in token_emb"
                t.want = f"token_emb.embedding_dim = {test_d_model}"
                t.got = f"token_emb.embedding_dim = {model.token_emb.embedding_dim}"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to check token_emb parameters"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 4: Check transformer encoder layer parameters
        t = test_case()
        try:
            # Test with unusual values (d_model must be even and divisible by nhead)
            test_d_model = 96  # Divisible by 6
            test_nhead = 6
            test_dim_feedforward = 384
            test_num_layers = 5
            
            model = learner_class(
                vocab_size=100, 
                d_model=test_d_model,
                nhead=test_nhead,
                dim_feedforward=test_dim_feedforward,
                num_layers=test_num_layers
            )
            
            # Check number of layers
            if len(model.transformer_encoder.layers) != test_num_layers:
                t.failed = True
                t.msg = "num_layers is hardcoded or incorrectly set"
                t.want = f"{test_num_layers} encoder layers"
                t.got = f"{len(model.transformer_encoder.layers)} encoder layers"
                return cases + [t]
            
            # Check first encoder layer parameters
            first_layer = model.transformer_encoder.layers[0]
            
            # Check self-attention parameters
            if hasattr(first_layer.self_attn, 'embed_dim'):
                if first_layer.self_attn.embed_dim != test_d_model:
                    t.failed = True
                    t.msg = "d_model is hardcoded in transformer encoder self-attention"
                    t.want = f"self_attn.embed_dim = {test_d_model}"
                    t.got = f"self_attn.embed_dim = {first_layer.self_attn.embed_dim}"
                    return cases + [t]
            
            if hasattr(first_layer.self_attn, 'num_heads'):
                if first_layer.self_attn.num_heads != test_nhead:
                    t.failed = True
                    t.msg = "nhead is hardcoded in transformer encoder"
                    t.want = f"self_attn.num_heads = {test_nhead}"
                    t.got = f"self_attn.num_heads = {first_layer.self_attn.num_heads}"
                    return cases + [t]
            
            # Check feedforward network dimensions
            if hasattr(first_layer, 'linear1'):
                if first_layer.linear1.in_features != test_d_model:
                    t.failed = True
                    t.msg = "d_model is hardcoded in feedforward network input"
                    t.want = f"linear1.in_features = {test_d_model}"
                    t.got = f"linear1.in_features = {first_layer.linear1.in_features}"
                    return cases + [t]
                    
                if first_layer.linear1.out_features != test_dim_feedforward:
                    t.failed = True
                    t.msg = "dim_feedforward is hardcoded in feedforward network"
                    t.want = f"linear1.out_features = {test_dim_feedforward}"
                    t.got = f"linear1.out_features = {first_layer.linear1.out_features}"
                    return cases + [t]
            
            if hasattr(first_layer, 'linear2'):
                if first_layer.linear2.in_features != test_dim_feedforward:
                    t.failed = True
                    t.msg = "dim_feedforward is hardcoded in feedforward network"
                    t.want = f"linear2.in_features = {test_dim_feedforward}"
                    t.got = f"linear2.in_features = {first_layer.linear2.in_features}"
                    return cases + [t]
                    
                if first_layer.linear2.out_features != test_d_model:
                    t.failed = True
                    t.msg = "d_model is hardcoded in feedforward network output"
                    t.want = f"linear2.out_features = {test_d_model}"
                    t.got = f"linear2.out_features = {first_layer.linear2.out_features}"
                    return cases + [t]
                    
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to check transformer encoder parameters"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 5: Check with another set of unusual values to be thorough
        t = test_case()
        try:
            # Different unusual values (must be even for d_model and d_model divisible by nhead)
            test_vocab = 37
            test_d_model = 36  # Even and divisible by 4
            test_nhead = 4
            test_layers = 7
            
            model = learner_class(
                vocab_size=test_vocab,
                d_model=test_d_model,
                nhead=test_nhead,
                num_layers=test_layers
            )
            
            # Quick checks on all parameters
            checks_failed = []
            
            if model.token_emb.num_embeddings != test_vocab:
                checks_failed.append(f"vocab_size: expected {test_vocab}, got {model.token_emb.num_embeddings}")
            
            if model.token_emb.embedding_dim != test_d_model:
                checks_failed.append(f"d_model in embedding: expected {test_d_model}, got {model.token_emb.embedding_dim}")
            
            if len(model.transformer_encoder.layers) != test_layers:
                checks_failed.append(f"num_layers: expected {test_layers}, got {len(model.transformer_encoder.layers)}")
            
            first_layer = model.transformer_encoder.layers[0]
            if hasattr(first_layer.self_attn, 'num_heads'):
                if first_layer.self_attn.num_heads != test_nhead:
                    checks_failed.append(f"nhead: expected {test_nhead}, got {first_layer.self_attn.num_heads}")
            
            if checks_failed:
                t.failed = True
                t.msg = "Some parameters appear to be hardcoded"
                t.want = "All parameters to match the input arguments"
                t.got = "; ".join(checks_failed)
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed secondary parameter check"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 6: Check correct number of parameters for standard config
        t = test_case()
        try:
            model = learner_class(
                vocab_size=5000,
                d_model=256,
                nhead=8,
                num_layers=3,
                dim_feedforward=512
            )
            
            actual_params = sum(p.numel() for p in model.parameters())
            SOLUTION_PARAMETERS = 2861312  # TO BE FILLED
            
            if abs(actual_params - SOLUTION_PARAMETERS) > 0:
                t.failed = True
                t.msg = "Incorrect total number of parameters"
                t.want = f"{SOLUTION_PARAMETERS} parameters"
                t.got = f"{actual_params} parameters"
                return cases + [t]
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to check parameter count"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 7: Check padding_idx
        t = test_case()
        try:
            model = learner_class(vocab_size=100, d_model=32)
            
            if model.token_emb.padding_idx != 0:
                t.failed = True
                t.msg = "padding_idx is not set to 0"
                t.want = "padding_idx=0"
                t.got = f"padding_idx={model.token_emb.padding_idx}"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to check padding_idx"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 8: Check forward pass
        t = test_case()
        try:
            batch_size = 2
            seq_len = 10
            test_vocab = 50
            test_d_model = 32  # Even number
            
            model = learner_class(vocab_size=test_vocab, d_model=test_d_model)
            model.eval()
            
            x = torch.randint(0, test_vocab, (batch_size, seq_len))
            
            with torch.no_grad():
                memory, padding_mask = model(x)
            
            # Check output shapes
            if memory.shape != (batch_size, seq_len, test_d_model):
                t.failed = True
                t.msg = "Incorrect memory output shape"
                t.want = f"Shape {(batch_size, seq_len, test_d_model)}"
                t.got = f"Shape {memory.shape}"
                return cases + [t]
                
            if padding_mask.shape != (batch_size, seq_len):
                t.failed = True
                t.msg = "Incorrect padding_mask shape"
                t.want = f"Shape {(batch_size, seq_len)}"
                t.got = f"Shape {padding_mask.shape}"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed forward pass test"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        # Test 9: Check padding mask creation
        t = test_case()
        try:
            batch_size = 2
            seq_len = 10
            vocab_size = 50
            d_model = 32
            
            model = learner_class(vocab_size=vocab_size, d_model=d_model)
            model.eval()
            
            # Create input with padding tokens (0s) at specific positions
            x = torch.ones((batch_size, seq_len), dtype=torch.long)  # All 1s
            # Add padding tokens (0s) at specific positions
            x[0, 5:] = 0  # First sequence has padding from position 5
            x[1, 7:] = 0  # Second sequence has padding from position 7
            
            with torch.no_grad():
                memory, padding_mask = model(x)
            
            # Check if padding mask correctly identifies padding positions
            expected_mask = (x == 0)  # True where there are 0s
            
            if not torch.equal(padding_mask, expected_mask):
                t.failed = True
                t.msg = "Padding mask not correctly identifying padding tokens (pad_idx=0)"
                t.want = f"Padding mask with True at positions with value 0"
                t.got = f"Incorrect padding mask. Expected True at padded positions."
                
                # More detailed error message
                for i in range(batch_size):
                    for j in range(seq_len):
                        if padding_mask[i, j] != expected_mask[i, j]:
                            t.got += f"\nPosition [{i},{j}]: token={x[i,j].item()}, expected_mask={expected_mask[i,j].item()}, got_mask={padding_mask[i,j].item()}"
                            break
                return cases + [t]
            
            # Additional test: Check with different padding pattern
            x2 = torch.ones((batch_size, seq_len), dtype=torch.long) * 2  # All 2s
            x2[0, [2, 4, 6]] = 0  # Sparse padding in first sequence
            x2[1, :3] = 0  # Padding at beginning of second sequence
            
            with torch.no_grad():
                memory2, padding_mask2 = model(x2)
            
            expected_mask2 = (x2 == 0)
            
            if not torch.equal(padding_mask2, expected_mask2):
                t.failed = True
                t.msg = "Padding mask fails with different padding patterns"
                t.want = f"Padding mask correctly identifying all 0-valued positions"
                t.got = f"Incorrect padding mask for sparse/beginning padding patterns"
                return cases + [t]
            
            # Test that non-zero values are not masked
            x3 = torch.randint(1, vocab_size, (batch_size, seq_len))  # No zeros
            with torch.no_grad():
                memory3, padding_mask3 = model(x3)
            
            if padding_mask3.any():
                t.failed = True
                t.msg = "Padding mask incorrectly masking non-zero tokens"
                t.want = "No masking when there are no padding tokens (0s)"
                t.got = f"Found {padding_mask3.sum().item()} incorrectly masked positions"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test padding mask creation"
            t.got = str(e)
            return cases + [t]
        cases.append(t)

        # Test 10: Check if positional encoding is being added
        t = test_case()
        try:
            vocab_size = 50
            d_model = 32
            model = learner_class(vocab_size=vocab_size, d_model=d_model)
            model.eval()
            
            # Create two identical token sequences
            x1 = torch.tensor([[1, 2, 3, 4, 5]])  # Same tokens
            x2 = torch.tensor([[1, 2, 3, 4, 5]])  # Same tokens
            
            with torch.no_grad():
                # Get embeddings directly (before positional encoding would be added)
                just_embed = model.token_emb(x1)
                
                # Get the full forward pass output
                memory, _ = model(x1)
                
                # Extract the representation before transformer (after embedding + positional)
                # We need to check if positional encoding was added
                
                # Method 1: Check if output differs from just embeddings at each position
                # If positional encoding is added, each position should be modified differently
                if torch.allclose(memory[0, 0], memory[0, 1], atol=1e-5):
                    # If the same token at different positions has identical representations,
                    # positional encoding was not added
                    pass  # This might not catch it if transformer modifies it
                
                # Method 2: Better test - check intermediate values
                # Create same token repeated
                x_repeated = torch.ones((1, 5), dtype=torch.long) * 2  # Token 2 repeated 5 times
                
                # Get just embeddings
                embed_only = model.token_emb(x_repeated)
                
                # Forward pass through the model
                memory_full, _ = model(x_repeated)
                
                # If no positional encoding, all positions with same token should start identical
                # before transformer processing
                # We can't directly access intermediate, so let's be clever
                
                # Check if the model has pos_enc being used
                # by seeing if same tokens at different positions produce different outputs
                
                # The key insight: without positional encoding, identical tokens would produce
                # identical embeddings going into transformer, and with no position info,
                # self-attention would treat them identically
                
                # Create a sequence with all same tokens
                all_same = torch.ones((1, 6), dtype=torch.long) * 3
                memory_same, _ = model(all_same)
                
                # Check if all positions have identical output (would indicate no positional encoding)
                first_pos = memory_same[0, 0]
                all_identical = True
                for i in range(1, 6):
                    if not torch.allclose(first_pos, memory_same[0, i], atol=1e-6):
                        all_identical = False
                        break
                
                if all_identical:
                    t.failed = True
                    t.msg = "Positional encoding not being added to embeddings"
                    t.want = "Different representations for same token at different positions"
                    t.got = "Identical representations, suggesting positional encoding is missing"
                    return cases + [t]
                    
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test positional encoding addition"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 11: More direct test - monkey patch to check if pos_enc is called
        t = test_case()
        try:
            vocab_size = 50
            d_model = 32
            model = learner_class(vocab_size=vocab_size, d_model=d_model)
            model.eval()
            
            # Track if pos_enc is called
            pos_enc_called = False
            original_pos_enc = model.pos_enc.forward
            
            def tracked_pos_enc(x):
                nonlocal pos_enc_called
                pos_enc_called = True
                return original_pos_enc(x)
            
            # Monkey patch the forward method
            model.pos_enc.forward = tracked_pos_enc
            
            # Run forward pass
            x = torch.ones((1, 5), dtype=torch.long)
            with torch.no_grad():
                memory, _ = model(x)
            
            # Restore original method
            model.pos_enc.forward = original_pos_enc
            
            if not pos_enc_called:
                t.failed = True
                t.msg = "pos_enc not being used in forward pass"
                t.want = "self.token_emb(x) + self.pos_enc(x)"
                t.got = "pos_enc.forward() was never called"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test if pos_enc is called"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 12: Check that positional encoding actually affects the output
        t = test_case()
        try:
            vocab_size = 50
            d_model = 32
            model = learner_class(vocab_size=vocab_size, d_model=d_model)
            model.eval()
            
            # Temporarily set positional encoding to zero to see difference
            original_pe = model.pos_enc.pe.clone()
            model.pos_enc.pe.zero_()
            
            x = torch.ones((1, 5), dtype=torch.long) * 2
            with torch.no_grad():
                output_without_pe, _ = model(x)
            
            # Restore positional encoding
            model.pos_enc.pe = original_pe
            
            with torch.no_grad():
                output_with_pe, _ = model(x)
            
            # If outputs are the same, positional encoding is not being added
            if torch.allclose(output_without_pe, output_with_pe, atol=1e-6):
                t.failed = True
                t.msg = "Positional encoding not affecting the output"
                t.want = "Output to change when positional encoding is modified"
                t.got = "Same output with and without positional encoding"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test positional encoding effect"
            t.got = str(e)
            return cases + [t]
        cases.append(t)

        # Test 13: Check if dropout is being applied
        t = test_case()
        try:
            vocab_size = 50
            d_model = 32
            dropout_rate = 0.5  # High dropout for testing
            model = learner_class(vocab_size=vocab_size, d_model=d_model, dropout=dropout_rate)
            
            # Set model to TRAINING mode (dropout active)
            model.train()
            
            x = torch.ones((2, 10), dtype=torch.long) * 2
            
            # Run multiple forward passes - with dropout they should differ
            outputs = []
            for _ in range(5):
                with torch.no_grad():
                    memory, _ = model(x)
                    outputs.append(memory.clone())
            
            # Check if outputs differ (they should due to dropout)
            all_identical = True
            for i in range(1, len(outputs)):
                if not torch.allclose(outputs[0], outputs[i], atol=1e-6):
                    all_identical = False
                    break
            
            if all_identical:
                t.failed = True
                t.msg = "Dropout not being applied in forward pass"
                t.want = "Different outputs in training mode due to dropout"
                t.got = "Identical outputs, suggesting dropout is missing"
                return cases + [t]
            
            # Now test in eval mode - outputs should be identical
            model.eval()
            eval_outputs = []
            for _ in range(3):
                with torch.no_grad():
                    memory, _ = model(x)
                    eval_outputs.append(memory.clone())
            
            # In eval mode, all outputs should be identical
            for i in range(1, len(eval_outputs)):
                if not torch.allclose(eval_outputs[0], eval_outputs[i], atol=1e-6):
                    t.failed = True
                    t.msg = "Model outputs vary in eval mode"
                    t.want = "Identical outputs in eval mode (dropout disabled)"
                    t.got = "Different outputs, suggesting improper dropout implementation"
                    return cases + [t]
                    
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test dropout application"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 14: More direct dropout test - monkey patch
        t = test_case()
        try:
            vocab_size = 50
            d_model = 32
            model = learner_class(vocab_size=vocab_size, d_model=d_model, dropout=0.1)
            model.train()
            
            # Track if dropout is called
            dropout_called = False
            original_dropout = model.dropout.forward
            
            def tracked_dropout(x):
                nonlocal dropout_called
                dropout_called = True
                return original_dropout(x)
            
            # Monkey patch
            model.dropout.forward = tracked_dropout
            
            x = torch.ones((1, 5), dtype=torch.long)
            with torch.no_grad():
                memory, _ = model(x)
            
            # Restore
            model.dropout.forward = original_dropout
            
            if not dropout_called:
                t.failed = True
                t.msg = "self.dropout() not being called in forward pass"
                t.want = "x = self.dropout(x) after embeddings"
                t.got = "dropout.forward() was never called"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test if dropout is called"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 15: Check dropout is applied AFTER embedding+positional encoding
        t = test_case()
        try:
            vocab_size = 50
            d_model = 32
            model = learner_class(vocab_size=vocab_size, d_model=d_model, dropout=0.5)
            
            # Track the order of operations
            operations_order = []
            
            # Patch embedding
            original_embed = model.token_emb.forward
            def tracked_embed(x):
                operations_order.append('embedding')
                return original_embed(x)
            
            # Patch positional encoding
            original_pos = model.pos_enc.forward
            def tracked_pos(x):
                operations_order.append('positional')
                return original_pos(x)
            
            # Patch dropout
            original_drop = model.dropout.forward
            def tracked_drop(x):
                operations_order.append('dropout')
                return original_drop(x)
            
            model.token_emb.forward = tracked_embed
            model.pos_enc.forward = tracked_pos
            model.dropout.forward = tracked_drop
            
            model.train()
            x = torch.ones((1, 5), dtype=torch.long)
            
            with torch.no_grad():
                memory, _ = model(x)
            
            # Restore
            model.token_emb.forward = original_embed
            model.pos_enc.forward = original_pos
            model.dropout.forward = original_drop
            
            # Check order
            if 'dropout' not in operations_order:
                t.failed = True
                t.msg = "Dropout not being applied"
                t.want = "Dropout to be applied after embeddings"
                t.got = f"Operations called: {operations_order}"
                return cases + [t]
            
            # Check dropout comes after embedding and positional
            if operations_order.index('dropout') < operations_order.index('embedding'):
                t.failed = True
                t.msg = "Dropout applied before embedding"
                t.want = "Dropout after embedding"
                t.got = f"Order: {operations_order}"
                return cases + [t]
                
            if 'positional' in operations_order:
                if operations_order.index('dropout') < operations_order.index('positional'):
                    t.failed = True
                    t.msg = "Dropout applied before positional encoding"
                    t.want = "Dropout after positional encoding"
                    t.got = f"Order: {operations_order}"
                    return cases + [t]
                    
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test dropout order"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        return cases
    
    cases = g()
    print_feedback(cases)


from typing import List
import torch
import torch.nn as nn
from dlai_grader.grading import print_feedback, test_case

def exercise_2(learner_class):
    def g():
        cases: List[test_case] = []
        class_name = "Decoder"
        
        # Test 1: Check if it's a class
        t = test_case()
        if not isinstance(learner_class, type):
            t.failed = True
            t.msg = f"{class_name} must be a class"
            t.want = f"a Python class called {class_name}."
            t.got = type(learner_class)
            return [t]
        cases.append(t)
        
        # Test 2: Check if it inherits from nn.Module
        t = test_case()
        if not issubclass(learner_class, nn.Module):
            t.failed = True
            t.msg = f"{class_name} must inherit from nn.Module"
            t.want = nn.Module
            t.got = learner_class.__base__
            return [t]
        cases.append(t)
        
        # Test 3: Check token_emb uses correct parameters
        t = test_case()
        try:
            # Test with unusual values (d_model must be even for positional encoding)
            test_vocab = 123
            test_d_model = 64  # Even number
            model = learner_class(vocab_size=test_vocab, d_model=test_d_model)
            
            # Check token_emb dimensions
            if model.token_emb.num_embeddings != test_vocab:
                t.failed = True
                t.msg = "vocab_size is hardcoded or incorrectly set in token_emb"
                t.want = f"token_emb.num_embeddings = {test_vocab}"
                t.got = f"token_emb.num_embeddings = {model.token_emb.num_embeddings}"
                return cases + [t]
            
            if model.token_emb.embedding_dim != test_d_model:
                t.failed = True
                t.msg = "d_model is hardcoded or incorrectly set in token_emb"
                t.want = f"token_emb.embedding_dim = {test_d_model}"
                t.got = f"token_emb.embedding_dim = {model.token_emb.embedding_dim}"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to check token_emb parameters"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 4: Check transformer decoder layer parameters
        t = test_case()
        try:
            # Test with unusual values (d_model must be even and divisible by nhead)
            test_d_model = 96  # Divisible by 6
            test_nhead = 6
            test_dim_feedforward = 384
            test_num_layers = 5
            
            model = learner_class(
                vocab_size=100, 
                d_model=test_d_model,
                nhead=test_nhead,
                dim_feedforward=test_dim_feedforward,
                num_layers=test_num_layers
            )
            
            # Check number of layers
            if len(model.transformer_decoder.layers) != test_num_layers:
                t.failed = True
                t.msg = "num_layers is hardcoded or incorrectly set"
                t.want = f"{test_num_layers} decoder layers"
                t.got = f"{len(model.transformer_decoder.layers)} decoder layers"
                return cases + [t]
            
            # Check first decoder layer parameters
            first_layer = model.transformer_decoder.layers[0]
            
            # Check self-attention parameters
            if hasattr(first_layer.self_attn, 'embed_dim'):
                if first_layer.self_attn.embed_dim != test_d_model:
                    t.failed = True
                    t.msg = "d_model is hardcoded in transformer decoder self-attention"
                    t.want = f"self_attn.embed_dim = {test_d_model}"
                    t.got = f"self_attn.embed_dim = {first_layer.self_attn.embed_dim}"
                    return cases + [t]
            
            if hasattr(first_layer.self_attn, 'num_heads'):
                if first_layer.self_attn.num_heads != test_nhead:
                    t.failed = True
                    t.msg = "nhead is hardcoded in transformer decoder"
                    t.want = f"self_attn.num_heads = {test_nhead}"
                    t.got = f"self_attn.num_heads = {first_layer.self_attn.num_heads}"
                    return cases + [t]
            
            # Check cross-attention parameters
            if hasattr(first_layer, 'multihead_attn'):
                if first_layer.multihead_attn.embed_dim != test_d_model:
                    t.failed = True
                    t.msg = "d_model is hardcoded in cross-attention"
                    t.want = f"multihead_attn.embed_dim = {test_d_model}"
                    t.got = f"multihead_attn.embed_dim = {first_layer.multihead_attn.embed_dim}"
                    return cases + [t]
            
            # Check feedforward network dimensions
            if hasattr(first_layer, 'linear1'):
                if first_layer.linear1.in_features != test_d_model:
                    t.failed = True
                    t.msg = "d_model is hardcoded in feedforward network input"
                    t.want = f"linear1.in_features = {test_d_model}"
                    t.got = f"linear1.in_features = {first_layer.linear1.in_features}"
                    return cases + [t]
                    
                if first_layer.linear1.out_features != test_dim_feedforward:
                    t.failed = True
                    t.msg = "dim_feedforward is hardcoded in feedforward network"
                    t.want = f"linear1.out_features = {test_dim_feedforward}"
                    t.got = f"linear1.out_features = {first_layer.linear1.out_features}"
                    return cases + [t]
                    
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to check transformer decoder parameters"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 5: Check output projection layer
        t = test_case()
        try:
            test_vocab = 77
            test_d_model = 48
            model = learner_class(vocab_size=test_vocab, d_model=test_d_model)
            
            if not hasattr(model, 'output_projection'):
                t.failed = True
                t.msg = "Missing output_projection layer"
                t.want = "An nn.Linear layer named 'output_projection'"
                t.got = "No 'output_projection' attribute found"
                return cases + [t]
            
            if not isinstance(model.output_projection, nn.Linear):
                t.failed = True
                t.msg = "output_projection is not nn.Linear"
                t.want = "nn.Linear"
                t.got = type(model.output_projection)
                return cases + [t]
            
            if model.output_projection.in_features != test_d_model:
                t.failed = True
                t.msg = "output_projection input dimension is incorrect"
                t.want = f"in_features = {test_d_model}"
                t.got = f"in_features = {model.output_projection.in_features}"
                return cases + [t]
            
            if model.output_projection.out_features != test_vocab:
                t.failed = True
                t.msg = "output_projection output dimension is incorrect"
                t.want = f"out_features = {test_vocab}"
                t.got = f"out_features = {model.output_projection.out_features}"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to check output_projection layer"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 6: Check with another set of unusual values
        t = test_case()
        try:
            # Different unusual values
            test_vocab = 37
            test_d_model = 36  # Even and divisible by 4
            test_nhead = 4
            test_layers = 7
            
            model = learner_class(
                vocab_size=test_vocab,
                d_model=test_d_model,
                nhead=test_nhead,
                num_layers=test_layers
            )
            
            checks_failed = []
            
            if model.token_emb.num_embeddings != test_vocab:
                checks_failed.append(f"vocab_size: expected {test_vocab}, got {model.token_emb.num_embeddings}")
            
            if model.token_emb.embedding_dim != test_d_model:
                checks_failed.append(f"d_model in embedding: expected {test_d_model}, got {model.token_emb.embedding_dim}")
            
            if len(model.transformer_decoder.layers) != test_layers:
                checks_failed.append(f"num_layers: expected {test_layers}, got {len(model.transformer_decoder.layers)}")
            
            if model.output_projection.out_features != test_vocab:
                checks_failed.append(f"output vocab_size: expected {test_vocab}, got {model.output_projection.out_features}")
            
            if checks_failed:
                t.failed = True
                t.msg = "Some parameters appear to be hardcoded"
                t.want = "All parameters to match the input arguments"
                t.got = "; ".join(checks_failed)
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed secondary parameter check"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 7: Check correct number of parameters for standard config
        t = test_case()
        try:
            model = learner_class(
                vocab_size=5000,
                d_model=256,
                nhead=8,
                num_layers=3,
                dim_feedforward=512
            )
            
            actual_params = sum(p.numel() for p in model.parameters())
            SOLUTION_PARAMETERS = 4937352  
            
            if abs(actual_params - SOLUTION_PARAMETERS) > 0:
                t.failed = True
                t.msg = "Incorrect total number of parameters"
                t.want = f"{SOLUTION_PARAMETERS} parameters"
                t.got = f"{actual_params} parameters"
                return cases + [t]
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to check parameter count"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 8: Check padding_idx
        t = test_case()
        try:
            model = learner_class(vocab_size=100, d_model=32)
            
            if model.token_emb.padding_idx != 0:
                t.failed = True
                t.msg = "padding_idx is not set to 0"
                t.want = "padding_idx=0"
                t.got = f"padding_idx={model.token_emb.padding_idx}"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to check padding_idx"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 9: Check forward pass output shape
        t = test_case()
        try:
            batch_size = 2
            tgt_seq_len = 8
            src_seq_len = 10
            test_vocab = 50
            test_d_model = 32
            
            model = learner_class(vocab_size=test_vocab, d_model=test_d_model)
            model.eval()
            
            # Create inputs
            x = torch.randint(0, test_vocab, (batch_size, tgt_seq_len))
            memory = torch.randn(batch_size, src_seq_len, test_d_model)
            memory_mask = torch.zeros(batch_size, src_seq_len).bool()
            
            with torch.no_grad():
                output = model(x, memory, memory_mask)
            
            # Check output shape
            expected_shape = (batch_size, tgt_seq_len, test_vocab)
            if output.shape != expected_shape:
                t.failed = True
                t.msg = "Incorrect output shape"
                t.want = f"Shape {expected_shape}"
                t.got = f"Shape {output.shape}"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed forward pass test"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 10: Check padding mask creation with pad_idx=0
        t = test_case()
        try:
            model = learner_class(vocab_size=50, d_model=32)
            model.eval()
            
            # Create input with known padding positions
            batch_size = 2
            seq_len = 10
            x = torch.ones((batch_size, seq_len), dtype=torch.long) * 2
            x[0, 5:] = 0  # Padding in first sequence
            x[1, 7:] = 0  # Padding in second sequence
            
            memory = torch.randn(batch_size, 12, 32)
            
            # Track if create_padding_mask is called correctly
            original_transformer_forward = model.transformer_decoder.forward
            padding_mask_correct = [False]
            
            def check_padding_mask(tgt, memory, **kwargs):
                if 'tgt_key_padding_mask' in kwargs:
                    mask = kwargs['tgt_key_padding_mask']
                    expected = (x == 0)
                    if torch.equal(mask, expected):
                        padding_mask_correct[0] = True
                return original_transformer_forward(tgt, memory, **kwargs)
            
            model.transformer_decoder.forward = check_padding_mask
            
            with torch.no_grad():
                output = model(x, memory)
            
            model.transformer_decoder.forward = original_transformer_forward
            
            if not padding_mask_correct[0]:
                t.failed = True
                t.msg = "Padding mask not correctly created with pad_idx=0"
                t.want = "Padding mask marking positions with value 0"
                t.got = "Incorrect padding mask"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test padding mask creation"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 11: Check if positional encoding is being added
        t = test_case()
        try:
            model = learner_class(vocab_size=50, d_model=32)
            model.eval()
            
            # Track if pos_enc is called
            pos_enc_called = False
            original_pos_enc = model.pos_enc.forward
            
            def tracked_pos_enc(x):
                nonlocal pos_enc_called
                pos_enc_called = True
                return original_pos_enc(x)
            
            model.pos_enc.forward = tracked_pos_enc
            
            x = torch.ones((1, 5), dtype=torch.long)
            memory = torch.randn(1, 8, 32)
            
            with torch.no_grad():
                output = model(x, memory)
            
            model.pos_enc.forward = original_pos_enc
            
            if not pos_enc_called:
                t.failed = True
                t.msg = "pos_enc not being used in forward pass"
                t.want = "self.token_emb(x) + self.pos_enc(x)"
                t.got = "pos_enc.forward() was never called"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test if pos_enc is called"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 12: Check if dropout is being applied
        t = test_case()
        try:
            vocab_size = 50
            d_model = 32
            model = learner_class(vocab_size=vocab_size, d_model=d_model, dropout=0.5)
            
            # Track if dropout is called
            dropout_called = False
            original_dropout = model.dropout.forward
            
            def tracked_dropout(x):
                nonlocal dropout_called
                dropout_called = True
                return original_dropout(x)
            
            model.dropout.forward = tracked_dropout
            model.train()  # Ensure dropout is active
            
            x = torch.ones((1, 5), dtype=torch.long)
            memory = torch.randn(1, 8, 32)
            
            with torch.no_grad():
                output = model(x, memory)
            
            model.dropout.forward = original_dropout
            
            if not dropout_called:
                t.failed = True
                t.msg = "self.dropout() not being called in forward pass"
                t.want = "x = self.dropout(x) after embeddings"
                t.got = "dropout.forward() was never called"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test if dropout is called"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 13: Check subsequent mask creation
        t = test_case()
        try:
            model = learner_class(vocab_size=50, d_model=32)
            model.eval()
            
            batch_size = 2
            seq_len = 6
            x = torch.ones((batch_size, seq_len), dtype=torch.long)
            memory = torch.randn(batch_size, 8, 32)
            
            # Track if subsequent mask is created correctly
            subsequent_mask_correct = [False]
            original_transformer_forward = model.transformer_decoder.forward
            
            def check_subsequent_mask(tgt, memory, **kwargs):
                if 'tgt_mask' in kwargs:
                    mask = kwargs['tgt_mask']
                    # Check if it's upper triangular (subsequent mask)
                    if mask is not None and len(mask.shape) == 2:
                        expected_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(mask.device)
                        if torch.equal(mask, expected_mask):
                            subsequent_mask_correct[0] = True
                return original_transformer_forward(tgt, memory, **kwargs)
            
            model.transformer_decoder.forward = check_subsequent_mask
            
            with torch.no_grad():
                output = model(x, memory)
            
            model.transformer_decoder.forward = original_transformer_forward
            
            if not subsequent_mask_correct[0]:
                t.failed = True
                t.msg = "Subsequent mask not correctly created"
                t.want = "Upper triangular mask preventing look-ahead"
                t.got = "Incorrect or missing subsequent mask"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test subsequent mask creation"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 14: Check that output projection is applied
        t = test_case()
        try:
            model = learner_class(vocab_size=50, d_model=32)
            model.eval()
            
            # Track if output_projection is called
            projection_called = False
            original_proj = model.output_projection.forward
            
            def tracked_projection(x):
                nonlocal projection_called
                projection_called = True
                return original_proj(x)
            
            model.output_projection.forward = tracked_projection
            
            x = torch.ones((1, 5), dtype=torch.long)
            memory = torch.randn(1, 8, 32)
            
            with torch.no_grad():
                output = model(x, memory)
            
            model.output_projection.forward = original_proj
            
            if not projection_called:
                t.failed = True
                t.msg = "output_projection not being called"
                t.want = "output = self.output_projection(decoded)"
                t.got = "output_projection.forward() was never called"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test if output_projection is called"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 15: Check order of operations in forward pass
        t = test_case()
        try:
            model = learner_class(vocab_size=50, d_model=32)
            
            operations_order = []
            
            # Patch all operations
            original_embed = model.token_emb.forward
            def tracked_embed(x):
                operations_order.append('embedding')
                return original_embed(x)
            
            original_pos = model.pos_enc.forward
            def tracked_pos(x):
                operations_order.append('positional')
                return original_pos(x)
            
            original_drop = model.dropout.forward
            def tracked_drop(x):
                operations_order.append('dropout')
                return original_drop(x)
            
            original_transformer = model.transformer_decoder.forward
            def tracked_transformer(tgt, memory, **kwargs):
                operations_order.append('transformer')
                return original_transformer(tgt, memory, **kwargs)
            
            original_proj = model.output_projection.forward
            def tracked_proj(x):
                operations_order.append('projection')
                return original_proj(x)
            
            model.token_emb.forward = tracked_embed
            model.pos_enc.forward = tracked_pos
            model.dropout.forward = tracked_drop
            model.transformer_decoder.forward = tracked_transformer
            model.output_projection.forward = tracked_proj
            
            model.train()
            x = torch.ones((1, 5), dtype=torch.long)
            memory = torch.randn(1, 8, 32)
            
            with torch.no_grad():
                output = model(x, memory)
            
            # Restore
            model.token_emb.forward = original_embed
            model.pos_enc.forward = original_pos
            model.dropout.forward = original_drop
            model.transformer_decoder.forward = original_transformer
            model.output_projection.forward = original_proj
            
            # Check order
            expected_order = ['embedding', 'positional', 'dropout', 'transformer', 'projection']
            
            # Check all operations are present
            for op in expected_order:
                if op not in operations_order:
                    t.failed = True
                    t.msg = f"Operation '{op}' not found in forward pass"
                    t.want = f"All operations: {expected_order}"
                    t.got = f"Found operations: {operations_order}"
                    return cases + [t]
            
            # Check order is correct
            actual_order = [op for op in operations_order if op in expected_order]
            if actual_order != expected_order:
                t.failed = True
                t.msg = "Operations in wrong order"
                t.want = f"Order: {expected_order}"
                t.got = f"Order: {actual_order}"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test operation order"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 16: Check that memory_padding_mask is passed correctly
        t = test_case()
        try:
            model = learner_class(vocab_size=50, d_model=32)
            model.eval()
            
            batch_size = 2
            tgt_len = 5
            src_len = 8
            x = torch.ones((batch_size, tgt_len), dtype=torch.long)
            memory = torch.randn(batch_size, src_len, 32)
            memory_padding_mask = torch.zeros(batch_size, src_len).bool()
            memory_padding_mask[0, 6:] = True  # Mark some positions as padding
            
            # Check if memory_padding_mask is passed to transformer
            memory_mask_passed = [False]
            original_transformer = model.transformer_decoder.forward
            
            def check_memory_mask(tgt, memory, **kwargs):
                if 'memory_key_padding_mask' in kwargs:
                    if kwargs['memory_key_padding_mask'] is not None:
                        if torch.equal(kwargs['memory_key_padding_mask'], memory_padding_mask):
                            memory_mask_passed[0] = True
                return original_transformer(tgt, memory = memory, **kwargs)
            
            model.transformer_decoder.forward = check_memory_mask
            
            with torch.no_grad():
                output = model(x, memory=memory, memory_padding_mask=memory_padding_mask)
            
            model.transformer_decoder.forward = original_transformer
            
            if not memory_mask_passed[0]:
                t.failed = True
                t.msg = "memory_padding_mask not passed correctly to transformer_decoder"
                t.want = "memory_key_padding_mask=memory_padding_mask"
                t.got = "memory_padding_mask not passed or incorrect"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test memory_padding_mask passing"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 17: Verify dropout effect in training mode
        t = test_case()
        try:
            model = learner_class(vocab_size=50, d_model=32, dropout=0.5)
            model.train()  # Training mode
            
            x = torch.ones((2, 5), dtype=torch.long) * 2
            memory = torch.randn(2, 8, 32)
            
            # Run multiple forward passes
            outputs = []
            for _ in range(5):
                with torch.no_grad():
                    output = model(x, memory)
                    outputs.append(output.clone())
            
            # Check if outputs differ (due to dropout)
            all_identical = True
            for i in range(1, len(outputs)):
                if not torch.allclose(outputs[0], outputs[i], atol=1e-6):
                    all_identical = False
                    break
            
            if all_identical:
                t.failed = True
                t.msg = "Dropout not having effect in training mode"
                t.want = "Different outputs due to dropout randomness"
                t.got = "Identical outputs, suggesting dropout not applied"
                return cases + [t]
            
            # Now check eval mode - outputs should be identical
            model.eval()
            eval_outputs = []
            for _ in range(3):
                with torch.no_grad():
                    output = model(x, memory)
                    eval_outputs.append(output.clone())
            
            for i in range(1, len(eval_outputs)):
                if not torch.allclose(eval_outputs[0], eval_outputs[i], atol=1e-6):
                    t.failed = True
                    t.msg = "Outputs vary in eval mode"
                    t.want = "Identical outputs in eval mode"
                    t.got = "Different outputs in eval mode"
                    return cases + [t]
                    
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test dropout effect"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        # Test 11: Check PositionalEncoding parameters
        t = test_case()
        try:
            test_max_len = 75
            test_d_model = 64
            model = learner_class(vocab_size=50, d_model=test_d_model, max_len=test_max_len)
            
            # Check if pos_enc has correct parameters
            if hasattr(model.pos_enc, 'max_len'):
                if model.pos_enc.max_len != test_max_len:
                    t.failed = True
                    t.msg = "max_len is hardcoded in PositionalEncoding"
                    t.want = f"pos_enc.max_len = {test_max_len}"
                    t.got = f"pos_enc.max_len = {model.pos_enc.max_len}"
                    return cases + [t]
            
            if hasattr(model.pos_enc, 'd_model'):
                if model.pos_enc.d_model != test_d_model:
                    t.failed = True
                    t.msg = "d_model is hardcoded in PositionalEncoding"
                    t.want = f"pos_enc.d_model = {test_d_model}"
                    t.got = f"pos_enc.d_model = {model.pos_enc.d_model}"
                    return cases + [t]
            
            # Test with different values
            test_max_len2 = 200
            test_d_model2 = 128
            model2 = learner_class(vocab_size=50, d_model=test_d_model2, max_len=test_max_len2)
            
            if hasattr(model2.pos_enc, 'max_len'):
                if model2.pos_enc.max_len != test_max_len2:
                    t.failed = True
                    t.msg = "max_len not properly passed to PositionalEncoding"
                    t.want = f"PositionalEncoding(max_len={test_max_len2}, d_model={test_d_model2})"
                    t.got = f"max_len={model2.pos_enc.max_len}"
                    return cases + [t]
            
            if hasattr(model2.pos_enc, 'd_model'):
                if model2.pos_enc.d_model != test_d_model2:
                    t.failed = True
                    t.msg = "d_model not properly passed to PositionalEncoding"
                    t.want = f"PositionalEncoding(max_len={test_max_len2}, d_model={test_d_model2})"
                    t.got = f"d_model={model2.pos_enc.d_model}"
                    return cases + [t]
                    
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to check PositionalEncoding parameters"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 12: Check if positional encoding is being added
        t = test_case()
        try:
            model = learner_class(vocab_size=50, d_model=32)
            model.eval()
            
            # Track if pos_enc is called
            pos_enc_called = False
            original_pos_enc = model.pos_enc.forward
            
            def tracked_pos_enc(x):
                nonlocal pos_enc_called
                pos_enc_called = True
                return original_pos_enc(x)
            
            model.pos_enc.forward = tracked_pos_enc
            
            x = torch.ones((1, 5), dtype=torch.long)
            memory = torch.randn(1, 8, 32)
            
            with torch.no_grad():
                output = model(x, memory)
            
            model.pos_enc.forward = original_pos_enc
            
            if not pos_enc_called:
                t.failed = True
                t.msg = "pos_enc not being used in forward pass"
                t.want = "self.token_emb(x) + self.pos_enc(x)"
                t.got = "pos_enc.forward() was never called"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test if pos_enc is called"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        return cases
    
    cases = g()
    print_feedback(cases)


from typing import List
import torch
import torch.nn as nn
from dlai_grader.grading import print_feedback, test_case

def exercise_3(learner_class, encoder_class, decoder_class):
    def g():
        cases: List[test_case] = []
        class_name = "EncoderDecoder"
        
        # Test 1: Check if it's a class
        t = test_case()
        if not isinstance(learner_class, type):
            t.failed = True
            t.msg = f"{class_name} must be a class"
            t.want = f"a Python class called {class_name}."
            t.got = type(learner_class)
            return [t]
        cases.append(t)
        
        # Test 2: Check if it inherits from nn.Module
        t = test_case()
        if not issubclass(learner_class, nn.Module):
            t.failed = True
            t.msg = f"{class_name} must inherit from nn.Module"
            t.want = nn.Module
            t.got = learner_class.__base__
            return [t]
        cases.append(t)
        
        # Test 3: Check encoder initialization with correct parameters
        t = test_case()
        try:
            # Use unusual values to catch hardcoding
            test_src_vocab = 123
            test_tgt_vocab = 456
            test_d_model = 64
            test_nhead = 4
            test_enc_layers = 2
            test_dec_layers = 3
            test_dim_ff = 128
            test_max_len = 75
            test_dropout = 0.2
            
            model = learner_class(
                src_vocab_size=test_src_vocab,
                tgt_vocab_size=test_tgt_vocab,
                d_model=test_d_model,
                nhead=test_nhead,
                num_enc_layers=test_enc_layers,
                num_dec_layers=test_dec_layers,
                dim_feedforward=test_dim_ff,
                max_len=test_max_len,
                dropout=test_dropout
            )
            
            # Check if encoder exists
            if not hasattr(model, 'encoder'):
                t.failed = True
                t.msg = "Missing encoder attribute"
                t.want = "self.encoder = Encoder(...)"
                t.got = "No encoder found"
                return cases + [t]
            
            # Check encoder parameters
            if hasattr(model.encoder, 'token_emb'):
                if model.encoder.token_emb.num_embeddings != test_src_vocab:
                    t.failed = True
                    t.msg = "src_vocab_size not properly passed to encoder"
                    t.want = f"encoder vocab_size = {test_src_vocab}"
                    t.got = f"encoder vocab_size = {model.encoder.token_emb.num_embeddings}"
                    return cases + [t]
                
                if model.encoder.token_emb.embedding_dim != test_d_model:
                    t.failed = True
                    t.msg = "d_model not properly passed to encoder"
                    t.want = f"encoder d_model = {test_d_model}"
                    t.got = f"encoder d_model = {model.encoder.token_emb.embedding_dim}"
                    return cases + [t]
            
            # Check encoder layers
            if hasattr(model.encoder, 'transformer_encoder'):
                if len(model.encoder.transformer_encoder.layers) != test_enc_layers:
                    t.failed = True
                    t.msg = "num_enc_layers not properly passed to encoder"
                    t.want = f"encoder num_layers = {test_enc_layers}"
                    t.got = f"encoder num_layers = {len(model.encoder.transformer_encoder.layers)}"
                    return cases + [t]
            
            # Check positional encoding in encoder
            if hasattr(model.encoder, 'pos_enc'):
                if hasattr(model.encoder.pos_enc, 'max_len'):
                    if model.encoder.pos_enc.max_len != test_max_len:
                        t.failed = True
                        t.msg = "max_len not properly passed to encoder's PositionalEncoding"
                        t.want = f"encoder.pos_enc.max_len = {test_max_len}"
                        t.got = f"encoder.pos_enc.max_len = {model.encoder.pos_enc.max_len}"
                        return cases + [t]
                
                if hasattr(model.encoder.pos_enc, 'd_model'):
                    if model.encoder.pos_enc.d_model != test_d_model:
                        t.failed = True
                        t.msg = "d_model not properly passed to encoder's PositionalEncoding"
                        t.want = f"encoder.pos_enc.d_model = {test_d_model}"
                        t.got = f"encoder.pos_enc.d_model = {model.encoder.pos_enc.d_model}"
                        return cases + [t]
                        
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to check encoder initialization"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 4: Check decoder initialization with correct parameters
        t = test_case()
        try:
            model = learner_class(
                src_vocab_size=test_src_vocab,
                tgt_vocab_size=test_tgt_vocab,
                d_model=test_d_model,
                nhead=test_nhead,
                num_enc_layers=test_enc_layers,
                num_dec_layers=test_dec_layers,
                dim_feedforward=test_dim_ff,
                max_len=test_max_len,
                dropout=test_dropout
            )
            
            # Check if decoder exists
            if not hasattr(model, 'decoder'):
                t.failed = True
                t.msg = "Missing decoder attribute"
                t.want = "self.decoder = Decoder(...)"
                t.got = "No decoder found"
                return cases + [t]
            
            # Check decoder parameters
            if hasattr(model.decoder, 'token_emb'):
                if model.decoder.token_emb.num_embeddings != test_tgt_vocab:
                    t.failed = True
                    t.msg = "tgt_vocab_size not properly passed to decoder"
                    t.want = f"decoder vocab_size = {test_tgt_vocab}"
                    t.got = f"decoder vocab_size = {model.decoder.token_emb.num_embeddings}"
                    return cases + [t]
                
                if model.decoder.token_emb.embedding_dim != test_d_model:
                    t.failed = True
                    t.msg = "d_model not properly passed to decoder"
                    t.want = f"decoder d_model = {test_d_model}"
                    t.got = f"decoder d_model = {model.decoder.token_emb.embedding_dim}"
                    return cases + [t]
            
            # Check decoder layers
            if hasattr(model.decoder, 'transformer_decoder'):
                if len(model.decoder.transformer_decoder.layers) != test_dec_layers:
                    t.failed = True
                    t.msg = "num_dec_layers not properly passed to decoder"
                    t.want = f"decoder num_layers = {test_dec_layers}"
                    t.got = f"decoder num_layers = {len(model.decoder.transformer_decoder.layers)}"
                    return cases + [t]
            
            # Check output projection
            if hasattr(model.decoder, 'output_projection'):
                if model.decoder.output_projection.out_features != test_tgt_vocab:
                    t.failed = True
                    t.msg = "tgt_vocab_size not properly used in decoder's output_projection"
                    t.want = f"output_projection.out_features = {test_tgt_vocab}"
                    t.got = f"output_projection.out_features = {model.decoder.output_projection.out_features}"
                    return cases + [t]
            
            # Check positional encoding in decoder
            if hasattr(model.decoder, 'pos_enc'):
                if hasattr(model.decoder.pos_enc, 'max_len'):
                    if model.decoder.pos_enc.max_len != test_max_len:
                        t.failed = True
                        t.msg = "max_len not properly passed to decoder's PositionalEncoding"
                        t.want = f"decoder.pos_enc.max_len = {test_max_len}"
                        t.got = f"decoder.pos_enc.max_len = {model.decoder.pos_enc.max_len}"
                        return cases + [t]
                
                if hasattr(model.decoder.pos_enc, 'd_model'):
                    if model.decoder.pos_enc.d_model != test_d_model:
                        t.failed = True
                        t.msg = "d_model not properly passed to decoder's PositionalEncoding"
                        t.want = f"decoder.pos_enc.d_model = {test_d_model}"
                        t.got = f"decoder.pos_enc.d_model = {model.decoder.pos_enc.d_model}"
                        return cases + [t]
                        
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to check decoder initialization"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 5: Check with different parameters to catch more hardcoding
        t = test_case()
        try:
            # Different unusual values
            test_src_vocab2 = 37
            test_tgt_vocab2 = 89
            test_d_model2 = 96
            test_nhead2 = 6
            test_enc_layers2 = 5
            test_dec_layers2 = 7
            
            model2 = learner_class(
                src_vocab_size=test_src_vocab2,
                tgt_vocab_size=test_tgt_vocab2,
                d_model=test_d_model2,
                nhead=test_nhead2,
                num_enc_layers=test_enc_layers2,
                num_dec_layers=test_dec_layers2
            )
            
            checks_failed = []
            
            # Check encoder
            if model2.encoder.token_emb.num_embeddings != test_src_vocab2:
                checks_failed.append(f"encoder vocab: expected {test_src_vocab2}, got {model2.encoder.token_emb.num_embeddings}")
            
            if len(model2.encoder.transformer_encoder.layers) != test_enc_layers2:
                checks_failed.append(f"encoder layers: expected {test_enc_layers2}, got {len(model2.encoder.transformer_encoder.layers)}")
            
            # Check decoder
            if model2.decoder.token_emb.num_embeddings != test_tgt_vocab2:
                checks_failed.append(f"decoder vocab: expected {test_tgt_vocab2}, got {model2.decoder.token_emb.num_embeddings}")
            
            if len(model2.decoder.transformer_decoder.layers) != test_dec_layers2:
                checks_failed.append(f"decoder layers: expected {test_dec_layers2}, got {len(model2.decoder.transformer_decoder.layers)}")
            
            if checks_failed:
                t.failed = True
                t.msg = "Some parameters appear to be hardcoded"
                t.want = "All parameters to match input arguments"
                t.got = "; ".join(checks_failed)
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed secondary parameter check"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 6: Check forward pass
        t = test_case()
        try:
            batch_size = 2
            src_len = 10
            tgt_len = 8
            test_src_vocab = 50
            test_tgt_vocab = 60
            test_d_model = 32
            
            model = learner_class(
                src_vocab_size=test_src_vocab,
                tgt_vocab_size=test_tgt_vocab,
                d_model=test_d_model
            )
            model.eval()
            
            # Create inputs
            x = torch.randint(0, test_src_vocab, (batch_size, src_len))
            tgt = torch.randint(0, test_tgt_vocab, (batch_size, tgt_len))
            
            with torch.no_grad():
                output = model(x, tgt)
            
            # Check output shape
            expected_shape = (batch_size, tgt_len, test_tgt_vocab)
            if output.shape != expected_shape:
                t.failed = True
                t.msg = "Incorrect output shape"
                t.want = f"Shape {expected_shape}"
                t.got = f"Shape {output.shape}"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed forward pass test"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 7: Check that encoder is called in forward pass
        t = test_case()
        try:
            model = learner_class(src_vocab_size=50, tgt_vocab_size=60, d_model=32)
            model.eval()
            
            encoder_called = False
            original_encoder = model.encoder.forward
            
            def tracked_encoder(x):
                nonlocal encoder_called
                encoder_called = True
                return original_encoder(x)
            
            model.encoder.forward = tracked_encoder
            
            x = torch.ones((1, 5), dtype=torch.long)
            tgt = torch.ones((1, 4), dtype=torch.long)
            
            with torch.no_grad():
                output = model(x, tgt)
            
            model.encoder.forward = original_encoder
            
            if not encoder_called:
                t.failed = True
                t.msg = "encoder not being called in forward pass"
                t.want = "memory, src_padding_mask = self.encoder(x)"
                t.got = "encoder.forward() was never called"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test encoder call"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
                # Test 8: Check that decoder is called with correct arguments
        t = test_case()
        try:
            model = learner_class(src_vocab_size=50, tgt_vocab_size=60, d_model=32)
            model.eval()
            
            decoder_called_correctly = [False]
            original_decoder = model.decoder.forward
            
            def tracked_decoder(tgt_input, memory_input, mask_input=None):
                nonlocal decoder_called_correctly
                # Check if decoder receives encoder output
                if memory_input is not None and len(memory_input.shape) == 3:
                    decoder_called_correctly[0] = True
                return original_decoder(tgt_input, memory_input, mask_input)
            
            model.decoder.forward = tracked_decoder
            
            x = torch.ones((1, 5), dtype=torch.long)
            tgt = torch.ones((1, 4), dtype=torch.long)
            
            with torch.no_grad():
                output = model(x, tgt)
        
            model.decoder.forward = original_decoder
            
            if not decoder_called_correctly[0]:
                t.failed = True
                t.msg = "decoder not receiving encoder output correctly"
                t.want = "output = self.decoder(tgt, memory, src_padding_mask)"
                t.got = "decoder not called with encoder's memory output"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test decoder call"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 9: Check that padding mask is passed from encoder to decoder
        t = test_case()
        try:
            model = learner_class(src_vocab_size=50, tgt_vocab_size=60, d_model=32)
            model.eval()
            
            # Create input with padding
            batch_size = 2
            x = torch.ones((batch_size, 8), dtype=torch.long)
            x[0, 5:] = 0  # Add padding
            x[1, 6:] = 0  # Add padding
            tgt = torch.ones((batch_size, 6), dtype=torch.long)
            
            mask_passed_correctly = [False]
            original_decoder = model.decoder.forward
            
            def check_mask_passing(tgt_input, memory_input, mask_input=None):
                if mask_input is not None:
                    # Check if mask matches the padding pattern
                    expected_mask = (x == 0)
                    if torch.equal(mask_input, expected_mask):
                        mask_passed_correctly[0] = True
                return original_decoder(tgt_input, memory_input, mask_input)
            
            model.decoder.forward = check_mask_passing
            
            with torch.no_grad():
                output = model(x, tgt)
            
            model.decoder.forward = original_decoder
            
            if not mask_passed_correctly[0]:
                t.failed = True
                t.msg = "src_padding_mask not passed correctly from encoder to decoder"
                t.want = "decoder called with src_padding_mask from encoder"
                t.got = "mask not passed or incorrect"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test mask passing"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 10: Check correct number of parameters for standard config
        t = test_case()
        try:
            model = learner_class(
                src_vocab_size=5000,
                tgt_vocab_size=5000,
                d_model=256,
                nhead=8,
                num_enc_layers=3,
                num_dec_layers=3,
                dim_feedforward=512
            )
            
            actual_params = sum(p.numel() for p in model.parameters())
            SOLUTION_PARAMETERS = 7798664  
            
            if abs(actual_params - SOLUTION_PARAMETERS) > 0:
                t.failed = True
                t.msg = "Incorrect total number of parameters"
                t.want = f"{SOLUTION_PARAMETERS} parameters"
                t.got = f"{actual_params} parameters"
                return cases + [t]
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to check parameter count"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 11: Check different encoder and decoder layer counts
        t = test_case()
        try:
            # Test that encoder and decoder can have different layer counts
            test_enc_layers = 4
            test_dec_layers = 6
            
            model = learner_class(
                src_vocab_size=100,
                tgt_vocab_size=100,
                num_enc_layers=test_enc_layers,
                num_dec_layers=test_dec_layers,
                d_model=32
            )
            
            if len(model.encoder.transformer_encoder.layers) != test_enc_layers:
                t.failed = True
                t.msg = "num_enc_layers not properly set"
                t.want = f"{test_enc_layers} encoder layers"
                t.got = f"{len(model.encoder.transformer_encoder.layers)} encoder layers"
                return cases + [t]
            
            if len(model.decoder.transformer_decoder.layers) != test_dec_layers:
                t.failed = True
                t.msg = "num_dec_layers not properly set"
                t.want = f"{test_dec_layers} decoder layers"
                t.got = f"{len(model.decoder.transformer_decoder.layers)} decoder layers"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test different layer counts"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 12: Check dropout parameter passing
        t = test_case()
        try:
            test_dropout = 0.35
            model = learner_class(
                src_vocab_size=50,
                tgt_vocab_size=60,
                dropout=test_dropout,
                d_model=32
            )
            
            # Check encoder dropout
            if hasattr(model.encoder, 'dropout'):
                if model.encoder.dropout.p != test_dropout:
                    t.failed = True
                    t.msg = "dropout not properly passed to encoder"
                    t.want = f"encoder.dropout.p = {test_dropout}"
                    t.got = f"encoder.dropout.p = {model.encoder.dropout.p}"
                    return cases + [t]
            
            # Check decoder dropout
            if hasattr(model.decoder, 'dropout'):
                if model.decoder.dropout.p != test_dropout:
                    t.failed = True
                    t.msg = "dropout not properly passed to decoder"
                    t.want = f"decoder.dropout.p = {test_dropout}"
                    t.got = f"decoder.dropout.p = {model.decoder.dropout.p}"
                    return cases + [t]
                    
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test dropout parameter"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 13: Check feedforward dimension parameter
        t = test_case()
        try:
            test_dim_ff = 256
            model = learner_class(
                src_vocab_size=50,
                tgt_vocab_size=60,
                dim_feedforward=test_dim_ff,
                d_model=32
            )
            
            # Check encoder feedforward
            enc_layer = model.encoder.transformer_encoder.layers[0]
            if hasattr(enc_layer, 'linear1'):
                if enc_layer.linear1.out_features != test_dim_ff:
                    t.failed = True
                    t.msg = "dim_feedforward not properly passed to encoder"
                    t.want = f"encoder linear1.out_features = {test_dim_ff}"
                    t.got = f"encoder linear1.out_features = {enc_layer.linear1.out_features}"
                    return cases + [t]
            
            # Check decoder feedforward
            dec_layer = model.decoder.transformer_decoder.layers[0]
            if hasattr(dec_layer, 'linear1'):
                if dec_layer.linear1.out_features != test_dim_ff:
                    t.failed = True
                    t.msg = "dim_feedforward not properly passed to decoder"
                    t.want = f"decoder linear1.out_features = {test_dim_ff}"
                    t.got = f"decoder linear1.out_features = {dec_layer.linear1.out_features}"
                    return cases + [t]
                    
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test dim_feedforward parameter"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 14: Check that encoder and decoder are instances of correct classes
        t = test_case()
        try:
            model = learner_class(
                src_vocab_size=50,
                tgt_vocab_size=60,
                d_model=32
            )
            
            if not isinstance(model.encoder, encoder_class):
                t.failed = True
                t.msg = "encoder is not an instance of Encoder class"
                t.want = f"isinstance(model.encoder, Encoder) = True"
                t.got = f"encoder is instance of {type(model.encoder)}"
                return cases + [t]
            
            if not isinstance(model.decoder, decoder_class):
                t.failed = True
                t.msg = "decoder is not an instance of Decoder class"
                t.want = f"isinstance(model.decoder, Decoder) = True"
                t.got = f"decoder is instance of {type(model.decoder)}"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test encoder/decoder class types"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 15: Check nhead parameter
        t = test_case()
        try:
            test_nhead = 2  # Small value for testing
            test_d_model = 32  # Must be divisible by nhead
            
            model = learner_class(
                src_vocab_size=50,
                tgt_vocab_size=60,
                d_model=test_d_model,
                nhead=test_nhead
            )
            
            # Check encoder attention heads
            enc_layer = model.encoder.transformer_encoder.layers[0]
            if hasattr(enc_layer.self_attn, 'num_heads'):
                if enc_layer.self_attn.num_heads != test_nhead:
                    t.failed = True
                    t.msg = "nhead not properly passed to encoder"
                    t.want = f"encoder self_attn.num_heads = {test_nhead}"
                    t.got = f"encoder self_attn.num_heads = {enc_layer.self_attn.num_heads}"
                    return cases + [t]
            
            # Check decoder attention heads
            dec_layer = model.decoder.transformer_decoder.layers[0]
            if hasattr(dec_layer.self_attn, 'num_heads'):
                if dec_layer.self_attn.num_heads != test_nhead:
                    t.failed = True
                    t.msg = "nhead not properly passed to decoder"
                    t.want = f"decoder self_attn.num_heads = {test_nhead}"
                    t.got = f"decoder self_attn.num_heads = {dec_layer.self_attn.num_heads}"
                    return cases + [t]
                    
        except Exception as e:
            t.failed = True
            t.msg = f"Failed to test nhead parameter"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        # Test 16: Final integration test with different vocab sizes
        t = test_case()
        try:
            # Source and target vocabularies should be different
            src_vocab = 100
            tgt_vocab = 200
            
            model = learner_class(
                src_vocab_size=src_vocab,
                tgt_vocab_size=tgt_vocab,
                d_model=32
            )
            
            x = torch.randint(0, src_vocab, (2, 5))
            tgt = torch.randint(0, tgt_vocab, (2, 4))
            
            with torch.no_grad():
                output = model(x, tgt)
            
            # Output should have target vocabulary size
            if output.shape[-1] != tgt_vocab:
                t.failed = True
                t.msg = "Output dimension doesn't match target vocabulary"
                t.want = f"output.shape[-1] = {tgt_vocab}"
                t.got = f"output.shape[-1] = {output.shape[-1]}"
                return cases + [t]
                
        except Exception as e:
            t.failed = True
            t.msg = f"Failed integration test"
            t.got = str(e)
            return cases + [t]
        cases.append(t)
        
        return cases
    
    cases = g()
    print_feedback(cases)