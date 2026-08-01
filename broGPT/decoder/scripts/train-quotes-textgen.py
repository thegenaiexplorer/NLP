import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler
import os
import argparse
import sys
from pathlib import Path
import logging
import shutil
import glob
import pandas as pd
import numpy as np
import math
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import partial
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftConfig, PeftModel 
from peft import PromptTuningConfig, PromptTuningInit
from transformers import TrainingArguments, Trainer
from torchinfo import summary


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_inputs():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning Job")
    parser.add_argument('--bs',
                        type=int,
                        default=32,
                        metavar="N",
                        help="Batch Size for training (default:32)"
                       )
    parser.add_argument('--lrate',
                        type=float,
                        default=0.0002,
                        metavar="LR",
                        help="Learning Rate for training (default:0.0002)"
                       )
    parser.add_argument('--num_epochs',
                        type=int,
                        default=3,
                        metavar="N",
                        help="Number of epochs to train (default:3)"
                       )
    parser.add_argument('--do',
                        type=float,
                        default=0.2,
                        help="Dropout to use for model training. Default: 0.2",
                        )
    parser.add_argument('--block_size',
                        type=int,
                        default=64,
                        help="Number of characters to fetch in one go. Default: 64",
                        )
    parser.add_argument('--environ',
                        type=str,
                        default="colab",
                        help="Set the cloud environment values: aws, colab. Default: colab",
                        )
    parser.add_argument('--num_layers',
                        type=int,
                        default=6,
                        help="Number of transformer blocks. Default: 6",
                        )
    parser.add_argument('--num_heads',
                        type=int,
                        default=12,
                        help="Number of heads in MHA block. Default: 12",
                        )
    parser.add_argument('--embed_dim',
                        type=int,
                        default=768,
                        help="Number of dimensions in embeddings. Default: 768",
                        )
    parser.add_argument('--k_dim',
                        type=int,
                        default=64,
                        help="Number of dimensions in key vector. Default: 64",
                        )
    parser.add_argument('--ctx_len',
                        type=int,
                        default=64,
                        help="Number of tokens that can be processed in one go. Default: 64",
                        )
    parser.add_argument('--ckpt',
                        type=str,
                        default="gpt2",
                        help="Model checkpoint. Default:gpt2",
                        )
    parser.add_argument('--quant_bits',
                        type=int,
                        default=0,
                        help="Quantization Bits for loading models. Default:0",
                        )
    parser.add_argument('--peft',
                        type=str,
                        default="none",
                        help="Peft technique to be used. Only lora supported. Default:none",
                        )
    parser.add_argument('--grad_accum_steps',
                        type=int,
                        default=1,
                        help="Quantization Bits for loading models. Default:0",
                        )

    args = parser.parse_args()
    logger.info(f'Starting training job with parameters: {args}')
    return args


class dset(Dataset):
    def __init__(self, df, tokenizer):
        super().__init__()
        self.data = df['full_text']
        self.tokenizer = tokenizer
        self.eos_tokid = [tokenizer.vocab[tokenizer.eos_token]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data.iloc[idx]
        x = self.tokenizer(x, truncation=True)
        y = x['input_ids'][1:] + self.eos_tokid
        return x, y


def collate_fn(batch, tokenizer):
    pad_tokenid = [tokenizer.vocab[tokenizer.pad_token]]
    inp_list, att_list, y_list = [],  [], []
    len_labs = [len(l) for i, l in batch]
    max_len = max(len_labs)
    for _text, _label in batch:
        inp_ids = _text['input_ids']
        am = _text['attention_mask']
        label = _label
        if len(inp_ids) < max_len:
            deficit = max_len - len(inp_ids)
            inp_ids = inp_ids + pad_tokenid*deficit
            am = am + [0]*deficit
            label = label + [-100]*deficit
        inp_list.append(torch.tensor(inp_ids))
        att_list.append(torch.tensor(am))
        y_list.append(torch.tensor(label))
    input_ids = torch.vstack(inp_list)
    attention_mask = torch.vstack(att_list)
    labels = torch.vstack(y_list)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}, labels


def create_data_loaders(args, data_path, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''

    def sample_sort_ds(df, tokenizer):
        '''
        1. This function converts dataset to pandas.
        2. combines summary and dialogue
        3. tokenizes the text in point no 3 above and calculates total number of tokens
        4. sorts the dataframe in ascending order of total number of tokens
        5. Selects all the records where total number of tokens is less than 512
        6. This also has benefit of combining similar length datapoints together to avoid
        wasting padding tokens
        '''
        df['input_ids_len'] = df['full_text'].apply(lambda x: len(tokenizer(x, truncation=True).input_ids))
        df = df.sort_values(by=['input_ids_len'], ascending=True)
        df_1 = df[df['input_ids_len'] < 256].copy()
        df_1.reset_index(inplace=True, drop=True)
        df_1['id'] = range(len(df_1))
        logger.info(f'length of dataframe is {len(df_1)}')
        logger.info(f'{df_1.describe()}')
        return df_1

    train_dir = data_path + 'train'
    cpu_cores = os.cpu_count()
    csv_path = train_dir+'/quotes.csv'
    df = pd.read_csv(csv_path)
    ckpt = args.ckpt
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt)
    vocab_size = model.config.vocab_size

    df = sample_sort_ds(df, tokenizer)
    train_ds = dset(df, tokenizer)

    wrapper_collate_fn = partial(
        collate_fn,
        tokenizer=tokenizer
        )
    train_dl = DataLoader(train_ds, shuffle=False, batch_size=batch_size, 
                          num_workers=cpu_cores, collate_fn=wrapper_collate_fn)
    del model
    return train_dl, train_ds, tokenizer, vocab_size


def net(args, device):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    ckpt = args.ckpt
    config = None
    peft_config = None
    ###################################################################
    prompt_config = PromptTuningConfig(
        task_type="CAUSAL_LM",
        prompt_tuning_init=PromptTuningInit.RANDOM,
        num_virtual_tokens=6,
        tokenizer_name_or_path=ckpt
    )
    ###################################################################
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        task_type="CAUSAL_LM" # TaskType.CAUSAL_LM",
        )
    config_4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
        )
    config_8bit = BitsAndBytesConfig(
        load_in_8bit=True,
        )

    orig_model = AutoModelForCausalLM.from_pretrained(ckpt)

    if args.peft == "lora":
        peft_config = lora_config
        if args.quant_bits == 4:
            config = config_4bit
        elif args.quant_bits == 8:
            config = config_8bit
        else:
            config = None
    elif args.peft == "prompt_tuning":
        peft_config = prompt_config
    else:
        pass

    if config:
        logger.info("Model Quantization requested")
        quant_model = AutoModelForCausalLM.from_pretrained(
            ckpt, 
            quantization_config=config
        )
        quant_model.gradient_checkpointing_enable()
        quant_model = prepare_model_for_kbit_training(quant_model)
        model = get_peft_model(quant_model, lora_config)
    elif args.peft != "none":
        logger.info(f'Only PEFT. PEFT method: {args.peft}. No Model Quantization requested')
        model = get_peft_model(orig_model, peft_config)
    else:
        logger.info("NO PEFT. No Model Quantization requested")
        model = orig_model

    #total_params = sum([p.numel() for p in model.parameters()])
    s = summary(model)
    logger.info(f"Total number of trainable parameters: {s.__dict__['trainable_params']}")
    #logger.info(f'There are {total_params/(1024**2)} parameters in {ckpt} model')

    model.to(device)
    return model


def train_model(args, brogpt, dl, device, vocab_size):
    epochs = args.num_epochs
    lr = args.lrate
    fn_loss = nn.CrossEntropyLoss()
    optimizer = AdamW(brogpt.parameters(), lr=lr)
    #num_params = sum([p.numel() for p in brogpt.parameters()])
    #logger.info(f'There are {num_params} parameters in our model')
    grad_accum_steps = args.grad_accum_steps # 4 - previous static value

    for i in range(epochs):
        loss_epoch = 0
        n_step = 0
        grad_accum_counter = 1

        for inputs in dl:
            data = inputs[0]
            label = inputs[1]
            batch_size = label.shape[0]
            ctx_size = label.shape[1]
            data = {i: k.to(device) for i, k in data.items()}
            label = label.to(device)
            out = brogpt(**data)
            out = out.logits  # Code added for hugging face based transformer models
            out = out.view(batch_size * ctx_size, vocab_size)
            label = label.view(batch_size * ctx_size)
            loss = fn_loss(out, label)
            loss = loss / grad_accum_steps
            loss.backward()
            if grad_accum_counter == grad_accum_steps:
                #logger.info(f'Steps: {grad_accum_counter}, adjusting learnable params now')
                optimizer.step()
                optimizer.zero_grad()
                grad_accum_counter = 0
            loss_epoch = loss_epoch + (loss.item() * batch_size * grad_accum_steps)
            if n_step % 100 == 0:
                logger.info(f'Step: {n_step}, Loss: {loss.item()}')
            grad_accum_counter += 1
            n_step += 1
            ##update so that if remaining dataloader runs are less than grad accum steps then at the last run i should
            #gradient update
        average_loss = loss_epoch/len(dl.dataset)
        perplexity = math.exp(average_loss) 
        logger.info(f'Epoch: {i} -- Average loss: {average_loss}')
        logger.info(f'Epoch: {i} -- Perplexity: {perplexity}')

    return brogpt, optimizer, average_loss, perplexity


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    data_path = '/opt/ml/input/data/'
    inference_path = '/opt/ml/model/code'
    ckpt_path = '/opt/ml/model/'
    if args.environ != 'aws':
        data_path = './opt/ml/input/data/'
        inference_path = './opt/ml/model/code'
        ckpt_path = './opt/ml/model/'
        os.makedirs(inference_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(ckpt_path, exist_ok=True)

    batch_size = args.bs
    block_size = args.block_size
    logger.info(f'data_path is {data_path}')
    logger.info(f'batch_size is {batch_size}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Training Device is {device}')

    logger.info("STEP 1: Creating training dataloaders...")
    train_dl, train_ds, tokenizer, vocab_size = create_data_loaders(args,
                                                                    data_path,
                                                                    batch_size
                                                                    )
    logger.info("STEP 1 completed successfully...")

    logger.info("STEP 2: Initializing and loading model for training")
    model = net(args, device)
    logger.info("STEP 2 completed successfully")

    logger.info("STEP 3: Initiating model training")
    logger.info(f'Model will be trained for {args.num_epochs} Epochs')
    model, optimizer, average_loss, perplexity = train_model(args, model,
                                                             train_dl,
                                                             device,
                                                             vocab_size)
    logger.info("STEP 3 completed successfully")
    logger.info("#############################################################")
    logger.info("########## MODEL TRAINING COMPLETED SUCCESSFULLY ############")

    logger.info(f'STEP 4: Saving model checkpoint to {ckpt_path}')
    os.makedirs(ckpt_path, exist_ok=True)
    if args.peft == "lora":
        tr = Trainer(
            model=model,
            tokenizer=tokenizer
            )
        save_path = ckpt_path + 'ckpt_' + "Epoch_" + str(args.num_epochs) + "_Prplxty_" + str(perplexity)
        tr.save_model(save_path)
    else:
        ckpt_name = ckpt_path + 'ckpt_' + "Epoch_" + str(args.num_epochs) + "_Prplxty_" + str(perplexity) + '.pt'
        torch.save({
            'epoch': args.num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_loss,
            'device': device
            }, ckpt_name)


if __name__ == '__main__':
    logger.info("Processing arguments now")
    args = parse_inputs()
    logger.info(f'Parsed arguments are {args}')
    logger.info("Invoking main function with parsed arguments")
    main(args)

    logger.info("Woohoo All Done Bro. Chillax!!")