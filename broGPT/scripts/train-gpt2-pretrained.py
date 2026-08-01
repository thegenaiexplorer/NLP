#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
##Implement OCLR/LR Scheduler + option to run full runing or only classifier layer
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
from transformers import AutoTokenizer, AutoModelForCausalLM


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

    args = parser.parse_args()
    logger.info(f'Starting training job with parameters: {args}')
    return args


class Bro_Shake_DS(Dataset):
    def __init__(self, vocab_text, tokenizer, bs):
        super().__init__()
        self.data = vocab_text
        self.bs = bs
        self.tokenizer = tokenizer
        self.toks_ds = []
        vocab_text_split = vocab_text.split()
        bs_chunks = int(len(vocab_text_split)/bs)
        for i in range(bs_chunks):
            start_indx = i*bs
            end_indx = start_indx + bs
            chunk = ' '.join(vocab_text_split[start_indx:end_indx])
            toks = tokenizer.encode(chunk)
            self.toks_ds.extend(toks)

    def __len__(self):
        return int((len(self.toks_ds) - self.bs)/self.bs)

    def __getitem__(self, idx):
        idx = idx * self.bs
        assert idx < len(self.data) - self.bs
        x = torch.tensor(self.toks_ds[idx:idx+self.bs])
        y = torch.tensor(self.toks_ds[idx+1:idx+1+self.bs])

        return x, y

def create_data_loaders(data_path, batch_size, block_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_dir = data_path + 'train'
    cpu_cores = os.cpu_count()
    vocab_text = Path(train_dir+'/shakes_ds.txt').read_text()
    ckpt = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    vocab_size = tokenizer.vocab_size
    
    train_ds = Bro_Shake_DS(vocab_text, tokenizer, block_size)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, 
                          num_workers=cpu_cores)

    return train_dl, train_ds, vocab_size, tokenizer


def net(args, vocab_size, device):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    ckpt = "openai-community/gpt2"
    model = AutoModelForCausalLM.from_pretrained(ckpt)
    total_params = sum([p.numel() for p in model.parameters()])
    logger.info(f'There are {total_params/(1024**2)} parameters in {ckpt} model')
    model.to(device)
    return model


def train_model(args, brogpt, dl, device, vocab_size):
    epochs = args.num_epochs
    lr = args.lrate
    fn_loss = nn.CrossEntropyLoss()
    optimizer = AdamW(brogpt.parameters(), lr=lr)
    num_params = sum([p.numel() for p in brogpt.parameters()])
    logger.info(f'There are {num_params} parameters in our model')

    for i in range(epochs):
        loss_epoch = 0
        n_step = 0
        for inputs in dl:
            data = inputs[0]
            label = inputs[1]
            batch_size = data.shape[0]
            ctx_size = data.shape[1]
            data, label = data.to(device), label.to(device)
            out = brogpt(data)
            out = out.logits  # Code added for hugging face based transformer models
            out = out.view(batch_size * ctx_size, vocab_size)
            label = label.view(batch_size * ctx_size)
            loss = fn_loss(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch = loss_epoch + loss.item()
            if n_step % 10 == 0:
                logger.info(f'Step: {n_step}, Loss: {loss.item()}')
            n_step += 1
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
    train_dl, train_ds, vocab_size, tokenizer = create_data_loaders(data_path,
                                                                    batch_size,
                                                                    block_size
                                                                    )
    logger.info("STEP 1 completed successfully...") #checks complete

    logger.info("STEP 2: Initializing and loading model for training")
    model = net(args, vocab_size, device)
    logger.info("STEP 2 completed successfully") #checks complete

    logger.info("STEP 3: Initiating model training")
    logger.info(f'Model will be trained for {args.num_epochs} Epochs')
    model, optimizer, average_loss, perplexity = train_model(args, model, train_dl, device, vocab_size)
    logger.info("STEP 3 completed successfully")
    logger.info("#############################################################")
    logger.info("########## MODEL TRAINING COMPLETED SUCCESSFULLY ############")

    logger.info(f'STEP 4: Saving model checkpoint to {ckpt_path}')
    os.makedirs(ckpt_path, exist_ok=True)
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