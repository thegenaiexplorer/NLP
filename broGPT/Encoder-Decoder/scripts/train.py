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


class Char_Tokenizer:
    def __init__(self, text):
        super().__init__()
        self.text = text
        self.vocab = None
        self.tok_list = None
        self.t2i = None
        self.i2t = None
        self.len_vocab = None

    def create_vocab(self):
        self.vocab = set(self.text)
        return sorted(list(self.vocab))

    def train_tokenizer(self):
        self.tok_list = self.create_vocab()
        # tokens created . lets create forwrd and reverse mapping
        self.t2i = {token: index for index, token in enumerate(self.tok_list)}
        self.i2t = {index: token for index, token in enumerate(self.tok_list)}
        self.len_vocab = self.vocab_size()

    def vocab_size(self):
        if self.tok_list is None:
            print("errrrrr Error dude.. you should train the tokenizer first")
        else:
            return len(self.tok_list)

    def encode(self, text, return_tensors=False):
        inputs = [self.t2i[c] for c in text]
        if return_tensors:
            inputs = torch.tensor(inputs)
        return inputs

    def decode(self, inputs, tensors=False):
        if tensors:
            inputs = inputs.squeeze().tolist()
        toks = [self.i2t[i] for i in inputs]
        join_toks = ''.join(toks)
        #right now support is only for one sentence
        return join_toks

    def __str__(self):
        return f"token2index: {self.t2i},\
                index2token: {self.i2t},\
                vocab_len: {self.len_vocab}"

    def __repr__(self):
        return f"token2index: {self.t2i},\
                index2token: {self.i2t},\
                vocab_len: {self.len_vocab}"

class Bro_Shake_DS(Dataset):
    def __init__(self, vocab_text, block_size, tokenizer):
        super().__init__()
        self.data = vocab_text
        self.bs = block_size
        self.tokenizer = tokenizer

    def __len__(self):
        return int((len(self.data) - self.bs)/self.bs)

    def __getitem__(self, idx):
        idx = idx * self.bs
        assert idx < len(self.data) - self.bs
        x = self.tokenizer.encode(self.data[idx:idx+self.bs], return_tensors=True)
        y = self.tokenizer.encode(self.data[idx+1:idx+1+self.bs], return_tensors=True)
        return x, y


def create_data_loaders(data_path, batch_size, block_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_dir = data_path + 'train'
    cpu_cores = os.cpu_count()
    vocab_text = Path(train_dir+'/shakes_ds.txt').read_text()

    tokenizer = Char_Tokenizer(vocab_text)
    # let us train our tokenizer
    tokenizer.train_tokenizer()
    vocab_size = tokenizer.len_vocab

    train_ds = Bro_Shake_DS(vocab_text, block_size, tokenizer)
    # sampler = RandomSampler(train_ds, replacement=False, generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, 
                          num_workers=cpu_cores)

    return train_dl, train_ds, vocab_size, tokenizer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len, device):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0., max_len,
                                device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2, device=device) * -(math.log(10000.0) / d_model))
        pe_pos = torch.mul(position, div_term)
        pe[:, 0::2] = torch.sin(pe_pos)
        pe[:, 1::2] = torch.cos(pe_pos)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        out = self.pe[:, :x.size(1)].requires_grad_(False)
        return out


class Embed(nn.Module):
    def __init__(self, vocab_size, embed_dim, ctx_len, do, device):
        super().__init__()
        self.tok_layer = nn.Embedding(vocab_size, embed_dim)
        self.pos_layer = PositionalEncoding(embed_dim, do, ctx_len, device)
        self.norm_do = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(do)
            )
        self.device = device
        self.ctx_len = ctx_len

    def forward(self, inp):
        inp = inp.to(self.device)
        len_inp = inp.shape[-1]
        try:
            assert len_inp <= self.ctx_len
        except:
            print("Err..Errr. Error...Bro..Length of supplied text exceeds context length defined.Exiting the program now")
            sys.exit(1)
        tok_embed = self.tok_layer(inp)
        pos_embed = self.pos_layer(inp)
        embed_tok_pos = tok_embed + pos_embed
        embed_out = self.norm_do(embed_tok_pos)
        return embed_out


def att_mask(attention_mask, lookahead, cross_att, x=None):
    if cross_att:
        batch_dim = x[0]
        repeat = x[1]
    else:
        batch_dim = attention_mask.shape[0]
        repeat = len(attention_mask[0])

    mask =[]
    for i in range(batch_dim):
        am_interim = [attention_mask[i].tolist()] * repeat
        am_interim = torch.tensor(am_interim).unsqueeze(0)
        mask.append(am_interim)
    mask = torch.vstack(mask)
    if lookahead:
        inp_save = mask
        mask = torch.tril(torch.ones(mask.shape))
    mask = torch.where(mask == 0, -torch.inf, 0.0)
    return mask

class Attention(nn.Module):
    def __init__(self, embed_dim, k_dim, do, device):
        super().__init__()
        self.embed_dim = embed_dim
        self.k_dim = k_dim
        self.query = nn.Linear(embed_dim, k_dim)
        self.key = nn.Linear(embed_dim, k_dim)
        self.value = nn.Linear(embed_dim, k_dim)
        self.att_do = nn.Dropout(do)
        self.device = device

    def forward(self, qry, ky, vlu, mask):
        q = self.query(qry)
        k = self.key(ky)
        v = self.value(vlu)
        qk = (q@k.transpose(1, 2))/(self.k_dim**0.5)
        mask = mask.to(self.device)
        qk_m = qk + mask
        qk_m_smax = torch.softmax(qk_m, dim=-1)
        qk_m_smax_do = self.att_do(qk_m_smax)
        qkv = qk_m_smax_do@v
        return qkv

class Attention_Block(nn.Module):
    def __init__(self, num_heads, embed_dim, k_dim, do, device):
        super().__init__()
        self.num_heads = num_heads
        self.heads_list = [Attention(embed_dim, k_dim, do, device) for i in range(num_heads)]
        self.heads = nn.ModuleList(self.heads_list)
        self.lin_do = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(do)
        )

    def forward(self, q, k, v, mask):
        heads_list_out = [head(q, k, v, mask) for head in self.heads]
        att_head = torch.cat(heads_list_out, dim=-1)
        att_head_out = self.lin_do(att_head)
        return att_head_out


class Decoder_Block(nn.Module):
    def __init__(self, num_heads, embed_dim, k_dim, do, device, decoderonly=False):
        super().__init__()
        self.MHA_CAUSAL_ATTN = Attention_Block(num_heads, embed_dim, k_dim, do, device)
        self.mha_causal_end_layernorm = nn.LayerNorm(embed_dim)
        self.decoderonly = decoderonly
        if not decoderonly:
            self.MHA_CROSS_ATTN = Attention_Block(num_heads, embed_dim, k_dim, do, device)
            self.mha_cross_end_layernorm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim),
            nn.Dropout(do)
            )
        self.dec_blk_end_layernorm = nn.LayerNorm(embed_dim)

    def forward(self, args_list):
        q = args_list[0]
        enc_k = args_list[1] 
        enc_v = args_list[2] 
        causal_mask = args_list[3] 
        padding_mask = args_list[4]

        # Causal Attention Block
        #################################################
        inp_start_causal_att_block = q
        x = self.MHA_CAUSAL_ATTN(q, q, q, causal_mask)
        x = x + inp_start_causal_att_block
        x = self.mha_causal_end_layernorm(x)

        # Cross Attention Block
        ###################################################
        if not self.decoderonly:
            inp_end_causal_att_block = x
            x = self.MHA_CROSS_ATTN(x, enc_k, enc_v, padding_mask)
            x = x + inp_end_causal_att_block
            x = self.mha_cross_end_layernorm(x)
        ###################################################

        inp_start_ff_block = x
        x = self.ff(x)
        x = x + inp_start_ff_block
        x = self.dec_blk_end_layernorm(x)
        return [x, enc_k, enc_v, causal_mask, padding_mask]


class BroGPT(nn.Module):
    def __init__(self, num_layers, num_heads, vocab_size,
                 embed_dim, ctx_len, do, device):
        super().__init__()
        k_dim = int(embed_dim/num_heads)
        self.emb = Embed(vocab_size, embed_dim, ctx_len, do, device)
        self.layer_list = [Decoder_Block(num_heads, embed_dim, k_dim, do, device, True) for i in range(num_layers)]
        self.layers = nn.Sequential(*self.layer_list)
        self.embed_vocab = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, attention_mask): # , enc_k=None, enc_v=None):
        x = self.emb(input_ids)
        causal_mask = att_mask(attention_mask, lookahead=True, cross_att=False)
        padding_mask = att_mask(attention_mask, lookahead=False, cross_att=False)
        #Let us fuse the two together. This is because causal should not be paying attention when there is a padding
        causal_mask = causal_mask + padding_mask
        padding_mask = None
        x = self.layers([x, x, x, causal_mask, padding_mask])
        x = self.embed_vocab(x[0])
        return x


def net(args, vocab_size, device):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''

    num_layers = args.num_layers
    num_heads = args.num_heads
    embed_dim = args.embed_dim
    ctx_len = args.ctx_len
    do = args.do

    model = BroGPT(num_layers, num_heads, vocab_size,
                   embed_dim, ctx_len, do, device)
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
            out = out.view(batch_size * ctx_size, vocab_size)
            label = label.view(batch_size * ctx_size)
            loss = fn_loss(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch = loss_epoch + loss.item()
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