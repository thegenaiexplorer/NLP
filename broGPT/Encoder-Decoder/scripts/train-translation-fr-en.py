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
from peft import AutoPeftModelForCausalLM
from transformers import TrainingArguments, Trainer
from torchinfo import summary
import tarfile
from sklearn.model_selection import train_test_split


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


class Encoder_Block(nn.Module):
    def __init__(self, num_heads, embed_dim, k_dim, do, device):
        super().__init__()
        self.MHA = Attention_Block(num_heads, embed_dim, k_dim, do, device)
        self.mha_blk_end_layernorm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim),
            nn.Dropout(do)
            )
        self.enc_blk_end_layernorm = nn.LayerNorm(embed_dim)

    def forward(self, args_list):  #q, k, v, mask):
        x = args_list[0]
        mask = args_list[1]

        inp_start_att_block = x
        x = self.MHA(x, x, x, mask)

        x = x + inp_start_att_block
        x = self.mha_blk_end_layernorm(x)

        inp_start_ff_block = x
        x = self.ff(x)

        x = x + inp_start_ff_block
        x = self.enc_blk_end_layernorm(x)
        return [x, mask]


class Encoder(nn.Module):
    def __init__(self, num_layers, num_heads, vocab_size, embed_dim, k_dim, ctx_len, do, device):
        super().__init__()
        self.emb = Embed(vocab_size, embed_dim, ctx_len, do, device)
        self.layer_list = [Encoder_Block(num_heads, embed_dim, k_dim, do, device) for i in range(num_layers)]
        self.layers = nn.Sequential(*self.layer_list)

    def forward(self, input_ids, attention_mask):
        x = self.emb(input_ids)
        mask = att_mask(attention_mask, lookahead=False, cross_att=False, x=None)
        x = self.layers([x, mask])
        return x[0]


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


class Decoder(nn.Module):
    def __init__(self, num_layers, num_heads, vocab_size, embed_dim, 
                 k_dim, ctx_len, do, device):
        super().__init__()
        self.emb = Embed(vocab_size, embed_dim, ctx_len, do, device)
        self.layer_list = [Decoder_Block(num_heads, embed_dim, k_dim, do, device) for i in range(num_layers)]
        self.layers = nn.Sequential(*self.layer_list)


    def forward(self, input_ids, attention_mask, enc_attention_mask, enc_k, enc_v):
        x = self.emb(input_ids)
        q = x
        dim = [x.shape[0], attention_mask.shape[1], enc_attention_mask.shape[1]]
        causal_mask = att_mask(attention_mask, lookahead=True, cross_att=False)
        padding_mask = att_mask(enc_attention_mask, lookahead=False, cross_att=True, x=dim)
        x = self.layers([q, enc_k, enc_v, causal_mask, padding_mask])
        return x[0]


class Bro_Transformer(nn.Module):

    def __init__(self, num_layers, embed_dim, num_heads,
                 input_vocab_size, target_vocab_size, enc_ctx_len,
                 dec_ctx_len, device, do=0.1):
        super().__init__()
        k_dim = int(embed_dim/num_heads)

        self.encoder = Encoder(num_layers, num_heads, input_vocab_size, embed_dim, k_dim, enc_ctx_len, do, device)
        self.decoder = Decoder(num_layers, num_heads, target_vocab_size, embed_dim, k_dim, dec_ctx_len, do, device)
        self.final_layer = nn.Linear(embed_dim, target_vocab_size)

    def forward(self, enc_input_ids, dec_input_ids, enc_attention_mask, dec_attention_mask):

        enc_output = self.encoder(enc_input_ids, enc_attention_mask)
        dec_output = self.decoder(dec_input_ids, dec_attention_mask,
                                  enc_attention_mask, enc_output, enc_output)
        final_output = self.final_layer(dec_output)
        return final_output


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




class EN_FR_DS(Dataset):
    def __init__(self, df):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fr_item = self.data.iloc[idx]['fr']
        en_item = '<|endoftext|>' + self.data.iloc[idx]['en']
        return fr_item, en_item


def collate_fn(batch, en_tokenizer, fr_tokenizer):
    eos_tokenid = fr_tokenizer.encode('<|endoftext|>')
    pad_tokenid = fr_tokenizer.encode('<|endoftext|>')
    ignore_loss_id = [-100]  #ignore loss
    ignore_token_id = [0]   #attention mask filler
    x_input_ids, x_am = [], []
    y_input_ids, y_am, lab_list = [], [], []

    for x, y in batch:
        x_tok = fr_tokenizer(x)
        y_tok = en_tokenizer(y)

        x_tok_input_ids = x_tok.input_ids
        x_tok_am = x_tok.attention_mask

        y_tok_input_ids = y_tok.input_ids
        y_tok_am = y_tok.attention_mask

        labs = y_tok_input_ids[1:] + eos_tokenid

        x_input_ids.append(x_tok_input_ids)
        x_am.append(x_tok_am)

        y_input_ids.append(y_tok_input_ids)
        y_am.append(y_tok_am)

        lab_list.append(labs)

    len_enc = [len(l) for l in x_input_ids]
    max_len_enc = max(len_enc)

    len_labs = [len(l) for l in lab_list]
    max_len_labs = max(len_labs)

    max_len = max([max_len_enc, max_len_labs])

    def pad_tokens_stack(in_list, max_len, padding):
        list_tensor = []
        for item in in_list:
            len_item = len(item)
            deficit = max_len - len_item
            if deficit > 0:
                item = item + padding*deficit
            list_tensor.append(torch.tensor(item))
        item_ids = torch.vstack(list_tensor)
        return item_ids

    enc_input_ids = pad_tokens_stack(x_input_ids, max_len, pad_tokenid)
    enc_am = pad_tokens_stack(x_am, max_len, ignore_token_id)

    dec_input_ids = pad_tokens_stack(y_input_ids, max_len, pad_tokenid)
    dec_am = pad_tokens_stack(y_am, max_len, ignore_token_id)

    labels = pad_tokens_stack(lab_list, max_len, ignore_loss_id)

    return {'enc_input_ids': enc_input_ids, 'dec_input_ids': dec_input_ids,
            'enc_attention_mask': enc_am, 'dec_attention_mask': dec_am}, labels


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
        df['input_ids_len'] = df['prompt'].apply(lambda x: len(tokenizer(x, truncation=True).input_ids))
        df = df.sort_values(by=['input_ids_len'], ascending=True)
        df_1 = df[df['input_ids_len'] <= 256].copy()
        df_1.reset_index(inplace=True, drop=True)
        df_1['id'] = range(len(df_1))
        logger.info(f'length of dataframe is {len(df_1)}')
        logger.info(f'{df_1.describe()}')
        return df_1

    cpu_cores = os.cpu_count()

    train_dir = data_path + 'train'
    csv_path = train_dir+'/en2fr.csv'
    df = pd.read_csv(csv_path)

    fr_tokenizer = AutoTokenizer.from_pretrained('Fardan/gpt2-frenchtranslation-tokenizer')
    en_tokenizer = AutoTokenizer.from_pretrained('Fardan/gpt2-englishtranslation-tokenizer')
    en_vocab_size = en_tokenizer.vocab_size
    fr_vocab_size = fr_tokenizer.vocab_size

    #df = sample_sort_ds(df, tokenizer)
    df_train, df_valid = train_test_split(df, test_size=0.01, random_state=42)
    train_ds = EN_FR_DS(df_train)
    valid_ds = EN_FR_DS(df_valid)

    wrapper_collate_fn = partial(
        collate_fn,
        en_tokenizer=en_tokenizer,
        fr_tokenizer=fr_tokenizer
        )
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size,
                          num_workers=cpu_cores, collate_fn=wrapper_collate_fn)
    valid_dl = DataLoader(valid_ds, shuffle=True, batch_size=batch_size,
                          num_workers=cpu_cores, collate_fn=wrapper_collate_fn)
    
    return train_dl, train_ds, valid_dl, valid_ds, fr_tokenizer, en_tokenizer, en_vocab_size, fr_vocab_size


def process_extract(data_path):
    logger.info('Entering process_extract function now')
    model_path = data_path + 'train/model_data/'
    logger.info(f'input_model path: {model_path}')
    logger.info('Extracting model.tar.gz')
    model_tar_path = model_path + 'model.tar.gz'
    model_tar = tarfile.open(model_tar_path)
    model_tar.extractall(model_path)
    model_tar.close()
    logger.info("Model extraction completed")
    return model_path


def create_model(args, data_path, device):
    model_dir = process_extract(data_path)
    dir_list = os.listdir(model_dir)
    ckpt_name = [i for i in dir_list if i.startswith('ckpt')][0]
    chkpt_file_path = model_dir + ckpt_name
    logger.info(f'model_dir path {model_dir}')
    logger.info(f'content of model_dir path {model_dir}: {dir_list}')
    logger.info(f'checkpoint file path {chkpt_file_path}')
    logger.info(f'contents of checkpoint file path {chkpt_file_path}: {os.listdir(chkpt_file_path)}')
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = args.ckpt
    orig_model = AutoModelForCausalLM.from_pretrained(ckpt)
    model = PeftModel.from_pretrained(orig_model, chkpt_file_path, is_trainable=True)
    model.to(device)
    logger.info(f'{summary(model)}')
    return model

#######################################################################################


def net(args, device, fr_vocab_size, en_vocab_size):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = Bro_Transformer(num_layers=args.num_layers, embed_dim=args.embed_dim,
                            num_heads=args.num_heads, input_vocab_size=fr_vocab_size,
                            target_vocab_size=en_vocab_size, enc_ctx_len=args.ctx_len,
                            dec_ctx_len=args.ctx_len, device=device, do=0.1)

    s = summary(model)
    logger.info(f"Total number of trainable parameters: {s.__dict__['trainable_params']}")

    model.to(device)
    return model


def train_model(args, ckpt_path, brogpt, train_dl, valid_dl, device, vocab_size):
    epochs = args.num_epochs
    lr = args.lrate
    fn_loss = nn.CrossEntropyLoss()
    optimizer = AdamW(brogpt.parameters(), lr=lr)
    grad_accum_steps = args.grad_accum_steps
    base_val_loss = torch.inf

    for i in range(epochs):
        brogpt.train()
        loss_epoch = 0
        n_step = 0
        grad_accum_counter = 1

        for inputs in train_dl:
            data = inputs[0]
            label = inputs[1]
            batch_size = label.shape[0]
            ctx_size = label.shape[1]
            data = {i: k.to(device) for i, k in data.items()}
            label = label.to(device)
            out = brogpt(**data)
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
        average_loss = loss_epoch/len(train_dl.dataset)
        perplexity = math.exp(average_loss)
        logger.info("############# STATS for Current Epoch - START ##############")
        logger.info(f'Epoch: {i} -- Average loss: {average_loss}')
        logger.info(f'Epoch: {i} -- Perplexity: {perplexity}')
        logger.info("############")
        average_valid_loss, valid_perplexity = validate_model(i, brogpt, valid_dl, device, vocab_size)
        logger.info("############## STATS for Current Epoch - END ############")
        if average_valid_loss < base_val_loss:
            base_val_loss = average_valid_loss
            save_model(ckpt_path, "model.pt", i, brogpt, optimizer, base_val_loss, device)

    return brogpt, optimizer, average_loss, perplexity, average_valid_loss, valid_perplexity


def save_model(ckpt_path, ckpt_name, epoch_value, model, optimizer, average_loss, device):
    os.makedirs(ckpt_path, exist_ok=True)
    ckpt_full_path = ckpt_path + ckpt_name
    torch.save({
        'epoch': epoch_value,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': average_loss,
        'device': device
        }, ckpt_full_path)
    logger.info(f"Checkpoint saved at epoch {epoch_value}")

def validate_model(i, brogpt, valid_dl, device, vocab_size):
    fn_loss = nn.CrossEntropyLoss()
    brogpt.eval()
    loss_epoch = 0
    for inputs in valid_dl:
        data = inputs[0]
        label = inputs[1]
        batch_size = label.shape[0]
        ctx_size = label.shape[1]
        data = {i: k.to(device) for i, k in data.items()}
        label = label.to(device)
        with torch.no_grad():
            out = brogpt(**data)
        out = out.view(batch_size * ctx_size, vocab_size)
        label = label.view(batch_size * ctx_size)
        loss = fn_loss(out, label)
        loss_epoch = loss_epoch + (loss.item() * batch_size)
    average_loss = loss_epoch/len(valid_dl.dataset)
    perplexity = math.exp(average_loss) 
    logger.info(f'Epoch: {i} -- Validation Average loss: {average_loss}')
    logger.info(f'Epoch: {i} -- Validation Perplexity: {perplexity}')
    return average_loss, perplexity


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
    train_dl, train_ds, valid_dl, valid_ds, fr_tokenizer, en_tokenizer, en_vocab_size, fr_vocab_size = create_data_loaders(args, data_path, batch_size)
    logger.info("STEP 1 completed successfully...")

    logger.info("STEP 2: Initializing and loading model for training")
    model = net(args, device, fr_vocab_size, en_vocab_size)
    logger.info("STEP 2 completed successfully")

    logger.info("STEP 3: Initiating model training")
    logger.info(f'Model will be trained for {args.num_epochs} Epochs')
    model, optimizer, average_loss, perplexity, average_valid_loss, valid_perplexity = train_model(args, ckpt_path, model, train_dl, valid_dl, device, en_vocab_size)

    logger.info("STEP 3 completed successfully")
    logger.info("#############################################################")
    logger.info("########## MODEL TRAINING COMPLETED SUCCESSFULLY ############")

    logger.info(f'STEP 4: Saving model checkpoint to {ckpt_path}')
    save_model(ckpt_path, "train_end_model.pt", args.num_epochs, model, optimizer, average_valid_loss, device)


if __name__ == '__main__':
    logger.info("Processing arguments now")
    args = parse_inputs()
    logger.info(f'Parsed arguments are {args}')
    logger.info("Invoking main function with parsed arguments")
    main(args)

    logger.info("Woohoo All Done Bro. Chillax!!")