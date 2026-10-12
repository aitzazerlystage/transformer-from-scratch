from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype=torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype=torch.int64)


    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids


        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # 2 because of SOS and EOS tokens
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # 1 because only SOS will be added to decoder, and the EOS of the sentence will be added to Target/ Label Values.


        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')


    # Add SOS and EOS to the Source text
        encoder_input = torch.cat(
            [
                self.sos_token,   #Start Token
                torch.tensor(enc_input_tokens, dtype=torch.int64), #Sentence Tokens
                self.eos_token, # End Token
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64) #Padding Tokens
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token, #Start Token
                torch.tensor(dec_input_tokens, dtype=torch.int64), #Sentence Tokens
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64) # Padding Tokens, Note: The EOS token is nto added to decoder but to the Target, and the Target does not have SOS Token.
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input" : encoder_input,
            "decoder_input" : decoder_input,
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int().int(),  # Not to Compare Padding tokens in Self attention
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0) & casual_mask(decoder_input.size(0)), # Not to compare padding tokens AND the Tokens after the word postiion. Meaning, only to compare the Token/word with the words that come before it. not with the words that come after it.
            "label" : label,
            "src_text" : src_text,
            "tgt_text" : tgt_text
        }

def casual_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
        

