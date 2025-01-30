import torch.nn as nn
from utils import AU_quantization_bit
from positional_encoding import PositionalEncoding
import torch
class AU_decoder(nn.Module):
    def __init__(self,d_model,nhead,num_decoder_layers):
        super(AU_decoder, self).__init__()
        self.transformer_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,batch_first =True), num_layers=num_decoder_layers)
        # initialize self.transformer_decoder
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)
        self.AU_encoder = nn.Embedding(AU_quantization_bit+4,d_model)
        self.linear = nn.Linear(d_model,AU_quantization_bit+4)

        self.pos_encoder = PositionalEncoding(d_model,max_len=50)

    def forward(self, x,au,AU_mask=None,F0_mask=None):

        au = self.AU_encoder(au.long())
        au = self.pos_encoder(au)
        au = self.transformer_decoder(au, x,tgt_key_padding_mask=AU_mask,memory_key_padding_mask=F0_mask)
        au = nn.functional.relu(au)
        au = self.linear(au)
        return au