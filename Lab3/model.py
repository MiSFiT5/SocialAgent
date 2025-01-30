import functorch.dim
import torch.nn as nn
from positional_encoding import PositionalEncoding
from decoder import AU_decoder
from F0encoder import F0_encoder
import torch
import torch.nn.functional as F
from utils import au_beg,au_end

class AU_predictor(nn.Module):
    def __init__(self,d_model,nhead,num_encoder_layers,num_decoder_layers):
        super(AU_predictor, self).__init__()
        self.F0_encoder = F0_encoder(d_model=d_model)

        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,batch_first =True), num_layers=num_encoder_layers)
        self.transformer_decoder_au01 = AU_decoder(d_model=d_model, nhead=nhead,num_decoder_layers=num_decoder_layers)
        self.transformer_decoder_au02 = AU_decoder(d_model=d_model, nhead=nhead,num_decoder_layers=num_decoder_layers)
        self.transformer_decoder_au04 = AU_decoder(d_model=d_model, nhead=nhead,num_decoder_layers=num_decoder_layers)
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)

        self.pos_encoder = PositionalEncoding(d_model,max_len=100)
    def forward(self, AUs, F0,AU_mask,F0_mask):

        if F0_mask is not None:
            F0_mask = F0_mask[:,3:-3].float()

        F0 = F0.float()
        F0 = self.F0_encoder(F0.unsqueeze(1))

        F0 = F0.permute(0,2,1)
        F0 = self.pos_encoder(F0)
        x = self.transformer_encoder(F0,src_key_padding_mask=F0_mask)

        au1 = AUs[:,0]

        au1 = self.transformer_decoder_au01(x,au1,AU_mask,F0_mask)

        au2 = AUs[:,1]
        au2 = self.transformer_decoder_au02(x,au2,AU_mask,F0_mask)

        au4 = AUs[:,2]
        au4 = self.transformer_decoder_au04(x,au4,AU_mask,F0_mask)



        return au1,au2,au4

    def generate(self,F0, F0_mask):

        if F0_mask is not None:
            F0_mask = F0_mask[:,3:-3].float()

        F0 = F0.float()
        F0 = self.F0_encoder(F0.unsqueeze(1))

        F0 = F0.permute(0,2,1)
        F0 = self.pos_encoder(F0)
        print('F0 = ',F0)
        x = self.transformer_encoder(F0,src_key_padding_mask=F0_mask)
        print("x = ",x)
        au1 = torch.ones((F0.shape[0],1)).to(F0.device)*au_beg
        au2 = torch.ones((F0.shape[0],1)).to(F0.device)*au_beg
        au4 = torch.ones((F0.shape[0],1)).to(F0.device)*au_beg
        i = 0
        while i<48 and au1[0,-1]!=au_end :
            i+=1
            y1 = self.transformer_decoder_au01(x, au1, None, F0_mask)
            y1 = torch.softmax(y1, dim=2)
            au1 = torch.concatenate((au1,torch.argmax(y1,dim=2)[:,-1:]),dim=1)
            y2 = self.transformer_decoder_au02(x, au2, None, F0_mask)
            y2 = torch.softmax(y2, dim=2)
            au2 = torch.concatenate((au2, torch.argmax(y2, dim=2)[:, -1:]), dim=1)
            y = self.transformer_decoder_au04(x, au4, None, F0_mask)
            y = torch.softmax(y,dim=2)
            au4 = torch.concatenate((au4, torch.argmax(y, dim=2)[:, -1:]), dim=1)


        return au1,au2,au4