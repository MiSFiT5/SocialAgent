import numpy as np
import pandas as pd
import seaborn as sns

AU_quantization_bit = 30
F0_quantization_bit = 600
au_pad = 0
au_beg = 1
au_end = 33
f0_pad = -1

min_clip_au1 = 0
max_clip_au1 = 4
min_clip_au2 = 0
max_clip_au2 = 2.5
min_clip_au4 = 0
max_clip_au4 = 2.5
min_clip_au5 = 0
max_clip_au5 = 2
min_clip_au6 = 0
max_clip_au6 = 2
min_clip_au7 = 0
max_clip_au7 = 3
min_clip_f0 = 50
max_clip_f0 = 400

def clip(X,clip_min,clip_max):
    for i,xs in enumerate(X) :
        for j,x in enumerate(xs) :
            if x >= 0 :
                X[i][j] = min(max(clip_min,x),clip_max)

def normalize(X,min,max):
    for i,xs in enumerate(X) :
        for j,x in enumerate(xs) :
            if x > au_pad :
                X[i][j] = (x-min)/(max-min)

def original_value(X,quantize_bit,min,max):
    Y = X.copy()
    for i,xs in enumerate(X) :
        for j,x in enumerate(xs) :
            if x >= 0 :
                Y[i][j] = x/quantize_bit
                Y[i][j] = Y[i][j]*(max-min)+min
    return Y

def replace_AU(AU,au_pad,au_beg,au_end,new_au_pad,new_au_end,new_au_beg):
    for i,aus in enumerate(AU) :
        for j,au in enumerate(aus) :
            if au == au_pad :
                AU[i][j] = new_au_pad
            elif au == au_end :
                AU[i][j] = new_au_end
            elif au == au_beg :
                AU[i][j] = new_au_beg
            else :
                AU[i][j] = AU[i][j]

# quantize AU with
def quantize(X,quantize_bit):
    for i,xs in enumerate(X) :
        for j,x in enumerate(xs) :
            if x >= 0 :
                X[i][j] = round(x*quantize_bit)

def plot_aus(AUs):
    A = np.transpose(AUs.copy(),(1,0,2))
    flatten_AUs = []
    for AU in A :
        replace_AU(AU,au_pad,au_beg,au_end,-1,-2,-3)
        flatten_AUs.append(np.delete(AU.flatten(), np.where(AU.flatten()<0)))
    data = pd.DataFrame({'AU01':flatten_AUs[0],'AU02':flatten_AUs[1],'AU04':flatten_AUs[2],'AU05':flatten_AUs[3],'AU06':flatten_AUs[4],'AU07':flatten_AUs[5]})
    # Facet grid of histplot
    g = sns.FacetGrid(data.melt(), col="variable",hue="variable", col_wrap=3, sharex=False, sharey=False).map(sns.histplot, "value")