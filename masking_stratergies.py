import numpy as np

def gaussian_mask(caption):
    mean = 0.5
    std = 0.2
    val = np.random.normal(mean,std, size=(1,))
    caption_split = caption.split(' ')
    len_to_mask = int(min(1,max(0,val)) * len(caption_split))
    masks = np.random.randint(len(caption_split), size=len_to_mask)
    for m in masks:
        caption_split[m] = "<mask>" 
    output = " ".join(caption_split)
    return output

def epoch_aware_mask(epoch,max_epochs,caption):
    mean = 1 - epoch/max_epochs
    std = 0.25
    val = np.random.normal(mean,std, size=(1,))
    caption_split = caption.split(' ')
    len_to_mask = int(min(1,max(0,val)) * len(caption_split))
    masks = np.random.randint(len(caption_split), size=len_to_mask)
    for m in masks:
        caption_split[m] = "<mask>" 
    output = " ".join(caption_split)
    return output