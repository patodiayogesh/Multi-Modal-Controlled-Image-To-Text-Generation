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

def epoch_aware_mask(epoch,caption,max_epochs=20):
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

def text_infilling(caption):
    caption_split = caption.split(' ')
    if len(caption_split) <= 4:
      return "<mask>"
    num_masks = np.random.randint(max(1,len(caption_split)//5),len(caption_split)//3)

    masks_pos = set()
    i=0
    while i<num_masks:
        x = np.random.randint(0,len(caption_split))
        if x not in masks_pos:
            masks_pos.add(x)
            i+=1
    sorted_masks_pos = sorted(list(masks_pos))
    output = []
    if sorted_masks_pos[0]!=0:
        output.append("<mask>")
    for pos in sorted_masks_pos:
        output.append(caption_split[pos])
        output.append("<mask>")

    if sorted_masks_pos[-1]!=len(caption_split)-1:
        output=output[:-1]
        
    output = " ".join(output)
    return output