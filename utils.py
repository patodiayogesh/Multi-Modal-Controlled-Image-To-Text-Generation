import random
import statistics

def random_mask_for_caption_prediction(captions):

    caption_index = random.randint(0, len(captions)-1)
    caption = captions[caption_index].split(' ')
    l_caption = len(caption)
    mask_locations = random.sample(range(l_caption), random.randint(l_caption-3, l_caption-1))
    prompt = []
    for i, word in enumerate(caption):

        if i in mask_locations:
            prompt.append('<mask>')
        else:
            prompt.append(word)
    return ' '.join(prompt)

def calculate_number_of_mask_tokens(dataset, filenames):

    sentence_lengths = []
    for img in filenames:
        for caption in dataset[img]:
            sentence_lengths.append(len(caption))
    median = statistics.median(sentence_lengths)
    return median

def mask_sentence_for_prediction(captions, median=None):

    """
    Function to generate <mask> prompt for epoch aware mask training strategy
    """

    if median is None:
        return random_mask_for_caption_prediction(captions)

    caption_index = random.randint(0, len(captions) - 1)
    caption = captions[caption_index].split(' ')
    l_caption = len(caption)
    mask_locations = random.sample(range(l_caption),
                                   median if median < (l_caption-1) else l_caption-1)
    prompt = []
    for i, word in enumerate(caption):

        if i in mask_locations:
            prompt.append('<mask>')
        else:
            prompt.append(word)
    return ' '.join(prompt)

if __name__ == '__main__':

    captions = ['My name is Yogesh Patodia it is a longer statement',
                'I am currently attending the class it is a longer statement',
                'Will this work it is a longer statement',
                ]
    for _ in range(10):
        print(mask_sentence_for_prediction(captions,10))
