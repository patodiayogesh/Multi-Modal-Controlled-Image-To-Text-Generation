import torch
from transformers import (
    ViTFeatureExtractor,
    ViTModel
)
from PIL import Image
import pickle
from torchvision.transforms import transforms

vision_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
vision_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
vision_model.eval()


def image_features(name,d):

    images = transforms.PILToTensor()(Image.open('datasets/flickr30k_images/'+name))
    inputs = vision_feature_extractor(images, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = vision_model(
            inputs
        ).last_hidden_state
    d[name] = outputs.detach().numpy()

if __name__ == '__main__':

    data = open('datasets/flickr30k/results_20130124.token', 'r').read().splitlines()
    image_filenames = [x.split('#')[0] for x in data]
    image_filenames = list(set(image_filenames))[:10]
    d = {}
    for img in image_filenames:
        image_features(img, d)

    with open('datasets/flickr30k/image_embeddings.pickle', 'wb') as handle:
        pickle.dump(d, handle)

    # with open('datasets/flickr30k/image_embeddings.pickle', 'rb') as handle:
    #     d = pickle.load(handle)
    # print()