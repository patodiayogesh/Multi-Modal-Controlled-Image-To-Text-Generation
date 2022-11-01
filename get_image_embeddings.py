from transformers import (
    ViTFeatureExtractor,
    ViTModel
)
from PIL import Image

vision_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
vision_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')


def image_features(name,df):
    images = Image.open(name)
    inputs = vision_feature_extractor(images, return_tensors="pt")

    outputs = vision_model(
        **inputs
    )
    new_row = {'name': name, 'features':outputs}
    df = df.append(new_row, ignore_index = True)
    return df