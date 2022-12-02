import argparse

from PIL import Image
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTModel, BartTokenizer

from multi_modal_module import MultiModalModel


def generate_caption(model_ckpt, image_location, mask):

    model = MultiModalModel(model_ckpt)

    image = image_location

    vision_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    vision_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    images = [transforms.PILToTensor()(Image.open(image))]
    image_features = vision_feature_extractor(images, return_tensors='pt').pixel_values
    img_embed = vision_model(
        image_features
    )

    # Alter Input text for different masking options to get controlled caption generation
    if mask == 'empty':
        input_text = ['A squirrel <mask>']
    elif mask == 'epoch_aware_mask':
        input_text = ['A squirrel <mask> nuts <mask> <mask>']
    else:
        input_text = ['<mask> gun <mask>']
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    input_ids = tokenizer(
        input_text,
        padding='longest',
        return_tensors='pt').input_ids
    o = model.model.generate(input_ids=input_ids,
                             image_embeddings=img_embed.last_hidden_state,
                             num_beams=5,
                             max_length=32
                             )
    text = tokenizer.batch_decode(o,
                                  skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)
    print(text)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--multi_modal', type=bool, default=True)
    parser.add_argument('--mask', type=str, default='empty', choices=['empty', 'epoch_aware_mask'])
    parser.add_argument('--model_ckpt', type=str, required=True)
    parser.add_argument('--image_location', type=str, required=True)

    args = parser.parse_args()
    generate_caption(args.model_ckpt, args.image_location, args.mask)