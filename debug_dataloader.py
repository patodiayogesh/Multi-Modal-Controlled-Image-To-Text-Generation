from PIL import Image
from transformers import GPT2TokenizerFast, ViTFeatureExtractor, VisionEncoderDecoderModel
from dataset import FlickrDatasetModule

if __name__ == '__main__':
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    dataset = FlickrDatasetModule()
    dataset.image_feature_extractor = feature_extractor
    dataset.decoder_tokenizer = tokenizer
    val_dataloader = dataset.val_dataloader()
    for x,y in val_dataloader:
        generated_ids = model.generate(x)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print()

