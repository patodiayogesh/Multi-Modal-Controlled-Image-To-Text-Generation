from PIL import Image
from transformers import GPT2TokenizerFast, ViTFeatureExtractor, VisionEncoderDecoderModel
from dataset import FlickrDatasetModule
import torch


if __name__ == '__main__':

    device = 'cpu'
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning",
                                                      )
    # feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # model.to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    dataset = FlickrDatasetModule()
    dataset.image_feature_extractor = feature_extractor
    dataset.decoder_tokenizer = tokenizer
    train_dataloader = dataset.train_dataloader()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, eps=1e-8, weight_decay=0.01)

    steps = 0
    model.train()
    for x,y in train_dataloader:
        # generated_ids = model(x)
        # generated_text = tokeniz
        # er.batch_decode(generated_ids, skip_special_tokens=True)
        # labels = tokenizer.batch_decode(y.input_ids, skip_special_tokens=True)
        # model_output=model(x)
        # x.to(device)
        # y.to(device)
        outputs = model(x,
                        labels=y.input_ids,
                        return_dict = True)
        loss = outputs.loss
        print(f"Steps: {steps}, Loss: {loss.item()}")

        model.zero_grad()
        loss.backward()
        optimizer.step()
        steps+=1
        if steps%10==0:
            generated_ids = model.generate(x)
            print(tokenizer.batch_decode(generated_ids,skip_special_tokens=True))

