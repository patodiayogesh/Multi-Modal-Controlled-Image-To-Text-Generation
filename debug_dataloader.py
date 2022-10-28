from PIL import Image
from transformers import GPT2TokenizerFast, ViTFeatureExtractor, VisionEncoderDecoderModel, GPT2Tokenizer
from dataset import FlickrDatasetModule
import torch


def _define_model(model,
                  tokenizer
                  ):

    config_decoder = model.config.decoder
    # config_decoder.is_decoder = True
    # config_decoder.add_cross_attention = True
    model.config.vocab_size = config_decoder.vocab_size

    # if text_decoder == 'gpt2':
    #     self.decoder_tokenizer.pad_token_id = self.decoder_tokenizer.eos_token_id
    #     self.model.config.pad_token_id = self.decoder_tokenizer.eos_token_id
    #     self.model.config.decoder_start_token_id = self.decoder_tokenizer.bos_token_id
    #
    #     self.end_token = self.decoder_tokenizer.eos_token
    #     self.start_token = self.decoder_tokenizer.bos_token
    # elif text_decoder == 'bert':
    #     self.model.config.pad_token_id = self.decoder_tokenizer.pad_token_id
    #     self.model.config.decoder_start_token_id = self.decoder_tokenizer.cls_token_id
    #
    #     self.end_token = self.decoder_tokenizer.pad_token
    #     self.start_token = self.decoder_tokenizer.cls_token

    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.pad_token_id


if __name__ == '__main__':

    device = 'cuda:0'

    # model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning",
    #                                                   )
    # tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    # feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        "google/vit-base-patch16-224-in21k",
        "gpt2")
    _define_model(model, tokenizer)

    dataset = FlickrDatasetModule()
    dataset.image_feature_extractor = feature_extractor
    dataset.decoder_tokenizer = tokenizer
    train_dataloader = dataset.train_dataloader()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, eps=1e-8, weight_decay=0.01)

    steps = 0
    model.train()
    for x,y in train_dataloader:
        # generated_ids = model(x)
        # generated_text = tokeniz
        # er.batch_decode(generated_ids, skip_special_tokens=True)
        # labels = tokenizer.batch_decode(y.input_ids, skip_special_tokens=True)
        # model_output=model(x)
        x = x.to(device)
        input_ids = y.input_ids
        input_ids = input_ids.to(device)
        outputs = model(x,
                        labels=input_ids,
                        return_dict=True)
        loss = outputs.loss
        print(f"Steps: {steps}, Loss: {loss.item()}")

        model.zero_grad()
        loss.backward()
        optimizer.step()
        steps += 1
        if steps % 10 == 0:
            generated_ids = model.generate(x)
            print(tokenizer.batch_decode(generated_ids,skip_special_tokens=True))
        if steps % 20 == 0:
            model.save_pretrained('vit-gpt2-scratch')
    model.save_pretrained('vit-gpt2-scratch')

