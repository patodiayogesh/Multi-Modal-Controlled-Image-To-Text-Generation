from transformers import (
    ViTFeatureExtractor,
    VisionEncoderDecoderModel,
    BartTokenizer,
)
import torch
import wandb
from tqdm import tqdm
from evaluate import compute_bleu_scores


class BaselineModel:

    def __init__(self,
                 model_ckpt=None,
                 beam_size=5):

        # Vit Image Extractor and Encoder and BART Decoder

        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        image_encoder = "google/vit-base-patch16-224-in21k"
        text_decoder = "facebook/bart-base"

        # Image and Text Tokenizers
        self.tokenizer = BartTokenizer.from_pretrained(text_decoder)
        self.image_feature_extractor = ViTFeatureExtractor.from_pretrained(image_encoder)

        # Model Initialization
        if model_ckpt is None:
            self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                image_encoder,
                text_decoder)
            self._set_bart_decoder()
        else:
            self.model = VisionEncoderDecoderModel.from_pretrained(model_ckpt)
        self.model.to(self.device)

        # Hyperparameters
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=0.0001,
                                           eps=1e-8,
                                           weight_decay=0.01
                                           )
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=0.9)
        self.beam_size = beam_size

        # Wandb
        self.log_freq = 10
        wandb.init(project='multi-modal-image-caption')
        wandb.watch(self.model, self.log_freq)

    def _set_bart_decoder(self):

        model_config = self.model.config
        model_config.pad_token_id = self.tokenizer.pad_token_id
        model_config.decoder_start_token_id = self.tokenizer.bos_token_id
        model_config.eos_token_id = self.tokenizer.eos_token_id
        model_config.vocab_size = model_config.decoder.vocab_size

    def train(self,
              epoch,
              train_dataloader):

        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_dataloader)
        for batch_idx, batch_data in enumerate(progress_bar):
            progress_bar.set_description(f'Train Epoch {epoch}')
            image_pixel_values = batch_data[0].to(self.device)
            label_encodings = batch_data[1]
            label_input_ids = label_encodings.input_ids.to(self.device)
            outputs = self.model(
                pixel_values=image_pixel_values,
                labels=label_input_ids,
                return_dict=True,
            )
            loss = outputs.loss
            progress_bar.set_postfix(loss=loss.item())
            total_loss += loss.item()
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_idx % self.log_freq == 0:
                wandb.log({"train/loss": loss.item()})

        return total_loss / (batch_idx + 1)

    def test(self,
             epoch,
             dataloader,
             validation=True):

        self.model.eval()
        loss_name = 'val/loss' if validation else 'test/loss'
        step = 'Val' if validation else 'Test'
        total_loss = 0.0
        progress_bar = tqdm(dataloader)
        for batch_idx, batch_data in enumerate(progress_bar):
            progress_bar.set_description(f'{step} Epoch {epoch}')
            image_pixel_values = batch_data[0].to(self.device)
            label_encodings = batch_data[1]
            label_input_ids = label_encodings.input_ids.to(self.device)
            outputs = self.model(
                pixel_values=image_pixel_values,
                labels=label_input_ids,
                return_dict=True,
            )
            loss = outputs.loss.item()
            progress_bar.set_postfix(loss=loss)
            total_loss += loss
            if batch_idx % self.log_freq == 0:
                wandb.log({loss_name: loss})

        return total_loss / (batch_idx + 1)

    def predict(self,
                dataloader,
                filename):

        self.model.eval()
        progress_bar = tqdm(dataloader)
        progress_bar.set_description('Inference')
        bleu_scores = []
        columns = ['Image', 'Generated Caption', 'Reference Captions', 'Bleu Score']
        wandb_table = wandb.Table(columns=columns)

        for batch_idx, batch_data in enumerate(progress_bar):
            image_pixel_values = batch_data[0].to(self.device)
            reference_captions = batch_data[1]
            image_file_name = batch_data[2]

            generated_ids = self.model.generate(image_pixel_values,
                                                num_beams=self.beam_size,
                                                max_length=24)
            generated_captions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            avg_bleu_score, bleu_score_list = compute_bleu_scores(generated_captions, reference_captions)
            bleu_scores += bleu_score_list
            progress_bar.set_postfix(bleu_score=avg_bleu_score)

            with open(f"{filename}_output.hyp", "a") as f:
                for pred in generated_captions:
                    f.write(f"{pred}\n")
            with open(f"{filename}_output.ref", "a") as f:
                for target in reference_captions:
                    f.write(f"{target}\n")

            if batch_idx % 10 == 0:
                wandb_table.add_data(
                    wandb.Image(f'datasets/flickr30k_images/{image_file_name[0]}'),
                    generated_captions[0],
                    reference_captions[0],
                    bleu_score_list[0]
                )

        wandb.log({'Bleu Score': sum(bleu_scores) / len(bleu_scores),
                   f'{filename} Prediction Samples': wandb_table,
                   f'{filename} Scores Plot': wandb.plot.histogram(
                       wandb.Table(data=[[s] for s in bleu_scores],
                                   columns=['bleu score'])
                   )
                   })


    def save_pretrained(self, path):
        self.model.save_pretrained(path)
