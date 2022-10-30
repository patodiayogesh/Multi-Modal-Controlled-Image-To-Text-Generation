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
        self.decoder_tokenizer = BartTokenizer.from_pretrained(text_decoder)
        self.image_feature_extractor = ViTFeatureExtractor.from_pretrained(image_encoder)

        # Model Initialization
        if model_ckpt is None:
            self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                image_encoder,
                text_decoder)
            self._set_bart_decoder(text_decoder)
        else:
            self.model = VisionEncoderDecoderModel.from_pretrained(model_ckpt)
        self.model.to(self.device)

        # Hyperparameters
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=0.0001,
                                           eps=1e-8,
                                           weight_decay=0.01
                                           )
        self.beam_size = beam_size

        # Wandb
        self.log_freq = 10
        wandb.init(project='multi-modal-image-caption')
        wandb.watch(self.model, self.log_freq)

    def _set_bart_decoder(self):

        model_config = self.model.config
        model_config.pad_token_id = self.decoder_tokenizer.pad_token_id
        model_config.decoder_start_token_id = self.decoder_tokenizer.bos_token_id
        model_config.eos_token_id = self.decoder_tokenizer.eos_token_id
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

        return total_loss / (batch_idx+1)

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

        return total_loss / (batch_idx+1)

    def predict(self,
                dataloader):

        self.model.eval()
        progress_bar = tqdm(dataloader)
        progress_bar.set_description('Inference')
        bleu_scores = []
        for batch_data in progress_bar:
            image_pixel_values = batch_data[0].to(self.device)
            label_input_ids = batch_data[1].input_ids.to(self.device)
            generated_ids = self.model.generate(image_pixel_values)
            pred_sequences = self.decoder_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            target_sequences = self.decoder_tokenizer.batch_decode(label_input_ids, skip_special_tokens=True)
            avg_bleu_score, bleu_score_list = compute_bleu_scores(pred_sequences, target_sequences)
            bleu_scores += bleu_score_list
            progress_bar.set_postfix(bleu_score=avg_bleu_score)
            with open("output.hyp", "a") as f:
                for pred in pred_sequences:
                    f.write(f"{pred}\n")
            with open("output.ref", "a") as f:
                for target in target_sequences:
                    f.write(f"{target}\n")
        self.log({'Bleu Score': sum(bleu_scores)*100/len(bleu_scores)})

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
