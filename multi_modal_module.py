from transformers import (
    ViTFeatureExtractor,
    ViTModel,
    BartTokenizer,
)
from modelling_bartMultiModal import BartMultiModalGenerationModel
import torch
import wandb
from tqdm import tqdm
from evaluation_metrics import compute_bleu_scores, compute_bert_score


class MultiModalModel:

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

        self.image_model = ViTModel.from_pretrained(image_encoder)
        self.image_model.to(self.device)
        self.image_model.eval()

        # Model Initialization
        if model_ckpt is None:
            self.model = BartMultiModalGenerationModel.from_pretrained(text_decoder)
        else:
            self.model = BartMultiModalGenerationModel.from_pretrained(model_ckpt)
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
        self.log_freq = 40
        wandb.init(project='multi-modal-image-caption-generation',
                   entity='multi-modal-image-caption-generation')
        wandb.watch(self.model, self.log_freq)

    def train(self,
              epoch,
              train_dataloader):

        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_dataloader)
        for batch_idx, batch_data in enumerate(progress_bar):
            progress_bar.set_description(f'Train Epoch {epoch}')
            image_pixel_values = batch_data[0].to(self.device)
            input_encodings = batch_data[1].to(self.device)
            input_ids = input_encodings.input_ids
            input_attention_mask = input_encodings.attention_mask
            label_input_ids = batch_data[2].input_ids.to(self.device)

            image_embeddings = self.image_model(image_pixel_values).last_hidden_state
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=input_attention_mask,
                image_embeddings=image_embeddings,
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
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(progress_bar):
                progress_bar.set_description(f'{step} Epoch {epoch}')
                image_pixel_values = batch_data[0].to(self.device)
                input_encodings = batch_data[1].to(self.device)
                input_ids = input_encodings.input_ids
                input_attention_mask = input_encodings.attention_mask
                label_input_ids = batch_data[2].input_ids.to(self.device)

                image_embeddings = self.image_model(image_pixel_values).last_hidden_state
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=input_attention_mask,
                    image_embeddings=image_embeddings,
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
        bleu_scores, bert_scores = [], []
        columns = ['Image', 'Generated Caption', 'Reference Captions', 'Bleu Score', 'Bert Score']
        wandb_table = wandb.Table(columns=columns)
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(progress_bar):
                image_pixel_values = batch_data[0].to(self.device)
                input_encodings = batch_data[1].to(self.device)
                input_ids = input_encodings.input_ids
                input_attention_mask = input_encodings.attention_mask
                reference_captions = batch_data[2]
                image_file_name = batch_data[3]

                image_embeddings = self.image_model(image_pixel_values).last_hidden_state
                generated_ids = self.model.generate(input_ids=input_ids,
                                                    attention_mask=input_attention_mask,
                                                    image_embeddings=image_embeddings,
                                                    num_beams=self.beam_size,
                                                    max_length=24)
                generated_captions = self.tokenizer.batch_decode(generated_ids,
                                                                 skip_special_tokens=True,
                                                                 clean_up_tokenization_spaces=False)
                avg_bleu_score, bleu_score_list = compute_bleu_scores(generated_captions, reference_captions)
                bleu_scores += bleu_score_list
                avg_bert_score, bert_score_list = compute_bert_score(generated_captions, reference_captions)
                bert_scores += bert_score_list
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
                        bleu_score_list[0],
                        bert_score_list[0]
                    )

        wandb.log({'Bleu Score': sum(bleu_scores) / len(bleu_scores),
                   'Bert Score': sum(bert_scores) / len(bert_scores),
                   f'{filename} Prediction Samples': wandb_table,
                   })
        with open(f"{filename} Bleu scores", "w") as f:
            for s in bleu_scores:
                f.write(str(s)+"\n")
        with open(f"{filename} Bert scores", "w") as f:
            for s in bert_scores:
                f.write(str(s)+"\n")

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
