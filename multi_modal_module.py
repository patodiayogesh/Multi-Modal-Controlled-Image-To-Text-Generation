from transformers import (
    ViTFeatureExtractor,
    ViTModel,
    BartTokenizer,
)
from modelling_bartMultiModal import BartMultiModalGenerationModel
import torch
import wandb
from tqdm import tqdm
import pickle
from evaluation_metrics import compute_bleu_scores, compute_bert_score, compute_rouge_score, compute_meteor_score


class MultiModalModel:

    def __init__(self,
                 model_ckpt=None,
                 beam_size=5):

        # Vit Image Extractor and Encoder and BART Decoder
        '''
        Set Modified BART Model architecture as model for text generation
        Pass Image Embeddings and Text as Input and Get Text as Output
        '''

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
            self.model_ckpt = '_'.join(model_ckpt.split('/'))
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
              train_dataloader,
              path):

        self.model.train()
        set_steps = set([10, 50, 100, 500, 1000, 5000, 10000])
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

            if batch_idx in set_steps and epoch == 0:
                self.model.save_pretrained(f'{path}_batch{batch_idx}/')

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
                filename,
                calculate_bleu_score=True,
                calculate_bert_score=False,
                calculate_meteor_score=True,
                calculate_rouge_score=True,
                experiment_setting='vqa'):

        self.model.eval()
        progress_bar = tqdm(dataloader)
        progress_bar.set_description('Inference')
        bleu_scores, bert_scores, rouge_scores, meteor_scores = [], [], [], []
        columns = self.wandb_column_names(experiment_setting)
        wandb_table = wandb.Table(columns=columns)

        model_inputs = []
        predictions = []
        targets = []
        images = []

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(progress_bar):
                image_pixel_values = batch_data[0].to(self.device)
                input_encodings = batch_data[1].to(self.device)
                input_ids = input_encodings.input_ids
                input_attention_mask = input_encodings.attention_mask
                reference_text = batch_data[2]
                image_file_name = batch_data[3]

                image_embeddings = self.image_model(image_pixel_values).last_hidden_state
                generated_ids = self.model.generate(input_ids=input_ids,
                                                    attention_mask=input_attention_mask,
                                                    image_embeddings=image_embeddings,
                                                    num_beams=self.beam_size,
                                                    max_length=24)
                generated_text = self.tokenizer.batch_decode(generated_ids,
                                                             skip_special_tokens=True,
                                                             clean_up_tokenization_spaces=False)

                input_text = self.tokenizer.batch_decode(input_ids,
                                                         skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False)
                model_inputs += input_text
                predictions += generated_text
                targets += reference_text
                images += image_file_name

                if experiment_setting == 'vqa':
                    with open(f"{filename}_input", "a") as f:
                        for txt, img in zip(input_text, image_file_name):
                            f.write(f"{txt} {img}\n")
                with open(f"{filename}_output.hyp", "a") as f:
                    for pred in generated_text:
                        f.write(f"{pred}\n")
                with open(f"{filename}_output.ref", "a") as f:
                    for target in reference_text:
                        f.write(f"{target}\n")

                # Evaluation Metrics
                if calculate_bleu_score:
                    avg_bleu_score, bleu_score_list = compute_bleu_scores(generated_text, reference_text)
                    bleu_scores += bleu_score_list
                if calculate_bert_score:
                    avg_bert_score, bert_score_list = compute_bert_score(generated_text, reference_text)
                    bert_scores += bert_score_list
                if calculate_rouge_score:
                    avg_rouge_score, rouge_score_list = compute_rouge_score(generated_text, reference_text)
                    rouge_scores += rouge_score_list
                if calculate_meteor_score:
                    avg_meteor_score, meteor_score_list = compute_meteor_score(generated_text, reference_text)
                    meteor_scores += meteor_score_list
                progress_bar.set_postfix(bleu_score=avg_bleu_score)

                # If batch size is 1, log every 10 the image else log first image in batch
                if len(image_file_name) == 1:
                    if batch_idx % 10 == 0:
                        self.update_wandb_table(wandb_table,
                                                image_file_name,
                                                input_text, generated_text, reference_text,
                                                bleu_score_list, rouge_score_list, meteor_score_list,
                                                experiment_setting)
                else:
                    self.update_wandb_table(wandb_table,
                                            image_file_name,
                                            input_text, generated_text, reference_text,
                                            bleu_score_list, rouge_score_list, meteor_score_list,
                                            experiment_setting)


        # Log and Save results after prediction
        wandb.log({'Bleu Score': round(sum(bleu_scores) / len(bleu_scores), 2),
                   # 'Bert Score': round(sum(bert_scores) / len(bert_scores), 2),
                   'Rouge Score': round(sum(rouge_scores) / len(rouge_scores), 2),
                   'Meteor Score': round(sum(meteor_scores) / len(meteor_scores), 2),
                   f'{filename} Prediction Samples': wandb_table,
                   })

        with open(f"{self.model_ckpt}_model_inputs.pkl", "wb") as save_file:
            pickle.dump(model_inputs, save_file)
        with open(f"{self.model_ckpt}_output_hyp.pkl", "wb") as save_file:
            pickle.dump(predictions, save_file)
        with open(f"{self.model_ckpt}_output_ref.pkl", "wb") as save_file:
            pickle.dump(targets, save_file)
        with open(f"{self.model_ckpt}_image_filenames.pkl", "wb") as save_file:
            pickle.dump(images, save_file)

    def wandb_column_names(self, experiment_setting):

        if experiment_setting == 'flickr30k':
            columns = ['Image',
                       'Generated Caption', 'Reference Captions',
                       'Bleu Score', 'Rouge Score', 'Meteor Score',
                       ]
        elif experiment_setting == 'vqa':
            columns = ['Image',
                       'Input Text',
                       'Generated Text', 'Reference Text',
                       'Bleu Score', 'Rouge Score', 'Meteor Score',
                       ]

        return columns

    def update_wandb_table(self, wandb_table, img_filename, input_text,
                           generated_text, reference_text,
                           bleu_scores, rouge_scores, meteor_scores,
                           experiment_setting):

        if experiment_setting == 'vqa':
            wandb_table.add_data(
                wandb.Image(f'datasets/vqa_images/val/{img_filename[0]}'),
                input_text[0],
                generated_text[0],
                reference_text[0],
                bleu_scores[0],
                # bert_score_list[0] if calculate_bert_score ,
                rouge_scores[0],
                meteor_scores[0]
            )
        else:
            wandb_table.add_data(
                wandb.Image(f'datasets/vqa_images/val/{img_filename[0]}'),
                generated_text[0],
                reference_text[0],
                bleu_scores[0],
                # bert_score_list[0] if calculate_bert_score ,
                rouge_scores[0],
                meteor_scores[0]
            )

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
