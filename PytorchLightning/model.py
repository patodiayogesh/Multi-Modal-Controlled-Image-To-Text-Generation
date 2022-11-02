import torch
from torch import nn
from transformers import VisionEncoderDecoderModel
from transformers import BeitFeatureExtractor, ViTFeatureExtractor
from transformers import BertTokenizer, GPT2Tokenizer, GPT2TokenizerFast
import pytorch_lightning as pl
from evaluate import compute_bleu_scores


class BaselineModel(pl.LightningModule):

    def __init__(self,
                 image_encoder,
                 text_decoder,
                 freeze_image_encoder=False,
                 beam_size=5,
                 model_ckpt=None,
                 ):

        super(BaselineModel, self).__init__()

        image_encoder = image_encoder.lower()
        text_decoder = text_decoder.lower()

        # if image_encoder == 'beit':
        #     image_feature_extractor = BeitFeatureExtractor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        #     image_encoder_path = "microsoft/beit-base-patch16-224-pt22k"
        # if image_encoder == 'vit':
        #     image_feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        #     image_encoder_path = 'google/vit-base-patch16-224-in21k'
        #
        # if text_decoder == 'bert':
        #     decoder_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        #     text_decoder_path = "bert-base-uncased"
        # if text_decoder == 'gpt2':
        #     decoder_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        #     text_decoder_path = "gpt2"
        #
        # self.image_feature_extractor = image_feature_extractor
        # self.decoder_tokenizer = decoder_tokenizer
        # self.beam_size = beam_size
        #
        # if model_ckpt is None or model_ckpt == '':
        #     self._define_model( image_encoder_path,
        #                         text_decoder_path,
        #                         text_decoder,
        #                         freeze_image_encoder,
        #                         )
        # else:
        #     self.model = VisionEncoderDecoderModel.from_pretrained(model_ckpt)
        #     if text_decoder == 'gpt2':
        #         self.decoder_tokenizer.pad_token_id = self.decoder_tokenizer.eos_token_id
        #         self.end_token = self.decoder_tokenizer.eos_token
        #         self.start_token = self.decoder_tokenizer.bos_token
        #     elif text_decoder == 'bert':
        #         self.end_token = self.decoder_tokenizer.pad_token
        #         self.start_token = self.decoder_tokenizer.cls_token

        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.decoder_tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.image_feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.start_token = None
        self.end_token =None
        self.beam_size = beam_size

        self.save_hyperparameters()

    def _define_model(self,
                      image_encoder_path,
                      text_decoder_path,
                      text_decoder,
                      freeze_image_encoder,
                      ):

        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            image_encoder_path,
            text_decoder_path,
            tie_encoder_decoder=True
        )

        config_decoder = self.model.config.decoder
        # config_decoder.is_decoder = True
        # config_decoder.add_cross_attention = True
        self.model.config.vocab_size = config_decoder.vocab_size

        if text_decoder == 'gpt2':
            self.decoder_tokenizer.pad_token_id = self.decoder_tokenizer.eos_token_id
            self.model.config.pad_token_id = self.decoder_tokenizer.eos_token_id
            self.model.config.decoder_start_token_id = self.decoder_tokenizer.bos_token_id

            self.end_token = self.decoder_tokenizer.eos_token
            self.start_token = self.decoder_tokenizer.bos_token
        elif text_decoder == 'bert':
            self.model.config.pad_token_id = self.decoder_tokenizer.pad_token_id
            self.model.config.decoder_start_token_id = self.decoder_tokenizer.cls_token_id

            self.end_token = self.decoder_tokenizer.pad_token
            self.start_token = self.decoder_tokenizer.cls_token

        self.model.config.eos_token_id = self.decoder_tokenizer.pad_token_id

        if freeze_image_encoder:
            for param in self.model.encoder.base_model.parameters():
                param.requires_grad = False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):

        one_cycle_lr_config = {
            'max_lr': 0.0001,
            'pct_start': 0.1,
            'div_factor': 1,
            'total_steps': 50,
            'anneal_strategy': 'linear'
        }
        exponential_lr_config = {
            'gamma': 0.5
        }
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001, eps=1e-8, weight_decay=0.01)
        #lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **lr_config)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **exponential_lr_config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    def training_step(self, batch, batch_idx):

        inputs, labels = batch
        labels_input_ids = labels.input_ids
        labels_attention_mask = labels.attention_mask
        outputs = self.model(
            inputs,
            # decoder_input_ids=labels_input_ids,
            # decoder_attention_mask=labels_attention_mask,
            labels=labels_input_ids,
            return_dict=True
        )
        train_loss = outputs.loss
        self.log_dict({"loss/train": train_loss.item()}, on_step=True)
        return train_loss

    def validation_step(self, batch, batch_idx):

        inputs, labels = batch
        labels_input_ids = labels.input_ids
        labels_attention_mask = labels.attention_mask
        outputs = self.model(
            inputs,
            # decoder_input_ids=labels_input_ids,
            # decoder_attention_mask=labels_attention_mask,
            labels=labels_input_ids,
            return_dict=True
        )
        val_loss = outputs.loss

        # output_sequences = self.model.generate(inputs,
        #                                        max_length=512,
        #                                        num_beams=self.beam_size,
        #                                        num_return_sequences=1
        #                                        )
        # output_sequences, target_seq = self.detokenize(output_sequences), self.detokenize(labels_input_ids)
        # _, bleu_scores = compute_bleu_scores(output_sequences, target_seq)
        #
        # s = ''
        # for i, _ in enumerate(output_sequences):
        #     s += f"# Example {i}\n\n"
        #     s += f"- gold\n```\n{target_seq[i]}\n```\n\n"
        #     s += f"- pred\n```\n{output_sequences[i]}\n```\n\n"
        #     # s += f"- metrics\n\n"
        #     # s += f"Bleu score: {bleu_scores[i]}\n"
        #     s += "\n"
        # self.logger.experiment.add_text("examples/val", s, global_step=self.global_step)
        self.log_dict({"loss/val": val_loss.item()}, on_step=True)
        return val_loss

    def detokenize(self, sequences):

        pred = []
        for seq in sequences:
            pred.append(self.decoder_tokenizer.decode(seq, skip_special_tokens=True))
        return pred

    def predict_step(self, batch, batch_idx=-1, dataloader_idx=None):

        inputs, labels = batch
        labels_input_ids = labels.input_ids
        labels_attention_mask = labels.attention_mask
        output_sequences = self.model.generate(inputs,
                                               max_length=12,
                                               num_beams=self.beam_size,
                                               num_return_sequences=1
                                               )
        pred_sequences = self.detokenize(output_sequences)
        target_sequences = self.detokenize(labels_input_ids)
        _, bleu_score_list = compute_bleu_scores(pred_sequences, target_sequences)
        with open("output.hyp", "a") as f:
            for pred in pred_sequences:
                f.write(f"{pred}\n")
        with open("output.ref", "a") as f:
            for target in target_sequences:
                f.write(f"{target}\n")

    def on_save_checkpoint(self, checkpoint):

        version_number = self.trainer.logger.version
        epoch_number = self.trainer.current_epoch
        path = f'lightning_logs/version_{version_number}/checkpoints/epoch_{epoch_number}/'
        self.model.save_pretrained(path)