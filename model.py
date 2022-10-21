import torch
from torch import nn
from transformers import VisionEncoderDecoderModel
from transformers import BeitModel, BeitFeatureExtractor
from transformers import BertTokenizer, BertModel

class BaselineModel(nn.Module):

    def __init__(self,
                 image_encoder,
                 text_decoder,
                 ):

        super().__init__()

        if torch.cuda.is_available():
            self.device = torch.device('gpu')
        else:
            self.device = torch.device('cpu')

        if image_encoder.lower() == 'beit':
            image_feature_extractor = BeitFeatureExtractor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
            image_encoder_path = "microsoft/beit-base-patch16-224-pt22k"

        if text_decoder.lower() == 'bert':
            decoder_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            text_decoder_path = "bert-base-uncased"

        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            image_encoder_path,
            text_decoder_path,
        )
        self.image_feature_extractor = image_feature_extractor
        self.decoder_tokenizer = decoder_tokenizer

        self.model.config.decoder_start_token_id = self.decoder_tokenizer.cls_token_id
        self.model.config.pad_token_id = self.decoder_tokenizer.pad_token_id
