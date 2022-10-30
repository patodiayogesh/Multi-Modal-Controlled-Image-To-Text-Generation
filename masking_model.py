import torch
from torch import nn 
from transformers import ViTConfig, BertConfig, VisionTextDualEncoderConfig, VisionTextDualEncoderModel, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from transformers import BartTokenizer, BartModel
from transformers.modeling_outputs import Seq2SeqModelOutput
from mask_data import mask

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class MaskingModel(nn.Module):
  def __init__(self):
    self.encoder_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    super(MaskingModel, self).__init__()
    self.model = BartModel.from_pretrained("facebook/bart-base")

    self.encoder_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    self.encoder = self.model.encoder
    self.decoder = self.model.decoder

  def forward(self,input,output):
      # input is the ViT encoded image output
      masked = mask(output)
      tokenized = self.encoder_tokenizer(masked,return_tensors = "pt")
      encoded_outputs = self.encoder(**tokenized,return_dict=True)
  
      outputs_concantentated = torch.concat((input,encoded_outputs.last_hidden_state),axis=1)
      attention_mask = torch.ones((outputs_concantentated.size()[0],outputs_concantentated.size()[1]))

      print(encoded_outputs.last_hidden_state.shape)
      print(input.shape)
      print(outputs_concantentated.shape)
      print(attention_mask.shape)
      decoder_input_ids = shift_tokens_right(
          tokenized['input_ids'], self.model.config.pad_token_id, self.model.config.decoder_start_token_id
      )
      
      decoder_outputs = self.decoder(
          input_ids=decoder_input_ids,
          encoder_hidden_states=outputs_concantentated,
          encoder_attention_mask=attention_mask,
          return_dict=True
      )
      return Seq2SeqModelOutput(
          last_hidden_state=decoder_outputs.last_hidden_state,
          past_key_values=decoder_outputs.past_key_values,
          decoder_hidden_states=decoder_outputs.hidden_states,
          decoder_attentions=decoder_outputs.attentions,
          cross_attentions=decoder_outputs.cross_attentions,
          encoder_last_hidden_state=encoded_outputs.last_hidden_state,
          encoder_hidden_states=encoded_outputs.hidden_states,
          encoder_attentions=encoded_outputs.attentions,
      )