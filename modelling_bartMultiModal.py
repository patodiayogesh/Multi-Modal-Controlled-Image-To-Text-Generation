from typing import Optional, List, Union, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqModelOutput, BaseModelOutput, Seq2SeqLMOutput
from transformers.models.bart.modeling_bart import (
    BartEncoder, BartDecoder,
    BartModel, BartForConditionalGeneration,
    BartConfig,
    shift_tokens_right
)


class BartMultiModalEncoder(BartEncoder):

    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens=None)
        self.config = config
        # self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=True,
                image_embeddings=None,
                ):
        # Image Embeddings are not passed to BartEncoder
        return super().forward(
            input_ids,
            attention_mask,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict
        )


class BartMultiModalModel(BartModel):

    def __init__(self, config: BartConfig):

        super(BartModel, self).__init__(config)
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        # ----- Modified-----#
        self.encoder = BartMultiModalEncoder(config, self.shared)
        # -------------------#
        self.decoder = BartDecoder(config, self.shared)

    def get_encoder(self):

        return self.encoder

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[List[torch.FloatTensor]] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            image_embeddings=None,
            **kwargs
    ) -> Union[Tuple, Seq2SeqModelOutput]:

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                image_embeddings=image_embeddings,
            )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        '''
        Beam Search creates (num_beams*enocder_input_ids) and passes to encoder which gives
        (num_beams*encoder_embeddings). 
        Multiply (num_beams*Image Embeddings) and attach them to encoder_embeddings and get
        new Image+Text embeddings.
        '''
        beam_batch_size = encoder_outputs.last_hidden_state.size()[0]
        if image_embeddings.size()[0] != beam_batch_size:
            image_embeddings = image_embeddings.repeat(beam_batch_size // image_embeddings.size()[0], 1, 1)

        image_text_embeddings = torch.concat((image_embeddings,
                                              encoder_outputs.last_hidden_state),
                                             axis=1)
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        if attention_mask is None:
            extended_attention_mask = torch.ones((image_text_embeddings.size()[0],
                                                  image_text_embeddings.size()[1])).to(device)
        else:
            image_attention_mask = torch.ones((image_embeddings.size()[0],
                                               image_embeddings.size()[1])).to(device)
            extended_attention_mask = torch.concat((image_attention_mask,
                                                    attention_mask),
                                                   axis=1)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=image_text_embeddings,
            encoder_attention_mask=extended_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class BartMultiModalGenerationModel(BartForConditionalGeneration):

    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: BartConfig):

        super(BartForConditionalGeneration, self).__init__(config)
        self.model = BartMultiModalModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

    def get_encoder(self):
        return self.model.get_encoder()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[List[torch.FloatTensor]] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            image_embeddings=None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                print("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_embeddings=image_embeddings,
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
            self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        output = {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

        # Modified
        if "image_embeddings" in kwargs:
            output["image_embeddings"] = kwargs['image_embeddings']

        return output


if __name__ == '__main__':
    from transformers import (
        ViTFeatureExtractor,
        ViTModel
    )
    from PIL import Image

    vision_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    vision_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    images = [Image.open('datasets/flickr30k_images/36979.jpg')]
    image_inputs = vision_feature_extractor(images, return_tensors='pt').pixel_values
    img_embed = vision_model(
        image_inputs
    )

    bartConfig = BartConfig()
    model = BartMultiModalGenerationModel.from_pretrained('facebook/bart-base')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    input_ids = tokenizer(
        ['<mask>'],
        padding='longest',
        return_tensors='pt').input_ids
    o = model(input_ids=input_ids,
              image_embeddings = img_embed.last_hidden_state,
              labels=input_ids)
    o = model.generate(input_ids=input_ids,
                       image_embeddings=img_embed.last_hidden_state,
                       num_beams=6
                       )
    text = tokenizer.batch_decode(o,
                                  skip_special_tokens=True)
    print(text)
