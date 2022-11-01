decoder_outputs = self.decoder(
    input_ids=decoder_input_ids,
    attention_mask=decoder_attention_mask,
    encoder_hidden_states=encoder_outputs[0],
    encoder_attention_mask=attention_mask,
    head_mask=decoder_head_mask,
    cross_attn_head_mask=cross_attn_head_mask,
    past_key_values=past_key_values,
    inputs_embeds=decoder_inputs_embeds,
    use_cache=use_cache,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
    return_dict=return_dict,
)

#TO

decoder_outputs = self.decoder(
    input_ids=decoder_input_ids,
    attention_mask=decoder_attention_mask,
    encoder_hidden_states=encoder_outputs[0], # Concat the image ViT output here
    encoder_attention_mask=attention_mask, # Concat all ones of size ViT input here
    head_mask=decoder_head_mask,
    cross_attn_head_mask=cross_attn_head_mask,
    past_key_values=past_key_values,
    inputs_embeds=decoder_inputs_embeds,
    use_cache=use_cache,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
    return_dict=return_dict,
)
