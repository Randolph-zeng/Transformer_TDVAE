import torch
import logging
import torch.nn as nn
from fengshen.models.tdvae.tdvae_utils import word_drop
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel, GPT2Block, GPT2Model, ACT2FN


logger = logging.getLogger(__name__)

class GPT2LatentDecoderModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config, latent_size=32, nclasses=0):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # VAE addition 
        self.split_num = config.n_head
        try:
            self.latent_size = config.latent_size
        except: 
            self.latent_size = latent_size # default size is 32

        self.linear_emb = nn.Linear(self.latent_size, config.hidden_size, bias=False) # share the same latent vector as the embeddings

        if nclasses > 0:# used for label-conditioned-VAE
            self.nclasses = nclasses
            self.linear_class = nn.Linear(nclasses, config.hidden_size * config.n_layer, bias=False)
            self.linear_class_emb = nn.Linear(nclasses, config.hidden_size, bias=False) 

        # Initialize weights and apply final processing
        self.post_init()

    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        class_label=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0 
            past_key_values = tuple([None] * len(self.h))
            latent_hint = None
        else:
            # NOTE: will be added into every token in every layer, thus no need to set position ids for it 
            # B, ZH -> B, H 
            latent_hint = self.linear_emb(past_key_values) 
            past_length = 0 
            past_key_values = tuple([None] * len(self.h))
            

        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        # NOTE add latent_hint to every token in every layer without any drop 
        if latent_hint is not None:
            # B, L, H + B, 1, H 
            hidden_states = hidden_states + latent_hint.unsqueeze(1)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            # NOTE add latent_hint to every token in every layer without any drop 
            if latent_hint is not None:
                hidden_states = hidden_states + latent_hint.unsqueeze(1)
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )




class GPT2ForDecoderLatentConnector(GPT2PreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-1`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import torch
        from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

    """
    def __init__(self, config, latent_size=32, nclasses=0):
        
        super(GPT2ForDecoderLatentConnector, self).__init__(config)

        
        self.transformer = GPT2LatentDecoderModel(config, latent_size=latent_size, nclasses=nclasses)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()
        self.tie_weights()




    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.transformer.wte)

    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, class_label=None, label_ignore=None, loss_mask=None):

        transformer_outputs = self.transformer(input_ids,
                                               past_key_values=past,
                                               class_label=class_label,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask)
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=label_ignore, reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))

            if loss_mask is not None:
                loss = loss*loss_mask
            loss = torch.sum(loss.view(-1, shift_labels.shape[-1]), -1)
            outputs = (loss,) + outputs


        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)




class LatentSelfAttention(nn.Module):
    def __init__(self, attention_size, activation_method='relu'):
        super(LatentSelfAttention, self).__init__()
        self.linear_trans = nn.Linear(attention_size, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = ACT2FN[activation_method]
        

    def forward(self, inputs, attention_mask=None):

        ##################################################################
        # STEP 1 - perform linear transformation to inputs x
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        key_z = self.linear_trans(inputs).squeeze(-1)
        scores = self.activation(key_z)

        if attention_mask is not None:
            scores = scores + attention_mask

        ##################################################################
        # Step 2 - Attention Multipliy
        # x as V_z, f(x) as K_z, E is the Q_z
        ##################################################################
        scores = self.softmax(scores / key_z.size(0)**0.5)

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        representations = weighted.sum(1).squeeze(1)

        return representations, scores


class GPT2ForEncoderLatentConnector(GPT2PreTrainedModel):

    def __init__(self, config):
        
        super(GPT2ForEncoderLatentConnector, self).__init__(config)

        self.transformer = GPT2Model(config)
        self.encoder_pooling_attention = LatentSelfAttention(config.hidden_size)
        self.hidden_size = config.hidden_size # for td-vae usage outside of this model 
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        paragraph_lengths=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        max_split_num=64,
        noise_type="discrete",
        unk_token_id=None
    ):
        # NOTE: This is to add some Gaussian noise to the encoder 
        # gaussian hypersphere, mu = 0, var = (zeta/3)^2
        # add some discrete noise to inputs
        # input_ids = word_drop(input_ids, 0.2, unk_token_id)
        # inputs_embeds = self.transformer.wte(input_ids)
        # zeta = 3.0 
        # noise = torch.rand_like(inputs_embeds) #LBE
        # embeds_magn = torch.norm(inputs_embeds, dim=-1) #LB
        # noise_magn = zeta/3 * torch.randn(embeds_magn.size(0), embeds_magn.size(1), device=inputs_embeds.device)#LB
        # new_noise = (noise_magn * embeds_magn).unsqueeze(-1) * torch.nn.functional.normalize(noise, p=2.0, eps=1e-12, dim=-1) #LBE
        # inputs_embeds = (inputs_embeds.clone() + new_noise).half()
        # input_ids = None  # you can not specify inputs_embeds and input_ids at the same time
            
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = transformer_outputs[0] # dim = B, L, H
        # TODO: carefully design the attention mask such that each sentence is pooled into a accumulated belief state
        batch_pooling_att_inputs, batch_actual_para_lengths = [], []
        for batch_idx, para_len_list in zip(range(last_hidden_state.shape[0]), paragraph_lengths):
            pooling_att_inputs, accm_idx = [], 0
            actual_para_lengths = []
            for para_len in para_len_list:
                if accm_idx+para_len > last_hidden_state.shape[1]: 
                    break  # avoid the unfinished sentence, no point to reconstruct them  
                pooling_att_inputs.append(last_hidden_state[batch_idx, :accm_idx+para_len, :])
                accm_idx += para_len
                actual_para_lengths.append(para_len)
                if len(actual_para_lengths) > max_split_num:
                    break  # avoid a very long paragraph for memory purpose
            pooling_att_inputs = pad_sequence(pooling_att_inputs, batch_first=True, padding_value=0)
            batch_pooling_att_inputs.append(pooling_att_inputs)
            batch_actual_para_lengths.append(actual_para_lengths)
        
        # for each batch sample(aka paragraph), we split to multiple sentences, thus 
        # we have dimension [B, Num_sent, H] for batch_belief_states
        batch_belief_states = [] 
        for pooling_att_inputs in batch_pooling_att_inputs:
            # Num_sent, Sent_len, H => Num_sent, H
            belief_state, scores = self.encoder_pooling_attention(pooling_att_inputs)
            batch_belief_states.append(belief_state)
        return batch_belief_states, batch_actual_para_lengths