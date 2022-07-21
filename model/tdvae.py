import torch
import torch.nn as nn
from fengshen.models.tdvae.tdvae_utils import *
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class newDBlock(nn.Module):
    """ A basic building block for computing parameters of a normal distribution."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logsigma = nn.Linear(hidden_size, output_size)
        self.act_func = torch.nn.GELU()

    def forward(self, input_):
        t = self.act_func(self.fc1(input_))
        mu = self.fc_mu(t)
        logsigma = self.fc_logsigma(t)
        return mu, logsigma


class DBlock(nn.Module):
    """ A basic building block for computing parameters of a normal distribution.
    Corresponds to D in the appendix."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logsigma = nn.Linear(hidden_size, output_size)

    def forward(self, input_):
        t = torch.tanh(self.fc1(input_))
        t = t * torch.sigmoid(self.fc2(input_))
        mu = self.fc_mu(t)
        logsigma = self.fc_logsigma(t)
        return mu, logsigma


class TDVAE(nn.Module):
    """TDVAE with RNN being replaced by transformer"""
    def __init__(self, encoder, decoder, latent_dim, max_split_num, pad_token_id, unk_token_id, bos_token_id, eos_token_id):
        super(TDVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.max_split_num = max_split_num
        # belief network is P(Z_t|S_t)
        belief_hidden_size = self.encoder.hidden_size * 3
        self.belief_net = DBlock(self.encoder.hidden_size, belief_hidden_size, latent_dim)
        # back_inference is P(Z_1|S_1, S_2, Z_2)
        back_inf_hidden_size = (self.encoder.hidden_size*2+latent_dim) * 3
        self.back_inference_net = DBlock(self.encoder.hidden_size*2+latent_dim, back_inf_hidden_size, latent_dim)
        # transition_net is P(Z_2|S_1,Z_1)
        trans_input_size = latent_dim+self.encoder.hidden_size
        trans_hidden_size = trans_input_size * 3 
        self.transition_net = DBlock(trans_input_size, trans_hidden_size, latent_dim)


    def get_belief_predict_network_output(self, belief_states, sample=True):
        # Num_sents, Hidden_size => Num_sents, Latent_size
        belief_sent_mus, belief_sent_logvars = self.belief_net(belief_states)
        belief_latent_z = connect(belief_sent_mus, belief_sent_logvars, sample=sample)
        # constructing the transition inputs P(Z2|S1,Z1)
        transition_inputs = []
        for i in range(belief_states.shape[0]):
            transition_inputs.append(torch.cat((belief_states[i,:], belief_latent_z[i, :]), dim=0))
        transition_inputs = torch.stack(transition_inputs, dim=0)
        predict_sent_mus, predict_sent_logvars = self.transition_net(transition_inputs)
        predict_latent_z = connect(predict_sent_mus, predict_sent_logvars, sample=sample)
        return belief_sent_mus, belief_sent_logvars, belief_latent_z, predict_sent_mus, predict_sent_logvars, predict_latent_z

    def get_back_inf_network_output(self, belief_states, belief_latent_z, actual_para_lengths, sample=True):
        # constructing the back-inference inputs P(Z1|S1,S2,Z2)
        back_inf_inputs = []
        for i in range(len(actual_para_lengths)-1):
            back_inf_inputs.append(torch.cat((belief_states[i,:], belief_states[i+1,:], 
                belief_latent_z[i+1, :]), dim=0))
        back_inf_inputs = torch.stack(back_inf_inputs, dim=0)
        inf_sent_mus, inf_sent_logvars = self.back_inference_net(back_inf_inputs)
        inf_latent_z = connect(inf_sent_mus, inf_sent_logvars, sample=sample)
        return inf_sent_mus, inf_sent_logvars, inf_latent_z

    def get_decoder_input_target(self, actual_para_lengths, labels, batch_idx, mlm_probability):
        accm_idx = 0
        decoder_input, decoder_target = [], []
        for idx in range(len(actual_para_lengths)):
            apl = actual_para_lengths[idx]
            # NOTE: add noise to decoder to enforce dependency on predicted latent variables! Also since there is no bos/eos in the inputs
            # we manually add it in decoder. We don't need to worry eos will predict anything since its corresponding class will only be pad token
            # and pad token will be ignored by cross entropy loss
            noisy_decoder_input = word_drop(labels[batch_idx, accm_idx:accm_idx+apl], 
                mlm_probability, self.unk_token_id)
            noisy_decoder_input = torch.cat([torch.tensor([self.bos_token_id], dtype=torch.long, device=labels.device), 
                noisy_decoder_input, torch.tensor([self.eos_token_id], dtype=torch.long, device=labels.device)], dim=0)  
            decoder_input.append(noisy_decoder_input)
            clean_decoder_target = torch.cat([torch.tensor([self.bos_token_id], dtype=torch.long, device=labels.device), 
                labels[batch_idx, accm_idx:accm_idx+apl], torch.tensor([self.eos_token_id], dtype=torch.long, device=labels.device)], dim=0)  
            decoder_target.append(clean_decoder_target)
            accm_idx += apl
        decoder_input = pad_sequence(decoder_input, batch_first=True, padding_value=self.pad_token_id)
        decoder_target = pad_sequence(decoder_target, batch_first=True, padding_value=self.pad_token_id)
        return decoder_input, decoder_target
    
    def get_decoder_loss(self, decoder_input, decoder_target, predict_latent_z, 
        belief_latent_z, inf_latent_z, actual_para_lengths):
        # In order to get same dimension loss for predict, inf and belief, we restrict belief,inf reconstruct to n-1
        inf_rec_loss = self.decoder(input_ids=decoder_input[:-1,:], past=inf_latent_z, 
            labels=decoder_target[:-1,:], label_ignore=self.pad_token_id)[0]
        belief_rec_loss = self.decoder(input_ids=decoder_input[:-1,:], past=belief_latent_z[:-1,:], 
            labels=decoder_target[:-1,:], label_ignore=self.pad_token_id)[0]
        predict_rec_loss = self.decoder(input_ids=decoder_input[1:,:], past=predict_latent_z[:-1,:], 
            labels=decoder_target[1:, :], label_ignore=self.pad_token_id)[0]
        reduce_inf_rec_loss = inf_rec_loss / torch.tensor(actual_para_lengths[:-1], device=inf_rec_loss.device)
        reduce_belief_rec_loss = belief_rec_loss / torch.tensor(actual_para_lengths[:-1], device=belief_rec_loss.device)
        reduce_predict_rec_loss = predict_rec_loss / torch.tensor(actual_para_lengths[1:], device=predict_rec_loss.device)
        # TODO: make this beta ratio adjustable ???
        return 0.1 * reduce_belief_rec_loss.mean(), reduce_predict_rec_loss.mean(), 0.1 * reduce_inf_rec_loss.mean()

    def reduce_all_losses(self, belief_rec_loss, predict_rec_loss, inf_rec_loss, belief_predict_kl_loss, infer_belief_kl_loss, kl_constraint_loss,
        total_infer_rec_loss, total_belief_rec_loss, total_predict_rec_loss, total_infer_belief_loss, total_belief_predict_loss, total_kl_constraint_loss):
        # accumulates all the losses
        total_infer_rec_loss = inf_rec_loss if total_infer_rec_loss is None else total_infer_rec_loss+inf_rec_loss
        total_belief_rec_loss = belief_rec_loss if total_belief_rec_loss is None else total_belief_rec_loss+belief_rec_loss
        total_predict_rec_loss = predict_rec_loss if total_predict_rec_loss is None else total_predict_rec_loss+predict_rec_loss
        
        total_infer_belief_loss = infer_belief_kl_loss if total_infer_belief_loss is None else total_infer_belief_loss+infer_belief_kl_loss
        total_belief_predict_loss = belief_predict_kl_loss if total_belief_predict_loss is None else total_belief_predict_loss+belief_predict_kl_loss
        total_kl_constraint_loss = kl_constraint_loss if total_kl_constraint_loss is None else total_kl_constraint_loss+kl_constraint_loss
        if any([torch.isnan(total_infer_rec_loss),
                torch.isnan(total_belief_rec_loss),
                torch.isnan(total_predict_rec_loss),
                torch.isnan(total_belief_predict_loss),
                torch.isnan(total_infer_belief_loss),
                torch.isnan(total_kl_constraint_loss)]):
            print("nan reached")
        return total_infer_rec_loss, total_belief_rec_loss, total_predict_rec_loss, total_infer_belief_loss, total_belief_predict_loss, total_kl_constraint_loss

    def reduce_losses_by_batch_size(self, total_infer_rec_loss, total_belief_rec_loss, total_predict_rec_loss, total_infer_belief_loss, total_belief_predict_loss, 
        total_kl_constraint_loss, batch_size):
        total_infer_rec_loss /= batch_size
        total_belief_rec_loss /= batch_size
        total_predict_rec_loss /= batch_size
        total_belief_predict_loss /= batch_size
        total_infer_belief_loss /= batch_size
        total_kl_constraint_loss /= batch_size
        total_loss = total_belief_predict_loss + total_infer_belief_loss + total_kl_constraint_loss + total_infer_rec_loss + total_belief_rec_loss + total_predict_rec_loss
        return total_loss, total_belief_predict_loss, total_infer_belief_loss, total_infer_rec_loss, total_belief_rec_loss, total_predict_rec_loss, total_kl_constraint_loss

    def compute_all_kl_constraints(self, belief_sent_mus, belief_sent_logvars, 
            predict_sent_mus, predict_sent_logvars, inf_sent_mus, inf_sent_logvars, 
            beta_belief_predict, beta_infer_belief, beta_kl_constraints,
            freebit_belief_predict, freebit_infer_belief, freebit_kl_constraints):
        # NOTE: BE VERY CAREFUL ABOUT THE ORDER OF KL DIVERGENCE!!! The constraint should be placed in the right side!
        # To calculate logpB(z2|s2) - logpP(z2|z1), we shift the belief_latent_z by one and bound this two with KL divergence 
        recursive_belief_predict_kl_loss = kl_loss(predict_sent_mus[:-1, :], predict_sent_logvars[:-1, :], 
            belief_sent_mus[1:,:], belief_sent_logvars[1:,:])
        # Calculate the kl divergence between back-inference and belief distribution            
        recursive_infer_belief_kl_loss = kl_loss(belief_sent_mus[:-1, :], belief_sent_logvars[:-1, :],
            inf_sent_mus, inf_sent_logvars)
        # NOTE we apply beta ratio and freebits here
        recursive_belief_predict_kl_loss = torch.max(recursive_belief_predict_kl_loss, torch.tensor(freebit_belief_predict, 
            device=recursive_belief_predict_kl_loss.device))
        recursive_infer_belief_kl_loss = torch.max(recursive_infer_belief_kl_loss, torch.tensor(freebit_infer_belief, 
            device=recursive_infer_belief_kl_loss.device))
        belief_predict_kl_loss = beta_belief_predict * recursive_belief_predict_kl_loss.mean() 
        infer_belief_kl_loss = beta_infer_belief * recursive_infer_belief_kl_loss.mean()
        # add posterior&prior constraint to all latent z prediction as all the VAEs do! 
        belief_kl_loss = kl_loss(belief_sent_mus, belief_sent_logvars, 
            torch.zeros_like(belief_sent_mus), torch.ones_like(belief_sent_logvars))
        belief_kl_loss = torch.max(belief_kl_loss, torch.tensor(freebit_kl_constraints, 
            device=belief_kl_loss.device))
        predict_kl_loss = kl_loss(predict_sent_mus, predict_sent_logvars,
            torch.zeros_like(predict_sent_mus), torch.ones_like(predict_sent_logvars))
        predict_kl_loss = torch.max(predict_kl_loss, torch.tensor(freebit_kl_constraints, 
            device=predict_kl_loss.device))
        back_inf_kl_loss = kl_loss(inf_sent_mus, inf_sent_logvars,
            torch.zeros_like(inf_sent_mus), torch.ones_like(inf_sent_logvars))
        back_inf_kl_loss = torch.max(back_inf_kl_loss, torch.tensor(freebit_kl_constraints, 
            device=back_inf_kl_loss.device))
        # NOTE: we added beta and freebit to kl constraint loss
        kl_constraint_loss = belief_kl_loss.mean() + predict_kl_loss.mean() + back_inf_kl_loss.mean()
        kl_constraint_loss = beta_kl_constraints * kl_constraint_loss
        return kl_constraint_loss, belief_predict_kl_loss, infer_belief_kl_loss

    def forward(self, inputs, labels, paragraph_lengths, beta_belief_predict, beta_infer_belief, beta_kl_constraints, 
        freebit_infer_belief, freebit_belief_predict, freebit_kl_constraints, mlm_probability, sample_z=False):
        # each list in batch_belief_states contain a tensor with shape Num_sents * H. Num_sents vary across batch
        batch_belief_states, batch_actual_para_lengths = self.encoder(inputs, paragraph_lengths, max_split_num=self.max_split_num, unk_token_id=self.unk_token_id)
        total_infer_belief_loss, total_belief_predict_loss, total_kl_constraint_loss = None, None, None
        total_infer_rec_loss, total_belief_rec_loss, total_predict_rec_loss = None, None, None
        # we iterate over the paragraphs in the batch one by one
        for batch_idx, belief_states, actual_para_lengths in zip(range(inputs.shape[0]), batch_belief_states, batch_actual_para_lengths):
            belief_sent_mus, belief_sent_logvars, belief_latent_z, predict_sent_mus, \
                predict_sent_logvars, predict_latent_z = self.get_belief_predict_network_output(belief_states, sample=sample_z)
            inf_sent_mus, inf_sent_logvars, inf_latent_z = self.get_back_inf_network_output(belief_states, 
                belief_latent_z, actual_para_lengths, sample=sample_z)
            kl_constraint_loss, belief_predict_kl_loss, infer_belief_kl_loss = self.compute_all_kl_constraints(belief_sent_mus, 
                belief_sent_logvars, predict_sent_mus, predict_sent_logvars, inf_sent_mus, inf_sent_logvars, 
                beta_belief_predict, beta_infer_belief, beta_kl_constraints,
                freebit_belief_predict, freebit_infer_belief, freebit_kl_constraints)
            # calculate the reconstruct decoding loss for both belief and predict latent z 
            decoder_input, decoder_target = self.get_decoder_input_target(actual_para_lengths, labels, batch_idx, mlm_probability)
            belief_rec_loss, predict_rec_loss, inf_rec_loss = self.get_decoder_loss(decoder_input, decoder_target, predict_latent_z, 
                belief_latent_z, inf_latent_z, actual_para_lengths)
            # reduce all losses to scalars here
            total_infer_rec_loss, total_belief_rec_loss, total_predict_rec_loss, total_infer_belief_loss, total_belief_predict_loss, total_kl_constraint_loss = self.reduce_all_losses(
                belief_rec_loss, predict_rec_loss, inf_rec_loss, belief_predict_kl_loss, infer_belief_kl_loss, kl_constraint_loss,
                total_infer_rec_loss, total_belief_rec_loss, total_predict_rec_loss, total_infer_belief_loss, total_belief_predict_loss, total_kl_constraint_loss)
        # reduce again according to the batch size
        return self.reduce_losses_by_batch_size(total_infer_rec_loss, total_belief_rec_loss, total_predict_rec_loss, total_infer_belief_loss, 
            total_belief_predict_loss, total_kl_constraint_loss, inputs.shape[0])
        

    def inference(self, inputs, paragraph_lengths, decoder_tokenizer, max_length, top_p, temperature, repetition_penalty, mode="belief", sample_z=False):
        # each list in batch_belief_states contain a tensor with shape Num_sents * H. Num_sents vary across batch
        batch_belief_states, batch_actual_para_lengths = self.encoder(inputs, paragraph_lengths, 
            max_split_num=self.max_split_num, unk_token_id=self.unk_token_id)
        generated_outputs = []
        for belief_states, actual_para_lengths in zip(batch_belief_states, batch_actual_para_lengths):
            # Num_sents, Hidden_size => Num_sents, Latent_size
            belief_sent_mus, belief_sent_logvars, belief_latent_z, predict_sent_mus, \
                predict_sent_logvars, predict_latent_z = self.get_belief_predict_network_output(belief_states, sample=sample_z)
            # decode process 
            generated = [[decoder_tokenizer.bos_token_id] for _ in range(len(actual_para_lengths) -1)] 
            generated = torch.tensor(generated, dtype=torch.long, device=inputs.device)
            with torch.no_grad():
                for _ in range(max_length):
                    # TODO: use predict latent z 
                    if mode=="belief":
                        # test belief
                        outputs = self.decoder(input_ids=generated, past=belief_latent_z[1:,:], labels=None,
                            label_ignore=self.pad_token_id)
                    elif mode == "predict":
                        # test predict 
                        outputs = self.decoder(input_ids=generated, past=predict_latent_z[:-1,:], labels=None,
                            label_ignore=self.pad_token_id)
                    else:
                        # for actual inference we only care about the last predict latent z 
                        if len(generated.shape) == 1 and generated.shape[0] == 0: 
                            generated = torch.tensor([[decoder_tokenizer.bos_token_id]] , dtype=torch.long, device=inputs.device)
                        if predict_latent_z.shape[0] != 1: 
                            predict_latent_z = predict_latent_z[-1,:].unsqueeze(0)
                        outputs = self.decoder(input_ids=generated, past=predict_latent_z, labels=None,
                            label_ignore=self.pad_token_id)
                    next_token_logits = outputs[0][:, -1, :] / temperature
                    filtered_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)
                    log_probs = F.softmax(filtered_logits, dim=-1)
                    if repetition_penalty != 1.0:
                        enforce_repetition_penalty(log_probs, generated, repetition_penalty)
                    next_token = torch.multinomial(log_probs, num_samples=1)
                    generated = torch.cat((generated, next_token), dim=1)
                    if all(next_token[idx,0].item() == decoder_tokenizer.eos_token_id for idx in range(next_token.shape[0])):
                        break  # if all samples predict eos in the batch. 
            generated_outputs.append(generated)        
        return generated_outputs
