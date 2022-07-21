from logging import raiseExceptions
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from types import SimpleNamespace
from fengshen.models.tdvae.tdvae import TDVAE
from pytorch_lightning.core.lightning import LightningModule
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.bert.tokenization_bert import BertTokenizer
from fengshen.models.tdvae.latent_connector import GPT2ForDecoderLatentConnector, GPT2ForEncoderLatentConnector
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


class TDVAEModule(LightningModule):
    @classmethod
    def add_module_specific_args(cls, parser):
        group = parser.add_argument_group('vae', 'configurations')
        group.add_argument("--checkpoint_path", type=str, default=None)
        group.add_argument("--max_split_num", default=12, type=int,
                        help="max split number of sentences for each paragraph")
        group.add_argument("-em","--encoder_model_path", type=str)
        group.add_argument("-dm","--decoder_model_path", type=str)
        group.add_argument("--beta_infer_belief", default=1, type=float,
                        help="max beta for infer_belief_kl_loss")
        group.add_argument("--beta_belief_predict", default=1, type=float,
                        help="max beta for belief_predict_kl_loss")
        group.add_argument("--beta_kl_constraints", default=1, type=float,
                        help="max beta for all the latent z posterior vs prior kl loss")
        group.add_argument("--beta_n_cycles", default=30, type=int,
                        help="number of cycles for kl loss ratio within an epoch") 
        group.add_argument("--freebit_infer_belief", default=.1, type=float,
                        help="free bit for infer_belief_kl_loss")
        group.add_argument("--freebit_belief_predict", default=.1, type=float,
                        help="free bit for belief_predict_kl_loss")
        group.add_argument("--freebit_kl_constraints", default=.1, type=float,
                        help="free bit for all the latent z kl loss")
        group.add_argument("--latent_dim", default=256, type=int,
                        help="latent dimension of TDVAE Z")

        group.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
        group.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
        group.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
        group.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
        group.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

        return parser
    
    @classmethod
    def load_model(cls, args, labels_dict=None):
        checkpoint = torch.load(os.path.join(args.checkpoint_path, 'mp_rank_00_model_states.pt'))
        args.encoder_model_path = args.encoder_model_path if hasattr(args, 'encoder_model_path') else checkpoint['hyper_args'].encoder_model_path
        args.decoder_model_path = args.decoder_model_path if hasattr(args, 'decoder_model_path') else checkpoint['hyper_args'].decoder_model_path 
        
        latent_size = checkpoint['latent_size'] if ('latent_size' in checkpoint.keys()) else args.latent_dim
        labels_dict = checkpoint['label_dict'] if ('label_dict' in checkpoint.keys()) else labels_dict
    
        anchor = 'module.model.encoder.'
        start = len(anchor)
        enc_dict = {key[start:]:val for key,val in checkpoint['module'].items() if anchor in key}
        enc_config = GPT2Config.from_pretrained(args.encoder_model_path)
        encoder_tokenizer = BertTokenizer.from_pretrained(args.encoder_model_path)
        special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>'}
        encoder_tokenizer.add_special_tokens(special_tokens_dict)
        nclasses = len(labels_dict) if labels_dict is not None else 0
        encoder_model = GPT2ForEncoderLatentConnector(config=enc_config)
        encoder_model.resize_token_embeddings(len(encoder_tokenizer)) 
        missing_keys,unexpected_keys = encoder_model.load_state_dict(enc_dict, strict=False)
        print(f"encoder loading process: missing keys {missing_keys}, unexpected keys {unexpected_keys}")

        anchor = 'module.model.decoder.'
        start = len(anchor)
        dec_dict = {key[start:]:val for key,val in checkpoint['module'].items() if anchor in key}
        dec_config = GPT2Config.from_pretrained(args.decoder_model_path)
        decoder_tokenizer = BertTokenizer.from_pretrained(args.decoder_model_path)
        special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>'}
        decoder_tokenizer.add_special_tokens(special_tokens_dict)
        nclasses = len(labels_dict) if labels_dict is not None else 0
        decoder_model = GPT2ForDecoderLatentConnector(config=dec_config, latent_size=latent_size, nclasses=nclasses)
        decoder_model.resize_token_embeddings(len(decoder_tokenizer)) 
        missing_keys,unexpected_keys = decoder_model.load_state_dict(dec_dict, strict=False)
        print(f"decoder loading process: missing keys {missing_keys}, unexpected keys {unexpected_keys}")
        pad_token_id = decoder_tokenizer.pad_token_id

        vae_model = TDVAE(encoder_model, decoder_model, latent_dim=latent_size, max_split_num=args.max_split_num,
            pad_token_id=decoder_tokenizer.pad_token_id, unk_token_id=decoder_tokenizer.unk_token_id,
            bos_token_id=decoder_tokenizer.bos_token_id, eos_token_id=decoder_tokenizer.eos_token_id)
        # NOTE: load the DBlock params!!!!!!!!
        anchor = 'module.model.belief_net.'
        start = len(anchor)
        net_dict = {key[start:]:val for key,val in checkpoint['module'].items() if anchor in key}
        missing_keys,unexpected_keys = vae_model.belief_net.load_state_dict(net_dict, strict=False)
        print(f"belief_net loading process: missing keys {missing_keys}, unexpected keys {unexpected_keys}")
        
        anchor = 'module.model.back_inference_net.'
        start = len(anchor)
        net_dict = {key[start:]:val for key,val in checkpoint['module'].items() if anchor in key}
        missing_keys,unexpected_keys = vae_model.back_inference_net.load_state_dict(net_dict, strict=False)
        print(f"back_inference_net loading process: missing keys {missing_keys}, unexpected keys {unexpected_keys}")
        
        anchor = 'module.model.transition_net.'
        start = len(anchor)
        net_dict = {key[start:]:val for key,val in checkpoint['module'].items() if anchor in key}
        missing_keys,unexpected_keys = vae_model.transition_net.load_state_dict(net_dict, strict=False)
        print(f"transition_net loading process: missing keys {missing_keys}, unexpected keys {unexpected_keys}")

        return vae_model, encoder_tokenizer, decoder_tokenizer, latent_size, labels_dict, args


    def __init__(
        self,
        args,
        train_steps=0,
        labels_dict=None
        ):
        super().__init__()
        # self.save_hyperparameters()
        self.args = args

        if args.checkpoint_path is not None:
            self.model, self.encoder_tokenizer, self.decoder_tokenizer, self.latent_size, \
                self.labels_dict, self.args =  TDVAEModule.load_model(self.args, labels_dict=labels_dict)
        else:
            self.encoder_tokenizer = BertTokenizer.from_pretrained(self.args.encoder_model_path)
            encoder_config = GPT2Config.from_pretrained(self.args.encoder_model_path)
            special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>'}
            self.encoder_tokenizer.add_special_tokens(special_tokens_dict)
            self.latent_size = self.args.latent_dim
            encoder = GPT2ForEncoderLatentConnector.from_pretrained(self.args.encoder_model_path, config=encoder_config)
            # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
            encoder.resize_token_embeddings(len(self.encoder_tokenizer))

            self.decoder_tokenizer = BertTokenizer.from_pretrained(self.args.decoder_model_path)
            self.decoder_tokenizer.add_special_tokens(special_tokens_dict)
            decoder_config = GPT2Config.from_pretrained(self.args.decoder_model_path)
            nclasses = len(labels_dict) if labels_dict is not None else 0
            self.labels_dict = labels_dict
            decoder = GPT2ForDecoderLatentConnector.from_pretrained(self.args.decoder_model_path, config=decoder_config, latent_size=self.latent_size, nclasses=nclasses)

            
            # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
            decoder.resize_token_embeddings(len(self.decoder_tokenizer))
            self.model = TDVAE(encoder, decoder, latent_dim=self.args.latent_dim, max_split_num=self.args.max_split_num,
                pad_token_id=self.decoder_tokenizer.pad_token_id, unk_token_id=self.decoder_tokenizer.unk_token_id,
                bos_token_id=self.decoder_tokenizer.bos_token_id, eos_token_id=self.decoder_tokenizer.eos_token_id)
        
        self.train_steps = train_steps
        self.beta_kl_constraints_list = self.get_cyclic_linear_beta_list(self.train_steps, 
            start=0, stop=args.beta_kl_constraints,  n_cycle=args.beta_n_cycles)
        self.mlm_probability_list = self.get_decoder_beta_list(self.train_steps, 
            start=0., stop=1.,  n_cycle=args.beta_n_cycles)
        self.beta_belief_predict_list = self.get_constant_ratio(self.train_steps, args.beta_belief_predict)
        self.beta_infer_belief_list = self.get_constant_ratio(self.train_steps, args.beta_infer_belief)
        # self.beta_kl_constraints_list = self.get_constant_ratio(self.train_steps, args.beta_kl_constraints)
        # self.mlm_probability_list = self.get_constant_ratio(self.train_steps, 0.)

        self.freebit_infer_belief = args.freebit_infer_belief
        self.freebit_belief_predict = args.freebit_belief_predict
        self.freebit_kl_constraints = args.freebit_kl_constraints

    def get_constant_ratio(self, n_steps, ratio):
        L = np.ones(n_steps)
        L *= ratio
        return L

    def get_decoder_beta_list(self, n_steps, start=0., stop=1.0, n_cycle=4):
        L = np.ones(n_steps)
        t_range = int(n_steps / n_cycle)
        for t_cur in range(n_steps):
            if t_cur > t_range:
                L[t_cur] = 0.
            else:    
                ratio = t_cur / t_range
                value = stop - ratio * (stop-start)
                L[t_cur] = value
        return L 

    def get_cyclic_linear_beta_list(self, n_steps, start=0.5, stop=1.0, n_cycle=4):
        L = np.ones(n_steps)
        t_range = int(n_steps / n_cycle)
        for t_cur in range(n_steps):
            loc = t_cur % t_range
            split_range = int(t_range * 0.5)
            if loc < split_range:
                ratio = (loc % split_range) / split_range
                value = ratio * (stop-start)
            elif split_range <= loc:
                value = stop
            L[t_cur] = value
        return L 

    #####
    # Torch lightning
    #####

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint['label_dict'] = self.labels_dict
        checkpoint['latent_size'] = self.latent_size

    def training_step(self, batch, batch_idx):
        encoder_inputs, decoder_labels, paragraph_lengths = batch[0], batch[1], batch[2]
        if encoder_inputs is None or decoder_labels is None or paragraph_lengths is None:
            total_loss = torch.Tensor([0.]).to(next(self.model.parameters()).device)
            total_loss.requires_grad=True
            return total_loss
        
        # encoder_inputs, _ = self.mask_tokens(inputs)
        total_loss, total_belief_predict_loss, total_infer_belief_loss, total_infer_rec_loss, total_belief_rec_loss, total_predict_rec_loss, total_kl_constraint_loss = \
            self.model(encoder_inputs, decoder_labels, 
            paragraph_lengths=paragraph_lengths, 
            beta_belief_predict=self.beta_belief_predict_list[batch_idx],
            beta_infer_belief=self.beta_infer_belief_list[batch_idx],
            beta_kl_constraints=self.beta_kl_constraints_list[batch_idx],
            freebit_infer_belief=self.freebit_infer_belief,
            freebit_belief_predict=self.freebit_belief_predict,
            freebit_kl_constraints=self.freebit_kl_constraints,
            mlm_probability=self.mlm_probability_list[batch_idx],
            sample_z=True)

        # the logging interval are set by the trainer_args log_every_n_steps
        for idx, pg in enumerate(self.optimizers().param_groups):
            self.log(f"learning_rate_{idx}", pg['lr'])
        unscaled_belief_predict_loss = 0. if self.beta_belief_predict_list[batch_idx] == 0. else total_belief_predict_loss/self.beta_belief_predict_list[batch_idx]
        unscaled_infer_belief_loss = 0. if self.beta_infer_belief_list[batch_idx] == 0. else total_infer_belief_loss/self.beta_infer_belief_list[batch_idx]
        unscaled_kl_constraint_loss = 0. if self.beta_kl_constraints_list[batch_idx] == 0. else total_kl_constraint_loss/self.beta_kl_constraints_list[batch_idx]
        self.log("total_loss", total_loss)
        self.log("total_infer_rec_loss", total_infer_rec_loss)
        self.log("total_belief_rec_loss", total_belief_rec_loss)
        self.log("total_predict_rec_loss", total_predict_rec_loss)
        self.log("total_belief_predict_loss", total_belief_predict_loss)
        self.log("total_infer_belief_loss", total_infer_belief_loss)
        self.log("total_kl_constraint_loss", total_kl_constraint_loss)
        self.log("unscaled_belief_predict_loss", unscaled_belief_predict_loss)
        self.log("unscaled_infer_belief_loss", unscaled_infer_belief_loss)
        self.log("unscaled_kl_constraint_loss", unscaled_kl_constraint_loss)
        self.log("beta_belief_predict", self.beta_belief_predict_list[batch_idx])
        self.log("beta_infer_belief", self.beta_infer_belief_list[batch_idx])
        self.log("beta_kl_constraints", self.beta_kl_constraints_list[batch_idx])
        self.log("decoder_mask_ratio", self.mlm_probability_list[batch_idx])
        
        return total_loss

    def training_step_end(self, batch_parts):
        pass

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        encoder_inputs, decoder_labels, paragraph_lengths = batch[0], batch[1], batch[2]
        if encoder_inputs is None or decoder_labels is None or paragraph_lengths is None:
            loss = torch.Tensor([0.]).to(next(self.model.parameters()).device)
            loss.requires_grad=True
            return loss
    
        # encoder_inputs, _ = self.mask_tokens(inputs)
        total_loss, total_belief_predict_loss, total_infer_belief_loss, total_infer_rec_loss, total_belief_rec_loss, total_predict_rec_loss, total_kl_constraint_loss = \
            self.model(encoder_inputs, decoder_labels, 
            paragraph_lengths=paragraph_lengths, 
            beta_belief_predict=1.,
            beta_infer_belief=1.,
            beta_kl_constraints=1.,
            freebit_infer_belief=self.freebit_infer_belief,
            freebit_belief_predict=self.freebit_belief_predict,
            freebit_kl_constraints=self.freebit_kl_constraints,
            mlm_probability=0.)

        # the logging interval are set by the trainer_args log_every_n_steps
        self.log("val_total_loss", total_loss)
        self.log("val_belief_predict_loss", total_belief_predict_loss)
        self.log("val_infer_belief_loss", total_infer_belief_loss)
        self.log("val_kl_constraint_loss", total_kl_constraint_loss)
        self.log("val_infer_recon_loss", total_infer_rec_loss)
        self.log("val_belief_recon_loss", total_belief_rec_loss)
        self.log("val_predict_recon_loss", total_predict_rec_loss)
        return total_loss

    def validation_epoch_end(self, outputs):
        pass


    def test_step(self, batch, batch_idx):
        encoder_inputs, decoder_labels, paragraph_lengths = batch[0], batch[1], batch[2]
        if encoder_inputs is None or decoder_labels is None or paragraph_lengths is None:
            loss = torch.Tensor([0.]).to(next(self.model.parameters()).device)
            loss.requires_grad=True
            return loss

        # encoder_inputs, _ = self.mask_tokens(inputs)
        total_loss, total_belief_predict_loss, total_infer_belief_loss, total_infer_rec_loss, total_belief_rec_loss, total_predict_rec_loss, total_kl_constraint_loss = \
            self.model(encoder_inputs, decoder_labels, 
            paragraph_lengths=paragraph_lengths, 
            beta_belief_predict=1.,
            beta_infer_belief=1.,
            beta_kl_constraints=1.,
            freebit_infer_belief=self.freebit_infer_belief,
            freebit_belief_predict=self.freebit_belief_predict,
            freebit_kl_constraints=self.freebit_kl_constraints,
            mlm_probability=0.)

        self.log("test_total_loss", total_loss)
        self.log("test_belief_predict_loss", total_belief_predict_loss)
        self.log("test_infer_belief_loss", total_infer_belief_loss)
        self.log("test_infer_recon_loss", total_infer_rec_loss)
        self.log("test_belief_recon_loss", total_belief_rec_loss)
        self.log("test_predict_recon_loss", total_predict_rec_loss)
        self.log("test_kl_constraint_loss", total_kl_constraint_loss)
        return total_loss

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.train_steps)
        
        return {'optimizer':optimizer,
                'lr_scheduler': {                                                                                                         
                    'scheduler': scheduler,                                                                                               
                    'interval': 'step',                                                                                                   
                    'frequency': 1                                                                                                        
                }
        }
