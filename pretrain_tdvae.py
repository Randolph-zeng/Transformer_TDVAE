import torch
import os
import math
import argparse
from fengshen.data.fs_datasets.fs_datamodule import FSDataModule
from fengshen.models.tdvae.vae_pl_module import TDVAEModule

from pytorch_lightning import (
    Trainer,
    loggers,
)

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.nn.utils.rnn import pad_sequence


# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

class RNNCollator:
    def __call__(self, samples):
        # when len(samples) is larger than one, we need to save the sentence length info 
        paragraph_lengths = []
        encoder_inputs, decoder_labels = [], []
        for sp in samples:
            # NOTE: in TD-VAE, both encoder and decoder are gpt2, thus use decoder sent twice !
            encoder_sent_lengths, decoder_sent_lengths = sp['decoder_sent_lengths'], sp['decoder_sent_lengths']  
            input_ids, decoder_target = sp['decoder_target'], sp['decoder_target']
            if len(encoder_sent_lengths) == 1 or encoder_sent_lengths[0] >= 512 or encoder_sent_lengths[0] + encoder_sent_lengths[1] >= 512:
                continue # we ignore paragraphs with only one sentence split
            encoder_inputs.append(torch.tensor(input_ids[:512], dtype=torch.long))
            decoder_labels.append(torch.tensor(decoder_target[:512], dtype=torch.long))
            paragraph_lengths.append(encoder_sent_lengths)
        if not encoder_inputs or not decoder_labels:
            return None, None, None  # if all the examples in the batch are single sentence
        encoder_inputs = pad_sequence(encoder_inputs, batch_first=True, padding_value=0)
        decoder_labels = pad_sequence(decoder_labels, batch_first=True, padding_value=0)
        return (encoder_inputs, decoder_labels, paragraph_lengths) 


class VAEModelCheckpoint:
    @ staticmethod
    def add_argparse_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')

        parser.add_argument('--monitor', default='total_loss', type=str)
        parser.add_argument('--mode', default='min', type=str)
        parser.add_argument('--dirpath', default='./log/', type=str)
        parser.add_argument('--filename', default='model-{epoch:2d}-{train_loss:.4f}', type=str)

        parser.add_argument('--save_top_k', default=-1, type=int)
        parser.add_argument('--every_n_train_steps', default=1000, type=float)
        parser.add_argument('--save_weights_only', default=True, type=bool)

        return parent_args

    @staticmethod
    def get_callback(args):
        return ModelCheckpoint(monitor=args.monitor,
                                    save_top_k=args.save_top_k,
                                    mode=args.mode,
                                    every_n_train_steps=args.every_n_train_steps,
                                    save_weights_only=args.save_weights_only,
                                    dirpath=args.dirpath,
                                    filename=args.filename)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()

    args_parser = FSDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = TDVAEModule.add_module_specific_args(args_parser)
    args_parser = VAEModelCheckpoint.add_argparse_args(args_parser)

    args = args_parser.parse_args()
    rnn_collator = RNNCollator()
    data_module = FSDataModule(args=args, collate_fn=rnn_collator)

    train_steps = math.ceil(len(data_module.train_dataset)*args.max_epochs/ \
                            args.train_batchsize / args.num_nodes / args.gpus)
    model = TDVAEModule(args, train_steps)

    logger = loggers.TensorBoardLogger(save_dir=os.path.join(
        args.default_root_dir, 'logs/'), name='tdvae_lightning')


    save_cpt_callback = VAEModelCheckpoint.get_callback(args)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer.from_argparse_args(args,
                                         callbacks=[save_cpt_callback, lr_monitor],
                                         logger=logger)
    trainer.fit(model, data_module)
