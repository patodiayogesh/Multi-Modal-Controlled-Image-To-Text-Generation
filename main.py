from dataset import FlickrDatasetModule
from model import BaselineModel
import pytorch_lightning as pl
import argparse
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from seutil import LoggingUtils
from pytorch_lightning.utilities.cli import (
    LR_SCHEDULER_REGISTRY,
    OPTIMIZER_REGISTRY,
)
import transformers
import torch

class BaselineTrainer:

    def __init__(self,
                 image_encoder,
                 text_decoder,
                 dataset,
                 model_ckpt,
                 freeze_image_encoder,
                 fast_dev,
                 predict):

        self.model = BaselineModel(image_encoder, text_decoder,
                                   freeze_image_encoder,
                                   model_ckpt=model_ckpt)

        if dataset == 'flickr30k':
            predict_file = predict if predict != '' else None
            self.dataModule = FlickrDatasetModule(predict_file=predict_file)
        else:
            raise RuntimeError("Incorrect Dataset")
        self.dataModule.set_encoder_and_decoder_tokenizer(
            self.model.image_feature_extractor,
            self.model.decoder_tokenizer
        )

        early_stopping_callback = EarlyStopping(monitor='loss/train',
                                                mode='min',
                                                )
        lr_monitor_callback = LearningRateMonitor(logging_interval='step')
        self.trainer = pl.Trainer(
            fast_dev_run=fast_dev,
            num_sanity_val_steps=1,
            max_epochs=20,
            callbacks=[
                early_stopping_callback,
                lr_monitor_callback
            ],
            # GPU specific
            accelerator='gpu',
            devices=1,
            # accumulate_grad_batches=12,
            # strategy='ddp',
        )

    def train_model(self):
        self.trainer.fit(self.model, self.dataModule)
        return

    def inference(self):
        self.trainer.predict(self.model, self.dataModule)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_encoder', type=str, default='beit', required=False)
    parser.add_argument('--text_decoder', type=str, default='bert', required=False)
    parser.add_argument('--dataset', type=str, default='flickr30k', required=False)
    parser.add_argument('--model_ckpt', type=str, required=False)
    parser.add_argument('--freeze_image_encoder', type=bool, default=False)
    parser.add_argument('--fast_dev', type=bool, default=False)
    parser.add_argument('--predict', type=str, default='')
    args = parser.parse_args()

    LoggingUtils.setup(LoggingUtils.INFO, 'baselineModel.log')
    OPTIMIZER_REGISTRY.register_classes(
        transformers.optimization, torch.optim.Optimizer, override=True
    )
    LR_SCHEDULER_REGISTRY.register_classes(
        transformers.optimization, torch.optim.lr_scheduler._LRScheduler, override=True
    )

    trainer = BaselineTrainer(
        args.image_encoder,
        args.text_decoder,
        args.dataset,
        args.model_ckpt,
        args.freeze_image_encoder,
        args.fast_dev,
        args.predict,
    )

    if args.predict == '':
        trainer.train_model()
    else:
        trainer.inference()
