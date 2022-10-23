from dataset import FlickrDatasetModule
from model import BaselineModel
import pytorch_lightning as pl

if __name__ == '__main__':

    dataset = FlickrDatasetModule()
    model = BaselineModel('beit', 'bert')
    dataset.set_encoder_and_decoder_tokenizer(model.image_feature_extractor, model.decoder_tokenizer)
    # train_dataloader = dataset.train_dataloader()
    # for x,y in train_dataloader:
    #     print()
    # print()
    trainer = pl.Trainer()
    trainer.fit(model, dataset)
    print()