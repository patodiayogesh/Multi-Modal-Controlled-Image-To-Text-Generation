from trainer import Trainer
from baseline_module import BaselineModel
from dataset import FlickrDatasetModule
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_ckpt', type=str, required=False)
    parser.add_argument('--predict', type=str, default=None)
    args = parser.parse_args()

    model = BaselineModel(args.model_ckpt)
    dataset = FlickrDatasetModule(predict_file=args.predict)
    trainer = Trainer(model, dataset)
    if args.predict:
        trainer.inference()
    else:
        trainer.fit()
