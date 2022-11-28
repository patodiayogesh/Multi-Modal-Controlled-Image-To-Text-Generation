from trainer import Trainer
from baseline_module import BaselineModel
from multi_modal_module import MultiModalModel
from vqa_dataset import VQADatasetModule
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='MultiModal')
    parser.add_argument('--dataset', type=str, default='flickr')
    parser.add_argument('--multi_modal', type=bool, default=True)
    parser.add_argument('--mask', type=str, default='empty', choices=['empty', 'epoch_aware_mask','text_infilling'])
    parser.add_argument('--model_ckpt', type=str, required=False)
    parser.add_argument('--predict', type=str, default=None)

    args = parser.parse_args()

    if args.model_name == 'MultiModal':
        model = MultiModalModel(args.model_ckpt)
    else:
        model = BaselineModel(args.model_ckpt)
    if args.dataset == 'flickr':
        dataset = VQADatasetModule()
    trainer = Trainer(model, dataset)
    if args.predict:
        trainer.inference()
    else:
        trainer.fit()
