import torchvision
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision.transforms import transforms
from collections import defaultdict
import random
import pytorch_lightning as pl


class FlickrDataset(Dataset):

    def __init__(self,
                 image_files,
                 dataset,
                 transform=None,
                 ):

        self.image_dir = 'datasets/flickr30k_images/'
        self.images, self.captions = [], []
        for image in image_files:
            for caption in dataset[image]:
                self.captions.append(caption)
                self.images.append(image)
        self.unique_images = image_files
        self.transform = transform


    def __len__(self):

        return len(self.unique_images)

    def __getitem__(self, index):

        caption = self.captions[index]
        image_filename = self.images[index]
        img = Image.open(self.image_dir + image_filename)
        if self.transform:
            img = self.transform(img)
        return img, caption


class FlickrDatasetModule(pl.LightningDataModule):

    def _load_dataset(self):

        data = open('datasets/flickr30k/results_20130124.token', 'r').read().splitlines()
        captions = [x.split('\t')[1] for x in data]
        image_filenames = [x.split('#')[0] for x in data]
        image_captions = defaultdict(list)
        for image, caption in zip(image_filenames, captions):
            image_captions[image].append(caption)
        return image_captions

    def __init__(self,
                 train_batch_size=16,
                 eval_batch_size=16,
                 transform=transforms.PILToTensor(),
                 num_workers=12,
                 predict_file=None):

        super().__init__()

        flickr_dataset = self._load_dataset()
        flickr_dataset_filenames = list(flickr_dataset.keys())
        random.seed(42)
        random.shuffle(flickr_dataset_filenames)
        dataset_length = len(flickr_dataset_filenames)
        train_length = int(dataset_length * 0.8)
        train_dataset_filenames = flickr_dataset_filenames[:train_length]
        test_dataset = flickr_dataset_filenames[train_length:]
        train_length -= int(train_length * 0.2)
        train_dataset = train_dataset_filenames[:train_length]
        val_dataset = train_dataset_filenames[train_length:]

        self.dataset = flickr_dataset
        self.train_filenames = train_dataset
        self.val_filenames = val_dataset
        self.test_filenames = test_dataset
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.transform = transform
        self.num_workers = num_workers
        self.predict_file = predict_file

    def setup(self, stage= None):

        if stage == 'fit' or stage is None:
            self.train_dataset = FlickrDataset(self.train_filenames, self.dataset, self.transform)
            self.val_dataset = FlickrDataset(self.val_filenames, self.dataset, self.transform)

        if stage == 'validate':
            self.val_dataset = FlickrDataset(self.val_filenames, self.dataset, self.transform)

        if stage == 'test':
            self.test_dataset = FlickrDataset(self.test_filenames, self.dataset, self.transform)

        if stage == 'predict':
            if self.predict_file is None:
                predict_dataset = FlickrDataset(self.test_filenames, self.dataset, self.transform)
            elif self.predict_file == 'train':
                predict_dataset = FlickrDataset(self.train_filenames, self.dataset, self.transform)
            elif self.predict_file == 'valid':
                predict_dataset = FlickrDataset(self.val_filenames, self.dataset, self.transform)
            elif self.predict_file == 'test':
                predict_dataset = FlickrDataset(self.test_filenames, self.dataset, self.transform)
            self.predict_dataset = predict_dataset

    def _set_image_feature_extractor(self, image_feature_extractor):
        self.image_feature_extractor = image_feature_extractor

    def _set_decoder_tokenizer(self, decoder_tokenizer):
        self.decoder_tokenizer = decoder_tokenizer

    def _set_tokens(self, model):
        self.start_token = model.start_token
        self.end_token = model.end_token

    def set_model_variables(self, model):

        self._set_image_feature_extractor(model.image_feature_extractor)
        self._set_decoder_tokenizer(model.decoder_tokenizer)
        #self._set_tokens(model)

    def tokenize_data(self, batch_data):

        image_tensors = [t[0] for t in batch_data]
        captions = [t[1] for t in batch_data]

        image_encodings = self.image_feature_extractor(image_tensors, return_tensors='pt').pixel_values
        labels = self.decoder_tokenizer(
            captions,
            padding="longest",
            truncation=True,
            #max_length=24,
            return_tensors="pt",
        )

        return image_encodings, labels

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.tokenize_data
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.tokenize_data
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.tokenize_data
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            shuffle=False,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.tokenize_data,
        )
