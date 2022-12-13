from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision.transforms import transforms
from collections import OrderedDict
import random
import pytorch_lightning as pl
from utils import calculate_number_of_mask_tokens
from masking_stratergies import epoch_aware_mask, text_infilling
import json

class VQATestDataset(Dataset):

    def __init__(self,
                 questions_file,
                 transform=None,
                 ):

        self.image_dir = 'datasets/vqa_images/test/'
        self.pairs = self.load_dataset(questions_file)
    
    def load_dataset(self,question_json):
      f = open(question_json)
      questions = json.load(f)

      questions_dict = {}
      images = {}
      grand_dict = []

      for val in questions['questions']:
        image_id = val['image_id']
        question = val['question']
        question_id = val['question_id']

        image_name = 'COCO_test2015_'+'0'*(12-len(str(image_id)))+str(image_id)+'.jpg'
        questions_dict[question_id] = {'Question':question}
        if not image_name in images:
          images[image_name] = []
        images[image_name].append(question_id)
      
      for v in images:
        image_name = v
        question_ids = images[v]
        for q in question_ids:
          obj = questions_dict[q]
          q = obj['Question']
          grand_dict.append([image_name,q,' '])
      
      return grand_dict

    def __len__(self):

        return len(self.pairs)

    def __getitem__(self, index):

        question = self.pairs[index][1]
        answer = self.pairs[index][2]
        image_filename = self.pairs[index][0]
        img = Image.open(self.image_dir + image_filename)
        if self.transform:
            img = self.transform(img)
        return img, question,answer, image_filename

class VQAPredictionDataset(Dataset):

    def __init__(self,
                 questions_file,
                 answers_file,
                 transform=None,
                 ):

        self.image_dir = 'datasets/vqa_images/val/'
        self.pairs = self.load_dataset(questions_file,answers_file)
        self.transform = transform
    def load_dataset(self,question_json,answer_json):
      f = open(answer_json)
      answers = json.load(f)
      f = open(question_json)
      questions = json.load(f)

      questions_dict = {}
      images = {}
      grand_dict = []

      for val in questions['questions']:
        image_id = val['image_id']
        question = val['question']
        question_id = val['question_id']

        image_name = 'COCO_val2014_'+'0'*(12-len(str(image_id)))+str(image_id)+'.jpg'
        questions_dict[question_id] = {'Question':question}
        if not image_name in images:
          images[image_name] = []
        images[image_name].append(question_id)
      
      for val in answers['annotations']:
        question_id = val['question_id']
        questions_dict[question_id]['answers'] = val['answers'][0]['answer']

      for v in images:
        image_name = v
        question_ids = images[v]
        for q in question_ids:
          obj = questions_dict[q]
          q = obj['Question']
          a = obj['answers']
          grand_dict.append([image_name,q,a])
      
      return grand_dict

    def __len__(self):

        return len(self.pairs)

    def __getitem__(self, index):

        question = self.pairs[index][1]
        answer = self.pairs[index][2]
        image_filename = self.pairs[index][0]
        img = Image.open(self.image_dir + image_filename)
        print(self.image_dir + image_filename,question,answer)
        if self.transform:
            img = self.transform(img)
        return img, question, answer, image_filename


class VQADataset(Dataset):

    def __init__(self,
                 questions_file,
                 answers_file,
                 split='train',
                 transform=None,
                 ):

        self.image_dir = 'datasets/vqa_images/'+split+'/'
        self.prefix = None
        if split == 'train':
            self.prefix = 'COCO_train2014_'
        elif split == 'val':
            self.prefix = 'COCO_val2014_'
        else:
            self.prefix = 'COCO_test2014_'

        self.pairs = self.load_dataset(questions_file,answers_file)
        self.transform = transform
    
    def load_dataset(self,question_json,answer_json):
      f = open(answer_json)
      answers = json.load(f)
      f = open(question_json)
      questions = json.load(f)

      questions_dict = {}
      images = {}
      grand_dict = []

      for val in questions['questions']:
        image_id = val['image_id']
        question = val['question']
        question_id = val['question_id']

        image_name = self.prefix+'0'*(12-len(str(image_id)))+str(image_id)+'.jpg'
        
        questions_dict[question_id] = {'Question':question}
        if not image_name in images:
          images[image_name] = []
        images[image_name].append(question_id)
      
      for val in answers['annotations']:
        question_id = val['question_id']
        print("DEBUG:",val['answers'])
        for ans in val['answers']:
            if ans['answer_confidence'] == 'yes':
                questions_dict[question_id]['answers'] = ans['answer']
                break
            else:
                questions_dict[question_id]['answers'] = val['answers'][0]['answer']

      for v in images:
        image_name = v
        question_ids = images[v]
        for q in question_ids:
          obj = questions_dict[q]
          q = obj['Question']
          a = obj['answers']
          grand_dict.append([image_name,q,a])
      
      return grand_dict

    def __len__(self):

        return len(self.pairs)

    def __getitem__(self, index):

        question = self.pairs[index][1]
        answer = self.pairs[index][2]
        image_filename = self.pairs[index][0]
        img = Image.open(self.image_dir + image_filename).convert('RGB')
        if self.transform:
            img = self.transform(img)
        #print("GET",question,answer)
        return img, question, answer,image_filename

class VQADatasetModule(pl.LightningDataModule):

    # def _load_dataset(self):

    #     data = open('datasets/flickr30k/results_20130124.token', 'r').read().splitlines()
    #     questions = None
    #     answers = None
    #     captions = [x.split('\t')[1] for x in data]
    #     image_filenames = [x.split('#')[0] for x in data]
    #     image_captions = OrderedDict()
    #     for image, caption in zip(image_filenames, captions):
    #         if image not in image_captions:
    #             image_captions[image] = []
    #         image_captions[image].append(caption)
    #     return image_captions

    def __init__(self,
                 train_batch_size=16,
                 eval_batch_size=16,
                 transform=transforms.PILToTensor(),
                 num_workers=12,
                 ):

        super().__init__()
        
        
        self.train_questions = './vqa_jsons/v2_OpenEnded_mscoco_train2014_questions.json'
        self.val_questions = './vqa_jsons/v2_OpenEnded_mscoco_val2014_questions.json'
        self.test_questions = './vqa_jsons/v2_OpenEnded_mscoco_test2015_questions.json'

        self.train_answers = './vqa_jsons/v2_mscoco_train2014_annotations.json'
        self.val_answers = './vqa_jsons/v2_mscoco_val2014_annotations.json'
        

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.transform = transform
        self.num_workers = num_workers

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            self.train_dataset = VQADataset(self.train_questions,self.train_answers,'train', self.transform)
            self.val_dataset = VQADataset(self.val_questions,self.val_answers,'val', self.transform)

        if stage == 'validate':
            self.val_dataset = VQADataset(self.val_questions,self.val_answers,'val', self.transform)

        if stage == 'test':
            self.test_dataset = VQADataset(self.val_questions,self.val_answers,'val', self.transform)

        if stage == 'predict':
            self.predict_file='val'
            self.predict_dataset = VQAPredictionDataset(self.val_questions,self.val_answers, self.transform)

    def _set_image_feature_extractor(self, image_feature_extractor):
        self.image_feature_extractor = image_feature_extractor

    def _set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _set_tokens(self, model):
        self.start_token = model.start_token
        self.end_token = model.end_token

    def set_model_variables(self, model):

        self._set_image_feature_extractor(model.image_feature_extractor)
        self._set_tokenizer(model.tokenizer)

    def tokenize_data(self, batch_data):

        image_tensors = [t[0] for t in batch_data]
        questions = [t[1] for t in batch_data]
        answers = [t[2] for t in batch_data]
        filenames = [t[3] for t in batch_data]
        #for img in image_tensors:
            #print(img.shape)
        #print("-------")
        print(questions,answers,filenames)
        try:
            image_encodings = self.image_feature_extractor(image_tensors, return_tensors='pt').pixel_values
        except Exception as e:
            print(e,filenames)
        #print(image_encodings,questions,answers)
        question_encodings = self.tokenizer(
            questions,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        answer_encodings = self.tokenizer(
            answers,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        return image_encodings, question_encodings, answer_encodings

    def prediction_tokenization(self, batch_data):

        '''

        :param batch_data: Each sample is (Image Tensor, Captions, Filename)
        :return: (Image Pixel Values, Input Text Encodings, Label Text Encodings, Filename)
        '''

        image_tensors = [t[0] for t in batch_data]
        questions = [t[1] for t in batch_data]
        answers = [t[2] for t in batch_data]
        image_filenames = [t[3] for t in batch_data]

        image_encodings = self.image_feature_extractor(image_tensors, return_tensors='pt').pixel_values

        question_encodings = self.tokenizer(
            questions,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        print(len(image_filenames))
        return image_encodings, question_encodings, answers, image_filenames

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
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=self.prediction_tokenization,
        )
# class VQADatasetModule(pl.LightningDataModule):

    # def _load_dataset(self):

    #     data = open('datasets/flickr30k/results_20130124.token', 'r').read().splitlines()
    #     questions = None
    #     answers = None
    #     captions = [x.split('\t')[1] for x in data]
    #     image_filenames = [x.split('#')[0] for x in data]
    #     image_captions = OrderedDict()
    #     for image, caption in zip(image_filenames, captions):
    #         if image not in image_captions:
    #             image_captions[image] = []
    #         image_captions[image].append(caption)
    #     return image_captions

    # def __init__(self,
    #              train_batch_size=16,
    #              eval_batch_size=16,
    #              transform=transforms.PILToTensor(),
    #              num_workers=12,
    #              predict_file=None,
    #              multi_modal=False,
    #              mask=False,
    #              ):

    #     super().__init__()
        
        
    #     self.train_questions = 'v2_OpenEnded_mscoco_train2014_questions.json'
    #     self.val_questions = 'v2_OpenEnded_mscoco_val2014_questions.json'
    #     self.test_questions = 'v2_OpenEnded_mscoco_test2015_questions.json'

    #     self.train_answers = 'v2_mscoco_val2014_annotations.json'
    #     self.val_answers = 'v2_mscoco_val2014_annotations.json'
        

    #     self.train_batch_size = train_batch_size
    #     self.eval_batch_size = eval_batch_size
    #     self.transform = transform
    #     self.num_workers = num_workers
    #     self.predict_file = predict_file
    #     self.multi_modal = multi_modal
    #     self.mask = mask

    # def setup(self, stage=None):

    #     if stage == 'fit' or stage is None:
    #         self.train_dataset = VQADataset(self.train_questions,self.train_answers, self.transform)
    #         self.val_dataset = VQADataset(self.val_questions,self.val_answers, self.transform)

    #     if stage == 'validate':
    #         self.val_dataset = VQADataset(self.val_questions,self.val_answers, self.transform)

    #     if stage == 'test':
    #         self.val_dataset = VQADataset(self.val_questions,self.val_answers, self.transform)

    #     if stage == 'predict':
    #         self.val_dataset = VQAPredictionDataset(self.val_questions,self.val_answers, self.transform)

    # def _set_image_feature_extractor(self, image_feature_extractor):
    #     self.image_feature_extractor = image_feature_extractor

    # def _set_tokenizer(self, tokenizer):
    #     self.tokenizer = tokenizer

    # def _set_tokens(self, model):
    #     self.start_token = model.start_token
    #     self.end_token = model.end_token

    # def set_model_variables(self, model):

    #     self._set_image_feature_extractor(model.image_feature_extractor)
    #     self._set_tokenizer(model.tokenizer)

    # def tokenize_data(self, batch_data):

    #     image_tensors = [t[0] for t in batch_data]
    #     questions = [t[1] for t in batch_data]
    #     answers = [t[2] for t in batch_data]

    #     image_encodings = self.image_feature_extractor(image_tensors, return_tensors='pt').pixel_values
    #     question_encodings = self.tokenizer(
    #         questions,
    #         padding="longest",
    #         truncation=True,
    #         return_tensors="pt",
    #     )
    #     answer_encodings = self.tokenizer(
    #         answers,
    #         padding="longest",
    #         truncation=True,
    #         return_tensors="pt",
    #     )
    #     if self.multi_modal:
    #         # if self.mask:
    #         #     if self.mask == 'empty':
    #         #         input_text = ['' for _ in batch_data]
    #         #     elif self.mask == 'epoch_aware_mask':
    #         #         input_text = [epoch_aware_mask(self.epoch, x) for x in captions]
    #         #     elif self.mask == 'text_infilling':
    #         #         input_text = [text_infilling(x) for x in captions]
    #         #     input_text_encodings = self.tokenizer(input_text,
    #         #                                           padding='longest',
    #         #                                           return_tensors='pt')
    #         # else:
    #         #     input_text_encodings = caption_encodings
    #         # labels = caption_encodings
    #         return image_encodings, question_encodings, answer_encodings
    #     else:
    #         return image_encodings, question_encodings

    # def prediction_tokenization(self, batch_data):

    #     '''

    #     :param batch_data: Each sample is (Image Tensor, Captions, Filename)
    #     :return: (Image Pixel Values, Input Text Encodings, Label Text Encodings, Filename)
    #     '''

    #     image_tensors = [t[0] for t in batch_data]
    #     captions = [t[1] for t in batch_data]
    #     image_filenames = [t[2] for t in batch_data]

    #     image_encodings = self.image_feature_extractor(image_tensors, return_tensors='pt').pixel_values
    #     if self.multi_modal:
    #         if self.mask == 'empty':
    #             input_text = ['' for _ in batch_data]
    #             input_text_encodings = self.tokenizer(input_text,
    #                                                   return_tensors='pt')
    #         elif self.mask == 'epoch_aware_mask':
    #             input_text = [' '.join(['<mask>']*self.median_length) for _ in batch_data]
    #             input_text_encodings = self.tokenizer(input_text,
    #                                                   return_tensors='pt')
            
    #         elif self.mask == 'text_infilling':
    #             input_text = ['<mask>' for _ in batch_data]
    #             input_text_encodings = self.tokenizer(input_text,
    #                                                   return_tensors='pt')

    #         return image_encodings, input_text_encodings, captions, image_filenames
    #     else:
    #         return image_encodings, captions, image_filenames

    # def train_dataloader(self):
    #     return torch.utils.data.DataLoader(
    #         self.train_dataset,
    #         batch_size=self.train_batch_size,
    #         shuffle=True,
    #         num_workers=self.num_workers,
    #         collate_fn=self.tokenize_data
    #     )

    # def val_dataloader(self):
    #     return torch.utils.data.DataLoader(
    #         self.val_dataset,
    #         batch_size=self.eval_batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         collate_fn=self.tokenize_data
    #     )

    # def test_dataloader(self):
    #     return torch.utils.data.DataLoader(
    #         self.test_dataset,
    #         batch_size=self.eval_batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         collate_fn=self.tokenize_data
    #     )

    # def predict_dataloader(self):
    #     return torch.utils.data.DataLoader(
    #         self.predict_dataset,
    #         shuffle=False,
    #         batch_size=self.eval_batch_size,
    #         num_workers=self.num_workers,
    #         collate_fn=self.prediction_tokenization,
    #     )
