# Multi-Modal Controlled Image To Text Generation

The project implements a BART model and alters the architecture 
to input image along-with text and generate corresponding text. 
The project also implements different masking strategies while training the model on the dataset.

We take a pre-trained BART and fine-tune it for image-caption generation (Flickr30k Dataset). 
We introduce masking strategies while fine-tuning to allow controlled image-caption generation during inference.

The image-caption fine-tuned model is then trained on VQA dataset.

We analyse the performance of the models trained using our masking strategies.

## Data Downloading

The datasets should be downloaded in datasets folder in root directory

### Flickr30K
The dataset is downloaded by filling the form at http://hockenmaier.cs.illinois.edu/DenotationGraph/. We use the file 'results_20130124.token' for textual data.
The pre-processing is done by dataset.py which need not be called explicitly.

## Training Baselines
Baseline Image-Caption Model is our BART Model Architecture that has been trained without masking strategy. 
It takes an image as input and generates corresponding caption.

Command:
```python
python experiments.py --model_name MultiModal --dataset flickr --multi_modal False --mask empty
```

## Training Experiments

### Training Model for Image-Caption Generation
We pass masked image caption as input along with image
We experiment with 2 masking strategies:
1. Epoch Aware Mask:
   Replace tokens with mask and increase the number of masks with epochs
    ```python
    python experiments.py --model_name MultiModal --dataset flickr --multi_modal True --mask epoch_aware_mask
   ```
2. Text Infilling:
   Replace contiguous tokens with single mask
    ```python
    python experiments.py --model_name MultiModal --dataset flickr --multi_modal True --mask text_infilling
   ```

### Training Model for Visual Question Answering
1. Train model directly on VQA dataset
   ```python
    python experiments.py --model_name MultiModal --dataset vqa
   ```
2. Use baseline model-ckpt trained on Image-Caption Dataset with different masking strategy and fine-tune for VQA
    ```python
    python experiments.py --model_ckpt checkpoint_location --model_name MultiModal --dataset vqa
   ```

## Evaluating Model

### Flickr30K Dataset
We calculate the BLEU and BERT Scores
1. Evaluate Baseline 
   ```python
    python experiments.py --model_ckpt checkpoint_location --model_name MultiModal --dataset flickr --multi_modal False --mask empty --predict test
    ```
2. Evaluate Epoch Aware Mask Strategy
   ```python
    python experiments.py --model_ckpt checkpoint_location --model_name MultiModal --dataset flickr --multi_modal True --mask epoch_aware_mask --predict test
    ```
3. Evaluate Text Infilling Strategy
   ```python
    python experiments.py --model_ckpt checkpoint_location --model_name MultiModal --dataset flickr --multi_modal True --mask text_infilling --predict test
    ```

### VQA Dataset
We calculate the BLEU, METEOR and ROUGE Scores
1. Evaluate Model trained only on VQA
   ```python
    python experiments.py --model_ckpt checkpoint_location --model_name MultiModal --dataset vqa --predict test
    ```
2. Evaluate Baseline Model trained on Flickr30K dataset and then fine-tuned on VQA
   ```python
    python experiments.py --model_ckpt checkpoint_location --model_name MultiModal --dataset vqa --predict test
    ```
3. Evaluate Text Infilling Strategy Model trained on Flickr30K dataset and then fine-tuned on VQA
   ```python
    python experiments.py --model_ckpt checkpoint_location --model_name MultiModal --dataset vqa --predict test
    ```
