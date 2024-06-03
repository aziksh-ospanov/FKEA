# FKEA Scores and Clustering

This repository contains a demo script for performing image and text clustering using the `FKEA_Evaluator` class from the FKEA package. The script demonstrates how to set up and run the FKEA evaluator for clustering tasks and compute diversity scores for both image and text datasets.

## Requirements

- Python 3.x
- FKEA package
- pickle
- torch
- torchvision (for image processing)
- openai package (for text-embedding-3 model)
- transformers, datasets

## Image Clustering & Scores
The script performs image clustering using a specified feature extractor and evaluates the diversity of the clustering. Set the parameters for the FKEA evaluator:

num_samples: Number of samples to use for clustering.
sigma: Gaussian Kernel Bandwidth Parameter.
rff_dim: Dimensionality for random Fourier features (note that final dimension of tge matrix is two times this number).
Initialize the FKEA_Evaluator with the desired settings and set the feature extractor.
Create an ImageFilesDataset with the path to your image dataset.

List of supported embeddings:
- InceptionV3
- DinoV2
- CLIP
- SwAV
- Colored Pixel: flattened pixel values of an RGB image. Used for MNIST-like image evaluation
- Mono Pixel: flattened pixel values of a GREYSCALE image. Used for MNIST-like image evaluation

Example code as follows:
```python
# Parameters
num_samples = 100
sigma = 20
rff_dim = 4000
batchsize = 32
result_name = 'your_result_name'
fe = 'dinov2' # feature embedding for img data

# Image Clustering & Scores
FKEA = FKEA_Evaluator(logger_path='./logs', batchsize=batchsize, sigma=sigma, num_samples=num_samples, result_name=result_name, rff_dim=rff_dim)
FKEA.set_feature_extractor(fe, save_path='./save')

img_pth = '/path/to/dataset'
dataset = ImageFilesDataset(img_pth, name=result_name, extension='JPEG')

FKEA.rff_clustering_modes_of_dataset(dataset)
alpha = [1, 2, 'inf']
fkea_approximated_scores = FKEA.f_diversity_scores(dataset, alpha=alpha)
true_scores = FKEA.non_f_diversity_scores(dataset, alpha=alpha)
```


## Text Clustering & Scores
The script also performs text clustering and evaluates the diversity of the clustering using a specified feature extractor for text embeddings. Set the parameters for the FKEA evaluator as in the image clustering section.

Initialize the FKEA_Evaluator and set the feature extractor. Load the text data as a folder of text files, the script will treat each paragraph as a sample.

List of supported embeddings:
- BERT: pretrained 'bert-base-uncased' offered by transformers
- GPT: text-embedding-3-large model offered by OpenAI

```python
result_name = 'your_result_name'
fe = 'gpt-large'

# Text Clustering & Scores
FKEA = FKEA_Evaluator(logger_path='./logs', batchsize=batchsize, sigma=sigma, num_samples=num_samples, result_name=result_name, rff_dim=rff_dim, 
                      api_key='your api key')
FKEA.set_feature_extractor(fe, save_path='./save')

txt_path = "/path/to/dataset"

dataset = TextFilesDataset(txt_path).dataset
    
FKEA.rff_text_clustering_modes_of_dataset(dataset)
alpha = 2
fkea_approximated_scores = FKEA.f_diversity_scores(dataset, alpha=alpha)
true_scores = FKEA.non_f_diversity_scores(dataset, alpha=alpha)
```

## Loading features directly 
The script allows to directly perform clustering/score computation if data is feed as list of embeddings and corresponding labels. Below is an example usage for text data.
```python
FKEA = FKEA_Evaluator(logger_path='./logs', batchsize=batchsize, sigma=sigma, num_samples=num_samples, result_name=result_name, rff_dim=rff_dim, 
                          api_key='your api key')
FKEA.set_feature_extractor(fe, save_path='./save')

txt_path = r"/path/to/dataset/file.txt"
with open(txt_path, 'r', encoding='utf-8') as f:
  texts = f.readlines()

embedding_pth = 'embedding/path/file.pickle' # In this example, we store embeddings as pickle file, should be convertible to torch tensor
with open(embedding_pth, 'rb') as f:
  embeddings = pickle.load(f)
assert len(texts) == embeddings.shape[0]
assert embeddings.shape[0] >= num_samples

FKEA.rff_text_clustering_modes_of_dataset(texts, test_feats = embeddings)
fkea_approximated_scores = FKEA.f_diversity_scores(texts, test_feats=embeddings)
true_scores = FKEA.non_f_diversity_scores(texts, test_feats=embeddings)
```

## Notes
- Ensure you update the paths and filenames to match your local environment.
- Adjust the result_name, fe, and other parameters as needed for your specific use case.
- The script assumes that the embeddings (test_feats parameter) are list-like object convertable to torch tensor.
- Please use your personal OpenAI api key to access gpt embeddings


This `README.md` file provides a clear and concise guide for users to understand and run the demo script, including installation instructions, usage examples, and parameter explanations. Adjust the paths and filenames in the script to match your specific environment and dataset.

