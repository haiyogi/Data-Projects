# Fake Review Detection Using Machine Learning and RoBERTa-Based Deep Learning

## Project Overview

This project develops and evaluates multiple machine-learning and deep-learning approaches for detecting whether an online product review is human-written or computer-generated.

The complete workflow includes data-quality analysis, duplicate detection, leakage prevention, exploratory data analysis, text preprocessing, TF-IDF feature extraction, traditional machine-learning models, recurrent deep-learning models using frozen RoBERTa features, end-to-end RoBERTa fine-tuning, and final performance comparison.

The notebook is designed to run in Google Colab or another Jupyter environment. When a CUDA-compatible GPU is available, the deep-learning models automatically use the GPU and mixed-precision training.

## Research Objective

The main objective is to classify product reviews into two classes:

```text
OR = Original / Human-Written Review
CG = Computer-Generated Review
```

For modelling, the labels are encoded as:

```text
OR = 0
CG = 1
```

The study investigates whether contextual transformer representations provide better fake-review detection performance than traditional TF-IDF representations.

## Dataset

The notebook expects the following dataset file:

```text
fake reviews dataset.csv
```

The original dataset contains:

```text
Rows    : 40,432
Columns : 4
```

The original columns are:

| Column | Description |
| category | Product category associated with the review |
| rating | Product rating |
| label | Review class |
| text_ | Full review text |

The dataset is approximately balanced between original reviews and computer-generated reviews.

## Project Workflow

```text
Dataset Loading
        ↓
Data Quality Analysis
        ↓
Duplicate and Label Integrity Checks
        ↓
Target Encoding
        ↓
Text Quality Analysis
        ↓
Text Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Duplicate and Leakage Removal
        ↓
Train / Validation / Test Split
        ↓
TF-IDF Feature Extraction
        ↓
Decision Tree
        ↓
Random Forest
        ↓
Gradient Boosting
        ↓
RoBERTa + Bidirectional LSTM
        ↓
RoBERTa + Bidirectional GRU
        ↓
Fine-Tuned RoBERTa
        ↓
Final Model Comparison
```

## Environment Setup

The project uses the following main libraries:

```text
NumPy
Pandas
Matplotlib
Seaborn
WordCloud
Scikit-learn
PyTorch
Transformers
Accelerate
```

The notebook automatically installs the additional transformer and word-cloud libraries using:

```text
transformers
accelerate
wordcloud
```

## Reproducibility

A fixed random seed of `42` is used throughout the project. The seed is applied to Python, NumPy, PyTorch, CUDA, and DataLoader shuffling. CuDNN deterministic behaviour is also enabled where possible.

## Data Quality and Leakage Prevention

Before model development, the dataset is examined for dimensions, data types, missing values, unique values, duplicates, repeated review text, conflicting labels, review length, URLs, email addresses, digits, punctuation, and uppercase words.

The original dataset contains 12 fully duplicated rows, which are removed before further analysis. No conflicting labels are found among duplicated review texts.

Leakage control includes exact duplicate removal, canonical-text deduplication, cleaned-text deduplication, removal of empty cleaned reviews, and a final overlap audit across original, canonical, classical-cleaned, and RoBERTa-preprocessed text. The final training, validation, and test sets contain zero overlap across these representations.

## Text Preprocessing

Classical machine-learning text is processed using HTML entity decoding, Unicode normalisation, URL and email replacement, HTML-tag removal, lowercasing, symbol removal, and whitespace normalisation. Numbers and stopwords are retained because they may contain useful stylistic information.

RoBERTa uses minimal cleaning: HTML entity decoding, Unicode normalisation, HTML-tag removal, and whitespace normalisation. Capitalisation, punctuation, numbers, sentence structure, contractions, and original word order are preserved.

## Exploratory Data Analysis

The notebook includes class, category, rating, review-length, category-by-class, and top TF-IDF feature plots, along with separate word clouds for real and computer-generated reviews. Linguistic and stylistic measures such as lexical diversity, average word and sentence length, punctuation, uppercase, exclamation, and question-mark ratios are calculated for exploratory analysis and are not directly supplied to predictive models.

## Final Dataset Split

The final cleaned dataset is split into 70% training, 15% validation, and 15% testing, stratified by target class and product category:

| Split | Samples |

| Training | 28,269 |
| Validation | 6,058 |
| Testing | 6,058 |

The validation set is used for deep-learning early stopping and model selection. Traditional models are trained on the training set and evaluated on the test set.

## TF-IDF Feature Extraction

Traditional models use a TF-IDF vectoriser fitted only on the training data.

```text
Maximum features          : 10,000
N-gram range              : 1–2
Minimum document frequency: 2
Maximum document frequency: 0.95
Sublinear term frequency  : Enabled
Normalisation             : L2
Data type                 : float32
```

## Machine-Learning Models

The traditional models are Decision Tree, Random Forest, and Gradient Boosting. Their main configurations are:

| Model | Main configuration |
| --- | --- |
| Decision Tree | Gini, max depth 50, min split 5, min leaf 2, balanced class weights |
| Random Forest | 300 trees, square-root features, min split 3, balanced class weights, parallel training |
| Gradient Boosting | 150 estimators, learning rate 0.05, max depth 3, min split 5, min leaf 2, subsample 0.8 |

All models use random seed 42.

## Evaluation Metrics

All models are evaluated using accuracy, weighted precision, weighted recall, weighted F1-score, confusion matrices, and detailed classification reports. Metrics are reported as percentages. Weighted averaging accounts for the number of samples in each class.

## Deep-Learning Setup

PyTorch automatically selects CUDA when a compatible GPU is available and otherwise falls back to CPU. GPU name, memory, and selected device are printed. Mixed-precision training is enabled when CUDA is available to reduce memory usage and accelerate training.

The transformer models use `FacebookAI/roberta-base`, whose hidden representation size is 768. RoBERTa tokenisation is analysed using candidate sequence lengths of 64, 128, 192, 256, 384, and 512; the final maximum sequence length is 256 tokens. The transformer DataLoader uses batch size 16, shuffling only the training data.

## RoBERTa + Bidirectional LSTM

The first recurrent model uses frozen RoBERTa token representations followed by a one-layer bidirectional LSTM, dense layer, dropout, and binary classification head. The recurrent hidden size is 128 and dropout is 0.30. Padding tokens are excluded using packed sequences.

Training uses Cross-Entropy Loss, AdamW with learning rate 0.001 and weight decay 0.0001, a maximum of 10 epochs, patience-3 early stopping, and gradient clipping at 1.0. RoBERTa remains frozen.

| Metric | Result |
| Accuracy | 95.97% |
| Weighted Precision | 95.99% |
| Weighted Recall | 95.97% |
| Weighted F1-Score | 95.97% |

## RoBERTa + Bidirectional GRU

The second recurrent model replaces the LSTM with a one-layer bidirectional GRU using hidden size 128 and dropout 0.30. It uses the same loss, optimiser, clipping, and early-stopping configuration.

| Metric | Result |
| --- | ---: |
| Accuracy | 97.67% |
| Weighted Precision | 97.68% |
| Weighted Recall | 97.67% |
| Weighted F1-Score | 97.67% |

The BiGRU produces better final performance than the BiLSTM in this experiment.

## Fine-Tuned RoBERTa

The final model performs full end-to-end RoBERTa fine-tuning. All transformer parameters are trainable, with approximately 124.65 million trainable parameters.

```text
Base model              : RoBERTa-base
Maximum sequence length : 256
Physical batch size     : 8
Gradient accumulation   : 2
Effective batch size    : 16
Optimizer               : AdamW
Learning rate           : 2e-5
Weight decay             : 0.01
Maximum epochs           : 5
Early stopping patience  : 2
Warm-up                  : 10%
Gradient clipping        : 1.0
```

| Metric | Result |

| Accuracy | 98.04% |
| Weighted Precision | 98.05% |
| Weighted Recall | 98.04% |
| Weighted F1-Score | 98.04% |

The classifier head is newly initialised because the pretrained checkpoint was originally trained for language modelling. This loading message is normal and does not indicate an implementation error.

## Final Model Results

| Model | Accuracy (%) | Weighted Precision (%) | Weighted Recall (%) | Weighted F1-Score (%) |

| Decision Tree | 70.43 | 75.59 | 70.43 | 69.75 |
| Random Forest | 80.77 | 82.38 | 80.77 | 80.77 |
| Gradient Boosting | 71.92 | 76.14 | 71.92 | 71.48 |
| RoBERTa + BiLSTM | 95.97 | 95.99 | 95.97 | 95.97 |
| RoBERTa + BiGRU | 97.67 | 97.68 | 97.67 | 97.67 |
| Fine-Tuned RoBERTa | 98.04 | 98.05 | 98.04 | 98.04 |

Transformer-based models substantially outperform the traditional TF-IDF-based models. Random Forest is the strongest traditional model, RoBERTa + BiGRU outperforms RoBERTa + BiLSTM, and Fine-Tuned RoBERTa achieves the strongest overall result.

## Early Stopping and Memory Management

Validation loss is monitored after every epoch for all deep-learning models, and the best checkpoint is restored before final testing. This limits overfitting when training loss continues improving while validation loss worsens.

RoBERTa features are generated batch-by-batch rather than precomputed for the entire dataset. Before full fine-tuning, frozen recurrent models and the frozen encoder are removed from memory and the CUDA cache is cleared when available.

## Running in Google Colab

1. Open `Fake_Review_Detection.ipynb` in Google Colab.
2. Upload `fake reviews dataset.csv`.
3. Enable GPU acceleration through **Runtime → Change runtime type → GPU**.
4. Run all notebook cells from top to bottom.

The notebook handles library installation, dataset analysis, preprocessing, visualisation, deduplication, leakage checking, splitting, TF-IDF extraction, traditional model training, GPU detection, transformer tokenisation, recurrent training, fine-tuning, evaluation, and final comparison.

## Recommended Project Structure

```text
Fake-Review-Detection/
│
├── Fake_Review_Detection.ipynb
├── fake reviews dataset.csv
└── README.md
```

## Key Methodological Strengths

- Exact and representation-level duplicates are removed before final splitting.
- Cross-split text leakage is explicitly tested.
- TF-IDF is fitted only on training data.
- The same final partitions are retained for all model families.
- Traditional models are evaluated on the test set without validation-set selection.
- The validation set is used for deep-learning early stopping.
- Recurrent models use complete token-level contextual RoBERTa sequences.
- Fine-tuned RoBERTa uses a small learning rate, warm-up, gradient clipping, accumulation, and early stopping.
- Precision, recall, and F1 values use weighted averaging.
- Random seed 42 is maintained throughout the project.

## Reproducibility Summary

```text
Random seed               : 42
Split                     : 70 / 15 / 15
TF-IDF maximum features   : 10,000
TF-IDF n-gram range       : 1–2
RoBERTa checkpoint        : FacebookAI/roberta-base
Maximum sequence length   : 256
LSTM hidden size          : 128
GRU hidden size           : 128
Recurrent dropout         : 0.30
LSTM/GRU maximum epochs    : 10
Fine-tuning maximum epochs : 5
```

The actual number of deep-learning epochs may be smaller because of early stopping.

## Final Conclusion

This project compares traditional machine-learning models with RoBERTa-based deep-learning models for detecting computer-generated product reviews. Random Forest achieves the strongest classical result, while transformer-based models provide substantially higher performance by capturing contextual and sequential information.

Fine-Tuned RoBERTa achieves the best overall performance, with 98.04% accuracy and a 98.04% weighted F1-score. RoBERTa + BiGRU achieves a close weighted F1-score of 97.67%, while RoBERTa + BiLSTM reaches 95.97%.

Fine-Tuned RoBERTa is the strongest choice when maximum predictive performance and sufficient computational resources are available. RoBERTa + BiGRU provides a strong efficiency-performance trade-off because the RoBERTa encoder remains frozen and substantially fewer parameters require optimisation. This can reduce training time, GPU memory requirements, and computational cost while retaining performance close to full fine-tuning.
