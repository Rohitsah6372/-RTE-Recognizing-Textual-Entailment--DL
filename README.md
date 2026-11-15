# RTE – Recognizing Textual Entailment using Deep Learning

A Deep Learning project developed as part of **CS6315E – Neural Networks and Deep Learning**.

The main objective of this project is to build a model that can understand the relationship between two sentences and determine whether one sentence logically follows from the other.

We use **DeBERTa-v3-base** as our main architecture and explore different fine-tuning strategies to improve classification performance while considering the number of trainable parameters.

---

## 1. About the Project

Recognizing Textual Entailment (RTE) is a Natural Language Processing task that focuses on understanding the relationship between two pieces of text.

Given a premise and a hypothesis, the model needs to determine whether the hypothesis can be logically inferred from the premise.

For example:

| Premise | Hypothesis | Prediction |
|---|---|---|
| A man is playing football on the ground. | A person is playing a sport. | Entailment |
| A woman is cooking food in the kitchen. | A man is driving a car. | Not Entailment |

The project uses a transformer-based language model to learn these relationships rather than relying only on traditional text-matching techniques.

## 2. Project Objectives

The main objectives of this project are:

- Understand and implement the Recognizing Textual Entailment task.
- Explore transformer-based architectures for text classification.
- Fine-tune DeBERTa-v3-base on the RTE dataset.
- Experiment with hyperparameter optimization using Optuna.
- Explore parameter-efficient fine-tuning using LoRA.
- Compare different training strategies using evaluation metrics.
- Study the relationship between model performance and the number of trainable parameters.

## 3. Dataset

The project uses an RTE dataset containing premise-hypothesis sentence pairs.

Each record contains the following fields:

| Column | Description |
|---|---|
| `premise` | The original statement or context. |
| `hypothesis` | The statement that needs to be evaluated. |
| `label` | The relationship between the two sentences. |
| `idx` | Unique record identifier. |

### Dataset labels

- **Entailment:** The hypothesis logically follows from the premise.
- **Not Entailment:** The hypothesis does not logically follow from the premise.

The dataset is provided in JSONL format.

### Dataset preparation

The preprocessing workflow includes:

- Loading the training and validation datasets.
- Checking missing values and duplicate sentence pairs.
- Removing duplicate training pairs.
- Converting labels into numerical representations.
- Tokenizing premise-hypothesis pairs.
- Preparing the data for transformer-based classification.

## 4. Model Architecture

We selected **DeBERTa-v3-base** from Microsoft as the main architecture.

Model: `microsoft/deberta-v3-base`

DeBERTa is a transformer-based language model that uses disentangled attention and an enhanced mask decoder to improve language understanding.

For this project, the model is adapted for binary sequence classification.

### Model configuration

| Component | Configuration |
|---|---|
| Base model | DeBERTa-v3-base |
| Model provider | Microsoft |
| Task | Binary text classification |
| Input | Premise + Hypothesis |
| Output | Entailment / Not Entailment |
| Maximum sequence length | 256 tokens |

We use the same base architecture across our experiments so that different fine-tuning strategies can be studied more consistently.

## 5. Experiments

We organized the project into three experiments.

### Experiment 1 – Baseline DeBERTa-v3-base

The first experiment establishes a baseline using the pretrained DeBERTa-v3-base model.

The model is fine-tuned on the RTE training dataset, and its performance is evaluated using the validation data.

This experiment provides a reference point for comparing the other approaches.

### Experiment 2 – Optuna Hyperparameter Optimization

In the second experiment, we use Optuna to explore different hyperparameter combinations.

Instead of manually selecting all training settings, Optuna runs multiple trials and searches for configurations that improve model performance.

The optimization focuses on finding suitable training parameters while keeping the DeBERTa-v3-base architecture unchanged.

We also visualize trial performance to understand how the optimization process progresses.

### Experiment 3 – LoRA Fine-Tuning

The third experiment explores Low-Rank Adaptation (LoRA), a parameter-efficient fine-tuning technique.

Instead of updating all the original model parameters, LoRA introduces additional trainable low-rank matrices into selected attention projection layers.

The original pretrained model weights are frozen, allowing the model to learn task-specific information with fewer trainable parameters.

**Recorded model configuration:**

| Parameter | Value |
|---|---:|
| Base model | DeBERTa-v3-base |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| Learning rate | 5e-5 |
| Batch size | 16 |
| Maximum epochs | 8 |
| Weight decay | 0.01 |

The recorded parameter analysis is:

| Parameter type | Count |
|---|---:|
| Original model parameters | 184,423,682 |
| Trainable parameters | 1,034,498 |
| Total after LoRA integration | 185,458,180 |

Only approximately 0.56% of the total parameters are trainable in this experiment.

## 6. Evaluation Metrics

We use standard classification metrics to evaluate the models.

- **Accuracy:** Measures the percentage of correctly classified examples.
- **Precision:** Measures how many predicted positive examples are actually positive.
- **Recall:** Measures how many actual positive examples are correctly identified.
- **Macro F1-score:** Calculates the F1-score independently for each class and takes their average.

We also use visualizations to understand model performance and compare experimental results.

## 7. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | Pretrained transformer models |
| DeBERTa-v3-base | Main language model |
| Optuna | Hyperparameter optimization |
| PEFT / LoRA | Parameter-efficient fine-tuning |
| Scikit-learn | Evaluation metrics |
| Matplotlib | Result visualization |
| Kaggle Notebooks | Model training and experimentation |
| Git & GitHub | Version control and project management |

## 8. Repository Structure

The current GitHub repository is organized as follows:

```text
RTE-Recognizing-Textual-Entailment--DL/
│
├── README.md
│
└── deberta-base.ipynb
```

- `README.md`: Project overview, methodology, experiments, and execution instructions.
- `deberta-base.ipynb`: Jupyter notebook containing the DeBERTa-based implementation.

The notebook and its associated experiment outputs are maintained through the Kaggle workflow.

## 9. How to Run the Project

The experiments were developed using Kaggle Notebooks with GPU acceleration.

### Step 1: Clone the repository

Open a terminal and execute:

```bash
git clone https://github.com/Rohitsah6372/RTE-Recognizing-Textual-Entailment--DL.git

cd RTE-Recognizing-Textual-Entailment--DL
```

### Step 2: Open the notebook

Open `deberta-base.ipynb` using Jupyter Notebook or import it into Kaggle.

For the recommended Kaggle workflow:

1. Open Kaggle.
2. Navigate to Notebooks.
3. Create a new notebook or import the existing notebook.
4. Upload or import `deberta-base.ipynb`.
5. Enable GPU acceleration from Notebook Settings.

### Step 3: Add the dataset

Attach the RTE dataset to your Kaggle notebook.

The dataset files are:

```text
train.jsonl
val.jsonl
```

The expected Kaggle dataset location used in our workflow is:

```python
DATA_DIR = "/kaggle/input/datasets/rohitsah402/rte-challenge"

TRAIN_PATH = f"{DATA_DIR}/train.jsonl"
VAL_PATH = f"{DATA_DIR}/val.jsonl"
```

Make sure the dataset is attached to the notebook before executing the training cells.

### Step 4: Install dependencies

Run the following command in a Kaggle notebook cell:

```python
!pip install -q transformers sentencepiece accelerate
!pip install -q peft optuna scikit-learn matplotlib pandas
```

PyTorch is generally available in the Kaggle environment.

### Step 5: Execute the notebook

Run the notebook cells sequentially.

The general workflow is:

1. Import the required libraries.
2. Load the RTE dataset.
3. Perform data preprocessing.
4. Load the pretrained DeBERTa-v3-base model.
5. Tokenize the premise-hypothesis pairs.
6. Configure model training.
7. Train and validate the model.
8. Calculate evaluation metrics.
9. Generate performance visualizations.
10. Save the trained model and experimental outputs.

**Note:** The exact cells and configuration depend on the experiment being executed. Optuna optimization and LoRA training require their respective experiment configurations.

## 10. Results and Observations

The three experiments allow us to investigate different approaches to fine-tuning the same pretrained architecture.

- The baseline experiment provides a reference for model performance.
- The Optuna experiment explores how hyperparameter selection affects performance.
- The LoRA experiment investigates whether competitive performance can be achieved with significantly fewer trainable parameters.

An important part of this project is understanding that improving model accuracy is not the only objective. Training efficiency and parameter efficiency are also important considerations.

## 11. Future Improvements

Some possible directions for further development include:

- Experimenting with different LoRA ranks and configurations.
- Exploring additional hyperparameter optimization strategies.
- Performing detailed error analysis on incorrectly classified sentence pairs.
- Comparing training time and memory requirements.
- Improving generalization on unseen RTE examples.
- Exploring other parameter-efficient fine-tuning methods.

## 12. Acknowledgement

This project was developed as part of the CS6315E – Neural Networks and Deep Learning course.

The work helped us gain practical experience with transformer architectures, NLP classification, hyperparameter optimization, and parameter-efficient fine-tuning.

---

**Project:** Recognizing Textual Entailment using Deep Learning  
**Primary Architecture:** DeBERTa-v3-base  
**Experiments:** Baseline, Optuna Optimization, and LoRA Fine-Tuning  
**Framework:** PyTorch and Hugging Face Transformers
