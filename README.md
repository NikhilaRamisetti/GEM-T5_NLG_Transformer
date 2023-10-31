# GEM-T5_NLG_Transformer


This repository contains code for loading GEM E2E NLG datasets, training a model using a T5 tokenizer, and generating sentences with the T5ForConditionalGeneration model. It also includes code for training and evaluating the model using METEOR, ROUGE, and BERT_score metrics.

## Installation

Use the following commands to install the required dependencies:

```shell
%%capture
!pip install git+https://github.com/huggingface/datasets.git
!pip install rouge_score
!pip install sentencepiece
!pip install transformers
!pip install bert-score
```

## Usage

1. Load the GEM E2E NLG datasets:

```python
import datasets
data = datasets.load_dataset('GEM/e2e_nlg')
```

2. Tokenize and preprocess the data for training:

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_NAME = "google/t5-v1_1-base"
MAX_LENGTH = 32

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

train_data_tokenized = data['train'].map(
    lambda batch: batch_tokenize(batch, tokenizer, max_length=MAX_LENGTH),
    batched=True
)
valid_data_tokenized = data['validation'].map(
    lambda batch: batch_tokenize(batch, tokenizer, max_length=MAX_LENGTH),
    batched=True
)
```

3. Train the model:

```python
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# Define training arguments
train_args = Seq2SeqTrainingArguments(
    output_dir="results/",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    # ...
)

trainer = Seq2SeqTrainer(
    model=model,
    args=train_args,
    train_dataset=train_data_tokenized,
    eval_dataset=valid_data_tokenized,
    tokenizer=tokenizer,
    compute_metrics=rouge_metric_fn,
)

trainer.train()
```

4. Load a trained model and generate text using beam search:

```python
checkpoint_path = "checkpoints/checkpoint-8382"
tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-base")
model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)

# Create a Seq2SeqTrainer for evaluation
trainer = Seq2SeqTrainer(
    model=model,
    args=train_args,
    train_dataset=train_data_tokenized,
    eval_dataset=valid_data_tokenized,
    tokenizer=tokenizer,
    compute_metrics=rouge_metric_fn,
)

# Generate text with beam search decoder
valid_output = data['validation'].map(
    lambda batch: {'generated': beam_generate_sentences(
        batch,
        model,
        tokenizer,
        num_beams=BEAM_SIZE,
        max_length=MAX_LENGTH,
        device=DEVICE)
    },
    batched=True,
    batch_size=8,
)
```

5. Evaluate the generated text with ROUGE metrics:

```python
rouge_results = rouge_scorer.compute(
    predictions=valid_output["generated"],
    references=valid_output["target"],
    rouge_types=["rouge2", "rougeL"],
    use_aggregator=True, use_stemmer=False,
)

f"R-2: {rouge_results['rouge2'].mid.fmeasure:.3f} R-L: {rouge_results['rougeL'].mid.fmeasure:.3f}"
```

6. Save the generated text as a JSON submission:

```python
# Save generated text as a JSON submission
submission_dict = {
    "submission_name": SUBMISSION_NAME,
    "param_count": sum(p.numel() for p in model.parameters()),
    "description": DESCRIPTION,
    "tasks": {
        "e2e_validation": {
            "values": valid_formatted,
            "keys": valid_keys
        }
    }
}

new_task_name = "e2e_challenged_train_sample"
new_task_data = {
    "values": challenge_train_sample_formatted,
    "keys": challenge_train_sample_keys
}

submission_dict["tasks"][new_task_name] = new_task_data

with open('results/gem_submission.json', 'w') as f:
    f.write(json.dumps(submission_dict)
```

## Human Evaluation

Here is a sample of human evaluation of generated phrases:

1. Input: "name[The Eagle], eatType[coffee shop], food[Japanese], priceRange[less than £20], customer rating[low], area[riverside], familyFriendly[yes], near[Burger King]"

   Output: "The Golden Curry is a family-friendly Japanese restaurant with a price range of less than £20. It is located in the city centre."

2. Input: "name[The Mill], eatType[coffee shop], food[French], area[riverside], near[The Sorrento]"

   Output: "The Golden Curry is a family-friendly restaurant with a price range of less than £20. It is located in the city centre near Café Rouge."

   (Note: PriceRange and Near parameters were inaccurately generated)

3. Input: "name[Loch Fyne], food[French], area[riverside], near[The Rice Boat]"

   Output: "The Golden Curry is a family-friendly Japanese restaurant with a price range of less than £20. It is located in the city centre."

   (Note: Near parameter was inaccurately generated)

## Metrics Evaluation

To evaluate the generated text with metrics, follow these steps:

```shell
%%capture

!git clone https://github.com/GEM-benchmark/GEM-metrics.git
%cd GEM-metrics
!pip install -r requirements.txt

!python run_metrics.py results/gem_submission.json
```

This will provide you with metrics for your generated text.
```

Feel free to adapt this `README.md` to your needs and provide additional documentation as required.