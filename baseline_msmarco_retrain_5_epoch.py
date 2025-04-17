# !pip install -U sentence-transformers
# !pip install datasets

import logging
import traceback

import torch
from datasets import load_dataset

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
from sentence_transformers.cross_encoder.losses import ListMLELoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

# model_name = "microsoft/MiniLM-L12-H384-uncased"
# model_name = "tomaarsen/reranker-ModernBERT-base-gooaq-bce"
model_name = "tomaarsen/reranker-MiniLM-L12-gooaq-bce" # 33.4M model trained on gooaq, converged at the end of the 1st epoch, Nanomsmarco R100 Mrr@10 about 0.53

# tomaarsen/reranker-msmarco-MiniLM-L12-H384-uncased-lambdaloss # 33.4M model trained on MS MARCO, converged at the beginning of the 1st epoch, Nanomsmarco R100 Mrr@10 about 0.530746
# cross-encoder/ms-marco-MiniLM-L6-v2 # 22.7M SOTA model trained on MS MARCO, converged at the beginning, Nanomsmarco R100 Mrr@10 about 0.54

# Set the log level to INFO to get more information
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
NUM = 64
train_batch_size = NUM
eval_batch_size = NUM
mini_batch_size = NUM
num_epochs = 3
max_docs = None
respect_input_order = True  # Whether to respect the original order of documents

# 1. Define our CrossEncoder model
torch.manual_seed(12)
model = CrossEncoder(model_name, num_labels=1)

print(model.model)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model max length:", model.max_length)
print("Model num labels:", model.num_labels)
print("Model is on device:", next(model.parameters()).device)

# 2. Load the MS MARCO dataset: https://huggingface.co/datasets/microsoft/ms_marco
logging.info("Read train dataset")
dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train")

def listwise_mapper(batch, max_docs: int | None = 10):
    processed_queries = []
    processed_docs = []
    processed_labels = []

    for query, passages_info in zip(batch["query"], batch["passages"]):
        # Extract passages and labels
        passages = passages_info["passage_text"]
        labels = passages_info["is_selected"]

        # Pair passages with labels and sort descending by label (positives first)
        paired = sorted(zip(passages, labels), key=lambda x: x[1], reverse=True)

        # Separate back to passages and labels
        sorted_passages, sorted_labels = zip(*paired) if paired else ([], [])

        # Filter queries without any positive labels
        if max(sorted_labels) < 1.0:
            continue

        # Truncate to max_docs
        if max_docs is not None:
            sorted_passages = list(sorted_passages[:max_docs])
            sorted_labels = list(sorted_labels[:max_docs])

        processed_queries.append(query)
        processed_docs.append(sorted_passages)
        processed_labels.append(sorted_labels)

    return {
        "query": processed_queries,
        "docs": processed_docs,
        "labels": processed_labels,
    }

# Create a dataset with a "query" column with strings, a "docs" column with lists of strings,
# and a "labels" column with lists of floats
dataset = dataset.map(
    lambda batch: listwise_mapper(batch=batch, max_docs=max_docs),
    batched=True,
    remove_columns=dataset.column_names,
    desc="Processing listwise samples",
)

dataset = dataset.train_test_split(test_size=1_000)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]
logging.info(train_dataset)

# 3. Define our training loss
loss = ListMLELoss(model, mini_batch_size=mini_batch_size, respect_input_order=respect_input_order)

# 4. Define the evaluator. We use the CENanoBEIREvaluator, which is a light-weight evaluator for English reranking
evaluator = CrossEncoderNanoBEIREvaluator(dataset_names=["msmarco"], batch_size=eval_batch_size)
evaluator(model)

# 5. Define the training arguments
short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
run_name = f"reranker-msmarco-v1.1-{short_model_name}-listmle"
args = CrossEncoderTrainingArguments(
    output_dir=f"models/{run_name}",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    load_best_model_at_end=True,
    metric_for_best_model="eval_NanoBEIR_R100_mean_ndcg@10",
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=250,
    logging_first_step=True,
    run_name=run_name,
    seed=12,
)

# 6. Create the trainer & start training
trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=evaluator,
)
trainer.train()

# 7. Evaluate the final model, useful to include these in the model card
evaluator(model)

# 8. Save the final model
final_output_dir = f"models/{run_name}/final"
model.save_pretrained(final_output_dir)
