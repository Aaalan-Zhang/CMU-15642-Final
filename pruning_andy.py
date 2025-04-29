import logging
import traceback

import torch
from datasets import load_dataset

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
from sentence_transformers.cross_encoder.losses import ListMLELoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
import torch.nn.functional as F

# model_name = "microsoft/MiniLM-L12-H384-uncased"
# model_name = "tomaarsen/reranker-ModernBERT-base-gooaq-bce"
model_name = "tomaarsen/reranker-MiniLM-L12-gooaq-bce"

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
num_epochs = 1
max_docs = None
respect_input_order = True  # Whether to respect the original order of documents

# 1. Define our CrossEncoder model
torch.manual_seed(12)
model = CrossEncoder(model_name, num_labels=1)

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
# evaluator = CrossEncoderNanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"], batch_size=eval_batch_size)
# evaluator(model)
evaluator = CrossEncoderNanoBEIREvaluator(dataset_names=["msmarco"], batch_size=eval_batch_size)

# 5. Define the training arguments
short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
run_name = f"reranker-msmarco-v1.1-{short_model_name}-listmle"
num_epochs = 2  # run 2 epochs only prune the first epoch
args = CrossEncoderTrainingArguments(
    output_dir=f"models/{run_name}",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,
    bf16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_NanoBEIR_R100_mean_ndcg@10",
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    logging_steps=250,
    logging_first_step=True,
    run_name=run_name,
    seed=12,
)

class PruningTrainer(CrossEncoderTrainer):
    def __init__(self, *args, ffn_norm=2, attn_norm=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.ffn_norm  = ffn_norm
        self.attn_norm = attn_norm
        self.step      = 0

        # Masks for each layer
        self.ffn_masks  = []
        self.attn_masks = []
        for blk in self.model.model.bert.encoder.layer:
            self.ffn_masks.append(torch.ones_like(
                blk.intermediate.dense.weight, dtype=torch.bool, device=blk.intermediate.dense.weight.device
            ))
            self.attn_masks.append(torch.ones_like(
                blk.attention.output.dense.weight, dtype=torch.bool, device=blk.attention.output.dense.weight.device
            ))

        # Custom schedule: step -> (action, target_layers, drop_rate)
        self.schedule = {
            150:  ("ffn",  "all",   0.12),    # FFN prune #1
            200:  ("ffn",[0,1,2],   0.2),   # Attn prune #1 (layers 0–1)
            300:  ("ffn",  [3,4,5],   0.2),    # FFN prune #2
            350:  ("ffn",[6,7,8],   0.2),   # Attn prune #2 (layers 4–5)
            450:  ("ffn",  [9,10,11],   0.2),    # FFN prune #3
            550: ("ffn","all",   0.12),   # Attn prune #3 (layers 8–9)
            650: ("ffn","all",   0.12),   # Attn prune #3 (layers 8–9)
        }
    def training_step(self, model: CrossEncoder, inputs: dict, batch_size: int) -> torch.Tensor:
        loss = super().training_step(model, inputs, batch_size)
        self.step += 1

        # only prune during the single epoch (epoch 0)
        epoch = getattr(self.state, "epoch", 0)
        with torch.no_grad():
            # re-apply existing masks
            for blk, fmask, amask in zip(model.bert.encoder.layer,
                                         self.ffn_masks,
                                         self.attn_masks):
                blk.intermediate.dense.weight.data.mul_(fmask)
                blk.attention.output.dense.weight.data.mul_(amask)

            # prune at scheduled steps in epoch 0
            if epoch < 1 and self.step in self.schedule:
                action, layers, drop_rate = self.schedule[self.step]
                print(f"— Scheduled prune at step {self.step} (epoch {epoch}):"
                      f" {action} on {layers} @ drop_rate={drop_rate} —")

                new_ffn_masks  = list(self.ffn_masks)
                new_attn_masks = list(self.attn_masks)

                # figure out which layer indices to touch
                if layers == "all":
                    target_idxs = list(range(len(model.bert.encoder.layer)))
                else:
                    target_idxs = layers

                for idx in target_idxs:
                    blk = model.bert.encoder.layer[idx]
                    if action == "ffn":
                        w, mask, p = blk.intermediate.dense.weight, self.ffn_masks[idx], self.ffn_norm
                    else:
                        w, mask, p = blk.attention.output.dense.weight, self.attn_masks[idx], self.attn_norm

                    # compute row norms (only on un-pruned rows)
                    norms = torch.norm(w * mask.float(), p=p, dim=1)
                    size  = norms.size(0)
                    # drop_rate-specific k
                    k     = max(1, int(drop_rate * size))

                    # never prune rows already zeroed
                    norms[~mask.any(dim=1)] = float("inf")
                    prune_idx = torch.topk(norms, k, largest=False).indices

                    row_mask = torch.ones(size, device=w.device, dtype=torch.bool)
                    row_mask[prune_idx] = False
                    full_mask = row_mask.view(-1,1).expand_as(w)
                    combined = mask & full_mask
                    w.data.mul_(combined)

                    if action == "ffn":
                        new_ffn_masks[idx] = combined
                    else:
                        new_attn_masks[idx] = combined

                self.ffn_masks  = new_ffn_masks
                self.attn_masks = new_attn_masks

                # report global sparsity
                total   = sum(p.numel() for p in model.parameters())
                nonzero = sum(torch.count_nonzero(p).item() for p in model.parameters())
                spar    = 100 * (1 - nonzero/total)
                print(f"   Sparsity now: {spar:.2f}% ({nonzero:,}/{total:,} non-zero)")

        return loss

# 6. Create the trainer & start training
trainer = PruningTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=evaluator,
)
trainer.train()
torch.cuda.empty_cache()

# # 7. Evaluate the final model, useful to include these in the model card
# evaluator(model)

# # 8. Save the final model
# final_output_dir = f"models/{run_name}/final"
# model.save_pretrained(final_output_dir)
