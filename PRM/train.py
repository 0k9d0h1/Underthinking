# train_regressor_ds.py
from datasets import load_from_disk
import os
os.environ["HF_HOME"] = "/home/kdh0901/Desktop/cache_dir/kdh0901/.cache/huggingface"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
os.environ["WANDB_PROJECT"] = "ppl_regression"
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
)
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForTokenClassification
import torch, torch.nn.functional as F
import numpy as np, scipy.stats as st


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.squeeze(-1)
    r, _ = st.pearsonr(preds, labels)
    return {"pearson_r": float(r)}

REG_DIR = "./reg_dp"
ds = load_from_disk(REG_DIR).train_test_split(test_size=0.05, seed=42)
tok     = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

def tok_fn(batch): return tok(batch["text"], truncation=True)
tok_ds = ds.map(tok_fn, batched=True, remove_columns=["text"])

# base    = AutoModelForTokenClassification.from_pretrained(
#             "Qwen/Qwen2.5-0.5B",
#             num_labels=1,
#             torch_dtype=torch.bfloat16,
#             device_map="auto",
#             trust_remote_code=True)

class Regressor(Qwen2ForTokenClassification):
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Call the parent's forward method to get the logits for all tokens.
        # Do NOT pass labels here, so it doesn't compute the default cross-entropy loss.
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the prediction for the very last token based on the attention mask.
        last_token_indices = attention_mask.sum(-1) - 1
        preds = outputs.logits.squeeze(-1)[torch.arange(input_ids.size(0)), last_token_indices]

        loss = None
        if labels is not None:
            # Compute our desired regression loss.
            loss = F.mse_loss(preds, labels.bfloat16())
        
        # Return in the standard format that the Trainer expects.
        return TokenClassifierOutput(
            loss=loss,
            logits=preds.unsqueeze(-1),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
dc = DataCollatorWithPadding(tok, pad_to_multiple_of=8, return_tensors="pt")
model = Regressor.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    num_labels=1,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
tr = Trainer(
    model           = model,
    args            = TrainingArguments(
        "/home/kdh0901/Desktop/cache_dir/kdh0901/prm_checkpoints",
        num_train_epochs           = 5,
        per_device_train_batch_size= 4,
        per_device_eval_batch_size = 4,
        learning_rate              = 2e-5,
        bf16                       = True,
        gradient_checkpointing     = True,
        deepspeed                  = "ds_zero2.json",   # optional
        logging_steps              = 100,
        eval_steps                 = 1000,
        save_steps                 = 5000,
        report_to                  = ["wandb"],         
        run_name                   = "qwen-regressor", 
        eval_strategy        = "steps",
    ),
    train_dataset   = tok_ds["train"],
    eval_dataset    = tok_ds["test"],
    data_collator   = dc,
    compute_metrics = compute_metrics
)
tr.train(resume_from_checkpoint=True)
tr.save_model("/home/kdh0901/Desktop/cache_dir/kdh0901/prm_checkpoints")