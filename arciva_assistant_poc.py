# %% [markdown]
# # Fine-Tuning Large Language Models for Domain-Specific QA
# 
# ## Executive Summary
# This script implements a scalable **MLOps pipeline** for adapting 7B-parameter Large Language Models (LLMs) to specialized corporate knowledge bases.

# %% [markdown]
# ## 1. Library Imports & Global Configuration
# %%
import os
import torch
import gc
import time
import transformers
import trl
import peft
import evaluate
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sacrebleu
from rouge_score import rouge_scorer

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig

# --- Global Hyperparameters ---
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATA_PATH = "arciva_qa_synthetic.jsonl"
OUTPUT_DIR = "model_output"
FINAL_SAVE_DIR = "final_merged_model"
USE_WANDB = False

print(f"{'-'*40}")
print(f"Target Architecture : {MODEL_NAME}")
print(f"Compute Device      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"{'-'*40}")

# %% [markdown]
# ## 2. Data Ingestion & Preprocessing
# %%
def prepare_data():
    if os.path.exists(DATA_PATH):
        print(f"[INFO] Loading dataset from source: {DATA_PATH}...")
        full_dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    else:
        print(f"[WARN] Source file {DATA_PATH} unavailable. Generating synthetic verification data.")
        full_dataset = Dataset.from_dict({
            "instruction": ["Explain quantum mechanics.", "How do I reset my password?"] * 50,
            "response": ["It is complex.", "Go to settings and click reset."] * 50
        })

    # Stratified Split (Approximated by random shuffle)
    # Split 1: 10% Test
    split_1 = full_dataset.train_test_split(test_size=0.1, seed=42)
    test_set = split_1["test"]
    remaining = split_1["train"]

    # Split 2: 10% Validation (of remaining -> ~9% of total)
    split_2 = remaining.train_test_split(test_size=0.1, seed=42)
    train_set = split_2["train"]
    val_set = split_2["test"]

    return train_set, val_set, test_set

def inspect_data_distribution(dataset, name="Training Set"):
    # Convert to Pandas for Analysis
    df = dataset.to_pandas()
    
    # Robust Length Calculation
    df['inst_len'] = df['instruction'].fillna("").apply(lambda x: len(str(x).split()))
    if 'response' in df.columns:
        df['resp_len'] = df['response'].fillna("").apply(lambda x: len(str(x).split()))
    elif 'output' in df.columns:
         df['resp_len'] = df['output'].fillna("").apply(lambda x: len(str(x).split()))
    else:
        df['resp_len'] = 0

    # Visualization - Saved to Disk for Script
    print(f"[ANALYSIS] Generating distribution plot for {name} -> dataset_distribution.png")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Instruction Length
    plt.subplot(1, 2, 1)
    sns.histplot(df['inst_len'], color="#1f77b4", kde=True)
    plt.title(f"{name}: Instruction Length Distribution")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    
    # Plot 2: Response Length
    plt.subplot(1, 2, 2)
    sns.histplot(df['resp_len'], color="#2a9d8f", kde=True)
    plt.title(f"{name}: Response Length Distribution")
    plt.xlabel("Word Count")
    
    plt.tight_layout()
    plt.savefig("dataset_distribution.png")
    plt.close() # Close figure to free memory

def inspect_data(train_set, val_set, test_set):
    print("\n" + "="*50)
    print("            DATASET STATISTICS            ")
    print("="*50)
    print(f"Total Corpus Size  : {len(train_set) + len(val_set) + len(test_set)} samples")
    print(f"Training Subset    : {len(train_set)} samples")
    print(f"Validation Subset  : {len(val_set)} samples")
    print(f"Test Subset        : {len(test_set)} samples")
    print("-"*50)

    print("\n[DATA SAMPLE: Index 0]")
    sample = train_set[0]
    print(f"> INSTRUCTION:\n{sample.get('instruction')}")
    print(f"\n> RESPONSE:\n{sample.get('response') or sample.get('output')}")
    print("-"*50)
    
    inspect_data_distribution(train_set)

# %% [markdown]
# ## 3. Model Fine-Tuning
# %%
class NotebookPlotCallback(TrainerCallback):
    def __init__(self):
        self.losses = []
        self.val_losses = []
        self.steps = []
        self.val_steps = []
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if 'loss' in logs:
                self.losses.append(logs['loss'])
                self.steps.append(state.global_step)
            if 'eval_loss' in logs:
                self.val_losses.append(logs['eval_loss'])
                self.val_steps.append(state.global_step)
            
            # CLI Status Update (Silent Plotting)
            if self.start_time:
                elapsed = time.time() - self.start_time
                avg_time = elapsed / state.global_step if state.global_step > 0 else 0
                remaining = int(avg_time * (state.max_steps - state.global_step))
                print(f"[TRAIN] Step {state.global_step}/{state.max_steps} | Loss: {logs.get('loss', 'N/A')} | ETA: {remaining//60}m {remaining%60}s")

def formatting_prompts_func(example):
    output_texts = []
    instruction = example.get('instruction')
    response = example.get('response') or example.get('output')
    if isinstance(instruction, list):
        for i, r in zip(instruction, response):
            output_texts.append(f"### Instruction:\n{i}\n\n### Response:\n{r}")
        return output_texts
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

def run_training_pipeline(train_data, val_data):
    # Hardware Acceleration Logic
    use_gpu = torch.cuda.is_available()
    compute_dtype = torch.float16
    bf16_kwargs = {"bf16": False, "fp16": True}
    
    if use_gpu and torch.cuda.is_bf16_supported():
        print("[INFO] BF16 acceleration enabled for compatible hardware.")
        compute_dtype = torch.bfloat16
        bf16_kwargs = {"bf16": True, "fp16": False}

    # 4-bit Quantization Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    ) if use_gpu else None

    print("[INFO] Initializing Base Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto" if use_gpu else "cpu",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model) if use_gpu else model
    peft_config = LoraConfig(
        lora_alpha=16, lora_dropout=0.1, r=64, bias="none", task_type="CAUSAL_LM"
    )

    # Training Hyperparameters
    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4 if use_gpu else 1,
        gradient_accumulation_steps=1 if use_gpu else 4,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=10,
        report_to="wandb" if USE_WANDB else "none",
        use_cpu=not use_gpu,
        max_length=512,
        **bf16_kwargs
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        processing_class=tokenizer,
        args=args,
        callbacks=[NotebookPlotCallback()]
    )

    print("[INFO] Starting Training Loop...")
    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)
    print(f"[INFO] Training concluded. Artifacts saved to: {OUTPUT_DIR}")
    
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

# %% [markdown]
# ## 4. Quantitative Analysis & Reporting
# %%
def evaluate_model_metrics(train_data, test_data):
    print("[INFO] Reloading model architecture for inference...")
    use_gpu = torch.cuda.is_available()

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True if use_gpu else False,
        device_map="auto" if use_gpu else "cpu",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    model.eval()
    
    # 1. Perplexity Calculation
    print("[METRICS] Calculating Perplexity...")
    eval_args = TrainingArguments(
        output_dir="temp_eval", 
        per_device_eval_batch_size=4, 
        report_to="none"
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data, 
        eval_dataset=test_data,
        formatting_func=formatting_prompts_func,
        processing_class=tokenizer,
        args=eval_args
    )
    results = trainer.evaluate()
    loss = results["eval_loss"]
    perp = torch.exp(torch.tensor(loss))
    
    # 2. Generation & Similarity Analysis
    print("[METRICS] Generating responses for semantic analysis...")
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    results_data = []
    all_preds = []
    all_refs = []
    
    for i, item in enumerate(test_data):
        prompt = f"### Instruction:\\n{item['instruction']}\n\n### Response:\\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, 
                max_new_tokens=100, 
                do_sample=True, 
                top_p=0.9, 
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )
        
        gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        ref_text = (item.get('response') or item.get('output')).strip()
        
        all_preds.append(gen_text)
        all_refs.append(ref_text)
        
        score = scorer.score(ref_text, gen_text)['rougeL'].fmeasure
        
        results_data.append({
            "Instruction": item['instruction'],
            "Prediction": gen_text,
            "Reference": ref_text,
            "ROUGE-L": score,
            "Pred_Len": len(gen_text.split()),
            "Ref_Len": len(ref_text.split())
        })

    # Calculate Corpus BLEU
    bleu = sacrebleu.corpus_bleu(all_preds, [all_refs])
    
    df_results = pd.DataFrame(results_data)
    
    # --- Visualization Panel (Saved to Disk) ---
    print("\n[INFO] Saving visualization panel to 'analysis_plots.png'...")
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.histplot(df_results["ROUGE-L"], bins=10, kde=True, ax=ax[0], color="#2a9d8f")
    ax[0].set_title("Distribution of Semantic Similarity (ROUGE-L)")
    
    df_melt = df_results.melt(value_vars=["Pred_Len", "Ref_Len"], var_name="Source", value_name="Length (Tokens)")
    sns.boxplot(x="Source", y="Length (Tokens)", hue="Source", data=df_melt, ax=ax[1], palette="viridis", legend=False)
    ax[1].set_title("Generation Length Analysis")
    
    plt.tight_layout()
    plt.savefig("analysis_plots.png")
    
    # --- Executive Summary Table ---
    print("\n" + "="*50)
    print("             PERFORMANCE SUMMARY             ")
    print("="*50)
    print(f"{ 'Metric':<20} | { 'Value':<10}")
    print("-"*40)
    print(f"{ 'Test Loss':<20} | {loss:.4f}")
    print(f"{ 'Perplexity':<20} | {perp:.4f}")
    print(f"{ 'BLEU Score':<20} | {bleu.score:.2f}")
    print(f"{ 'Avg ROUGE-L':<20} | {df_results['ROUGE-L'].mean():.4f}")
    print("="*50)
    
    print("\n[ANALYSIS] Top Performing Generations:")
    print(df_results.nlargest(3, "ROUGE-L")[["Instruction", "Prediction", "ROUGE-L"]])
    
    print("\n[ANALYSIS] Low Performing Generations:")
    print(df_results.nsmallest(3, "ROUGE-L")[["Instruction", "Prediction", "ROUGE-L"]])

    gc.collect()
    torch.cuda.empty_cache()

# %% [markdown]
# ## 5. Evaluation Data Export
# %%
def generate_evaluation_report(train_set, val_set, test_set, n_samples=20):
    print(f"[REPORT] Generating CSV evaluation report (Sampling N={n_samples} per split)...")
    
    # Reload architecture
    use_gpu = torch.cuda.is_available()
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        load_in_4bit=True if use_gpu else False, 
        device_map="auto" if use_gpu else "cpu", 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    model.eval()

    splits = {"Train": train_set, "Validation": val_set, "Test": test_set}
    dashboard_data = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for split_name, dataset in splits.items():
        n = min(n_samples, len(dataset))
        subset = dataset.shuffle(seed=42).select(range(n))
        
        print(f"> Processing split: {split_name} ({n} items)...")
        for item in subset:
            prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                out = model.generate(
                    **inputs, 
                    max_new_tokens=100, 
                    do_sample=True, 
                    top_p=0.9, 
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            ref_text = (item.get('response') or item.get('output')).strip()
            score = scorer.score(ref_text, gen_text)['rougeL'].fmeasure
            
            dashboard_data.append({
                "Split": split_name,
                "Instruction": item['instruction'],
                "Reference": ref_text,
                "Prediction": gen_text,
                "ROUGE-L": round(score, 4)
            })

    df = pd.DataFrame(dashboard_data)
    csv_path = "model_evaluation_report.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Evaluation report saved to: {csv_path}")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()

# %% [markdown]
# ## 6. Artifact Compilation & Export
# %%
def export_final_model():
    print(f"[EXPORT] Initializing merge sequence. Target: {FINAL_SAVE_DIR}...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    model = model.merge_and_unload()
    
    model.save_pretrained(FINAL_SAVE_DIR)
    tokenizer.save_pretrained(FINAL_SAVE_DIR)
    print(f"[SUCCESS] Standalone model exported to ./{FINAL_SAVE_DIR}")

# %% [markdown]
# ## Run
# %%
if __name__ == "__main__":
    train_set, val_set, test_set = prepare_data()
    inspect_data(train_set, val_set, test_set)
    run_training_pipeline(train_set, val_set)
    evaluate_model_metrics(train_set, test_set)
    generate_evaluation_report(train_set, val_set, test_set)
    export_final_model()
