import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from google.cloud import storage

def formatting_prompts_func(example):
    output_texts = []
    
    # helper to get value from potential keys
    instruction = example.get('instruction')
    response = example.get('response') or example.get('output') # Support both keys

    # Handle batch (list) vs single (scalar)
    if isinstance(instruction, list):
        # Batch mode
        for i, r in zip(instruction, response):
            text = f"### Instruction:\n{i}\n\n### Response:\n{r}"
            output_texts.append(text)
        return output_texts
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        return text 

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    print(f"Downloading {source_blob_name} from bucket {bucket_name} to {destination_file_name}...")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print("Download successful.")
    except Exception as e:
        print(f"Error downloading from GCS: {e}")
        raise

def train():
    # 1. Prepare Data from GCS
    # Expected Env Vars: GCS_BUCKET_NAME, GCS_BLOB_NAME
    # Defaults set based on user configuration
    bucket_name = os.environ.get("GCS_BUCKET_NAME", "arciva-llm-mlops-data")
    blob_name = os.environ.get("GCS_BLOB_NAME", "01_raw_data/qa_pairs_v1/arciva_qa_synthetic.jsonl") 
    local_data_path = "data_gcs.jsonl"

    if bucket_name:
        download_from_gcs(bucket_name, blob_name, local_data_path)
    else:
        print("⚠️ GCS_BUCKET_NAME not set. Checking if file exists locally...")
        if not os.path.exists(local_data_path):
             print(f"Error: {local_data_path} not found and no GCS bucket provided.")
             # Fallback for testing if just trying to run build
             local_data_path = "data/raw/arciva_qa_synthetic.jsonl" 
             if not os.path.exists(local_data_path):
                 raise FileNotFoundError("Could not find data file.")

    # 2. Load Data
    dataset = load_dataset("json", data_files=local_data_path, split="train")

    # 3. Hardware & Model Configuration
    use_gpu = torch.cuda.is_available()
    output_dir = os.environ.get("AIP_MODEL_DIR", "model_output")

    if use_gpu:
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        # base_model_name = "meta-llama/Llama-2-7b-hf" # Uncomment for production
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs = {
            "quantization_config": bnb_config,
            "device_map": "auto",
        }
        max_steps = -1 # Full training
        fp16 = False 
    else:
        print("⚠️ No GPU detected! Running in CPU DEBUG mode.")
        print("Using 'HuggingFaceM4/tiny-random-LlamaForCausalLM' for fast validation.")
        base_model_name = "HuggingFaceM4/tiny-random-LlamaForCausalLM"
        model_kwargs = {"device_map": "cpu"}
        max_steps = 1 # Single step validation
        fp16 = False 

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        **model_kwargs
    )
    
    # 4. QLoRA Configuration
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    if use_gpu:
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        model = prepare_model_for_kbit_training(model)

    # 5. Training Arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4 if use_gpu else 1,
        gradient_accumulation_steps=1 if use_gpu else 4,
        optim="paged_adamw_32bit" if use_gpu else "adamw_torch",
        save_steps=25,
        logging_steps=1,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=fp16,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=max_steps,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        use_cpu=not use_gpu,
        max_length=512 if use_gpu else 128, 
        packing=False, 
        report_to="none" if not use_gpu else "wandb",
    )

    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train()

    # 7. Save Model
    trainer.model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    # Optional: Upload model back to GCS if needed
    # (Implementation dependent on requirements)

if __name__ == "__main__":
    train()
