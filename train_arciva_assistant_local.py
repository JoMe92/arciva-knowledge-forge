import os
import subprocess
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def formatting_prompts_func(example):
    output_texts = []
    
    # helper to get value from potential keys
    instruction = example.get('instruction')
    response = example.get('response') or example.get('output') # Support both keys

    # Handle batch (list) vs single (scalar)
    if isinstance(instruction, list):
        # Batch mode
        for i, r in zip(instruction, response):
            text = f"### Instruction:\\n{i}\\n\\n### Response:\\n{r}"
            output_texts.append(text)
        return output_texts
    else:
        # Single mode - logic suggests we should return a list for SFTTrainer even if single?
        # But debug showed returning list caused AttributeError in add_eos (text became list).
        # So for single input, we probably return single string, OR SFTTrainer maps it correctly?
        # If the map is unbatched, returning [text] might make the column value [text].
        # Returning text makes it text.
        # However, SFTTrainer documentation emphasizes "list of strings".
        # Let's try returning a list first (standard for SFTTrainer), but if that fails, 
        # it might be because map(batched=False). 
        # But wait, if I return list of 1 string here, and it failed before...
        # Before it failed because zip(str, str) produced list of MANY strings (chars).
        # Example: zip("No", "Yes") -> [('N','Y'), ('o','e'), ...]
        # So text column became a list of strings ["...","..."].
        # add_eos checks text.endswith -> fails on list.
        # If I return [ "full text" ], then text column becomes "full text"?
        # Or ["full text"]?
        # Usually datasets.map(batched=False) expects dict. SFTTrainer wrapper expects list of strings?
        # Let's assume standard SFT usage: return a LIST OF STRINGS which corresponds to the batch.
        # If input is single (scalar), then "batch" size is 1.
        # So I should return ["full text"].
        # If that fails, then SFTTrainer really expects scalar.
        # But "zip" on strings produced a list of length N.
        # And that caused "text" to be a list. 
        # So if I return list of length 1, "text" might be list of length 1?
        # OR "text" might be the string?
        # A scalar return is safer for unbatched map.
        text = f"### Instruction:\\n{instruction}\\n\\n### Response:\\n{response}"
        return text 



def prepare_data():
    """
    Pulls data using DVC.
    """
    print("Pulling data via DVC...")
    try:
        subprocess.check_call(["dvc", "pull"])
        print("Data pulled successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error pulling data: {e}")
        # We might want to raise here, but for now just print
        raise

def train():
    # 1. Prepare Data
    try:
        prepare_data()
    except subprocess.CalledProcessError:
        print("Warning: DVC pull failed. Checking for local data...")
    except Exception as e:
        print(f"Warning: Failed to pull data: {e}")

    # 2. Load Data
    # Assuming the data is pulled to data/raw/arciva_qa_synthetic.jsonl
    data_path = "data/raw/arciva_qa_synthetic.jsonl"
    if not os.path.exists(data_path):
        # Fallback for testing if file doesn't exist
        print(f"Dataset not found at {data_path}. Creating dummy data for testing purposes.")
        from datasets import Dataset
        dataset = Dataset.from_dict({
            "instruction": ["Test instruction"] * 5,
            "response": ["Test response"] * 5
        })
    else:
        dataset = load_dataset("json", data_files=data_path, split="train")

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
        fp16 = False # usually True for Llama 2 but depends on card, leaving False safe default or use bf16
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
    # We apply PEFT even on CPU to test the flow, though not strictly necessary for tiny model.
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
    # Using SFTConfig for newer trl versions
    from trl import SFTConfig
    
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
        max_length=512 if use_gpu else 128, # Changed to max_length per error suggestion
        packing=False, # Moved here
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

if __name__ == "__main__":
    train()
