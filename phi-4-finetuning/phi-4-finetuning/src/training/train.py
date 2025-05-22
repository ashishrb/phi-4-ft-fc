# src/training/train.py

import os
import logging
import argparse
import math
from typing import Dict, List, Optional, Any

import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    set_seed
)
from datasets import load_dataset
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
import evaluate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "training.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Phi4Training")

# src/training/train.py

import os
import logging
import argparse
import math
from typing import Dict, List, Optional, Any

import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback,
    set_seed
)
from datasets import load_dataset
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
import evaluate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Phi4Training")

# Add the disk space checking function
def check_disk_space(directory: str, required_gb: float = 5.0) -> bool:
    """
    Check if there's enough disk space available in the specified directory
    
    Args:
        directory: Directory to check
        required_gb: Required space in GB
        
    Returns:
        True if enough space is available, False otherwise
    """
    try:
        import shutil
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Get disk usage statistics
        total, used, free = shutil.disk_usage(directory)
        
        # Convert to GB
        free_gb = free / (1024 ** 3)
        
        logger.info(f"Available disk space in {directory}: {free_gb:.2f} GB")
        
        # Check if there's enough space
        if free_gb < required_gb:
            logger.warning(f"Low disk space warning: Only {free_gb:.2f} GB available in {directory}, required {required_gb} GB")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error checking disk space: {str(e)}")
        return False

# Add the custom early stopping callback
class CustomEarlyStoppingCallback(EarlyStoppingCallback):
    """
    Custom early stopping callback that can detect loss spikes
    """
    
    def __init__(self, early_stopping_patience: int = 3, early_stopping_threshold: float = 0.0, max_loss_spike: float = 10.0):
        """
        Initialize custom early stopping callback
        
        Args:
            early_stopping_patience: Number of evaluation calls with no improvement after which training will be stopped
            early_stopping_threshold: Minimum change in the monitored quantity to qualify as an improvement
            max_loss_spike: Maximum allowed loss spike multiplier before stopping
        """
        super().__init__(early_stopping_patience=early_stopping_patience, early_stopping_threshold=early_stopping_threshold)
        self.max_loss_spike = max_loss_spike
        self.prev_loss = float('inf')
        self.best_loss = float('inf')
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """
        Event called after an evaluation phase
        """
        # First, apply the regular early stopping logic
        super().on_evaluate(args, state, control, metrics, **kwargs)
        
        # Get the current loss
        current_loss = metrics.get("eval_loss", None)
        
        if current_loss is not None:
            # Update best loss
            if current_loss < self.best_loss:
                self.best_loss = current_loss
            
            # Check for loss spike
            if self.prev_loss != float('inf') and current_loss > self.max_loss_spike * self.best_loss:
                logger.warning(f"Loss spike detected: current_loss={current_loss:.4f}, best_loss={self.best_loss:.4f}, " 
                              f"spike_factor={current_loss / self.best_loss:.2f} (max allowed: {self.max_loss_spike})")
                logger.warning("Early stopping due to loss spike")
                
                print(f"\n{'='*50}")
                print(f"LOSS SPIKE DETECTED: {current_loss:.4f} (best: {self.best_loss:.4f}, spike factor: {current_loss / self.best_loss:.2f}x)")
                print(f"Training stopped to prevent divergence")
                print(f"{'='*50}\n")
                
                # Set control flag to stop training
                control.should_training_stop = True
            
            # Update previous loss
            self.prev_loss = current_loss

# Then continue with the existing FunctionCallingDataset class and other functions...

class FunctionCallingDataset(Dataset):
    """
    Dataset for Phi-4 function calling finetuning
    """
    
    def __init__(self, tokenizer, dataset, max_length=2048):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        messages = example.get("messages", [])
        
        # Convert messages to prompt format
        prompt = self.convert_messages_to_prompt(messages)
        
        # Tokenize with padding and truncation
        encodings = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create input_ids and labels (for causal LM, labels are the same as input_ids)
        input_ids = encodings.input_ids[0]
        attention_mask = encodings.attention_mask[0]
        
        # Add special handling for labels (we don't want to compute loss on system prompt)
        labels = input_ids.clone()
        
        # Find the first user message
        first_user_token = self.find_first_user_token(prompt)
        if first_user_token > 0:
            # Mask out system message from loss computation
            labels[:first_user_token] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def convert_messages_to_prompt(self, messages):
        """Convert a list of message dictionaries to the Phi-4 prompt format"""
        prompt = ""
        
        for message in messages:
            role = message.get("role", "").lower()
            content = message.get("content", "")
            
            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
            elif role == "function":
                # Handle function calls differently - include function name if available
                function_name = message.get("name", "function")
                prompt += f"<|function|>\nname: {function_name}\ncontent: {content}\n"
        
        return prompt.strip()
    
    def find_first_user_token(self, prompt):
        """Find the position of the first user message token"""
        system_end = prompt.find("<|user|>")
        if system_end == -1:
            return 0
        
        # Tokenize up to the user token to get its position
        system_part = prompt[:system_end]
        return len(self.tokenizer.encode(system_part)) - 1  # -1 because we don't want to include the user token itself


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Phi-4 for function calling")
    
    # Model and dataset arguments
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/Phi-4-mini-instruct",
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name or path of the dataset to use")
    parser.add_argument("--dataset_config_name", type=str, default=None,
                        help="The configuration name of the dataset to use")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="The output directory where the model will be saved")
    
    # Training hyperparameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size per GPU/TPU core/CPU for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="Batch size per GPU/TPU core/CPU for evaluation")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="The initial learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for gradient clipping")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs to perform")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="If > 0, overrides num_train_epochs and is total number of training steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Linear warmup over warmup_ratio fraction of total steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="The scheduler type to use", 
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--max_loss_spike", type=float, default=10.0,
                       help="Maximum allowed loss spike factor before stopping training")
    
    # LoRA parameters
    parser.add_argument("--use_lora", action="store_true", 
                        help="Whether to use LoRA for parameter-efficient finetuning")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout probability")
    
    # Additional training arguments
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch",
                        help="The evaluation strategy to use", choices=["no", "steps", "epoch"])
    parser.add_argument("--save_strategy", type=str, default="epoch",
                        help="The checkpoint save strategy to use", choices=["no", "steps", "epoch"])
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X updates steps when save_strategy='steps'")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Run evaluation every X updates steps when evaluation_strategy='steps'")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every X updates steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="If a value is passed, will limit the total amount of checkpoints")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use fp16 16-bit (mixed) precision training")
    parser.add_argument("--bf16", action="store_true",
                        help="Whether to use bf16 16-bit (mixed) precision training (requires Ampere or newer NVIDIA GPU)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help="The list of integrations to report the results and logs to")
    parser.add_argument("--run_name", type=str, default=None,
                        help="A name for the training run, used by WandB and other integrations")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Number of evaluation calls with no improvement after which training will be stopped")
    
    # Load balancing for multi-GPU training
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="Number of subprocesses to use for data loading")
    
    args = parser.parse_args()
    return args


def load_and_prepare_datasets(tokenizer, args):
    """Load and prepare datasets for training and evaluation"""
    try:
        # Check if the dataset is a local path or a registered dataset
        if os.path.exists(args.dataset_name):
            # Load from local path
            dataset = load_dataset("json", data_files=args.dataset_name)
        else:
            # Use Azure ML dataset path if provided
            azure_ml_dataset_path = os.environ.get("AZURE_ML_DATASET_PATH")
            if azure_ml_dataset_path:
                dataset_path = os.path.join(azure_ml_dataset_path, args.dataset_name)
                if os.path.exists(dataset_path):
                    dataset = load_dataset("json", data_files=dataset_path)
                else:
                    # Try to load from Hugging Face hub
                    dataset = load_dataset(args.dataset_name, args.dataset_config_name)
            else:
                # Try to load from Hugging Face hub
                dataset = load_dataset(args.dataset_name, args.dataset_config_name)
        
        # Check if dataset has train and validation splits
        if "train" not in dataset:
            # If only one split, create train/val/test splits
            dataset = dataset["train"].train_test_split(test_size=0.1)
            dataset["validation"] = dataset["test"]
        
        logger.info(f"Loaded dataset: {args.dataset_name}")
        
        # Create custom datasets for training and evaluation
        train_dataset = FunctionCallingDataset(tokenizer, dataset["train"])
        eval_dataset = FunctionCallingDataset(tokenizer, dataset["validation"])
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Eval dataset size: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def train():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Check available disk space
    output_dir = args.output_dir
    model_size_gb = 5.0  # Estimate model size based on model type
    if "mini" in args.model_name_or_path.lower():
        model_size_gb = 5.0
    elif "base" in args.model_name_or_path.lower():
        model_size_gb = 10.0
    elif "instruct" in args.model_name_or_path.lower():
        model_size_gb = 15.0
    else:
        model_size_gb = 20.0  # Default assumption for other model sizes
    
    # Account for checkpoints, optimizer states, and other files
    required_space_gb = model_size_gb * 5  # Multiply by 5 to account for multiple checkpoints, optimizer states, etc.
    
    if not check_disk_space(output_dir, required_space_gb):
        logger.warning(f"Low disk space detected. Need at least {required_space_gb:.1f} GB free space for model checkpoints.")
        proceed = input(f"Only limited disk space available. Continue anyway? (yes/no): ").strip().lower()
        if proceed not in ["yes", "y"]:
            logger.info("Training aborted due to low disk space")
            return
    
    # Also check /tmp space for cache
    check_disk_space("/tmp", 2.0)  # Check if at least 2GB available in /tmp

    # Check if output directory exists, create if not
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Add special tokens if they don't exist
    special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|function|>"]
    special_tokens_dict = {"additional_special_tokens": special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    # Load model
    logger.info(f"Loading model from {args.model_name_or_path}")
    
    # Load model with 4-bit quantization if using LoRA
    if args.use_lora:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32),
            load_in_4bit=True,
            device_map="auto"
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Define LoRA config
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Get PEFT model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        # Load model normally
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32),
        )
    
    # Resize token embeddings to account for new special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Load and prepare datasets
    train_dataset, eval_dataset = load_and_prepare_datasets(tokenizer, args)
    
    # Initialize metrics
    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        
        # Filter out -100 labels (padding/ignored positions)
        valid_mask = labels != -100
        filtered_predictions = predictions[valid_mask]
        filtered_labels = labels[valid_mask]
        
        # Compute accuracy
        return metric.compute(predictions=filtered_predictions, references=filtered_labels)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else None,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        report_to=args.report_to.split(",") if args.report_to else None,
        run_name=args.run_name,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=args.dataloader_num_workers,
        group_by_length=True,  # Group sequences of similar length to minimize padding
        remove_unused_columns=False,  # Required when using custom datasets
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        compute_metrics=compute_metrics,
        callbacks=[
            CustomEarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                max_loss_spike=10.0  # Stop if loss spikes to 10x best loss
            )
        ]
    )

    # Periodically check disk space during training (add disk space monitor)
    class DiskSpaceCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 500 == 0:  # Check every 500 steps
                if not check_disk_space(args.output_dir, required_space_gb):
                    logger.warning("Critical disk space shortage detected during training")
                    control.should_training_stop = True
    
    trainer.add_callback(DiskSpaceCallback())
    
    # Train the model
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Log and save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Save the model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training arguments
    trainer.save_state()
    
    # Evaluate the model
    if eval_dataset is not None:
        logger.info("Evaluating the model...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    
    logger.info("Training completed successfully!")
    return trainer


if __name__ == "__main__":
    train()