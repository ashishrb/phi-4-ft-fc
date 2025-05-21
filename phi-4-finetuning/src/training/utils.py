# src/training/utils.py

import os
import json
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger("TrainingUtils")

def set_gpu_allocation():
    """Optimize GPU memory allocation"""
    try:
        # First check if CUDA is available
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Running on CPU only.")
            print("NOTE: CUDA is not available. Training will run on CPU, which will be very slow.")
            return False
        
        # Set PyTorch to use TF32 precision if available
        if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer GPU
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled TF32 precision for Ampere or newer GPU")
        
        # Get number of GPUs
        num_gpus = torch.cuda.device_count()
        logger.info(f"Found {num_gpus} GPU(s)")
        
        # Log GPU info
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9  # Convert to GB
            logger.info(f"GPU {i}: {gpu_name} with {gpu_memory:.2f} GB memory")
            
        # For A100, set optimal memory allocation
        for i in range(num_gpus):
            if "A100" in torch.cuda.get_device_name(i):
                # Set memory allocation strategies optimal for A100
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
                logger.info("Set optimal CUDA memory allocation for A100 GPUs")
                break
                
        return True
        
    except Exception as e:
        logger.warning(f"Error in GPU setup: {str(e)}")
        logger.warning("Continuing with default GPU settings")
        return False


def format_metrics(metrics: Dict[str, Any]) -> str:
    """
    Format metrics dictionary as a string for logging
    
    Args:
        metrics: Dictionary with metrics
        
    Returns:
        Formatted string with metrics
    """
    return ", ".join([f"{k}: {v:.4f}" if isinstance(v, (float, np.float32, np.float64)) else f"{k}: {v}" 
                     for k, v in metrics.items()])


def save_model_card(output_dir: str, model_name: str, dataset_name: str, hyperparams: Dict[str, Any]):
    """
    Save a model card with information about the training
    
    Args:
        output_dir: Directory where to save the model card
        model_name: Name of the base model
        dataset_name: Name of the dataset used for finetuning
        hyperparams: Dictionary with training hyperparameters
    """
    # Create model card content - using triple quotes with no nesting
    model_card = f"""# Phi-4 Finetuned Model

This model is a finetuned version of [{model_name}](https://huggingface.co/{model_name}) on the {dataset_name} dataset.

## Training Details

- **Base Model:** {model_name}
- **Dataset:** {dataset_name}
- **Training Type:** Supervised Finetuning
- **Hyperparameters:**
"""
    
    # Add hyperparameters
    for key, value in hyperparams.items():
        if key.startswith("_"):  # Skip private attributes
            continue
        model_card += f"  - {key}: {value}\n"
    
    # Add usage section without nested triple quotes
    model_card += """
## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("PATH_TO_MODEL")
model = AutoModelForCausalLM.from_pretrained("PATH_TO_MODEL")

# Example usage for function calling
messages = [
    {"role": "system", "content": "You are a helpful assistant with function calling capabilities."},
    {"role": "user", "content": "What's the weather like in New York?"}
]

# Convert messages to prompt format
prompt = ""
for message in messages:
    if message["role"] == "system":
        prompt += f"<|system|>\\n{message['content']}\\n"
    elif message["role"] == "user":
        prompt += f"<|user|>\\n{message['content']}\\n"
    elif message["role"] == "assistant":
        prompt += f"<|assistant|>\\n{message['content']}\\n"
    elif message["role"] == "function":
        function_name = message.get("name", "function")
        prompt += f"<|function|>\\nname: {function_name}\\ncontent: {message['content']}\\n"

# Add assistant prefix for generation
prompt += "<|assistant|>\\n"

# Generate response
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(response)
"""

# Save model card to README.md
try:
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(model_card)
    
    logger.info(f"Model card saved to {os.path.join(output_dir, 'README.md')}")
except Exception as e:
    logger.error(f"Error saving model card: {str(e)}")
    logger.warning("Continuing without saving model card")
