# src/data_preparation/tokenize.py

import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/tokenization.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("TokenizationProcess")

class Tokenizer:
    """
    Class to handle tokenization of data for Phi-4 finetuning
    """
    
    def __init__(self, model_name_or_path: str = None, config_path: str = None):
        """
        Initialize the tokenizer
        
        Args:
            model_name_or_path: HuggingFace model name or path to load tokenizer from
            config_path: Path to the config file with model info
        """
        from src.msazure.config import AzureConfig
        
        if config_path:
            # Load model name from config
            config_handler = AzureConfig(config_path)
            config = config_handler.load_config()
            self.model_name_or_path = config['config']['HF_MODEL_NAME_OR_PATH']
            self.hf_token = config['config']['HF_TOKEN']
        else:
            # Use provided model name
            self.model_name_or_path = model_name_or_path or "microsoft/Phi-4-mini-instruct"
            self.hf_token = os.environ.get("HF_TOKEN", None)
        
        logger.info(f"Initializing tokenizer from {self.model_name_or_path}")
        self.tokenizer = self._load_tokenizer()
    
    def _load_tokenizer(self):
        """
        Load the tokenizer from HuggingFace
        
        Returns:
            The loaded tokenizer
        """
        try:
        # Load tokenizer with token if available
            if self.hf_token:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name_or_path,
                        token=self.hf_token,
                        trust_remote_code=True
                    )
                    logger.info(f"Tokenizer loaded successfully with auth token: {tokenizer.__class__.__name__}")
                except Exception as auth_error:
                    logger.error(f"Authentication error with HF token: {str(auth_error)}")
                    logger.info("Attempting to load tokenizer without authentication token...")
                    # Fall back to loading without token
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name_or_path,
                        trust_remote_code=True
                    )
                    logger.info(f"Tokenizer loaded successfully without auth token: {tokenizer.__class__.__name__}")
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True
                )
                logger.info(f"Tokenizer loaded successfully: {tokenizer.__class__.__name__}")
        
            return tokenizer
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            raise
    
    def convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert a list of messages to a prompt format suitable for Phi-4 finetuning
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            Formatted prompt string
        """
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
    
    def tokenize_example(self, example: Dict[str, Any], max_length: int = 2048) -> Dict[str, Any]:
        """
        Tokenize a single example
        
        Args:
            example: Dictionary containing the example data
            max_length: Maximum token sequence length
            
        Returns:
            Dictionary with tokenized inputs
        """
        try:
            # Get messages from example
            messages = example.get("messages", [])
            
            # Convert messages to prompt
            prompt = self.convert_messages_to_prompt(messages)
            
            # Check prompt length before tokenization (rough estimate)
            if len(prompt) > max_length * 4:  # Assuming 4 chars per token on average
                logger.warning(f"Prompt is very long ({len(prompt)} chars), may exceed max token length")
            
            # Tokenize the prompt
            tokenized = self.tokenizer(
                prompt,
                truncation=True,
                padding=False,
                max_length=max_length,
                return_tensors=None
            )
            
            # Convert BatchEncoding to a regular dictionary
            tokenized_dict = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            }
            
            # If there are other keys in the BatchEncoding, add them too
            for key in tokenized.keys():
                if key not in tokenized_dict:
                    tokenized_dict[key] = tokenized[key]
            
            # Convert all numpy arrays and tensors to lists for JSON serialization
            for key, value in tokenized_dict.items():
                if hasattr(value, "tolist"):  # Handle numpy arrays and torch tensors
                    tokenized_dict[key] = value.tolist()
            
            # Add original data to tokenized output
            tokenized_dict["original_example"] = example
            
            # Check if sequence was truncated
            if len(tokenized_dict["input_ids"]) >= max_length:
                logger.warning(f"Example was truncated to {max_length} tokens (original length was longer)")
                tokenized_dict["was_truncated"] = True
            else:
                tokenized_dict["was_truncated"] = False
                
            return tokenized_dict
        except Exception as e:
            logger.error(f"Error tokenizing example: {str(e)}")
            logger.error(f"Problematic example: {example}")
            raise
    
    def tokenize_file(self, input_file: str, output_file: str, max_length: int = 2048) -> None:
        """
        Tokenize all examples in a file
        
        Args:
            input_file: Path to the input JSONL file
            output_file: Path to the output JSONL file
            max_length: Maximum token sequence length
        """
        try:
            logger.info(f"Tokenizing examples from {input_file} with max_length={max_length}")
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Load and tokenize examples
            tokenized_examples = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        example = json.loads(line.strip())
                        tokenized = self.tokenize_example(example, max_length)
                        tokenized_examples.append(tokenized)
                    except Exception as e:
                        logger.error(f"Error processing example {i}: {str(e)}")
                        continue
            
            # Save tokenized examples
            with open(output_file, 'w', encoding='utf-8') as f:
                for tokenized in tokenized_examples:
                    f.write(json.dumps(tokenized) + '\n')
            
            logger.info(f"Tokenized {len(tokenized_examples)} examples and saved to {output_file}")
        except Exception as e:
            logger.error(f"Error tokenizing file: {str(e)}")
            raise
    
    def tokenize_dataset(self, input_dir: str, output_dir: str, max_length: int = 2048) -> Dict[str, str]:
        """
        Tokenize all splits in a dataset
        
        Args:
            input_dir: Path to the directory containing the dataset splits
            output_dir: Path to the directory where to save the tokenized dataset
            max_length: Maximum token sequence length
            
        Returns:
            Dictionary with paths to the tokenized dataset files
        """
        try:
            logger.info(f"Tokenizing dataset from {input_dir} with max_length={max_length}")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Paths to splits
            train_input = os.path.join(input_dir, "train", "train.jsonl")
            val_input = os.path.join(input_dir, "val", "val.jsonl")
            test_input = os.path.join(input_dir, "test", "test.jsonl")
            
            train_output = os.path.join(output_dir, "train", "train_tokenized.jsonl")
            val_output = os.path.join(output_dir, "val", "val_tokenized.jsonl")
            test_output = os.path.join(output_dir, "test", "test_tokenized.jsonl")
            
            # Create output subdirectories
            os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
            
            # Tokenize each split
            logger.info(f"Tokenizing training set (max_length={max_length})")
            self.tokenize_file(train_input, train_output, max_length)
            
            logger.info(f"Tokenizing validation set (max_length={max_length})")
            self.tokenize_file(val_input, val_output, max_length)
            
            logger.info(f"Tokenizing test set (max_length={max_length})")
            self.tokenize_file(test_input, test_output, max_length)
            
            logger.info("Dataset tokenization completed successfully")
            
            # Return paths to tokenized files
            return {
                "train": train_output,
                "val": val_output,
                "test": test_output
            }
        except Exception as e:
            logger.error(f"Error tokenizing dataset: {str(e)}")
            raise
    
    def analyze_token_lengths(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze token lengths in a file
        
        Args:
            file_path: Path to the tokenized JSONL file
            
        Returns:
            Dictionary with statistics about token lengths
        """
        try:
            logger.info(f"Analyzing token lengths in {file_path}")
            
            lengths = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    example = json.loads(line.strip())
                    lengths.append(len(example.get("input_ids", [])))
            
            # Calculate statistics
            min_length = min(lengths) if lengths else 0
            max_length = max(lengths) if lengths else 0
            avg_length = sum(lengths) / len(lengths) if lengths else 0
            
            stats = {
                "count": len(lengths),
                "min_length": min_length,
                "max_length": max_length,
                "avg_length": avg_length,
                "p90_length": sorted(lengths)[int(0.9 * len(lengths))] if lengths else 0,
                "p95_length": sorted(lengths)[int(0.95 * len(lengths))] if lengths else 0,
                "p99_length": sorted(lengths)[int(0.99 * len(lengths))] if lengths else 0,
            }
            
            logger.info(f"Token length statistics: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error analyzing token lengths: {str(e)}")
            raise


def main():
    """
    Main function to test tokenization
    """
    # Initialize tokenizer
    tokenizer = Tokenizer(config_path="configs/azure_config.yaml")
    
    # Define input and output directories
    input_dir = "data/processed"
    output_dir = "data/processed/tokenized"
    
    try:
        # Tokenize dataset
        tokenized_paths = tokenizer.tokenize_dataset(input_dir, output_dir)
        
        print("\n" + "="*50)
        print("Dataset tokenization completed successfully!")
        print(f"Tokenized train set: {tokenized_paths['train']}")
        print(f"Tokenized validation set: {tokenized_paths['val']}")
        print(f"Tokenized test set: {tokenized_paths['test']}")
        print("="*50 + "\n")
        
        # Analyze token lengths
        for split, path in tokenized_paths.items():
            stats = tokenizer.analyze_token_lengths(path)
            print(f"{split.capitalize()} set statistics:")
            print(f"  - Count: {stats['count']} examples")
            print(f"  - Token length: min={stats['min_length']}, max={stats['max_length']}, avg={stats['avg_length']:.1f}")
            print(f"  - 90th percentile: {stats['p90_length']} tokens")
            print(f"  - 95th percentile: {stats['p95_length']} tokens")
            print(f"  - 99th percentile: {stats['p99_length']} tokens")
            print()
    except Exception as e:
        print(f"Error during tokenization: {str(e)}")


if __name__ == "__main__":
    main()