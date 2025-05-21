# src/data_preparation/preprocess.py

import os
import json
import logging
import random
from typing import Dict, List, Tuple, Any, Optional
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_preprocessing.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("DataPreprocessing")

class DataPreprocessor:
    """
    Class to handle data preprocessing and splitting for Phi-4 finetuning
    """
    
    def __init__(self, 
                 input_file: str = None,
                 output_dir: str = None,
                 train_ratio: float = 0.8, 
                 val_ratio: float = 0.1, 
                 test_ratio: float = 0.1,
                 random_seed: int = 42):
        """
        Initialize the data preprocessor
        
        Args:
            input_file: Path to the input JSONL file
            output_dir: Path to the output directory
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            test_ratio: Ratio of data to use for testing
            random_seed: Random seed for reproducibility
        """
        self.input_file = input_file or os.path.join("data", "raw", "function_calling_dataset.jsonl")
        self.output_dir = output_dir or os.path.join("data", "processed")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)

        # Create output directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "val"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "test"), exist_ok=True)
    
    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load data from a JSONL file
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            List of dictionaries containing the data
        """
        data = []
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File {file_path} not found")
                raise FileNotFoundError(f"File {file_path} not found")
                
            # Check if file is empty
            if os.path.getsize(file_path) == 0:
                logger.error(f"File {file_path} is empty")
                raise ValueError(f"File {file_path} is empty")
        
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))

            # Check if any data was loaded
            if not data:
                logger.error(f"No valid data found in {file_path}")
                raise ValueError(f"No valid data found in {file_path}")
        
            logger.info(f"Loaded {len(data)} examples from {file_path}")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in file {file_path}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def save_jsonl(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """
        Save data to a JSONL file
        
        Args:
            data: List of dictionaries to save
            file_path: Path where to save the JSONL file
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
            logger.info(f"Saved {len(data)} examples to {file_path}")
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {str(e)}")
            raise
    
    def split_data(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split data into train, validation, and test sets
        
        Args:
            data: List of dictionaries containing the data
            
        Returns:
            Tuple with train, validation, and test data
        """
        # Shuffle data
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        
        # Calculate split indices
        train_end = int(len(shuffled_data) * self.train_ratio)
        val_end = train_end + int(len(shuffled_data) * self.val_ratio)
        
        # Split data
        train_data = shuffled_data[:train_end]
        val_data = shuffled_data[train_end:val_end]
        test_data = shuffled_data[val_end:]
        
        logger.info(f"Split data into {len(train_data)} train, {len(val_data)} validation, and {len(test_data)} test examples")
        
        return train_data, val_data, test_data
    
    def preprocess_and_split(self) -> Dict[str, str]:
        """
        Preprocess and split data into train, validation, and test sets
        
        Returns:
            Dictionary with paths to the output files
        """
        try:
            # Load data
            logger.info(f"Loading data from {self.input_file}")
            data = self.load_jsonl(self.input_file)
            
            # Split data
            logger.info("Splitting data into train, validation, and test sets")
            train_data, val_data, test_data = self.split_data(data)
            
            # Save data
            train_path = os.path.join(self.output_dir, "train", "train.jsonl")
            val_path = os.path.join(self.output_dir, "val", "val.jsonl")
            test_path = os.path.join(self.output_dir, "test", "test.jsonl")
            
            logger.info(f"Saving train data to {train_path}")
            self.save_jsonl(train_data, train_path)
            
            logger.info(f"Saving validation data to {val_path}")
            self.save_jsonl(val_data, val_path)
            
            logger.info(f"Saving test data to {test_path}")
            self.save_jsonl(test_data, test_path)
            
            # Also save a copy of the full dataset in the processed directory
            full_dataset_path = os.path.join(self.output_dir, "full_dataset.jsonl")
            logger.info(f"Saving full dataset to {full_dataset_path}")
            shutil.copy(self.input_file, full_dataset_path)
            
            logger.info("Data preprocessing and splitting completed successfully")
            
            # Return paths to the output files
            return {
                "train": train_path,
                "val": val_path,
                "test": test_path,
                "full": full_dataset_path
            }
            
        except Exception as e:
            logger.error(f"Error during data preprocessing and splitting: {str(e)}")
            raise
    
    def validate_data(self) -> bool:
        """
        Validate that the data is in the correct format for Phi-4 finetuning
        
        Returns:
            True if the data is valid, False otherwise
        """
        try:
            # Load a sample of the data
            data = self.load_jsonl(self.input_file)
            
            # Check if each example has the required fields
            for i, example in enumerate(data):
                if "messages" not in example:
                    logger.error(f"Example {i} does not have 'messages' field")
                    return False
                
                # Check if messages field is a list
                if not isinstance(example["messages"], list):
                    logger.error(f"Example {i}: 'messages' field is not a list")
                    return False
                
                # Check if each message has the required fields
                for j, message in enumerate(example["messages"]):
                    if "role" not in message:
                        logger.error(f"Example {i}, message {j} does not have 'role' field")
                        return False
                    
                    if "content" not in message:
                        logger.error(f"Example {i}, message {j} does not have 'content' field")
                        return False
            
            logger.info("Data validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Error during data validation: {str(e)}")
            return False


def main():
    """
    Main function to test data preprocessing
    """
    # Initialize data preprocessor
    preprocessor = DataPreprocessor()
    
    # Validate data
    if not preprocessor.validate_data():
        print("Data validation failed. Please check the input data format.")
        return
    
    # Preprocess and split data
    try:
        output_paths = preprocessor.preprocess_and_split()
        
        print("\n" + "="*50)
        print("Data preprocessing completed successfully!")
        print(f"Train data: {output_paths['train']} ({sum(1 for _ in open(output_paths['train']))} examples)")
        print(f"Validation data: {output_paths['val']} ({sum(1 for _ in open(output_paths['val']))} examples)")
        print(f"Test data: {output_paths['test']} ({sum(1 for _ in open(output_paths['test']))} examples)")
        print("="*50 + "\n")
    except Exception as e:
        print(f"Error during data preprocessing: {str(e)}")


if __name__ == "__main__":
    main()