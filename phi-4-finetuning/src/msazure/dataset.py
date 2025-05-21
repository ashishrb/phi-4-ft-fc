# src/azure/dataset.py (updated)

print("Executing dataset.py")
print("Trying to import azure.ai.ml")
try:
    import azure.ai.ml
    print("Successfully imported azure.ai.ml!")
except ImportError as e:
    print(f"Failed to import azure.ai.ml: {e}")

import os
import yaml
import logging
from typing import Dict, Any, Optional
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

from src.msazure.config import AzureConfig
from src.data_preparation.preprocess import DataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/azure_dataset.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("AzureDataset")

class AzureDatasetManager:
    """
    Class to handle Azure ML dataset registration and management
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize Azure Dataset Manager
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_handler = AzureConfig(config_path)
        self.config = self.config_handler.load_config()
        self.ml_client = None
        self.dataset = None
        
    def initialize_ml_client(self) -> None:
        """
        Initialize the Azure ML client
        """
        try:
            logger.info("Initializing Azure ML client")
            
            # Get Azure credentials
            credential = DefaultAzureCredential()
            
            # Create ML client
            self.ml_client = MLClient(
                credential=credential,
                subscription_id=self.config['config']['AZURE_SUBSCRIPTION_ID'],
                resource_group_name=self.config['config']['AZURE_RESOURCE_GROUP'],
                workspace_name=self.config['config']['AZURE_WORKSPACE']
            )
            
            logger.info("Azure ML client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Azure ML client: {str(e)}")
            raise
    
    def prepare_dataset(self, input_file: str = None, output_dir: str = None) -> Dict[str, str]:
        """
        Prepare the dataset by preprocessing and splitting
        
        Args:
            input_file: Path to the input JSONL file
            output_dir: Path to the output directory
            
        Returns:
            Dictionary with paths to the output files
        """
        logger.info("Preparing dataset for registration")
        
        input_file = input_file or os.path.join("data", "raw", "function_calling_dataset.jsonl")
        output_dir = output_dir or self.config['config']['SFT_DATA_DIR']
        
        # Initialize data preprocessor
        preprocessor = DataPreprocessor(input_file=input_file, output_dir=output_dir)
        
        # Validate data
        if not preprocessor.validate_data():
            raise ValueError("Data validation failed. Please check the input data format.")
        
        # Preprocess and split data
        output_paths = preprocessor.preprocess_and_split()
        logger.info(f"Dataset prepared successfully: {output_paths}")
        
        return output_paths
    
    def register_dataset(self, data_path: Optional[str] = None, preprocess: bool = True, data_format: str = "jsonl") -> Data:
        """
        Register a dataset in Azure ML
        
        Args:
            data_path: Path to the dataset file or directory to register
                    If not provided, uses the path from config
            preprocess: Whether to preprocess the data before registration
            data_format: Format of the data (jsonl, csv, tsv, txt, etc.)
        
        Returns:
            Data: The registered dataset
        """
        try:
            if not self.ml_client:
                self.initialize_ml_client()
            
            # Preprocess data if required
            if preprocess:
                output_paths = self.prepare_dataset()
                # Use the processed directory as data path
                data_path = self.config['config']['SFT_DATA_DIR']
            else:
                # Use provided path or default from config
                data_path = data_path or self.config['config']['SFT_DATA_DIR']
            
            data_name = self.config['config']['AZURE_SFT_DATA_NAME']
            
            logger.info(f"Registering dataset '{data_name}' from {data_path}")
            
            # Check if path exists
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset path {data_path} does not exist")
            
            # Validate data format
            if os.path.isdir(data_path):
                # Check if directory contains files of the specified format
                found_files = False
                for root, _, files in os.walk(data_path):
                    for file in files:
                        if file.endswith(f".{data_format}"):
                            found_files = True
                            break
                    if found_files:
                        break
                        
                if not found_files:
                    logger.warning(f"No files with .{data_format} extension found in {data_path}")
                    
                    # Try to detect format
                    extensions = {}
                    for root, _, files in os.walk(data_path):
                        for file in files:
                            if "." in file:
                                ext = file.split(".")[-1].lower()
                                extensions[ext] = extensions.get(ext, 0) + 1
                    
                    if extensions:
                        most_common_ext = max(extensions.items(), key=lambda x: x[1])[0]
                        logger.info(f"Most common file extension detected: .{most_common_ext}")
                        data_format = most_common_ext
                        
            # Determine if directory or file
            if os.path.isdir(data_path):
                # Register directory
                self.dataset = Data(
                    name=data_name,
                    version="1",
                    description=f"Phi-4 finetuning dataset: {data_name} ({data_format} format)",
                    path=data_path,
                    type=AssetTypes.URI_FOLDER
                )
            else:
                # Register file
                self.dataset = Data(
                    name=data_name,
                    version="1",
                    description=f"Phi-4 finetuning dataset: {data_name} ({data_format} format)",
                    path=data_path,
                    type=AssetTypes.URI_FILE
                )
            
            # Register the dataset
            registered_dataset = self.ml_client.data.create_or_update(self.dataset)
            logger.info(f"Dataset registered successfully with id: {registered_dataset.id}")
            
            # Verify dataset registration
            try:
                verification = self.ml_client.data.get(name=data_name, version=registered_dataset.version)
                if verification and verification.id == registered_dataset.id:
                    logger.info(f"Dataset registration verified: {verification.id}")
                else:
                    logger.warning(f"Dataset registration verification failed. Retrieved ID: {verification.id if verification else 'None'}")
            except Exception as e:
                logger.warning(f"Could not verify dataset registration: {str(e)}")
            
            print("\n" + "="*50)
            print(f"Dataset '{data_name}' registered successfully!")
            print(f"Dataset ID: {registered_dataset.id}")
            print(f"Dataset Version: {registered_dataset.version}")
            print(f"Dataset Format: {data_format}")
            print("="*50 + "\n")
            
            return registered_dataset
        
        except Exception as e:
            logger.error(f"Error registering dataset: {str(e)}")
            raise
    
    def get_dataset(self, name: Optional[str] = None, version: str = "latest") -> Data:
        """
        Get a registered dataset from Azure ML
        
        Args:
            name: Name of the dataset to retrieve. If not provided, uses the name from config
            version: Version of the dataset to retrieve
        
        Returns:
            Data: The registered dataset
        """
        try:
            if not self.ml_client:
                self.initialize_ml_client()
            
            name = name or self.config['config']['AZURE_SFT_DATA_NAME']
            
            logger.info(f"Getting dataset '{name}' (version: {version})")
            
            dataset = self.ml_client.data.get(name=name, version=version)
            logger.info(f"Dataset retrieved successfully: {dataset.id}")
            
            return dataset
        
        except Exception as e:
            logger.error(f"Error getting dataset: {str(e)}")
            raise


def main():
    """
    Main function to test dataset preparation and registration
    """
    # Initialize dataset manager
    dataset_manager = AzureDatasetManager()
    
    # Confirm configuration
    if not dataset_manager.config_handler.confirm_config():
        print("Please update the configuration and try again.")
        return
    
    # Prepare and register dataset
    try:
        registered_dataset = dataset_manager.register_dataset(preprocess=True)
        print(f"Dataset registered successfully: {registered_dataset.id}")
    except Exception as e:
        print(f"Error preparing and registering dataset: {str(e)}")


if __name__ == "__main__":
    main()