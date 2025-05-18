# src/azure/config.py

import os
import yaml
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/azure_config.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("AzureConfig")

class AzureConfig:
    """
    Class to handle Azure configuration loading and validation
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize Azure configuration
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path or os.path.join("configs", "azure_config.yaml")
        self.config = None
        self.resource_group = None
        self.subscription_id = None
        self.workspace = None
        self.sft_data_name = None
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Returns:
            Dict containing the configuration
        """
        try:
            logger.info(f"Loading configuration from {self.config_path}")

                        # Check if file exists
            if not os.path.exists(self.config_path):
                logger.error(f"Configuration file {self.config_path} not found")
                raise FileNotFoundError(f"Configuration file {self.config_path} not found")
            
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            
            # Validate required fields
            required_fields = [
                ('config', 'AZURE_SUBSCRIPTION_ID'),
                ('config', 'AZURE_RESOURCE_GROUP'),
                ('config', 'AZURE_WORKSPACE'),
                ('config', 'HF_MODEL_NAME_OR_PATH'),
                ('train', 'azure_compute_cluster_size'),
                ('train', 'azure_env_name'),
                ('train', 'epoch')
            ]

            for section, field in required_fields:
                if section not in self.config:
                    logger.error(f"Missing section '{section}' in configuration")
                    raise ValueError(f"Missing section '{section}' in configuration")
                    
                if field not in self.config[section]:
                    logger.error(f"Missing required field '{field}' in section '{section}'")
                    raise ValueError(f"Missing required field '{field}' in section '{section}'")

            # Extract key Azure details
            self.subscription_id = self.config['config']['AZURE_SUBSCRIPTION_ID']
            self.resource_group = self.config['config']['AZURE_RESOURCE_GROUP']
            self.workspace = self.config['config']['AZURE_WORKSPACE']
            self.sft_data_name = self.config['config']['AZURE_SFT_DATA_NAME']
            
            return self.config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def display_config(self) -> None:
        """
        Display the loaded configuration to the user
        """
        if not self.config:
            logger.warning("Configuration not loaded. Please call load_config() first.")
            return
        
        print("\n" + "="*50)
        print("Azure Configuration:")
        print("="*50)
        print(f"Subscription ID: {self.subscription_id}")
        print(f"Resource Group: {self.resource_group}")
        print(f"Workspace: {self.workspace}")
        print(f"SFT Data Name: {self.sft_data_name}")
        print("\nTraining Configuration:")
        print(f"Model: {self.config['config']['HF_MODEL_NAME_OR_PATH']}")
        print(f"Environment Name: {self.config['train']['azure_env_name']}")
        print(f"Compute Cluster: {self.config['train']['azure_compute_cluster_name']}")
        print(f"VM Size: {self.config['train']['azure_compute_cluster_size']}")
        print(f"Training Epochs: {self.config['train']['epoch']}")
        print(f"Batch Size: {self.config['train']['train_batch_size']}")
        print("="*50 + "\n")
    
    def confirm_config(self) -> bool:
        """
        Ask user to confirm the configuration
        
        Returns:
            Boolean indicating if the user confirmed the configuration
        """
        self.display_config()
        confirmation = input("Is this configuration correct? (yes/no): ").strip().lower()
        
        if confirmation in ('yes', 'y'):
            logger.info("Configuration confirmed by user")
            return True
        else:
            logger.warning("Configuration rejected by user")
            return False
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration dictionary
        
        Returns:
            Dict containing the configuration
        """
        if not self.config:
            self.load_config()
        return self.config


def main():
    """
    Main function to test the configuration loading
    """
    config_handler = AzureConfig()
    config_handler.load_config()
    confirmed = config_handler.confirm_config()
    
    if confirmed:
        print("Proceeding with the configuration...")
    else:
        print("Please update the configuration and try again.")


if __name__ == "__main__":
    main()