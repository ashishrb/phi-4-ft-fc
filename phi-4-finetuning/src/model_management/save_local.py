# src/model_management/save_local.py

import os
import logging
import shutil
from typing import Dict, Any, Optional, List
import json
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/model_management.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ModelManagement")

class ModelManager:
    """
    Class to handle model saving and loading
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize Model Manager
        
        Args:
            config_path: Path to the YAML configuration file
        """
        from src.msazure.config import AzureConfig
        
        self.config_handler = AzureConfig(config_path)
        self.config = self.config_handler.load_config()
    
    def download_model_from_azure(self, job_id: str, output_dir: Optional[str] = None, 
                             verify_disk_space: bool = True, compress_model: bool = False) -> str:
        """
        Download a model from Azure ML to local directory
        
        Args:
            job_id: Azure ML job ID
            output_dir: Local directory to save the model. If None, use the one from config
            verify_disk_space: Whether to verify disk space before downloading
            compress_model: Whether to compress the model after downloading
            
        Returns:
            Path to the downloaded model
        """
        try:
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential
            
            logger.info(f"Downloading model from Azure ML job {job_id}")
            
            # Get output directory
            if output_dir is None:
                output_dir = os.path.join(self.config['model_save']['local_dir'], f"job_{job_id}")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Check model size before downloading
            model_size_gb = self._estimate_model_size(job_id)
            
            # Verify disk space if requested
            if verify_disk_space and model_size_gb > 0:
                available_space_gb = self._get_available_disk_space(output_dir)
                
                # Need at least the model size plus 5GB for extraction and processing
                required_space_gb = model_size_gb + 5
                
                if available_space_gb < required_space_gb:
                    logger.error(f"Insufficient disk space. Available: {available_space_gb:.1f} GB, Required: {required_space_gb:.1f} GB")
                    raise ValueError(f"Insufficient disk space. Available: {available_space_gb:.1f} GB, Required: {required_space_gb:.1f} GB")
                
                logger.info(f"Sufficient disk space available ({available_space_gb:.1f} GB, Need {required_space_gb:.1f} GB)")
            
            # Initialize Azure ML client
            credential = DefaultAzureCredential()
            ml_client = MLClient(
                credential=credential,
                subscription_id=self.config['config']['AZURE_SUBSCRIPTION_ID'],
                resource_group_name=self.config['config']['AZURE_RESOURCE_GROUP'],
                workspace_name=self.config['config']['AZURE_WORKSPACE']
            )
            
            # Get the job
            job = ml_client.jobs.get(job_id)
            
            # Get the outputs
            outputs = job.outputs
            
            # Download the model with progress tracking
            if "output_dir" in outputs:
                logger.info(f"Downloading model to {output_dir}")
                
                print(f"\nDownloading model from job {job_id}...")
                print(f"Estimated model size: {model_size_gb:.1f} GB")
                print(f"Target directory: {output_dir}")
                print("This may take several minutes. Please wait...\n")
                
                # Download with progress tracking if tqdm is available
                try:
                    from tqdm import tqdm
                    
                    # Create a progress callback
                    class DownloadProgressCallback:
                        def __init__(self):
                            self.pbar = None
                            self.downloaded = 0
                            self.total_size = model_size_gb * 1024 * 1024 * 1024  # Convert GB to bytes
                        
                        def __call__(self, downloaded_bytes, total_bytes):
                            if self.pbar is None:
                                self.pbar = tqdm(total=total_bytes if total_bytes > 0 else self.total_size, 
                                                unit='B', unit_scale=True)
                            
                            # Update progress
                            delta = downloaded_bytes - self.downloaded
                            if delta > 0:
                                self.pbar.update(delta)
                                self.downloaded = downloaded_bytes
                                
                    callback = DownloadProgressCallback()
                    
                    # Download with progress callback
                    ml_client.jobs.download(
                        name=job_id,
                        output_name="output_dir",
                        download_path=output_dir,
                        progress_callback=callback
                    )
                    
                    if callback.pbar is not None:
                        callback.pbar.close()
                        
                except ImportError:
                    # Fall back to regular download without progress tracking
                    ml_client.jobs.download(
                        name=job_id,
                        output_name="output_dir",
                        download_path=output_dir
                    )
                
                # Compress the model if requested
                if compress_model:
                    compressed_path = self._compress_model(output_dir)
                    logger.info(f"Model compressed to {compressed_path}")
                    print(f"Model compressed to {compressed_path}")
                
                logger.info(f"Model downloaded successfully to {output_dir}")
                print(f"Model downloaded successfully to {output_dir}")
                
                return output_dir
            else:
                logger.error(f"Job {job_id} does not have output_dir")
                raise ValueError(f"Job {job_id} does not have output_dir")
        
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            raise

    def _estimate_model_size(self, job_id: str) -> float:
        """
        Estimate the model size based on the job ID
        
        Args:
            job_id: Azure ML job ID
            
        Returns:
            Estimated model size in GB
        """
        try:
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential
            
            # Initialize Azure ML client
            credential = DefaultAzureCredential()
            ml_client = MLClient(
                credential=credential,
                subscription_id=self.config['config']['AZURE_SUBSCRIPTION_ID'],
                resource_group_name=self.config['config']['AZURE_RESOURCE_GROUP'],
                workspace_name=self.config['config']['AZURE_WORKSPACE']
            )
            
            # Get the job
            job = ml_client.jobs.get(job_id)
            
            # Try to get model size from job properties/metrics if available
            # This is a simplified approach - in a real implementation, you would parse metrics
            
            # Default size estimates based on model type from config
            model_name = self.config['config']['HF_MODEL_NAME_OR_PATH'].lower()
            
            if "mini" in model_name:
                return 5.0  # Phi-4-mini: ~5GB
            elif "base" in model_name:
                return 10.0  # Phi-4-base: ~10GB
            elif "instruct" in model_name:
                return 15.0  # Phi-4-instruct: ~15GB
            else:
                return 20.0  # Default conservative estimate
                
        except Exception as e:
            logger.warning(f"Error estimating model size: {str(e)}")
            return 20.0  # Conservative default if estimation fails

    def _get_available_disk_space(self, directory: str) -> float:
        """
        Get available disk space in GB
        
        Args:
            directory: Directory to check
            
        Returns:
            Available disk space in GB
        """
        try:
            import shutil
            
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            
            # Get disk usage statistics
            total, used, free = shutil.disk_usage(directory)
            
            # Convert to GB
            free_gb = free / (1024 ** 3)
            
            return free_gb
        
        except Exception as e:
            logger.warning(f"Error checking disk space: {str(e)}")
            return 0.0  # Return 0 to indicate unknown space (will trigger verification failure)

    def _compress_model(self, model_dir: str) -> str:
        """
        Compress the model directory
        
        Args:
            model_dir: Directory containing the model
            
        Returns:
            Path to the compressed file
        """
        try:
            import tarfile
            import shutil
            
            # Create a timestamp for the filename
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # Get output path
            output_path = f"{model_dir}_compressed_{timestamp}.tar.gz"
            
            print(f"Compressing model to {output_path}...")
            
            # Compress the directory
            with tarfile.open(output_path, "w:gz") as tar:
                tar.add(model_dir, arcname=os.path.basename(model_dir))
            
            # Get sizes for logging
            original_size = self._get_directory_size(model_dir) / (1024 ** 3)  # Convert to GB
            compressed_size = os.path.getsize(output_path) / (1024 ** 3)  # Convert to GB
            
            logger.info(f"Model compressed from {original_size:.2f} GB to {compressed_size:.2f} GB")
            print(f"Model compressed from {original_size:.2f} GB to {compressed_size:.2f} GB")
            
            # Ask if user wants to delete the original directory
            delete_original = input("Do you want to delete the original uncompressed model directory? (yes/no): ").strip().lower()
            
            if delete_original in ('yes', 'y'):
                shutil.rmtree(model_dir)
                logger.info(f"Original model directory {model_dir} deleted")
                print(f"Original model directory deleted")
            
            return output_path
        
        except Exception as e:
            logger.error(f"Error compressing model: {str(e)}")
            return model_dir  # Return original path if compression fails

    def _get_directory_size(self, directory: str) -> float:
        """
        Get the size of a directory in bytes
        
        Args:
            directory: Directory to measure
            
        Returns:
            Directory size in bytes
        """
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        
        return total_size
    
    def save_model_to_hub(self, model_dir: str, repo_name: Optional[str] = None, private: bool = False) -> str:
        """
        Save a model to Hugging Face Hub
        
        Args:
            model_dir: Directory containing the model
            repo_name: Name of the repository on HF Hub. If None, use the one from config
            private: Whether the repository should be private
            
        Returns:
            URL of the model on HF Hub
        """
        try:
            from huggingface_hub import HfApi
            
            logger.info(f"Saving model to Hugging Face Hub")
            
            # Get HF token
            hf_token = self.config['config'].get('HF_TOKEN')
            
            if not hf_token:
                logger.error("HF_TOKEN not found in config")
                raise ValueError("HF_TOKEN not found in config. Cannot upload to Hugging Face Hub.")
            
            # Get repository name
            if repo_name is None:
                # Extract repo name from config or generate from job ID
                repo_name = self.config['model_save'].get('hf_repo_name')
                
                if not repo_name:
                    # Generate a repo name from directory name
                    repo_name = os.path.basename(model_dir)
            
            # Initialize HF API
            api = HfApi(token=hf_token)
            
            # Create repository if it doesn't exist
            api.create_repo(
                repo_id=repo_name,
                private=private,
                exist_ok=True
            )
            
            # Upload model files
            logger.info(f"Uploading model to {repo_name}")
            api.upload_folder(
                folder_path=model_dir,
                repo_id=repo_name,
                commit_message="Upload finetuned Phi-4 model"
            )
            
            # Get repository URL
            repo_url = f"https://huggingface.co/{repo_name}"
            
            logger.info(f"Model uploaded successfully to {repo_url}")
            print(f"Model uploaded successfully to {repo_url}")
            
            return repo_url
        
        except Exception as e:
            logger.error(f"Error saving model to Hugging Face Hub: {str(e)}")
            raise
    
    def download_model_to_local(self, azure_job_id: str, local_dir: Optional[str] = None,
                           verify_disk_space: bool = True, compress_model: bool = False) -> str:
        """
        Download model from Azure ML to local directory
        
        Args:
            azure_job_id: Azure ML job ID
            local_dir: Local directory to save the model. If None, use the one from config
            verify_disk_space: Whether to verify disk space before downloading
            compress_model: Whether to compress the model after downloading
            
        Returns:
            Path to the downloaded model
        """
        try:
            if local_dir is None:
                local_dir = self.config['model_save'].get('local_dir', os.path.join("results", "local_model"))
                
                if not local_dir:
                    local_dir = os.path.join("results", "local_model", f"job_{azure_job_id}")
            
            # Create local directory
            os.makedirs(local_dir, exist_ok=True)
            
            # Ask if user wants to compress the model
            if not compress_model:
                compress_model = input("Do you want to compress the model after downloading? (yes/no): ").strip().lower() in ('yes', 'y')
            
            # Download model from Azure ML
            model_path = self.download_model_from_azure(
                azure_job_id, 
                local_dir, 
                verify_disk_space=verify_disk_space,
                compress_model=compress_model
            )
            
            print("\n" + "="*50)
            print(f"Model downloaded successfully to {model_path}")
            print(f"You can load this model using:")
            print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
            print(f"  tokenizer = AutoTokenizer.from_pretrained('{model_path}')")
            print(f"  model = AutoModelForCausalLM.from_pretrained('{model_path}')")
            print("="*50 + "\n")
            
            return model_path
        
        except Exception as e:
            logger.error(f"Error downloading model to local: {str(e)}")
            raise
    
    def list_saved_models(self, base_dir: Optional[str] = None) -> List[str]:
        """
        List all saved models in the local directory
        
        Args:
            base_dir: Base directory to look for models. If None, use the one from config
            
        Returns:
            List of model directories
        """
        try:
            if base_dir is None:
                base_dir = self.config['model_save'].get('local_dir', "results/local_model")
            
            # Check if directory exists
            if not os.path.exists(base_dir):
                logger.warning(f"Directory {base_dir} does not exist")
                return []
            
            # Get all subdirectories with model files
            model_dirs = []
            
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                
                # Check if it's a directory and contains model files
                if os.path.isdir(item_path) and self._is_model_dir(item_path):
                    model_dirs.append(item_path)
            
            return model_dirs
        
        except Exception as e:
            logger.error(f"Error listing saved models: {str(e)}")
            raise
    
    def _is_model_dir(self, directory: str) -> bool:
        """
        Check if a directory contains a HuggingFace model
        
        Args:
            directory: Directory to check
            
        Returns:
            True if the directory contains a model, False otherwise
        """
        # Check for common model files
        model_files = ["config.json", "pytorch_model.bin", "model.safetensors"]
        
        for file in model_files:
            if os.path.exists(os.path.join(directory, file)):
                return True
        
        return False


def main():
    """
    Main function to test model management
    """
    # Initialize model manager
    model_manager = ModelManager()
    
    # Confirm configuration
    if not model_manager.config_handler.confirm_config():
        print("Please update the configuration and try again.")
        return
    
    # Ask for job ID
    job_id = input("Enter Azure ML job ID: ").strip()
    
    if not job_id:
        print("Job ID is required.")
        return
    
    # Ask if user wants to download the model
    download = input("Do you want to download the model locally? (yes/no): ").strip().lower()
    
    if download in ('yes', 'y'):
        # Ask for local directory
        local_dir = input("Enter local directory to save the model (leave empty for default): ").strip()
        
        if not local_dir:
            local_dir = None
        
        try:
            # Download model
            model_path = model_manager.download_model_to_local(job_id, local_dir)
            print(f"Model downloaded successfully to {model_path}")
            
            # Ask if user wants to push to HF Hub
            push_to_hub = input("Do you want to push the model to Hugging Face Hub? (yes/no): ").strip().lower()
            
            if push_to_hub in ('yes', 'y'):
                # Ask for repository name
                repo_name = input("Enter repository name (leave empty for default): ").strip()
                
                if not repo_name:
                    repo_name = None
                
                # Ask if repository should be private
                private = input("Should the repository be private? (yes/no): ").strip().lower() in ('yes', 'y')
                
                # Push to hub
                repo_url = model_manager.save_model_to_hub(model_path, repo_name, private)
                print(f"Model pushed successfully to {repo_url}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    else:
        print("Model will not be downloaded.")


if __name__ == "__main__":
    main()