# src/azure/job.py

import os
import logging
import time
from typing import Dict, Any, Optional, List
import json

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment, CommandJob, AmlCompute
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError

from src.azure.config import AzureConfig
from src.azure.environment import AzureEnvironmentManager
from src.azure.dataset import AzureDatasetManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/azure_job.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("AzureJob")

class AzureJobManager:
    """
    Class to handle Azure ML job creation and management
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize Azure Job Manager
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_handler = AzureConfig(config_path)
        self.config = self.config_handler.load_config()
        self.ml_client = None
        self.env_manager = AzureEnvironmentManager(config_path)
        self.dataset_manager = AzureDatasetManager(config_path)
        
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
    
    def ensure_compute_exists(self) -> str:
        """
        Ensure that the compute cluster exists, create it if it doesn't
        
        Returns:
            The name of the compute cluster
        """
        try:
            if not self.ml_client:
                self.initialize_ml_client()
            
            compute_name = self.config['train']['azure_compute_cluster_name']
            vm_size = self.config['train']['azure_compute_cluster_size']
            
            try:
                # Try to get the compute target to check if it exists
                compute = self.ml_client.compute.get(compute_name)
                logger.info(f"Compute cluster '{compute_name}' already exists")
                
                # Check compute status
                if compute.provisioning_state == "Failed":
                    logger.error(f"Compute cluster '{compute_name}' is in a failed state: {compute.provisioning_errors}")
                    raise Exception(f"Compute cluster '{compute_name}' is in a failed state: {compute.provisioning_errors}")
                
                # Check if VM size matches
                if compute.size.lower() != vm_size.lower():
                    logger.warning(f"Existing compute cluster '{compute_name}' has VM size '{compute.size}' but config specifies '{vm_size}'")
                
                return compute_name
                
            except ResourceNotFoundError:
                # Check quotas before creating
                self._check_compute_quota(vm_size)
                
                # Create the compute target if it doesn't exist
                logger.info(f"Creating compute cluster '{compute_name}'")
                
                # Determine if low priority VM should be used
                low_priority = self.config['config'].get('USE_LOWPRIORITY_VM', False)
                
                # Create compute
                compute = AmlCompute(
                    name=compute_name,
                    size=vm_size,
                    min_instances=0,
                    max_instances=1,
                    idle_time_before_scale_down=120,
                    tier="Dedicated",
                    priority="LowPriority" if low_priority else "Normal"
                )
                
                try:
                    operation = self.ml_client.compute.begin_create_or_update(compute)
                    created_compute = operation.result()
                    logger.info(f"Compute cluster '{compute_name}' created successfully")
                    return compute_name
                except Exception as creation_error:
                    # Check if the error is due to quota limits
                    error_message = str(creation_error).lower()
                    if "quota" in error_message or "limit" in error_message:
                        logger.error(f"Failed to create compute due to quota limits: {error_message}")
                        raise Exception(f"Failed to create compute due to quota limits. Please request a quota increase for VM size {vm_size} or choose a different VM size.")
                    else:
                        logger.error(f"Failed to create compute: {error_message}")
                        raise
            
        except Exception as e:
            logger.error(f"Error ensuring compute exists: {str(e)}")
            raise

    def _check_compute_quota(self, vm_size: str) -> None:
        """
        Check if there's enough quota for the specified VM size
        
        Args:
            vm_size: VM size to check quota for
        """
        try:
            # A100 specific check
            if "a100" in vm_size.lower():
                logger.info(f"Checking quota for A100 GPU VM: {vm_size}")
                
                # Get subscription and region from config
                subscription_id = self.config['config']['AZURE_SUBSCRIPTION_ID']
                
                # We can't directly query quota through the SDK, so we'll just log a warning
                logger.warning(f"A100 GPUs have limited availability. Please ensure your subscription has enough quota for {vm_size} VMs.")
                print(f"\nNOTE: A100 GPUs have limited availability. Please ensure your subscription {subscription_id} has enough quota for {vm_size} VMs.\n")
            
            # General advice for checking quotas
            print("\nNOTE: If job submission fails due to quota issues, you can check your quotas in the Azure Portal:")
            print("1. Go to https://portal.azure.com")
            print("2. Navigate to 'Subscriptions' > Your subscription > 'Usage + quotas'")
            print("3. Check for the relevant VM family quota\n")
            
        except Exception as e:
            logger.warning(f"Error checking compute quota: {str(e)}")
            # Continue anyway as this is just a warning
    
    def display_and_confirm_hyperparameters(self) -> bool:
        """
        Display the training hyperparameters and ask for user confirmation
        
        Returns:
            Boolean indicating if the user confirmed the hyperparameters
        """
        try:
            # Extract hyperparameters from config
            hyperparams = {
                "model_name": self.config['config']['HF_MODEL_NAME_OR_PATH'],
                "training_data": self.config['config']['AZURE_SFT_DATA_NAME'],
                "compute": self.config['train']['azure_compute_cluster_name'],
                "vm_size": self.config['train']['azure_compute_cluster_size'],
                "num_epochs": self.config['train']['epoch'],
                "learning_rate": self.config['train']['learning_rate'],
                "train_batch_size": self.config['train']['train_batch_size'],
                "eval_batch_size": self.config['train']['eval_batch_size'],
                "warmup_ratio": self.config['train']['warmup_ratio'],
                "weight_decay": self.config['train']['weight_decay'],
                "gradient_accumulation_steps": self.config['train']['gradient_accumulation_steps'],
                "lr_scheduler_type": self.config['train']['lr_scheduler_type'],
                "fp16": self.config['train']['fp16']
            }
            
            # Display hyperparameters
            print("\n" + "="*50)
            print("Finetuning Hyperparameters:")
            print("="*50)
            for key, value in hyperparams.items():
                print(f"{key}: {value}")
            print("="*50 + "\n")
            
            # Ask for confirmation
            confirmation = input("Do you want to proceed with these hyperparameters? (yes/no): ").strip().lower()
            
            return confirmation in ('yes', 'y')
        except Exception as e:
            logger.error(f"Error displaying hyperparameters: {str(e)}")
            raise
    
    def prepare_training_script_args(self) -> List[str]:
        """
        Prepare the arguments for the training script
        
        Returns:
            List of arguments for the training script
        """
        args = [
            f"--model_name_or_path={self.config['config']['HF_MODEL_NAME_OR_PATH']}",
            f"--dataset_name={self.config['config']['AZURE_SFT_DATA_NAME']}",
            f"--output_dir=${{output_dir}}",
            f"--num_train_epochs={self.config['train']['epoch']}",
            f"--per_device_train_batch_size={self.config['train']['train_batch_size']}",
            f"--per_device_eval_batch_size={self.config['train']['eval_batch_size']}",
            f"--learning_rate={self.config['train']['learning_rate']}",
            f"--warmup_ratio={self.config['train']['warmup_ratio']}",
            f"--weight_decay={self.config['train']['weight_decay']}",
            f"--logging_steps={self.config['train']['logging_steps']}",
            f"--gradient_accumulation_steps={self.config['train']['gradient_accumulation_steps']}",
            f"--lr_scheduler_type={self.config['train']['lr_scheduler_type']}",
            f"--evaluation_strategy={self.config['train']['evaluation_strategy']}",
            f"--save_strategy={self.config['train']['save_strategy']}",
            f"--save_total_limit={self.config['train']['save_total_limit']}",
        ]
        
        # Add fp16 flag if enabled
        if self.config['train']['fp16']:
            args.append("--fp16")
        
        # Add W&B configuration if API key is provided
        wandb_api_key = self.config['train'].get('wandb_api_key')
        if wandb_api_key and wandb_api_key.strip():
            args.append(f"--report_to=wandb")
            args.append(f"--run_name=phi4-finetune-{int(time.time())}")
        else:
            args.append("--report_to=tensorboard")
        
        return args
    
    def submit_finetuning_job(self) -> Any:
        """
        Submit a finetuning job to Azure ML
        
        Returns:
            The submitted job
        """
        try:
            if not self.ml_client:
                self.initialize_ml_client()
            
            # Ensure compute exists
            compute_name = self.ensure_compute_exists()
            
            # Check compute status before submitting
            self._verify_compute_resources(compute_name)
            
            # Ensure environment exists
            try:
                environment = self.env_manager.get_environment()
                logger.info(f"Using existing environment: {environment.name} (version: {environment.version})")
            except Exception:
                logger.info("Environment not found, creating a new one")
                environment = self.env_manager.create_environment(wait_for_completion=True)
            
            # Ensure dataset exists
            try:
                dataset = self.dataset_manager.get_dataset()
                logger.info(f"Using existing dataset: {dataset.name} (version: {dataset.version})")
            except Exception:
                logger.info("Dataset not found, creating a new one")
                dataset = self.dataset_manager.register_dataset()
            
            # Prepare job name
            job_name = f"phi4-finetune-{int(time.time())}"
            
            # Prepare training script args
            script_args = self.prepare_training_script_args()
            
            # Set environment variables
            env_vars = {}
            
            # Add W&B API key if provided
            wandb_api_key = self.config['train'].get('wandb_api_key')
            if wandb_api_key and wandb_api_key.strip():
                env_vars["WANDB_API_KEY"] = wandb_api_key
                env_vars["WANDB_PROJECT"] = self.config['train'].get('wandb_project', 'phi4-finetuning')
                env_vars["WANDB_WATCH"] = self.config['train'].get('wandb_watch', 'gradients')
            
            # Add HF token if provided
            hf_token = self.config['config'].get('HF_TOKEN')
            if hf_token and hf_token.strip():
                env_vars["HF_TOKEN"] = hf_token
            
            # Create the job
            job = command(
                name=job_name,
                display_name="Phi-4 Finetuning",
                description="Finetuning Phi-4 model on custom dataset",
                compute=compute_name,
                environment=f"{environment.name}:{environment.version}",
                code="./src",  # Root directory for the code
                command="python training/train.py " + " ".join(script_args),
                environment_variables=env_vars,
                inputs={
                    "dataset": dataset
                },
                outputs={
                    "output_dir": None  # This will create a new output
                },
                # Add resource requirements
                resources={
                    "instance_count": 1
                }
            )
            
            # Submit the job
            returned_job = self.ml_client.jobs.create_or_update(job)
            job_url = returned_job.studio_url
            
            logger.info(f"Job '{job_name}' submitted successfully")
            logger.info(f"Job URL: {job_url}")
            
            print("\n" + "="*50)
            print(f"Finetuning job '{job_name}' submitted successfully!")
            print(f"Job ID: {returned_job.id}")
            print(f"Job URL: {job_url}")
            print("="*50 + "\n")
            
            return returned_job
        
        except Exception as e:
            logger.error(f"Error submitting finetuning job: {str(e)}")
            raise

    def _verify_compute_resources(self, compute_name: str) -> None:
        """
        Verify that compute resources are available
        
        Args:
            compute_name: Name of the compute cluster
        """
        try:
            # Get compute
            compute = self.ml_client.compute.get(compute_name)
            
            # Check compute status
            if compute.provisioning_state not in ["Succeeded", "Updating"]:
                logger.warning(f"Compute cluster '{compute_name}' is in state '{compute.provisioning_state}', which may cause job submission to fail")
                print(f"\nWARNING: Compute cluster '{compute_name}' is in state '{compute.provisioning_state}'")
                print("Job submission may fail if the compute is not in a ready state.")
                
                # Ask user if they want to continue
                continue_anyway = input("Do you want to continue with job submission anyway? (yes/no): ").strip().lower()
                if continue_anyway not in ['yes', 'y']:
                    raise Exception(f"Job submission canceled due to compute cluster state: {compute.provisioning_state}")
            
            # Check if compute is scaled to 0 nodes
            current_nodes = getattr(compute, "scale_settings", {}).get("current_node_count", 0)
            if current_nodes == 0:
                logger.info(f"Compute cluster '{compute_name}' is currently scaled to 0 nodes. It will automatically scale up when the job is submitted.")
                print(f"\nNOTE: Compute cluster '{compute_name}' is currently scaled to 0 nodes.")
                print("It will automatically scale up when the job is submitted, which may take a few minutes.\n")
            
            logger.info(f"Compute resources verified for '{compute_name}'")
            
        except Exception as e:
            logger.error(f"Error verifying compute resources: {str(e)}")
            raise
    
    def monitor_job(self, job_id: str, poll_interval_seconds: int = 30) -> Dict[str, Any]:
        """
        Monitor a job until it completes
        
        Args:
            job_id: The job ID to monitor
            poll_interval_seconds: Interval in seconds to poll for job status
            
        Returns:
            Dictionary with job status and metrics
        """
        try:
            if not self.ml_client:
                self.initialize_ml_client()
            
            logger.info(f"Monitoring job {job_id}")
            print(f"Monitoring job {job_id}")
            
            while True:
                job = self.ml_client.jobs.get(job_id)
                status = job.status
                
                # Display current status
                print(f"Job status: {status}")
                
                if status in ["Failed", "Canceled", "NotResponding"]:
                    logger.error(f"Job failed with status: {status}")
                    print(f"Job failed with status: {status}")
                    print(f"Error details: {job.error}")
                    return {"status": status, "error": job.error}
                
                if status == "Completed":
                    logger.info(f"Job completed successfully")
                    print(f"Job completed successfully")
                    
                    # Get metrics
                    metrics = {}
                    try:
                        # Try to get metrics from the job
                        # This depends on how metrics are logged in the training script
                        run = self.ml_client.jobs.get(job_id)
                        metrics = run.metrics
                    except Exception as e:
                        logger.warning(f"Error getting metrics: {str(e)}")
                    
                    return {"status": status, "metrics": metrics}
                
                # If job is still running, wait and check again
                time.sleep(poll_interval_seconds)
        
        except Exception as e:
            logger.error(f"Error monitoring job: {str(e)}")
            raise
    
    def get_job_metrics(self, job_id: str) -> Dict[str, Any]:
        """
        Get metrics from a completed job
        
        Args:
            job_id: The job ID to get metrics for
            
        Returns:
            Dictionary with job metrics
        """
        try:
            if not self.ml_client:
                self.initialize_ml_client()
            
            logger.info(f"Getting metrics for job {job_id}")
            
            run = self.ml_client.jobs.get(job_id)
            
            # Get metrics
            metrics = {}
            try:
                metrics = run.metrics
            except Exception as e:
                logger.warning(f"Error getting metrics: {str(e)}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error getting job metrics: {str(e)}")
            raise


def main():
    """
    Main function to test job submission and monitoring
    """
    # Initialize job manager
    job_manager = AzureJobManager()
    
    # Confirm configuration
    if not job_manager.config_handler.confirm_config():
        print("Please update the configuration and try again.")
        return
    
    # Display and confirm hyperparameters
    if not job_manager.display_and_confirm_hyperparameters():
        print("Please update the hyperparameters and try again.")
        return
    
    # Submit finetuning job
    try:
        job = job_manager.submit_finetuning_job()
        print(f"Finetuning job submitted successfully with ID: {job.id}")
        
        # Ask if user wants to monitor the job
        monitor = input("Do you want to monitor the job until completion? (yes/no): ").strip().lower()
        if monitor in ('yes', 'y'):
            job_status = job_manager.monitor_job(job.id)
            
            if job_status["status"] == "Completed":
                print("\n" + "="*50)
                print("Finetuning completed successfully!")
                print("Metrics:")
                for key, value in job_status.get("metrics", {}).items():
                    print(f"  {key}: {value}")
                print("="*50 + "\n")
            else:
                print(f"Job ended with status: {job_status['status']}")
                if "error" in job_status:
                    print(f"Error: {job_status['error']}")
    
    except Exception as e:
        print(f"Error submitting or monitoring finetuning job: {str(e)}")


if __name__ == "__main__":
    main()