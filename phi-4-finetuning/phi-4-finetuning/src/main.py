# src/main.py
import os
import logging
import argparse
import time
import sys
from typing import Dict, Any, Optional

from src.msazure.config import AzureConfig
from src.data_preparation.preprocess import DataPreprocessor
from src.data_preparation.tokenize import Tokenizer
from src.msazure.dataset import AzureDatasetManager
from src.msazure.environment import AzureEnvironmentManager
from src.msazure.job import AzureJobManager
from src.monitoring.metrics import MetricsTracker
from src.model_management.save_local import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "main.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Phi4Finetuning")

class PipelineStatus:
    """Track pipeline status and provide user feedback"""
    
    def __init__(self):
        self.current_step = ""
        self.steps_completed = []
        self.total_steps = 7
        
    def update_status(self, step: str, status: str = "in_progress"):
        self.current_step = step
        if status == "completed":
            self.steps_completed.append(step)
        
        # Display progress
        progress = len(self.steps_completed)
        print(f"\n{'='*60}")
        print(f"🚀 Phi-4 Finetuning Pipeline Progress: {progress}/{self.total_steps}")
        print(f"📋 Current Step: {step}")
        if status == "completed":
            print(f"✅ {step} - COMPLETED")
        elif status == "skipped":
            print(f"⏭️  {step} - SKIPPED (already exists)")
        elif status == "in_progress":
            print(f"⏳ {step} - IN PROGRESS...")
        elif status == "failed":
            print(f"❌ {step} - FAILED")
        print(f"{'='*60}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Phi-4 Finetuning Pipeline - Single Command Execution")
    
    parser.add_argument("--config", type=str, default="configs/azure_config.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--raw_data_path", type=str, default=None,
                        help="Path to the raw data file")
    parser.add_argument("--force_recreate_dataset", action="store_true",
                        help="Force recreation of dataset even if it exists")
    parser.add_argument("--force_recreate_environment", action="store_true",
                        help="Force recreation of environment even if it exists")
    parser.add_argument("--monitoring_interval", type=int, default=60,
                        help="Interval in seconds for job monitoring polling")
    parser.add_argument("--max_monitor_time", type=int, default=480,
                        help="Maximum time in minutes to monitor the job")
    parser.add_argument("--auto_download_model", action="store_true",
                        help="Automatically download model after training completes")
    
    return parser.parse_args()

def check_resource_exists(manager, resource_type: str) -> tuple:
    """Check if a resource already exists"""
    try:
        if resource_type == "dataset":
            resource = manager.get_dataset()
            return True, resource
        elif resource_type == "environment":
            resource = manager.get_environment()
            return True, resource
        else:
            return False, None
    except Exception:
        return False, None

def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize status tracker
    status = PipelineStatus()
    
    try:
        # 1. Load and confirm configuration
        status.update_status("Configuration Loading")
        logger.info(f"Loading configuration from {args.config}")
        config_handler = AzureConfig(args.config)
        config = config_handler.load_config()
        
        if not config_handler.confirm_config():
            logger.info("Configuration not confirmed by user. Exiting.")
            return
        
        status.update_status("Configuration Loading", "completed")
        
        # Set up paths
        raw_data_path = args.raw_data_path or os.path.join("data", "raw", "function_calling_dataset.jsonl")
        
        # 2. Data Preprocessing
        status.update_status("Data Preprocessing")
        
        # Check if processed data already exists
        processed_train_path = os.path.join(config['config']['SFT_DATA_DIR'], "train", "train.jsonl")
        if os.path.exists(processed_train_path) and not args.force_recreate_dataset:
            logger.info("Processed data already exists, skipping preprocessing")
            status.update_status("Data Preprocessing", "skipped")
        else:
            logger.info("Preprocessing data")
            data_processor = DataPreprocessor(input_file=raw_data_path)
            
            # Validate data
            if not data_processor.validate_data():
                logger.error("Data validation failed. Please check the input data.")
                status.update_status("Data Preprocessing", "failed")
                return
            
            # Preprocess and split data
            output_paths = data_processor.preprocess_and_split()
            logger.info(f"Data preprocessed and split successfully: {output_paths}")
            status.update_status("Data Preprocessing", "completed")
        
        # 3. Data Tokenization
        status.update_status("Data Tokenization")
        
        # Check if tokenized data already exists
        tokenized_train_path = os.path.join(config['config']['SFT_DATA_DIR'], "tokenized", "train", "train_tokenized.jsonl")
        if os.path.exists(tokenized_train_path) and not args.force_recreate_dataset:
            logger.info("Tokenized data already exists, skipping tokenization")
            status.update_status("Data Tokenization", "skipped")
        else:
            logger.info("Tokenizing data")
            tokenizer = Tokenizer(config_path=args.config)
            
            tokenized_paths = tokenizer.tokenize_dataset(
                input_dir=config['config']['SFT_DATA_DIR'],
                output_dir=os.path.join(config['config']['SFT_DATA_DIR'], "tokenized")
            )
            
            logger.info(f"Data tokenized successfully: {tokenized_paths}")
            status.update_status("Data Tokenization", "completed")
        
        # 4. Dataset Registration
        status.update_status("Dataset Registration")
        
        dataset_manager = AzureDatasetManager(args.config)
        exists, existing_dataset = check_resource_exists(dataset_manager, "dataset")
        
        if exists and not args.force_recreate_dataset:
            logger.info(f"Dataset already exists: {existing_dataset.name} (version: {existing_dataset.version})")
            registered_dataset = existing_dataset
            status.update_status("Dataset Registration", "skipped")
        else:
            logger.info("Registering dataset in Azure ML")
            registered_dataset = dataset_manager.register_dataset(preprocess=False)
            logger.info(f"Dataset registered successfully: {registered_dataset.id}")
            status.update_status("Dataset Registration", "completed")
        
        # 5. Environment Creation
        status.update_status("Environment Creation")
        
        env_manager = AzureEnvironmentManager(args.config)
        exists, existing_env = check_resource_exists(env_manager, "environment")
        
        if exists and not args.force_recreate_environment:
            logger.info(f"Environment already exists: {existing_env.name} (version: {existing_env.version})")
            created_env = existing_env
            status.update_status("Environment Creation", "skipped")
        else:
            logger.info("Creating environment in Azure ML")
            created_env = env_manager.create_environment(wait_for_completion=True)
            logger.info(f"Environment created successfully: {created_env.id}")
            status.update_status("Environment Creation", "completed")
        
        # 6. Job Submission
        status.update_status("Job Submission")
        
        job_manager = AzureJobManager(args.config)
        
        # Display and confirm hyperparameters
        if not job_manager.display_and_confirm_hyperparameters():
            logger.info("Hyperparameters not confirmed by user. Exiting.")
            status.update_status("Job Submission", "failed")
            return
        
        # Submit job
        logger.info("Submitting finetuning job")
        job = job_manager.submit_finetuning_job()
        logger.info(f"Finetuning job submitted successfully with ID: {job.name}")
        print(f"\n🎯 Job Details:")
        print(f"   Job ID: {job.name}")
        print(f"   Job URL: {job.studio_url}")
        print(f"   Status: {job.status}")
        
        status.update_status("Job Submission", "completed")
        
        # 7. Job Monitoring
        status.update_status("Job Monitoring")
        
        logger.info(f"Monitoring job {job.name}")
        metrics_tracker = MetricsTracker(job.name, args.config)
        
        print(f"\n🔍 Monitoring job {job.name}")
        print(f"   Polling interval: {args.monitoring_interval} seconds")
        print(f"   Maximum monitoring time: {args.max_monitor_time} minutes")
        print(f"   Press Ctrl+C to stop monitoring (job will continue running)")
        
        try:
            # Monitor with specified polling interval and timeout
            result = metrics_tracker.monitor_job(args.monitoring_interval, args.max_monitor_time)
            
            if result["status"] == "Completed":
                logger.info(f"Job completed successfully")
                print(f"\n🎉 Training completed successfully!")
                
                # Plot final metrics
                print("📊 Generating metrics visualizations...")
                plot_paths = metrics_tracker.plot_metrics()
                logger.info(f"Metrics plots saved to: {plot_paths}")
                
                # Display final metrics
                print(f"\n📈 Final Training Metrics:")
                for metric_name, metric_values in result["metrics"].items():
                    if not metric_values:
                        continue
                    final_value = metric_values[-1]['value']
                    print(f"   {metric_name}: {final_value:.6f}")
                
                status.update_status("Job Monitoring", "completed")
                
                # 8. Model Download (if requested)
                if args.auto_download_model:
                    print(f"\n📥 Downloading trained model...")
                    model_manager = ModelManager(args.config)
                    model_path = model_manager.download_model_to_local(
                        job.name, 
                        compress_model=True
                    )
                    print(f"✅ Model downloaded to: {model_path}")
                
            else:
                logger.error(f"Job ended with status: {result['status']}")
                print(f"❌ Job ended with status: {result['status']}")
                if "error" in result:
                    print(f"   Error: {result['error']}")
                status.update_status("Job Monitoring", "failed")
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            print(f"\n⏹️  Monitoring stopped by user")
            print(f"   Job {job.name} is still running in Azure ML")
            print(f"   Monitor at: {job.studio_url}")
            status.update_status("Job Monitoring", "completed")
        
        # Final summary
        print(f"\n" + "="*60)
        print(f"🏁 Phi-4 Finetuning Pipeline Summary")
        print(f"="*60)
        print(f"✅ Steps completed: {len(status.steps_completed)}/{status.total_steps}")
        print(f"🔗 Job ID: {job.name}")
        print(f"🌐 Monitor at: {job.studio_url}")
        print(f"📁 Results will be saved to: {config['train']['results_dir']}")
        print(f"="*60)
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print(f"\n\n⏹️  Process interrupted by user. Exiting.")
        return
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        print(f"❌ Pipeline Error: {str(e)}")
        status.update_status(status.current_step, "failed")
        return


if __name__ == "__main__":
    main()