# src/main.py
import os
import logging
import argparse
import time
import sys
from typing import Dict, Any, Optional

from src.azure.config import AzureConfig
from src.data_preparation.preprocess import DataPreprocessor
from src.data_preparation.tokenize import Tokenizer
from src.azure.dataset import AzureDatasetManager
from src.azure.environment import AzureEnvironmentManager
from src.azure.job import AzureJobManager
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

def parse_args():
    parser = argparse.ArgumentParser(description="Phi-4 Finetuning Pipeline")
    
    parser.add_argument("--config", type=str, default="configs/azure_config.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--skip_preprocessing", action="store_true",
                        help="Skip data preprocessing")
    parser.add_argument("--skip_tokenization", action="store_true",
                        help="Skip data tokenization")
    parser.add_argument("--skip_dataset_registration", action="store_true",
                        help="Skip dataset registration")
    parser.add_argument("--skip_environment_creation", action="store_true",
                        help="Skip environment creation")
    parser.add_argument("--skip_job_submission", action="store_true",
                        help="Skip job submission")
    parser.add_argument("--skip_monitoring", action="store_true",
                        help="Skip job monitoring")
    parser.add_argument("--skip_model_download", action="store_true",
                        help="Skip model download")
    parser.add_argument("--raw_data_path", type=str, default=None,
                        help="Path to the raw data file")
    parser.add_argument("--job_id", type=str, default=None,
                        help="Azure ML job ID (for monitoring or model download)")
    parser.add_argument("--local_model_dir", type=str, default=None,
                        help="Local directory to save the model")
    parser.add_argument("--compress_model", action="store_true",
                        help="Compress the model after downloading")
    parser.add_argument("--monitoring_interval", type=int, default=60,
                        help="Interval in seconds for job monitoring polling")
    parser.add_argument("--max_monitor_time", type=int, default=480,
                        help="Maximum time in minutes to monitor the job")
    
    args = parser.parse_args()
    
    # Validate arguments
    validate_arguments(args)
    
    return args

def parse_args():
    parser = argparse.ArgumentParser(description="Phi-4 Finetuning Pipeline")
    
    parser.add_argument("--config", type=str, default="configs/azure_config.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--skip_preprocessing", action="store_true",
                        help="Skip data preprocessing")
    parser.add_argument("--skip_tokenization", action="store_true",
                        help="Skip data tokenization")
    parser.add_argument("--skip_dataset_registration", action="store_true",
                        help="Skip dataset registration")
    parser.add_argument("--skip_environment_creation", action="store_true",
                        help="Skip environment creation")
    parser.add_argument("--skip_job_submission", action="store_true",
                        help="Skip job submission")
    parser.add_argument("--skip_monitoring", action="store_true",
                        help="Skip job monitoring")
    parser.add_argument("--skip_model_download", action="store_true",
                        help="Skip model download")
    parser.add_argument("--raw_data_path", type=str, default=None,
                        help="Path to the raw data file")
    parser.add_argument("--job_id", type=str, default=None,
                        help="Azure ML job ID (for monitoring or model download)")
    parser.add_argument("--local_model_dir", type=str, default=None,
                        help="Local directory to save the model")
    parser.add_argument("--compress_model", action="store_true",
                        help="Compress the model after downloading")
    parser.add_argument("--monitoring_interval", type=int, default=60,
                        help="Interval in seconds for job monitoring polling")
    parser.add_argument("--max_monitor_time", type=int, default=480,
                        help="Maximum time in minutes to monitor the job")
    
    args = parser.parse_args()
    
    # Validate arguments
    validate_arguments(args)
    
    return args

def validate_arguments(args):
    """
    Validate command line arguments
    
    Args:
        args: Arguments from argparse
    """
    # Check if config file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file {args.config} not found")
    
    # Check if raw data path exists if provided
    if args.raw_data_path and not os.path.exists(args.raw_data_path):
        raise FileNotFoundError(f"Raw data file {args.raw_data_path} not found")
    
    # If skipping to monitoring or model download, job_id is required
    if (args.skip_preprocessing and args.skip_tokenization and args.skip_dataset_registration and 
        args.skip_environment_creation and args.skip_job_submission) and not args.job_id:
        raise ValueError("Job ID is required when skipping all preparation steps")
    
    # Check monitoring interval range
    if args.monitoring_interval < 10 or args.monitoring_interval > 600:
        raise ValueError(f"Monitoring interval {args.monitoring_interval} is out of range (10-600 seconds)")
    
    # Check max monitor time range
    if args.max_monitor_time < 10 or args.max_monitor_time > 1440:
        raise ValueError(f"Max monitor time {args.max_monitor_time} is out of range (10-1440 minutes)")
    
    # If local_model_dir is provided, ensure parent directory exists
    if args.local_model_dir:
        parent_dir = os.path.dirname(args.local_model_dir)
        if parent_dir and not os.path.exists(parent_dir):
            raise FileNotFoundError(f"Parent directory {parent_dir} for local_model_dir does not exist")

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up signal handlers for graceful termination
    import signal
    
    # Define signal handler
    def signal_handler(sig, frame):
        logger.info("Received termination signal. Cleaning up...")
        print("\nReceived termination signal. Cleaning up...")
        # Perform any necessary cleanup here
        print("Cleanup complete. Exiting.")
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    try:
        # Try to import tqdm for progress bars
        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False
        
        # 1. Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config_handler = AzureConfig(args.config)
        config = config_handler.load_config()
        
        # 2. Display and check configuration with user
        if not config_handler.confirm_config():
            logger.info("Configuration not confirmed by user. Exiting.")
            return
        
        # Set up paths
        raw_data_path = args.raw_data_path or os.path.join("data", "raw", "function_calling_dataset.jsonl")
        
        # 3. Preprocess data
        if not args.skip_preprocessing:
            logger.info("Preprocessing data")
            print("\n=== Step 1: Data Preprocessing ===")
            
            data_processor = DataPreprocessor(input_file=raw_data_path)
            
            # Validate data
            print("Validating data...")
            if not data_processor.validate_data():
                logger.error("Data validation failed. Please check the input data.")
                print("❌ Data validation failed. Please check the input data.")
                return
            
            # Preprocess and split data
            print("Preprocessing and splitting data...")
            if has_tqdm:
                with tqdm(total=3, desc="Preprocessing") as pbar:
                    output_paths = data_processor.preprocess_and_split()
                    pbar.update(3)
            else:
                output_paths = data_processor.preprocess_and_split()
                
            logger.info(f"Data preprocessed and split successfully: {output_paths}")
            print("✅ Data preprocessed and split successfully")
            
            for split, path in output_paths.items():
                count = sum(1 for _ in open(path))
                print(f"  - {split}: {count} examples")
        
        # 4. Tokenize data
        if not args.skip_tokenization:
            logger.info("Tokenizing data")
            print("\n=== Step 2: Data Tokenization ===")
            
            tokenizer = Tokenizer(config_path=args.config)
            
            # Tokenize dataset
            print("Tokenizing dataset...")
            if has_tqdm:
                with tqdm(total=3, desc="Tokenizing") as pbar:
                    # Each tokenize_file call updates the progress bar
                    def update_progress(*args, **kwargs):
                        pbar.update(1)
                    
                    # Patch the tokenize_file method to update progress
                    original_tokenize_file = tokenizer.tokenize_file
                    def wrapped_tokenize_file(*args, **kwargs):
                        result = original_tokenize_file(*args, **kwargs)
                        update_progress()
                        return result
                    
                    tokenizer.tokenize_file = wrapped_tokenize_file
                    
                    tokenized_paths = tokenizer.tokenize_dataset(
                        input_dir=config['config']['SFT_DATA_DIR'],
                        output_dir=os.path.join(config['config']['SFT_DATA_DIR'], "tokenized")
                    )
            else:
                tokenized_paths = tokenizer.tokenize_dataset(
                    input_dir=config['config']['SFT_DATA_DIR'],
                    output_dir=os.path.join(config['config']['SFT_DATA_DIR'], "tokenized")
                )
                
            logger.info(f"Data tokenized successfully: {tokenized_paths}")
            print("✅ Data tokenized successfully")
            
            # Analyze token lengths
            print("Analyzing token lengths...")
            for split, path in tokenized_paths.items():
                stats = tokenizer.analyze_token_lengths(path)
                print(f"  - {split} set: min={stats['min_length']}, max={stats['max_length']}, avg={stats['avg_length']:.1f}")
        
        # Rest of the function with similar progress tracking and status updates...
        
        # 5. Register dataset in Azure ML
        if not args.skip_dataset_registration:
            logger.info("Registering dataset in Azure ML")
            print("\n=== Step 3: Dataset Registration ===")
            
            dataset_manager = AzureDatasetManager(args.config)
            
            # Register dataset
            print("Registering dataset in Azure ML...")
            registered_dataset = dataset_manager.register_dataset(preprocess=False)
            logger.info(f"Dataset registered successfully: {registered_dataset.id}")
            print(f"✅ Dataset registered successfully: {registered_dataset.name} (version: {registered_dataset.version})")
        
        # 6. Create environment in Azure ML
        if not args.skip_environment_creation:
            logger.info("Creating environment in Azure ML")
            print("\n=== Step 4: Environment Creation ===")
            
            env_manager = AzureEnvironmentManager(args.config)
            
            # Create environment
            print("Creating environment in Azure ML...")
            created_env = env_manager.create_environment(wait_for_completion=True)
            logger.info(f"Environment created successfully: {created_env.id}")
            print(f"✅ Environment created successfully: {created_env.name} (version: {created_env.version})")
        
        # 7. Confirm hyperparameters and submit job
        if not args.skip_job_submission:
            logger.info("Submitting finetuning job")
            print("\n=== Step 5: Job Submission ===")
            
            job_manager = AzureJobManager(args.config)
            
            # Display and confirm hyperparameters
            if not job_manager.display_and_confirm_hyperparameters():
                logger.info("Hyperparameters not confirmed by user. Exiting.")
                print("❌ Hyperparameters not confirmed by user. Exiting.")
                return
            
            # Submit job
            print("Submitting finetuning job...")
            job = job_manager.submit_finetuning_job()
            logger.info(f"Finetuning job submitted successfully with ID: {job.id}")
            print(f"✅ Finetuning job submitted successfully with ID: {job.id}")
            
            # Save job ID for later steps
            args.job_id = job.id
        
        # 8. Monitor job
        if not args.skip_monitoring and args.job_id:
            logger.info(f"Monitoring job {args.job_id}")
            print(f"\n=== Step 6: Job Monitoring ===")
            
            metrics_tracker = MetricsTracker(args.job_id, args.config)
            
            # Monitor job
            print(f"Monitoring job {args.job_id}. Press Ctrl+C to stop monitoring (the job will continue running).")
            
            try:
                # Monitor with specified polling interval and timeout
                result = metrics_tracker.monitor_job(args.monitoring_interval, args.max_monitor_time)
                
                if result["status"] == "Completed":
                    logger.info(f"Job completed successfully")
                    print(f"✅ Job completed successfully")
                    
                    # Plot final metrics
                    print("Generating metrics visualizations...")
                    plot_paths = metrics_tracker.plot_metrics()
                    logger.info(f"Metrics plots saved to: {plot_paths}")
                    print(f"✅ Metrics plots saved")
                    
                    # Display final metrics
                    print("\n" + "="*50)
                    print(f"Job {args.job_id} completed successfully!")
                    print("\nFinal Metrics:")
                    
                    for metric_name, metric_values in result["metrics"].items():
                        if not metric_values:
                            continue
                        
                        final_value = metric_values[-1]['value']
                        print(f"  {metric_name}: {final_value:.6f}")
                    
                    print("="*50 + "\n")
                else:
                    logger.error(f"Job ended with status: {result['status']}")
                    print(f"❌ Job ended with status: {result['status']}")
                    if "error" in result:
                        print(f"Error: {result['error']}")
                        
                        # If job failed, exit
                        return
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                print("\nMonitoring stopped. The job will continue running.")
        
        # 9. Download model
        if not args.skip_model_download and args.job_id:
            logger.info(f"Downloading model from job {args.job_id}")
            print(f"\n=== Step 7: Model Download ===")
            
            # Ask user if they want to download the model
            download = input("Do you want to download the model locally? (yes/no): ").strip().lower()
            
            if download in ('yes', 'y'):
                model_manager = ModelManager(args.config)
                
                # Download model
                print("Downloading model...")
                model_path = model_manager.download_model_to_local(
                    args.job_id, 
                    args.local_model_dir, 
                    compress_model=args.compress_model
                )
                logger.info(f"Model downloaded successfully to {model_path}")
                print(f"✅ Model downloaded successfully to {model_path}")
                
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
                    print("Pushing model to Hugging Face Hub...")
                    repo_url = model_manager.save_model_to_hub(model_path, repo_name, private)
                    logger.info(f"Model pushed successfully to {repo_url}")
                    print(f"✅ Model pushed successfully to {repo_url}")
        
        # 10. Greet user for successful completion
        print("\n" + "="*50)
        print("🎉 Phi-4 Finetuning Process Completed Successfully! 🎉")
        print("="*50)
        
        if args.job_id:
            print(f"Job ID: {args.job_id}")
            print(f"You can check the job in the Azure ML Studio.")
        
        print(f"Thank you for using the Phi-4 Finetuning Pipeline!")
        print("="*50 + "\n")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        print(f"❌ Error: {str(e)}")
        return
        
    except ValueError as e:
        logger.error(f"Invalid argument: {str(e)}")
        print(f"❌ Error: {str(e)}")
        return
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print("\n\nProcess interrupted by user. Exiting.")
        return
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        print(f"❌ Error: {str(e)}")
        return


if __name__ == "__main__":
    main()