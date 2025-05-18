# src/azure/environment.py

import os
import logging
import time
from typing import Dict, Any, Optional, List

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import DefaultAzureCredential

from src.azure.config import AzureConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/azure_environment.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("AzureEnvironment")

class AzureEnvironmentManager:
    """
    Class to handle Azure ML environment setup and management
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize Azure Environment Manager
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_handler = AzureConfig(config_path)
        self.config = self.config_handler.load_config()
        self.ml_client = None
        self.environment = None
        
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
    
    def create_environment(self, wait_for_completion: bool = True) -> Environment:
        """
        Create a training environment in Azure ML
        
        Args:
            wait_for_completion: Whether to wait for the environment creation to complete
            
        Returns:
            The created environment
        """
        try:
            if not self.ml_client:
                self.initialize_ml_client()
            
            # Get environment name from config
            env_name = self.config['train']['azure_env_name']
            dockerfile_path = self.config['train'].get('docker_image_path', os.path.join("docker", "Dockerfile"))

            
            logger.info(f"Creating environment '{env_name}' using Dockerfile at {dockerfile_path}")
            
            # Check if Dockerfile exists
            if not os.path.exists(dockerfile_path):
                raise FileNotFoundError(f"Dockerfile not found at {dockerfile_path}")
            
            # Validate Dockerfile content
            self._validate_dockerfile(dockerfile_path)
            
            # Create environment
            self.environment = Environment(
                name=env_name,
                version="1",
                description=f"Phi-4 finetuning environment: {env_name}",
                tags={"model": "phi-4", "type": "finetuning", "created_at": datetime.now().isoformat()},
                build=BuildContext(path=os.path.dirname(dockerfile_path))
            )
            
            # Create the environment in Azure ML
            created_env = self.ml_client.environments.create_or_update(self.environment)
            logger.info(f"Environment creation job submitted: {created_env.name} (version: {created_env.version})")
            
            print("\n" + "="*50)
            print(f"Environment '{env_name}' creation job submitted successfully!")
            print(f"Environment ID: {created_env.id}")
            print(f"Environment Version: {created_env.version}")
            print("="*50 + "\n")
            
            # If wait_for_completion is True, wait for the environment to be created
            if wait_for_completion:
                print("Waiting for environment creation to complete...")
                creation_successful = self.wait_for_environment_creation(env_name, created_env.version)
                
                if not creation_successful:
                    logger.error(f"Environment creation failed or timed out for '{env_name}'")
                    raise Exception(f"Environment creation failed or timed out for '{env_name}'")
            
            return created_env
        
        except Exception as e:
            logger.error(f"Error creating environment: {str(e)}")
            raise

    def _validate_dockerfile(self, dockerfile_path: str) -> None:
        """
        Validate the Dockerfile content
        
        Args:
            dockerfile_path: Path to the Dockerfile
        """
        try:
            with open(dockerfile_path, 'r') as f:
                dockerfile_content = f.read()
            
            # Check for basic Dockerfile elements
            if not dockerfile_content.strip():
                raise ValueError("Dockerfile is empty")
            
            required_elements = ["FROM", "RUN", "WORKDIR"]
            missing_elements = [elem for elem in required_elements if elem not in dockerfile_content]
            
            if missing_elements:
                logger.warning(f"Dockerfile missing recommended elements: {', '.join(missing_elements)}")
            
            logger.info(f"Dockerfile validation passed for {dockerfile_path}")
        except Exception as e:
            logger.error(f"Error validating Dockerfile: {str(e)}")
            raise
    
    def wait_for_environment_creation(self, env_name: str, version: str, timeout_minutes: int = 60, check_interval_seconds: int = 30) -> bool:
        """
        Wait for an environment creation to complete
        
        Args:
            env_name: Name of the environment
            version: Version of the environment
            timeout_minutes: Maximum time to wait in minutes
            check_interval_seconds: Interval in seconds to check status
            
        Returns:
            True if the environment was created successfully, False otherwise
        """
        try:
            logger.info(f"Waiting for environment '{env_name}' (version: {version}) creation to complete")
            
            # Calculate timeout
            timeout = timeout_minutes * 60
            start_time = time.time()
            
            # Track previous status for change detection
            prev_status = None
            status_unchanged_count = 0
            MAX_UNCHANGED_COUNT = 10  # If status unchanged for this many checks, log a warning
            
            # Poll environment status
            while time.time() - start_time < timeout:
                try:
                    # Get environment
                    env = self.ml_client.environments.get(name=env_name, version=version)
                    
                    # Check status
                    status = getattr(env, "provisioning_state", None)
                    
                    # Check if status is unchanged
                    if status == prev_status:
                        status_unchanged_count += 1
                        if status_unchanged_count >= MAX_UNCHANGED_COUNT:
                            logger.warning(f"Environment status '{status}' has not changed for {MAX_UNCHANGED_COUNT * check_interval_seconds} seconds")
                            status_unchanged_count = 0  # Reset to avoid repeated warnings
                    else:
                        status_unchanged_count = 0
                        prev_status = status
                    
                    if status == "Succeeded":
                        logger.info(f"Environment '{env_name}' (version: {version}) created successfully")
                        print(f"Environment '{env_name}' (version: {version}) created successfully!")
                        return True
                    elif status in ["Failed", "Canceled"]:
                        error_message = getattr(env, "error_message", "No error message available")
                        logger.error(f"Environment creation failed with status: {status}. Error: {error_message}")
                        print(f"Environment creation failed with status: {status}")
                        print(f"Error: {error_message}")
                        return False
                    else:
                        # Still in progress
                        elapsed_minutes = (time.time() - start_time) / 60
                        remaining_minutes = timeout_minutes - elapsed_minutes
                        print(f"Environment creation in progress... Status: {status} (Elapsed: {elapsed_minutes:.1f}m, Remaining: {remaining_minutes:.1f}m)")
                        # Wait for check_interval_seconds before checking again
                        time.sleep(check_interval_seconds)
                except Exception as e:
                    logger.warning(f"Error checking environment status: {str(e)}")
                    time.sleep(check_interval_seconds)
            
            logger.error(f"Environment creation timed out after {timeout_minutes} minutes")
            print(f"Environment creation timed out after {timeout_minutes} minutes")
            return False
        
        except Exception as e:
            logger.error(f"Error waiting for environment creation: {str(e)}")
            raise
    
    def get_environment(self, name: Optional[str] = None, version: str = "latest") -> Environment:
        """
        Get an environment from Azure ML
        
        Args:
            name: Name of the environment to retrieve. If not provided, uses the name from config
            version: Version of the environment to retrieve
            
        Returns:
            The retrieved environment
        """
        try:
            if not self.ml_client:
                self.initialize_ml_client()
            
            name = name or self.config['train']['azure_env_name']
            
            logger.info(f"Getting environment '{name}' (version: {version})")
            
            environment = self.ml_client.environments.get(name=name, version=version)
            logger.info(f"Environment retrieved successfully: {environment.id}")
            
            return environment
        
        except Exception as e:
            logger.error(f"Error getting environment: {str(e)}")
            raise
    
    def list_environments(self) -> List[Environment]:
        """
        List all environments in the workspace
        
        Returns:
            List of environments
        """
        try:
            if not self.ml_client:
                self.initialize_ml_client()
            
            logger.info("Listing environments in workspace")
            
            environments = list(self.ml_client.environments.list())
            logger.info(f"Retrieved {len(environments)} environments")
            
            return environments
        
        except Exception as e:
            logger.error(f"Error listing environments: {str(e)}")
            raise


def main():
    """
    Main function to test environment creation
    """
    # Initialize environment manager
    env_manager = AzureEnvironmentManager()
    
    # Confirm configuration
    if not env_manager.config_handler.confirm_config():
        print("Please update the configuration and try again.")
        return
    
    # Create environment
    try:
        created_env = env_manager.create_environment(wait_for_completion=True)
        print(f"Environment created successfully: {created_env.id}")
    except Exception as e:
        print(f"Error creating environment: {str(e)}")


if __name__ == "__main__":
    main()