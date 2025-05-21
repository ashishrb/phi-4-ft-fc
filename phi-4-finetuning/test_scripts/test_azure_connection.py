from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

try:
    # Get Azure credentials
    print("Attempting to get Azure credentials...")
    credential = DefaultAzureCredential()
    
    # Load config to get subscription and workspace info
    from src.msazure.config import AzureConfig
    config_handler = AzureConfig()
    config = config_handler.load_config()
    
    # Try to connect to Azure ML workspace
    print("Attempting to connect to Azure ML workspace...")
    ml_client = MLClient(
        credential=credential,
        subscription_id=config['config']['AZURE_SUBSCRIPTION_ID'],
        resource_group_name=config['config']['AZURE_RESOURCE_GROUP'],
        workspace_name=config['config']['AZURE_WORKSPACE']
    )
    
    # Test connection by getting workspace info
    workspace = ml_client.workspaces.get(config['config']['AZURE_WORKSPACE'])
    print("✅ Azure connection successful!")
    print(f"Connected to workspace: {workspace.name}")
    print(f"Location: {workspace.location}")
    
except Exception as e:
    print(f"❌ Azure connection failed: {str(e)}")