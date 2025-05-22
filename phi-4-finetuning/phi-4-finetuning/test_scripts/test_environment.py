from src.msazure.environment import AzureEnvironmentManager

try:
    # Initialize environment manager
    print("Initializing environment manager...")
    env_manager = AzureEnvironmentManager()
    
    # Initialize ML client
    print("Initializing Azure ML client...")
    env_manager.initialize_ml_client()
    
    # List existing environments
    print("Listing existing environments...")
    environments = env_manager.list_environments()
    print(f"Found {len(environments)} existing environments:")
    
    for env in environments:
        print(f"  - {env.name} (version: {env.version})")
    
    print("✅ Environment manager test successful!")
    
except Exception as e:
    print(f"❌ Environment manager test failed: {str(e)}")