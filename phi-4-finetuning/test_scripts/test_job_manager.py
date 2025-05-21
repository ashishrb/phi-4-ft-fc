from src.msazure.job import AzureJobManager

try:
    # Initialize job manager
    print("Initializing job manager...")
    job_manager = AzureJobManager()
    
    # Initialize ML client
    print("Initializing Azure ML client...")
    job_manager.initialize_ml_client()
    
    # Check compute existence (but don't create)
    print("Checking compute cluster...")
    compute_name = job_manager.config['train']['azure_compute_cluster_name']
    
    try:
        compute = job_manager.ml_client.compute.get(compute_name)
        print(f"✅ Compute cluster '{compute_name}' exists!")
        print(f"  - VM Size: {compute.size}")
        print(f"  - State: {compute.provisioning_state}")
    except Exception:
        print(f"ℹ️ Compute cluster '{compute_name}' does not exist yet (will be created during actual run)")
    
    # Prepare training script args
    print("Testing hyperparameter preparation...")
    script_args = job_manager.prepare_training_script_args()
    
    print("✅ Job manager test successful!")
    print("Sample script arguments:")
    for arg in script_args[:5]:  # Show first 5 args
        print(f"  - {arg}")
    
except Exception as e:
    print(f"❌ Job manager test failed: {str(e)}")