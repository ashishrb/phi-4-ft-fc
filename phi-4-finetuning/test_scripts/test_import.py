try:
    from azure.ai.ml import MLClient
    print("Successfully imported MLClient!")
except ImportError as e:
    print(f"Import error: {e}")
    
    # Check Python path
    import sys
    print("\nPython path:")
    for path in sys.path:
        print(f"  {path}")
        
    # Try to find azure packages
    import pkg_resources
    print("\nInstalled packages with 'azure' in the name:")
    for package in pkg_resources.working_set:
        if "azure" in package.key:
            print(f"  {package.key} {package.version}")