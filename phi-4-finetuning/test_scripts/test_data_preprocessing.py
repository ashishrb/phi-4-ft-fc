import os
from src.data_preparation.preprocess import DataPreprocessor

try:
    # Make sure data directory exists
    raw_data_path = os.path.join("data", "raw", "function_calling_dataset.jsonl")
    if not os.path.exists(raw_data_path):
        print(f"❌ Raw data file not found at {raw_data_path}")
        exit(1)
    
    # Initialize data processor
    print("Initializing data processor...")
    processor = DataPreprocessor(input_file=raw_data_path)
    
    # Test data validation
    print("Validating data...")
    if processor.validate_data():
        print("✅ Data validation successful!")
    else:
        print("❌ Data validation failed!")
        exit(1)
    
    # Process a small sample (first 10 examples)
    print("Testing data preprocessing with a small sample...")
    
    # Count total examples
    with open(raw_data_path, 'r') as f:
        total_examples = sum(1 for _ in f)
    
    print(f"Total examples in dataset: {total_examples}")
    print("Processing and splitting...")
    
    # Process and split data
    output_paths = processor.preprocess_and_split()
    
    print("✅ Data preprocessing successful!")
    for split, path in output_paths.items():
        count = sum(1 for _ in open(path))
        print(f"  - {split}: {count} examples")
        
except Exception as e:
    print(f"❌ Data preprocessing test failed: {str(e)}")