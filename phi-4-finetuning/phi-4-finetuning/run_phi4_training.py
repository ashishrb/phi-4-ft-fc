#!/usr/bin/env python3
"""
Simple Phi-4 Finetuning Pipeline Runner
Single command to run the entire pipeline
"""

import subprocess
import sys
import os

def main():
    print("🚀 Starting Phi-4 Finetuning Pipeline")
    print("="*60)
    
    # Check if we're in the right directory
    if not os.path.exists("src/main.py"):
        print("❌ Error: Please run this script from the phi-4-finetuning directory")
        print("   Expected to find src/main.py")
        sys.exit(1)
    
    # Check if config exists
    if not os.path.exists("configs/azure_config.yaml"):
        print("❌ Error: configs/azure_config.yaml not found")
        print("   Please ensure your configuration file exists")
        sys.exit(1)
    
    # Run the pipeline
    try:
        cmd = [
            sys.executable, "src/main.py",
            "--config", "configs/azure_config.yaml",
            "--auto_download_model"  # Automatically download model when complete
        ]
        
        print(f"🔧 Running command: {' '.join(cmd)}")
        print("="*60)
        
        # Run the pipeline
        result = subprocess.run(cmd, check=True)
        
        print("\n" + "="*60)
        print("🎉 Pipeline completed successfully!")
        print("="*60)
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed with exit code: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print(f"\n⏹️  Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()