# src/monitoring/logger.py

import os
import logging
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

class TrainingLogger:
    """
    Class to handle training logging
    """
    
    def __init__(self, log_dir: str = None, experiment_name: str = None):
        """
        Initialize Training Logger
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = log_dir or os.path.join("logs", "training")
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set experiment name
        if experiment_name:
            self.experiment_name = experiment_name
        else:
            self.experiment_name = f"phi4_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Set log file path
        self.log_file = os.path.join(self.log_dir, f"{self.experiment_name}.log")
        
        # Configure logging
        self.logger = self._setup_logger()
        
        # Initialize events list
        self.events = []
        
    def _setup_logger(self) -> logging.Logger:
        """
        Set up logger
        
        Returns:
            Configured logger
        """
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        if logger.handlers:
            for handler in logger.handlers:
                logger.removeHandler(handler)
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Log an event
        
        Args:
            event_type: Type of event
            event_data: Event data
        """
        # Create event
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": event_data
        }
        
        # Add event to list
        self.events.append(event)
        
        # Log event
        self.logger.info(f"{event_type}: {json.dumps(event_data)}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics
        
        Args:
            metrics: Metrics to log
            step: Training step
        """
        # Create event data
        event_data = {**metrics}
        
        if step is not None:
            event_data["step"] = step
        
        # Log event
        self.log_event("metrics", event_data)
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """
        Log hyperparameters
        
        Args:
            hyperparameters: Hyperparameters to log
        """
        # Log event
        self.log_event("hyperparameters", hyperparameters)
    
    def log_artifact(self, name: str, path: str) -> None:
        """
        Log an artifact
        
        Args:
            name: Artifact name
            path: Path to the artifact
        """
        # Create event data
        event_data = {
            "name": name,
            "path": path
        }
        
        # Log event
        self.log_event("artifact", event_data)
    
    def log_status(self, status: str, message: Optional[str] = None) -> None:
        """
        Log a status
        
        Args:
            status: Status to log
            message: Optional message
        """
        # Create event data
        event_data = {
            "status": status
        }
        
        if message:
            event_data["message"] = message
        
        # Log event
        self.log_event("status", event_data)
    
    def export_events(self, output_file: Optional[str] = None) -> str:
        """
        Export events to a JSON file
        
        Args:
            output_file: Path to the output file. If None, use default
            
        Returns:
            Path to the exported file
        """
        if output_file is None:
            output_file = os.path.join(self.log_dir, f"{self.experiment_name}_events.json")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Write events to file
        with open(output_file, "w") as f:
            json.dump(self.events, f, indent=2)
        
        self.logger.info(f"Events exported to {output_file}")
        
        return output_file


def main():
    """
    Main function to test logging
    """
    # Create logger
    logger = TrainingLogger()
    
    # Log hyperparameters
    logger.log_hyperparameters({
        "model_name": "microsoft/Phi-4-mini-instruct",
        "learning_rate": 2e-5,
        "batch_size": 8,
        "epochs": 3
    })
    
    # Simulate training
    for epoch in range(3):
        # Log status
        logger.log_status("training", f"Starting epoch {epoch + 1}")
        
        # Simulate steps
        for step in range(10):
            # Simulate metrics
            metrics = {
                "loss": 1.0 - (epoch * 0.1 + step * 0.01),
                "accuracy": 0.7 + (epoch * 0.05 + step * 0.005)
            }
            
            # Log metrics
            logger.log_metrics(metrics, epoch * 10 + step)
            
            # Sleep to simulate training time
            time.sleep(0.1)
        
        # Log status
        logger.log_status("validation", f"Validating epoch {epoch + 1}")
        
        # Simulate validation metrics
        val_metrics = {
            "val_loss": 1.1 - (epoch * 0.15),
            "val_accuracy": 0.65 + (epoch * 0.07)
        }
        
        # Log validation metrics
        logger.log_metrics(val_metrics, (epoch + 1) * 10)
    
    # Log artifact
    logger.log_artifact("model", "results/model")
    
    # Log status
    logger.log_status("completed", "Training completed successfully")
    
    # Export events
    events_file = logger.export_events()
    print(f"Events exported to {events_file}")


if __name__ == "__main__":
    main()