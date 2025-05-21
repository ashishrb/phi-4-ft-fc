# src/monitoring/metrics.py

import os
import logging
import json
import time
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/monitoring.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("MetricsMonitoring")

class MetricsTracker:
    """
    Class to track and visualize training metrics
    """
    
    def __init__(self, job_id: str, config_path: str = None, max_history_points: int = 1000):
        """
        Initialize Metrics Tracker
        
        Args:
            job_id: Azure ML job ID
            config_path: Path to the YAML configuration file
            max_history_points: Maximum number of data points to keep in history per metric
        """
        from src.msazure.config import AzureConfig
        
        self.job_id = job_id
        self.config_handler = AzureConfig(config_path)
        self.config = self.config_handler.load_config()
        self.ml_client = None
        self.metrics_history = {}
        self.last_update_time = 0
        self.max_history_points = max_history_points
        
        # Configure matplotlib for headless environments
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            logger.info("Configured matplotlib for non-interactive backend")
        except Exception as e:
            logger.warning(f"Error configuring matplotlib: {str(e)}")
        
    def initialize_ml_client(self) -> None:
        """
        Initialize the Azure ML client
        """
        try:
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential
            
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
    
    def get_job_metrics(self) -> Dict[str, Any]:
        """
        Get metrics from a job
        
        Returns:
            Dictionary with job metrics
        """
        try:
            if not self.ml_client:
                self.initialize_ml_client()
            
            logger.info(f"Getting metrics for job {self.job_id}")
            
            run = self.ml_client.jobs.get(self.job_id)
            
            # Get metrics
            metrics = {}
            try:
                metrics = run.metrics
            except Exception as e:
                logger.warning(f"Error getting metrics: {str(e)}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error getting job metrics: {str(e)}")
            raise
    
    def get_job_status(self) -> str:
        """
        Get the status of a job
        
        Returns:
            Job status
        """
        try:
            if not self.ml_client:
                self.initialize_ml_client()
            
            job = self.ml_client.jobs.get(self.job_id)
            return job.status
        
        except Exception as e:
            logger.error(f"Error getting job status: {str(e)}")
            raise
    
    def update_metrics_history(self) -> Dict[str, Any]:
        """
        Update the metrics history
        
        Returns:
            Updated metrics history
        """
        try:
            # Get current metrics
            current_metrics = self.get_job_metrics()
            current_time = time.time()
            
            # Update history
            for metric_name, metric_value in current_metrics.items():
                if isinstance(metric_value, (int, float)):
                    # Add to history
                    if metric_name not in self.metrics_history:
                        self.metrics_history[metric_name] = []
                    
                    # Add new data point
                    self.metrics_history[metric_name].append({
                        'value': metric_value,
                        'timestamp': current_time
                    })
                    
                    # Trim history if it exceeds maximum size
                    if len(self.metrics_history[metric_name]) > self.max_history_points:
                        # Keep the first point, the last max_points/2, and evenly spaced points in between
                        if len(self.metrics_history[metric_name]) > self.max_history_points + 1:
                            first_point = [self.metrics_history[metric_name][0]]
                            last_points = self.metrics_history[metric_name][-self.max_history_points//2:]
                            
                            # Calculate how many points to keep from the middle
                            middle_count = self.max_history_points - len(first_point) - len(last_points)
                            
                            # Get evenly spaced indexes from the middle section
                            middle_section = self.metrics_history[metric_name][1:-self.max_history_points//2]
                            if len(middle_section) > middle_count and middle_count > 0:
                                step = len(middle_section) / middle_count
                                middle_indices = [int(i * step) for i in range(middle_count)]
                                middle_points = [middle_section[i] for i in middle_indices if i < len(middle_section)]
                            else:
                                middle_points = middle_section
                            
                            # Combine sections
                            self.metrics_history[metric_name] = first_point + middle_points + last_points
                            logger.info(f"Pruned history for {metric_name} from {len(self.metrics_history[metric_name])} to {len(first_point) + len(middle_points) + len(last_points)} points")
            
            # Update last update time
            self.last_update_time = current_time
            
            return self.metrics_history
        
        except Exception as e:
            logger.error(f"Error updating metrics history: {str(e)}")
            raise
    
    def plot_metrics(self, save_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Plot metrics and save the plots
        
        Args:
            save_dir: Directory to save the plots. If None, use results/metrics
            
        Returns:
            Dictionary mapping metric names to plot file paths
        """
        try:
            if not self.metrics_history:
                logger.warning("No metrics history to plot")
                return {}
            
            if save_dir is None:
                save_dir = os.path.join("results", "metrics", self.job_id)
            
            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            # Import matplotlib here to allow changing the backend
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime
            
            # Plot each metric
            plot_paths = {}
            
            for metric_name, metric_data in self.metrics_history.items():
                # Skip if not enough values
                if len(metric_data) < 2:
                    continue
                
                # Extract values and timestamps
                values = [point['value'] for point in metric_data]
                timestamps = [point['timestamp'] for point in metric_data]
                
                # Convert timestamps to datetime objects for better plotting
                dates = [datetime.fromtimestamp(ts) for ts in timestamps]
                
                # Create figure
                plt.figure(figsize=(10, 6))
                plt.plot(dates, values)
                plt.title(f"{metric_name} over Time")
                plt.xlabel("Time")
                plt.ylabel(metric_name)
                plt.grid(True)
                
                # Format x-axis to show dates nicely
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                plt.gcf().autofmt_xdate()  # Rotate date labels
                
                # Save figure
                plot_path = os.path.join(save_dir, f"{metric_name.replace('/', '_')}.png")
                plt.savefig(plot_path)
                plt.close()
                
                plot_paths[metric_name] = plot_path
            
            return plot_paths
        
        except Exception as e:
            logger.error(f"Error plotting metrics: {str(e)}")
            # Continue execution even if plotting fails
            return {}
    
    def monitor_job(self, poll_interval_seconds: int = 60, max_duration_minutes: int = 480) -> Dict[str, Any]:
        """
        Monitor a job until it completes
        
        Args:
            poll_interval_seconds: Interval in seconds to poll for job status
            max_duration_minutes: Maximum duration to monitor the job in minutes
            
        Returns:
            Dictionary with job status, metrics and plot paths
        """
        try:
            logger.info(f"Monitoring job {self.job_id}")
            print(f"Monitoring job {self.job_id}")
            
            # Calculate timeout
            timeout = max_duration_minutes * 60
            start_time = time.time()
            
            # Initialize metrics plot directory
            metrics_dir = os.path.join("results", "metrics", self.job_id)
            os.makedirs(metrics_dir, exist_ok=True)
            
            while time.time() - start_time < timeout:
                # Get job status
                status = self.get_job_status()
                
                # Display current status
                print(f"Job status: {status}")
                
                if status in ["Failed", "Canceled", "NotResponding"]:
                    logger.error(f"Job failed with status: {status}")
                    print(f"Job failed with status: {status}")
                    return {"status": status, "error": "Job failed"}
                
                if status == "Completed":
                    logger.info(f"Job completed successfully")
                    print(f"Job completed successfully")
                    
                    # Final metrics update
                    self.update_metrics_history()
                    
                    # Plot metrics
                    plot_paths = self.plot_metrics(metrics_dir)
                    
                    return {
                        "status": status, 
                        "metrics": self.metrics_history,
                        "plots": plot_paths
                    }
                
                # Update metrics
                self.update_metrics_history()
                
                # Plot metrics every 5 minutes
                if time.time() - self.last_update_time > 300:
                    self.plot_metrics(metrics_dir)
                
                # Display current metrics
                self._print_current_metrics()
                
                # If job is still running, wait and check again
                time.sleep(poll_interval_seconds)
            
            logger.warning(f"Monitoring timed out after {max_duration_minutes} minutes")
            print(f"Monitoring timed out after {max_duration_minutes} minutes")
            
            return {"status": "Timeout", "metrics": self.metrics_history}
        
        except Exception as e:
            logger.error(f"Error monitoring job: {str(e)}")
            raise
    
    def _print_current_metrics(self) -> None:
        """
        Print current metrics
        """
        print("\nCurrent Metrics:")
        print("-" * 50)
        
        for metric_name, metric_data in self.metrics_history.items():
            if not metric_data:
                continue
            
            current_value = metric_data[-1]['value']
            
            # Calculate trend if possible
            trend = ""
            if len(metric_data) > 1:
                previous_value = metric_data[-2]['value']
                diff = current_value - previous_value
                trend = f"({'↑' if diff > 0 else '↓'} {abs(diff):.4f})"
            
            print(f"{metric_name}: {current_value:.6f} {trend}")
        
        print("-" * 50)


def main():
    """
    Main function to test metrics tracking
    """
    # Ask for job ID
    job_id = input("Enter Azure ML job ID to monitor: ").strip()
    
    if not job_id:
        print("Job ID is required.")
        return
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(job_id)
    
    # Ask for poll interval
    poll_interval = input("Enter poll interval in seconds (default: 60): ").strip()
    poll_interval = int(poll_interval) if poll_interval else 60
    
    # Ask for max duration
    max_duration = input("Enter maximum monitoring duration in minutes (default: 480): ").strip()
    max_duration = int(max_duration) if max_duration else 480
    
    try:
        # Monitor job
        result = metrics_tracker.monitor_job(poll_interval, max_duration)
        
        # Display final results
        if result["status"] == "Completed":
            print("\n" + "="*50)
            print(f"Job {job_id} completed successfully!")
            print("\nFinal Metrics:")
            
            for metric_name, metric_values in result["metrics"].items():
                if not metric_values:
                    continue
                
                final_value = metric_values[-1]
                print(f"{metric_name}: {final_value:.6f}")
            
            print("\nMetrics plots saved to:")
            for metric_name, plot_path in result.get("plots", {}).items():
                print(f"{metric_name}: {plot_path}")
            
            print("="*50 + "\n")
        else:
            print(f"Job ended with status: {result['status']}")
            if "error" in result:
                print(f"Error: {result['error']}")
    
    except Exception as e:
        print(f"Error monitoring job: {str(e)}")


if __name__ == "__main__":
    main()