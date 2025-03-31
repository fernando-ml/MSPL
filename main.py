"""
Dual-Space (DOS) Project - Main script

This is the main entry point for the DOS project that handles loading configuration,
setting up the environment, and running the specified scenarios.
"""

import sys
import os
import argparse
from datetime import datetime
from utils.config_parser import load_config, get_output_paths, get_scenario_name
from utils.utils import get_device, print_section
from trainer import train_prototypical_model, train_traditional_model
from dataloader import DatasetManager


def save_results(config, metrics, scenario_name, dataset_name):
    """Save experiment results to file.
    
    Args:
        config (dict): Configuration dictionary
        metrics (dict): Metrics to save
        scenario_name (str): Name of the scenario
        dataset_name (str): Name of the dataset
    """
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{dataset_name}_{scenario_name}_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Scenario: {scenario_name}\n")
        f.write(f"Mode: {config['mode']}\n\n")
        f.write("Results:\n")
        for metric_name, value in metrics.items():
            f.write(f"{metric_name}: {value}\n")
    
    print(f"Results saved to {filename}")


def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='DOS (Dual-Space) Project - Multi-space episodic training')
    
    # Required arguments
    parser.add_argument('config_file', help='Path to the configuration YAML file')
    parser.add_argument('--dataset', '-d', required=True, 
                      help='Dataset to use: CICEVSE_Network2024, CICIDS2017, CICEVSE_PowerB2024, CICIoV2024, etc.')
    
    # Optional arguments
    parser.add_argument('--mode', '-m', choices=['prototypical', 'traditional'], 
                      help='Training mode (overrides config file)')
    parser.add_argument('--output-dir', '-o', help='Custom output directory')
    parser.add_argument('--epochs', '-e', type=int, help='Number of epochs (overrides config file)')
    parser.add_argument('--samples', '-s', type=int, help='Number of samples (overrides config file)')
    
    return parser.parse_args()


def main():
    """Main function for the DOS project."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config_file)
    
    # Set dataset from command line
    config['selected-dataset'] = args.dataset
    
    if args.mode:
        config['mode'] = args.mode
    
    if args.epochs and 'epochs' in config['params']:
        config['params']['epochs'] = args.epochs
    
    if args.samples and 'n_samples' in config['params']:
        config['params']['n_samples'] = args.samples
    
    # Create output directories
    if args.output_dir:
        config['output']['train-history'] = os.path.join(args.output_dir, "train_history")
        config['output']['best_models'] = os.path.join(args.output_dir, "models/best_model_")
    
    output_paths = get_output_paths(config)
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Print run info
    if config['mode'] == 'prototypical':
        print_section("Starting Deep Learning Run")
        run_type = "Deep Learning (Prototypical Network)"
    else:
        print_section("Starting Traditional ML Run")
        run_type = "Traditional Machine Learning"
    
    print(f"Run Type: {run_type}")
    print(f"Dataset: {config['selected-dataset']}")
    print(f"Configuration File: {args.config_file}")
    
    # Initialize data manager and preprocess data
    print_section("Loading and Preprocessing Data")
    dataset_manager = DatasetManager(config)
    X_train, y_train, val_dataloader = dataset_manager.preprocess_data()
    
    # Run scenarios
    total_scenarios = len(config['scenarios'])
    print(f"Total scenarios to run: {total_scenarios}")
    
    for i, scenario in enumerate(config['scenarios']):
        # Get scenario configuration
        distances_weights = None
        if config['mode'] == 'prototypical':
            config_setup_name, distances_weights = get_scenario_name(scenario, config['mode'])
        else:
            config_setup_name = scenario
            
        print_section(f"Starting scenario {i+1}/{total_scenarios}: {config_setup_name}")
        
        # Train model based on mode
        if config['mode'] == 'prototypical':
            best_metrics = train_prototypical_model(
                config=config,
                X_train=X_train,
                y_train=y_train,
                val_dataloader=val_dataloader,
                config_setup_name=config_setup_name,
                distances_weights=distances_weights
            )
        else:
            best_metrics = train_traditional_model(
                config=config,
                X_train=X_train,
                y_train=y_train,
                val_dataloader=val_dataloader,
                model_name=config_setup_name
            )
        
        # Save results
        save_results(config, best_metrics, config_setup_name, config['selected-dataset'])
        
    print_section(f"All {total_scenarios} scenarios completed")


if __name__ == "__main__":
    main()
