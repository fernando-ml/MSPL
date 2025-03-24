"""
Configuration handling module for the DOS project.

This module provides functionality for loading, validating and accessing
configuration settings throughout the project.
"""

import os
import sys
import yaml
import itertools
from pathlib import Path


def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate the config has required sections
        required_sections = ['selected-dataset', 'mode', 'params', 'output']
        for section in required_sections:
            if section not in config:
                print(f"Error: Missing required configuration section '{section}'")
                sys.exit(1)
                
        # Process the scenario configuration based on mode
        if config['mode'] == 'prototypical':
            # Generate prototypical network scenarios
            if 'dl_config' not in config or 'polyak_values' not in config['dl_config'] or 'weights_configs' not in config['dl_config']:
                print("Error: For prototypical mode, 'dl_config' with 'polyak_values' and 'weights_configs' must be specified")
                sys.exit(1)
                
            # Generate all scenario combinations for DL models
            config['scenarios'] = generate_dl_scenarios(config['dl_config']['polyak_values'], config['dl_config']['weights_configs'])
        else:
            # Traditional ML models
            if 'nonDL_config' not in config or 'model_types' not in config['nonDL_config']:
                print("Error: For non-DL mode, 'nonDL_config' with 'model_types' must be specified")
                sys.exit(1)
                
            # Use model types as scenarios for non-DL
            config['scenarios'] = config['nonDL_config']['model_types']
        
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)


def generate_dl_scenarios(polyak_values, weights_configs):
    """
    Generate all combinations of polyak settings and weights configurations for DL models.
    
    Args:
        polyak_values (list): List of boolean values for polyak setting
        weights_configs (list): List of weight arrays for different metrics
        
    Returns:
        list: List of scenario dictionaries with all combinations
    """
    scenarios = []
    
    # Generate all combinations
    for polyak in polyak_values:
        for weights in weights_configs:
            scenario = {
                'polyak': polyak,
                'vals': weights
            }
            scenarios.append(scenario)
    
    return scenarios


def get_output_paths(config):
    """
    Create and return output paths based on configuration.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Dictionary of output paths
    """
    paths = {}
    
    # Ensure train history directory exists
    train_history_path = config['output']['train-history']
    Path(train_history_path).mkdir(parents=True, exist_ok=True)
    paths['train_history'] = train_history_path
    
    # Ensure results directory exists
    results_path = "results"
    Path(results_path).mkdir(parents=True, exist_ok=True)
    paths['results'] = results_path
    
    # Ensure models directory exists
    models_path = os.path.dirname(config['output']['best_models'])
    Path(models_path).mkdir(parents=True, exist_ok=True)
    paths['models'] = models_path
    
    return paths


def get_scenario_name(scenario, mode='prototypical'):
    """
    Generate a standardized name for a scenario configuration.
    
    Args:
        scenario (dict): Scenario configuration
        mode (str): Training mode
        
    Returns:
        tuple: (config_name, distances_weights) - name and (for prototypical) distance weights
    """
    if mode != 'prototypical':
        return scenario, None
        
    weights = ["euclidean", "chebyshev", "cosine", "wasserstein"]
    config_name = 'polyak' if scenario['polyak'] else 'no-polyak'
    
    # Create map of distance weights
    distances_weights = dict(zip(weights, scenario['vals']))
    
    # Format the name
    config_name = config_name + '_' + '_'.join([f"{k}-{v}" for k, v in distances_weights.items()])
    
    # Convert string fractions to actual values
    distances_weights = {key: eval(val) for key, val in distances_weights.items()}
    
    return config_name, distances_weights


def sanitize_filename(name):
    """
    Convert scenario name to a safe filename.
    
    Args:
        name (str): Original name
        
    Returns:
        str: Sanitized filename
    """
    return name.replace("/", "over").replace(".", "point")
