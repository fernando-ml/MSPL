# WandB Integration for DOS Project

This guide explains how to use the Weights & Biases (WandB) logging features integrated into the DOS Project.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Login to your WandB account:
   ```
   wandb login
   ```
   If you don't have an account, you can create one for free at [wandb.ai](https://wandb.ai/).

## Running Experiments with WandB Logging

The project now automatically logs all training metrics and visualizations to WandB. You can run experiments as usual:

```bash
# Run a prototypical network experiment on the CIC-IDS-2017 dataset
python main.py --dataset CIC-IDS-2017 --mode prototypical

# Run a traditional ML experiment on the UNSW-NB15 dataset
python main.py --dataset UNSW-NB15 --mode traditional
```

## Tracked Metrics & Visualizations

### General Metrics
- Balanced accuracy (with std dev across experiments)
- F1-score (macro & micro)
- Matthews Correlation Coefficient (MCC)
- Area Under Precision-Recall Curve (AUPRC)
- Training time
- Dataset statistics

### Per-Class Metrics
- Precision
- Recall
- F1-score
- Accuracy
- Class distribution

### Visualizations
- Confusion matrices
- Per-class metrics heatmaps
- Feature importance (for traditional ML models)
- Coefficient analysis (for logistic regression)
- t-SNE embeddings visualization (for prototypical networks)
- Metric distribution histograms

## Organization

Experiments are organized in WandB according to:
- Project: Each approach (prototypical or traditional) has its own project
- Run name: Automatically generated based on dataset and model configuration
- Tags: Standard metrics tags are applied to enable filtering and comparison

## Comparing Experiments

1. Go to your WandB dashboard
2. Navigate to the DOS-Project-Prototypical or DOS-Project-Traditional project
3. Use the table view to compare metrics across runs
4. Create custom visualizations by selecting metrics to plot
5. Group runs by configuration parameters to analyze the impact of different settings

## Tips for Analysis

- Compare final metrics between traditional and prototypical approaches
- Look at per-class performance to identify imbalances or problematic classes
- Use t-SNE visualizations to understand feature space structure
- Analyze feature importance to identify most relevant features
- Monitor standard deviation across experiments to assess model stability

## Extending WandB Logging

If you need to add custom visualizations or metrics:

1. Add your visualization code to `utils/wandb_logger.py`
2. Call your custom logging function from the appropriate training function in `trainer.py`

## Troubleshooting

If you encounter issues with WandB logging:

- Make sure you're logged in (`wandb login`)
- Check your internet connection
- Try running in offline mode (`wandb offline`) and sync later
- Check the WandB logs in the `wandb/` directory
