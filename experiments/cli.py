import click
import torch
from .train import run_training

@click.group()
def cli():
    """SlowFast Anticipation CLI"""
    pass

@cli.command()
@click.option('--variant', type=click.Choice(['baseline', 'attention', 'roi_guidance', 'hybrid']), required=True, help='Model variant to train')
@click.option('--data-path', type=str, required=True, help='Path to dataset directory (containing train.csv/val.csv)')
@click.option('--epochs', type=int, default=10, help='Number of epochs')
@click.option('--batch-size', type=int, default=8, help='Batch size')
@click.option('--lr', type=float, default=0.01, help='Learning rate')
@click.option('--num-classes', type=int, default=400, help='Number of classes')
@click.option('--experiment-name', type=str, default='SlowFast_Experiment', help='MLflow experiment name')
def train(variant, data_path, epochs, batch_size, lr, num_classes, experiment_name):
    """Train a model variant."""
    run_training(
        variant=variant,
        data_path=data_path,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        mlflow_experiment=experiment_name
    )

from .eval import run_evaluation

@cli.command()
@click.option('--variant', type=click.Choice(['baseline', 'attention', 'roi_guidance', 'hybrid']), required=True, help='Model variant')
@click.option('--data-path', type=str, required=True, help='Path to dataset')
@click.option('--checkpoint', type=str, required=True, help='Path to checkpoint.pth')
@click.option('--num-classes', type=int, default=400, help='Number of classes')
@click.option('--batch-size', type=int, default=8, help='Batch size')
def eval(variant, data_path, checkpoint, num_classes, batch_size):
    """Evaluate a trained model."""
    run_evaluation(
        variant=variant,
        data_path=data_path,
        checkpoint_path=checkpoint,
        num_classes=num_classes,
        batch_size=batch_size
    )

if __name__ == '__main__':
    cli()
