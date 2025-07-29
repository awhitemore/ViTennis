# Dataset: @https://www.kaggle.com/datasets/orvile/tennis-player-actions-dataset
import dataset
from torch.utils.data import DataLoader, random_split
import torch
import random 
import numpy as np
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTImageProcessor
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback
import evaluate
from PIL import Image
import PIL
from tqdm import tqdm
import torch.utils.data

def load_datasets(data_dir="data/images", train_split=0.8, batch_size=32):
    """Load dataset and return train/val DataLoaders with 80/20 split."""
    full_dataset = dataset.TennisActionDataset(data_dir=data_dir)
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Dataset loaded: {total_size} total images")
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    print(f"Classes: {full_dataset.classes()}")
    return train_dataset, val_dataset, full_dataset.classes()

if __name__ == "__main__":
    # Load datasets (return splits, not DataLoaders)
    train_dataset, val_dataset, classes = load_datasets()

    # Load the VIT image processor
    image_processor = ViTImageProcessor.from_pretrained("facebook/deit-small-patch16-224")

    # Load evaluation metrics
    metric = evaluate.load("accuracy")

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids) or {}

    # Prepare the model
    num_classes = len(classes)
    model = ViTForImageClassification.from_pretrained(
        "facebook/deit-small-patch16-224",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        use_safetensors=True,
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir = "./temp/4-tennis",
        per_device_eval_batch_size = 16,  # Increased batch size for faster evaluation, but still CPU-friendly
        per_device_train_batch_size = 16,  # Increased batch size for more stable gradients, but not too large for CPU
        num_train_epochs = 3,  # Train for more epochs to improve learning
        max_steps = 100,  # Allow up to 100 steps, but will likely stop earlier due to dataset size
        eval_strategy = "steps",
        save_strategy = "steps",
        save_steps = 20,  # Save less frequently to reduce I/O overhead
        eval_steps = 20,  # Evaluate less frequently to reduce overhead
        logging_steps = 5,  # Log more frequently for better monitoring
        learning_rate = 2e-4,  # Keep learning rate the same (already reasonable)
        save_total_limit = 1,  # Only keep the best checkpoint
        load_best_model_at_end = True,
        remove_unused_columns = False,
        push_to_hub = False,
        dataloader_num_workers = 0,  # for CPU
    )

    # Define early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience = 10,
        early_stopping_threshold = 0.0,
    )

    # Define the Trainer (use datasets, not DataLoaders)
    trainer = Trainer(
        model = model,
        args = training_args,
        compute_metrics = compute_metrics,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        callbacks = [early_stopping_callback],
    )

    # Train the model
    train_results = trainer.train()
    # Save model in safetensors format for security
    model.save_pretrained("./temp/4-tennis", safe_serialization=True)
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    image_processor.save_pretrained("./temp/4-tennis")

    # Evaluate the model
    metrics = trainer.evaluate(eval_dataset=val_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

