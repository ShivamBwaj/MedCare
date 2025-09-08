import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import os
import logging
import datetime
import time
from collections import Counter

# Setup logging
log_dir = "training_logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"{log_dir}/nail_transfer_training_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Training configuration
data_dir = "ai/nail_data"
skin_model_path = "ai/skin_model.pt"
batch_size = 32
epochs = 15
lr = 0.0001
device = "cuda" if torch.cuda.is_available() else "cpu"

logging.info("=== NAIL CNN TRANSFER LEARNING FROM SKIN MODEL ===")
logging.info(f"Device: {device}")
logging.info(f"Loading pre-trained skin model: {skin_model_path}")

# Data transformations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1)
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    # Load datasets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "valid"), transform=val_transforms)
    
    classes = train_dataset.classes
    num_classes = len(classes)
    
    logging.info(f"Nail classes: {classes}")
    logging.info(f"Training samples: {len(train_dataset)}")
    logging.info(f"Validation samples: {len(val_dataset)}")
    
    # Check class distribution
    train_targets = [train_dataset[i][1] for i in range(len(train_dataset))]
    class_counts = Counter(train_targets)
    logging.info(f"Class distribution: {dict(class_counts)}")
    
    # Create weighted sampler for balanced training
    class_weights = [1.0 / class_counts[i] for i in range(num_classes)]
    sample_weights = [class_weights[target] for target in train_targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Load pre-trained skin model
    if os.path.exists(skin_model_path):
        logging.info("Loading skin model for transfer learning...")
        skin_checkpoint = torch.load(skin_model_path, map_location=device)
        
        # Create MobileNetV2 model (same as skin model)
        model = models.mobilenet_v2(weights=None)  # No ImageNet weights
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)  # Original skin classes
        
        # Load skin model weights
        model.load_state_dict(skin_checkpoint["state_dict"])
        logging.info("✅ Loaded skin model weights successfully!")
        
        # Adapt for nail classes
        if num_classes != 3:
            logging.info(f"Adapting classifier for {num_classes} nail classes...")
            model.classifier[1] = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(model.classifier[1].in_features, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
        else:
            # Same number of classes, just fine-tune
            logging.info("Same number of classes - fine-tuning existing classifier...")
            
    else:
        logging.warning(f"Skin model not found at {skin_model_path}")
        logging.info("Using ImageNet pre-trained MobileNetV2...")
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    model = model.to(device)
    
    # Freeze feature extractor, only train classifier initially
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float().to(device))
    optimizer = optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    logging.info("Starting transfer learning training...")
    best_val_acc = 0.0
    patience = 7
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        scheduler.step()
        
        logging.info(f"Epoch {epoch+1}/{epochs}:")
        logging.info(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        logging.info(f"  Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        logging.info(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "model_type": "mobilenet_v2_transfer_from_skin",
                "best_val_acc": best_val_acc,
                "epoch": epoch + 1,
                "source_model": "skin_model.pt"
            }, "ai/nail_transfer_model.pt")
            logging.info(f"🎯 New best model saved! Val Acc: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
        
        # Unfreeze more layers after epoch 5 for fine-tuning
        if epoch == 4:
            logging.info("Unfreezing last feature layers for fine-tuning...")
            for param in model.features[-3:].parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=lr/2, weight_decay=0.01)
    
    logging.info("=== TRANSFER LEARNING SUMMARY ===")
    logging.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    logging.info(f"Improvement over random: {best_val_acc - 100/num_classes:.2f}%")
    logging.info(f"Model saved: ai/nail_transfer_model.pt")
    
except Exception as e:
    logging.error(f"Transfer learning failed: {e}")
    import traceback
    logging.error(traceback.format_exc())

print(f"\n✅ Transfer learning completed! Check logs at: {log_file}")
