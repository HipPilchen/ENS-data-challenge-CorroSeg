from utils import iou_score
# from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
import argparse
import pandas as pd
import wandb
from tqdm import tqdm
from model import get_model
from dataloader import CorroSeg
import numpy as np 
import os

def main(args):
    if(args.wandb):
        wandb.init(
            name=args.experiment_name,
            id=args.wandb_id,
            project="corroseg",
        )
        
        wandb.config = {
            "architecture":args.model_name,
            "epochs":args.num_epochs,
            "learning_rate":args.learning_rate,
        }
        
    device ='cpu'
    model = get_model(args.model_name).to(device)
    
    corro_seg = CorroSeg('data', 'y_train.csv', shuffle = True,
                 batch_size = args.batch_size, valid_ratio = args.valid_ratio, transform_img=None, transform_mask=None, 
                 transform_test=None, test_params={'batch_size': args.batch_size, 'shuffle': False})
    train_loader, val_loader, test_loader = corro_seg.get_loaders()

    # Loss function and optimizer definition
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    for epoch in tqdm(range(args.num_epochs)):
        # Defreezing strategy
        if epoch % args.unfreeze_at_epoch == 0:
            layers_to_unfreeze = (epoch // args.unfreeze_at_epoch) * args.layers_to_unfreeze_each_time
            model.unfreeze_layers(layers_to_unfreeze)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        
        for image, mask, well in tqdm(train_loader):
            mask = mask.view(-1, 1, 36, 36)
            optimizer.zero_grad()
            outputs = model(image.repeat(1, 3, 1, 1))
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * image.size(0)
            preds = outputs > args.threshold  # Apply threshold to get binary predictions
            train_iou += iou_score(preds, mask).item() * image.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_iou /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for image, mask, well in tqdm(val_loader):
                mask = mask.view(-1, 1, 36, 36)
                outputs = model(image.repeat(1, 3, 1, 1))
                loss = criterion(outputs, mask)
                val_loss += loss.item() * image.size(0)
                preds = outputs > 0.5  # Apply threshold to get binary predictions
                val_iou += iou_score(preds, mask).item() * image.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_iou /= len(val_loader.dataset)
        
        # Logging to Weights and Biases
        if(args.wandb):
            wandb.log({'Train Loss': train_loss, 'Train IoU': train_iou,
                    'Validation Loss': val_loss, 'Validation IoU': val_iou}, step=epoch)
        
        print(f'Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Validation Loss: {val_loss:.4f}, Validation IoU: {val_iou:.4f}')

    # Testing phase
    model.eval()
    predicted_masks = []  # List to store predicted masks  
    with torch.no_grad():
        for images, _ in test_loader:  # Ignore the masks in the test loader
            images = images.repeat(1, 3, 1, 1)  # Repeat the images 3 times to match the input channels of the model
            
            # Forward pass
            outputs = model(images)
            
            # Convert predicted masks to numpy arrays
            predicted_masks.extend(outputs.cpu().numpy())
    
    # Save predicted masks to a CSV file
    predicted_masks = np.array(predicted_masks)
    df = pd.DataFrame(predicted_masks)
    df.to_csv("predicted_masks.csv", index=False)
    
    print("Predicted masks saved to predicted_masks.csv")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()  
    parser.add_argument(
        "--wandb", action="store_true", default=True,
        help="Whether to log metrics to Weights & Biases"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None,
        help="Name of the current experiment. Used for wandb logging"
    )
    parser.add_argument('-output_dir', default='wandb', type=int)
    parser.add_argument(
        "--wandb_id", type=str, default=None,
        help="ID of a previous run to be resumed"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None,
        help="wandb username or team name to which runs are attributed"
    )
    parser.add_argument('-n', '--num-epochs', default=5, type=int,
                        help="number of epochs to run") 
    parser.add_argument('-bs','--batch-size', default=1, type=int)
    parser.add_argument('--valid-ratio', default=0.1, type=int)
    parser.add_argument('--model-name', default='resnet18', type=str)
    parser.add_argument('-lr', '--learning-rate', default=2e-5, type=float,
                        help="learning rate for Adam optimizer")
    parser.add_argument('-eps', '--threshold', default=0.5, type=float,
                        help="Threshold for binary classification")
    parser.add_argument('-unfreeze_at_epoch', default=3, type=int, 
                        help="Epoch to start unfreezing")
    parser.add_argument('-layers_to_unfreeze_each_time', default=1, type=int,
                        help="Number of layers to unfreeze")
    parser.add_argument('-wd','--weight-decay',type=float, default = 0.01, help = 'Weight decay')

    args = parser.parse_args()
    main(args)
    
# python3 train.py --experiment_name "ResNet18" --output_dir "wandb" --num_epochs 500 --batch_size 1 --valid_ratio 0.1 --model_name "resnet18" --learning_rate 2e-5 --threshold 0.5 --unfreeze_at_epoch 3 --layers_to_unfreeze_each_time 1 --weight_decay 0.01