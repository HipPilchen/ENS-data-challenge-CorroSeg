from utils import iou_score, RollTransform
from losses import SoftIoULoss, FocalLoss
# from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
import argparse
import pandas as pd
import wandb
from tqdm import tqdm
from model import get_model
from dataloader import CorroSeg, CorroSegDataset
import torchvision.transforms as transforms
import numpy as np 
import os
import datetime
from datetime import datetime
from skimage.segmentation import random_walker



def main(args):
    
    if args.experiment_name is None:
            args.experiment_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
    if(args.wandb):
        wandb.init(
            name=args.experiment_name,
            id=args.wandb_id,
            entity=args.wandb_entity,
            project="corroseg",
            config={
            "architecture":args.model_name,
            "learning_rate":args.learning_rate,
            "batch_size":args.batch_size,
            "dropout":args.dropout,
            "p_dropout":args.p_dropout,
            "pretrained":args.pretrained,
            "backbone":args.backbone,
            "loss":args.criterion,
            "threshold":args.threshold,
            "alpha":args.alpha,
            "gamma":args.gamma,
            'Number transforms':args.n_transforms, 
            })
        
        wandb.config = {
            "architecture":args.model_name,
            "epochs":args.num_epochs,
            "learning_rate":args.learning_rate,
        }
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name=args.model_name, backbone_name=args.backbone, 
                      backbone_pretrained=args.pretrained, dropout = args.dropout, pdrop = args.p_dropout).to(device)
    

    # Possible transforms: transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), t
    all_transforms = [None,
        transforms.RandomHorizontalFlip(1),
        transforms.RandomVerticalFlip(1),RollTransform(),
        transforms.Compose([transforms.RandomVerticalFlip(1),transforms.RandomHorizontalFlip(1)]),]
    transform_img = all_transforms[:args.n_transforms]
    
    corro_seg = CorroSeg('data', 'y_train.csv', shuffle = True,
                 batch_size = args.batch_size, valid_ratio = args.valid_ratio, transform_img=transform_img,  
                 transform_test=None, test_params={'batch_size': 1, 'shuffle': False})
    train_loader, val_loader, test_loader = corro_seg.get_loaders()
    print("Data loaded")
    # print("Number of training images: ", len(train_loader.dataset))
    # print("Number of validation images: ", len(val_loader.dataset))
    # print("Number of test images: ", len(test_loader.dataset))

    # Loss function and optimizer definition
    if args.criterion == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.criterion == 'iou':
        criterion = SoftIoULoss()
    elif args.criterion == 'focal':
        criterion = FocalLoss(args.gamma,args.alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.scheduler:  
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, eta_min=1e-5)
        
    best_val_loss = 1000000
    count_loss_decrease = 0
        
    for epoch in tqdm(range(args.num_epochs)):
        # Defreezing strategy
        if args.defreezing_strategy and (epoch % args.unfreeze_at_epoch == 0):
            model.freeze_encoder()
            blocks_to_unfreeze = epoch // args.unfreeze_at_epoch
            model.unfreeze_blocks(blocks_to_unfreeze)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        iou_score_class = SoftIoULoss()
        for image, mask, well in tqdm(train_loader):
            if args.model_need_GRAY:
                image = torch.mean(image, dim=1, keepdim=True)
            mask = torch.mean(mask, dim=1, keepdim = True)
            optimizer.zero_grad()
            image = image.to(device)  # Move image to device
            mask = mask.to(device)  # Move mask to device
            outputs = model(image)
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * image.size(0)
            # Apply threshold to get binary predictions
            preds = (outputs - args.threshold).round()
            train_iou += (-iou_score_class(outputs, mask)).item() * image.size(0)


        if args.scheduler:
            scheduler.step(epoch)
        
        train_loss /= len(train_loader.dataset)
        train_iou /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for image, mask, well in tqdm(val_loader):
                if args.model_need_GRAY:
                    image = torch.mean(image, dim=1, keepdim=True)
                mask = torch.mean(mask, dim=1, keepdim = True)
                image = image.to(device)  # Move image to device
                mask = mask.to(device)  # Move mask to device
                outputs = model(image)
                outputs = outputs.detach()  # Detach outputs from the computation graph
                loss = criterion(outputs, mask)
                val_loss += loss.item() * image.size(0)
                # Apply threshold to get binary predictions
                preds = (outputs - args.threshold).round()
                val_iou += (-iou_score_class(outputs, mask)).item()  * image.size(0)
          
        print('Example of outputs',outputs[0,0,0,:10])
        val_loss /= len(val_loader.dataset)
        val_iou /= len(val_loader.dataset)
        best_val_loss = min(best_val_loss, val_loss)
        if best_val_loss == val_loss:
                    count_loss_decrease = 0        
        else :
            count_loss_decrease += 1
            if count_loss_decrease == args.early_stopping:
                print('Training stopped at %i epochs'%epoch)
                break
        
        # Logging to Weights and Biases
        if(args.wandb):
            wandb.log({'Train Loss': train_loss, 'Train IoU': train_iou,
                    'Validation Loss': val_loss, 'Validation IoU': val_iou}, step=epoch)
        
        print(f'Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Validation Loss: {val_loss:.4f}, Validation IoU: {val_iou:.4f}')
            
    # Testing phase
    model.eval()
    predicted_masks = []  # List to store predicted masks
    with torch.no_grad():
        for image, _, _ in test_loader:  # Ignore the masks in the test loader
            if args.model_need_GRAY:
                image = torch.mean(image, dim=1, keepdim=True)
            
            image = image.to(device)
            output = model(image).detach()
            if not args.random_walk:
                preds = output > args.threshold  # Apply threshold to get binary predictions
                preds = preds.int()

                # Check the unique values and values less than -100
                unique_values = torch.unique(image)
                # if len(unique_values) < 10 or torch.any(image < -100):
                if torch.any(image < -100):
                    preds = torch.zeros_like(preds).int()  # Reset preds to zeros if conditions are met

                # Ensure consistent shape for all flattened masks
                flattened_mask = preds.cpu().numpy().reshape(-1, 36*36)  # Explicitly specify the flattened shape
                predicted_masks.extend(flattened_mask)
            else:
                output = torch.mean(output,dim=1).squeeze(0)
                # print('Output mean',torch.mean(output))
                # print('Output first values',output[:10,0])
                preds_background = output < 0.3
                preds_corrosion = output > 0.5
                if not np.any(preds_corrosion.cpu().numpy()) or not np.any(preds_background.cpu().numpy()):
                    labels  = np.zeros(output.shape)
                else: 
                    markers = np.zeros(output.shape)
                    markers[preds_background.cpu().numpy()] = 1
                    markers[preds_corrosion.cpu().numpy()] = 2
                    image = torch.mean(image,dim=1).squeeze(0)
                    image = (image.cpu().numpy() - np.min(image.cpu().numpy()))*255/(np.max(image.cpu().numpy()) - np.min(image.cpu().numpy()))
                    labels = random_walker(image, markers, beta=1, mode='bf')
                    
                    labels = labels - 1
                predicted_masks.append(labels.flatten())
                # print(predicted_masks)
                
    # Save predicted masks to a CSV file
    predicted_masks = np.vstack(predicted_masks)  # Stack the list of arrays into a single 2D array
    
    df = pd.DataFrame(predicted_masks)
    # print(df)
    files = [f.replace('.npy','') for f in os.listdir('data/processed/images_test')]
    df.index = files
    
    prediction_path = "data/predictions/submission_" + args.experiment_name + '.csv'
    df.to_csv(prediction_path, index=True)

    print("Predicted masks saved to submission_"+args.experiment_name+".csv")
    if args.wandb:
        wandb.finish()
    
"""Parser arguments
"""
def parser_args(parser):
        parser.add_argument(
        "--wandb", action="store_true", default=False,
        help="Whether to log metrics to Weights & Biases"
        )
        parser.add_argument(
            "--experiment_name", type=str, default=None,
            help="Name of the current experiment. Used for wandb logging"
        )
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
        parser.add_argument('--criterion', default='bce', type=str)
        parser.add_argument('-bs','--batch-size', default=64, type=int)
        parser.add_argument('--valid-ratio', default=0.1, type=int)
        parser.add_argument('--model-name', default='cnn', type=str)
        parser.add_argument('--backbone', default='efficientnet-v2-m', type=str)
        parser.add_argument('-lr', '--learning-rate', default=2e-5, type=float,
                            help="learning rate for Adam optimizer")
        parser.add_argument('-eps', '--threshold', default=0.5, type=float,
                            help="Threshold for binary classification")
        parser.add_argument('--defreezing-strategy', action="store_true")
        parser.add_argument('--unfreeze-at-epoch', default=0, type=int, 
                            help="Epoch to start unfreezing")
        parser.add_argument('-wd','--weight-decay',type=float, default = 0.01, help = 'Weight decay')
        parser.add_argument('--gamma',type=float, default = 3, help = 'Gamma for focal loss')
        parser.add_argument('--alpha',type=float, default = 15, help = 'Alpha for focal loss')
        parser.add_argument('--model_need_GRAY',action="store_true", help = 'Whether to tile in 3 channels or not, by default RGB 3 channels')
        parser.add_argument('--pretrained',action="store_true", help="Whether to use a pretrained model or not")
        parser.add_argument('--scheduler',action="store_true", help="Whether to use a scheduler or not")
        parser.add_argument('--dropout',action="store_true", help="Whether to use a dropout or not")
        parser.add_argument('--p_dropout',type=float, default = None, help="Probability of dropout if None standard value is used")
        parser.add_argument('--random_walk',action="store_true", help="Whether to use a random walk or not")
        parser.add_argument('--early_stopping', default=20, type=int, help="Number of epochs to wait before early stopping")
        parser.add_argument('--n_transforms', default=5, type=int, help="Number of transforms to use")
        return parser

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser = parser_args(parser) 
    args = parser.parse_args()  
    main(args)
    
# python3 train.py --wandb --wandb_entity lucasgascon --batch-size 128 --num-epochs 100 --model-name seg_model --experiment_name seg_model --criterion iou

# python3 train.py --wandb --wandb_entity lucasgascon --batch-size 128 --num-epochs 100 --model-name unet --criterion iou --pretrained --defreezing-strategy --unfreeze-at-epoch 10 -lr 10e-4 --weight-decay 10e-3 --n_transforms 3 --experiment_name unet_pretrained_unfreezed_strategy_iou_best_params

# python3 train.py --wandb --wandb_entity lucasgascon --batch-size 128 --num-epochs 100 -lr 10e-4 --weight-decay 10e-3 --n_transforms 3 --model-name unet --criterion iou --experiment_name unet_iou_best_params

# python3 train.py --wandb --wandb_entity lucasgascon --batch-size 128 --num-epochs 100 -lr 10e-4 --weight-decay 10e-3 --n_transforms 3 --model-name unet --criterion iou --experiment_name unet_iou_best_params