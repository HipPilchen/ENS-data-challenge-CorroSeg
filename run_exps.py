import argparse
import numpy as np
from train import main as train_main
from train import parser_args as train_parser_args
from experiments import dict_exp, new_experiments
import json

def create_commands(wandb=True, experiment_name=None, wandb_id=None, wandb_entity=None, num_epochs=None, 
                    criterion=None, batch_size=None, valid_ratio=None, model_name=None, backbone=None, learning_rate=None, 
                    threshold=None, defreezing_strategy=None, unfreeze_at_epoch=None, layers_to_unfreeze_each_time=None, 
                    weight_decay=None, gamma=None, alpha=None, model_need_GRAY=None, pretrained=None, scheduler=None, dropout=None, 
                    random_walk=None, early_stopping=None, p_dropout = None, n_transforms = None):
    
    commands = []
    if wandb:
        commands.append('--wandb')
    if experiment_name is not None:
        commands.append('--experiment_name')
        commands.append(experiment_name)
    if wandb_id is not None:
        commands.append('--wandb_id')
        commands.append(str(wandb_id))
    if wandb_entity is not None:
        commands.append('--wandb_entity')
        commands.append(wandb_entity)
    if num_epochs is not None:
        commands.append('--num-epochs')
        commands.append(str(num_epochs))
    if criterion is not None:
        commands.append('--criterion')
        commands.append(criterion)
    if batch_size is not None:
        commands.append('--batch-size')
        commands.append(str(batch_size))
    if valid_ratio is not None:
        commands.append('--valid-ratio')
        commands.append(str(valid_ratio))
    if model_name is not None:
        commands.append('--model-name')
        commands.append(model_name)
    if backbone is not None:
        commands.append('--backbone')
        commands.append(backbone)
    if learning_rate is not None:
        commands.append('--learning-rate')
        commands.append(str(learning_rate))
    if threshold is not None:
        commands.append('--threshold')
        commands.append(str(threshold))
    if defreezing_strategy:
        commands.append('--defreezing-strategy')
    if unfreeze_at_epoch is not None:
        commands.append('--unfreeze_at_epoch')
        commands.append(str(unfreeze_at_epoch))
    if layers_to_unfreeze_each_time is not None:
        commands.append('--layers_to_unfreeze_each_time')
        commands.append(str(layers_to_unfreeze_each_time))
    if weight_decay is not None:
        commands.append('--weight-decay')
        commands.append(str(weight_decay))
    if gamma is not None:
        commands.append('--gamma')
        commands.append(str(gamma))
    if alpha is not None:
        commands.append('--alpha')
        commands.append(str(alpha))
    if model_need_GRAY:
        commands.append('--model_need_GRAY')
    if pretrained:
        commands.append('--pretrained')
    if scheduler:
        commands.append('--scheduler')
    if dropout:
        commands.append('--dropout')
    if random_walk:
        commands.append('--random_walk')
    if early_stopping is not None:
        commands.append('--early_stopping')
        commands.append(str(early_stopping))
    if p_dropout is not None:
        commands.append('--p_dropout')
        commands.append(str(p_dropout)) 
    if n_transforms is not None:
        commands.append('--n_transforms')
        commands.append(str(n_transforms))
    return commands
    
def run_exps(dico_exp):
    for ind, (num_exp, params) in enumerate(dico_exp.items()):
        print('Running experiment', num_exp)
        parser = argparse.ArgumentParser()
        parser = train_parser_args(parser) 
        try:               
            args = parser.parse_args(create_commands(**params))
            print('Command:', args)
            train_main(args)
        except:
            print('Error in experiment', num_exp)
            continue

    
if __name__ == "__main__":
    run_exps(new_experiments)
    print("All experiments run successfully")
    