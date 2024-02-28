import argparse
import numpy as np
from train import main as train_main
from train import parser_args as train_parser_args
from experiments import dict_exp
import json

def create_commands(wandb, experiment_name, output_dir, wandb_id, wandb_entity, num_epochs, 
                    criterion, batch_size, valid_ratio, model_name, backbone, learning_rate, 
                    threshold, defreezing_strategy, unfreeze_at_epoch, layers_to_unfreeze_each_time, 
                    weight_decay, gamma, alpha, model_need_GRAY, pretrained, scheduler, dropout, 
                    random_walk, early_stopping):
    
    commands = []
    if wandb:
        commands.append('--wandb')
    if experiment_name is not None:
        commands.append(f'--experiment_name {experiment_name}')
    if output_dir is not None:
        commands.append(f'--output_dir {output_dir}')
    if wandb_id is not None:
        commands.append(f'--wandb_id {wandb_id}')
    if wandb_entity is not None:
        commands.append(f'--wandb_entity {wandb_entity}')
    if num_epochs is not None:
        commands.append(f'--num-epochs {num_epochs}')
    if criterion is not None:
        commands.append(f'--criterion {criterion}')
    if batch_size is not None:
        commands.append(f'--batch-size {batch_size}')
    if valid_ratio is not None:
        commands.append(f'--valid-ratio {valid_ratio}')
    if model_name is not None:
        commands.append(f'--model-name {model_name}')
    if backbone is not None:
        commands.append(f'--backbone {backbone}')
    if learning_rate is not None:
        commands.append(f'--learning-rate {learning_rate}')
    if threshold is not None:
        commands.append(f'--threshold {threshold}')
    if defreezing_strategy:
        commands.append('--defreezing-strategy')
    if unfreeze_at_epoch is not None:
        commands.append(f'--unfreeze_at_epoch {unfreeze_at_epoch}')
    if layers_to_unfreeze_each_time is not None:
        commands.append(f'--layers_to_unfreeze_each_time {layers_to_unfreeze_each_time}')
    if weight_decay is not None:
        commands.append(f'--weight-decay {weight_decay}')
    if gamma is not None:
        commands.append(f'--gamma {gamma}')
    if alpha is not None:
        commands.append(f'--alpha {alpha}')
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
        commands.append(f'--early_stopping {early_stopping}')
    return commands
    
def run_exps():
    accuracies = {}
    i = 0
    file_path = 'all_experiment_results_valid_final.json'
    for ind, (num_exp, params) in enumerate(dict_exp.items()):
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
    run_exps()
    print("All experiments run successfully")
    