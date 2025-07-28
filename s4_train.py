import argparse
import math
import os
import yaml
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from os.path import join
import wandb
import multiprocessing

from helpers.predict_utils import Config, get_data, train_model, rescale_data, evaluate_model
from helpers.models import TimeSeriesRegressorWrapper

def wandb_process(arguments):
    config = arguments['config']
    wandb.agent(arguments['sweep_id'],
                lambda: train_model(arguments['trainsets'], arguments['valsets'], arguments['testsets'], arguments['device'], config.wandb_mode, config.wandb_project, config.name))
    # print(arguments['id'])

def load_controller_configs(person_dir):
    """
    Load configs from configs folder using standard naming:
    - modular_fs.yaml for free space
    - modular_inter.yaml for interaction model
    """
    configs = {}
    config_dir = join('data', person_dir, 'configs')

    if not os.path.exists(config_dir):
        print(f"Error: Config directory not found: {config_dir}")
        return configs
    
    # Free space congif
    fs_config_path = join(config_dir, 'modular_fs.yaml')
    if os.path.exists(fs_config_path):
        with open(fs_config_path, 'r') as file:
            wandb_config = yaml.safe_load(file)
            configs['free_space'] = Config(wandb_config)
            print(f"Loaded free space config file")
    else:
        print(f"Warning: free space config not found")

    # Interaction congif
    inter_config_path = join(config_dir, 'modular_inter.yaml')
    if os.path.exists(inter_config_path):
        with open(inter_config_path, 'r') as file:
            wandb_config = yaml.safe_load(file)
            configs['interaction'] = Config(wandb_config)
            print(f"Loaded interaction config file")
    else:
        print(f"Warning: Interaction config not found")

    return configs

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Timeseries data analysis')
    parser.add_argument('--person_dir', type=str, required=True, help='Person directory')
    parser.add_argument('--intact_hand', type=str, required=True, help='Intact hand (Right/Left)')
    # parser.add_argument('--config_name', type=str, required=True, help='Training configuration')
    parser.add_argument('--controller_mode', choices=['free_space', 'interaction', 'both'], 
                   default='both', help='Which controller(s) to train')
    parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--allow_tf32', action='store_true', help='Allow TF32')
    parser.add_argument('-v', '--visualize', action='store_true', help='Plot data exploration results')
    parser.add_argument('-hs', '--hyperparameter_search', action='store_true', help='Perform hyperparameter search')
    parser.add_argument('-t', '--test', action='store_true', help='Test the model')
    parser.add_argument('-s', '--save_model', action='store_true', help='Save a model')
    parser.add_argument('-e', '--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--perturb', action='store_true', help='Perturb the data')
    args = parser.parse_args()

    sampling_frequency = 60
    experiment_name = 'perturb' if args.perturb else 'non_perturb'

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('Using CUDA')
    #
    #     # List available GPUs
    #     if args.multi_gpu:
    #         n_gpus = torch.cuda.device_count()
    #         print(f'Number of available GPUs: {n_gpus}')
    #         for i in range(n_gpus):
    #             print(f'GPU{i}: {torch.cuda.get_device_name(i)}')

    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     print('Using MPS')
    else:
        device = torch.device("cpu")
        print('Using CPU')

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        print('TF32 enabled')

    # with open(join('data', args.person_dir, 'configs', f'{args.config_name}.yaml'), 'r') as file:
    #     wandb_config = yaml.safe_load(file)
    #     config = Config(wandb_config)

    print(f"Loading configs from: data/{args.person_dir}/configs/")
    configs = load_controller_configs(args.person_dir)
    # Validate Configs based on mode
    if args.controller_mode == 'both' and len(configs) <2:
        print("Error: Both configs required for 'both' mode")
        print(f"Found configs: {list(configs.keys())}")
        print("Expected: modular_fs.yaml and modular_inter.yaml")
        exit(1)
    elif args.controller_mode == 'free_space' and 'free_space' not in configs:
        print("Error: modular_fs.yaml required for free_space mode")
        exit(1)
    elif args.controller_mode == 'interaction' and 'interaction' not in configs:
        print("Error: modular_inter.yaml required for interaction mode")
        exit(1)


    # print("Config dict BEFORE passing to wandb:", config.to_dict())


    # data_dirs = [join('data', args.person_dir, 'recordings', recording, 'experiments', '1') for recording in
    #              config.recordings]

    # test_dirs = [join('data', args.person_dir, 'recordings', recording, 'experiments', '1') for recording in
    #              config.test_recordings] if config.test_recordings is not None else []
    if args.evaluate:
        if args.perturb:
            perturb_file = join('data', args.person_dir, 'online_trials', experiment_name,
                                'perturber.npy')
        else:
            perturb_file = join('data', 'eye.npy')
    else:
        perturb_file = None

    # trainsets, valsets, combined_sets, testsets = get_data(config, data_dirs, args.intact_hand, visualize=args.visualize, test_dirs=test_dirs, perturb_file=perturb_file)
    # print(f"Number of valsets: {len(valsets)}")
    # for i, val_set in enumerate(valsets):
    #     print(f"Val set {i} length: {len(val_set)}")

    if args.controller_mode in ['free_space', 'both'] and 'free_space' in configs:
        print("\n" + "="*60)
        print("TRAINING FREE-SPACE CONTROLLER")
        print("="*60)

        config = configs['free_space']

        # Build data directordies from config
        data_dirs = [join('data', args.person_dir, 'recordings', recording, 'experiments', '1')
                     for recording in config.recordings]
        test_dirs = [join('data', args.person_dir, 'recordings', recording, 'experiments', '1')
                     for recording in config.test_recordings] if config.test_recordings is not None else []
        
        print(f"Free-space recordings: {config.recordings}")
        print(f"Input features: {len(config.features)} (EMG only)")
        print(f"Output targets: {len(config.targets)} (positions)")

        # Get free-space data
        trainsets, valsets, combined_sets, testsets = get_data(
            config, data_dirs, args.intact_hand, 
            visualize=args.visualize, test_dirs=test_dirs, perturb_file=perturb_file
        )
    
        if args.hyperparameter_search:
            sweep_id = wandb.sweep(config.to_dict(), project=config.wandb_project)
            wandb.agent(sweep_id, lambda: train_model(trainsets, valsets, testsets, device, config.wandb_mode, config.wandb_project, config.name))
        
        if args.test:
            print("Training free-space model...")
            free_model = train_model(trainsets, valsets, testsets, device, 
                                config.wandb_mode, config.wandb_project, 
                                config.name, config, args.person_dir)
            
            # Generate predictions for test sets
            for set_id, test_set in enumerate(valsets + testsets):
                val_pred = free_model.predict(test_set, config.features, config.targets).squeeze(0).to('cpu').detach().numpy()
                test_set[config.targets] = val_pred
                test_set = rescale_data(test_set, args.intact_hand)
                test_set.to_parquet(join((data_dirs + test_dirs)[set_id], f'pred_angles-{config.name}.parquet'))
            
            # Save model
            if args.save_model:
                free_model.to(torch.device('cpu'))
                os.makedirs(join('data', args.person_dir, 'models'), exist_ok=True)
                free_model.save(join('data', args.person_dir, 'models', f'{config.name}.pt'))
                print(f"Saved free-space model: {config.name}.pt")

    # if args.hyperparameter_search:  # training on training set, evaluation on test set
    #     sweep_id = wandb.sweep(wandb_config, project=config.wandb_project)
    #     # wandb.agent(sweep_id, lambda: train_model(trainsets, valsets, testsets, device, config.wandb_mode, config.wandb_project, config.name))
    #     pool = multiprocessing.Pool(processes=4)
    #     pool.map(wandb_process, [{'id': i, 'config': config, 'sweep_id': sweep_id, 'trainsets': trainsets, 'valsets': valsets, 'testsets': testsets, 'device': device} for i in range(4)])


    if args.controller_mode in ['interaction', 'both'] and 'interaction' in configs:
        print("\n" + "="*60)
        print("TRAINING INTERACTION CONTROLLER") 
        print("="*60)

        config = configs['interaction']

        data_dirs = [join('data', args.person_dir, 'recordings', recording, 'experiments', '1') 
                 for recording in config.recordings]
        test_dirs = [join('data', args.person_dir, 'recordings', recording, 'experiments', '1') 
                 for recording in config.test_recordings] if config.test_recordings is not None else []
        
        print(f"Interaction recordings: {config.recordings}")
        print(f"Input features: {len(config.features)} (EMG + Force)")
        print(f"Output targets: {len(config.targets)} (positions)")
        
        # Get interaction data
        trainsets, valsets, combined_sets, testsets = get_data(
            config, data_dirs, args.intact_hand,
            visualize=args.visualize, test_dirs=test_dirs, perturb_file=perturb_file
        )
        
        if args.hyperparameter_search:
            sweep_id = wandb.sweep(config.to_dict(), project=config.wandb_project)
            wandb.agent(sweep_id, lambda: train_model(trainsets, valsets, testsets, device, config.wandb_mode, config.wandb_project, config.name))
        
        if args.test:
            print("Training interaction model...")
            interaction_model = train_model(trainsets, valsets, testsets, device,
                                        config.wandb_mode, config.wandb_project,
                                        config.name, config, args.person_dir)
            
            # Generate predictions for test sets
            for set_id, test_set in enumerate(valsets + testsets):
                val_pred = interaction_model.predict(test_set, config.features, config.targets).squeeze(0).to('cpu').detach().numpy()
                test_set[config.targets] = val_pred
                test_set = rescale_data(test_set, args.intact_hand)
                test_set.to_parquet(join((data_dirs + test_dirs)[set_id], f'pred_angles-{config.name}.parquet'))
            
            # Save model
            if args.save_model:
                interaction_model.to(torch.device('cpu'))
                os.makedirs(join('data', args.person_dir, 'models'), exist_ok=True)
                interaction_model.save(join('data', args.person_dir, 'models', f'{config.name}.pt'))
                print(f"Saved interaction model: {config.name}.pt")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

    if args.controller_mode == 'both':
        print("Free-space controller: EMG -> Position")
        print("Interaction controller: EMG + Force → Position")
    elif args.controller_mode == 'free_space':
        print("Free-space controller: EMG -> Position")
    elif args.controller_mode == 'interaction':
        print("Interaction controller: EMG + Force -> Position")

    if args.evaluate:
        ###### to generate trajectory:
        # config.person_dir = args.person_dir
        # config.intact_hand = args.intact_hand
        # config.experiment_name = experiment_name
        # config.perturb = args.perturb
        # config.wandb_project = 'study_participants_online'
        # config.wandb_mode = 'disabled'
        # if args.perturb:
        #     config.name = config.name + '_perturb'
        #
        #
        # wandb.init(mode=config.wandb_mode, project=config.wandb_project, name=config.name, config=config)
        # config = wandb.config
        #
        # model = TimeSeriesRegressorWrapper(device=device, input_size=len(config.features),
        #                                    output_size=len(config.targets),
        #                                    **config)
        # model.to('cpu')
        # model.load(join('data', args.person_dir, 'online_trials', experiment_name, 'models',
        #                 f'{args.person_dir}-online_last.pt'))
        # model.eval()
        # for set_id, test_set in enumerate(valsets + testsets):
        #     val_pred = model.predict(test_set, config.features, config.targets).squeeze(0).to('cpu').detach().numpy()
        #     test_set.loc[:, config.targets] = val_pred
        #
        #
        #     test_set = rescale_data(test_set, args.intact_hand)
        #
        #     test_set.to_parquet(join((data_dirs + test_dirs)[set_id], f'pred_angles-{config.name}_last.parquet'))

        ###### to evaluate model:
        config.person_dir = args.person_dir
        config.intact_hand = args.intact_hand
        config.experiment_name = experiment_name
        config.perturb = args.perturb
        config.wandb_project = 'study_participants_online'
        config.wandb_mode = 'online'
        if args.perturb:
            config.name = config.name + '_perturb'


        wandb.init(mode=config.wandb_mode, project=config.wandb_project, name=config.name, config=config)
        config = wandb.config

        # model = TimeSeriesRegressorWrapper(device=device, input_size=len(config.features),
        #                                    output_size=len(config.targets),
        #                                    **config)
        model = TimeSeriesRegressorWrapper(
            input_size=len(config.features),
            output_size=len(config.targets),
            device=device,
            n_epochs=config.n_epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            model_type=config.model_type,
            **{k: v for k, v in config.__dict__.items() if k not in (
                'n_epochs', 'learning_rate', 'weight_decay', 'warmup_steps', 'model_type'
            )}
        )
        model.to('cpu')
        model.eval()
        epoch = 0
        best_val_loss = math.inf
        while True:
            try:
                model.load(join('data', args.person_dir, 'online_trials', experiment_name, 'models', f'{args.person_dir}-online_{epoch}.pt'))
            except:
                break
            model.to(device)
            val_loss, test_loss, all_losses = evaluate_model(model, valsets, testsets, device, config)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wandb.run.summary['best_epoch'] = epoch
                wandb.run.summary['best_val_loss'] = best_val_loss
            if test_loss < wandb.run.summary.get('best_test_loss', math.inf):
                wandb.run.summary['best_test_loss'] = test_loss
                wandb.run.summary['best_test_epoch'] = epoch
            wandb.run.summary['used_epochs'] = epoch

            test_recording_names = config.test_recordings if config.test_recordings is not None else []
            log = {f'val_loss/{(config.recordings + test_recording_names)[set_id]}': loss for set_id, loss in
                   enumerate(all_losses)}
            log['total_val_loss'] = val_loss
            log['total_test_loss'] = test_loss
            log['epoch'] = epoch
            wandb.log(log)
            print(val_loss)

            epoch += 1

        try:
            model.load(join('data', args.person_dir, 'online_trials', experiment_name, 'models', f'{args.person_dir}-online_last.pt'))
        except:
            print('No last model found')
        model.to(device)
        val_loss, test_loss, all_losses = evaluate_model(model, valsets, testsets, device, config)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wandb.run.summary['best_epoch'] = epoch
            wandb.run.summary['best_val_loss'] = best_val_loss
        if test_loss < wandb.run.summary.get('best_test_loss', math.inf):
            wandb.run.summary['best_test_loss'] = test_loss
            wandb.run.summary['best_test_epoch'] = epoch
        wandb.run.summary['used_epochs'] = epoch


        test_recording_names = config.test_recordings if config.test_recordings is not None else []
        log = {f'val_loss/{(config.recordings + test_recording_names)[set_id]}': loss for set_id, loss in
               enumerate(all_losses)}
        log['total_val_loss'] = val_loss
        log['total_test_loss'] = test_loss
        log['epoch'] = epoch
        wandb.log(log)
        wandb.run.summary['last'] = log
        print(val_loss)
