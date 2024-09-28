import argparse
import math
import os
import yaml
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Timeseries data analysis')
    parser.add_argument('--person_dir', type=str, required=True, help='Person directory')
    parser.add_argument('--intact_hand', type=str, required=True, help='Intact hand (Right/Left)')
    parser.add_argument('--config_name', type=str, required=True, help='Training configuration')
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

    with open(join('data', args.person_dir, 'configs', f'{args.config_name}.yaml'), 'r') as file:
        wandb_config = yaml.safe_load(file)
        config = Config(wandb_config)

    data_dirs = [join('data', args.person_dir, 'recordings', recording, 'experiments', '1') for recording in
                 config.recordings]

    test_dirs = [join('data', args.person_dir, 'recordings', recording, 'experiments', '1') for recording in
                 config.test_recordings] if config.test_recordings is not None else []
    if args.evaluate:
        if args.perturb:
            perturb_file = join('data', args.person_dir, 'online_trials', experiment_name,
                                'perturber.npy')
        else:
            perturb_file = join('data', 'eye.npy')
    else:
        perturb_file = None

    trainsets, valsets, combined_sets, testsets = get_data(config, data_dirs, args.intact_hand, visualize=args.visualize, test_dirs=test_dirs, perturb_file=perturb_file)

    if args.hyperparameter_search:  # training on training set, evaluation on test set
        sweep_id = wandb.sweep(wandb_config, project=config.wandb_project)
        # wandb.agent(sweep_id, lambda: train_model(trainsets, valsets, testsets, device, config.wandb_mode, config.wandb_project, config.name))
        pool = multiprocessing.Pool(processes=4)
        pool.map(wandb_process, [{'id': i, 'config': config, 'sweep_id': sweep_id, 'trainsets': trainsets, 'valsets': valsets, 'testsets': testsets, 'device': device} for i in range(4)])



    if args.test:  # trains on the training set and saves the test set predictions
        os.makedirs(join('data', args.person_dir, 'models'), exist_ok=True)

        model = train_model(trainsets, valsets, testsets, device, config.wandb_mode, config.wandb_project, config.name, config, args.person_dir)

        for set_id, test_set in enumerate(valsets + testsets):
            val_pred = model.predict(test_set, config.features, config.targets).squeeze(0).to('cpu').detach().numpy()
            test_set[config.targets] = val_pred

            test_set = rescale_data(test_set, args.intact_hand)

            test_set.to_parquet(join((data_dirs + test_dirs)[set_id], f'pred_angles-{config.name}.parquet'))


        if args.save_model:  # trains on the whole dataset and saves the model
            # model = train_model(combined_sets, valsets, testsets, device, config.wandb_mode, config.wandb_project, config.name, config)

            model.to(torch.device('cpu'))
            os.makedirs(join('data', args.person_dir, 'models'), exist_ok=True)
            model.save(join('data', args.person_dir, 'models', f'{config.name}.pt'))

    elif args.evaluate:
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

        model = TimeSeriesRegressorWrapper(device=device, input_size=len(config.features),
                                           output_size=len(config.targets),
                                           **config)
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
