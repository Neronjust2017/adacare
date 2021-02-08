import numpy as np
import argparse
import os
import imp
import re
import pickle
import random
# import matplotlib.pyplot as plt
# import matplotlib as mpl
from thop import profile
RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
from datetime import datetime

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic=True

from utils import utils
from utils.readers import DecompensationReader
from utils.preprocessing import Discretizer, Normalizer
from utils import metrics
from utils import common_utils
from utils.lr_scheduler import GradualWarmupScheduler
from model import AdaCare
from tensorboardX import SummaryWriter
from hyperopt import hp, tpe, fmin, Trials
from hyper_opt.util import init_obj, to_np, get_mnt_mode, save_checkpoint, \
    write_json, get_logger, analyze, progress, load_checkpoint

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


save_path = './hyperopt'

space = {
    'arch':
        {
            "input_dim": 76,
            "output_dim": 1,
            "rnn_dim": hp.choice('rnn_dim', [384]),
            "kernel_size": hp.choice('kernel_size', [2]),
            "kernel_num": hp.choice('kernel_num', [384]),
            "dropout_rate": hp.choice('dropout_rate', [0.5]),
        },

    'batch_size': hp.choice('batch_size', [128]),
    'epochs': hp.choice('epochs', [50]),

    'optimizer':
        hp.choice('optimizer', [
            {
                'type': 'Adam',
                'args':
                    {
                        'lr': hp.choice('lr', [0.001, 0.0005]),
                        'weight_decay': hp.choice('weight_decay', [1e-3, 1e-4]),
                        'amsgrad': True
                    }
            }
        ]),

    'lr_scheduler':
        hp.choice('lr_scheduler', [
            {
                "type": "ReduceLROnPlateau",
                "args": {
                    "mode": "min",
                    "factor": hp.choice('factor', [0.1, 0.5]),
                    "patience": 6,
                    "verbose": False,
                    "threshold": 0.0001,
                    "threshold_mode": "rel",
                    "cooldown": 0,
                    "min_lr": 0,
                    "eps": 1e-08
                }

            },
            {
                "type": "StepLR",
                "args":
                    {
                        "step_size": hp.choice('step_size', [30, 40]),
                        "gamma": hp.choice('StepLR_gamma', [0.1, 0.5])
                    }
            },

            {
                "type": "GradualWarmupScheduler",
                "args": {
                    "multiplier": hp.choice('multiplier', [1, 1.5]),
                    "total_epoch": hp.choice('total_epoch', [5, 10]),
                    "after_scheduler": {
                        "type": "ReduceLROnPlateau",
                        "args": {
                            "mode": "min",
                            "factor": hp.choice('factor2', [0.1, 0.5]),
                            "patience": 6,
                            "verbose": False,
                            "threshold": 0.0001,
                            "threshold_mode": "rel",
                            "cooldown": 0,
                            "min_lr": 0,
                            "eps": 1e-08
                        }

                    }
                }
            }
        ]),
}


def opt(hype_space):
    data_path = 'data/'
    small_part = 0

    # Paths to save log, checkpoint, tensorboard logs and results
    run_id = datetime.now().strftime(r'%m%d_%H%M%S')
    base_path = os.path.join(save_path, run_id)
    result = os.path.join(base_path, 'results.txt')
    os.makedirs(base_path)
    write_json(hype_space, base_path + '/hype_space.json')

    try:
        loss = []
        acc = []
        prec0 = []
        prec1 = []
        rec0 = []
        rec1 = []
        auroc = []
        auprc = []
        minpse = []

        for split in range(4):

            base_path_split = os.path.join(base_path, 'split_' + str(split))
            # result_dir = os.path.join(base_path_split, 'result')
            file_name = os.path.join(base_path_split, 'model')
            tb_dir = os.path.join(base_path_split, 'tb_log')
            # os.makedirs(result_dir)
            os.makedirs(tb_dir)

            log_writer = SummaryWriter(tb_dir)

            ''' Prepare training data'''
            print('Preparing training data ... ')
            train_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(
                data_path, 'train'), listfile=os.path.join(data_path, 'train_listfile_{}.csv'.format(split)),
                small_part=small_part)

            val_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(
                data_path, 'train'), listfile=os.path.join(data_path, 'val_listfile_{}.csv'.format(split)),
                small_part=small_part)

            discretizer = Discretizer(timestep=1.0, store_masks=True,
                                      impute_strategy='previous', start_time='zero')

            discretizer_header = discretizer.transform(train_data_loader._data["X"][0])[1].split(',')
            cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

            normalizer = Normalizer(fields=cont_channels)
            normalizer_state = 'decomp_normalizer'
            normalizer_state = os.path.join(os.path.dirname(data_path), normalizer_state)
            normalizer.load_params(normalizer_state)

            train_data_gen = utils.BatchGenDeepSupervision(train_data_loader, discretizer,
                                                           normalizer, hype_space['batch_size'], shuffle=True, return_names=True)
            val_data_gen = utils.BatchGenDeepSupervision(val_data_loader, discretizer,
                                                         normalizer, hype_space['batch_size'], shuffle=False, return_names=True)
            '''Model structure'''
            print('Constructing model ... ')
            device = torch.device("cuda" if torch.cuda.is_available() == True else 'cpu')
            print("available device: {}".format(device))

            model = AdaCare(**hype_space['arch']).to(device)

            optimizer = init_obj(hype_space, 'optimizer', torch.optim, model.parameters())

            if hype_space['lr_scheduler']['type'] == 'GradualWarmupScheduler':
                params = hype_space["lr_scheduler"]["args"]
                scheduler_steplr_args = dict(params["after_scheduler"]["args"])
                scheduler_steplr = getattr(torch.optim.lr_scheduler, params["after_scheduler"]["type"])(optimizer,
                                                                                                        **scheduler_steplr_args)
                lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=params["multiplier"],
                                                      total_epoch=params["total_epoch"],
                                                      after_scheduler=scheduler_steplr)
            else:
                lr_scheduler = init_obj(hype_space, 'lr_scheduler', torch.optim.lr_scheduler, optimizer)

            '''Train phase'''
            print('Start training ... ')

            train_loss = []
            val_loss = []
            batch_loss = []
            max_auprc = 0


            for each_chunk in range(hype_space['epochs']):
                cur_batch_loss = []
                model.train()
                for each_batch in range(train_data_gen.steps):
                    batch_data = next(train_data_gen)
                    batch_name = batch_data['names']
                    batch_data = batch_data['data']

                    batch_x = torch.tensor(batch_data[0][0], dtype=torch.float32).to(device)
                    batch_mask = torch.tensor(batch_data[0][1], dtype=torch.float32).unsqueeze(-1).to(device)
                    batch_y = torch.tensor(batch_data[1], dtype=torch.float32).to(device)

                    if batch_mask.size()[1] > 400:
                        batch_x = batch_x[:, :400, :]
                        batch_mask = batch_mask[:, :400, :]
                        batch_y = batch_y[:, :400, :]

                    optimizer.zero_grad()
                    cur_output, _ = model(batch_x, device)
                    masked_output = cur_output * batch_mask
                    loss = 0.7 * batch_y * torch.log(masked_output + 1e-7) + 0.3 * (1 - batch_y) * torch.log(
                        1 - masked_output + 1e-7)
                    # loss = 0.7 * torch.pow(batch_y,1.5) * torch.log(masked_output + 1e-7) + 0.3 * torch.pow((1-batch_y),1.5) * torch.log(1 - masked_output + 1e-7)
                    loss = torch.sum(loss, dim=1) / torch.sum(batch_mask, dim=1)
                    loss = torch.neg(torch.sum(loss))
                    cur_batch_loss.append(loss.cpu().detach().numpy())

                    loss.backward()
                    optimizer.step()

                    if each_batch % 50 == 0:
                        print('Chunk %d, Batch %d: Loss = %.4f' % (each_chunk, each_batch, cur_batch_loss[-1]))

                batch_loss.append(cur_batch_loss)
                train_loss.append(np.mean(np.array(cur_batch_loss)))
                log_writer.add_scalar('train_loss', np.mean(np.array(cur_batch_loss)), each_chunk)

                print("\n==>Predicting on validation")
                with torch.no_grad():
                    model.eval()
                    cur_val_loss = []
                    valid_true = []
                    valid_pred = []
                    for each_batch in range(val_data_gen.steps):
                        valid_data = next(val_data_gen)
                        valid_name = valid_data['names']
                        valid_data = valid_data['data']

                        valid_x = torch.tensor(valid_data[0][0], dtype=torch.float32).to(device)
                        valid_mask = torch.tensor(valid_data[0][1], dtype=torch.float32).unsqueeze(-1).to(device)
                        valid_y = torch.tensor(valid_data[1], dtype=torch.float32).to(device)

                        if valid_mask.size()[1] > 400:
                            valid_x = valid_x[:, :400, :]
                            valid_mask = valid_mask[:, :400, :]
                            valid_y = valid_y[:, :400, :]

                        valid_output, valid_dis = model(valid_x, device)
                        flops, params = profile(model, inputs=(valid_x, device))
                        print('--------------------------parameters & flops---------------------------')
                        print('flops: ', flops, ' params: ', params)
                        masked_valid_output = valid_output * valid_mask

                        valid_loss = valid_y * torch.log(masked_valid_output + 1e-7) + (1 - valid_y) * torch.log(
                            1 - masked_valid_output + 1e-7)
                        valid_loss = torch.sum(valid_loss, dim=1) / torch.sum(valid_mask, dim=1)
                        valid_loss = torch.neg(torch.sum(valid_loss))
                        cur_val_loss.append(valid_loss.cpu().detach().numpy())

                        for m, t, p in zip(valid_mask.cpu().numpy().flatten(), valid_y.cpu().numpy().flatten(),
                                           valid_output.cpu().detach().numpy().flatten()):
                            if np.equal(m, 1):
                                valid_true.append(t)
                                valid_pred.append(p)

                    val_loss.append(np.mean(np.array(cur_val_loss)))
                    print('Valid loss = %.4f' % (val_loss[-1]))
                    print('\n')
                    valid_pred = np.array(valid_pred)
                    valid_pred = np.stack([1 - valid_pred, valid_pred], axis=1)
                    ret = metrics.print_metrics_binary(valid_true, valid_pred)

                    log_writer.add_scalar('val_loss', val_loss[-1], each_chunk)
                    log_writer.add_scalar('val_acc', ret['acc'], each_chunk)
                    log_writer.add_scalar('val_prec0', ret['prec0'], each_chunk)
                    log_writer.add_scalar('val_prec1', ret['prec1'], each_chunk)
                    log_writer.add_scalar('val_rec0', ret['rec0'], each_chunk)
                    log_writer.add_scalar('val_rec1', ret['rec1'], each_chunk)
                    log_writer.add_scalar('val_auroc', ret['auroc'], each_chunk)
                    log_writer.add_scalar('val_auprc', ret['auprc'], each_chunk)
                    log_writer.add_scalar('val_minpse', ret['minpse'], each_chunk)
                    print()

                    cur_auprc = ret['auprc']
                    if cur_auprc > max_auprc:
                        max_auprc = cur_auprc
                        state = {
                            'net': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'chunk': each_chunk
                        }
                        torch.save(state, file_name)
                        print('\n------------ Save best model ------------\n')

            '''Parameters phase'''
            print('Parameters...')
            # Find total parameters and trainable parameters
            total_params = sum(p.numel() for p in model.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')

            '''Evaluate phase'''
            print('Testing model ... ')

            checkpoint = torch.load(file_name)
            save_chunk = checkpoint['chunk']
            print("last saved model is in chunk {}".format(save_chunk))
            model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.eval()

            test_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(data_path, 'test'),
                                                                      listfile=os.path.join(data_path, 'test',
                                                                                            'listfile.csv'),
                                                                      small_part=small_part)
            test_data_gen = utils.BatchGenDeepSupervision(test_data_loader, discretizer,
                                                          normalizer, hype_space['batch_size'],
                                                          shuffle=False, return_names=True)

            with torch.no_grad():
                torch.manual_seed(RANDOM_SEED)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(RANDOM_SEED)

                cur_test_loss = []
                test_true = []
                test_pred = []

                for each_batch in range(test_data_gen.steps):
                    test_data = next(test_data_gen)
                    test_name = test_data['names']
                    test_data = test_data['data']

                    test_x = torch.tensor(test_data[0][0], dtype=torch.float32).to(device)
                    test_mask = torch.tensor(test_data[0][1], dtype=torch.float32).unsqueeze(-1).to(device)
                    test_y = torch.tensor(test_data[1], dtype=torch.float32).to(device)

                    if test_mask.size()[1] > 400:
                        test_x = test_x[:, :400, :]
                        test_mask = test_mask[:, :400, :]
                        test_y = test_y[:, :400, :]

                    test_output, test_dis = model(test_x, device)
                    masked_test_output = test_output * test_mask

                    test_loss = test_y * torch.log(masked_test_output + 1e-7) + (1 - test_y) * torch.log(
                        1 - masked_test_output + 1e-7)
                    test_loss = torch.sum(test_loss, dim=1) / torch.sum(test_mask, dim=1)
                    test_loss = torch.neg(torch.sum(test_loss))
                    cur_test_loss.append(test_loss.cpu().detach().numpy())

                    for m, t, p in zip(test_mask.cpu().numpy().flatten(), test_y.cpu().numpy().flatten(),
                                       test_output.cpu().detach().numpy().flatten()):
                        if np.equal(m, 1):
                            test_true.append(t)
                            test_pred.append(p)

                print('Test loss = %.4f' % (np.mean(np.array(cur_test_loss))))
                print('\n')
                test_pred = np.array(test_pred)
                test_pred = np.stack([1 - test_pred, test_pred], axis=1)
                test_ret = metrics.print_metrics_binary(test_true, test_pred)

                with open(result, "a") as f:
                    f.write('split_' + str(split) + "\n")
                    f.write("accuracy = {}".format(test_ret['acc']))
                    f.write("precision class 0 = {}".format(test_ret['prec0']))
                    f.write("precision class 1 = {}".format(test_ret['prec1']))
                    f.write("recall class 0 = {}".format(test_ret['rec0']))
                    f.write("recall class 1 = {}".format(test_ret['rec1']))
                    f.write("AUC of ROC = {}".format(test_ret['auroc']))
                    f.write("AUC of PRC = {}".format(test_ret['auprc']))
                    f.write("min(+P, Se) = {}".format(test_ret['minpse']))

                loss.append(np.mean(np.array(cur_test_loss)))
                acc.append(test_ret['acc'])
                prec0.append(test_ret['prec0'])
                prec1.append(test_ret['prec1'])
                rec0.append(test_ret['rec0'])
                rec1.append(test_ret['rec1'])
                auroc.append(test_ret['auroc'])
                auprc.append(test_ret['auprc'])
                minpse.append(test_ret['minpse'])

        with open(result, "a") as f:
            f.write('AVG' + "\n")
            f.write("accuracy = {}".format(np.mean(acc)))
            f.write("precision class 0 = {}".format(np.mean(prec0)))
            f.write("precision class 1 = {}".format(np.mean(prec1)))
            f.write("recall class 0 = {}".format(np.mean(rec0)))
            f.write("recall class 1 = {}".format(np.mean(rec1)))
            f.write("AUC of ROC = {}".format(np.mean(auroc)))
            f.write("AUC of PRC = {}".format(np.mean(auprc)))
            f.write("min(+P, Se) = {}".format(np.mean(minpse)))

    except:
        return 0

    return - np.mean(auprc)


def run_trials():
    """Run one TPE meta optimisation step and save its results."""
    max_evals = nb_evals = 15

    logger = get_logger(save_path + '/trials.log', name='trials')

    logger.info("Attempt to resume a past training if it exists:")

    try:
        # https://github.com/hyperopt/hyperopt/issues/267
        trials = pickle.load(open(save_path + "/results.pkl", "rb"))
        logger.info("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        logger.info("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))
    except:
        trials = Trials()
        logger.info("Starting from scratch: new trials.")

    best = fmin(
        opt,
        space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals,
    )

    logger.info("Best: {}".format(best))
    pickle.dump(trials, open(save_path + "/results.pkl", "wb"))
    logger.info("\nOPTIMIZATION STEP COMPLETE.\n")
    logger.info("Trials:")

    for trial in trials:
        logger.info(trial)


if __name__ == "__main__":
    try:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        run_trials()
    except Exception as err:
        err_str = str(err)
        print(err_str)
        # traceback_str = str(traceback.format_exc())
        # print(traceback_str)
