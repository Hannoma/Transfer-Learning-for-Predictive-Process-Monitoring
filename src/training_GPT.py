import torch
import torch.nn as nn
import numpy as np
import datetime
import socket
import json
import argparse

from definitions import CONFIG_DIR, LOGS_DIR
from src import data_preprocessing, models
import random
import os
import math
import copy
from fast_transformers.masking import TriangularCausalMask


# Training loop for transformer models:
def iterate_over_traces(log_with_traces,
                        model=None,
                        device=None,
                        categorical_criterion=None,
                        regression_criterion=None,
                        subset=None,
                        optimizer=None,
                        lagrange_a=None,
                        causal_mask=None):

    summa_categorical_loss = 0.0
    summa_regression_loss = 0.0
    steps = 0

    data_loader = log_with_traces[subset + '_torch_data_loaders']

    for mini_batch in iter(data_loader):
        if subset == 'training':
            optimizer.zero_grad()

        if device == 'GPU':
            if model.architecture == 'GPT':
                a_s_i = mini_batch[0].cuda()
                t_s_i = mini_batch[1].cuda()
            a_s_t = mini_batch[2].cuda()
            t_s_t = mini_batch[3].cuda()
        else:
            if model.architecture == 'GPT':
                a_s_i = mini_batch[0]
                t_s_i = mini_batch[1]
            a_s_t = mini_batch[2]
            t_s_t = mini_batch[3]

        if model.architecture == 'GPT':
            prediction = model(x=(a_s_i, t_s_i), attn_mask=causal_mask)
        elif model.architecture == 'BERT':
            # the variable categorical target is an unsqueezed long so it has to be transformed into an unsqueezed float input:
            prediction = model(x=(a_s_t.unsqueeze(2).float(), t_s_t), attn_mask=None)

        categorical_criterion.reduction = 'mean'
        if model.architecture == 'GPT':
            categorical_loss = categorical_criterion(prediction[0].transpose(2, 1), a_s_t)
        # Loss is calculated only for the masked positions:
        elif model.architecture == 'BERT':
            categorical_loss = categorical_criterion(prediction[0][:, model.mlm.masked_indexes, :].transpose(2, 1), a_s_t[:, model.mlm.masked_indexes])

        # If time attribute and time prediction present:
        if len(prediction) > 1:
            regression_criterion.reduction = 'mean'
            if model.architecture == 'GPT':
                regression_loss = regression_criterion(prediction[1], t_s_t)
            elif model.architecture == 'BERT':
                regression_loss = regression_criterion(prediction[1][:, model.mlm.masked_indexes, :], t_s_t[:, model.mlm.masked_indexes, :])

            if subset == 'training':
                (categorical_loss + lagrange_a * regression_loss).backward()

            summa_categorical_loss += categorical_loss
            summa_regression_loss += regression_loss
        else:
            if subset == 'training':
                categorical_loss.backward()

            summa_categorical_loss += categorical_loss

        steps += 1

        if subset == 'training':
            optimizer.step()

    if len(prediction) > 1:
        return summa_categorical_loss.item() / steps, summa_regression_loss.item() / steps
    else:
        return (summa_categorical_loss.item() / steps, )

def main(args, dt_object, pre_train=True, combi=[], layers=[]):
    if not args.random:
        # RANDOM SEEDs:
        random_seed = args.random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        rng = np.random.default_rng(seed=random_seed)
        torch.backends.cudnn.deterministic = True
        random.seed(a=args.random_seed)
    if not pre_train:
        args.nb_epoch = 200
    
    with open(os.path.join(CONFIG_DIR, 'logs_meta.json')) as f:
        logs_meta = json.load(f)
    
    # data_preprocessing.download_logs(logs_meta, logs_dir)
    
    distributions, logs = data_preprocessing.create_distributions(LOGS_DIR)
   

    ####################################################################################
    #Data set 1
    ###################################################################################
    if not pre_train:
        log_name = combi[0]
        if args.device == 'GPU':
            print('total GPU memory: ' + str(torch.cuda.get_device_properties(device=args.gpu).total_memory))
            print('allocated GPU memory: ' + str(torch.cuda.memory_allocated(device=args.gpu)))

        processed_log = data_preprocessing.create_structured_log(logs[log_name], log_name=log_name)
        path = os.path.join('results', str(args.architecture), str(processed_log['id']))
        if not os.path.exists(path): os.makedirs(path)

        vars(args)['dataset'] = str(processed_log['id'])
        
        if os.path.isdir(os.path.join('split_logs', log_name)):
            for file_name in sorted(os.listdir(os.path.join('split_logs', log_name))):
                if file_name.startswith('split_log_'):
                    split_log_file_name = os.path.join('split_logs', log_name, file_name)
                    with open(split_log_file_name) as f_in:
                        split_log = json.load(f_in)
                    print(split_log_file_name + ' is used as common data')
            del processed_log
        else:
            split_log = data_preprocessing.create_split_log(processed_log, validation_ratio=args.validation_split)

        with open(os.path.join(path, 'split_log_' + dt_object.strftime("%Y%m%d%H%M") + '.json'), 'w') as f:
            json.dump(split_log, f)

        augmented_log = data_preprocessing.create_transformer_augmentation(split_log,
                                                                           pad_token=args.pad_token,
                                                                           training_batch_size=args.training_batch_size,
                                                                           validation_batch_size=args.validation_batch_size)

        # [EOS], [SOS], [PAD] and [MASK]
        nb_special_tokens = 4
        attributes_meta = {0: {'nb_special_tokens': nb_special_tokens, 'vocabulary_size': augmented_log['vocabulary_size']},
                        1: {'min_value': 0.0, 'max_value': 1.0}}

        # [MASK] special token is about to be the largest integer of our vocabulary:
        mask_token = max(augmented_log['sos_token'], augmented_log['eos_token']) + 1
        vars(args)['mask_token'] = mask_token
        vars(args)['sos_token'] = augmented_log['sos_token']
        vars(args)['eos_token'] = augmented_log['eos_token']
        vars(args)['nb_special_tokens'] = nb_special_tokens
        vars(args)['vocabulary_size'] = augmented_log['vocabulary_size']
        vars(args)['longest_trace_length'] = augmented_log['longest_trace_length']

        # All traces are longer by one position due to the closing [EOS]:
        max_length = augmented_log['longest_trace_length'] + 1

        vars(args)['max_length'] = max_length

        # Fix mask creation (15%) for BERT pseudo-pseudo-LL validation:
        amount_of_fix_masked_positions = max(1, math.floor(max_length / 7.5))
        nb_unmasked_positions_between_masked_ones = math.ceil(
            max_length / (amount_of_fix_masked_positions + 1))
        fix_masks = []
        for i in range(amount_of_fix_masked_positions):
            index_to_append = nb_unmasked_positions_between_masked_ones + i * nb_unmasked_positions_between_masked_ones
            if index_to_append < max_length: fix_masks.append(index_to_append)
        vars(args)['fix_masks'] = fix_masks

        with open(os.path.join(path, 'experiment_parameters.json'), 'a') as fp:
            json.dump(vars(args), fp)
            fp.write('\n')

        ######################################################################################
        #Build pretrained model
        #####################################################################################

        model = models.Transformer(d_model=args.hidden_dim,
                                   sequence_length=max_length,
                                   n_layers=args.n_layers,
                                   n_heads=args.n_heads,
                                   d_query=int(args.hidden_dim / args.n_heads),
                                   dropout_prob=args.dropout_prob,
                                   attention_dropout_prob=args.dropout_prob,
                                   mlm_prob=args.mlm_masking_prob,
                                   vocab_size=attributes_meta[0]['vocabulary_size'],
                                   pad_token=args.pad_token,
                                   mask_token=mask_token,
                                   sos_token=augmented_log['sos_token'],
                                   eos_token=augmented_log['eos_token'],
                                   architecture=args.architecture,
                                   attributes_meta=attributes_meta,
                                   time_attribute_concatenated=args.time_attribute_concatenated,
                                   nb_special_tokens=attributes_meta[0]['nb_special_tokens']).to(device=args.gpu)

        checkpoint = torch.load(f'./results/GPT/{combi[0]}/checkpoints/model-{combi[0]}.pt')
        model.load_state_dict(checkpoint['model_state_dict'])

    #######################################################################################
    #Dataset 2
    ######################################################################################
    model_log = combi[-1]

    new_processed_log = data_preprocessing.create_structured_log(logs[model_log], log_name=model_log)

    path = os.path.join('results', str(args.architecture), str(new_processed_log['id']))
    if not os.path.exists(path): os.makedirs(path)

    vars(args)['dataset'] = str(new_processed_log['id'])

    if os.path.isdir(os.path.join('split_logs', model_log)):
        for file_name in sorted(os.listdir(os.path.join('split_logs', model_log))):
            if file_name.startswith('split_log_'):
                split_log_file_name = os.path.join('split_logs', model_log, file_name)
                with open(split_log_file_name) as f_in:
                    split_log = json.load(f_in)
                print(split_log_file_name + ' is used as common data')
        del processed_log
    else:
        new_split_log = data_preprocessing.create_split_log(new_processed_log, validation_ratio=args.validation_split)

    with open(os.path.join(path, 'split_log_' + dt_object.strftime("%Y%m%d%H%M") + '.json'), 'w') as f:
        json.dump(new_split_log, f)

    new_augmented_log = data_preprocessing.create_transformer_augmentation(new_split_log,
                                                                           pad_token=args.pad_token,
                                                                           training_batch_size=args.training_batch_size,
                                                                           validation_batch_size=args.validation_batch_size)
    # [EOS], [SOS], [PAD] and [MASK]
    nb_special_tokens = 4
    attributes_meta = {0: {'nb_special_tokens': nb_special_tokens, 'vocabulary_size': new_augmented_log['vocabulary_size']},
                    1: {'min_value': 0.0, 'max_value': 1.0}}

    # [MASK] special token is about to be the largest integer of our vocabulary:
    mask_token = max(new_augmented_log['sos_token'], new_augmented_log['eos_token']) + 1
    vars(args)['mask_token'] = mask_token
    vars(args)['sos_token'] = new_augmented_log['sos_token']
    vars(args)['eos_token'] = new_augmented_log['eos_token']
    vars(args)['nb_special_tokens'] = nb_special_tokens
    vars(args)['vocabulary_size'] = new_augmented_log['vocabulary_size']
    vars(args)['longest_trace_length'] = new_augmented_log['longest_trace_length']

    # All traces are longer by one position due to the closing [EOS]:
    new_max_length = new_augmented_log['longest_trace_length'] + 1
    vars(args)['new_max_length'] = new_max_length

    with open(os.path.join(path, 'experiment_parameters.json'), 'a') as fp:
        json.dump(vars(args), fp)
        fp.write('\n')


    ################################################################################################
    # New model:
    ###################################################################################################
    i = 0
    for layer in layers:
        i += 1
        new_model = models.Transformer(d_model=args.hidden_dim,
                                       sequence_length=new_max_length,
                                       n_layers=args.n_layers,
                                       n_heads=args.n_heads,
                                       d_query=int(args.hidden_dim / args.n_heads),
                                       dropout_prob=args.dropout_prob,
                                       attention_dropout_prob=args.dropout_prob,
                                       mlm_prob=args.mlm_masking_prob,
                                       vocab_size=attributes_meta[0]['vocabulary_size'],
                                       pad_token=args.pad_token,
                                       mask_token=mask_token,
                                       sos_token=new_augmented_log['sos_token'],
                                       eos_token=new_augmented_log['eos_token'],
                                       architecture=args.architecture,
                                       attributes_meta=attributes_meta,
                                       time_attribute_concatenated=args.time_attribute_concatenated,
                                       nb_special_tokens=attributes_meta[0]['nb_special_tokens']).to(device=args.gpu)

        # The will be only one instance of casual mask which is instanciated once here outside of all loops:
        if args.architecture == 'GPT':
            causal_mask = TriangularCausalMask(new_max_length, device=next(new_model.parameters()).device)
        else:
            causal_mask = None

        categorical_criterion = nn.CrossEntropyLoss()
        regression_criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(new_model.parameters(),
                                        lr=args.training_learning_rate,
                                        weight_decay=args.training_gaussian_process)

        training_log_filename = "training_figures_" + dt_object.strftime("%Y%m%d%H%M") + ".csv"
        
        
        if args.architecture == 'BERT':
            with open(os.path.join(path, training_log_filename), "a") as myfile:
                myfile.write(
                    'datetime'
                    ',epoch'
                    ',training_loss_activity'
                    ',training_loss_time,training_loss'
                    ',validation_loss_activity'
                    ',validation_loss_time'
                    ',validation_loss'
                    ',elapsed_seconds'
                    #',masked_positions'
                    ',validation_loss_activity_fix_masks'
                    ',validation_loss_time_fix_masks'
                    ',validation_loss_fix_masks'
                    '\n')
        else:
            with open(os.path.join(path, training_log_filename), "a") as myfile:
                myfile.write(
                    'datetime'
                    ',epoch'
                    ',training_loss_activity'
                    ',training_loss_time,training_loss'
                    ',validation_loss_activity'
                    ',validation_loss_time'
                    ',validation_loss'
                    ',elapsed_seconds'
                    '\n')

        # not saving all version of model:
        min_loss_threshold = args.save_criterion_threshold

        if not os.path.exists(os.path.join(path, 'checkpoints')): os.makedirs(os.path.join(path, 'checkpoints'))

        model_to_save = {}
        total_validation_losses = []
        total_validation_losses_fix_masks = []

        ####################################################################################################################
        #Copying the model
        ###################################################################################################################
        if not pre_train: 
            with torch.no_grad():
                for name, param in new_model.named_parameters():
                    if 'embedding' not in name:
                        if 'readout' not in name:
                            if max_length >= new_max_length:
                                param.copy_(model.state_dict()[name])
                            elif max_length < new_max_length:
                                param.copy_(model.state_dict()[name])

                                

        ######################################################################################################################
        #Training the model
        #####################################################################################################################
        
        for e in range(args.nb_epoch):
            if not e%10:
                print('training epoch ' + str(e) + '/' + str(args.nb_epoch) + ' of ' + str(new_augmented_log['id']))
            if not pre_train:
                for name, param in new_model.named_parameters():
                    for single in layer:
                        if f'layers.{single}.' in name:
                            param.requires_grad = False

            for name,param in new_model.named_parameters():
                print(param.requires_grad)

            new_model.train()

            if args.architecture == 'BERT':
                model.mlm.method = 'u-PMLM'

            dt_object_training_start = datetime.datetime.now()

            training_loss = iterate_over_traces(log_with_traces=new_augmented_log,
                                                model=new_model,
                                                device=args.device,
                                                categorical_criterion=categorical_criterion,
                                                regression_criterion=regression_criterion,
                                                subset='training',
                                                optimizer=optimizer,
                                                lagrange_a=args.lagrange_a,
                                                causal_mask=causal_mask)

            dt_object_training_end = datetime.datetime.now()

            training_duration = (dt_object_training_end - dt_object_training_start).total_seconds()
            new_model.eval()
            with torch.no_grad():
                validation_loss = iterate_over_traces(log_with_traces=new_augmented_log,
                                                        model=new_model,
                                                        device=args.device,
                                                        categorical_criterion=categorical_criterion,
                                                        regression_criterion=regression_criterion,
                                                        subset='validation',
                                                        causal_mask=causal_mask)

                if args.architecture == 'BERT':
                    model.mlm.method = 'fix_masks'
                    model.mlm.fix_masks = torch.tensor(fix_masks, device=next(model.parameters()).device)
                    validation_loss_fix_masks = iterate_over_traces(log_with_traces=augmented_log,
                                                                    model=model,
                                                                    device=args.device,
                                                                    categorical_criterion=categorical_criterion,
                                                                    regression_criterion=regression_criterion,
                                                                    subset='validation',
                                                                    causal_mask=causal_mask)
                else:
                    # an arbritary value:
                    validation_loss_fix_masks = (99, 99)
                    
            if len(validation_loss) > 1:
                total_validation_loss = validation_loss[0] + args.lagrange_a * validation_loss[1]
                total_training_loss = training_loss[0] + args.lagrange_a * training_loss[1]
                if args.architecture == 'BERT':
                    total_validation_loss_fix_masks = validation_loss_fix_masks[0] + args.lagrange_a * validation_loss_fix_masks[1]
                else:
                    # an arbritary value:
                    total_validation_loss_fix_masks = 99
                with open(os.path.join(path, training_log_filename), "a") as myfile:
                    myfile.write(dt_object.strftime("%Y%m%d%H%M")
                                    + ',' + str(e)
                                    + ',' + "{:.4f}".format(training_loss[0])
                                    + ',' + "{:.4f}".format(training_loss[1])
                                    + ',' + "{:.4f}".format(total_training_loss)
                                    + ',' + "{:.4f}".format(validation_loss[0])
                                    + ',' + "{:.4f}".format(validation_loss[1])
                                    + ',' + "{:.4f}".format(total_validation_loss)
                                    + ',' + "{:.3f}".format(training_duration)
                                    + ',' + "{:.4f}".format(validation_loss_fix_masks[0])
                                    + ',' + "{:.4f}".format(validation_loss_fix_masks[1])
                                    + ',' + "{:.4f}".format(total_validation_loss_fix_masks)
                                    + '\n')
                                    #+ ',' + str('|'.join(map(str, model.mlm.masked_indexes.tolist()))) + '\n')
            else:
                total_validation_loss = validation_loss[0]
                total_training_loss = training_loss[0]
                total_validation_loss_fix_masks = validation_loss_fix_masks[0]
                with open(os.path.join(path, training_log_filename), "a") as myfile:
                    myfile.write(dt_object.strftime("%Y%m%d%H%M")
                                    + ',' + str(e)
                                    + ',' + "{:.4f}".format(training_loss[0])
                                    + ',' + 'NA'
                                    + ',' + "{:.4f}".format(total_training_loss)
                                    + ',' + "{:.4f}".format(validation_loss[0])
                                    + ',' + 'NA'
                                    + ',' + "{:.4f}".format(total_validation_loss)
                                    + ',' + "{:.3f}".format(training_duration)
                                    + ',' + "{:.4f}".format(validation_loss_fix_masks[0])
                                    + ',' + 'NA'
                                    + ',' + "{:.4f}".format(total_validation_loss_fix_masks)
                                    + '\n')
                                    #+ ',' + str('|'.join(map(str, model.mlm.masked_indexes.tolist()))) + '\n')

            total_validation_losses.append(total_validation_loss)
            total_validation_losses_fix_masks.append(total_validation_loss_fix_masks)

            if args.early_stopping:
                early_stopping_var = 50
                if args.architecture == 'BERT':
                    if len(total_validation_losses_fix_masks) > early_stopping_var:
                        if np.all(np.array(total_validation_losses_fix_masks)[-(early_stopping_var-1):] > np.array(total_validation_losses_fix_masks)[-early_stopping_var]):
                            print("early stopping")
                            break
                else:
                    if len(total_validation_losses) > early_stopping_var:
                        if np.all(np.array(total_validation_losses)[-(early_stopping_var-1):] > np.array(total_validation_losses)[-early_stopping_var]):
                            print("early stopping")
                            break


            # if total_validation_loss < min_loss_threshold:
            model_to_save['model_state_dict'] = copy.deepcopy(new_model.state_dict())
            model_to_save['optimizer_state_dict'] = copy.deepcopy(optimizer.state_dict())
            model_to_save['loss'] = copy.deepcopy(total_validation_loss)
            model_to_save['epoch'] = copy.deepcopy(e)
            model_to_save['total_validation_loss'] = copy.deepcopy(total_validation_loss)
            min_loss_threshold = total_validation_loss

            if pre_train:
                checkpoint_name = 'model-' + f'{combi[0]}'+'.pt'
            elif not pre_train:
                checkpoint_name = f'model-{combi[0]}-{combi[1]}-[{i}]' + '.pt'
        
            new_path = os.path.join('results', 'GPT', combi[-1])
            if not os.path.exists(new_path): os.makedirs(new_path)

            torch.save({
                'model_state_dict': model_to_save['model_state_dict'],
                'optimizer_state_dict': model_to_save['optimizer_state_dict'],
                'loss': model_to_save['loss'],
            }, os.path.join(new_path, 'checkpoints', checkpoint_name))


if __name__ == '__main__':
    dt_object = datetime.datetime.now()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datetime', help='datetime', default=dt_object.strftime("%Y%m%d%H%M"), type=str)
    parser.add_argument('--hidden_dim', help='hidden state dimensions', default=128, type=int)
    parser.add_argument('--n_layers', help='number of layers', default=4, type=int)
    parser.add_argument('--n_heads', help='number of heads', default=4, type=int)
    parser.add_argument('--nb_epoch', help='training iterations', default=400, type=int)
    parser.add_argument('--training_batch_size', help='number of training samples in mini-batch', default=512, type=int)
    parser.add_argument('--validation_batch_size', help='number of validation samples in mini-batch', default=512, type=int)
    parser.add_argument('--training_mlm_method', help='training MLM method', default='BERT', type=str)
    parser.add_argument('--validation_mlm_method', help='validation MLM method', default='fix_masks', type=str) # we would like to end up with some non-stochastic & at least pseudo likelihood metric
    parser.add_argument('--mlm_masking_prob', help='mlm_masking_prob', default=0.15, type=float)
    parser.add_argument('--dropout_prob', help='dropout_prob', default=0.3, type=float)
    parser.add_argument('--training_learning_rate', help='GD learning rate', default=1e-4, type=float)
    parser.add_argument('--training_gaussian_process', help='GP', default=1e-5, type=float)
    parser.add_argument('--validation_split', help='validation_split', default=0.2, type=float)
    parser.add_argument('--dataset', help='dataset', default='', type=str)
    parser.add_argument('--random_seed', help='random_seed', default=1982, type=int)
    parser.add_argument('--random', help='if random', default=True, type=bool)
    parser.add_argument('--gpu', help='gpu', default=0, type=int)
    parser.add_argument('--validation_indexes', help='list of validation_indexes NO SPACES BETWEEN ITEMS!', default='[0,1,4,10,15]', type=str)
    parser.add_argument('--ground_truth_p', help='ground_truth_p', default=0.0, type=float)
    parser.add_argument('--architecture', help='BERT or GPT', default='GPT', type=str)
    parser.add_argument('--time_attribute_concatenated', help='time_attribute_concatenated', default=False, type=bool)
    parser.add_argument('--device', help='GPU or CPU', default='GPU', type=str)
    parser.add_argument('--lagrange_a', help='Langrange multiplier', default=1.0, type=float)
    parser.add_argument('--save_criterion_threshold', help='save_criterion_threshold', default=4.0, type=float)
    parser.add_argument('--pad_token', help='pad_token', default=0, type=int)
    parser.add_argument('--to_wrap_into_torch_dataset', help='to_wrap_into_torch_dataset', default=True, type=bool)
    parser.add_argument('--seq_ae_teacher_forcing_ratio', help='seq_ae_teacher_forcing_ratio', default=1.0, type=float)
    parser.add_argument('--early_stopping', help='early_stopping', default=True, type=bool)
    parser.add_argument('--single_position_target', help='single_position_target', default=False, type=bool)

    args = parser.parse_args()
    
    vars(args)['hostname'] = str(socket.gethostname())
    
    print('This is training of: ' + dt_object.strftime("%Y%m%d%H%M"))
    
    if args.device == 'GPU':
        torch.cuda.set_device(args.gpu)
        print('This is training at gpu: ' + str(args.gpu))

    layers = [['0','1','2','3'], ['0','1','2'],
            ['1','2','3'], ['0','2','3'], 
            ['0','1','3'], ['0','1'],['0']]

    for log in ['BPI_Challenge_2013_closed_problems.xes.gz','BPI_Challenge_2012.xes.gz',
                'BPI_Challenge_2013_incidents.xes.gz','BPI_Challenge_2013_open_problems.xes.gz',
                'BPI%20Challenge%202017.xes.gz','BPIC15_1.xes',
                'Road_Traffic_Fine_Management_Process.xes.gz','Sepsis%20Cases%20-%20Event%20Log.xes.gz',
                'helpdesk.csv']:
        main(args, dt_object, combi=[log], layers=[[]])
        # for transfer in [#'BPI_Challenge_2013_closed_problems.xes.gz','BPI_Challenge_2012.xes.gz',
        #                  #'BPI_Challenge_2013_incidents.xes.gz','BPI_Challenge_2013_open_problems.xes.gz',
        #                  #'BPI%20Challenge%202017.xes.gz','BPIC15_1.xes',
        #                  #'Road_Traffic_Fine_Management_Process.xes.gz','Sepsis%20Cases%20-%20Event%20Log.xes.gz',
        #                  'helpdesk.csv']:
        #         if log != transfer:
        #             main(args, dt_object, pre_train=False, combi=[log, transfer], layers=layers)

