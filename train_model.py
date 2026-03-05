import argparse

import numpy as np
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
import time
from scipy.stats import mode
import model.ASAM as ARMBANDGNN_last


torch.manual_seed(0)
np.random.seed(0)
import warnings
warnings.filterwarnings(action='ignore')

total_loss = []
total_acc = []
valid_loss = []
valid_acc = []


def add_args(parser):
    parser.add_argument('--gpu', type=int, default=0, metavar='N', help='GPU index.')
    
    parser.add_argument('--epoch', type=int, default=150, metavar='N',
                        help='number of training')
    parser.add_argument('--epoch_pre', type=int, default=150, metavar='N',
                    help='number of training')
    parser.add_argument('--dataset', default='skku', metavar='N', help='dataset type: skku, nina, nina_18')


    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                        help='learning rate')

    parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                        help='insert batch size for training(default 128)')
    parser.add_argument('--feature_dim', type=int, default=100, metavar='N',
                        help='feature dimension(default 128)')
    parser.add_argument('--precision', type=float, default=1e-6, metavar='N',
                        help='reducing learning rate when a metric has stopped improving(default = 0.0000001')

    parser.add_argument('--channel',default='[24, 16, 8, 4]',metavar='N', help=' 3 channel')

    parser.add_argument('--dropout', type=float, default=0.2, metavar='N',
                        help='probability of elements to be zero')
    parser.add_argument('--type', type=int, default=2, metavar='N',
                        help='0: GNN, 1: concat version, 2: GAT version')
    parser.add_argument('--cand_num', type=int, default=4, metavar='N',
                        help='number of candidates for each dataset, 10, 36, 17')
    parser.add_argument('--load_data', default='./utils/saved_model/4th_sep4111.pt', metavar='N',
                        help='saved model name(no duplicate)')
    parser.add_argument('--num_label', type=int, default= 18, metavar='N',
                        help = 'numbe of label')
    parser.add_argument('--channel_electrode', type=int, default=8, metavar='N')
    parser.add_argument('--l1', default = 1e-1, type=float, help = 'spatial hyperparameter')
    parser.add_argument('--l2', default = 1e-1, type=float, help='temporal hyperparamter')
    parser.add_argument('--l3', default=0.3, type=float, help='invariance hyperparamter')
    parser.add_argument('--l4', default=1e-2, type=float, help='DA spatial hyperparameter')
    parser.add_argument('--print', default=0, help='print the loss')
    parser.add_argument('--eeg_data_divider', default=10, help='the number of patients in the first phase the last 1~14 (the last 5 patients are used in the inference)')

    args = parser.parse_args()

    return args


def scramble(examples, labels):
    random_vec = np.arange(len(labels))
    np.random.shuffle(random_vec)

    new_labels, new_examples = [], []
    for i in random_vec:
        new_labels.append(labels[i])
        new_examples.append(examples[i])

    return new_examples, new_labels

##
def gaussian(u, sigma=3.0):
    if abs(u) <= 1:
        return np.exp(-u**2 / (2 * sigma**2))
    else:
        return 0
    

def biweight_kernel(u):
    if abs(u) <= 1:
        return (15/16) * (1 - u**2)**2
    else:
        return 0

def triweight_kernel(u):
    if abs(u) <= 1:
        return (35/32) * (1 - u**2)**3
    else:
        return 0
   
def tricube(u):
    if abs(u) <= 1:
        return (70/81) * (1 - u**3)**3
    else:
        return 0

def epanechnikov(u):
    if abs(u) <= 1:
        return (3/4) * (1 - u**2)
    else:
        return 0


def create_diagonal_tensor(area_ratio, kernel_f, size, device):
    # Initialize a tensor with zeros
    tensor = torch.zeros(size, size, device=device)
    area_size = int(area_ratio*size)
    area = torch.zeros(area_size, area_size, device=device)

    for i in range(area_size):
        for j in range(area_size):
            if(area_size ==1):
                distance = abs(i-j)
            else:
                distance = abs(i - j)/(area_size -1)
            area[i, j] = kernel_f(distance, sigma=2)
                
    for i in range(size - area_size+1):
        tensor[i:i+area_size, i:i+area_size] = area        
    return tensor


def train_basic(model, criterion, optimizer, scheduler, dataloaders, num_epochs, precision):
    since = time.time()
    best_loss = float('inf')
    patience = 60
    patience_increase = 20
    hundred = False

    for epoch in range(num_epochs):
        epoch_start = time.time()
        if args.print:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10, "^ㅁ^bbbbb")

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.
            running_corrects = 0
            total = 0
            for i, data in enumerate(dataloaders[phase]):
                inputs, labels = data
                # inputs = torch.nn.functional.normalize(inputs)
                inputs, labels = Variable(inputs), Variable(labels)
                inputs = inputs.cuda(args.gpu)
                labels = labels.cuda(args.gpu)
                optimizer.zero_grad()

                if phase == 'train':
                    model.train()
                    z_s1, z_t1, outputs, z_s1_pos = model(inputs) # forward
                    

                    z_s1 = z_s1_pos

                    # loss1(cca-ssg)
                    c_s1 = torch.bmm(z_s1, z_s1.transpose(1, 2))  # batch matrix multiplication
                    c_t1 = torch.bmm(z_t1, z_t1.transpose(1, 2))

                    N_s = z_s1.size()[2]
                    N_t = z_t1.size()[2]
                    c_s1 = (c_s1 / N_s)

                    c_t1 = (c_t1 / N_t)
                    
                    iden_s1 = torch.eye(c_s1.shape[1], device=z_s1.device).unsqueeze(0)

                    sym_t1 = create_diagonal_tensor(1, gaussian, c_t1.shape[1], z_s1.device).unsqueeze(0).to(z_s1.device)

                    loss_dec_s1 = (iden_s1 - c_s1).pow(2).sum(dim=(1, 2))
                    loss_reg_t1 = (sym_t1 - c_t1).pow(2).sum(dim=(1, 2))

                    lambd_t = args.l1#1e-1 # trade-off ratio lambda. default set is 1e-3.
                    lambd_s = args.l2#1e-3

                    loss1 = (lambd_s * loss_dec_s1 + lambd_t * loss_reg_t1).mean()
                    
                    _, predictions = torch.max(outputs.data, 1)
                    labels = labels.long()
                    loss2 = criterion(outputs, labels)
                    
                    # loss = 0.3*loss1 + loss2
                    loss = 0.3 * loss1 + loss2
                    # print(loss1, loss2)
                    #loss = loss2
                    loss.backward()
                    optimizer.step()
                else:# phase == 'val'
                    model.eval()
                    with torch.no_grad():
                        z_s1, z_t1, outputs, z_s1_pos = model(inputs) # forward
                        z_s1 = z_s1.clone().detach()
                        z_t1 = z_t1.clone().detach()
                        
                        # loss1(cca-ssg)
                        c_s1 = torch.bmm(z_s1, z_s1.transpose(1, 2))  # batch matrix multiplication
                        c_t1 = torch.bmm(z_t1, z_t1.transpose(1, 2))

                        N_s = z_s1.size()[2]
                        N_t = z_t1.size()[2]
                        c_s1 = (c_s1 / N_s)#.pow(3)
                        c_t1 = (c_t1 / N_t)#.pow(3)
                        iden_s1 = torch.eye(c_s1.shape[1]).unsqueeze(0).to(z_s1.device)

                        sym_t1 = create_diagonal_tensor(1, gaussian, c_t1.shape[1], z_s1.device).unsqueeze(0).to(z_s1.device)

                        loss_dec_s1 = (iden_s1 - c_s1).pow(2).sum(dim=(1, 2))
                        loss_reg_t1 = (sym_t1 - c_t1).pow(2).sum(dim=(1, 2))

                        lambd_t = args.l1#1e-1  # trade-off ratio lambda. default set is 1e-3.
                        lambd_s = args.l2#1e-3

                        loss1 = (lambd_s * loss_dec_s1 + lambd_t * loss_reg_t1).mean()

                        accumulated_predicted = Variable(torch.zeros(len(inputs), 53))
                        loss_intermediary = 0.
                        total_sub_pass = 0
                        labels = labels.long().cpu()
                        loss2 = criterion(outputs.cpu(), labels)
                        # loss = 0.3 * loss1 + loss2
                        loss = 0.3 * loss1 + loss2
                        #loss = loss2
                        
                        if loss_intermediary == 0.:
                            loss_intermediary = loss.item()
                        else:
                            loss_intermediary += loss.item()
                        _, prediction_from_this_sub_network = torch.max(outputs.data, 1)
                        accumulated_predicted[range(len(inputs)),
                                                prediction_from_this_sub_network.cpu().numpy().tolist()] += 1
                        total_sub_pass += 1
                        _, predictions = torch.max(accumulated_predicted.data, 1)
                        loss = loss_intermediary / total_sub_pass

                running_loss += loss
                running_corrects += torch.sum(predictions == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            if phase == 'val':
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            else:
                total_loss.append(epoch_loss)
                total_acc.append(epoch_acc)
            if args.print:
                print('{} Loss: {:.8f} Acc: {:.8}'.format(
                    phase, epoch_loss, epoch_acc))

            # earlystopping
            if phase == 'val': #'val':  #TODO changed to train
                scheduler.step(epoch_loss)
                if epoch_loss + precision < best_loss:
                    if args.print:
                        print("New best validation loss:", epoch_loss)
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), args.load_data)
                    patience = patience_increase + epoch
                if epoch_acc == 1:
                    if args.print:
                        print("stopped because of 100%")
                    hundred = True

        if args.print:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - epoch_start))
        if epoch > patience :
            break
    
    
    time_elapsed = time.time() - since
    if args.print:
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val loss: {:4f}'.format(best_loss))
    model_weights = torch.load(args.load_data)
    model.load_state_dict(model_weights)
    model.eval()
    return model, num_epochs



def adapt_da_inv(model, criterion, optimizer, scheduler, dataloaders, num_epochs, precision):
    since = time.time()
    best_loss = float('inf')
    patience = 60
    patience_increase = 20
    hundred = False

    for epoch in range(num_epochs):
        epoch_start = time.time()
        if args.print:
            print('Domain Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10, "^ㅁ^bbbbb")

        # Each epoch has a training and validation phase
        torch.cuda.empty_cache()
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.
            running_corrects = 0
            total = 0
            for i, data in enumerate(dataloaders[phase]):
                inputs, labels = data
                # inputs = torch.nn.functional.normalize(inputs)
                inputs, labels = Variable(inputs), Variable(labels)

                inputs[:, :4] = inputs[:, torch.randperm(4)] # inputs.size(1)

                inputs = inputs.cuda(args.gpu)
                labels = labels.cuda(args.gpu)
                optimizer.zero_grad()

                if phase == 'train':
                    model.train()
                    inv_1, inv_2, outputs, inv_pos_emb = model(inputs) # forward
                    loss_inv = (inv_1 - inv_2).pow(2).sum(dim=(1, 2))
                    ## spatial decorrelation loss
                    z_s1 = inv_2.clone().detach()
                    inv_pos = inv_pos_emb.clone().detach()
                    # loss1(cca-ssg)
                    c_s1 = torch.bmm(z_s1, inv_pos.transpose(1, 2))  # batch matrix multiplication

                    N_s = z_s1.size()[2]
                    c_s1 = (c_s1 / N_s)  # .pow(3)
                    iden_s1 = torch.eye(c_s1.shape[1]).unsqueeze(0).to(z_s1.device)

                    loss_dec_s1 = (iden_s1 - c_s1).pow(2).sum(dim=(1, 2))

                    ######



                    # loss_inv = (inv_1 - inv_2).pow(2).sum(dim=(1, 2))
                    
                    loss1 = loss_inv#loss_inv.mean()
                    _, predictions = torch.max(outputs.data, 1)
                    labels = labels.long()
                    loss2 = criterion(outputs, labels)
                    # print(loss_dec_s1.shape, loss1.shape, loss_inv.shape, loss2.shape, loss_dec_s1.shape)
                    loss = args.l3*loss1.mean() + loss2 + args.l4 * loss_dec_s1.mean()
                    #loss = loss2
                    loss.backward()
                    optimizer.step()
                else:# phase == 'val'
                    model.eval()
                    with torch.no_grad():
                        inv_1, inv_2, outputs, inv_pos_emb = model(inputs) # forward
                        inv_1 = inv_1.clone().detach()
                        inv_2 = inv_2.clone().detach()
                        
                        loss_inv = (inv_1 - inv_2).pow(2).sum(dim=(1, 2))
                    
                        loss1 = loss_inv.mean()

                        z_s1 = inv_2.clone().detach()
                        inv_pos = inv_pos_emb.clone().detach()
                        # loss1(cca-ssg)
                        c_s1 = torch.bmm(z_s1, inv_pos.transpose(1, 2))  # batch matrix multiplication

                        N_s = z_s1.size()[2]
                        c_s1 = (c_s1 / N_s)  # .pow(3)
                        iden_s1 = torch.eye(c_s1.shape[1]).unsqueeze(0).to(z_s1.device)

                        loss_dec_s1 = (iden_s1 - c_s1).pow(2).sum(dim=(1, 2))

                        accumulated_predicted = Variable(torch.zeros(len(inputs), 53))
                        loss_intermediary = 0.
                        total_sub_pass = 0
                        labels = labels.long().cpu()
                        loss2 = criterion(outputs.cpu(), labels)
                        loss = args.l3*loss1.mean() + loss2 + args.l4 * loss_dec_s1.mean()
                        #loss = loss2
                        
                        if loss_intermediary == 0.:
                            loss_intermediary = loss.item()
                        else:
                            loss_intermediary += loss.item()
                        _, prediction_from_this_sub_network = torch.max(outputs.data, 1)
                        accumulated_predicted[range(len(inputs)),
                                                prediction_from_this_sub_network.cpu().numpy().tolist()] += 1
                        total_sub_pass += 1
                        _, predictions = torch.max(accumulated_predicted.data, 1)
                        loss = loss_intermediary / total_sub_pass

                running_loss += loss
                running_corrects += torch.sum(predictions == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            if phase == 'val':
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            else:
                total_loss.append(epoch_loss)
                total_acc.append(epoch_acc)
            if args.print:
                print('{} Loss: {:.8f} Acc: {:.8}'.format(
                    phase, epoch_loss, epoch_acc))

            # earlystopping
            if phase == 'val': 
                scheduler.step(epoch_loss)
                if epoch_loss + precision < best_loss:
                    if args.print:
                        print("New best validation loss:", epoch_loss)
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), args.load_data)
                    patience = patience_increase + epoch
                if epoch_acc == 1:
                    if args.print:
                        print("stopped because of 100%")
                    hundred = True
        if args.print:
            print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - epoch_start))
        if epoch > patience :
            break
    
    
    time_elapsed = time.time() - since
    if args.print:
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val loss: {:4f}'.format(best_loss))
    model_weights = torch.load(args.load_data)
    model.load_state_dict(model_weights)
    model.eval()
    return model, num_epochs

def fit_freeze(args, examples_training, labels_training):
    accuracy_test0, accuracy_test1 = [], []
    X_fine_tune_train, Y_fine_tune_train = [], []
    X_fine_tune_pretrain, Y_fine_tune_pretrain = [], []
    X_fine_tune_test, Y_fine_tune_test = [], []
    if(args.dataset == 'skku'):
        # a = torch.randperm(4)
        # a = a.tolist()
        # a += [4, 5, 6, 7]
        # for dataset_index in range(0, 1):
        #     for label_index in range(len(labels_training)):
        #         if label_index == dataset_index:
        #             for example_index in range(len(examples_training[label_index])):
        #                     X_fine_tune_train.extend(examples_training[label_index][example_index][:, a])
        #                     Y_fine_tune_train.extend(labels_training[label_index][example_index])
        for dataset_index in range(0, 2):
            for label_index in range(len(labels_training)):
                if label_index == dataset_index:
                    for example_index in range(len(examples_training[label_index])):
                            X_fine_tune_train.extend(examples_training[label_index][example_index])
                            Y_fine_tune_train.extend(labels_training[label_index][example_index])
            # print("{}-th data set open~~~".format(dataset_index))
        for dataset_index in range(2, 3):
            for label_index in range(len(labels_training)):
                if label_index == dataset_index:
                    for example_index in range(len(examples_training[label_index])):
                            X_fine_tune_pretrain.extend(examples_training[label_index][example_index])
                            Y_fine_tune_pretrain.extend(labels_training[label_index][example_index])
            # print("{}-th data set open~~~".format(dataset_index))
        for dataset_index in range(3, 4):
            for label_index in range(len(labels_training)):
                if label_index == dataset_index:
                    for example_index in range(len(examples_training[label_index])):
                        X_fine_tune_test.extend(examples_training[label_index][example_index])
                        Y_fine_tune_test.extend(labels_training[label_index][example_index])
            # print("{}-th data set open~~~".format(dataset_index))
    elif (args.dataset == 'smap'):
        for dataset_index in [0,2]: #[0, 1, 2, 3]
            for label_index in range(len(labels_training)):
                # if label_index % 2 == 0:
                X_fine_tune_train.extend(examples_training[label_index])
                Y_fine_tune_train.extend(labels_training[label_index])
                # else:
                #     X_fine_tune_pretrain.extend(examples_training[label_index])
                #     Y_fine_tune_pretrain.extend(labels_training[label_index])
            # print("{}-th data set open~~~".format(dataset_index))
        for dataset_index in [1,3]: #[0, 1, 2, 3]
            for label_index in range(len(labels_training)):
                X_fine_tune_pretrain.extend(examples_training[label_index])
                Y_fine_tune_pretrain.extend(labels_training[label_index])
            # print("{}-th data set open~~~".format(dataset_index))
        for dataset_index in range(4, 7):
            for label_index in range(len(labels_training)):
                X_fine_tune_test.extend(examples_training[label_index])
                Y_fine_tune_test.extend(labels_training[label_index])
    
    elif(args.dataset in ['HAR']):
        #pick test subject and phase1-phase2 division
        # test_subject_num = 0
        X_test, Y_test = examples_training.pop(test_subject_num), labels_training.pop(test_subject_num)
        # if args.print:
        #     print(X_test.shape)
        # X_phase1_train, Y_phase1_train = torch.cat(examples_training[27:28], dim=0), torch.cat(labels_training[27:28], dim=0)
        X_fine_tune_train, Y_fine_tune_train = torch.cat(examples_training[:14], dim=0), torch.cat(labels_training[0:14], dim=0)
        # X_phase1_train, Y_phase1_train = torch.cat(examples_training[:29], dim=0), torch.cat(labels_training[0:29], dim=0)
        # X_phase2_train, Y_phase2_train = torch.cat(examples_training[28:29], dim=0), torch.cat(labels_training[28:29], dim=0)
        X_fine_tune_pretrain, Y_fine_tune_pretrain = torch.cat(examples_training[14:29], dim=0), torch.cat(labels_training[14:29], dim=0)
        # X_phase2_train, Y_phase2_train = torch.cat(examples_training[0:29], dim=0), torch.cat(labels_training[0:29], dim=0)

        #flatten and formatted
        X_fine_tune_train, Y_fine_tune_train = X_fine_tune_train.float(), Y_fine_tune_train
        X_fine_tune_pretrain, Y_fine_tune_pretrain = X_fine_tune_pretrain.float(), Y_fine_tune_pretrain
        X_fine_tune_test, Y_fine_tune_test = X_test.float(), Y_test
    
    elif 'boiler' in args.dataset:#(args.dataset == 'boiler'):
        for dataset_index in range(0, 1):
            for label_index in range(len(labels_training)):
                if label_index % 1 == 0:
                    X_fine_tune_train.extend(examples_training[label_index])
                    Y_fine_tune_train.extend(labels_training[label_index])
                # else:
                #     X_fine_tune_pretrain.extend(examples_training[label_index])
                #     Y_fine_tune_pretrain.extend(labels_training[label_index])
            # print("{}-th data set open~~~".format(dataset_index))
        for dataset_index in range(1, 2):
            for label_index in range(len(labels_training)):
                X_fine_tune_pretrain.extend(examples_training[label_index])
                Y_fine_tune_pretrain.extend(labels_training[label_index])
            # print("{}-th data set open~~~".format(dataset_index))
        for dataset_index in range(2, 3):
            for label_index in range(len(labels_training)):
                X_fine_tune_test.extend(examples_training[label_index])
                Y_fine_tune_test.extend(labels_training[label_index])
    else:
        for dataset_index in range(0, 2):
            for label_index in range(len(labels_training)):
                if label_index == dataset_index:
                    for example_index in range(len(examples_training[label_index])):
                            X_fine_tune_train.extend(examples_training[label_index][example_index])
                            Y_fine_tune_train.extend(labels_training[label_index][example_index])
            # print("{}-th data set open~~~".format(dataset_index))
        for dataset_index in range(2, 4):
            for label_index in range(len(labels_training)):
                if label_index == dataset_index:
                    for example_index in range(len(examples_training[label_index])):
                            X_fine_tune_pretrain.extend(examples_training[label_index][example_index])
                            Y_fine_tune_pretrain.extend(labels_training[label_index][example_index])
            # print("{}-th data set open~~~".format(dataset_index))
        for dataset_index in range(4, 6):
            for label_index in range(len(labels_training)):
                if label_index == dataset_index:
                    for example_index in range(len(examples_training[label_index])):
                        X_fine_tune_test.extend(examples_training[label_index][example_index])
                        Y_fine_tune_test.extend(labels_training[label_index][example_index])
            # print("{}-th data set open~~~".format(dataset_index))
        


    X_fine_tunning, Y_fine_tunning = scramble(X_fine_tune_train, Y_fine_tune_train)
    X_fine_pretrain, Y_fine_pretrain = scramble(X_fine_tune_pretrain, Y_fine_tune_pretrain)
    X_test_0, Y_test_0 = scramble(X_fine_tune_test, Y_fine_tune_test)


    valid_examples = X_fine_tunning[0:int(len(X_fine_tunning) * 0.2)]
    labels_valid = Y_fine_tunning[0:int(len(Y_fine_tunning) * 0.2)]
    X_fine_tune = X_fine_tunning[int(len(X_fine_tunning) * 0.2):]
    Y_fine_tune = Y_fine_tunning[int(len(Y_fine_tunning) * 0.2):]
    
    X_fine_pretrain_valid = X_fine_pretrain[0:int(len(X_fine_tunning) * 0.2)]
    Y_fine_pretrain_valid = Y_fine_pretrain[0:int(len(Y_fine_tunning) * 0.2)]
    X_fine_pretrain = X_fine_pretrain[int(len(X_fine_tunning) * 0.2):]
    Y_fine_pretrain = Y_fine_pretrain[int(len(Y_fine_tunning) * 0.2):]
    
    if args.print:
        print("total data size :", len(X_fine_tune_train), np.shape(np.array(X_fine_tune_train)))

    X_fine_tune = torch.from_numpy(np.array(X_fine_tune, dtype=np.float32))
    X_fine_pretrain = torch.from_numpy(np.array(X_fine_pretrain, dtype=np.float32))
    if args.print:
        print("train data :", np.shape(np.array(X_fine_tune)))
    Y_fine_tune = torch.from_numpy(np.array(Y_fine_tune, dtype=np.float32))
    Y_fine_pretrain = torch.from_numpy(np.array(Y_fine_pretrain, dtype=np.float32))
    
    X_fine_pretrain_valid = torch.from_numpy(np.array(X_fine_pretrain_valid, dtype=np.float32))
    Y_fine_pretrain_valid = torch.from_numpy(np.array(Y_fine_pretrain_valid, dtype=np.float32))
    
    
    valid_examples = torch.from_numpy(np.array(valid_examples, dtype=np.float32))
    if args.print:
        print("valid data :", np.shape(np.array(valid_examples)))
    labels_valid = torch.from_numpy(np.array(labels_valid, dtype=np.float32))
    
    X_test_0 = torch.from_numpy(np.array(X_test_0, dtype=np.float32))
    Y_test_0 = torch.from_numpy(np.array(Y_test_0, dtype=np.float32))


    train_fine = TensorDataset(X_fine_pretrain, Y_fine_pretrain) #for domain adptation
    valid_fine = TensorDataset(X_fine_pretrain_valid, Y_fine_pretrain_valid)
    train = TensorDataset(X_fine_tune, Y_fine_tune)
    valid = TensorDataset(valid_examples, labels_valid)
    test_0 = TensorDataset(X_test_0, Y_test_0)
    if args.print:
        print(torch.unique(Y_fine_tune))
        print(torch.unique(labels_valid))
        print(torch.unique(Y_test_0))
    # data loading
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=args.batch_size, shuffle=True)
    test_0_loader = torch.utils.data.DataLoader(test_0, batch_size=args.batch_size, shuffle=False)

    train_loader_fine = torch.utils.data.DataLoader(train_fine, batch_size=args.batch_size, shuffle=True)
    valid_loader_fine = torch.utils.data.DataLoader(valid_fine, batch_size=args.batch_size, shuffle=True)

    
    stgcn = ARMBANDGNN_last.ARMBANDGNN_modified_rnn_raw(args.channel_electrode, eval(args.channel), args.num_label, args.feature_dim).cuda(args.gpu)
    
    precision = 1e-8
    criterion = nn.NLLLoss(size_average=False)
    optimizer = optim.Adam(stgcn.parameters(), lr=args.lr, weight_decay=1e-4)
    verbose_bool = False
    if args.print:
        verbose_bool = True
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.4, patience=5,
                                                     verbose=verbose_bool, eps=precision)
    
    #training
    model_basic, num_epoch = train_basic(stgcn, criterion, optimizer, scheduler, {"train": train_loader, "val": valid_loader}, args.epoch, precision)

    for name, param in model_basic.named_parameters():
        param.requires_grad = True
    
    adaptation_model = ARMBANDGNN_last.DA_gnn_invariance_ver2(model_basic, args.channel_electrode, args.feature_dim).cuda(args.gpu)
    optimizer_ft = optim.Adam(adaptation_model.parameters(), lr=args.lr) 
    scheduler_ft = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_ft, mode='min', factor=.4, patience=5,
                                                        verbose=verbose_bool, eps=precision)
    model, num_epoch = adapt_da_inv(adaptation_model, criterion, optimizer_ft, scheduler_ft,\
                                    {"train": train_loader_fine, "val": valid_loader_fine}, args.epoch_pre, args.precision)
    
    model.eval()
    
    all_dict = dict()
    correct_dict = dict()
    for i in range(args.num_label):
        all_dict[i] = 0
        correct_dict[i] = 0

    acc_perlabel = []
    total = 0
    correct_prediction_test_0 = 0
    time_list = []
    for k, data_test_0 in enumerate(test_0_loader):
        start_time = time.time()
        inputs_test_0, ground_truth_test_0 = data_test_0

        # inputs_test_0[:, [2,3]] = inputs_test_0[:, [3,2]]

        ## channel deletion
        # target = [0, 1, 2]
        # # count_list = []
        # # for i in range(8):
        # #     if i != target:
        # #         count_list.append(i)
        # # inputs_test_0 = inputs_test_0[:, count_list]
        # # inputs_test_0[:, target] = torch.randn(inputs_test_0[:, target].shape, device=inputs_test_0.device)
        # inputs_test_0[:, target] -= torch.randn(inputs_test_0[:, target].shape, device=inputs_test_0.device)#2

        # inputs_test_0 = inputs_test_0[:, torch.randperm(inputs_test_0.size(1))]
        inputs_test_0[:, :4] = inputs_test_0[:, torch.randperm(4)] # inputs.size(1)

        ### channel addition (random noise)
        # add_channel_num = 3
        # random_noise = torch.randn(inputs_test_0.shape[0], add_channel_num, inputs_test_0.shape[2], device=inputs_test_0.device)
        # inputs_test_0 = torch.cat((inputs_test_0, random_noise), dim=1)

        if args.dataset == 'smap':
            one_idx = (ground_truth_test_0 == 1)
            inputs_test_0 = inputs_test_0[one_idx]
            ground_truth_test_0 = ground_truth_test_0[one_idx]


        inputs_test_0, ground_truth_test_0 = Variable(inputs_test_0).cuda(args.gpu), Variable(ground_truth_test_0).cuda(args.gpu)
        concat_input = inputs_test_0

        _, _, outputs_test_0, _ = model(concat_input)
        # model.plot(concat_input, k)
        # outputs_test_0 = outputs_test_0
        # _, predicted = torch.max(outputs_test_0.data, 1)
        # all_dict[int(ground_truth_test_0)] += 1

        outputs_test_0 = outputs_test_0
        _, predicted = torch.max(outputs_test_0.data, 1)
        total += ground_truth_test_0.size(0)
        correct_prediction_test_0 += (predicted == ground_truth_test_0).sum().item()
        # print(predicted)
        # if mode(predicted.cpu().numpy())[0][0] == ground_truth_test_0.data.cpu().numpy():
        #     correct_dict[int(ground_truth_test_0)] += 1
        # correct_prediction_test_0 += (mode(predicted.cpu().numpy())[0][0] ==
        #                               ground_truth_test_0.data.cpu().numpy()).sum()
        # total += ground_truth_test_0.size(0)
        end = time.time()
        time_list.append(end-start_time)
    accuracy_test0.append(100 * float(correct_prediction_test_0) / float(total))
    print("ACCURACY TESƒT_0 FINAL : %.3f %%" % (100 * float(correct_prediction_test_0) / float(total)))


    f = open('result_nina_hyper.txt', 'a')
    f.write(f'{np.array(accuracy_test0).mean()}, {args.l1}, {args.l2}, {args.l3}, {args.l4} \n')
    #result
    print("AVERAGE ACCURACY TEST 0:   %.3f" % np.array(accuracy_test0).mean())
    return accuracy_test0 , num_epoch


if __name__ == "__main__":
    # loading...
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    if (args.dataset == 'nina'):
        data_file = "['NINA_1_sep1_data_raw.npy', 'NINA_1_sep2_data_raw.npy', 'NINA_1_sep3_data_raw.npy', 'NINA_1_sep4_data_raw.npy', 'NINA_1_sep5_data_raw.npy', 'NINA_1_sep6_data_raw.npy']"
        label_file = "['NINA_1_sep1_label_raw.npy', 'NINA_1_sep2_label_raw.npy', 'NINA_1_sep3_label_raw.npy', 'NINA_1_sep4_label_raw.npy', 'NINA_1_sep5_label_raw.npy', 'NINA_1_sep6_label_raw.npy']"
        examples_training = np.stack([np.load(f"./data/CWT_dataset/{eval(data_file)[i]}", encoding="bytes", allow_pickle=True) for i in range(6)])
        labels_training = np.stack([np.load(f"./data/CWT_dataset/{eval(label_file)[i]}", encoding="bytes", allow_pickle=True) for i in range(6)])
    elif (args.dataset == 'nina_18'):
        data_file = "['NINA_1_sep1_data_raw_18.npy', 'NINA_1_sep2_data_raw_18.npy', 'NINA_1_sep3_data_raw_18.npy', 'NINA_1_sep4_data_raw_18.npy', 'NINA_1_sep5_data_raw_18.npy', 'NINA_1_sep6_data_raw_18.npy']"
        label_file = "['NINA_1_sep1_label_raw_18.npy', 'NINA_1_sep2_label_raw_18.npy', 'NINA_1_sep3_label_raw_18.npy', 'NINA_1_sep4_label_raw_18.npy', 'NINA_1_sep5_label_raw_18.npy', 'NINA_1_sep6_label_raw_18.npy']"
        examples_training = np.stack([np.load(f"./data/CWT_dataset/{eval(data_file)[i]}", encoding="bytes", allow_pickle=True) for i in range(6)])
        labels_training = np.stack([np.load(f"./data/CWT_dataset/{eval(label_file)[i]}", encoding="bytes", allow_pickle=True) for i in range(6)])
    elif (args.dataset =='HAR'):
            global test_subject_num
            test_subject_num = 1
            examples_training = [] # list of npy for each subject
            labels_training = []
            for i in range(1, 31): #subject
                data_file = [f'train_{str(i)}.pt', f'test_{str(i)}.pt']
                examples_labels= [torch.load(f"./data/HAR/{data_file[i]}", encoding="bytes") for i in range(2)]
                examples_subject = torch.cat((torch.tensor(examples_labels[0]['samples']), torch.tensor(examples_labels[1]['samples'])), dim=0)
                labels_subject = torch.cat((torch.tensor(examples_labels[0]['labels']), torch.tensor(examples_labels[1]['labels'])), dim=0)
                examples_training.append(examples_subject) #shape [subject, repetition_num, ]
                labels_training.append(labels_subject)
    
    elif (args.dataset == 'smap'):
        examples_training = []
        labels_training = []
        
        for i in range(1,8):
            # filename = [f'./data/boiler/test_{i}.pt', f'./data/boiler/train_{i}.pt']
            # temp = torch.load(filename[0])['samples']
            # data = np.concatenate([torch.load(filename[i])['samples'] for i in range(2)])
            # label = np.concatenate([torch.load(filename[i])['labels'] for i in range(2)])

            filename = [ f'./data/SMAP/train_{i}.pt']
            temp = torch.load(filename[0])['samples']
            data = np.concatenate([torch.load(filename[i])['samples'] for i in range(1)])
            label = np.concatenate([torch.load(filename[i])['labels'] for i in range(1)])
            # print(label.shape)
            # exit(0)
            examples_training.append(data)
            labels_training.append(label)
        
        
        # for i in range(args.eeg_data_divider):
    
    elif 'boiler' in args.dataset:
        examples_training = []
        labels_training = []
        
        for i in range(1,4):
            # filename = [f'./data/boiler/test_{i}.pt', f'./data/boiler/train_{i}.pt']
            # temp = torch.load(filename[0])['samples']
            # data = np.concatenate([torch.load(filename[i])['samples'] for i in range(2)])
            # label = np.concatenate([torch.load(filename[i])['labels'] for i in range(2)])

            filename = [ f'./data/{args.dataset}/train_{i}.pt']
            temp = torch.load(filename[0])['samples']
            data = np.concatenate([torch.load(filename[i])['samples'] for i in range(1)])
            label = np.concatenate([torch.load(filename[i])['labels'] for i in range(1)])
            
            examples_training.append(data)
            labels_training.append(label)
        
        
        # for i in range(args.eeg_data_divider):


    else:
        data_file = "['NM_정희수2_sep1_data230724_raw.npy', 'NM_정희수2_sep2_data230724_raw.npy', 'NM_정희수2_sep3_data230724_raw.npy','NM_정희수2_sep4_data230724_raw.npy']"
        label_file = "['NM_정희수2_sep1_label230724_raw.npy', 'NM_정희수2_sep2_label230724_raw.npy', 'NM_정희수2_sep3_label230724_raw.npy','NM_정희수2_sep4_label230724_raw.npy']"
        examples_training = np.concatenate([np.load(f"./data/CWT_dataset/{eval(data_file)[i]}", encoding="bytes", allow_pickle=True) for i in range(4)])
        labels_training = np.concatenate([np.load(f"./data/CWT_dataset/{eval(label_file)[i]}", encoding="bytes", allow_pickle=True) for i in range(4)])

    
    accuracy_test_0, num_epochs = fit_freeze(args, examples_training, labels_training)
    
