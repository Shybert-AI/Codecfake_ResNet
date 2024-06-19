import torch
import torch.nn as nn
import argparse
import os
import json
import shutil
import numpy as np
from model import *
from dataset import *
from CSAM import *
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler, Sampler
import torch.utils.data.sampler as torch_sampler

from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from collections import defaultdict
from tqdm import tqdm, trange
import random
from utils import *
import eval_metrics as em
import warnings
from sklearn import metrics
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")

torch.set_default_tensor_type(torch.FloatTensor)
torch.multiprocessing.set_start_method('spawn', force=True)

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed', type=int, help="random number seed", default=688)

    # Data folder prepare
    parser.add_argument("-a", "--access_type", type=str, help="LA or PA", default='LA')
    parser.add_argument("-d", "--path_to_database", type=str, help="dataset path",
                        default='/data2/xyk/codecfake')
    parser.add_argument("-f", "--path_to_features", type=str, help="features path",
                        default='/data2/xyk/codecfake/preprocess_xls-r-5')
    
    parser.add_argument("-f1", "--path_to_features1", type=str, help="cotrain_dataset1_path",
                        default='/data2/xyk/asv2019/preprocess_xls-r-5')
    
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=False, default='./models/try/')

    # Dataset prepare
    parser.add_argument("--feat", type=str, help="which feature to use", default='xls-r-5',
                        choices=["mel", "xls-r-5"])
    parser.add_argument("--feat_len", type=int, help="features length", default=201)
    parser.add_argument('--pad_chop', type=str2bool, nargs='?', const=True, default=False,
                        help="whether pad_chop in the dataset")
    parser.add_argument('--padding', type=str, default='repeat', choices=['zero', 'repeat', 'silence'],
                        help="how to pad short utterance")

    parser.add_argument('-m', '--model', help='Model arch', default='W2VAASIST',
                        choices=['lcnn','W2VAASIST'])

    # Training hyperparameters
    parser.add_argument('--train_task', type=str, default='co-train', choices=['19LA','codecfake','co-train'], help="training dataset")

    parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=128, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0005, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=2, help="interval to decay lr")

    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="7")
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers")

    parser.add_argument('--base_loss', type=str, default="ce", choices=["ce", "bce"],
                        help="use which loss for basic training")
    parser.add_argument('--continue_training', action='store_true', help="continue training with trained model")

    # generalized strategy 
    parser.add_argument('--SAM', type= bool, default= False, help="use SAM")
    parser.add_argument('--ASAM', type= bool, default= False, help="use ASAM")
    parser.add_argument('--CSAM', type= bool, default= False, help="use CSAM")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds
    setup_seed(args.seed)

    if args.continue_training:
        pass
    else:
        # Path for output data
        if not os.path.exists(args.out_fold):
            os.makedirs(args.out_fold)
        else:
            shutil.rmtree(args.out_fold)
            os.mkdir(args.out_fold)

        # Folder for intermediate results
        if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
            os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
        else:
            shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
            os.mkdir(os.path.join(args.out_fold, 'checkpoint'))

        # Path for input data
        # assert os.path.exists(args.path_to_database)
        assert os.path.exists(args.path_to_features)

        # Save training arguments
        with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
            file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args


def metrics_scores(output, target):
    output = output.detach().cpu().numpy().argmax(axis=1)
    target = target.detach().cpu().numpy()

    accuracy = metrics.accuracy_score(target, output)
    recall = metrics.recall_score(target, output)
    precision = metrics.precision_score(target, output)
    F1 = metrics.f1_score(target, output)

    return accuracy * 100, recall * 100, precision, F1

def plot_draw(save_dir,metrics):
    train_loss,val_loss,train_acc,val_acc,train_prec,val_prec,train_F1,val_F1 = [],[],[],[],[],[],[],[]
    for i in metrics:
        train_loss.append(i[1])
        val_loss.append(i[2])
        train_acc.append(i[3])
        val_acc.append(i[4])
        train_prec.append(i[5])
        val_prec.append(i[6])
        train_F1.append(i[7])
        val_F1.append(i[8])

    fig=plt.figure(figsize=(10,6))
    fig.suptitle('performance metrics')
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(train_loss,label="train_loss")
    ax1.plot(val_loss,label="val_loss")
    ax1.set_title("train_loss/val_loss")
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(train_acc,label="train_acc")
    ax2.plot(val_acc,label="val_acc")
    ax2.set_title("train_acc/val_acc")
    ax2.legend(loc="upper right")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(train_prec,label="train_prec")
    ax3.plot(val_prec,label="val_prec")
    ax3.set_title("train_prec/val_prec")
    ax3.legend(loc="upper right")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(train_F1,label="train_F1")
    ax4.plot(val_F1,label="val_F1")
    ax4.set_title("train_F1/val_F1")
    ax4.legend(loc="upper right")
    plt.savefig(os.path.join(save_dir,"result.png"))
    #plt.show()

def adjust_learning_rate(args, lr, optimizer, epoch_num):
    lr = lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def shuffle(feat,  labels):
    shuffle_index = torch.randperm(labels.shape[0])
    feat = feat[shuffle_index]
    labels = labels[shuffle_index]
    # this_len = this_len[shuffle_index]
    return feat, labels


def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)

    # initialize model
    if args.model == 'W2VAASIST':
        feat_model = W2VAASIST().cuda()

    if args.continue_training:
        feat_model = torch.load(os.path.join(args.out_fold,'checkpoint', 'anti-spoofing_feat_model.pt')).to(args.device)
    #feat_model = nn.DataParallel(feat_model, list(range(torch.cuda.device_count())))  # for multiple GPUs

    feat_optimizer = torch.optim.Adam(feat_model.parameters(), lr=args.lr,
                                      betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)
    
    if args.SAM or args.CSAM:
        feat_optimizer = torch.optim.Adam
        feat_optimizer = SAM(
            feat_model.parameters(),
            feat_optimizer,
            lr=args.lr,
            betas=(args.beta_1, args.beta_2),
            weight_decay=0.0005
        )

    if args.ASAM:
        feat_optimizer = torch.optim.Adam
        feat_optimizer = SAM(
            feat_model.parameters(),
            feat_optimizer,
            lr=args.lr,
            adaptive = True,
            betas=(args.beta_1, args.beta_2),
            weight_decay=0.0005
        )

    if args.train_task == '19LA':
        asv_training_set = ASVspoof2019(args.access_type, args.path_to_features1, 'train',
                                    args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
        asv_validation_set = ASVspoof2019(args.access_type, args.path_to_features1, 'dev',
                                      args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
        trainOriDataLoader = DataLoader(asv_training_set, batch_size=int(args.batch_size),
                                        shuffle=False, num_workers=args.num_workers,
                                        sampler=torch_sampler.SubsetRandomSampler(range(25380)))
        valOriDataLoader = DataLoader(asv_validation_set, batch_size=int(args.batch_size),
                                      shuffle=False, num_workers=args.num_workers,
                                      sampler=torch_sampler.SubsetRandomSampler(range(24844)))

    if args.train_task == 'codecfake':
        codec_training_set = codecfake(args.access_type, args.path_to_features, 'train',
                                    args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
        codec_validation_set = codecfake(args.access_type, args.path_to_features, 'dev',
                                      args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
        trainOriDataLoader = DataLoader(codec_training_set, batch_size=int(args.batch_size ),
                                        shuffle=False, num_workers=args.num_workers, persistent_workers=True,pin_memory= True,
                                        sampler=torch_sampler.SubsetRandomSampler(range(len(codec_training_set))))
        valOriDataLoader = DataLoader(codec_validation_set, batch_size=int(args.batch_size),
                                      shuffle=False, num_workers=args.num_workers,persistent_workers=True,
                                      sampler=torch_sampler.SubsetRandomSampler(range(len(codec_validation_set))))

    if args.train_task == 'co-train':
        # domain_19train,dev
        asv_training_set = ASVspoof2019(args.access_type, args.path_to_features1, 'train',
                                    args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
        asv_validation_set = ASVspoof2019(args.access_type, args.path_to_features1, 'dev',
                                      args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)

        # domain_codectrain, dev
        codec_training_set = codecfake(args.access_type, args.path_to_features, 'train',
                                    args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
        codec_validation_set = codecfake(args.access_type, args.path_to_features, 'dev',
                                      args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)

        # concat dataset
        training_set = ConcatDataset([codec_training_set, asv_training_set])
        validation_set = ConcatDataset([codec_validation_set, asv_validation_set])

        train_total_samples_codec = len(codec_training_set)
        train_total_samples_asv = len(asv_training_set)
        train_total_samples_combined = len(training_set)
        train_codec_weight = train_total_samples_codec / train_total_samples_combined
        train_asv_weight = train_total_samples_asv / train_total_samples_combined

        if args.CSAM:
            trainOriDataLoader = DataLoader(training_set, batch_size=int(args.batch_size),
                                            shuffle=False, num_workers=args.num_workers,
                                            sampler=CSAMSampler(dataset=training_set,
                                batch_size=int(args.batch_size),ratio_dataset1= train_codec_weight,ratio_dataset2 = train_asv_weight))

        if args.SAM or args.ASAM:
            trainOriDataLoader = DataLoader(training_set, batch_size=int(args.batch_size * args.ratio),
                                shuffle=False, num_workers=args.num_workers,pin_memory=True,
                                sampler=torch_sampler.SubsetRandomSampler(range(len(training_set))))
        valOriDataLoader = DataLoader(validation_set, batch_size=int(args.batch_size),
                                      shuffle=False, num_workers=args.num_workers,
                                      sampler=torch_sampler.SubsetRandomSampler(range(len(validation_set))))


    trainOri_flow = iter(trainOriDataLoader)
    valOri_flow = iter(valOriDataLoader)


    weight = torch.FloatTensor([1,1]).to(args.device)   # concentrate on real 0

    if args.base_loss == "ce":
        criterion = nn.CrossEntropyLoss(weight=weight)

    else:
        criterion = nn.functional.binary_cross_entropy()

    #prev_loss = 1e8
    prev_loss = 0
    monitor_loss = 'base_loss'
    metrics_all = []
    for epoch_num in tqdm(range(args.num_epochs)):
        feat_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)
        testlossDict = defaultdict(list)
        adjust_learning_rate(args, args.lr, feat_optimizer, epoch_num)

        acc_t_train, recall_train, prec_train, F1_train = [],[],[],[]
        acc_t_val, recall_val, prec_val, F1_val = [], [], [], []
        for i in tqdm(range(len(trainOriDataLoader))):
            try:
                featOri, audio_fnOri,  labelsOri = next(trainOri_flow)
            except StopIteration:
                trainOri_flow = iter(trainOriDataLoader)
                featOri, audio_fnOri,  labelsOri = next(trainOri_flow)


            feat = featOri
            labels = labelsOri

            feat = feat.transpose(2, 3).to(args.device)
            labels = labels.to(args.device)

            if args.SAM or args.ASAM or args.CSAM:
                enable_running_stats(feat_model)
                # [32, 1, 1024, 201]
                # [16, 1, 1920, 201]
                feats, feat_outputs = feat_model(feat)
                feat_loss = criterion(feat_outputs, labels)
                feat_loss.mean().backward()
                feat_optimizer.first_step(zero_grad=True)

                disable_running_stats(feat_model)
                feats, feat_outputs = feat_model(feat)
                criterion(feat_outputs, labels).mean().backward()
                feat_optimizer.second_step(zero_grad=True)
            
            else:
                feat_optimizer.zero_grad()
                feats, feat_outputs = feat_model(feat)
                feat_loss = criterion(feat_outputs, labels)
                feat_loss.backward()
                feat_optimizer.step()

            acc_t, recall_t, prec_t, F1_t = metrics_scores(feat_outputs, labels)
            acc_t_train.append(acc_t)
            recall_train.append(recall_t)
            prec_train.append(prec_t)
            F1_train.append(F1_t)

            trainlossDict['base_loss'].append(feat_loss.item())

        feat_model.eval()
        with torch.no_grad():
            ip1_loader,  idx_loader, score_loader = [],  [], []
            #for i in trange(0, len(valOriDataLoader), total=len(valOriDataLoader), initial=0):
            for i in tqdm(range(len(valOriDataLoader))):
                try:
                    featOri, audio_fnOri, labelsOri= next(valOri_flow)
                except StopIteration:
                    valOri_flow = iter(valOriDataLoader)
                    featOri, audio_fnOri, labelsOri= next(valOri_flow)
                feat = featOri
                labels = labelsOri

                feat = feat.transpose(2, 3).to(args.device)
                labels = labels.to(args.device)
                feats, feat_outputs = feat_model(feat)

                if args.base_loss == "bce":
                    feat_loss = criterion(feat_outputs, labels.unsqueeze(1).float())
                    score = feat_outputs[:, 0]
                else:
                    feat_loss = criterion(feat_outputs, labels)
                    score = F.softmax(feat_outputs, dim=1)[:, 0]

                acc_v, recall_v, prec_v, F1_v = metrics_scores(feat_outputs, labels)
                acc_t_val.append(acc_v)
                recall_val.append(recall_v)
                prec_val.append(prec_v)
                F1_val.append(F1_v)

                ip1_loader.append(feats)
                idx_loader.append((labels))

                devlossDict["base_loss"].append(feat_loss.item())
                score_loader.append(score)
                scores = torch.cat(score_loader, 0).data.cpu().numpy()
                labels = torch.cat(idx_loader, 0).data.cpu().numpy()

                acc_t_val.append(acc_v)
                recall_val.append(recall_v)
                prec_val.append(prec_v)
                F1_val.append(F1_v)

        p1 = round(np.nanmean(trainlossDict[monitor_loss]),4)
        p2 = round(np.nanmean(devlossDict[monitor_loss]),4)
        p3 = round(np.nanmean(acc_t_train),4)
        p4 = round(np.nanmean(acc_t_val),4)
        p5 = round(np.nanmean(prec_train),4)
        p6 = round(np.nanmean(prec_val),4)
        p7 = round(np.nanmean(F1_train),4)
        p8 = round(np.nanmean(F1_val),4)
        metrics_all.append([epoch_num,p1,p2,p3,p4,p5,p6,p7,p8])
        with open(os.path.join(args.out_fold, "metrics.log"), "a") as log:
            log.write(f"{epoch_num} train_loss={p1} val_loss ={p2} train_acc = {p3} val_acc = {p4} train_prec = {p5} val_prec = {p6} train_F1 = {p7} val_F1 = {p8}\n")

        #valLoss = np.nanmean(devlossDict[monitor_loss])
        valLoss = 0.4*np.nanmean(recall_val) + 0.6*np.nanmean(F1_val)
        if (epoch_num + 1) % 5 == 0:
            torch.save(feat_model, os.path.join(args.out_fold, 'checkpoint',
                                                'anti-spoofing_feat_model_%d.pt' % (epoch_num + 1)))

        if valLoss > prev_loss:
            # Save the model checkpoint
            torch.save(feat_model, os.path.join(args.out_fold,'checkpoint', f'anti-spoofing_feat_model_best_{epoch_num}.pt'))
            prev_loss = valLoss
        if epoch_num == args.num_epochs - 1 :
            with open(os.path.join(args.out_fold, "metrics.log"), "a") as log:
                log.write(f"==================metrics===========================\n")
                log.write(f"{metrics_all}\n")
            plot_draw(args.out_fold,metrics_all)
            torch.save(feat_model, os.path.join(args.out_fold, 'checkpoint', 'last.pt'))
    return feat_model


if __name__ == "__main__":
    args = initParams()
    train(args)
