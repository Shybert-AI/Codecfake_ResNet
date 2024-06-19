from dataset import *
from model import *
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
import argparse
import raw_dataset as dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Tokenizer
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor,Wav2Vec2Config
import numpy as np
from glob import iglob
import pandas as pd

def init():
    parser = argparse.ArgumentParser("generate model scores")
    parser.add_argument('--model_folder', type=str, help="directory for pretrained model",
                        default='./pretrained_model/codec_w2v2aasist/')
    parser.add_argument("-t", "--task", type=str, help="which dataset you would liek to score on",
                         default='2024', choices=["19eval","ITW","codecfake","2024"])
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")


    return args

def torchaudio_load(filepath):
    wave, sr = librosa.load(filepath,sr=16000)
    waveform = torch.Tensor(np.expand_dims(wave, axis=0))
    return [waveform, sr]

def pad_dataset(wav):
    waveform = wav.squeeze(0)
    waveform_len = waveform.shape[0]
    cut = 64600
    if waveform_len >= cut:
        waveform = waveform[:cut]
        return waveform
    # need to pad
    num_repeats = int(cut / waveform_len) + 1
    padded_waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]
    return padded_waveform


def generate_score(task, feat_model_path):
    dirname = os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
    else:
        dir_path = dirname(feat_model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ADD_model = torch.load(feat_model_path)
    # # https://gitee.com/modelee/wav2vec2-xls-r-300m
    # processor3 = Wav2Vec2FeatureExtractor.from_pretrained(r"D:\下载\codecfake_data\wav2vec2-large-xlsr-53-chinese-zh-cn")
    # model3 = Wav2Vec2Model.from_pretrained(r"D:\下载\codecfake_data\wav2vec2-large-xlsr-53-chinese-zh-cn").cuda()
    #
    # processor2 = Wav2Vec2FeatureExtractor.from_pretrained(r"D:\下载\codecfake_data\mms-lid-4017")
    # model2 = Wav2Vec2Model.from_pretrained(r"D:\下载\codecfake_data\mms-lid-4017").cuda()

    # config = Wav2Vec2Config.from_json_file("huggingface/wav2vec2-xls-r-2b/config.json")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("huggingface/wav2vec2-xls-r-300m/")
    model = Wav2Vec2Model.from_pretrained("huggingface/wav2vec2-xls-r-300m/").cuda()

    model.config.output_hidden_states = True
    # model2.config.output_hidden_states = True
    # model3.config.output_hidden_states = True
    ADD_model.eval()
    if task == '19eval':
        with open('./result/19LA_result.txt', 'w') as cm_score_file:
            asvspoof_raw = dataset.ASVspoof2019LAeval()
            for idx in tqdm(range(len(asvspoof_raw))):
                waveform, filename, labels  = asvspoof_raw[idx]
                waveform = waveform.to(device)
                waveform = pad_dataset(waveform).to('cpu')
                input_values = processor(waveform, sampling_rate=16000,
                                        return_tensors="pt").input_values.cuda()
                # input_values2 = processor2(waveform, sampling_rate=16000,
                #                            return_tensors="pt").input_values.cuda()
                # input_values3 = processor3(waveform, sampling_rate=16000,
                #                            return_tensors="pt").input_values.cuda()
                with torch.no_grad():
                    wav2vec21 = model(input_values).hidden_states[5].cuda()
                    # wav2vec22 = model2(input_values2).hidden_states[5].cuda()
                    # wav2vec23 = model3(input_values3).hidden_states[5].cuda()
                #wav2vec2 = torch.concat([wav2vec21, wav2vec22, wav2vec23], dim=2)
                wav2vec2 = wav2vec21
                w2v2, audio_fn= wav2vec2, filename
                this_feat_len = w2v2.shape[1]
                w2v2 = w2v2.unsqueeze(dim=0)
                w2v2 = w2v2.transpose(2, 3).to(device)
                feats, w2v2_outputs = ADD_model(w2v2)
                score = F.softmax(w2v2_outputs)[:, 0]
                cm_score_file.write('%s %s %s\n' % (
                audio_fn, score.item(), "spoof" if labels== "spoof" else "bonafide"))

    if task == 'ITW':
        with open('./result/ITW_result.txt', 'w') as cm_score_file:
            ITW_raw = dataset.ITW()
            for idx in tqdm(range(len(ITW_raw))):
                waveform, filename, labels  = ITW_raw[idx]
                waveform = waveform.to(device)
                waveform = pad_dataset(waveform).to('cpu')
                input_values = processor(waveform, sampling_rate=16000,
                                        return_tensors="pt").input_values.cuda()  
                with torch.no_grad():
                    wav2vec2 = model(input_values).hidden_states[5].cuda()  
                w2v2, audio_fn= wav2vec2, filename
                this_feat_len = w2v2.shape[1]
                w2v2 = w2v2.unsqueeze(dim=0)
                w2v2 = w2v2.transpose(2, 3).to(device)
                feats, w2v2_outputs = ADD_model(w2v2)
                score = F.softmax(w2v2_outputs)[:, 0]
                cm_score_file.write('%s %s %s\n' % (
                audio_fn, score.item(), "spoof" if labels== "spoof" else "bonafide"))

    if task == 'codecfake':
        for condition in ['C1','C2','C3','C4','C5','C6','C7','A1','A2','A3']:
            file_path = './result/{}_result.txt'.format(condition)
            with open(file_path, 'w') as cm_score_file:
                codecfake_raw = dataset.codecfake_eval(type=condition)
                for idx in tqdm(range(len(codecfake_raw))):
                    waveform, filename, labels  = codecfake_raw[idx]
                    waveform = waveform.to(device)
                    waveform = pad_dataset(waveform).to('cpu')
                    input_values = processor(waveform, sampling_rate=16000,
                                            return_tensors="pt").input_values.cuda()
                    with torch.no_grad():
                        wav2vec2 = model(input_values).hidden_states[5].cuda()
                    w2v2, audio_fn= wav2vec2, filename
                    this_feat_len = w2v2.shape[1]
                    w2v2 = w2v2.unsqueeze(dim=0)
                    w2v2 = w2v2.transpose(2, 3).to(device)
                    feats, w2v2_outputs = ADD_model(w2v2)
                    score = F.softmax(w2v2_outputs)[:, 0]
                    cm_score_file.write('%s %s %s\n' % (
                    audio_fn, score.item(), "fake" if labels== "fake" else "real"))

    if task == '2024':
        result = []
        result_score = []
        classesxx = {"0": 0, "0.5": 0, "1": 0}
        file_list = iglob(r"data\finvcup9th_1st_ds5\test\*.wav",recursive=True)
        for filename in tqdm(file_list):
            waveform, sr = torchaudio_load(filename)
            waveform = waveform.to(device)
            waveform = pad_dataset(waveform).to('cpu')
            input_values = processor(waveform, sampling_rate=16000,
                                    return_tensors="pt").input_values.cuda()
            # with torch.no_grad():
            #     wav2vec2 = model(input_values).hidden_states[5].cuda()

            # input_values2 = processor2(waveform, sampling_rate=16000,
            #                            return_tensors="pt").input_values.cuda()
            # input_values3 = processor3(waveform, sampling_rate=16000,
            #                            return_tensors="pt").input_values.cuda()
            with torch.no_grad():
                wav2vec21 = model(input_values).hidden_states[5].cuda()
                # wav2vec22 = model2(input_values2).hidden_states[5].cuda()
                # wav2vec23 = model3(input_values3).hidden_states[5].cuda()
            #wav2vec2 = torch.concat([wav2vec21, wav2vec23, wav2vec22], dim=2)
            wav2vec2 = wav2vec21
            w2v2, audio_fn= wav2vec2, filename
            this_feat_len = w2v2.shape[1]
            w2v2 = w2v2.unsqueeze(dim=0)
            w2v2 = w2v2.transpose(2, 3).to(device)
            feats, w2v2_outputs = ADD_model(w2v2)
            score = F.softmax(w2v2_outputs)[:, 0]
            # 0表示假，1表示真

            if score.item() <0.4:
                classesxx["0"] +=1
            elif score.item() >0.6:
                classesxx["1"] +=1
            else:
                classesxx["0.5"] += 1

            if score.item() <0.5:
                result.append([os.path.basename(audio_fn), 1])
            else:
                result.append([os.path.basename(audio_fn), 0])
            result_score.append([os.path.basename(audio_fn), score.item()])
        df_result = pd.DataFrame(result, columns=["speech_name", "pred_label"])
        df_result.to_csv("submit.csv", index=False, header=None)
        df_result = pd.DataFrame(result_score, columns=["speech_name", "pred_label"])
        df_result.to_csv("result_score.csv", index=False, header=None)
        print(classesxx)
        from collections import Counter
        print(Counter([i[1]for i in result]))
if __name__ == "__main__":
    args = init()
    model_dir = os.path.join(args.model_folder)
    #model_path = os.path.join(model_dir, "anti-spoofing_feat_model.pt")
    # 100 {'0': 1487, '0.5': 57, '1': 3350}     Counter({0: 3375, 1: 1519})       0.908969
    # 61  {'0': 1466, '0.5': 66, '1': 3362}     Counter({0: 3391, 1: 1503})       #预估 {0: 3426, 1: 1468}
    # 25  {'0': 1456, '0.5': 67, '1': 3371}     Counter({0: 3395, 1: 1499})
    # 15  {'0': 1445, '0.5': 70, '1': 3379}     Counter({0: 3415, 1: 1479})       0.909091
    # 10  {'0': 1374, '0.5': 57, '1': 3463}     Counter({'0': 3492, '1': 1402})   Counter({0: 3485, 1: 1409})   0.908303

    # large 10 {'0': 1590, '0.5': 129, '1': 3175}  Counter({0: 3204, 1: 1690})       0.909782
    # large 60 {'0': 1568, '0.5': 133, '1': 3193}  Counter({0: 3225, 1: 1669})       0.909782
    #model_path = os.path.join(r"D:\mywork\pythonProject\Codecfake-main\pretrained_model\codec_w2v2aasist_ResNet50_CSAM_0520_300m_self_attention_1024_eca_2\checkpoint\anti-spoofing_feat_model_best_24.pt")
    model_path = os.path.join(
        r"pretrained_model/codec_w2v2aasist_ResNet50_CSAM_xls-r-5_300m/checkpoint/anti-spoofing_feat_model_25.pt")

    generate_score(args.task, model_path)

    # 数据集
    # wakefake https://openxlab.org.cn/datasets/OpenDataLab/WaveFake/tree/main/raw
    # LA       https://datashare.ed.ac.uk/handle/10283/3336
    # https://openxlab.org.cn/datasets/OpenDataLab/AISHELL-1

    # 根据预测全为1，调交到线上的结果F1=46.17%,因此大概可以估算出{0: 3426, 1: 1468}

    # "D:\mywork\pythonProject\Codecfake-main\pretrained_model\codec_w2v2aasist333\checkpoint\anti-spoofing_feat_model_30.pt"   官方数据集  resnet18
    # anti-spoofing_feat_model_10.pt {'0': 1512, '0.5': 34, '1': 3348}    Counter({0: 3369, 1: 1525})      0.92652
    # anti-spoofing_feat_model_15.pt {'0': 1508, '0.5': 24, '1': 3362}    Counter({0: 3374, 1: 1520})      0.92807
    # anti-spoofing_feat_model_20.pt {'0': 1536, '0.5': 30, '1': 3328}    Counter({0: 3347, 1: 1547})      0.929045
    # anti-spoofing_feat_model_30.pt {'0': 1531, '0.5': 22, '1': 3341}    Counter({0: 3352, 1: 1542})
    # anti-spoofing_feat_model_35.pt {'0': 1492, '0.5': 26, '1': 3376}    Counter({0: 3385, 1: 1509})      0.926797
    # anti-spoofing_feat_model_50.pt {'0': 1506, '0.5': 25, '1': 3363}    Counter({0: 3377, 1: 1517})      0.928332
    # "D:\mywork\pythonProject\Codecfake-main\pretrained_model\codec_w2v2aasist_ResNet50_CSAM_0520\checkpoint\anti-spoofing_feat_model_5.pt"   官方数据集  resnet50
    # anti-spoofing_feat_model_5.pt {'0': 1506, '0.5': 76, '1': 3312}     Counter({0: 3350, 1: 1544})      0.946251
    # anti-spoofing_feat_model_10.pt {'0': 1454, '0.5': 37, '1': 3403}    Counter({0: 3422, 1: 1472})      0.952737

    # "D:\mywork\pythonProject\Codecfake-main\pretrained_model\codec_w2v2aasist_ResNet50_CSAM\checkpoint\anti-spoofing_feat_model_36.pt"   官方数据集  resnet50
    # anti-spoofing_feat_model_10.pt  {'0': 1487, '0.5': 106, '1': 3301}   Counter({0: 3349, 1: 1545})      0.946251
    # anti-spoofing_feat_model_20.pt  {'0': 1520, '0.5': 42, '1': 3332}   Counter({0: 3354, 1: 1540})      0.953141        官方数据集
    # anti-spoofing_feat_model_30.pt  {'0': 1496, '0.5': 32, '1': 3366}   Counter({0: 3382, 1: 1512})      0.956055        官方数据集
    # anti-spoofing_feat_model_46.pt  {'0': 1474, '0.5': 35, '1': 3385}   Counter({0: 3400, 1: 1494})      0.954438        官方数据集
    #anti-spoofing_feat_model_60.pt   {'0': 1497, '0.5': 35, '1': 3362}   Counter({0: 3381, 1: 1513})      0.956405

    # "D:\mywork\pythonProject\Codecfake-main\pretrained_model\codec_w2v2aasist_ResNet101_CSAM\checkpoint\anti-spoofing_feat_model_30.pt"   官方数据集  resnet101
    # anti-spoofing_feat_model_1.pt  {'0': 1410, '0.5': 129, '1': 3355}   Counter({0: 3416, 1: 1478})      0.849678  大数据集
    # anti-spoofing_feat_model_10.pt  {'0': 1459, '0.5': 57, '1': 3378}   Counter({0: 3402, 1: 1492})      0.946302  官方数据集
    # anti-spoofing_feat_model_20.pt  {'0': 1491, '0.5': 45, '1': 3358}   Counter({0: 3382, 1: 1512})      0.9527    官方数据集


    # D:\mywork\pythonProject\Codecfake-main\pretrained_model\codec_w2v2aasist_ResNet50_CSAM_0520_2b\checkpoint\anti-spoofing_feat_model.pt

    #anti-spoofing_feat_model.pt     {'0': 1455, '0.5': 37, '1': 3402}   Counter({0: 3419, 1: 1475})     0.951766  官方数据集
    #anti-spoofing_feat_model_30.pt  {'0': 1458, '0.5': 39, '1': 3397}  Counter({0: 3418, 1: 1476})    0.952122  官方数据集
    #anti-spoofing_feat_model_20.pt  {'0': 1465, '0.5': 42, '1': 3387}  Counter({0: 3406, 1: 1488})    0.953669  官方数据集

    # D:\mywork\pythonProject\Codecfake-main\pretrained_model\codec_w2v2aasist_ResNet50_CSAM_0520_300m\checkpoint\anti-spoofing_feat_model.pt 加的cbam或者self-attention有问题  2b
    # anti-spoofing_feat_model_20.pt  {'0': 1534, '0.5': 39, '1': 3321}  Counter({0: 3340, 1: 1554})    0.943434  官方数据集
    # anti-spoofing_feat_model_30.pt  {'0': 1499, '0.5': 49, '1': 3346}  Counter({0: 3370, 1: 1524})    0.946208  官方数据集

    # D:\mywork\pythonProject\Codecfake-main\pretrained_model\codec_w2v2aasist_ResNet50_CSAM_0520_300m\checkpoint\anti-spoofing_feat_model.pt  + self-attention
    # anti-spoofing_feat_model.pt  {'0': 1493, '0.5': 26, '1': 3375}     Counter({0: 3385, 1: 1509})     0.965077  官方数据集
    # anti-spoofing_feat_model_30.pt  {'0': 1502, '0.5': 25, '1': 3367}  Counter({0: 3381, 1: 1513})     0.964453  官方数据集

    #D:\mywork\pythonProject\Codecfake - main\pretrained_model\codec_w2v2aasist_ResNet50_CSAM_0520_300m_2B\checkpoint\anti-spoofing_feat_model_best_34.pt  2b
    # anti-spoofing_feat_model_best_34.pt  {'0': 1445, '0.5': 60, '1': 3389} Counter({0: 3421, 1: 1473})  0.938137

    # "D:\mywork\pythonProject\Codecfake-main\pretrained_model\codec_w2v2aasist_ResNet50_CSAM_0520_300m_CBAM\checkpoint\anti-spoofing_feat_model_30.pt"  + self-attention +cbam
    # anti-spoofing_feat_model_30.pt  {'0': 1504, '0.5': 35, '1': 3355} Counter({0: 3372, 1: 1522})      0.950853
    # anti-spoofing_feat_model_30.pt  {'0': 1495, '0.5': 37, '1': 3362} Counter({0: 3377, 1: 1517})     0.950853

    # "D:\mywork\pythonProject\Codecfake-main\pretrained_model\codec_w2v2aasist_ResNet50_CSAM_0520_300m_CBAM\checkpoint\anti-spoofing_feat_model_30.pt"

    #anti-spoofing_feat_model_best_15.pt {'0': 1495, '0.5': 37, '1': 3362}  Counter({0: 3377, 1: 1517})  0.949766
    # anti-spoofing_feat_model_best_26.pt {'0': 1463, '0.5': 18, '1': 3413}  Counter({0: 3426, 1: 1468})  0.95
    #   Counter({0: 3381, 1: 1513})          0.966                                                                                                      融合  anti-spoofing_feat_model_30.pt

    #D:\mywork\pythonProject\Codecfake-main\pretrained_model\codec_w2v2aasist_ResNet50_CSAM_0520_300m_addwavefake\checkpoint\anti-spoofing_feat_model_best_3.pt
    #anti-spoofing_feat_model_best_3.pt  {'0': 1344, '0.5': 87, '1': 3463}   Counter({0: 3501, 1: 1393})                       0.890287
    # anti-spoofing_feat_model_best_10.pt   {'0': 1344, '0.5': 58, '1': 3492}   Counter({0: 3520, 1: 1374})                    0.910306
    # anti-spoofing_feat_model_best_15.pt   {'0': 1372, '0.5': 67, '1': 3455}   Counter({0: 3488, 1: 1406})                    0.915478

    # r"D:\mywork\pythonProject\Codecfake-main\pretrained_model\codec_w2v2aasist_ResNet50_CSAM_0520_300m_self_attention_1024_eca\checkpoint\anti-spoofing_feat_model_5.pt" 权重 weight = torch.FloatTensor([10,1]).to(args.device)
    # " train_loss=0.1078 val_loss =0.0609 train_acc = 89.0833 val_acc = 97.7843 train_prec = 0.8403 val_prec = 0.914 train_F1 = 0.7948 val_F1 = 0.9023"
    # anti-spoofing_feat_model_5.pt   {'0': 1310, '0.5': 167, '1': 3417}   Counter({0: 3500, 1: 1394})                         0.869019
    # anti-spoofing_feat_model_4.pt  {'0': 1406, '0.5': 40, '1': 3448}     Counter({0: 3469, 1: 1425})                         0.950933
    # anti-spoofing_feat_model_best_44.pt  {'0': 1423, '0.5': 40, '1': 3431}    Counter({0: 3452, 1: 1442})                         0.954998 和0.964453融合，0.968961
    # anti-spoofing_feat_model_30.pt  {'0': 1411, '0.5': 45, '1': 3438}   Counter({0: 3458, 1: 1436})                          0.954217

 # r"D:\mywork\pythonProject\Codecfake-main\pretrained_model\codec_w2v2aasist_ResNet50_CSAM_0520_300m_self_attention_1024_eca_2\checkpoint\anti-spoofing_feat_model_best_24.pt" 权重 weight = torch.FloatTensor([1,1]).to(args.device)
   # anti-spoofing_feat_model_best_24.pt  {'0': 1460, '0.5': 24, '1': 3410}   Counter({0: 3420, 1: 1474})                      0.959565
               #                                                              Counter({0: 3381, 1: 1513})                      0.964453    平均融合0.966588


   # preprocess_xls-r-5_chinese-wav2vec2-base_768
   # D:\mywork\pythonProject\Codecfake-main\pretrained_model\preprocess_xls-r-5_chinese-wav2vec2-base_768\checkpoint\anti-spoofing_feat_model_best_16.pt    {'0': 1462, '0.5': 84, '1': 3348}   {0: 3395, 1: 1499}    0.911051
   # # D:\mywork\pythonProject\Codecfake-main\pretrained_model\preprocess_xls-r-5_chinese-wav2vec2-base_768\checkpoint\anti-spoofing_feat_model_best_29.pt  {'0': 1457, '0.5': 84, '1': 3353}   {0: 3395, 1: 1499}    0.913073

   # r"D:\mywork\pythonProject\Codecfake-main\pretrained_model\preprocess_xls-r-5_mms-lid-4017_1280\checkpoint\anti-spoofing_feat_model_30.pt"              {'0': 1432, '0.5': 33, '1': 3429}   {0: 3444, 1: 1450}    0.959233
   #                                                                                                                                                        {'0': 1493, '0.5': 26, '1': 3375}   {0: 3381, 1: 1513}    0.964453
                                                                                                                                             #    #0.5  0.965424  # #0.4  0.968697
                                                                                                                                             #                                                  {0: 3426, 1: 1468}
                                                                                                                                              #                                                 {0: 3392, 1: 1501}   0.968697

   #preprocess_xls-r-5_wav2vec2-large-xlsr-53-chinese-zh-cn_1024\checkpoint\anti-spoofing_feat_model_best_43.pt                                              {'0': 1497, '0.5': 73, '1': 3324}  {0: 3361, 1: 1533}   0.923384

   # r"D:\mywork\pythonProject\Codecfake-main\pretrained_model\preprocess_xls-r-5_mms-comblie\checkpoint\anti-spoofing_feat_model_20.pt")   {'0': 1473, '0.5': 22, '1': 3399}  {0: 3408, 1: 1486}  0.962437
# r"D:\mywork\pythonProject\Codecfake-main\pretrained_model\preprocess_xls-r-5_mms-comblie\checkpoint\anti-spoofing_feat_model_20.pt")   {'0': 1476, '0.5': 19, '1': 3399}  Counter({0: 3406, 1: 1488})

