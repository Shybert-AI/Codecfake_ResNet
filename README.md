# The Codecfake Dataset and Countermeasures for the Universally Detection of Deepfake Audio
Baseline code: https://github.com/xieyuankun/Codecfake

![image](https://github.com/Shybert-AI/Codecfake_ResNet/assets/82042336/962a4bc7-e7a1-4d78-ae23-00a3c29d92f4)

博客：https://blog.csdn.net/weixin_43509698/article/details/139812979

1.下载finvcup9th_1st_ds5数据集，解压到data目录下
2.执行data_prepare.py 脚本生成训练的csv文件，修改finvcup9th_1st_ds5_valid_data.csv为finvcup9th_1st_ds5_dev_data.csv
python data_prepare.py
3.执行提取特征文件
python preprocess.py 
4.训练
python main_train.py  --path_to_features preprocess_xls-r-5  -f1 preprocess_xls-r-5 --out_fold ./pretrained_model/codec_w2v2aasist_ResNet50_CSAM_xls-r-5_300m/ --CSAM True --train_task codecfake  --num_epochs 50  --batch_size 16 --lr 0.001  --gpu 0   --seed  2024   --num_workers 1
5.预测
python predict.py
