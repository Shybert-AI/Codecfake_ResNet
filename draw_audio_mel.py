"""Python绘制语谱图"""
"""Python绘制时域波形"""

# 导入相应的包
import numpy, wave
import matplotlib.pyplot as plt
import numpy as np
import os

filepath = './data/finvcup9th_1st_ds5/train/'  # 添加路径
filename = ["1faa35017631429207ac7c1d5544ff94.wav"]
# for i in range(len(filename)):
#     f = wave.open(filepath + filename[i], 'rb')  # 调用wave模块中的open函数，打开语音文件。
#     params = f.getparams()  # 得到语音参数
#     nchannels, sampwidth, framerate, nframes = params[:4]  # nchannels:音频通道数，sampwidth:每个音频样本的字节数，framerate:采样率，nframes:音频采样点数
#     strData = f.readframes(nframes)  # 读取音频，字符串格式
#     wavaData = np.fromstring(strData, dtype=np.int16)  # 得到的数据是字符串，将字符串转为int型
#     wavaData = wavaData * 1.0/max(abs(wavaData))  # wave幅值归一化
#     wavaData = np.reshape(wavaData, [nframes, nchannels]).T  # .T 表示转置
#     f.close()
#
#     #（1）绘制语谱图
#     plt.figure()
#     plt.specgram(wavaData[0], Fs=framerate, scale_by_freq=True, sides='default')  # 绘制频谱
#     plt.xlabel('Time(s)')
#     plt.ylabel('Frequency')
#     plt.title("Spectrogram_{}".format(i+1))
#     plt.show()
#
#     #（2）绘制时域波形
#     time = np.arange(0, nframes) * (1.0 / framerate)
#     time = np.reshape(time, [nframes, 1]).T
#     plt.plot(time[0, :nframes], wavaData[0, :nframes], c="b")
#     plt.xlabel("time(seconds)")
#     plt.ylabel("amplitude")
#     plt.title("Original wave")
#     plt.show()

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt

audio_file = './data/finvcup9th_1st_ds5/train/1faa35017631429207ac7c1d5544ff94.wav'
audio, sr = librosa.load(audio_file, sr=16000)
plt.rcParams.update({"font.size": 10})  # 设置图中字体大小
# fig, axes = plt.subplots(
#     nrows=1, ncols=1, figsize=(10, 6), dpi=80, facecolor="w", edgecolor="k"
# )
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(2, 1, 1)
audio_spectrogram = librosa.amplitude_to_db(librosa.stft(audio))  # 得到语谱图
librosa.display.specshow(audio_spectrogram, y_axis='log', cmap='coolwarm')  # 绘制语谱图

plt.colorbar(format='%+2.0f dB')  # 转换至dB标度
ax1.set_title('spectrogram')
ax1.set_xlabel('time(s)')
ax1.set_ylabel('freq(Hz)')

# （2）绘制时域波形
ax2 = fig.add_subplot(2, 1, 2)
time = np.arange(0, audio.shape[0]) * (1.0 / sr)
time = np.reshape(time, [audio.shape[0], 1]).T
ax2.plot(time[0, :audio.shape[0]], audio, c="b")
ax2.set_xlabel("time(seconds)")
ax2.set_ylabel("amplitude")
plt.show()