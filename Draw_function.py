import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import stft
import pywt
import dtcwt
import seaborn as sns

class Draw_diagram():
    '''
    params:
    data: 输入数据
    length: 输入数据长度
    T: 输入数据时间间隔
    fs: 输入数据采样频率
    nperseg: 窗长
    noverlap: 窗重叠长度
    levels:  decomposition levels
    '''
    def __init__(self,data,length,T,fs,nperseg,noverlap,levels):
        self.data = np.array(data)
        self.T = T
        self.length = length
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap   
        self.levels = levels
    def draw_2D_timeDomain(self,):
        t = np.linspace(0, self.T*self.length, self.length, endpoint=True)  # 时间向量
        fig, ax = plt.subplots(figsize=(13, 8))
        fig = plt.figure(dpi=100)

        ax.plot(t, self.data, color='#276fbcff', linewidth=1.0)
        # ax.set_title('Time domain diagram', fontsize=22)  # 标题字号
        # ax.set_xlabel('Time (s)', fontsize=20)  # x轴标签字号
        # ax.set_ylabel('Amplitude', fontsize=20)  # y轴标签字号
        ax.tick_params(axis='both', which='major', labelsize=32)
        plt.show()
  
    def draw_2D_fft(self,):
        N = self.length
        T = self.T
        yf = fft(self.data)  # 快速傅里叶变换 得到复数数组
        xf = fftfreq(N, T)  

        fig, ax = plt.subplots(figsize=(13, 8))
        # 1/N 双边谱归一化因子,将FFT的幅度结果调整到正确的尺度,2/N 单边谱，表示舍弃对称的负频。
        # 频率轴 FFT对称,负频是镜像,舍弃与正频对应的负频。不损失信息
        y = 2.0/N * np.abs(yf[:N//2])  # abs取复数的模表示振频 实部和虚部的比值包含相位信息 
        x = xf[:N//2]
        Max_y = []
        Max_x = []
        Window = 5 # 窗口长度 找出最大值 使图像平滑
        for i in range(0,len(x)//Window):
            y1 = y[i*Window : (i+1)*Window]
            x1 = x[i*Window : (i+1)*Window]
            MaxY = max(y1)
            # MiddleX = (max(x1)+min(x1))/2
            MaxX = max(x1)

            Max_y.append(MaxY)
            Max_x.append(MaxX)

        ax.plot(Max_x,Max_y,color='#276fbcff', linewidth=1.0)  
        # ax.set_title('fft domain diagram')
        # ax.set_xlabel('frequency (Hz)')
        # ax.set_ylabel('megnitude')
        ax.tick_params(axis='both', which='major', labelsize=32)
        plt.show()

    def draw_2D_stft(self,):
        f, t, complex_list = stft(self.data, self.fs, nperseg = self.nperseg ,noverlap =  self.noverlap) 
        magnitude = np.abs(complex_list)
        # new_magnitude = np.transpose(magnitude) # 转置,使x轴为频率轴

        plt.figure(figsize=(13, 8))
        plt.tick_params(axis='both', which='major', labelsize=32)

        plt.pcolormesh(t, f, magnitude, cmap='viridis', shading='gouraud')

        yticks = plt.yticks()[0]  # 读取y轴刻度值
        plt.yticks(yticks[1:-1])  # 从索引 1 开始，去掉最底部刻度(避免和x轴0刻度重复) 和最上方冗余刻度 

        # plt.title('STFT Magnitude', fontsize=22)
        # plt.ylabel('Frequency [Hz]', fontsize=22)
        # plt.xlabel('Time [s]', fontsize=22)
        # plt.colorbar(label='Magnitude')
        plt.show()


    def draw_3D_stft(self,):
        f, t, complex_list = stft(self.data, self.fs, nperseg = self.nperseg ,noverlap =  self.noverlap) 
        magnitude = np.abs(complex_list)  # complex_list复数列表取正值 
        T, F = np.meshgrid(t, f)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(T, F, magnitude, cmap='viridis')

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_zlabel('magnitude')
        ax.set_title('3D Time-Frequency Representation using STFT')

        plt.show()
        return complex_list

    def draw_2D_cwt(self,):

        totalscal = 128
        wavename1 = 'cmor' 
        fc1 = pywt.central_frequency(wavename1)
        # 计算所选小波的中心频率。对于复Morlet小波，pywt.central_frequency 函数返回一个与采样周期相关的归一化中心频率。
        cparam1 = 2 * fc1 * totalscal  
        scales = cparam1 / np.arange(totalscal, 0, -1)
        # 尺度选的越大，表示母小波越宽，对低频分辨率越高
        # 此处有128个尺度，表示使用了128个不同宽度(不同频率强度)的子小波来分析信号
        # 参数a = scale，表示频率的高低 （频域特征），(表示某个点是高频子小波组成还是低频子小波组成）
        # 参数 tao（t） 平移参数，(pywt)自动计算得出，表示信号的时域特征，(表示高频子小波/低频子小波所出现的位置)
        coefficients, frequencies = pywt.cwt(self.data, scales, wavename1, sampling_period=1/self.fs)
        amp1 = abs(coefficients)

        t = np.linspace(0, self.T*self.length, self.length, endpoint=False)

        plt.figure(figsize=(13, 8))
        plt.tick_params(axis='both', which='major', labelsize=32)
        plt.contourf(t, frequencies, amp1, cmap='jet')
        # plt.pcolormesh(t,scales,amp1,shading='gouraud')
        # plt.title('CWT Magnitude')
        # plt.ylabel('Scales')
        # plt.xlabel('Time [s]')

        plt.show()

    def draw_3D_cwt(self,):
        totalscal = 128
        wavename1 = 'cmor' 
        fc1 = pywt.central_frequency(wavename1)
        cparam1 = 2 * fc1 * totalscal  
        scales = cparam1 / np.arange(totalscal, 0, -1)
        coefficients, frequencies = pywt.cwt(self.data, scales, wavename1, sampling_period=1/self.fs)

        t = np.linspace(0, self.T*self.length, self.length,endpoint=False)  
        T, F = np.meshgrid(t, frequencies)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(T, F, np.abs(coefficients), cmap='viridis') #z轴为wavelet系数绝对值(振幅)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_zlabel('Amplitude')
        ax.set_title('3D Time-Frequency Representation using CWT')
        plt.show()
        return coefficients, frequencies

    
    def DTCWT_re_construct(self,):

        t = np.linspace(0, self.T*self.length, self.length,endpoint=False)  
    
        # 进行双树复小波变换（DTCWT）
        transform = dtcwt.Transform1d()
        coeffs = transform.forward(self.data, nlevels=self.levels)  # x 级分解

        # 可视化每一级小波系数的模值
        plt.figure(figsize=(10, 6))
        for i in range(len(coeffs.highpasses)):
            plt.subplot(len(coeffs.highpasses), 1, i + 1)
            plt.plot(np.abs(coeffs.highpasses[i]), label=f'Level {i+1}')
            plt.legend()
            plt.xlabel('Samples')
            plt.ylabel('Magnitude')

        plt.suptitle('DTCWT Highpass Coefficients')
        plt.tight_layout()
        plt.show()

        # 进行逆双树复小波变换（重构信号）
        reconstructed_signal = transform.inverse(coeffs)

        # 绘制原始信号与重构信号对比
        plt.figure(figsize=(10, 4))
        plt.plot(t, self.data, label="Original Signal",linestyle='solid',linewidth=2,color='#1f77b4ff')
        plt.plot(t, reconstructed_signal, label="Reconstructed Signal", linestyle='dashed',linewidth=1,color='#ffa710')
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Original vs Reconstructed Signal")
        plt.show()

    def Draw_2D_DTCWT(self,):
        t = np.linspace(0, self.T*self.length, self.length,endpoint=False)  

        # 进行 DTCWT 变换
        transform = dtcwt.Transform1d()
        coeffs = transform.forward(self.data, nlevels=self.levels)  # x 级分解
        ## 得到 level 个高频分量和 1 个低频分量
        ## coeffs 理论上每一级分解出高频和低频，并将低频送入下一级分解。因此第一级为最高频。
        # print("低频系数 (approximation):", coeffs.lowpass.shape)
        # for i, highpass in enumerate(coeffs.highpasses):
        #     print(f"高频系数 Level {i+1}: {highpass.shape}")

        # 高频分量，构造二维矩阵
        time_steps = len(self.data)
        freq_levels = len(coeffs.highpasses)
        dtcwt_matrix = np.zeros((freq_levels + 1, time_steps)) # +1 --> +low_level
        # 提取每一级的高频系数
        for i in range(freq_levels):
            highpass = np.array(coeffs.highpasses[i])
            magnitudex = np.abs(highpass)
            magnitude = np.concatenate(magnitudex)
            xp = np.linspace(0, time_steps, len(magnitude))
            fp = magnitude
            # 插值
            dtcwt_matrix[i, :] = np.interp(np.linspace(0, time_steps, time_steps), 
                                           xp, 
                                           fp)
         # 低频分量
        lowpass = np.array(coeffs.lowpass)
        magnitudelow = np.concatenate(abs(lowpass))
        # 提取低频系数
        dtcwt_matrix[freq_levels, :] = np.interp(np.linspace(0, time_steps, time_steps), 
                                       np.linspace(0, time_steps, len(magnitudelow)), 
                                       magnitudelow)
        # 绘制时频图
        plt.figure(figsize=(13, 8))
        sns.heatmap(dtcwt_matrix, cmap="jet", cbar=False)  # cbar 右边竖条


        # 设置纵坐标
        num_levels = dtcwt_matrix.shape[0]  # 分解层数
        level_labels = [f"Level {i+1}" for i in range(num_levels)] 
        plt.yticks(ticks=np.arange(num_levels) + 0.5, labels=level_labels, rotation=0)

        # 设置横坐标
        xlabels = [f"{i}" for i in range(0,self.length,400)]
        plt.xticks(ticks=np.linspace(0, dtcwt_matrix.shape[1], num=len(xlabels)), labels=xlabels,rotation=1)
        plt.tick_params(axis='both', which='major', labelsize=30)

        # plt.title("DTCWT Time-Frequency Representation")
        # plt.xlabel("Sample points")
        # plt.ylabel("Decomposition Level")
        plt.show()
        
        # # 查看每一级的频率范围
        # levels --> frequency   [0,  fs/2**n+1  ,  fs/2**n]
        # freq_list = []
        # start_f = 0
        # freq_list.append(start_f)
        # low_f = int( self.fs/2 ** (self.levels + 1) )
        # freq_list.append(low_f)
        # for i in range(self.levels):
        #     low_f = low_f * 2
        #     freq_list.append(low_f)
        # print(freq_list)

        # 融合时频图
        # plt.figure(figsize=(10, 6))
        # plt.imshow(dtcwt_matrix, aspect='auto', cmap='jet', origin='lower',
        #         extent=[t[0], t[-1], 1, freq_levels])
        # plt.colorbar(label="Magnitude")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Decomposition Level")
        # plt.title("DTCWT Time-Frequency Representation")
        # plt.show()