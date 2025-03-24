import numpy as np
import matplotlib.pyplot as plt
from tftb.processing import WignerVilleDistribution
from scipy.signal import hilbert,stft,windows
from Operate_RawData import Operate_RawData
from scipy.interpolate import interp1d
from Draw_function import Draw_diagram
from scipy.stats import mode
# 将窗口边界设置为[16，128],一段完整的信号长度为2048,将信号分成16段，求每段的最佳窗口。
# 输入xju六种信号，每种信号进行10次PSO算法，取最佳窗口
def getdata(datapath):

    Raw_Data = Operate_RawData(   
        txtdata_path = datapath,
        channel = 3,
        start_idx = 1000, 
        length = 2048
        )
    raw_data = Raw_Data.get_raw_data()
    return np.array(raw_data)
def ideal_wvd_fitness(signal):
    # analytic_signal = hilbert(signal)
    wvd = WignerVilleDistribution(signal)
    tfr, times, freqs = wvd.run()
    tfr = np.abs(tfr) ** 2  # 获取能量分布
    tfr = tfr / np.max(tfr)  # 归一化
    # 计算改进的Renyi熵（抑制交叉项） 
    alpha = 3  # 使用更高α值抑制交叉项影响 增强高频
    tfr_flat = tfr.flatten()
    tfr_flat = tfr_flat[tfr_flat > 0]  # 避免log(0)
    # 负renyi熵计算公式，越低集中度却高，越高集中度越低，表示能量分布散，有交叉项或噪声
    ideal_renyi_entropy = (1/(1-alpha)) * np.log(np.sum(tfr_flat ** alpha)) 
    print("Ideal_WVD_Renyi Entropy:", ideal_renyi_entropy) # Ideal_WVD_Renyi Entropy: -2.1874789270138786
    return ideal_renyi_entropy
# 改进的适应度函数（基于WVD的时频集中度）
def wvd_fitness(window_params, signal, fs , ideal_entropy):
    win_len = int(np.clip(window_params, 16,127))  # 约束窗口长度范围

    # 使用高斯窗
    gaussian_window = windows.gaussian(win_len,std = win_len)
    f, t, Zxx = stft(signal, fs, window = gaussian_window, nperseg = win_len)
    # 使用默认窗
    # f, t, Zxx = stft(signal, fs, nperseg = win_len)

    Zxx = np.abs(Zxx) ** 2  # 获取能量分布
    Zxx = Zxx / np.max(Zxx)  # 归一化

    # 计算改进的Renyi熵（抑制交叉项） 
    alpha = 3  # 使用更高α值抑制交叉项影响 增强高频
    Zxx_flat = Zxx.flatten()
    Zxx_flat = Zxx_flat[Zxx_flat > 0]  # 避免log(0)
    # 负renyi熵计算公式，越低集中度却高，越高集中度越低，表示能量分布散，有交叉项或噪声
    renyi_entropy1 = (1/(1-alpha)) * np.log(np.sum(Zxx_flat ** alpha)) 

    renyi_entropy_loss = renyi_entropy1 - ideal_entropy
    # 计算当前窗口 的 负熵与WVD 的负熵之间的 差值  使插值越小则越逼近
    # 添加窗口长度惩罚项，防止窗口过大
    length_penalty = (win_len / 500) ** 2  # 经验系数
    # print("one's window_length:",win_len,"one's entrooy:",renyi_entropy1)
    return renyi_entropy_loss + 0.4 * length_penalty # 0.4 看作自设超参数，惩罚权重 

class WVD_PSO:
    def __init__(self, n_particles, dimensions, bounds, max_iter):
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.max_iter = max_iter
        
        # 改进的粒子初始化策略
        self.positions = np.random.uniform(
            low=self.bounds[0], 
            high=self.bounds[1],
            size=(n_particles, dimensions)
        )
        self.velocities = 0.1 * (self.bounds[1] - self.bounds[0]) * np.random.randn(n_particles, dimensions)
        
        # 列表字典数组-要加copy   嵌套--加deepcopy  copy为浅层(第一层)
        self.best_positions = self.positions.copy()     # 个体最优 加上copy后为两个对象，修改其中一个不会影响另一个
        self.best_scores = np.full(n_particles, np.inf) # 最优适应度 初始化无穷大 (要找适应度最小)
        self.global_best_position = None                # 全局最优
        self.global_best_score = np.inf                 # 全局最优适应度
        
        # 自适应参数
        self.w_min = 0.4
        self.w_max = 0.9   # 变化的惯性权重
        self.c1 = 2.0      # 个体 及 全局 认知因子
        self.c2 = 2.0
    
    def optimize(self, objective_func, signal, fs,ideal_entropy):
        for iter in range(self.max_iter):
            # 动态调整惯性权重(相当于调整学习率)
            inertia = self.w_max - (self.w_max - self.w_min) * iter / self.max_iter
            
            for i in range(self.n_particles):
                current_score = objective_func(self.positions[i], signal, fs,ideal_entropy)
                
                if current_score < self.best_scores[i]:
                    self.best_scores[i] = current_score
                    self.best_positions[i] = self.positions[i].copy()
                    
                    if current_score < self.global_best_score:
                        self.global_best_score = current_score
                        self.global_best_position = self.positions[i].copy()

            # 带约束的速度更新
            r1 = np.random.rand(self.n_particles, self.dimensions)
            r2 = np.random.rand(self.n_particles, self.dimensions)

            cognitive = self.c1 * r1 * (self.best_positions - self.positions)
            social = self.c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = inertia * self.velocities + cognitive + social
            
            # 位置更新与边界处理
            self.positions += self.velocities
            self.positions = np.clip(self.positions, 
                                     self.bounds[0], 
                                     self.bounds[1])
            
            # 添加随机扰动（防止早熟）
            if iter % 10 == 0:
                noise = 0.1 * (self.bounds[1] - self.bounds[0]) * np.random.randn(*self.positions.shape)
                self.positions = np.clip(self.positions + noise, 
                                         self.bounds[0], 
                                         self.bounds[1])
            print(f"Iteration: {iter}, Best Score: {self.global_best_score}")
            print(f"Best G Position: {self.global_best_position} ")
        return self.global_best_position, self.global_best_score

def draw_spectrogram(winlist,allsignal,fs,segmentation_times):
    t_lengthx = 0
    magnitude_list = []
    for i in range(segmentation_times):
        seg_signal = allsignal[i*128:(i+1)*128]
        win_length = winlist[i]
        gaussian_window = windows.gaussian(win_length,std = win_length)
        f, t, complex_list = stft(seg_signal, fs , window = gaussian_window, nperseg = win_length)

        magnitude = np.abs(complex_list)   # 128sample / [16,128]win_len  -->  [(9,17),(65,3)]
        # new_magnitude = np.transpose(magnitude) # 转置,使x轴为频率轴
        f_length , t_length = magnitude.shape
        target_f_length = 64

        f_old = np.linspace(0, 1, f_length)  
        f_new = np.linspace(0, 1, target_f_length)  
        new_magnitude = np.zeros((target_f_length, t_length))

        for j in range(t_length):  # 时间轴不变 对 频率轴插值
            interpidFunction = interp1d(f_old, magnitude[:, j], kind='linear', fill_value="extrapolate")
            new_magnitude[:, j] = interpidFunction(f_new)  # 计算新行数对应的值 可上下采样

        magnitude_list.append(new_magnitude)

        result = np.concatenate(magnitude_list, axis=1) # (freq,times)时间轴拼接
        t_lengthx = t_lengthx + t_length

    plt.figure(figsize=(13, 8))
    plt.tick_params(axis='both', which='major', labelsize=32)
    tx = np.linspace(0, len(seg_signal)*segmentation_times / fs, t_lengthx) # 起点，终点，总点数
    fy = np.linspace(0, max(f), target_f_length)
    plt.pcolormesh(tx, fy, result, cmap='viridis', shading='gouraud')

    yticks = plt.yticks()[0]  # 读取y轴刻度值
    plt.yticks(yticks[1:-1])  # 从索引 1 开始，去掉最底部刻度(避免和x轴0刻度重复) 和最上方冗余刻度 
    # plt.title('STFT Magnitude', fontsize=22)
    # plt.ylabel('Frequency [Hz]', fontsize=22)
    # plt.xlabel('Time [s]', fontsize=22)
    # plt.colorbar(label='Magnitude')
    plt.show()


if __name__ == "__main__":

    file_paths = [
            "C:/1/振动试验台数据/断半齿/0.4负载转速1500/WithoutLossSamp0144.txt",  # 144 - 154
            "C:/1/振动试验台数据/断齿/0.4负载转速1500/WithoutLossSamp0099.txt",    # 99 - 109
            "C:/1/振动试验台数据/偏心齿轮/0.4负载转速1500/WithoutLossSamp0091.txt",# 91 - 101
            "C:/1/振动试验台数据/正常/0.4负载转速1500/WithoutLossSamp0066.txt",    # 66 - 76
            "C:/1/振动试验台数据/齿轮裂纹/0.4负载转速1500/WithoutLossSamp0073.txt",# 73 - 83
            "C:/1/振动试验台数据/齿面磨损/0.5负载转速1500/WithoutLossSamp0116.txt" # 116 - 127
            ]
 
    for datapath in file_paths:
        signalX = getdata(datapath)

        ## 设置每种信号的PSO 实验次数
        experiment_times = 10
        # 生成信号
        fs = 20480
        # t, signal = generate_signal(fs)
        win_list_all = []
        segmentation_times = 16
        n_particles = 20
        dimensions = 1  # 优化窗口长度
        bounds = [16, 127]  # 窗口长度范围
        max_iter = 30
        for i in range(experiment_times): #防止陷入局部最优，设置10组实验
        # 优化参数设置
            print(f"The {i} experiment ---------------------------------------------")
            win_list = []
            for j in range(0,segmentation_times): #分16段
                signal = signalX[j*128:(j+1)*128]
                print(f"----the {j}th segmentation ---",signal[1])
                # PSO算法
                pso = WVD_PSO(n_particles, dimensions, bounds, max_iter)
                Ideal = ideal_wvd_fitness(signal)
                best_params, best_score = pso.optimize(wvd_fitness, signal, fs, Ideal)
                win_list.append(int(best_params))
                print(f"Optimized window length: {int(best_params)} samples")
                print(f"Best fitness score: {best_score:.4f}")
            win_list_all.append(win_list)
        mode_values, counts = mode(win_list_all, axis=0) #计算每一列众数
        for i in range(experiment_times):    
            print(f"the NO.{i} experiments:{win_list_all[i]}")  
    # missing tooth
    # [21, 28, 50, 22, 82, 118, 16, 18, 19, 109, 116, 16, 76, 16, 18, 104]
    # [21, 28, 50, 22, 82, 118, 16, 18, 19, 109, 43, 16, 76, 16, 18, 104]
    # [21, 28, 50, 22, 82, 118, 16, 18, 19, 109, 43, 16, 76, 16, 18, 104]
    # [21, 28, 50, 22, 82, 118, 16, 18, 19, 109, 43, 16, 76, 16, 18, 104]
    # [21, 28, 50, 22, 16, 118, 16, 18, 19, 109, 43, 16, 76, 16, 18, 104]
    # [21, 28, 50, 22, 82, 118, 16, 18, 19, 109, 43, 16, 76, 16, 128, 104]
    # [21, 28, 50, 22, 16, 118, 16, 18, 19, 109, 43, 16, 76, 16, 18, 104]
    # [21, 28, 50, 22, 16, 118, 16, 18, 19, 109, 43, 16, 76, 16, 128, 104]
    # [21, 99, 50, 22, 82, 118, 16, 18, 19, 109, 43, 16, 76, 16, 18, 104]
    # [21, 99, 50, 22, 82, 118, 16, 18, 19, 109, 43, 16, 76, 16, 18, 104]

        best_win_list = mode_values
        print(f"adaptive windows:{best_win_list}")
        draw_spectrogram(best_win_list,signalX,fs,segmentation_times)

