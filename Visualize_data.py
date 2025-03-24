from Operate_RawData import Operate_RawData
import numpy as np
from Draw_function import Draw_diagram

txtdata_path0 = "C:/1/振动试验台数据/断半齿/0.4负载转速1500/WithoutLossSamp0144.txt"# 144 - 154
txtdata_path1 = "C:/1/振动试验台数据/断齿/0.4负载转速1500/WithoutLossSamp0099.txt"# 99 - 109
txtdata_path2 = "C:/1/振动试验台数据/偏心齿轮/0.4负载转速1500/WithoutLossSamp0091.txt"# 91 - 101
txtdata_path3 = "C:/1/振动试验台数据/正常/0.4负载转速1500/WithoutLossSamp0066.txt"# 66 - 76
txtdata_path4 = "C:/1/振动试验台数据/齿轮裂纹/0.4负载转速1500/WithoutLossSamp0073.txt"# 73 - 83
txtdata_path5 = "C:/1/振动试验台数据/齿面磨损/0.5负载转速1500/WithoutLossSamp0116.txt"# 116 - 127

def main():
    lengthx = 2048
    T = 1/20480 
    fs = 20480          # 采样频率
    nperseg = 32        # stft窗口
    noverlap = nperseg/2       # stft窗口重叠
    levels = 6 # DTCWT 分解层数

    file_paths = [
        "C:/1/振动试验台数据/断半齿/0.4负载转速1500/WithoutLossSamp0144.txt",  # 144 - 154
        "C:/1/振动试验台数据/断齿/0.4负载转速1500/WithoutLossSamp0099.txt",    # 99 - 109
        "C:/1/振动试验台数据/偏心齿轮/0.4负载转速1500/WithoutLossSamp0091.txt",# 91 - 101
        "C:/1/振动试验台数据/正常/0.4负载转速1500/WithoutLossSamp0066.txt",    # 66 - 76
        "C:/1/振动试验台数据/齿轮裂纹/0.4负载转速1500/WithoutLossSamp0073.txt",# 73 - 83
        "C:/1/振动试验台数据/齿面磨损/0.5负载转速1500/WithoutLossSamp0116.txt" # 116 - 127
        ]
 
    for datapath in file_paths:
        Raw_Data = Operate_RawData(   
            txtdata_path = datapath,
            channel = 3,
            start_idx = 1000, 
            length = lengthx
            )
        raw_data = Raw_Data.get_raw_data()
        data_ndarry = np.array(raw_data)

        ### 归一化
        # min_val = np.min(data_ndarry)
        # max_val = np.max(data_ndarry)
        # normalized_data = (data_ndarry - min_val) / (max_val - min_val)      # [0,1]
        # normalized_data =normalized_data * 2 - 1                             # [-1,1]
        # pic = Draw_diagram(normalized_data,lengthx,T,fs,nperseg,noverlap)

        pic = Draw_diagram(data_ndarry,lengthx,T,fs,nperseg,noverlap,levels)

        # pic.draw_2D_timeDomain()
        
        # pic.draw_2D_fft()

        # pic.draw_2D_stft()

        # pic.draw_3D_stft()

        # pic.draw_2D_cwt()

        # pic.draw_3D_cwt()

        # pic.DTCWT_re_construct()

        pic.Draw_2D_DTCWT()

if __name__ == "__main__":
    main()


