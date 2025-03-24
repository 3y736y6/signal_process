
class Operate_RawData:
    '''
    specify the [channel,start_idx,length],get data fragment

    parameters
    ---
    - txtdata_path: the path of the txt data
    - channel: the sample_channel of the data
    - start_idx: the start index of the data
    - length: the length of the data
    '''
    def __init__(self,txtdata_path,channel,start_idx,length):
        self.data_path = txtdata_path
        self.data_list = []

        with open(self.data_path, 'r') as file:
            for i in range(0,start_idx + length):
                line = file.readline().split() # one row has 5 number
                self.data_list.append(float(line[channel-1]))
            self.data_list = self.data_list[start_idx:start_idx+length]
    def get_raw_data_example(self,idx):
        return self.data_list[idx]
    def get_raw_data(self):
        return self.data_list
    
