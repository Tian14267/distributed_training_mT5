import random
import json
from loguru import logger
import pandas as pd

# 准备你自己的数据集，返回为train data list以及valid data list。
# 数据格式：[{"source": source_query， "target": target_query}, ...]

def prepare_data():
    # example
    train_data_list = [{"source": "你今天好吗", "target": "挺好的"}]
    valid_data_list = [{"source": "你好吗", "target": "还行"}]

    return train_data_list, valid_data_list


def generate_prompt(instruction, input=None, output=None):
    if input:
        out = f"""用户:{instruction}。{input}### 小催:{output}"""
        return out
    else:
        out = f"""用户：:{instruction}### 小催:{output}"""
        return out


def generate_prompt_2(instruction, input=None):
    if input:
        out = f"""用户:{instruction}。{input}### 小催:"""
        return out
    else:
        out = f"""用户：:{instruction}### 小催:"""
        return out



def prepare_data_new(data_file,max_source_text_len=64 ,max_target_text_len=512,if_shuffle=True):
    with open(data_file,"r") as f:
        lines = f.readlines()
        #lines = lines[:10000]  ####   截取数据
    f.close()

    all_data_list = []

    for one_line in lines:
        line_json = json.loads(one_line.strip())
        #if len(str(line_json["instruction"])) <max_source_text_len and len(str(line_json["output"])) <max_target_text_len:
        prompt_out = generate_prompt_2(instruction=line_json["instruction"])
        new_json = {"source": prompt_out, "target": line_json["output"]}
        all_data_list.append(new_json)
    print("#########   数据总量：",len(all_data_list))
    #exit()
    if if_shuffle:
        random.shuffle(all_data_list)

    train_data_rate = 0.9

    train_num = int(len(all_data_list)*train_data_rate)
    train_data_list = all_data_list[:train_num]
    valid_data_list = all_data_list[train_num:]

    return train_data_list,valid_data_list

if __name__ == '__main__':
    prepare_data_new(data_file="/data1/fffan/5_NLP/3_ChatYuan/data/cuishou_train_file_vicuna.json")