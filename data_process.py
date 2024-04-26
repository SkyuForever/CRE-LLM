import json
import re
import random

def data_split(full_list, ratio, shuffle=False):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


# 从TXT文件中读取数据
with open('data/FinRE/train.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

data = []

# 创建一个字典以存储相同的输入句子对应的输出
output_dict = {}

def bracket_entities(text, entities):
    for entity in entities:
        text = re.sub(re.escape(entity), f'[{entity}]', text)
    return text

# 解析每一行数据
for line in lines:
    parts = line.strip().split('\t')
    if len(parts) == 4:
        instruction = "请根据给定句子和实体抽取其关系"
        input_text = parts[3]
        entities=[parts[0],parts[1]]
        entity_relation = f'([{parts[0]}],?,[{parts[1]}])'
        input_text = bracket_entities(input_text, entities)
        input_text=input_text+' '+entity_relation
        # output = f'([{parts[0]}],{parts[2]},[{parts[1]}])'
        output = f'{parts[2]}'
        
        # 如果input_text已经存在于字典中，将output合并
        if input_text in output_dict:
            output_dict[input_text] = output_dict[input_text] + ',' + output
        else:
            output_dict[input_text] = output
    
# 创建JSON格式的数据
for input_text, output in output_dict.items():
    item = {
        "instruction": instruction,
        "input": f'{input_text}',
        "output": output
    }
    data.append(item)

# 将数据转化为JSON格式
json_data = json.dumps(data, ensure_ascii=False, indent=2)
# # 将JSON数据写入文件
with open('data/FinRE/train.json', 'w', encoding='utf-8') as output_file:
     output_file.write(json_data)
