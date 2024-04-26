import json
import re
from simcse import SimCSE
model = SimCSE("sup-simcse-roberta-large")
# 读取txt文件
with open('data/FinRE/relation2id.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 提取每行中的单词（关系）
extracted_relations = []

for line in lines:
    # 分割每行数据
    parts = line.strip().split()

    # 检查是否有足够的部分
    if len(parts) >= 1:
        relation = parts[0]
        extracted_relations.append(relation)


with open('FinRE_test/generated_predictions.jsonl', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

id = 0
# 遍历每个示例
for example in data:
    predict = example["predict"]
    predict_relations = []
    start_index = 0
    for i in range(1, 9):
        left_bracket_index = predict.find('[', start_index)
        right_bracket_index = predict.find(
            ']', left_bracket_index) if left_bracket_index != -1 else -1
        if left_bracket_index != -1 and right_bracket_index != -1:
            if i % 2 == 0:
                # 第偶数组左括号 '[' 的索引位置
                left = left_bracket_index
                relation = predict[right+2:left-1]
                predict_relations.append(relation)
            else:
                # 第奇数组右括号 ']' 的索引位置
                right = right_bracket_index

            # 将下一轮查找的起始位置设为 ']'
            start_index = right_bracket_index + 1 if right_bracket_index != -1 else start_index
        else:
            break

    for relation in predict_relations:
        if relation not in extracted_relations:
            model.build_index(extracted_relations)
            result = model.search(relation)
            print("{}:{}-{}".format(id, relation, result))
    id += 1
