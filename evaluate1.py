import json

# 从JSON文件中读取数据
with open('FinRE_test1/generated_predictions.jsonl', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

# 初始化列表以存储每个示例的 Precision、Recall 和 F1 值
precisions = []
recalls = []
f1_values = []
# 遍历每个示例
for example in data:
    label = example["label"]
    predict = example["predict"]
    label_relations=[]
    predict_relations=[]
    start_index = 0
    for i in range(1, 9):
        left_bracket_index = label.find('(', start_index)
        right_bracket_index = label.find(')', left_bracket_index) if left_bracket_index != -1 else -1
        if left_bracket_index != -1 and right_bracket_index != -1:
            left=label.find(',',left_bracket_index)
            right=label.find(',',left+1)
            relation=label[left+1:right]
            label_relations.append(relation)
            
            start_index = right_bracket_index + 1 if right_bracket_index != -1 else start_index
        else:break
    start_index = 0
    for i in range(1, 9):
        left_bracket_index = predict.find('(', start_index)
        right_bracket_index = predict.find(')', left_bracket_index) if left_bracket_index != -1 else -1
        if left_bracket_index != -1 and right_bracket_index != -1:
            # 偶数组左括号 '[' 的索引位置
            left=predict.find(',',left_bracket_index)
            relation=predict[left+1:right]
            predict_relations.append(relation)

            # 将下一轮查找的起始位置设为 ']'
            start_index = right_bracket_index + 1 if right_bracket_index != -1 else start_index
        else:break
    
    # 计算 Precision 和 Recall
    tp = len(set(label_relations).intersection(predict_relations))
    fp = len(set(predict_relations).difference(label_relations))
    fn = len(set(label_relations).difference(predict_relations))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    # 计算 F1 值
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    # 将 Precision、Recall 和 F1 值添加到列表中
    precisions.append(precision)
    recalls.append(recall)
    f1_values.append(f1)

# 计算平均 Precision、Recall 和 F1 值
average_precision = sum(precisions) / len(precisions)
average_recall = sum(recalls) / len(recalls)
average_f1 = sum(f1_values) / len(f1_values)
print("Average Precision: {:.4f}".format(average_precision))
print("Average Recall: {:.4f}".format(average_recall))
print("Average F1: {:.4f}".format(average_f1))
