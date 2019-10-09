import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, AdamW, WarmupLinearSchedule

import csv
import json
import numpy as np
import pandas as pd


# 超参数
EPOCHS = 10  # 训练的轮数
BATCH_SIZE = 10  # 批大小
MAX_LEN = 300  # 文本最大长度
LR = 1e-5  # 学习率
WARMUP_STEPS = 100  # 热身步骤
T_TOTAL = 1000  # 总步骤

# pytorch的dataset类 重写getitem,len方法
class Custom_dataset(Dataset):
    def __init__(self, dataset_list):
        self.dataset = dataset_list

    def __getitem__(self, item):
        text = self.dataset[item][1]
        label = self.dataset[item][2]

        return text, label

    def __len__(self):
        return len(self.dataset)


# 加载数据集
def load_dataset(filepath, max_len):
    dataset_list = []
    f = open(filepath, 'r', encoding='utf-8')
    r = csv.reader(f)
    for item in r:
        if r.line_num == 1:
            continue
        dataset_list.append(item)
    
    # 根据max_len参数进行padding
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    for item in dataset_list:
        item[1] = item[1].replace(' ','')
        num = max_len - len(item[1])
        if num < 0:
            item[1] = item[1][:max_len]
            item[1] = tokenizer.encode(item[1]) 
            num_temp = max_len - len(item[1])
            if num_temp > 0:
                for _ in range(num_temp):
                    item[1].append(0)
            # 在开头和结尾加[CLS] [SEP]
            item[1] = [101] + item[1] + [102]
            item[1] = str(item[1])
            continue

        for _ in range(num):
            item[1] = item[1] + '[PAD]'
        item[1] = tokenizer.encode(item[1])
        num_temp = max_len - len(item[1])
        if num_temp > 0:
            for _ in range(num_temp):
                item[1].append(0)
        item[1] = [101] + item[1] + [102]
        item[1] = str(item[1])

    return dataset_list


# 计算每个batch的准确率
def  batch_accuracy(pre, label):
    pre = pre.argmax(dim=1)
    correct = torch.eq(pre, label).sum().float().item()
    accuracy = correct / float(len(label))

    return accuracy


if __name__ == "__main__":

    # 生成数据集以及迭代器
    train_dataset = load_dataset('data/Train.csv', max_len = MAX_LEN)  # 7337 * 3
    test_dataset = load_dataset('data/Test.csv', max_len = MAX_LEN)  # 7356 * 3
  
    train_cus = Custom_dataset(train_dataset)
    train_loader = DataLoader(dataset=train_cus, batch_size=BATCH_SIZE, shuffle=False)

    # Bert模型以及相关配置
    config = BertConfig.from_pretrained('bert-base-chinese')
    config.num_labels = 3
    model = BertForSequenceClassification(config = config)
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', config=config)
    model.cuda()

 
    optimizer = AdamW(model.parameters(), lr=LR, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps = WARMUP_STEPS, t_total = T_TOTAL)

    # optimizer = optim.Adam(model.parameters(), lr=LR)

    model.train()
    print('开始训练...')
    for epoch in range(EPOCHS):
        for text, label in train_loader:
            text_list = list(map(json.loads, text))
            label_list = list(map(json.loads, label))
            
            text_tensor = torch.tensor(text_list).cuda()
            label_tensor = torch.tensor(label_list).cuda()

            outputs = model(text_tensor, labels=label_tensor)
            loss, logits = outputs[:2]
            optimizer.zero_grad()
            loss.backward()
            scheduler.step()
            optimizer.step()

            acc = batch_accuracy(logits, label_tensor)
            print('epoch:{} | acc:{} | loss:{}'.format(epoch, acc, loss))

    torch.save(model.state_dict(), 'bert_cla.ckpt')
    print('保存训练完成的model...')


    # 测试
    
    print('开始加载训练完成的model...')
    model.load_state_dict(torch.load('bert_cla.ckpt'))

    print('开始测试...')
    model.eval()
    test_result = []
    for item in test_dataset:

        text_list = list(json.loads(item[1]))
        text_tensor = torch.tensor(text_list).unsqueeze(0).cuda()

        with torch.no_grad():

            # print('list', text_list)
            # print('tensor', text_tensor)
            # print('tensor.shape', text_tensor.shape)
            outputs = model(text_tensor, labels=None)

            print(outputs[0])
            
            pre = outputs[0].argmax(dim=1)
            test_result.append([item[0], pre.item()])

    # 写入csv文件
    df = pd.DataFrame(test_result)
    df.to_csv('test_result.csv',index=False, header=['id', 'label'])

    print('测试完成，快提交结果吧')


