# 互联网新闻情感分析-基于Transformers2.0-Bert

## 1 赛题简介
本项目基于Transformer2.0库中的中文Bert模型，对新闻语料进行三分类。具体赛题信息见本人另一个项目[“互联网新闻情感分析”](https://github.com/DefuLi/Emotional-Analysis-of-Internet-News)<br>

## 2 项目结构
项目文件夹共包括以下文件及文件夹：<br>
main.py 主程序，里面包括自定义生成Dataset子类，BertConfig的实例化，BertTokenizer类的实例化以及BertForSequenceClassification类的实例化。<br>
data 文件夹中包括了训练集和测试集，具体格式为id, text, label.<br>

## 3 main程序运行流程
<b>加载数据集</b>：对每一个文本text进行\[CLS]+text+\[SEP]+\[PAD]拼接操作
```python
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
```

<b>生成Dataset子类</b>：通过继承Dataset类，自定义读取数据集的Custom_dataset子类
```python
class Custom_dataset(Dataset):
    def __init__(self, dataset_list):
        self.dataset = dataset_list

    def __getitem__(self, item):
        text = self.dataset[item][1]
        label = self.dataset[item][2]

        return text, label

    def __len__(self):
        return len(self.dataset)
```

<b>实例化Bert相关类，以及实例化优化器等</b>：BertConfig、BertForSequenceClassification、AdamW、WarmupLinearSchedule的实例化
```python
    config = BertConfig.from_pretrained('bert-base-chinese')
    config.num_labels = 3
    model = BertForSequenceClassification(config = config)
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', config=config)
    model.cuda()
    
    optimizer = AdamW(model.parameters(), lr=LR, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps = WARMUP_STEPS, t_total = T_TOTAL)
```

<b>模型的训练与测试</b>：对模型进行了8个epoch左右的训练，最后测试时F1值在0.78左右，较之于微调前的模型提高不少
```python
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
```

## 4 注意事项
本程序使用了PyTorch的[transformers2.0库](https://huggingface.co/transformers/index.html)，请先安装该库，然后进行微调训练模型。<br>
本程序使用了GPU加速，如不需要，请在main.py文件中删除相关源代码。<br>
本程序的超参数设置如下
```python
EPOCHS = 10  # 训练的轮数
BATCH_SIZE = 10  # 批大小
MAX_LEN = 300  # 文本最大长度
LR = 1e-5  # 学习率
WARMUP_STEPS = 100  # 热身步骤
T_TOTAL = 1000  # 总步骤
```
