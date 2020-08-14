# BERT-PreTrain

## 数据说明
- data 目录下为最终要训练的数据和测试数据
格式为两列：保准问ID 和 用户问句 ，中间使用 \t 隔开

label      | txt 
---------- | :-------------:
1014       | 190 26 6 7 154 41 6 7 17 117 8 43 40 153 313
364        | 0 43 40 60 63 139 44 211 26

- pre-train 目录下为预训练使用数据，包含文件有：
  - pre_train_data        预训练语料
  - bert_config.json      bert模型配置
  - vocab                 word字典文件

## 执行步骤
- 1.执行 pre-train-1.sh ，生成 tf_pre_train.tfrecord 文件
- 2.执行 pre-train-2.sh ，即可开始预训练模型训练

```
训练一天最终预训练效果为：
global_step = 200000
loss = 1.3791736
masked_lm_accuracy = 0.744027
masked_lm_loss = 1.152022
next_sentence_accuracy = 0.915125
next_sentence_loss = 0.22917493
```
loss 图为：
![image](https://github.com/syzong/images/blob/master/58_pre_train_loss.png)


- 3.执行 train.sh 开始模型训练
- 4.执行 test.sh 测试模型最终效果，生成的 sub.csv 文件去除第三列概率值即为提交文件

requirements:
```
tensorflow >= 1.11.0   # CPU Version of TensorFlow.
tensorflow-gpu  >= 1.11.0  # GPU version of TensorFlow.
```
