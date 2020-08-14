# BERT-PreTrain

按以下步骤执行：
- 将 pre_train_data 文件放在 pre-train 目录下。(原文件太大，只上传了部分文件)
- 执行 train-pre.sh ，生成 tf_pre_train.tfrecord 文件
- 执行 train.sh ，即可开始预训练模型训练

requirements:
'''
tensorflow >= 1.11.0   # CPU Version of TensorFlow.
tensorflow-gpu  >= 1.11.0  # GPU version of TensorFlow.
'''
