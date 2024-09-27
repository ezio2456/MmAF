# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer, BertShard_1, BertShard_2
import threading


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集
        self.benchmark_path = dataset + '/data/benchmark.txt'
        # 去除字符串两边的空格，生成类别列表
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.require_improvement = 600  # 若超过600batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 2  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # 预训练部分
        self.bert = BertModel.from_pretrained(config.bert_path, shard_flag='full')
        for param in self.bert.parameters():
            # print(param)
            param.requires_grad = True
        # 微调部分
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子  --> (token_ids, .....)
        print(context.shape)  # torch.size([128, 32]), 如果要设置微批应该对x[0]进行切分
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out

    def freeze_layer(self, freeze_layer):
        for n, p in self.named_parameters():
            n_split = n.split('.')
            if freeze_layer in n_split:
                p.requires_grad = False


"""
class Model1(nn.Module):

    def __init__(self, config):
        super(Model1, self).__init__()
        # 模型预训练部分
        config.model_name = 'bert_1'
        self.name = 'bert_1'
        self.bert_1 = BertShard_1.from_pretrained(config.bert_path, shard_flag=1)
        for param in self.bert_1.parameters():
            # print(param)
            param.requires_grad = True

    def forward(self, x):
        context = x[0]
        mask = x[2]
        encoder_layers, extended_attention_mask = self.bert_1(context, attention_mask=mask, output_all_encoded_layers=False)

        return encoder_layers, extended_attention_mask


class Model2(nn.Module):

    def __init__(self, config):
        super(Model2, self).__init__()
        # 模型预训练部分
        config.model_name = 'bert_2'
        self.name = 'bert_2'
        self.bert_2 = BertShard_2.from_pretrained(config.bert_path, shard_flag=2)
        for param in self.bert_2.parameters():
            # print(param)
            param.requires_grad = True
        print(self.bert_2.parameter_rref())
        # 微调部分
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, encoder_layers, extended_attention_mask):
        pooled = self.bert_2(encoder_layers, extended_attention_mask)
        out = self.fc(pooled)

        return out

"""


# class Model1(nn.Module):
#
#     def __init__(self, config):
#         super(Model1, self).__init__()
#         # 模型预训练部分
#         config.model_name = 'bert_1'
#         self.config = config
#         self.name = 'bert_1'
#         self._lock = threading.Lock()
#         self.bert_1 = BertShard_1.from_pretrained(config.bert_path, shard_flag=1)
#         for param in self.bert_1.parameters():
#             # print(param)
#             param.requires_grad = True
#
#     def forward(self, context_rref, mask_rref):
#         # context = x[0]
#         # mask = x[2]
#         context = context_rref.to_here().to(self.config.device)
#         mask = mask_rref.to_here().to(self.config.device)
#         with self._lock:
#             encoder_layers, extended_attention_mask = self.bert_1(context, attention_mask=mask,
#                                                                   output_all_encoded_layers=False)
#
#         return encoder_layers.cpu(), extended_attention_mask.cpu()
#
#     def parameter_rrefs(self):
#         return [RRef(p) for p in self.parameters()]
#
#
# class Model2(nn.Module):
#
#     def __init__(self, config):
#         super(Model2, self).__init__()
#         # 模型预训练部分
#         config.model_name = 'bert_2'
#         self.config = config
#         self._lock = threading.Lock()
#         self.name = 'bert_2'
#         self.bert_2 = BertShard_2.from_pretrained(config.bert_path, shard_flag=2)
#         for param in self.bert_2.parameters():
#             # print(param)
#             param.requires_grad = True
#         # print(self.bert_2.parameter_rref())
#         # 微调部分
#         self.fc = nn.Linear(config.hidden_size, config.num_classes)
#
#     def forward(self, encoder_layers_rref, extended_attention_mask_rref):
#         encoder_layers = encoder_layers_rref.to_here().to(self.config.device)
#         extended_attention_mask = extended_attention_mask_rref.to_here().to(self.config.device)
#         with self._lock:
#             pooled = self.bert_2(encoder_layers, extended_attention_mask)
#             out = self.fc(pooled)
#
#         return out.cpu()
#
#     def parameter_rrefs(self):
#         return [RRef(p) for p in self.parameters()]
