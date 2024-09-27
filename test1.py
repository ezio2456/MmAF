# coding: UTF-8
import time
import os
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
from torch.distributed.optim import DistributedOptimizer
from pytorch_pretrained.optimization import BertAdam
import torch.nn as nn
import torch.distributed.autograd as dist_autograd
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from pytorch_pretrained import BertShard_1, BertShard_2, BertShard_3
import threading
from gpu_mem_track import MemTracker

os.environ['MASTER_ADDR'] = '10.129.112.170'
os.environ['MASTER_PORT'] = '7856'
os.environ['GLOO_SOCKET_IFNAME'] = 'wlx0013ef4f5ec5'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()
gpu_tracker = MemTracker()


class Model1(nn.Module):

    def __init__(self, config):
        super(Model1, self).__init__()
        # 模型预训练部分
        config.model_name = 'bert_1'
        self.config = config
        self.name = 'bert_1'
        self.bert_1 = BertShard_1.from_pretrained(config.bert_path, shard_flag=1).to("cuda")
        for param in self.bert_1.parameters():
            # print(param)
            param.requires_grad = True

    def forward(self, data):
        context, mask = data
        encoder_layers, extended_attention_mask = self.bert_1(context, attention_mask=mask,
                                                              output_all_encoded_layers=False)

        return encoder_layers[-1].cpu(), extended_attention_mask.cpu()

    def save_model(self):
        torch.save(self.state_dict(), 'THUCNews/saved_dict/bert_1.ckpt')
        print("Model has been saved in : THUCNews/saved_dict/bert_1.ckpt")

    def load_model(self):
        self.load_state_dict(torch.load('THUCNews/saved_dict/bert_1.ckpt'))
        self.eval()
        print("Model1 state dict has been loaded succ...")

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]

    def freeze_param_rrefs(self):
        return [RRef(p) for p in self.parameters() if p.requires_grad]

    def freeze_layer(self, freeze):
        for n, p in self.named_parameters():
            n_split = n.split(".")
            if freeze in n_split:
                p.requires_grad = False


class Model2(nn.Module):

    def __init__(self, config):
        super(Model2, self).__init__()
        # 模型预训练部分
        config.model_name = 'bert_2'
        self.config = config
        self.name = 'bert_2'
        self.bert_2 = BertShard_2.from_pretrained(config.bert_path, shard_flag=2)
        for param in self.bert_2.parameters():
            param.requires_grad = True

    def forward(self, y_rref, inference=False):
        if inference:
            with torch.no_grad():
                data = y_rref.to_here()
                encoder_layers, extended_attention_mask = data
                encoder_layers = encoder_layers.to("cuda")
                extended_attention_mask = extended_attention_mask.to("cuda")
                out, mask = self.bert_2(encoder_layers, extended_attention_mask)
                return out.cpu(), mask.cpu()
        else:
            data = y_rref.to_here()
            encoder_layers, extended_attention_mask = data
            encoder_layers = encoder_layers.to("cuda")
            extended_attention_mask = extended_attention_mask.to("cuda")
            out, mask = self.bert_2(encoder_layers, extended_attention_mask)
            return out.cpu(), mask.cpu()

    def save_model(self):
        torch.save(self.state_dict(), 'THUCNews/saved_dict/bert_2.ckpt')
        print("Model has been saved in : THUCNews/saved_dict/bert_2.ckpt")

    def load_model(self):
        self.load_state_dict(torch.load('THUCNews/saved_dict/bert_2.ckpt'))
        self.eval()
        print("Model2 state dict has been loaded succ...")

    def freeze_layer(self, freeze):
        for n, p in self.named_parameters():
            n_split = n.split(".")
            if freeze in n_split:
                p.requires_grad = False

    def comp_grad(self, freeze_layer):
        temp = 0
        for name, param in self.named_parameters():
            name_split = name.split(".")
            if freeze_layer in name_split:
                temp += torch.norm(param.grad.data.clone(), p=2).item()
        return temp

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]

    def freeze_param_rrefs(self):
        return [RRef(p) for p in self.parameters() if p.requires_grad]


class Model3(nn.Module):
    def __init__(self, config):
        super(Model3, self).__init__()
        self.config = config
        self.bert_3 = BertShard_3.from_pretrained(config.bert_path, shard_flag=3)
        for param in self.bert_3.parameters():
            param.requires_grad = True
        # 微调部分
        self.fc = nn.Linear(config.hidden_size, config.num_classes).to("cuda")

    def forward(self, y_rref, inference=False):
        if inference:
            with torch.no_grad():
                data = y_rref.to_here()
                encoder_layers, extended_attention_mask = data
                encoder_layers = encoder_layers.to("cuda")
                extended_attention_mask = extended_attention_mask.to("cuda")
                pooled = self.bert_3(encoder_layers, extended_attention_mask)
                out = self.fc(pooled)
                return out.cpu()
        else:
            data = y_rref.to_here()
            encoder_layers, extended_attention_mask = data
            encoder_layers = encoder_layers.to("cuda")
            extended_attention_mask = extended_attention_mask.to("cuda")
            pooled = self.bert_3(encoder_layers, extended_attention_mask)
            out = self.fc(pooled)
            return out.cpu()

    def save_model(self):
        torch.save(self.state_dict(), 'THUCNews/saved_dict/bert_3.ckpt')
        print("Model1 has been saved in : THUCNews/saved_dict/bert_3.ckpt")

    def load_model(self):
        self.load_state_dict(torch.load('THUCNews/saved_dict/bert_3.ckpt'))
        self.eval()
        print("Model3 state dict has been loaded succ...")

    def freeze_layer(self, freeze):
        for n, p in self.named_parameters():
            n_split = n.split(".")
            if freeze in n_split:
                p.requires_grad = False

    def comp_grad(self, freeze_layer):
        temp = 0
        for name, param in self.named_parameters():
            name_split = name.split(".")
            if freeze_layer in name_split:
                temp += torch.norm(param.grad.data.clone(), p=2).item()
        return temp

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]

    def freeze_param_rrefs(self):
        return [RRef(p) for p in self.parameters() if p.requires_grad]


# 分布式训练配置
class DistBert(nn.Module):
    def __init__(self, config, workers):
        super(DistBert, self).__init__()
        self.config = config
        self.p1_rref = Model1(self.config).to("cuda")
        self.p2_rref = rpc.remote(
            workers[1],
            Model2,
            args=(self.config,)
        )
        self.p3_rref = rpc.remote(
            workers[2],
            Model3,
            args=(self.config,)
        )

    def forward(self, context, attention_mask, inference):
        # print(context.shape, attention_mask.shape)
        encoder_layers, mask = self.p1_rref([context, attention_mask])
        # print("p2_rref start")
        y_rref = RRef([encoder_layers, mask])
        z_fut = self.p2_rref.rpc_async().forward(y_rref, inference)
        # print("DistBert finished")
        out, mask = z_fut.wait()
        z_rref = RRef([out, mask])
        final = self.p3_rref.rpc_async().forward(z_rref, inference)
        out = final.wait()
        # print(out.dtype, out.device)
        return out

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.p1_rref.parameter_rrefs())
        print("p1_rref_param finished")
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
        print("p2_rref_param finished")
        remote_params.extend(self.p3_rref.remote().parameter_rrefs().to_here())
        print("p3_rref_param finished")
        # print("p2_rref:", remote_params[-1].to_here().dtype)
        return remote_params

    def freeze_param_rrefs(self):
        remote_params = []
        remote_params.extend(self.p1_rref.freeze_param_rrefs())
        print("Model1 Freeze param fininshed")
        remote_params.extend(self.p2_rref.remote().freeze_param_rrefs().to_here())
        print("Model2 Freeze param finished")
        remote_params.extend(self.p3_rref.remote().freeze_param_rrefs().to_here())
        print("Model3 Freeze param finished")
        return remote_params


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集
    # model_name = args.model  # bert
    # 动态加载模型：models.bert
    x = import_module('models.bert')
    # 配置模型训练的相关参数
    config = x.Config(dataset)
    # 保证每次结果一样
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print("Loading data...")
    # 准备数据集
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Data load finished Time usage:", time_dif)

    # 分布式RPC配置
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128, rpc_timeout=300)
    print("start init worker1 rpc")
    rpc.init_rpc("worker1", rank=0, world_size=3, rpc_backend_options=options)
    print("worker1 rpc init finish")
    tik = time.time()
    model = DistBert(config, ["worker1", "worker2", "worker3"])
    # print("model created:", model)
    param_optimizer = model.parameter_rrefs()  # [[], [], [], ...]
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    opt = DistributedOptimizer(
        optim.Adam,
        param_optimizer,
        lr=config.learning_rate,
    )
    print("Optimizer init finished")
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    previous_grad = {}
    current_grad = {}
    init_flag = True
    grad_norm_diff = {}
    freeze_layer = "embeddings"
    freeze_worker = "1"
    stop_freeze = False
    layer_alloc_dict = {
        "1": ["embeddings", "0", "1", "2", "3", "4", "5"],
        "2": ["6", "7", "8", "9"],
        "3": ["10", "11"]
    }
    async_flag = False

    # 评估函数
    def evaluate(config, model, data_iter, test=False):
        model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            for texts, labels in data_iter:
                context = texts[0]
                mask = texts[2]
                # gpu_tracker.track()
                outputs = model(context, mask, True)
                # gpu_tracker.track()
                outputs = outputs.to("cuda")
                # gpu_tracker.track()
                loss = F.cross_entropy(outputs, labels).to("cuda")
                # gpu_tracker.track()
                loss_total += loss
                # print("loss tatal:", loss_total)
                labels = labels.data.cpu().numpy()
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predic)
        acc = metrics.accuracy_score(labels_all, predict_all)
        if test:
            report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
            confusion = metrics.confusion_matrix(labels_all, predict_all)
            return acc, loss_total / len(data_iter), report, confusion
        print("eval finished")
        return acc, loss_total / len(data_iter)

    # 测试集函数
    def test(config, Model, test_iter):
        Model.p1_rref.load_model()
        future = Model.p2_rref.rpc_async().load_model()
        future.wait()
        future_1 = Model.p3_rref.rpc_async().load_model()
        future_1.wait()
        print("All Model ckpt has been loaded ")
        start_time = time.time()
        test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
        msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
        print(msg.format(test_loss, test_acc))
        print("Precision, Recall and F1-Score...")
        print(test_report)
        print("Confusion Matrix...")
        print(test_confusion)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)


    # 训练过程
    print("Start Training...")
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # print("batch size:", total_batch)
        torch.cuda.empty_cache()
        for i, (trains, labels) in enumerate(train_iter):
            print("batch size:", total_batch)
            context = trains[0]
            mask = trains[2]
            with dist_autograd.context() as context_id:
                outputs = model(context, mask, False)
                outputs = outputs.to("cuda")
                # print("out_put finished", outputs.device)
                model.zero_grad()
                # print("model zero grad")
                loss = F.cross_entropy(outputs, labels).to("cuda")
                print("Loss:", loss)
                x = []
                x.append(loss)
                dist_autograd.backward(context_id, x)
                # 计算梯度
                if total_batch % 95 == 0 and epoch + 1 > 1 and not stop_freeze:
                    # 查看冻结层所处服务器的位置，选择是需要异步调用
                    for k, v in layer_alloc_dict.items():
                        if freeze_layer in v:
                            freeze_worker = k
                            if k == "1":
                                async_flag = False
                            else:
                                async_flag = True
                    # 计算冻结层的梯度
                    if async_flag:
                        if freeze_worker == "2":
                            grad_fut = model.p2_rref.rpc_async().comp_grad(freeze_layer)
                            grad_data = grad_fut.wait()
                            previous_grad[freeze_layer] = grad_data
                        if freeze_worker == "3":
                            grad_fut = model.p3_rref.rpc_async().comp_grad(freeze_layer)
                            grad_data = grad_fut.wait()
                            current_grad[freeze_layer] = grad_data
                    else:
                        for name, param in model.p1_rref.named_parameters():
                            name_split = name.split(".")
                            if freeze_layer in name_split:
                                if freeze_layer not in previous_grad.keys():
                                    previous_grad[freeze_layer] = torch.norm(param.data.clone(), p=2).item()
                                else:
                                    previous_grad[freeze_layer] += torch.norm(param.data.clone(), p=2).item()
                if total_batch % 100 == 0 and epoch + 1 > 1 and not stop_freeze:
                    # 计算冻结层的梯度
                    if async_flag:
                        if freeze_worker == "2":
                            grad_fut = model.p2_rref.rpc_async().comp_grad(freeze_layer)
                            grad_data = grad_fut.wait()
                            current_grad[freeze_layer] = grad_data
                        if freeze_worker == "3":
                            grad_fut = model.p3_rref.rpc_async().comp_grad(freeze_layer)
                            grad_data = grad_fut.wait()
                            current_grad[freeze_layer] = grad_data
                    else:
                        for name, param in model.p1_rref.named_parameters():
                            name_split = name.split(".")
                            if freeze_layer in name_split:
                                if freeze_layer not in previous_grad.keys():
                                    current_grad[freeze_layer] = torch.norm(param.data.clone(), p=2).item()
                                else:
                                    current_grad[freeze_layer] += torch.norm(param.data.clone(), p=2).item()
                    grad_norm_diff[freeze_layer] = abs(
                                previous_grad[freeze_layer] - current_grad[freeze_layer]) / previous_grad[freeze_layer]
                    # 清除累积梯度
                    previous_grad = {key: 0 for key in previous_grad}
                    current_grad = {key: 0 for key in current_grad}
                    for k, v in grad_norm_diff.items():
                        print(k, v)
                opt.step(context_id)
                # 进行层冻结
                if epoch + 1 > 1 and total_batch % 100 == 0 and not stop_freeze:
                    try:
                        if freeze_layer == "embeddings":
                            if grad_norm_diff[freeze_layer] < 0.01:
                                print("Start Freezing!!!!")
                                model.p1_rref.freeze_layer(freeze_layer)
                                print("model1 param has been Freezed")
                                param_optimizer = model.freeze_param_rrefs()
                                opt = DistributedOptimizer(
                                    optim.Adam,
                                    param_optimizer,
                                    lr=2e-5,
                                )
                                print("Model Freeze finished")
                                freeze_layer = "0"
                        else:
                            if grad_norm_diff[freeze_layer] < 0.01:
                                print("Start Freezing!!!!")
                                if async_flag:
                                    if freeze_worker == "2":
                                        fut = model.p2_rref.rpc_async().freeze_layer(freeze_layer)
                                        fut.wait()
                                        print("model2 param has been Freezed")
                                    if freeze_worker == "3":
                                        fut = model.p3_rref.rpc_async().freeze_layer(freeze_layer)
                                        fut.wait()
                                        print("model3 param has been Freezed")
                                    param_optimizer = model.freeze_param_rrefs()
                                    opt = DistributedOptimizer(
                                        optim.Adam,
                                        param_optimizer,
                                        lr=1e-5,
                                    )
                                    print("Model Freeze finished")
                                    freeze_layer = str(int(freeze_layer) + 1)
                                    if freeze_layer == '12':
                                        stop_freeze = True
                                else:
                                    model.p1_rref.freeze_layer(freeze_layer)
                                    param_optimizer = model.freeze_param_rrefs()
                                    opt = DistributedOptimizer(
                                        optim.Adam,
                                        param_optimizer,
                                        lr=1e-5,
                                    )
                                    print("Model Freeze finished")
                                    freeze_layer = str(int(freeze_layer) + 1)
                    except KeyError:
                        pass
                if total_batch % 100 == 0:
                    print("start dev")
                    true = labels.data.cpu()
                    predic = torch.max(outputs.data, 1)[1].cpu()
                    train_acc = metrics.accuracy_score(true, predic)
                    torch.cuda.empty_cache()
                    dev_acc, dev_loss = evaluate(config, model, dev_iter)
                    if dev_loss < dev_best_loss:
                        dev_best_loss = dev_loss
                        model.p1_rref.save_model()
                        future = model.p2_rref.rpc_async().save_model()
                        future.wait()
                        future_1 = model.p3_rref.rpc_async().save_model()
                        future_1.wait()
                        print("All model ckpt saved")
                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''
                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                    print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                    model.train()
                total_batch += 1
                # if total_batch - last_improve > config.require_improvement:
                #     # 验证集loss超过1000batch没下降，结束训练
                #     print("No optimization for a long time, auto-stopping...")
                #     flag = True
                #     break
            if flag:
                break
    tok = time.time()
    print(f"execution time = {tok - tik}")
    print("Start Test")
    test(config, model, test_iter)
    rpc.shutdown()
