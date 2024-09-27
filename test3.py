import os
import torch.distributed.rpc as rpc
import torch.nn as nn
from torch.distributed.rpc import RRef
from pytorch_pretrained import BertShard_3
import threading
from gpu_mem_track import MemTracker
import torch
os.environ['MASTER_ADDR'] = '10.129.112.170'
os.environ['MASTER_PORT'] = '7856'
os.environ['TP_SOCKET_IFNAME'] = 'eth0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

#gpu_tracker = MemTracker()


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
        print("Model3 has been saved in : THUCNews/saved_dict/bert_3.ckpt")

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


if __name__ == '__main__':
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128, rpc_timeout=300)
    rpc.init_rpc("worker3", rank=2, world_size=3, rpc_backend_options=options)
    print("worker3 rpc init finish")
    rpc.shutdown()