import torch
from utils.Utils import OCEMOTION, OCNLI, TNEWS


class PGD():
    def __init__(self, model):
        self.model = model
        self.config = model.config
        self.emb_backup = {}
        self.grad_backup = {}
        self.epsilon = self.config.pgd_epsilon
        self.alpha = self.config.pgd_alpha
        self.embedding_name = self.config.pgd_embedding_name
        self.k = self.config.pgd_k

    def attack(self,is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.embedding_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.embedding_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        exception = ['bert.pooler.dense.weight', 'bert.pooler.dense.bias']
        for name, param in self.model.named_parameters():
            if param.requires_grad and name not in exception:
                if param.grad is None:
                    print(name)
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        exception = ['bert.pooler.dense.weight', 'bert.pooler.dense.bias']
        for name, param in self.model.named_parameters():
            if param.requires_grad and name not in exception:
                param.grad = self.grad_backup[name]
                
    def train(self, **kwargs):
        ocemotion_batch = kwargs['ocemotion_batch']
        ocnli_batch = kwargs['ocnli_batch']
        tnews_batch = kwargs['tnews_batch']
        self.backup_grad()
        for t in range(self.k):
            self.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
            if t != self.k-1:
                self.model.zero_grad()
            else:
                self.restore_grad()
            _, ocemotion_loss_adv = self.model(*ocemotion_batch, OCEMOTION)
            _, ocnli_loss_adv = self.model(*ocnli_batch, OCNLI)
            _, tnews_loss_adv = self.model(*tnews_batch, TNEWS)
            ocemotion_loss_adv.backward()
            ocnli_loss_adv.backward()
            tnews_loss_adv.backward()
        self.restore()
