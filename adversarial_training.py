import torch.nn as nn
import torch.nn.functional as F
from pgd import LinfPGD
from occlusion_attack import OcclusionAttack


class AdversarialTraining(nn.Module):
    def __init__(self, model, eps=0.1, step=5, step_size=0.025, random_start=True, criterion=F.cross_entropy, is_at=False):
        super(AdversarialTraining, self).__init__()
        self.model = model
        self.adversary = LinfPGD(self.model, 
                                 eps=eps, 
                                 step_size=step_size, 
                                 step=step, 
                                 random_start=random_start, 
                                 criterion=criterion)
        self.is_at = is_at
    
    def forward(self, x, y=None):
        if self.is_at:
            # 记录模型状态
            training = self.model.training

            assert y is not None
            
            # 在创造对抗性样本时，模型开启测试模式
            self.model.eval()
            x_adv = self.adversary((x, y))
            
            # 如果在训练过程中，需要将模型转化为训练模式；否则保持测试模式
            if training:
                self.model.train()
            return self.model(x_adv)
        else:
            return self.model(x)


class OcclusionAdversarialTraining(nn.Module):
    def __init__(self, model, top_k=9, occlu_color=0.0, kernel_size=3, is_at=False):
        super(OcclusionAdversarialTraining, self).__init__()
        self.model = model
        self.adversary = OcclusionAttack(self.model,
                                         top_k=top_k,
                                         occlu_color=occlu_color,
                                         kernel_size=kernel_size)
        self.is_at = is_at

    def forward(self, x, y=None):
        if self.is_at:
            training = self.model.training

            assert y is not None

            self.model.eval()
            x_adv = self.adversary((x, y))

            if training:
                self.model.train()
            return self.model(x_adv)
        else:
            return self.model(x)
