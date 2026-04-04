import torch.nn as nn
import torch.nn.functional as F
from pgd import LinfPGD, L2PGD, FGSM, CWAttack, MSDAttack
from occlusion_attack import OcclusionAttack, AdaptiveOcclusionAttack


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


class AdaptiveOcclusionAdversarialTraining(nn.Module):
    def __init__(self, model, N=5, R=3, c=0.0, is_at=False):
        super(AdaptiveOcclusionAdversarialTraining, self).__init__()
        self.model = model
        self.adversary = AdaptiveOcclusionAttack(self.model, N=N, R=R, c=c)
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


# ========== 新增：L2 PGD 对抗性训练 ==========
class L2PGDAdversarialTraining(nn.Module):
    """L2范数PGD对抗性训练"""

    def __init__(self, model, eps=1.0, step=10, step_size=0.1, random_start=True, criterion=F.cross_entropy, is_at=False):
        super(L2PGDAdversarialTraining, self).__init__()
        self.model = model
        self.adversary = L2PGD(self.model,
                               eps=eps,
                               step_size=step_size,
                               step=step,
                               random_start=random_start,
                               criterion=criterion)
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


# ========== 新增：FGSM 对抗性训练 ==========
class FGSMAdversarialTraining(nn.Module):
    """FGSM对抗性训练（单步攻击，速度快）"""

    def __init__(self, model, eps=8/255, criterion=F.cross_entropy, is_at=False):
        super(FGSMAdversarialTraining, self).__init__()
        self.model = model
        self.adversary = FGSM(self.model, eps=eps, criterion=criterion)
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


# ========== 新增：CW 对抗性训练 ==========
class CWAdversarialTraining(nn.Module):
    """Carlini-Wagner对抗性训练"""

    def __init__(self, model, eps=8/255, step=10, step_size=2/255,
                 confidence=50, num_classes=10, random_start=True, is_at=False):
        super(CWAdversarialTraining, self).__init__()
        self.model = model
        self.adversary = CWAttack(self.model,
                                  eps=eps,
                                  step_size=step_size,
                                  step=step,
                                  confidence=confidence,
                                  num_classes=num_classes,
                                  random_start=random_start)
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


# ========== 新增：MSD 对抗性训练 ==========
class MSDAdversarialTraining(nn.Module):
    """Multi-Scale Diversity对抗性训练（同时使用L2和L∞攻击）"""

    def __init__(self, model, epsilon_l_inf=8/255, epsilon_l_2=1.0,
                 alpha_l_inf=2/255, alpha_l_2=0.1, num_iter=10, is_at=False):
        super(MSDAdversarialTraining, self).__init__()
        self.model = model
        self.adversary = MSDAttack(self.model,
                                   epsilon_l_inf=epsilon_l_inf,
                                   epsilon_l_2=epsilon_l_2,
                                   alpha_l_inf=alpha_l_inf,
                                   alpha_l_2=alpha_l_2,
                                   num_iter=num_iter)
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
