from dataclasses import dataclass, field

import torch


@dataclass
class LossLogger:
    loss: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))

    def __iadd__(self, other):
        for key in self.__dict__.keys():
            if type(self.__dict__[key]) is not int:
                self.__dict__[key] += other.__dict__[key]

        return self

    def update_losses(self, *args):
        raise NotImplementedError

    def prepare_losses(self):
        raise NotImplementedError

    def detach_loss(self):
        self.loss = self.loss.item()


@dataclass
class A2CLosses(LossLogger):
    policy: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    value: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    entropy: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    temp_entropy: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))

    def prepare_losses(self):
        self.loss = self.policy.sum() + self.value.sum() + self.entropy.sum() + self.temp_entropy.sum()

        # detach items
        for key in self.__dict__.keys():
            if key not in ["loss"]:
                self.__dict__[key] = self.__dict__[key].item()


@dataclass
class ICMLosses(LossLogger):
    inverse: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    forward: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    long_horizon_forward: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))

    def prepare_losses(self):
        self.loss = self.inverse.sum() + self.forward.sum()

        # detach items
        for key in self.__dict__.keys():
            if key not in ["loss"]:
                self.__dict__[key] = self.__dict__[key].item()


@dataclass
class LocalizationLosses(LossLogger):
    x: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    y: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    c: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    s: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    c_h: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    s_h: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    corr: torch.Tensor = field(default_factory=lambda: torch.tensor([0.0, 0.0, 0.0]))

    def update_losses(self, loss_x: torch.Tensor, loss_y: torch.Tensor,
                      loss_c: torch.Tensor, loss_s: torch.Tensor,
                      loss_c_h: torch.Tensor, loss_s_h: torch.Tensor,
                      corr: torch.Tensor):

        self.x += loss_x
        self.y += loss_y
        self.c += loss_c
        self.s += loss_s
        self.c_h += loss_c_h
        self.s_h += loss_s_h
        self.corr += corr

    def cuda(self):
        for key in self.__dict__.keys():
            if type(self.__dict__[key]) is not int:
                self.__dict__[key] = self.__dict__[key].cuda()

    def div(self, other):
        for key in self.__dict__.keys():
            if type(self.__dict__[key]) is not int:
                self.__dict__[key] = self.__dict__[key] / other

    def prepare_losses(self, numSteps = 1):
        self.x /= numSteps
        self.y /= numSteps
        self.c /= numSteps
        self.s /= numSteps
        self.c_h /= numSteps
        self.s_h /= numSteps
        self.loss = self.x.sum() + self.y.sum() + self.c.sum() + self.s.sum() + self.c_h.sum() + self.s_h.sum()

        self.corr /= numSteps

        # detach items
        for key in self.__dict__.keys():
            if key not in ["loss", "corr"]:
                self.__dict__[key] = self.__dict__[key].item()

    def finalize_corr(self):
        self.corr *= 100

    def __repr__(self):
        return f"Localization Loss: {self.loss:.4f}, X: {self.x:.4f}, Y: {self.y:.4f}, " \
               f"C: {self.c:.4f}, S: {self.s:.4f}, C_h: {self.c_h:.4f}, S_h: {self.s_h:.4f}, " \
               f"Correct: [{self.corr[0].item():.2f}, {self.corr[1].item():.2f}, {self.corr[2].item():.2f}]"


@dataclass
class ReconLosses(LossLogger):
    x: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    y: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    confidence: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    binary: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    continuous: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    cls: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    num_classes: int = None
    num_thresh: int = None

    def __post_init__(self):
        if self.num_classes is None:
            raise ValueError("num_classes should not be None")
        elif self.num_classes == 1:
            self.precision = torch.tensor(0.0)
            self.recall = torch.tensor(0.0)
        else:
            self.precision = torch.zeros((self.num_classes, self.num_thresh))
            self.recall = torch.zeros((self.num_classes, self.num_thresh))
            self.totalRecall = torch.zeros((self.num_classes, self.num_thresh))
            self.Ps = torch.zeros((self.num_thresh,))
            self.Rs = torch.zeros((self.num_thresh,))
            self.TRs = torch.zeros((self.num_thresh,))
            self.APs = torch.zeros((self.num_thresh,))

    def div(self, other):
        for key in self.__dict__.keys():
            if type(self.__dict__[key]) is not int:
                self.__dict__[key] = self.__dict__[key] / other


    def cuda(self):
        for key in self.__dict__.keys():
            if type(self.__dict__[key]) is not int:
                self.__dict__[key] = self.__dict__[key].cuda()

    def update_losses(self, loss_x: torch.Tensor, loss_y: torch.Tensor, loss_confidence: torch.Tensor,
                      loss_continuous: torch.Tensor, loss_binary: torch.Tensor, loss_cls: torch.Tensor):

        self.x += loss_x
        self.y += loss_y
        self.confidence += loss_confidence
        self.continuous += loss_continuous
        self.binary += loss_binary
        self.cls += loss_cls

    def prepare_losses(self):
        self.loss = self.x + self.y + self.confidence + self.continuous + self.binary + self.cls

        # detach items
        for key in self.__dict__.keys():
            if key in ["recall", "precision", "totalRecall"]:
                self.__dict__[key] = self.__dict__[key].mean(dim=0).detach()
            elif key not in ["loss", "num_classes", "num_thresh", "APs", "Ps", "Rs", "TRs"]:
                self.__dict__[key] = self.__dict__[key].item()

    def update_stats(self, nCorrect: list, nCorrectPrec: list, nPred: int, nTotal: int, nObjs: int, idx: int):
        for i in range(self.num_thresh):
            self.precision[idx,i] = float(nCorrectPrec[i] / nPred) if nPred else 1
            self.recall[idx, i] = float(nCorrect[i] / nTotal) if nTotal else 1
            self.totalRecall[idx, i] = float(nCorrect[i] / nObjs) if nObjs else 1

    def compute_APs(self):
        self.Ps = (self.precision).mean(dim=0) * 100.0
        self.Rs = (self.recall).mean(dim=0) * 100.0
        self.TRs = (self.totalRecall).mean(dim=0) * 100
        self.APs = (self.Ps + self.Rs)/2.0

    def __repr__(self):
        return f"Reconstruction Loss: {self.loss:.4f}, X: {self.x:.4f}, Y: {self.y:.4f}, Conf: {self.confidence:.4f}," \
               f" Bin: {self.binary:.4f}, Cont: {self.continuous:.4f}, Cls: {self.cls:.4f}" \
               f" [Precs: {self.Ps[0].item():.2f}, {self.Ps[1].item():.2f}, {self.Ps[2].item():.2f}]" \
               f" [Recs: {self.Rs[0].item():.2f}, {self.Rs[1].item():.2f}, {self.Rs[2].item():.2f}]" \
               f" [TRecs: {self.TRs[0].item():.2f}, {self.TRs[1].item():.2f}, {self.TRs[2].item():.2f}]"