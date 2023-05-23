from torch import nn
from abc import ABC, abstractmethod

from .indexer import Indexer


class IndexedModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.modules = dict()

    def init(self):
        self.collect_modules(self)

    def __getitem__(self, key: str):
        if "." in key:
            key, attribute = key.split(".")
            return getattr(self.modules[key].operator, attribute)
        return self.modules[key]

    # def __setitem__(self, key: str, value):
    #     if "." in key:
    #         key, attribute = key.split(".")
    #         setattr(self.modules[key].operator, attribute, value)
    #     else:
    #         raise NotImplementedError("setting modules not implemented")

    def __len__(self):
        return len(self.modules)

    def collect_modules(self, module: nn.Module):
        for _, submodule in module.named_children():
            if isinstance(submodule, Indexer):
                self.modules[submodule.key] = submodule
            self.collect_modules(submodule)

    def start_tracking(self):
        for tracker in self.modules.values():
            tracker.tracking = True

    def stop_tracking(self):
        for tracker in self.modules.values():
            tracker.tracking = False

    def reset_statistics(self):
        for tracker in self.modules.values():
            tracker.reset_statistics()

    def track(self):
        return TrackingContextManager(self)

    def requires_grad(self, requires_grad: bool = True):
        if requires_grad:
            for param in self.parameters():
                param.requires_grad = True
        else:
            for param in self.parameters():
                param.requires_grad = False

    @abstractmethod
    def copy(self):
        pass


class TrackingContextManager:
    def __init__(self, model: IndexedModel):
        self.model = model

    def __enter__(self):
        self.model.reset_statistics()
        self.model.start_tracking()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.stop_tracking()
        return False
