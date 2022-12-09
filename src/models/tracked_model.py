from torch import nn

from .activation_tracker import Tracker


class TrackedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trackers = dict()

    def init(self):
        self.collect_trackers(self)

    def __getitem__(self, key: str):
        if "." in key:
            key, attribute = key.split(".")
            return getattr(self.trackers[key].operator, attribute)
        return self.trackers[key]

    def __len__(self):
        return len(self.trackers)

    def collect_trackers(self, module: nn.Module):
        for _, submodule in module.named_children():
            if isinstance(submodule, Tracker):
                self.trackers[submodule.key] = submodule
            self.collect_trackers(submodule)

    def start_tracking(self):
        for tracker in self.trackers.values():
            tracker.tracking = True

    def stop_tracking(self):
        for tracker in self.trackers.values():
            tracker.tracking = False

    def reset_statistics(self):
        for tracker in self.trackers.values():
            tracker.reset_statistics()

    def track(self):
        return TrackingContextManager(self)


class TrackingContextManager:
    def __init__(self, model: TrackedModel):
        self.model = model

    def __enter__(self):
        self.model.reset_statistics()
        self.model.start_tracking()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.stop_tracking()
        return False
