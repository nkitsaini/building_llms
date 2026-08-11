from os import environ
import torch
from plotly.subplots import make_subplots

class AutoFig:
    def __init__(self):
        self.items = []
    def add_scatter(self, **kw):
        self.items.append(("scatter", kw))
    def add_heatmap(self, **kw):
        self.items.append(("heatmap", kw))
    def add_histogram(self, **kw):
        self.items.append(("histogram", kw))

    def add_torch_histogram(self, *, y: torch.Tensor, name: str, **kwargs):
        self.add_histogram(y=y.view(-1).numpy(), name=name + "-histogram", **kwargs)

    def show(self):
        titles = [kw.get('name', '-') for _, kw in self.items]
        fig = make_subplots(rows=len(self.items), cols=1, subplot_titles=titles)
        for i, (kind, kw) in enumerate(self.items, start=1):
            getattr(fig, f"add_{kind}")(row=i, col=1, **kw)
        fig.update_layout(height=300 * len(self.items))
        if environ.get("PLOT") == "0":
            return
        fig.show()
