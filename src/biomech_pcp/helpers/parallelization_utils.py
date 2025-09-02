import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


class UnevenDataParallel(nn.DataParallel):
    def __init__(
        self, module, device_ids=None, output_device=None, dim=0, device_ratio=None
    ):
        super(UnevenDataParallel, self).__init__(
            module, device_ids=device_ids, output_device=output_device, dim=dim
        )
        self.device_ratio = device_ratio
        if device_ratio:
            self.device_ratio = [r / sum(device_ratio) for r in device_ratio]

    def scatter(self, inputs, kwargs, device_ids):
        if self.device_ratio:
            total_size = inputs[0].size(self.dim)
            sizes = [int(r * total_size) for r in self.device_ratio]
            sizes[-1] = total_size - sum(
                sizes[:-1]
            )  # Adjust the last size to match the total
            inputs = [torch.split(inp, sizes, dim=self.dim) for inp in inputs]
            inputs = [
                [chunk.to(device) for chunk, device in zip(inp, device_ids)]
                for inp in inputs
            ]
            return inputs, kwargs
        else:
            return super(UnevenDataParallel, self).scatter(inputs, kwargs, device_ids)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class UnevenDistributedDataParallel(DDP):
    def __init__(
        self, module, device_ids=None, output_device=None, dim=0, device_ratio=None
    ):
        super(UnevenDistributedDataParallel, self).__init__(
            module, device_ids=device_ids, output_device=output_device, dim=dim
        )
        self.device_ratio = device_ratio
        if device_ratio:
            self.device_ratio = [r / sum(device_ratio) for r in device_ratio]

    def __getattr__(self, name):
        try:
            return super(UnevenDistributedDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def scatter(self, inputs, kwargs, device_ids):
        if self.device_ratio:
            total_size = inputs[0].size(self.dim)
            sizes = [int(r * total_size) for r in self.device_ratio]
            sizes[-1] = total_size - sum(
                sizes[:-1]
            )  # Adjust the last size to match the total
            inputs = [torch.split(inp, sizes, dim=self.dim) for inp in inputs]
            inputs = [
                [chunk.to(device) for chunk, device in zip(inp, device_ids)]
                for inp in inputs
            ]
            return inputs, kwargs
        else:
            return super(UnevenDistributedDataParallel, self).scatter(
                inputs, kwargs, device_ids
            )
