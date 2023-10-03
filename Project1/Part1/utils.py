import numpy as np
import collections


class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class Sampler:
    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, data_source):
        super(SequentialSampler, self).__init__(data_source)
        self.data_source = data_source
        self.num_samples = len(self.data_source)

    def __iter__(self):
        return iter(range(self.num_samples))

    def __len__(self):
        return self.num_samples


class RandomSampler(Sampler):
    def __init__(self, data_source):
        super(RandomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.num_samples = len(self.data_source)

    def __iter__(self):
        """
        返回一个实现了 __next__ 方法的对象。

        返回:
        - (Generator[int]): 通过 yield 关键字返回生成器对象，生成的索引来自 shuffle 后的结果。

        注意:
        - 每次调用都会刷新一次随机索引序列。
        """
        yield from np.random.permutation(self.num_samples).tolist()

    def __len__(self):
        return self.num_samples


class BatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        """
        返回一个实现了 __next__ 方法的对象。

        返回:
        - (Generator[list]): 通过 yield 关键字返回生成器对象，生成的索引列表元素通过迭代 sampler 获得。

        注意:
        - for idx in self.sampler 中会隐式地 sampler_iter = iter(self.sampler) \
            然后不断地 next(sampler_iter) 获取元素。
        """
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


def default_collate(batch):
    """
    默认数据整理函数，将数据打包成第一维是 batch_size 的 ndarray 。

    参数:
    - batch (Sequence): 其中的 ndarray 必须形状相同。

    返回:
    - stacked (ndarray): 打包后的 ndarray 。如果 batch 中的元素形状为 (d_1, d_2, ...) \
        则返回的数组的形状为 (batch_size, d_1, d_2, ...) 。

    注意:
    - 也可以传入 list[list], list[dict] 等。
    """
    elem_type = type(batch[0])
    if elem_type.__module__ == "numpy":
        return np.stack(batch, 0)
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]
    else:
        raise NotImplementedError


class DataLoader:
    def __init__(
        self, dataset, batch_size=1, shuffle=False, collate_fn=None, drop_last=False
    ):
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        if collate_fn is None:
            collate_fn = default_collate

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.batch_sampler_iter = None
        self.collate_fn = collate_fn

    def __len__(self):
        return len(self.batch_sampler)

    def __next__(self):
        indices = next(self.batch_sampler_iter)
        data = [self.dataset[idx] for idx in indices]
        stacked = self.collate_fn(data)
        return stacked

    def __iter__(self):
        """
        返回一个实现了 __next__ 方法的对象。

        返回:
        - self (DataLoader): 对象本身。

        注意:
        - 返回自身，因为自身已经实现了 __next__ 。
        - 在这里调用 iter 创建 BatchSampler 的迭代器。每次 for batch in dataloader \
            其实隐式地调用 iter(dataloader) 同时执行 iter(self.batch_sampler) ，也就是 \
            开始进行一轮新的采样。
        """
        self.batch_sampler_iter = iter(self.batch_sampler)
        return self
