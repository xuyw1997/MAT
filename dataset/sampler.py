from torch.utils.data.sampler import Sampler
import torch
import torch.distributed as dist
import math
from itertools import chain
from torch.utils.data.distributed import DistributedSampler
import numpy as np

class ClipSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """


    def __init__(self, data_source, num_clip_per_video, batch_size, generator=None) -> None:
        self.data_source = data_source
        self.generator = generator
        self.num_clip_per_video = num_clip_per_video
        self.batch_size = batch_size
        n = len(self.data_source)
        assert n % self.num_clip_per_video == 0
        self.num_video = n // self.num_clip_per_video
        num_pad = self.num_video % self.batch_size
        if num_pad > 0:
            self.num_pad = self.batch_size - num_pad


    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __len__(self) -> int:
        return (self.num_video + self.num_pad) * self.num_clip_per_video

    def __iter__(self):

        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        start_list = torch.randperm(self.num_video, generator=generator).tolist()
        start_list = [x * self.num_clip_per_video for x in start_list]
        start_list += start_list[:self.num_pad]
        ans = []
        for i in range(0, len(start_list), self.batch_size):
            tmp = []
            for j in range(self.num_clip_per_video):
                tmp += [x+j for x in start_list[i:i+self.batch_size]]
            ans += tmp
        return iter(ans)



class DistributedClipSampler(Sampler):
    r"""Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.
    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.
    Example::

    """

    def __init__(self, dataset, num_clip_per_video, batch_size, num_replicas = None,
                 rank = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_clip_per_video = num_clip_per_video
        self.batch_size = batch_size
        assert len(self.dataset) % self.num_clip_per_video == 0
        self.total_num_video = len(self.dataset) // self.num_clip_per_video

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_video = math.ceil(self.total_num_video / self.num_replicas)
        num_pad = self.num_video % self.batch_size
        if num_pad > 0:
            num_pad = self.batch_size - num_pad
        self.num_video += num_pad
        self.total_video_size = self.num_video * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.total_num_video, generator=g).tolist()
        else:
            indices = list(range(self.total_num_video))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_video_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_video_size]
        assert len(indices) == self.total_video_size

        # subsample
        indices = indices[self.rank:self.total_video_size:self.num_replicas]
        assert len(indices) == self.num_video

        start_list = [x * self.num_clip_per_video for x in indices]

        ans = []
        for i in range(0, len(start_list), self.batch_size):
            tmp = []
            for j in range(self.num_clip_per_video):
                tmp += [x + j for x in start_list[i:i + self.batch_size]]
            ans += tmp
        return iter(ans)

    def __len__(self) -> int:
        return self.num_video * self.num_clip_per_video

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch



class MyDistributedSampler(Sampler):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, total_bsz, num_clip_per_video, num_replicas = None,
                 rank = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.total_bsz = total_bsz
        self.num_clip_per_video = num_clip_per_video
        self.num_video = len(self.dataset) // self.num_clip_per_video
        assert (self.num_video * self.num_clip_per_video) == len(self.dataset), 'len(dataset) must be divisible by num_clip_per_video'
        assert self.num_video % self.total_bsz == 0, 'num_video must be divisible by total_bsz'
        self.num_iter = (self.num_video // self.total_bsz) * self.num_clip_per_video
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        video_begin = [(i // self.total_bsz)* self.total_bsz* self.num_clip_per_video + j  for i in range(self.num_video) for j in range(self.total_bsz)]

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(video_begin), generator=g).tolist()  # type: ignore[arg-type]

        else:
            indices = list(range(len(video_begin)))  # type: ignore[arg-type]
        reoredered = [video_begin[i] for i in indices]
        video_begin = reoredered
        assert len(video_begin) == self.num_video, f"video_begin:{video_begin}, self.num_video:{self.num_video}"
        clip_id_2d = []
        idx = 0
        while idx < len(video_begin):
            for i in range(self.num_clip_per_video):
                clip_id_2d.append(list(map(lambda x:x + i*self.total_bsz, video_begin[idx:idx + self.total_bsz])))
            idx += self.total_bsz
        clip_id_2d = torch.tensor(clip_id_2d).transpose()

                
        # subsample
        clip_id_2d = clip_id_2d[self.rank:self.num_video:self.num_replicas]
        clip_id_1d = clip_id_2d.transpose().contiguous().view(-1).tolist()
       
        return iter(clip_id_1d)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch