"""
Benchmark comparing the performance of random write accesses into numpy arrays
vs torch tensors for small containers.

We noticed a major performance hit when using a torch.Tensor as a buffer during
the tokenization/encoding of chess games. This benchmark demonstrates the
problem: Random write access into a torch tensor is more than one order of
magnitude slower compare to numpy, at least for small tensor sizes.

Relevant: https://discuss.pytorch.org/t/torch-is-slow-compared-to-numpy/117502
"""

import pyperf
import numpy as np
import torch


def random_access_numpy(num_elem: int, num_iter: int):
    idx = np.random.randint(num_elem, size=(int(0.9 * num_elem),))
    data = np.empty((num_elem,))
    for i in range(num_iter):
        for j in range(len(idx)):
            data[j] = i


def random_access_torch(num_elem: int, num_iter: int):
    idx = np.random.randint(num_elem, size=(int(0.9 * num_elem),))
    data = torch.empty((num_elem,))
    for i in range(num_iter):
        for j in range(len(idx)):
            data[j] = i


def main():
    runner = pyperf.Runner()

    num_elems = [100, 10000, 1000000]
    num_iters = [4]

    for num_elem in num_elems:
        for num_iter in num_iters:
            runner.bench_func(
                f"{random_access_numpy.__name__}(num_elem={num_elem}, num_iter={num_iter})",
                random_access_numpy,
                num_elem,
                num_iter,
                metadata={"num_elem": num_elem, "num_iter": num_iter},
            )
            runner.bench_func(
                f"{random_access_torch.__name__}(num_elem={num_elem}, num_iter={num_iter})",
                random_access_torch,
                num_elem,
                num_iter,
                metadata={"num_elem": num_elem, "num_iter": num_iter},
            )


if __name__ == "__main__":
    main()
