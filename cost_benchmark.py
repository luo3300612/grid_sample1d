import time
import torch
from op import GridSample1d
from acc_benchmark import original, mine, args_groups

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU

N = 4  # 4
channel = 256
L_in = 16

# Note the device=cuda_device arguments here
x = torch.randn(N, 64, 64, 64, channel).cuda()
lines = torch.randn((N, channel, L_in)).cuda()

module = GridSample1d()

forward = 0
backward = 0
N_iters = 1000


for args in args_groups:
    with torch.no_grad():
        for _ in range(N_iters):
            start = time.time()
            output = mine(x, lines, module, **args['mine'])
            torch.cuda.synchronize()
            forward += time.time() - start

            start = time.time()
            output.sum() .backward()
            torch.cuda.synchronize()
            backward += time.time() - start

    print('Forward: {:.3f} ms | Backward {:.3f} ms'.format(forward * 1e3 / N_iters, backward * 1e3 / N_iters))
