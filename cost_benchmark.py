import time
import torch
from grid_sample1d import GridSample1d
from acc_benchmark import original, mine, args_groups
import json

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU

Ns = [4, 8, 16, 32, 64, 128, 256]
Cs = [64, 128, 256, 512, 1024]
L_ins = [16, 32, 64, 128, 256, 512]
L_outs = [32, 64, 128, 256, 512]

N_fix = 32
C_fix = 256
L_in_fix = 64
L_out_fix = 128


def get_forward_backward_speed(func, inputs, N_iters):
    forward = 0
    backward = 0
    for _ in range(N_iters):
        start = time.time()
        output = func(*inputs)
        torch.cuda.synchronize()
        forward += time.time() - start

        start = time.time()
        output.sum().backward()
        torch.cuda.synchronize()
        backward += time.time() - start
    print(func)
    print('Forward: {:.3f} ms | Backward {:.3f} ms'.format(forward * 1e3 / N_iters, backward * 1e3 / N_iters))
    return forward * 1e3 / N_iters, backward * 1e3 / N_iters


if __name__ == '__main__':
    N = 4  # 4
    C = 256
    L_in = 16
    L_out = 32
    N_iters = 1000

    N_ys = [[] for _ in range(len(args_groups))]
    N_yso = [[] for _ in range(len(args_groups))]
    for N in Ns:
        print('N=', N)
        input = torch.randn((N, C_fix, L_in_fix), requires_grad=True).cuda()
        grids = torch.randn((N, L_out_fix), requires_grad=True).cuda()

        func = torch.sin

        sin_forward, sin_backward = get_forward_backward_speed(func, [grids], N_iters=N_iters)

        for i, args in enumerate(args_groups):
            func = mine
            module = GridSample1d(**args['mine'])
            inputs = [input, grids, module]
            forward, backward = get_forward_backward_speed(func, inputs, N_iters=N_iters)

            real_forward = forward - sin_forward
            real_backward = backward - sin_backward

            N_ys[i].append((real_forward, real_backward))

            print('-' * 50)
            print('start test original')
            func = original
            inputs = [input, grids, args['original']['padding_mode'], args['original']['align_corners']]
            forward, backward = get_forward_backward_speed(func, inputs, N_iters=N_iters)
            real_forward = forward - sin_forward
            real_backward = backward - sin_backward
            N_yso[i].append((real_forward, real_backward))

    C_ys = [[] for _ in range(len(args_groups))]
    C_yso = [[] for _ in range(len(args_groups))]
    for C in Cs:
        print('C=', C)
        input = torch.randn((N_fix, C, L_in_fix), requires_grad=True).cuda()
        grids = torch.randn((N_fix, L_out_fix), requires_grad=True).cuda()

        func = torch.sin

        sin_forward, sin_backward = get_forward_backward_speed(func, [grids], N_iters=N_iters)

        for i, args in enumerate(args_groups):
            func = mine
            module = GridSample1d(**args['mine'])
            inputs = [input, grids, module]
            forward, backward = get_forward_backward_speed(func, inputs, N_iters=N_iters)

            real_forward = forward - sin_forward
            real_backward = backward - sin_backward

            C_ys[i].append((real_forward, real_backward))

            print('-' * 50)
            print('start test original')
            func = original
            inputs = [input, grids, args['original']['padding_mode'], args['original']['align_corners']]
            forward, backward = get_forward_backward_speed(func, inputs, N_iters=N_iters)
            real_forward = forward - sin_forward
            real_backward = backward - sin_backward
            C_yso[i].append((real_forward, real_backward))

    L_in_ys = [[] for _ in range(len(args_groups))]
    L_in_yso = [[] for _ in range(len(args_groups))]
    for L_in in L_ins:
        print('L_in=', L_in)
        input = torch.randn((N_fix, C_fix, L_in), requires_grad=True).cuda()
        grids = torch.randn((N_fix, L_out_fix), requires_grad=True).cuda()

        func = torch.sin

        sin_forward, sin_backward = get_forward_backward_speed(func, [grids], N_iters=N_iters)

        for i, args in enumerate(args_groups):
            func = mine
            module = GridSample1d(**args['mine'])
            inputs = [input, grids, module]
            forward, backward = get_forward_backward_speed(func, inputs, N_iters=N_iters)

            real_forward = forward - sin_forward
            real_backward = backward - sin_backward

            L_in_ys[i].append((real_forward, real_backward))

            print('-' * 50)
            print('start test original')
            func = original
            inputs = [input, grids, args['original']['padding_mode'], args['original']['align_corners']]
            forward, backward = get_forward_backward_speed(func, inputs, N_iters=N_iters)
            real_forward = forward - sin_forward
            real_backward = backward - sin_backward
            L_in_yso[i].append((real_forward, real_backward))

    L_out_ys = [[] for _ in range(len(args_groups))]
    L_out_yso = [[] for _ in range(len(args_groups))]
    for L_out in L_outs:
        print('L_out=', L_out)
        input = torch.randn((N_fix, C_fix, L_in_fix), requires_grad=True).cuda()
        grids = torch.randn((N_fix, L_out), requires_grad=True).cuda()

        func = torch.sin

        sin_forward, sin_backward = get_forward_backward_speed(func, [grids], N_iters=N_iters)

        for i, args in enumerate(args_groups):
            func = mine
            module = GridSample1d(**args['mine'])
            inputs = [input, grids, module]
            forward, backward = get_forward_backward_speed(func, inputs, N_iters=N_iters)

            real_forward = forward - sin_forward
            real_backward = backward - sin_backward

            L_out_ys[i].append((real_forward, real_backward))

            print('-' * 50)
            print('start test original')
            func = original
            inputs = [input, grids, args['original']['padding_mode'], args['original']['align_corners']]
            forward, backward = get_forward_backward_speed(func, inputs, N_iters=N_iters)
            real_forward = forward - sin_forward
            real_backward = backward - sin_backward
            L_out_yso[i].append((real_forward, real_backward))

    res = {
        'N': [Ns, N_ys, N_yso],
        'C': [Cs, C_ys, C_yso],
        'L_in': [L_ins, L_in_ys, L_in_yso],
        'L_out': [L_outs, L_out_ys, L_out_yso]
    }
    json.dump(res, open('speed_res.json','w'))
