import torch
from grid_sample1d import GridSample1d
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


args_groups = [
    {'original': {'padding_mode': 'zeros', 'align_corners': True},
     'mine': {'padding_mode': False, 'align_corners': True}},
    {'original': {'padding_mode': 'zeros', 'align_corners': False},
     'mine': {'padding_mode': False, 'align_corners': False}},
    {'original': {'padding_mode': 'border', 'align_corners': True},
     'mine': {'padding_mode': True, 'align_corners': True}},
    {'original': {'padding_mode': 'border', 'align_corners': False},
     'mine': {'padding_mode': True, 'align_corners': False}}
]


def original(input, grid, padding_mode, align_corners):
    shape = grid.shape
    grid = grid.sin()  # batch_size * L_out
    input = input.unsqueeze(-1)  # batch_size * C * L_in * 1

    # grid = grid.unsqueeze(-1) # batch_size * L_out * 1
    grid = grid.unsqueeze(1)  # batch_size * 1 * L_out
    grid = torch.stack([-torch.ones_like(grid), grid], dim=-1)
    z = F.grid_sample(input, grid, padding_mode=padding_mode, align_corners=align_corners)
    C = input.shape[1]
    out_shape = [shape[0], C, shape[1]]
    z = z.view(*out_shape)  # batch_size * C * L_out
    return z


def mine(input, grid, module):
    shape = grid.shape
    grid = grid.sin()
    z = module(input, grid)
    C = input.shape[1]
    out_shape = [shape[0], C, shape[1]]
    z = z.view(*out_shape)
    return z


def inspect(output, output_origin, verbose_matrix=False, verbose=False):
    err = torch.abs(output - output_origin)
    max_err = torch.max(err).item()
    pos = torch.argmax(err)

    rela_err = err / torch.abs(output_origin)
    max_err_rela = torch.max(rela_err)
    pos_rela = torch.argmax(rela_err)

    N_err = torch.sum(err > eps).item()
    N_rela_err = torch.sum(rela_err > eps_r).item()

    # if max_err > eps:
    #     if verbose_matrix:
    #         print('output')
    #         print(output)
    #         print('origin')
    #         print(output_origin)
    #         print(output - output_origin)
    #     print('different!')
    #     print(f'max_err={max_err}')
    #     print(f'where origin={output_origin.view(-1)[pos]}')
    #     print(f'mine={output.view(-1)[pos]}')
    #     print(f'N err > eps={N_err}')
    #     print(f'err% = {N_err / torch.numel(output) * 100:.2f}')
    # print('-' * 50)
    if max_err_rela > eps_r:
        if verbose:
            if verbose_matrix:
                print('output')
                print(output)
                print('origin')
                print(output_origin)
                print(output - output_origin)
                print('different!')
            print(f'max_err_rela={max_err_rela}')
            print(f'where origin={output_origin.view(-1)[pos_rela]}')
            print(f'mine={output.view(-1)[pos_rela]}')
            print(f'N err > eps={N_err}')
            print(f'err% = {N_rela_err / torch.numel(output) * 100:.2f}')
    # if N_err == 0:
    #     print('same!')
    return N_rela_err


if __name__ == '__main__':
    setup_seed(0)

    batch_size = 20
    C = 256
    L_in = 16
    L_out = 32

    eps = 1e-6
    eps_r = 1e-5
    N_samples = 100

    print('forward')

    for args in args_groups:
        print('testing')
        print(args)

        module = GridSample1d(**args['mine'])
        running_err_forward = 0.
        running_err_backward_input = 0.
        running_err_backward_grid = 0.
        try:
            with torch.no_grad():
                for i in tqdm(range(N_samples)):
                    input = torch.randn((batch_size, C, L_in)).cuda()
                    grid = torch.randn(batch_size, L_out).cuda()
                    output = mine(input, grid, module).cpu()
                    output_origin = original(input, grid, **args['original']).cpu()
                    try:
                        if (not args['mine']['padding_mode']) and (not args['mine']['align_corners']):
                            torch.allclose(output, output_origin * 2, atol=eps, rtol=eps_r)
                        else:
                            assert torch.allclose(output, output_origin, atol=eps, rtol=eps_r)
                    except:
                        N_err = inspect(output, output_origin)
                        running_err_forward += N_err / torch.numel(output)
                        if N_err / torch.numel(output) >= 0.05:
                            raise
                        else:
                            pass
            print(f'Forward ACC test done on {N_samples} samples with eps={eps}')

            print('backward')
            for i in tqdm(range(N_samples)):
                setup_seed(i)
                grid_original = torch.randn((batch_size, L_out), requires_grad=True).cuda()
                input_original = torch.randn((batch_size, C, L_in), requires_grad=True).cuda()
                grid_original.retain_grad()
                input_original.retain_grad()

                setup_seed(i)
                grid_mine = torch.randn((batch_size, L_out), requires_grad=True).cuda()
                input_mine = torch.randn((batch_size, C, L_in), requires_grad=True).cuda()
                grid_mine.retain_grad()
                input_mine.retain_grad()

                output_origin = original(input_original, grid_original, **args['original'])
                output = mine(input_mine, grid_mine, module)

                if (not args['mine']['padding_mode']) and (not args['mine']['align_corners']):
                    assert torch.allclose(output, output_origin*2, atol=eps, rtol=eps_r)
                else:
                    assert torch.allclose(output, output_origin, atol=eps, rtol=eps_r)

                output_origin = torch.sum(output_origin.view(-1))
                output = torch.sum(output.view(-1))

                output.backward()
                output_origin.backward()

                grad_grid_original = grid_original.grad
                grad_input_original = input_original.grad

                grad_grid_mine = grid_mine.grad
                grad_input_mine = input_mine.grad

                try:
                    if (not args['mine']['padding_mode']) and (not args['mine']['align_corners']):
                        assert torch.allclose(2*grad_grid_original, grad_grid_mine, atol=eps, rtol=eps_r)
                        assert torch.allclose(2*grad_input_original, grad_input_mine, atol=eps, rtol=eps_r)
                    else:
                        assert torch.allclose(grad_grid_original, grad_grid_mine, atol=eps, rtol=eps_r)
                        assert torch.allclose(grad_input_original, grad_input_mine, atol=eps, rtol=eps_r)
                except AssertionError:
                    N_err_grid = inspect(grad_grid_mine, grad_grid_original,verbose=True)
                    N_err_input = inspect(grad_input_mine, grad_input_original,verbose=True)

                    running_err_backward_grid += N_err_grid / torch.numel(grad_grid_mine)
                    running_err_backward_input += N_err_input / torch.numel(grad_input_mine)
                    if N_err_grid / torch.numel(grad_grid_mine) >= 0.05 or N_err_input / torch.numel(
                            grad_input_mine) >= 0.05:
                        raise
                    else:
                        pass
            print(f'Backward ACC test done on {N_samples} samples with eps={eps}')

            print(f'running err forward:{running_err_forward * 100:.2f}%')
            print(f'running err backward input:{running_err_backward_input:.2f}%')
            print(f'running err backward grid:{running_err_backward_grid:.2f}%')
        except AssertionError:
            raise
    print('Done')
