import torch
from grid_sample1d.op import GridSample1d
from tqdm import tqdm
from acc_benchmark import args_groups, mine, inspect, setup_seed

if __name__ == '__main__':
    setup_seed(0)

    batch_size = 20
    C = 512
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
        prev_output = None
        input = torch.randn((batch_size, C, L_in)).cuda()
        grid = torch.randn(batch_size, L_out).cuda()
        try:
            with torch.no_grad():
                for i in tqdm(range(N_samples)):
                    output = mine(input, grid, module).cpu()

                    if prev_output is not None:
                        try:
                            assert torch.allclose(output, prev_output, atol=eps, rtol=eps_r)
                        except:
                            N_err = inspect(output, prev_output)
                            running_err_forward += N_err / torch.numel(output)
                            if N_err / torch.numel(output) >= 0.05:
                                raise
                            else:
                                pass
                    prev_output = output
            print(f'Forward Det test done on {N_samples} samples with eps={eps}')

            print('backward')
            prev_grad_grid = None
            prev_grad_input = None
            for i in tqdm(range(N_samples)):
                setup_seed(0)
                grid_mine = torch.randn((batch_size, L_out), requires_grad=True).cuda()
                input_mine = torch.randn((batch_size, C, L_in), requires_grad=True).cuda()
                grid_mine.retain_grad()
                input_mine.retain_grad()

                output = mine(input_mine, grid_mine, module)

                output = torch.sum(output.view(-1))

                output.backward()

                grad_grid_mine = grid_mine.grad
                grad_input_mine = input_mine.grad

                if prev_grad_grid is not None:
                    try:
                        assert torch.allclose(prev_grad_grid, grad_grid_mine, atol=eps, rtol=eps_r)
                        assert torch.allclose(prev_grad_input, grad_input_mine, atol=eps, rtol=eps_r)
                    except AssertionError:
                        N_err_grid = inspect(grad_grid_mine, prev_grad_grid,verbose=True)
                        N_err_input = inspect(grad_input_mine, prev_grad_input,verbose=True)

                        running_err_backward_grid += N_err_grid / torch.numel(grad_grid_mine)
                        running_err_backward_input += N_err_input / torch.numel(grad_input_mine)
                        if N_err_grid / torch.numel(grad_grid_mine) >= 0.05 or N_err_input / torch.numel(
                                grad_input_mine) >= 0.05:
                            raise
                        else:
                            pass
            print(f'Backward Det test done on {N_samples} samples with eps={eps}')

            print(f'running err forward:{running_err_forward * 100:.2f}%')
            print(f'running err backward input:{running_err_backward_input:.2f}%')
            print(f'running err backward grid:{running_err_backward_grid:.2f}%')
        except AssertionError:
            raise
    print('Done')
