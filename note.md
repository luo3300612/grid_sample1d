# Ref

* [grid sample pytorch](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html?highlight=grid_sample#torch.nn.functional.grid_sample)
* [grid sample cuda](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/GridSampler.cu)
* [pytorch C++ doc](https://pytorch.org/cppdocs/notes/tensor_creation.html)
* [cuda doc](https://docs.nvidia.com/cuda/)

## 在相同位置创建tensor

```c++
auto output = torch::zeros({batch_size, C, L_out}, input.options());
```

这样就在input相同的位置创建了output

## accessor是inefficient的

源文档中说了，即便accessor可以方便代码的可读性，但非常低效， 从pytorch官方代码中没用使用accessor也可以发现，这玩意根本就不好用

## benchmark
100 iter average
```python
grid # 4 * 64**3 * 256
input # 4 * 256 * 16
```
|item|speed|RAM|
|----|----|-----|
|lxy original|552ms|10825MB|
|2d grid sample forward|198ms|9800MB|
|lxy dog grid sample forward|137ms|5705MB|

## 进度
forward backward done

但backward error会随着C的增大而增大