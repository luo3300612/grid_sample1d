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

修改为C上串行后，除了当padding mode=zeros,align_corners=True以外，不再存在上述问题

尝试修改了backward中grid梯度两个计算Block的顺序，结果会与2d的sample不同，这也表明了浮点数累加不满足结合律这一问题

修复了align=false,zero padding下结果是两倍的问题
## 一个问题
pytorch中的2d grid sample，在双线性插值，且zero padding模式align_corners=False时是不适用H/W中有一个是1的情况的，最终的结果会是真实结果的0.5倍，这主要是因为align_corner为False时，1维的坐标变成-0.5，
而生成出来的需要插值坐标的左右近邻必有一个不满足within_bounds的判断，最终会忽略掉恰好一半的数值
