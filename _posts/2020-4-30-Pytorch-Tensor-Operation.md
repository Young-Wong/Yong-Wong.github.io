---
layout: post
title: Pytorch Tensor Operation
date: 2020-03-12
categories: Pytorch
tags: Pytorch
---

### torch.cat() 
<code>torch.cat(tensors, dim=0, out=None)</code>

Concatenates the given sequence of seq tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty.


```markdown
t = torch.ones((2,3))
t_0 = torch.cat([t,t], dim = 0)
t_1 = torch.cat([t,t], dim = 1)

t_0
>>> tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]])

t_1
>>> tensor([[1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1.]])
```


### torch.chunk()
<code>torch.gather(input, dim, index, out=None, sparse_grad=False)</code>

Splits a tensor into a specific number of chunks. Each chunk is a view of the input tensor.
Last chunk will be smaller if the tensor size along the given dimension dim is not divisible by chunks


```
a = torch.ones((2,7))
list_of_tensors = torch.chunk(a, dim=1, chunks =3)

for idx, t in enumerate(list_of_tensors):
    print("No.{} tensor: {}, shape is {}".format(idx+1, t, t.shape))

>>> No.1 tensor: tensor([[1., 1., 1.],
        [1., 1., 1.]]), shape is torch.Size([2, 3])
    No.2 tensor: tensor([[1., 1., 1.],
            [1., 1., 1.]]), shape is torch.Size([2, 3])
    No.3 tensor: tensor([[1.],
            [1.]]), shape is torch.Size([2, 1])
```
### torch.split()
<code>torch.split(tensor, split_size_or_sections, dim=0)</code>

Splits the tensor into chunks. Each chunk is a view of the original tensor.
* If split_size_or_sections is an integer type, then tensor will be split into equally sized chunks (if possible). Last chunk will be smaller if the tensor size along the given dimension dim is not divisible by split_size.
* If split_size_or_sections is a list, then tensor will be split into len(split_size_or_sections) chunks with sizes in dim according to split_size_or_sections.

```
a = torch.ones((2,5))
list_of_tensors = torch.split(a, [2,1,2], dim=1)
​
for idx, t in enumerate(list_of_tensors):
    print("No.{} tensor: {}, shape is {}".format(idx+1, t, t.shape))

>>> No.1 tensor: tensor([[1., 1.],
        [1., 1.]]), shape is torch.Size([2, 2])
    No.2 tensor: tensor([[1.],
        [1.]]), shape is torch.Size([2, 1])
    No.3 tensor: tensor([[1., 1.],
        [1., 1.]]), shape is torch.Size([2, 2])
```

### torch.index_select()
<code>torch.index_select(input, dim, index, out=None)</code>

Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.
```
t = torch.randint(0,9, size = (3,3))
idx = torch.tensor([0,2])
t_select = torch.index_select(t, dim=0, index=idx)

print("t:\n{}\nt_select:\n{}".format(t, t_select))

>>> t:
    tensor([[1, 5, 5],
            [8, 8, 7],
            [8, 3, 8]])
    t_select:
    tensor([[1, 5, 5],
            [8, 3, 8]])
```

### torch.masked_select()
<code>torch.masked_select(input, mask, out=None)</code>

Returns a new 1-D tensor which indexes the input tensor according to the boolean mask mask which is a BoolTensor.
```
t = torch.randint(0,9, size = (3,3))
mask = t.ge(5)
t_select = torch.masked_select(t, mask)

print("t:\n{}\nmask:\n{}\nt_select:\n{}".format(t, mask, t_select))

>>> t:
    tensor([[0, 1, 7],
            [4, 4, 5],
            [0, 4, 6]])
    mask:
    tensor([[False, False,  True],
            [False, False,  True],
            [False, False,  True]])
    t_select:
    tensor([7, 5, 6])
```

### torch.reshape()
<code>torch.reshape(input, shape)</code>
* input (Tensor) – the tensor to be reshaped
* shape (tuple of python:ints) – the new shape

Returns a tensor with the same data and number of elements as input, but with the specified shape. When possible, the returned tensor will be a view of input. Otherwise, it will be a copy. Contiguous inputs and inputs with compatible strides can be reshaped without copying, but you should not depend on the copying vs. viewing behavior.
```
t = torch.randperm(8)
t_reshape = torch.reshape(t, (2,4))

print("t:\n{}\nt_reshape:\n{}".format(t, t_reshape))

>>> t:
    tensor([7, 5, 3, 4, 2, 6, 1, 0])
    t_reshape:
    tensor([[7, 5, 3, 4],
            [2, 6, 1, 0]])
```

### torch.add()
<code>torch.add(input, other, *, alpha=1, out=None)</code>

Each element of the tensor other is multiplied by the scalar alpha and added to each element of the tensor input. The resulting tensor is returned.
The shapes of input and other must be broadcastable.
out = input + alpha × other
```
t_0 = torch.randn((3,3))
t_1 = torch.ones_like(t_0)
t_add = torch.add(t_0, 10, t_1)

print("t_0:\n{}\nt_1:\n{}\nt_add:\n{}".format(t_0, t_1, t_add))

>>> t_0:
    tensor([[-0.6974, -0.7791, -0.4557],
            [-0.7229, -1.5595,  1.5851],
            [ 1.1120,  0.3162, -0.4607]])
    t_1:
    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]])
    t_add:
    tensor([[ 9.3026,  9.2209,  9.5443],
            [ 9.2771,  8.4405, 11.5851],
            [11.1120, 10.3162,  9.5393]])
```

