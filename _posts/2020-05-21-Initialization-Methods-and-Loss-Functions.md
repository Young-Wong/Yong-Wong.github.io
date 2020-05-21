
---
layout: post
title: Initialization Methods and Loss Functions
date: 2020-05-21
categories: Pytorch
tags: Pytorch
---

## Initialization Methods

<code>torch.nn.init</code>
This operation is used to solve gradient vanishing and gradient exploding

#### init.calculate_gain()
<code>torch.nn.init.calculate_gain(nonlinearity, param=None)</code>
Return the recommended gain value for the given nonlinearity function

```
>>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2
```
<br>

#### init.uniform_()
<code>torch.nn.init.uniform_(tensor, a=0.0, b=1.0)</code>

Fills the input Tensor with values drawn from the uniform distribution \mathcal{U}(a, b)U(a,b) .

**Parameters**
* tensor – an n-dimensional torch.Tensor
* a – the lower bound of the uniform distribution
* b – the upper bound of the uniform distribution

**Examples**
```
>>> w = torch.empty(3, 5)
>>> nn.init.uniform_(w)
```
<br>

#### init.normal_()
<code>torch.nn.init.normal_(tensor, mean=0.0, std=1.0)</code>

Fills the input Tensor with values drawn from the normal distribution 

**Parameters**
* tensor – an n-dimensional torch.Tensor
* mean – the mean of the normal distribution
* std – the standard deviation of the normal distribution

**Examples**
```
>>> w = torch.empty(3, 5)
>>> nn.init.normal_(w)
```
<br>

####init.xavier_uniform_()
<code>torch.nn.init.xavier_uniform_(tensor, gain=1.0)</code>

Fills the input Tensor with values according to the method described in Understanding the difficulty of training deep feedforward neural networks

**Parameters**
* tensor – an n-dimensional torch.Tensor
* gain – an optional scaling factor
torch.nn.init.xavier_normal_(tensor, gain=1.0)

**Examples**
```
>>> w = torch.empty(3, 5)
>>> nn.init.xavier_normal_(w)
```
<br>

#### init.kaiming_uniform_()
```
torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu’)
```
The resulting tensor will have values sampled from(−bound,bound) 


**Parameters**
- mode – either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.

**Examples**
```
>>> w = torch.empty(3, 5)
>>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
```
<br>

#### init.kaiming_normal_()
```
torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu’)
```
The resulting tensor will have values sampled from (0, $std^2$)

**Examples**
```
>>> w = torch.empty(3, 5)
>>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu’)
```
<br><br>


## Loss function

#### Cross entropy
```
torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
```
<br>

This criterion combines <code>nn.LogSoftmax()</code> and <code>nn.NLLLoss()</code> in one single class

**Parameters**
- reduction: 'none' , 'mean', 'sum'.
    - none: no reduction will be applied
    - mean: the sum of the output will be divided by the number of elements in the output
    - sum: the output will be summed. 
    - Default: 'mean'

<br>
#### Binary Cross Entropy <br>
```
torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
```


The unreduced (i.e. with reduction set to 'none') loss can be described as:

$l_n = -w_n[y_nlogx_n + (1-y_n)log(1-x_n)]$ 

**Examples**
```
>>> m = nn.Sigmoid()
>>> loss = nn.BCELoss()
>>> input = torch.randn(3, requires_grad=True)
>>> target = torch.empty(3).random_(2)
>>> output = loss(m(input), target)
>>> output.backward()
```

#### BCEWithLogitsLoss<br>
```
torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
```
This loss combines a Sigmoid layer and the BCELoss in one single class. This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.

***Note:*** we don't add sigmoid layer to the neural network model. Sigmoid layer is only used in the loss function for this case.
​
The unreduced (i.e. with reduction set to 'none') loss can be described as:
$l_n = -w_n[y_nlogσ(x_n) + (1-y_n)log(1-σ(x_n)]$ 

**Parameters**
- pos_weight (Tensor, optional) – a weight of positive examples. Must be a vector with length equal to the number of classes

#### Other Loss Functions:
- nn.L1Loss
- nn.MSELoss
- nn.SmoothL1Loss
- nn.PoissonNLLLoss
- nn.KLDivLoss
- nn.MarginRankingLoss
- nn.MultiLabelMarginLoss
- nn.SoftMarginLoss
- nn.MultiLabelSoftMarginLoss
- nn.MultiMarginLoss
- nn.TripletMarginLoss
- nn.HingeEmbeddingLoss
- nn.CosineEmbeddingLoss
- nn.CTCLoss