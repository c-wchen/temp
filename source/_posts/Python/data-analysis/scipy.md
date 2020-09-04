---
title: scipy
date: 2019/7/13 20:46:25
tags:
---

`scipy`包包含许多专注于科学计算中的常见问题的工具箱。它的子模块对应于不同的应用，比如插值、积分、优化、图像处理、统计和特殊功能等。

`scipy`可以与其他标准科学计算包相对比，比如GSL (C和C++的GNU科学计算包), 或者Matlab的工具箱。`scipy`是Python中科学程序的核心程序包；这意味着有效的操作`numpy`数组，因此，numpy和scipy可以一起工作。

在实现一个程序前，有必要确认一下需要的数据处理时候已经在scipy中实现。作为非专业程序员，科学家通常倾向于**重新发明轮子**，这产生了小玩具、不优化、很难分享以及不可以维护的代码。相反，scipy的程序是优化并且测试过的，因此应该尽可能使用。

<!-- more -->


### 定积分求解

> https://zhuanlan.zhihu.com/p/100951677
>
> https://www.qikegu.com/docs/3499

$$
I = \int_{0}^{\frac{\pi}{2}} \frac{xsinx}{1+cos^2x}dx = \frac{1}{2}ln^2(1 + \sqrt2) + \frac{\pi^2}{8}
$$



```python
from  scipy.integrate import quad
from math import cos, sin, pi, log, sqrt
fx = lambda x: x * sin(x) / (1 + cos(x) ** 2)
# quad函数返回两个值，第一个值是积分的值，第二个值是对积分值的绝对误差估计
val, err = quad(fx, 0, pi / 2)
t_val = -1 / 2 * log(1 + sqrt(2)) ** 2 + pi ** 2 / 8
print(val, t_val)
'''
0.845290850188322 0.8452908501883218  
'''
```

### 线性代数
<!-- 
$$
\left\{  
             \begin{array}{**lr**}  
             x + 3y + 5z = 10 &  \\  
             2x+ 5y + z = 8\\  
             2x + 3y + 8z = 3 &    
             \end{array}  
\right.
$$


$$


\left[
    \begin{array}{ccc}
        1 & 3 & 5\\
        2 & 5 & 1\\
        2 & 3 & 8\\
    \end{array}
\right] 
\left[
    \begin{array}{ccc}
        x \\
        y \\
        z \\
    \end{array}
\right]=
\left[
    \begin{array}{ccc}
        10 \\
        8 \\
        3 \\
    \end{array}
\right]
$$
 -->


```python
from scipy import linalg
import numpy as np

a = np.array([[1, 3, 5], [2, 5, 1], [2, 3, 8]])
b = np.array([10, 8, 3])
# np.linalg.solve(a, b)
s = linalg.solve(a, b)
print (s)
```

### 参考文献

[常量](https://www.qikegu.com/docs/3488)