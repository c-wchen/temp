---
title: numpy
date: 2019/7/13 20:46:25
tags:
---

广播原则： 如果两个数组的后缘维度（trailing dimension， 即从末尾开始算起的维度）的轴长相符合其中一方的长度为1， 则认为他们是广播兼容的。广播会在缺失和（或）长度为1的维度进行

<!-- more -->


```python
import numpy as np
t1 = np.arange(5).reshape((5, 1)) #(10, 1)
t2 = np.arange(25).reshape((5,5)) #(10, 10)
t1 * t2
'''
array([[ 0,  0,  0,  0,  0],
       [ 5,  6,  7,  8,  9],
       [20, 22, 24, 26, 28],
       [45, 48, 51, 54, 57],
       [80, 84, 88, 92, 96]])
'''

```

### dtype

```python
import numpy as np

stu = np.dtype([
    ('name', np.object),
    ('age', np.int32),
    ('marks', np.object)
])

s1 = np.array([
    ('chen', 21, 'student 1'),
    ('chen', 21, 'student 2'),
    ('chen', 21, 'student 3')
], dtype=stu)

print(s1, s1.itemsize)
'''
[('chen', 21, 'student 1') ('chen', 21, 'student 2') ('chen', 21, 'student 3')]  20
'''
```

### newaxis

---

https://www.cnblogs.com/onemorepoint/p/8110523.html

```python
import numpy as np

a = np.arange(3)
# 行向量变为列向量
b = a[:, np.newaxis]
c = a[:, None]
print(b, c, np.newaxis is None, sep='\n')
'''
[[0]
 [1]
 [2]]
[[0]
 [1]
 [2]]
True
'''
```

### 分割和堆积（\[h|v][split|stack]）

---

```python
import numpy as np

a = np.arange(16).reshape(4, 4)
b = np.arange(16).reshape(4, 4)
# 水平堆积
c = np.hstack((a, b))
# 垂直堆积
d = np.vstack((a, b))

# 水平分割
e = np.hsplit(a, 2)
# 垂直分割
f = np.vsplit(a, 2)
print(c, d, e, f, sep='\n')
'''
# hstack(a, b)
[[ 0  1  2  3  0  1  2  3]
 [ 4  5  6  7  4  5  6  7]
 [ 8  9 10 11  8  9 10 11]
 [12 13 14 15 12 13 14 15]]
# vstack(a, b)
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]
 [ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]
# hsplit(a, 2)
[array([[ 0,  1], [ 4,  5], [ 8,  9], [12, 13]]),       
 array([[ 2,  3], [ 6,  7], [10, 11], [14, 15]])]
# vsplit(a, 2)
[array([[0, 1, 2, 3], [4, 5, 6, 7]]), 
 array([[ 8,  9, 10, 11], [12, 13, 14, 15]])]

'''
```

### view & copy

---

```python
import numpy as np

t1 = np.arange(10)
t1_view = t1.view()
# numpy切片数组会返回一个视图,
t2 = t1[::2]
# 深复制
t3 = t1.copy()
print(t1_view, t1_view.base is t1, t2, t2.base is t1, t3 is t1, sep='\n')
'''
[0 1 2 3 4 5 6 7 8 9]
True
[0 2 4 6 8]
True
False
'''
#  视图数据和原始数据的修改相互影响
t1[2] = 1000
t2[4] = -1000
print(t1, t2, sep='\n')
'''
[    0     1  1000     3     4     5     6     7  -1000     9]
[    0  1000     4     6  -1000]
'''
```

### numpy.axis

---

> [https://juejin.im/post/5d4943b2518825262a6bc553](https://juejin.im/post/5d4943b2518825262a6bc553)

```python
import numpy as np

m1 = np.arange(24).reshape((2, 3, 4))

# m1[0, ...] + m1[1, ...]
# np.sum 和 obj.sum效果相同
print(np.sum(m1, axis=0))
# m1[:, 0, :] + m1[:, 1, :] + m1[:, 2, :]
print(np.sum(m1, axis=1))
# m1[..., 0] + m1[..., 1] + m1[..., 2] + m1[..., 3]
print(np.sum(m1, axis=2))
```

$$
m.sum(axis=k) = \sum_i^n(m[x_1, x_2, ..., x_{k-1}, i, x_{k+1}, ...x_n])
$$

注意 : 从上面可以理解， axis=k相当去掉k象限，新生成的值为第k象限求和值

### np.all & np.any

---

```python
import numpy as np
m1 = np.arange(10)
# all => x1 && x2 && x3...&&xn
# any => x1 || x2 || x3...||xn
print(np.all(m1), np.any(m1))
'''
False True
'''
```

### numpy.where

---

> https://www.cnblogs.com/massquantity/p/8908859.html

```python
# numpy.where(condition[, x, y])
t1 = np.arange(16).reshape((4, 4)).astype(float)
t1[1:-1, 1:-1] = np.nan
# 替换nan值
np.where(t1==t1, t1, 0) 
# 返回满足条件下标
# np.where(condition)
np.transpose(np.where(t1 > 0))
'''
array([[0, 1],
       [0, 2],
       [0, 3],
       [1, 0],
       [1, 3],
       [2, 0],
       [2, 3],
       [3, 0],
       [3, 1],
       [3, 2],
       [3, 3]], dtype=int64)
'''
```

### numpy.loadtxt

---

```python
import numpy as np
import matplotlib.pyplot as plt

# views likes dislikes comment_total
gb_view_data = np.loadtxt('GB_video_data_numbers.csv', delimiter=',', dtype=int)

comments = gb_view_data[:, -1]
comments = comments[comments < 10000]
# 组距
d = 200
# 组数
bins = (comments.max() - comments.min()) // d

plt.figure(figsize=(20, 8), dpi=80)
plt.hist(comments, bins)
plt.xticks(range(comments.min(), comments.max() + d, d), rotation=90)
plt.show()
```

### 索引和切片

---

```python
import numpy as np

m = np.arange(24).reshape((4, 6))
# 索引
i = 1, 2
# m[1, 2] = m[1][2]
# m[(1, 2)] = m[1, 2]
# m[[1, 2]] = [m[1], m[2]]
# m[array([1, 2])] =  [m[1], m[2]]
print(m[1, 2], m[i], m[list(i)], m[np.array(i)], sep='\n')
'''
8
8
[[ 6  7  8  9 10 11]
 [12 13 14 15 16 17]]
[[ 6  7  8  9 10 11]
 [12 13 14 15 16 17]]
'''
j = 2, 3
# m[[1, 2], [2, 3]] = [m[1, 2], [2, 3]]
# m[(i, j)] = m[i, j]
# m[np.array([(1, 2), (2, 3)])] = [[m[1], m[2]], [m[2], m[3]]]
print(m[i, j], m[(i, j)], m[[i, j], :], m[np.array([i, j])], sep='\n')
'''
[ 8 15]
[ 8 15]
[[[ 6  7  8  9 10 11]
  [12 13 14 15 16 17]]
 [[12 13 14 15 16 17]
  [18 19 20 21 22 23]]]
[[[ 6  7  8  9 10 11]
  [12 13 14 15 16 17]]
 [[12 13 14 15 16 17]
  [18 19 20 21 22 23]]]
'''
# 切片（与列表类似）
print(m[::2, 3:])
'''
[[ 3  4  5]
 [15 16 17]]
'''
# 布尔索引
print(m[m[:, -1] > 10])
'''
[[ 6  7  8  9 10 11]
 [12 13 14 15 16 17]
 [18 19 20 21 22 23]]
'''
```

注意：  索引和切片返回时视图， 对试图的修改即对原始数据的修改

### linalg = linear algebra线性代数

---

```python
# np.linalg.*
#计算特征值和特征向量
import numpy as np
a = np.full((2, 2), 2)
x, y = np.linalg.eig(a)
print(x, y)
```

## numpy.random

---

```python
import numpy as np
import matplotlib.pyplot as plt
m = np.arange(100).reshape((10, 10))
# 随机洗牌
np.random.shuffle(m)
print(m)
'''
array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
       [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
       [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
       [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
       [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
       [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
       [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
       [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
       [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
       [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]])
'''
# 取值为标准正态分布X~N(0, 1)
x = np.random.randn(10000)
plt.figure(figsize=(20, 8))
plt.hist(x, 1000)
plt.show()
```

### 参考文献

---

[内置标量类型]([https://www.numpy.org.cn/reference/arrays/scalars.html#%E5%86%85%E7%BD%AE%E6%A0%87%E9%87%8F%E7%B1%BB%E5%9E%8B](https://www.numpy.org.cn/reference/arrays/scalars.html#内置标量类型))

[指定和构造数据类型]([https://www.numpy.org.cn/reference/arrays/dtypes.html#%E6%8C%87%E5%AE%9A%E5%92%8C%E6%9E%84%E9%80%A0%E6%95%B0%E6%8D%AE%E7%B1%BB%E5%9E%8B](https://www.numpy.org.cn/reference/arrays/dtypes.html#指定和构造数据类型))

[numpy常用方法](https://blog.csdn.net/mengenqing/article/details/80574755)

[numpy随机数组](https://www.cnblogs.com/jason--/p/11567316.html)