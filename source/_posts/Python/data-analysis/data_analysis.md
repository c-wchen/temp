---
title: data-analysis
date: 2019/7/13 20:46:25
tags:
---
YouTube影片趋势的统计资料（观看次数，喜欢，类别，评论+）

数据来源： https://www.kaggle.com/datasnaek/youtube/data

<!-- more -->

```python
# 对'views', 'likes', 'dislikes', 'comment_tota' 单独保存为csv文件
import pandas as pd
import numpy as np

if __name__ == '__main__':
    gb_file_path = './data/GBvideos.csv'
    df = pd.read_csv(gb_file_path, error_bad_lines=False)
    # temp = df[['views', 'likes', 'dislikes', 'comment_total']].to_numpy(dtype='int')
    # np.savetxt('data/gb_video_nums.csv', temp, fmt='%d', delimiter=',')
    temp = df[['views', 'likes', 'dislikes', 'comment_total']]
    temp.to_csv('data/gb_video_nums.csv', index=False, header=False)
```

分析喜欢的评论数分布

```ptyhon
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # 获取喜欢序列
    y = np.loadtxt('data/gb_video_nums.csv', delimiter=',')[:, 1].astype('int')
    y = y[y < 180000]
    # 组距
    d = 6000
    # 盒子数目
    bins = (y.max() - y.min()) // d
    plt.figure(figsize=(20, 8), dpi=80)
    plt.hist(y, bins=bins)
    plt.xticks(range(y.min(), y.max() + d, d), rotation=45)
    plt.show()
```

![image-20200704151311049](data_analysis/image-20200704151311049.png)

> 从图上可以看出， 喜欢数集中在0-6000左右， 随着数目的增大， 越来越少