# Python Code Tips


# **>pandas**
```
    df.sub(a, axis=0)   # 按行减去a这一行
    df.drop(a, axis=0)  # 删除a这一列
    df.dropna()         # 删除空值
    df.cov(ddof=n)      # 求自由度为n的协方差
    df.sort_values(by=aattribute)   # 按照某个属性的值排序
    df.set_index()      # 设置索引，可double sorting
    df.isin(values)     # values:iterable, setries, Dataframe or dict
    df.swaplevel()      # 交换双重index的位置
    df.reset_index()    # 重置index, index作为data储存
    df.fillna(num)      # 用num补充所有的nan
    df.cumprod()        # 每个位置的value有前面所有balues累乘
    df.cumsum()         # 累加
    df.drop_duplicates()    # 删除重复值        
    df.astype()         # 整张表修改数据格式         

    pd.isnull(df).sum() # 空值统计

    pd.merge(df1, df2, on= )    # 相同索引会归在同一行，按列拼接
    pd.concat([df1, df2], axis=0)   # 横向拼接

    pd.DatetimeIndex(Growth_Rate.index.get_level_values(1)).year==Year   # 双重index内索引为时间，取出某一年的数据
```

# **>numpy**
```
    np.ones(n)              # 一维
    np.ones((n,m))          # 二维
    np.zeros()
    np.ones()
    np.linalg.inv()         # 求逆矩阵
    np.dot(data1, data2)    # 矩阵相乘
    df1.dot(df2)
    np.split(data, [多个拆分位置])   # 拆分数据
    np.array_split(data, n)         # 将数据平分成为n列
    np.maximum.accumulate(data)
    np.maximum.accumulate(date)     # 计算数组（或dataframe特定轴）的累积最大值（用于最大回撤）
    np.unique(data)         # 找出不重复的值
    np.argmax(data)         # 返回一个numpy数组中最大值的索引值，多个是返回第一个
    np.sign(data)           # 取数字符号（正负号）的函数（-1，0，1）
    np.clip(data, a, b)     # 剪枝，数据范围限制在[a,b]之间
    np.quantile(data, quantile)     # 取出data的quantile的值
    np.vstack((array1, array2))     # 纵向堆叠数据
    np.hstack((array1, array2))     # 水平堆叠数据

    array.tolist()          # array转为list
    array.cumsum()          # 累加，可用于多维数组

```

# **>统计方法**
## 回归
```
    y = 
    x =
    x = sm.add_constant(x)
    model = sm.OLS(y,x)
    result = model.fit()

    p = scipy.stats.f.sf(n:标的数量， m-n-1:自由度)
    ststs.ttest_1samp(data,values)  # t_test:data可以是多列，同时values要为对应数量的值，比如检验25列数据是否显著不为0，values应该为25个0的list
```

# **>数据保存**
## 保存到excel
```
    savelocal = 
    writer = pd.ExcelWriter(savelocal)
    df.to_excel(writer, sheet_name='')

    writer.save()
    writer.close()
```

# **>多进程**
```
    p = mp.Pool(n):     # n:多进程数量
    result = []
    result.append(p.apply_async(function, args=()))
    
    for i in result：
    print(b.get())      # 多进程debug输出

    p.close()
    p.join()
```

# **>Leetcode**
```
    enumerate(x)         # 给x中的元素贴上序号，并返回序号与元素值
    dict().get()         # 根据index取出值
```
