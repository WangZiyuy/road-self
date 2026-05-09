# -*- coding: UTF-8 -*-
'''
@Project ：VecRoad 
@File    ：excel.py
@IDE     ：PyCharm 
@Author  ：wzy
@Date    ：2025/9/21 12:07:32
'''
import pandas as pd
import re
# 读取Excel文件
df = pd.read_excel('vote.xlsx')

# 假设你要统计的列名是 'Names'
column_name = 'c'

# 将该列中的每个单元格中的人名按逗号和换行符拆分
names = df[column_name].apply(lambda x: re.split(r'[\t\n,， ]+', str(x).strip()))

# 将拆分出来的人名展平为一个列表
names_flat = names.explode().reset_index(drop=True)

# 统计每个人名的出现频次
name_counts = names_flat.value_counts()

# 输出结果
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(name_counts)
