import pandas as pd
import numpy as np
from DISP_get import DISP_get
import itertools
import random

random.seed(6)
length = 30  # Input base assignment sequence length
N_test = 5  # Input base assignment sequence to generate the number of strips
orders = DISP_get(length, N_test)   # Generate a random base assignment order

'''Generate sequencing signals corresponding to all base assignment sequences (example 10)'''
read = open("sequence_list.csv") 
seq = np.array(pd.read_csv(read)).T  
"""//seq: The untidy places are occupied by Nans，eg.[['CACATG' 'CTCCTT' 'CACCTG' nan], ['ACTGG' 'AGTGG' 'ACTGA' 'AGTGA']]"""

N_order = orders.shape[0]  # Number of base assignment sequence items obtained (number of items)
N_MH = seq.shape[0]  # The number of microhaplotypes is obtained
N_allele_max = seq.shape[1]  # Get the number of alleles

order_part = np.zeros([1, length])  # Generates an empty array to hold the base assignment order of the signal "" incomplete """
order_phase = np.zeros([1, length])  # Generate an empty array to hold the "" complete and phase ready "" base assignment order in the phase step
"""order_phase: All alleles of MH can be covered completely, and the phase base assignment sequence can be carried out for any allelic combination in any MH"""

signal_all = np.zeros([N_order, N_MH, N_allele_max, length])  # Create an empty array to hold "all" base assignment sequence sequencing signals
"""//singal_all: From the outer layer to the inner layer, the index numbers are N_order, N_MH, N_allele_max, length【After the item is stored, it is not used in subsequent code】"""
signal_sum = np.zeros([N_order, N_MH, N_allele_max, 1])  # Create an empty array to hold the sum of "all" base assignment sequence sequencing signals, i.e. "Total number of bases for this signal"
"""//singal_sum: From the outer layer to the inner layer, the index numbers are respectively N_order, N_MH, N_allele_max, 1"""
signal_phase = []  # Create an empty array to hold "all complete and phasable" base assignment sequence sequencing signals
"""//signal_phase: From the outer layer to the inner layer, the index numbers are respectively N_order_phase, N_MH, N_allele_max, length"""

for order in range(N_order):
    dispensation = orders[order]  
    current_signal = np.zeros([N_MH, N_allele_max, length])  

    for mh in range(N_MH):  

        for allele in range(N_allele_max):  
            signal = [0]*length  
            current_seq = seq[mh][allele]  
            len_allele = len(seq[mh][0]) 

            j = 0  # 在“待测序列”上进行碱基个数的计数
            for i in range(1, len(dispensation)+1):  # 对碱基分配顺序的碱基个数进行计数：‘1~碱基分配顺序总长度’（range为左闭右开）
                """//j索引的是MH的等位基因的第j个碱基；i索引的是碱基分配顺序中的第i-1个碱基"""

                count = 0  # 在“碱基分配顺序”上进行测序信号的计数（0-若干）
                """//对碱基分配顺序中的每个碱基都单独进行测序信号计数"""
                try:

                    while current_seq[j] == dispensation[i - 1]:  # 当“待测序列”当前碱基和“分配顺序”当前碱基一致时，进行信号的叠加
                        """
                        //判断当前提取的MH等位基因碱基是否与测序信号碱基匹配，匹配则认为测序信号覆盖了该碱基
                        //当覆盖了某个碱基时，在当前碱基分配顺序上信号+1，并将该结果储存到signal中；同时将比对的等位基因碱基后移一位，判
                        //断第j+1个等位基因碱基是否被接下来的测序信号覆盖，依次循环
                        """
                        count += 1  # 在“碱基分配顺序“上该位置的测序信号+1
                        signal[i - 1] = count  # 当前测序信号加1
                        j += 1  # ”待测序列“上该位置碱基个数加1，”待测序列“往后移一位

                        if j >= len(current_seq):  # 当待测序列的碱基计数超过待测序列最大长度时
                            break  # 跳出while循环

                    if j >= len(current_seq):  # 当待测序列的碱基计数超过待测序列最大长度时
                        break  # 跳出for循环

                except TypeError:  # 待测序列中有些MH的allele数量较少，因此这些位置是NaN
                    break

            current_signal[mh, allele] = signal  # 获得“当前”碱基分配顺序下的测序信号
            """//current_signal: 此处索引的是单个MH下单个等位基因，在当前操作的碱基分配顺序下的信号值"""
            signal_sum[order, mh, allele] = sum(signal)  # “当前”碱基分配顺序下的测序信号之和，即获得”当前“碱基分配顺序下的待测序列的碱基总数
            """//signal_sum: 此处索引的是单个MH下单个等位基因，在当前碱基分配顺序下的信号总值（也即该等位基因被检测到的碱基数）"""
    signal_all[order] = current_signal  # 将“当前”碱基分配顺序下的测序信号，存放在“所有”碱基分配顺序的测序信号变量中
    """//singal_all: 此处索引的是单个MH下单个等位基因，在操作的碱基分配顺序下的信号值，该数组储存的是所有剪辑分配顺序下的结果"""


"""
//part2
//func: 依据刚刚储存的sum(signal)信息，判断每一种碱基分配顺序是否能完整覆盖特定MH的特定等位基因，对于不能覆盖特定MH的特定等位基因者，将其
//输出至order_part中储存
"""
'''找出无法生成完整信号的碱基分配顺序相应'''
'''在全部碱基分配顺序（orders）中，对每一条碱基分配顺序分别进行相应测序信号的加和（signal_sum），并与相应待测序列上的碱基总数比较'''
'''若 信号加和 小于 待测序列碱基总数，那么该碱基分配顺序无法生成完整信号，存在order_part中。'''
for order in range(N_order):  # 在所有碱基分配顺序中遍历
    order_current = np.array(orders[order])  # 获得当前碱基分配顺序
    """//order_current: 用于在满足条件时储存的碱基分配顺序"""

    for mh in range(N_MH):  # 在所有MH中遍历
        len_allele = len(seq[mh][0])  # 获得当前MH的待测序列碱基个数，即待测序列长度
        """len_allele: 用于与sum(signal)信息判断该条碱基分配顺序是否可用"""

        for allele in range(N_allele_max):  # 在所有allele中遍历（包含NaN)

            if 0 < signal_sum[order, mh, allele] < len_allele:  # 若信号的碱基个数小于待测序列碱基个数，说明该碱基分配顺序只能生成不完整的信号，后续将会删除。
                """//比较sum(signal)和len_allele的关系"""

                order_part = np.vstack((order_part, order_current))  # 将当前碱基分配顺序存在order_part中
                current_signal_part = [signal_all[order]]  # 获得不完整的信号
                """//【current_signal_part在此代码块、整个文件中均未使用到】"""
                break  # 跳出最内层for循环

        else:
            """//【此处else的位置是不是错位了？应该对应的是上面的if？下面continue同】"""
            continue

        break  # 跳出第二层for循环，开始检查下一个碱基分配顺序是否可以得到完整型号


"""
//part3
//func: 在所有生成的碱基分配顺序中，剔除无法完整覆盖某一MH中某一等位基因的碱基分配顺序（也即保留能覆盖所有MH中所有等位基因的碱基分配顺序）
"""
order_part = np.delete(order_part, 0, 0)  # 把order_part第一行的0删去，余下各行均为无法生成完整信号的碱基分配顺序
'''通过取差集，获得生成信号完整的碱基分配顺序'''
# 复制存放“全部信号”的碱基分配顺序数组（orders），存为orders_copy
orders_copy = orders.view([('', orders.dtype)]*orders.shape[1])
"""//orders_copy在生成时，在最内层增加了一层'()'，其类型为np.void（eg.[('A', 'C', 'T', 'A', 'T', ...)]）"""
# 复制存放“不完整信号”的碱基分配顺序数组（order_part），存为order_part_copy
order_part_copy = order_part.view([('', order_part.dtype)]*order_part.shape[1])
"""//order_part_copy生成时，也在最内层增加了一层'()'，其类型为np.void（eg.[('A', 'C', 'T', 'A', 'T', ...)]）"""
# 用上述两者的差集，获得“完整信号”的碱基分配顺序，存在order_full中
order_full = np.setdiff1d(orders_copy, order_part_copy).view(orders.dtype).reshape(-1, orders.shape[1])
"""//order_full: 取完差集后恢复为正常的np.array"""

"""
//part4
//func: 按照part1的逻辑，对于能生成完整信号的碱基分配顺序进行信号记录
"""
'''生成信号完整的碱基分配顺序的相应信号，为下一步phase检验做准备'''
N_order_full = order_full.shape[0]  # 获得完整信号的碱基分配顺序条数
signal_full = np.zeros([N_order_full, N_MH, N_allele_max, length])  # 创建一个空数组,用来存放“完整信号”碱基分配顺序的测序信号
"""//signal_full格式与signal_all一致，储存测序信号，由外层到内层，索引数分别为N_order, N_MH, N_allele_max, length"""
for order in range(N_order_full):
    dispensation2 = order_full[order]  # 导入碱基分配顺序
    current_signal_full = np.zeros([N_MH, N_allele_max, length])  # 创建一个空数组,用来存放“当前”碱基分配顺序下的测序信号

    for mh in range(N_MH):  # 在每个MH中遍历

        for allele in range(N_allele_max):
            signal = [0]*length  # 焦磷酸测序信号的长度,并在此基础上进行相应信号的计数
            current_seq = seq[mh][allele]  # 获取“当前”待测序列
            len_allele = len(seq[mh][0])  # 获取“当前”待测序列的碱基个数
            """//【len_allele在此代码块中未使用到】"""

            j = 0  # 在“待测序列”上进行碱基个数计数
            for i in range(1, len(dispensation2)+1):  # 对碱基分配顺序的碱基个数进行计数：‘1~碱基分配顺序总长度’（range为左闭右开）

                count = 0  # 在“碱基分配顺序”上测序信号进行计数
                try:

                    while current_seq[j] == dispensation2[i - 1]:  # 当"待测序列"当前碱基和"分配顺序"当前碱基一致时，进行信号的叠加
                        count += 1  # 在“碱基分配顺序”上该位置的测序信号+1
                        signal[i - 1] = count  # 当前测序信号加1
                        j += 1  # “待测序列”往后移一位

                        if j >= len(current_seq):  # 当待测序列的碱基计数超过待测序列最大长度时
                            break  # 跳出while循环

                    if j >= len(current_seq):  # 当待测序列的碱基计数超过待测序列最大长度时
                        break  # 跳出for循环

                except TypeError:  # 待测序列中有些MH的allele数量较少，因此这些位置是NaN
                    break

            current_signal_full[mh, allele] = signal  # 获得“当前”碱基分配顺序下的测序信号
            """//【原本为current_signal，现改为current_signal_full】"""
    signal_full[order] = current_signal_full
    """//【原本为current_signal，现改为current_signal_full】"""


"""
//part5
//func: 在每一碱基分配顺序下，对每一MH任意两个等位基因信号叠加后的信号值进行检测，保留能识别所有不同等位基因信号叠加情况的碱基分配顺序
"""
'''进行phase检验'''
for order in range(N_order_full):  # 在每个碱基分配顺序下

    x = 0  # 用来对能够phase的MH进行计数，当x = dim1时，即所有待测MH都可以phase, 则认为该碱基分配顺序达到能够phase的要求
    """//x: 每有一个MH可以被phase，则计数+1，当x==N_MH时，代表任意MH在该碱基分配顺序均可被phase"""
    for mh in range(N_MH):  # 遍历每个MH

        current_mh = signal_full[order][mh]  # 获得当前MH下各个allele信号
        """//current_mh: 二维数组，储存在特定碱基分配顺序下特定MH下各个等位基因的信号值，用于所有不同等位基因信号叠加情况下能否被识别"""
        if np.all(current_mh[-1] == 0):  # 因为有的MH的allele数量较少，这些MH的最后一行信号一定全为0
            """
            //针对MH最后一个等位基因的信号值是否全0来进行不同的信号叠加方式
            //若最后一个等位基因信号值为全0，则需先从current_mh中去除掉所有全0信号，再进行组合、判断
            //若最后一个等位基因信号值非全0，则直接进行组合并进行判断
            """
            signal_pair = []  # 用来存放两两allele信号加和的列表signal_pair

            current_mh_copy = current_mh.view([('', current_mh.dtype)] * current_mh.shape[1])  # 复制当前MH下各个allele信号
            current_0_copy = current_mh[-1].view([('', current_mh[-1].dtype)] * current_mh[-1].shape[0])   # 复制当前MH下全为0的信号
            c = np.setdiff1d(current_mh_copy, current_0_copy).view(current_mh.dtype).reshape(-1, current_mh.shape[1])  # 取上述两者的差集，即为各个allele的信号

            f = list(itertools.combinations_with_replacement(c, 2))  # 将allele信号两两加和（有放回的全组合）
            """//f: 列表，其中每一项是一个元组，每一元组中包含任意两个等位基因信号值的np.array；有放回的组合是考虑了纯合子情况"""
            """
            //Q: 此处为两两加和，考虑的就是该组信号分配顺序能否对单一个体某一MH上2个等位基因进行phase，但用PSQ解决混合问题时，一组信号
            //分配顺序是要对mix中的>2个等位基因进行识别？适用于单一样本的碱基分配顺序能在mix样本中同样适用吗？不知道我理解的对不对，就是
            //看到这想到的一个问题
            """

            for i in f:  # 信号相加
                """//i: 列表f中的元组i，i[0]、i[1]分别为两个等位基因信号值的np.array"""
                add = i[0] + i[1]
                """//add: 两等位基因加和后的信号值，仍为np.array"""
                signal_pair.append(add)  # 加和信号存放在f中

            w1 = []  # 对“一个MH下各个等位基因信号两两加和的信号”是否含有重复值进行检查
            for j in signal_pair:  # 遍历d中的所有加和信号
                j = j.tolist()

                if j not in w1:
                    w1.append(j)

            if len(w1) == len(signal_pair):  # 检查最终w1和d的长度是否相等，若相等，则说明该碱基分配顺序对当前MH信号可以phase
                x += 1  # 可以phase 则x加1

        else:
            signal_pair = []  # 用来存放两两allele信号加和的列表
            f = list(itertools.combinations_with_replacement(current_mh, 2))

            for i in f:
                add = i[0] + i[1]
                signal_pair.append(add)
            w1 = []  # 对“一个MH下各个等位基因信号两两加和”进行去重

            for j in signal_pair:
                j = j.tolist()

                if j not in w1:
                    w1.append(j)

            if len(w1) == len(signal_pair):
                x += 1  # 用来对能够phase的MH进行计数，x = dim1时，认为可以phase

    if x == N_MH:
        order_phase = np.vstack((order_phase, order_full[order]))  # 把可以phase的碱基分配顺序保存下来
        signal_phase.append(signal_full[order])  # 把可以phase的碱基分配顺序保存下来
        """//signal_phase: 为在列表中储存碱基分配顺序的信号值（np.array）；在后续已使用np.array()进行了转化"""

order_phase = np.delete(order_phase, 0, 0)
signal_phase = np.array(signal_phase)


"""
//part6
//func: 计算各个
"""
'''对测序信号进行相关系数计算'''
cor = []  # 创建一个列表，用来存放各个碱基分配顺序下各个MH的"最大""的相关系数
N_order_phase = len(order_phase)
"""//N_order_phase: 可完整覆盖所有MH的等位基因，且能对任意MH中任意等位基因组合进行phase的碱基分配顺序数"""
for order in range(N_order_phase):  # 在每个碱基分配顺序中遍历

    cor_MH = []  # 创建一个列表，用来存放"各个MH"的allele的最大相关系数
    for mh in range(N_MH):  # 在每个MH中遍历，生成测序信号
        current_signal = signal_phase[order][mh]  # 获得当前信号
        # 把current_signal中全为0的行删去
        current_signal = current_signal[[not np.all(current_signal[i] == 0) for i in range(current_signal.shape[0])], :]
        """//current_signal: 去除全0行后的、单个碱基分配顺序下MH所有等位基因的信号值"""

        cor_matrix = np.corrcoef(current_signal)  # 对矩阵进行相关系数计算
        """
        //np.correcoef: 计算行与行之间的相关系数，返回Pearson乘积矩相关系数（eg.cor_matrix[i][j]计算的是第i行和第j行的相关系数）
        //此处计算的即为各个等位基因生成的信号值数组之间的相关性
        """
        row, col = np.diag_indices_from(cor_matrix)  # 提取相关系数矩阵对角线
        cor_matrix[row, col] = 0  # 对角线赋值为0

        cor_allele = np.nanmax(cor_matrix)  # 把当前MH中各个allele间相关系数最大的值存入列表
        """//np.nanmax: 排除nan值后的最大值计算"""
        """Q: 将各个等位基因信号值之间的最大相关性的目的是什么呢？"""
        cor_MH.append(cor_allele)  # 每个MH的最大相关系数存入列表

    cor_MH_max = max(cor_MH)  # Assign cor_MH_max the maximum correlation coefficient for each MH in the current base classification order
    cor.append(cor_MH_max) 

cor = np.array(cor)  # Convert the maximum correlation coefficients corresponding to all base assignment orders into an array
cor = cor.reshape(cor.shape[0], 1)  

"""
//part7
//func: Format adjustment and result storage: base assignment order + correlation (in descending order of correlation)
"""
final_result = np.concatenate((order_phase, cor), axis=1)
final_result = pd.DataFrame(final_result)
final_result['dispensation order'] = final_result[final_result.columns[0:length]].apply(
    lambda x: ''.join(x.astype(str)), axis=1)
final_result['cor_matrix'] = final_result[final_result.columns[length]]
final_result = final_result.drop(final_result.columns[0:length+1], axis = 1)

final_result1 = final_result.sort_values(by=['cor_matrix'], ascending=True)  # 排序
#final_result1.to_csv("sequence_list-disp order_test.csv")

import os
print(os.getcwd())