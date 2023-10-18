import statsmodels.api as sm
from dtw import dtw
import math

import pandas as pd
import numpy as np
from cif import cif
# from QuantLib.utils import align_dataframes
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set(style='whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

'''ZScore'''
def ZScore(Data):
    '''
    :param Data: 待标准化数据
    :return: 标准化后的数据
    '''
    Data_ZScore=(Data-Data.mean())/Data.std()
    return Data_ZScore

'''滤波器'''
def HP_Filter_Oneside(Data, Lambda):
    '''
    :param Data: 待滤波数据
    :param Lambda: 滤波参数 年频：100；季频：1600；月频：14400
    :return: HP滤波后的Cycle项和Trend项
    '''

    def At(t: int, Lambda):
        # e_t:tx1，最后一行是1，其余是0
        e_t = np.zeros(t)
        e_t[-1] = 1

        # I_t:t-2的单位阵
        I_t = np.identity(t - 2)

        # Q_t:二阶差分矩阵，(t-2)xt
        Q_t = np.zeros((t - 2, t))  # 先设置shape
        # 再通过循环设置每一行的值
        for i in range(t - 2):
            Q_t[i, i], Q_t[i, i + 1], Q_t[i, i + 2] = 1, -2, 1

        # 通过矩阵运算，计算常数阵
        # @:矩阵乘法； Matrix.T:矩阵转置； Matrix.I：矩阵求逆
        A_t = np.matrix(e_t) @ (Q_t.T) @ (np.linalg.inv(Q_t @ (Q_t.T) + I_t / Lambda)) @ Q_t
        # 结果是一个1xt的矩阵
        return A_t

    Data_Local = Data.copy()
    data_series = np.array(Data_Local)  # nx1的matrix
    length = len(Data)

    list_cycle = [math.nan, math.nan]  # t=1,2时是没有的，用math.nan填充
    for i in range(2, length):
        # t=i+1
        sub_series = data_series[:i + 1]  # 一共有i+1=t项
        sub_A_t = At(i + 1, Lambda)
        cycle_t = (sub_A_t @ sub_series)[0, 0]
        list_cycle.append(cycle_t)
    Data_Local['Cycle'] = list_cycle
    Data_Local['Trend'] = Data[Data.columns[0]] - np.array(list_cycle)
    return Data_Local['Cycle'], Data_Local['Trend']

def HP_Filter_Twoside(Data, Lambda=1600):
    '''
    :param Data: 待滤波数据
    :param Lambda: 滤波参数 年频：100；季频：1600；月频：14400
    :return: HP滤波后的Cycle项和Trend项
    '''
    Cycle, Trend = sm.tsa.filters.hpfilter(Data, Lambda)
    return Cycle, Trend


'''领先性判断算法'''
#时滞相关性
def Shift_Corr(Data1, Data2, ShiftRange=range(-20, 21), Method='pearson', ShowFig=False):
    '''
    :param Data1: 数据1（DataFrame格式）
    :param Data2: 数据2（DataFrame格式，被移动，Shift负期数可以理解为，数据1对数据2的领先，如Shift -5相关性最强，则数据1领先数据2 5期）
    :param ShiftRange: 回测移动期数范围
    :param Method:  相关系数计算方法 ①’Pearson‘-普通IC；②’Spearman‘-RankIC
    :param ShowFig: 是否要显示相关性序列图
    :return: Data1 和 Data2 前后滞后的相关性序列
    '''

    Corr_Series = pd.DataFrame(index=ShiftRange, columns=['Corr'])

    for _ in ShiftRange:
        Shift_Matrix = pd.concat([Data1, Data2], axis=1).dropna(axis=0)
        Shift_Matrix.columns = ['Data1', 'Data2']
        Shift_Matrix.sort_index(ascending=True, inplace=True)

        Shift_Matrix['Data2'] = Shift_Matrix['Data2'].shift(_)
        Shift_Matrix.dropna(axis=0,inplace=True)
        Corr_Series.loc[_, 'Corr'] = Shift_Matrix['Data1'].corr(Shift_Matrix['Data2'], method=Method)

    LeadingTerm = Corr_Series.index[np.argmax(Corr_Series)]

    if ShowFig:
        plt.plot(Corr_Series)
        plt.title('Corr')

        plt.show()

    return Corr_Series, LeadingTerm

#K-L距离
def K_L(Data1, Data2, ShiftRange=range(-20, 21), NegCorr_Indicator=False, ShowFig=False):
    '''
    :param Data1: 数据1（DataFrame格式）
    :param Data2: 数据2（DataFrame格式，被移动，Shift负期数可以理解为，数据1对数据2的领先，如Shift -5相关性最强，则数据1领先数据2 5期）
    :param ShiftRange: 回测移动期数范围
    :param NegCorr_Indicator: 如果两个指标本身具有逻辑负相关性，则需要先取相反数，然后再进行计算
    :param ShowFig: 是否要显示相关性序列图
    :return: Data1 和 Data2 前后滞后的相关性序列
    '''
    Corr_Series = pd.DataFrame(index=ShiftRange, columns=['K_L_Info'])

    if NegCorr_Indicator:  # 如果指标间本身逻辑存在负相关性，则需要先取相反数，然后再进行计算
        Data2 = -Data2

    for _ in ShiftRange:
        Shift_Matrix = pd.concat([Data1, Data2], axis=1).dropna(axis=0)
        Shift_Matrix.columns = ['Data1', 'Data2']
        Shift_Matrix.sort_index(ascending=True, inplace=True)

        Shift_Matrix['Data2'] = Shift_Matrix['Data2'].shift(_)
        Shift_Matrix.dropna(axis=0,inplace=True)

        Shift_Matrix['Data1'] = Shift_Matrix['Data1'] - Shift_Matrix['Data1'].min() + 1
        Shift_Matrix['Data2'] = Shift_Matrix['Data2'] - Shift_Matrix['Data2'].min() + 1
        #
        Data1_Prob = Shift_Matrix['Data1'] / Shift_Matrix['Data1'].sum()
        Data2_Prob = Shift_Matrix['Data2'] / Shift_Matrix['Data2'].sum()

        K_L = Data1_Prob.values * np.log(Data1_Prob.values / Data2_Prob.values)
        Corr_Series.loc[_, 'K_L_Info'] = K_L.sum()

    LeadingTerm = Corr_Series.index[np.argmin(Corr_Series)]

    if ShowFig:
        plt.plot(Corr_Series)
        plt.title('K_L_Info')

        plt.show()

    return Corr_Series,LeadingTerm

#DTW距离
def DTW(Data1, Data2):
    '''
    :param Data1: 数据1（DataFrame格式）
    :param Data2: 数据2（DataFrame格式）
    :return: DTW距离
    '''
    Data = pd.concat([Data1, Data2], axis=1).dropna(axis=0)
    Data.columns = ['Data1', 'Data2']
    Data1 = np.array(Data['Data1'].sort_index(ascending=True).dropna().values).reshape(-1, 1)
    Data2 = np.array(Data['Data2'].sort_index(ascending=True).dropna().values).reshape(-1, 1)

    Manhattan_Distance = lambda x, y: np.abs(x - y)
    DTW_Distance, Cost_Matrix, Acc_Cost_Matrix, Path = dtw(Data1, Data2, dist=Manhattan_Distance)

    return DTW_Distance / len(Data)

#Bry-Boschan算法
def Detect_Tp_of_Time_Series(series: pd.Series, minimal_cycle_length=15, minimal_phase_length=5, log_file=None):
    """
    Bry-Boschan算法识别时间序列拐点

    Parameters:
    ------
    series: Series
        时间序列
    minimal_cycle_length: int
        一个周期最小持续长度要求
    minimal_phase_length: int
        一个波峰或者波谷的最小长度要求
    log_file: str
        日志文件名称

    Returns:
    -------
    indicator: series
        拐点标志，1代表顶点，-1代表底点，0代表其它
    """
    df = series.to_frame()

    if isinstance(log_file, str):
        log_file = open(log_file, 'w')

    # a) Looking for local minimal/maximal
    col_ind_local = cif.getLocalExtremes(df, showPlots=False)

    # b) Check the turning points alterations
    col_ind_neigh = cif.checkNeighbourhood(df=df, indicator=col_ind_local, showPlots=False, saveLogs=log_file)
    col_ind_alter = cif.checkAlterations(df=df, indicator=col_ind_neigh, keepFirst=False, showPlots=False, saveLogs=log_file)

    # c) Check minimal lenth of cycles
    col_ind_cyclelength = cif.checkCycleLength(df=df, indicator=col_ind_alter, cycleLength=minimal_cycle_length,
                                               showPlots=False, saveLogs=log_file)

    # d) Check the turning points alterations again
    col_ind_neigh_again = cif.checkNeighbourhood(df=df, indicator=col_ind_cyclelength, showPlots=False, saveLogs=log_file)
    col_ind_alter_again = cif.checkAlterations(df=df, indicator=col_ind_neigh_again, keepFirst=False, showPlots=False,
                                               saveLogs=log_file)

    # e) Check minimal length of phase
    col_ind_phaselength = cif.checkPhaseLength(df=df, indicator=col_ind_alter_again, keepFirst=False,
                                               phaseLength=minimal_phase_length, showPlots=False,
                                               saveLogs=log_file)

    # f) Check the turning points alterations for the last time
    col_ind_neigh_last = cif.checkNeighbourhood(df=df, indicator=col_ind_phaselength, showPlots=False, saveLogs=log_file)
    col_ind_turning_points = cif.checkAlterations(df=df, indicator=col_ind_neigh_last, keepFirst=False, showPlots=False,
                                                  saveLogs=log_file)

    return col_ind_turning_points[col_ind_turning_points.columns[0]]

def Match_Tp_of_Two_Series(series1=None, series2=None, indicator1=None, indicator2=None,
                           freq_scaler=1, minimal_cycle_length1=15, minimal_cycle_length2=15,
                           minimal_phase_length1=5, minimal_phase_length2=5, log_file=None,
                           look_back_cache=6, lookforward_cache=6, return_details=False):
    """两个序列的拐点匹配"""

    if isinstance(log_file, str):
        log_file = open(log_file, 'w')

    if series1 is not None:
        indicator1 = Detect_Tp_of_Time_Series(series1, minimal_cycle_length1, minimal_phase_length1, log_file)
    indicator1 = indicator1[indicator1 != 0.0]
    if series2 is not None:
        indicator2 = Detect_Tp_of_Time_Series(series2, minimal_cycle_length2, minimal_phase_length2, log_file)
    indicator2 = indicator2[indicator2 != 0.0]

    if indicator2.index.min() < indicator1.index.min():
        indicator2 = indicator2.iloc[np.where(indicator2.index < indicator1.index.min())[0][-1]:]
    if indicator2.index.max() > indicator1.index.max():
        indicator2 = indicator2.iloc[:np.where(indicator2.index > indicator1.index.max())[0][0]+1]

    matched = 0
    no_data = len(indicator1[indicator1.index < indicator2.index.min()])
    leading_periods = []
    matched_periods_of_series1 = []
    matched_periods_of_series2 = []
    for dt1, tp1 in indicator1.iloc[no_data:].iteritems():

        # 匹配规则：寻找series1拐点前后相临近的series2拐点,若是向前匹配，则还要判断向前时段是否符合条件。
        idx = np.where(indicator2.index <= dt1)[0][-1]
        latest_tp2 = indicator2.iat[idx]
        lastet_dt2 = indicator2.index[idx]

        # Check if the turning point of series2 matches the one of series1
        if tp1 == latest_tp2:
            if ((dt1 - lastet_dt2) / pd.Timedelta(1, 'D') <= look_back_cache * freq_scaler):
            # if ((dt1 - lastet_dt2) / pd.Timedelta(1, 'D') >= look_back_cache * freq_scaler) or (len(indicator1.loc[lastet_dt2:dt1]) >1): #原始版本
                # continue
                matched += 1
                leading_periods.append(int((dt1 - lastet_dt2) / pd.Timedelta(1, 'D') / freq_scaler))
                matched_periods_of_series1.append(dt1)
                matched_periods_of_series2.append(lastet_dt2)
        else:
            idx = np.where(indicator2.index <= dt1+pd.Timedelta(int(lookforward_cache*freq_scaler), 'D'))[0][-1]
            latest_tp2 = indicator2.iat[idx]
            lastet_dt2 = indicator2.index[idx]
            if tp1 == latest_tp2:
                matched += 1
                leading_periods.append(int((dt1 - lastet_dt2) / pd.Timedelta(1, 'D') / freq_scaler))
                matched_periods_of_series1.append(dt1)
                matched_periods_of_series2.append(lastet_dt2)

    # 匹配率：序列1的匹配数 / (序列1的总拐点数 - 无数据拐点数)
    mached_ratio = matched / (len(indicator1) - no_data)

    # 多余率: 指标多余拐点 / 指标总拐点
    redundant_ratio = (len(indicator2) - matched) / len(indicator2)

    # 平均领先阶数
    if leading_periods:
        avg_leading_periods = np.mean(leading_periods)
        std_leading_periods = np.std(leading_periods)
    else:
        avg_leading_periods = np.nan
        std_leading_periods = np.nan

    result_dict = {
        '无数据拐点': no_data,
        '可匹配拐点': matched,
        '有效拐点总数': len(indicator1) - no_data,
        '匹配率': mached_ratio,
        '多余率': redundant_ratio,
        '平均领先阶数': avg_leading_periods,
        '领先阶数标准差': std_leading_periods
    }
    mached_periods = pd.DataFrame(
        [matched_periods_of_series1, matched_periods_of_series2],
        index=[indicator1.name, indicator2.name]).T

    if return_details:
        return result_dict, {'匹配时点': mached_periods}
    # return result_dict
    return result_dict,leading_periods

def Compare_Two_Series(series1, series2, ind1, ind2, save_plots=False, show_plots=True,
                       matched_periods=None):
    """比较两个时间序列的拐点"""
    # series1, series2, ind1, ind2 = align_dataframes(series1, series2, ind1, ind2, axis='index', join='inner')
    times = np.arange(len(series1))

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    ax2 = ax1.twinx()

    # 设置x坐标, 6个ticks
    xtick_nums = 6
    xtick_values = np.linspace(0, len(times)-1, xtick_nums, dtype='int32', endpoint=True)
    xtick_labels = list(series1.index.strftime("%Y-%m-%d")[xtick_values])
    plt.xticks(xtick_values, xtick_labels, rotation=45)

    # 划曲线，高点和底点用几何图形标注
    peak_marker = '^'
    peak_xticks1 = times[ind1.values == 1]
    peak_yticks1 = series1.values[peak_xticks1]

    trough_marker = 'v'
    trough_xticks1 = times[ind1.values == -1]
    trough_yticks1 = series1.values[trough_xticks1]

    color1 = 'firebrick'
    color2 = 'grey'
    line1 = ax1.plot(times, series1, linestyle='-', color=color1, label=series1.name)[0]
    line2 = ax2.plot(times, series2, linestyle='-', color=color2, label=series2.name)[0]

    import matplotlib.ticker
    nticks = 6
    ax1.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
    ax2.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))

    ax2.grid(None)
    ax1.grid(True, linestyle='--', linewidth=1.0)
    scatter1 = ax1.scatter(peak_xticks1, peak_yticks1, marker=peak_marker, color='cornflowerblue', s=100)
    scatter2 = ax1.scatter(trough_xticks1, trough_yticks1, marker=trough_marker, color='navajowhite', s=100)
    ax1.legend([line1, line2, scatter1, scatter2], [series1.name, series2.name, '峰', '谷'], fontsize='x-small')

    peak_xticks2 = times[ind2.values == 1]
    peak_yticks2 = series2.values[peak_xticks2]
    trough_xticks2 = times[ind2.values == -1]
    trough_yticks2 = series2.values[trough_xticks2]

    ax2.scatter(peak_xticks2, peak_yticks2, marker=peak_marker, color='cornflowerblue', s=100)
    ax2.scatter(trough_xticks2, trough_yticks2, marker=trough_marker, color='navajowhite', s=100)

    # 匹配的拐点用箭头表示
    from matplotlib.patches import ConnectionPatch
    if isinstance(matched_periods, pd.DataFrame):
        for dt1, dt2 in matched_periods[[ind1.name, ind2.name]].values:

            xtick_ax1 = times[np.where(ind1.index == dt1)[0][0]]
            ytick_ax1 = series1.loc[dt1]
            xtick_ax2 = times[np.where(ind2.index == dt2)[0][0]]
            ytick_ax2 = series2.loc[dt2]

            con = ConnectionPatch(xyA=(xtick_ax1, ytick_ax1), xyB=(xtick_ax2, ytick_ax2),
                                  coordsA="data", coordsB="data", arrowstyle="<->",
                                  axesA=ax1, axesB=ax2, shrinkA=0, shrinkB=0)
            con.set_color('black')
            con.set_linewidth(1)
            con.set_linestyle('--')
            ax2.add_artist(con)

    plt.subplots_adjust(hspace=0.0)
    fig.tight_layout()

    if show_plots:
        plt.show()

    if save_plots:
        plt.savefig(save_plots, dpi=300)

    plt.close(fig)

'''PMI数据预处理'''
def PMI_To_YOY(Data,Oneside=True,Cycle_Term=14400):
    '''
    PMI环比 -> PMI指数 -> 滤波后取趋势项
    :param Data: PMI原始数据
    :param Oneside: 是否进行单边HP滤波
    :param Cycle_Term: HP滤波参数 年频：100；季频：1600；月频：14400
    :return:PMI指数滤波后的循环项
    '''
    PMI = 1 + (Data - 50) / 100
    PMI_Index=PMI.cumprod() / PMI.values[0]

    if Oneside:
        Cycle,Trend=HP_Filter_Oneside(PMI_Index,Cycle_Term)
    else:
        Cycle,Trend=HP_Filter_Twoside(PMI_Index,Cycle_Term)

    return PMI_Index,Cycle,Trend,pd.DataFrame(Cycle.diff(12).dropna(axis=0))

'''数据领先性检验'''
def LeadingEffect_Test(Data,Test_Series_Name,ShiftRange=range(-20,21)):
    Test_Series= Data[[Test_Series_Name]]
    Indicators=Data.columns[1:]
    Test_Result=pd.DataFrame(index=Indicators,columns=['时滞相关性','时滞检验领先期数','K-L信息量','K-L领先期数','DTW距离','Bry-Boschan匹配率','Bry-Boschan领先期数','Bry-Boschan领先期数标准差','Bry-Boschan领先期数序列'])

    for Indicator in Indicators:
        Tested_Series=Data[[Indicator]]
        Data_New=pd.concat([Test_Series,Tested_Series],axis=1).dropna(axis=0)
        Data1=Data_New[[Test_Series_Name]]
        Data1_=Data_New[Test_Series_Name]
        Data2=Data_New[[Indicator]]
        Data2_=Data_New[Indicator]
        Corr_Series, Corr_LeadingTerm=Shift_Corr(Data1,Data2,ShiftRange=ShiftRange,Method='pearson',ShowFig=False)
        K_L_Corr_Series,K_L_LeadingTerm=K_L(Data1,Data2,ShiftRange=ShiftRange,NegCorr_Indicator=False,ShowFig=False)
        DTW_Distance=DTW(ZScore(Data1),ZScore(Data2))

        Data1_Ind=Detect_Tp_of_Time_Series(Data1_, minimal_cycle_length=15, minimal_phase_length=5, log_file=None)
        Data2_Ind=Detect_Tp_of_Time_Series(Data2_, minimal_cycle_length=15, minimal_phase_length=5, log_file=None)


        Bry_Boschan_Result_dict,Bry_Boschan_Leading_periods=Match_Tp_of_Two_Series(series1=Data1_, series2=Data2_, indicator1=None, indicator2=None,
                                                                                   freq_scaler=28, minimal_cycle_length1=15, minimal_cycle_length2=15,
                                                                                   minimal_phase_length1=5, minimal_phase_length2=5, log_file=None,
                                                                                   look_back_cache=8, lookforward_cache=8, return_details=False)

        Compare_Two_Series(Data1_, Data2_, Data1_Ind, Data2_Ind, save_plots=False, show_plots=True,
                           matched_periods=None)
        print(Bry_Boschan_Leading_periods)

        Test_Result.loc[Indicator,'时滞相关性']=Corr_Series.max().mean() #取值
        Test_Result.loc[Indicator,'时滞检验领先期数']=Corr_LeadingTerm
        Test_Result.loc[Indicator,'K-L信息量']=K_L_Corr_Series.min().mean() #取值
        Test_Result.loc[Indicator,'K-L领先期数']=K_L_LeadingTerm
        Test_Result.loc[Indicator,'DTW距离']=DTW_Distance
        Test_Result.loc[Indicator,'Bry-Boschan匹配率']=Bry_Boschan_Result_dict['匹配率']
        Test_Result.loc[Indicator,'Bry-Boschan领先期数']=Bry_Boschan_Result_dict['平均领先阶数']
        Test_Result.loc[Indicator,'Bry-Boschan领先期数标准差']=Bry_Boschan_Result_dict['领先阶数标准差']
        Test_Result.loc[Indicator,'Bry-Boschan领先期数序列']=Bry_Boschan_Leading_periods




    return Test_Result

'''发电量'''

OriginalData=pd.read_excel('D:\Python\MacroFactorModel\MacroDataAnalysis\PMI_LeadingEffects.xlsx',sheet_name='PMI_Diff12',index_col=0,header=0)
Data_New=LeadingEffect_Test(OriginalData,'PMI_Diff12')
#
# Data_New.to_excel('Test_Result.xlsx')

Data1_= OriginalData['PMI_Diff12']
Data2_=OriginalData['铝材产量']

# Bry_Boschan_Result_dict,Bry_Boschan_Leading_periods=Match_Tp_of_Two_Series(series1=Data1_, series2=Data2_, indicator1=None, indicator2=None,
#                                                                                    freq_scaler=28, minimal_cycle_length1=15, minimal_cycle_length2=15,
#                                                                                    minimal_phase_length1=5, minimal_phase_length2=5, log_file=None,
#                                                                                    look_back_cache=12, lookforward_cache=12, return_details=False)



# freq_scaler=28
# minimal_cycle_length1=15
# minimal_cycle_length2=15
# minimal_phase_length1=5
# minimal_phase_length2=5
# look_back_cache=0
# lookforward_cache=6
#
# Data1_Ind = Detect_Tp_of_Time_Series(Data1_, minimal_cycle_length=15, minimal_phase_length=5, log_file=None)
# Data2_Ind = Detect_Tp_of_Time_Series(Data2_, minimal_cycle_length=15, minimal_phase_length=5, log_file=None)
# indicator1 = Data1_Ind[Data1_Ind != 0.0]
# indicator2 = Data2_Ind[Data2_Ind != 0.0]
#
# if indicator2.index.min() < indicator1.index.min():
#     indicator2 = indicator2.iloc[np.where(indicator2.index < indicator1.index.min())[0][-1]:]
# if indicator2.index.max() > indicator1.index.max():
#     indicator2 = indicator2.iloc[:np.where(indicator2.index > indicator1.index.max())[0][0] + 1]
#
# matched = 0
# no_data = len(indicator1[indicator1.index < indicator2.index.min()])
# leading_periods = []
# matched_periods_of_series1 = []
# matched_periods_of_series2 = []
#
# _=0
# dt1,tp1=list(indicator1.iloc[no_data:].iteritems())[_]
# idx = np.where(indicator2.index <= dt1)[0][-1]
# latest_tp2 = indicator2.iat[idx]
# lastet_dt2 = indicator2.index[idx]




# Compare_Two_Series(Data1_, Data2_, Data1_Ind, Data2_Ind, save_plots=False, show_plots=True,matched_periods=None)