from WindPy import w
import datetime
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ttest_1samp
import xlwings as xw
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

'''登录Choice量化接口'''
# UserName = 'slsc0258'
# Password = 'lx329680'
# startoptions = "ForceLogin=1" + ",UserName=" + UserName + ",Password=" + Password;
# loginResult = c.start(startoptions, '')

'''数据处理'''


def Shift_Corr(Data1, Data2, ShiftRange=range(-20, 21), Method='pearson', ShowFig=False):
    '''
    :param Data1: 数据1
    :param Data2: 数据2（被移动，Shift负期数可以理解为，数据1对数据2的领先，如Shift -5相关性最强，则数据1领先数据2 5期）
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

        Shift_Matrix['Data2'] = Shift_Matrix['Data2'].shift(_).dropna(axis=0)
        Corr_Series.loc[_, 'Corr'] = Shift_Matrix['Data1'].corr(Shift_Matrix['Data2'], method=Method)

    if ShowFig:
        plt.plot(Corr_Series)
        plt.title('Corr')

        plt.show()

    return Corr_Series

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

def HalfLife(H,T):
    t=np.arange(T+1)[1:]
    wt=2**((t-T-1)/H) #半衰权重
    res=wt/wt.sum() #归一化
    return res

'''数据获取'''


def FetchData_MacroFactor_W(Code, StartDate, EndDate):
    '''
    :param Code: 基础宏观变量代码
    :param StartDate: 开始日期
    :param EndDate: 结束日期
    :return: 宏观数据DataFrame
    '''
    w.start()
    error_code, edb_data = w.edb(Code, StartDate, EndDate, "Fill=Previous", usedf=True)
    if error_code == 0:
        pass
    else:
        print("Error Code:", error_code)
        print("Error Message:", edb_data.iloc[0, 0])
    w.stop()
    return edb_data

def FetchData_MacroFactor_LocalDataBase(DB_Add, Data, Indicator, StartDate, EndDate):
    Data = pd.read_excel(DB_Add, sheet_name=Data, index_col=0, header=1)
    Data = Data[Data['披露时间'] == Data['披露时间']]
    Data_Indicator = Data[[Indicator]]
    Data_Indicator.index = Data['披露时间'].values.astype('datetime64[ns]')
    return Data_Indicator[Data_Indicator.index >= StartDate][Data_Indicator.index <= EndDate]

def FetchData_AssetClose_W(Code, StartDate, EndDate, PriceType='CLOSE'):
    '''
    :param Code: 基础资产价格代码
    :param StartDate: 开始日期
    :param EndDate: 结束日期
    :return:
    '''
    w.start()
    error_code, wsd_data = w.wsd(Code, "open,high,low,close", StartDate, EndDate, 'PriceAdj=B', usedf=True)
    if error_code == 0:
        pass
    else:
        print("Error Code:", error_code)
        print("Error Message:", wsd_data.iloc[0, 0])
    w.stop()
    return wsd_data[PriceType]


'''IC-IR计算'''


def MacroFactor_IC(MacroFactor, AssetClose, IC_RWindow=36, Ret_Period=20, Method='pearson'):
    '''
    :param MacroFactor:宏观因子DataFrame
    :param AssetClose: 资产价格因子DataFrame
    :param IC_RWindow: 计算IC的滚动窗口
    :param Ret_Period: 计算未来资产收益率的窗口
    :param Method: IC计算方法：①’Pearson‘-普通IC；②’Spearman‘-RankIC
    :return: IC_Avg, IC_Std, IC_IR
    '''

    Ret_NxtPeriod = AssetClose.pct_change(Ret_Period).shift(-Ret_Period)
    Ret_NxtPeriod.index = Ret_NxtPeriod.index.astype('datetime64[ns]')
    MacroFactor['Ret_NxtPeriod'] = np.nan
    for _ in range(len(MacroFactor)):
        Distance = (Ret_NxtPeriod.index - MacroFactor.index[_]).days
        NxtDate = np.min(np.where(Distance >= 0))
        MacroFactor['Ret_NxtPeriod'][_] = Ret_NxtPeriod.iloc[NxtDate, 0]

    MacroFactor.dropna(axis=0, inplace=True)
    MacroFactor.columns = ['MacroFactor', 'Ret_NxtPeriod']

    IC_Series=pd.DataFrame(index=MacroFactor.index,columns=['IC_Series'])

    for _ in range(IC_RWindow-1,len(MacroFactor)):
        IC_Series.iloc[_,0]=MacroFactor.iloc[_-IC_RWindow+1:_+1].corr(method=Method).iloc[0][1]

    IC_Avg = IC_Series.mean()
    IC_Std = IC_Series.std()
    if IC_Avg > 0:
        IC_Series_AbsRatio = len(IC_Series[IC_Series > 0]) / len(IC_Series)
    else:
        IC_Series_AbsRatio = len(IC_Series[IC_Series <= 0]) / len(IC_Series)
    IC_IR = IC_Avg / IC_Std

    return IC_Avg, IC_Std, IC_IR, IC_Series_AbsRatio

def MacroFactor_WindowsBacktest(MacroFactor, AssetClose, IC_RWindow=range(1, 37, 1), Ret_Period=range(1, 20, 1),
                                Method='pearson', Output_To_File=False, Output_File_Add=None):
    '''
    :param MacroFactor:宏观因子DataFrame
    :param AssetClose:资产价格因子DataFrame
    :param IC_RWindow:计算IC的滚动窗口序列
    :param Ret_Period:计算未来资产收益率的窗口序列
    :param Method:①’Pearson‘-普通IC；②’Spearman‘-RankIC
    :return:IC_Avg_Matrix, IC_Std_Matrix, IC_IR_Matrix
    '''

    IC_Avg_Matrix = pd.DataFrame(columns=IC_RWindow, index=Ret_Period)
    IC_Std_Matrix = pd.DataFrame(columns=IC_RWindow, index=Ret_Period)
    IC_IR_Matrix = pd.DataFrame(columns=IC_RWindow, index=Ret_Period)
    IC_Series_AbsRatio_Matrix = pd.DataFrame(columns=IC_RWindow, index=Ret_Period)

    for IC_RWindow_ in tqdm(IC_RWindow):
        for Ret_Period_ in Ret_Period:
            IC_Avg, IC_Std, IC_IR, IC_Series_AbsRatio = MacroFactor_IC(MacroFactor, AssetClose, IC_RWindow=IC_RWindow_,
                                                                       Ret_Period=Ret_Period_, Method=Method)
            IC_Avg_Matrix.loc[Ret_Period_, IC_RWindow_] = IC_Avg
            IC_Std_Matrix.loc[Ret_Period_, IC_RWindow_] = IC_Std
            IC_IR_Matrix.loc[Ret_Period_, IC_RWindow_] = IC_IR
            IC_Series_AbsRatio_Matrix.loc[Ret_Period_, IC_RWindow_] = IC_Series_AbsRatio

    if Output_To_File:
        app = xw.App(visible=False, add_book=False)
        workbook = app.books.open(Output_File_Add)
        worksheet = workbook.sheets['IC-BacktestResult']
        worksheet.range('B2').value = IC_Avg_Matrix
        worksheet.range('B20').value = IC_Std_Matrix
        worksheet.range('B38').value = IC_IR_Matrix
        worksheet.range('B56').value = IC_Series_AbsRatio_Matrix

        workbook.save()
        app.kill()

    return IC_Avg_Matrix, IC_Std_Matrix, IC_IR_Matrix

def MacroFactor_RollingIC(MacroFactor, AssetClose, IC_RWindow=36, Ret_Period=20, Method='spearman'):
    '''
    :param MacroFactor:宏观因子DataFrame
    :param AssetClose: 资产价格因子DataFrame
    :param IC_RWindow: 计算IC的滚动窗口
    :param Ret_Period: 计算未来资产收益率的窗口
    :param Method: IC计算方法：①’pearson‘-普通IC；②’spearman‘-RankIC(默认）
    :return: IC_Data 包含：
    ① IC_Series历史IC时间序列；
    ② IC_Avg，IC_Series的滚动均值；
    ③ IC_Std,IC_Series的滚动标准差；
    ④ IC_IR,IC_Series的滚动IR；
    ⑤ IC_WinRate,IC_Series的滚动胜率
    '''

    Ret_NxtPeriod = AssetClose.pct_change(Ret_Period).shift(-Ret_Period)
    Ret_NxtPeriod.index = Ret_NxtPeriod.index.astype('datetime64[ns]')
    MacroFactor_=MacroFactor.copy()
    MacroFactor_['Ret_NxtPeriod'] = np.nan
    for _ in range(len(MacroFactor)):
        Distance = (MacroFactor_.index[_]-Ret_NxtPeriod.index).days #开盘买入逻辑
        # Distance = (Ret_NxtPeriod.index - MacroFactor_.index[_]).days #收盘买入逻辑
        NxtDate = np.max(np.where(Distance >= 0))#开盘买入逻辑
        # NxtDate = np.min(np.where(Distance >= 0)) #收盘买入逻辑
        MacroFactor_['Ret_NxtPeriod'][_] = Ret_NxtPeriod.iloc[NxtDate, 0]

    MacroFactor_.dropna(axis=0, inplace=True)
    MacroFactor_.columns = ['MacroFactor', 'Ret_NxtPeriod']

    IC_Series=pd.DataFrame(index=MacroFactor_.index,columns=['IC_Series'])
    #
    for _ in range(IC_RWindow-1,len(MacroFactor_)):
        IC_Series.iloc[_,0]=MacroFactor_.iloc[_-IC_RWindow+1:_+1].corr(method=Method).iloc[0][1]
    IC_Series=IC_Series.dropna()

    IC_Avg = IC_Series.rolling(IC_RWindow).mean()
    IC_Std = IC_Series.rolling(IC_RWindow).std()
    IC_IR = IC_Avg / IC_Std

    IC_Data = pd.concat([IC_Series, IC_Avg, IC_Std, IC_IR], axis=1)
    IC_Data.columns = ['IC_Series', 'IC_Avg', 'IC_Std', 'IC_IR']
    IC_Data['IC_WinRate'] = np.nan

    for _ in range(IC_RWindow - 1, len(IC_Data)):
        IC_Data.iloc[_,4] = len(IC_Data['IC_Series'][_ - (IC_RWindow - 1):_ + 1][IC_Data['IC_Series'][_ - (IC_RWindow - 1):_ + 1] * IC_Data['IC_Avg'][_] > 0]) / IC_RWindow

    return IC_Data
    # return MacroFactor_, IC_Data

def MacroFactor_RollingIC_WindowsBacktest(MacroFactor, AssetClose, DataName,IC_RWindow=range(1, 37, 1),Ret_Period=range(1,20,1),
                                          Method='spearman', Output_To_File=False, Output_File_Add=None):

    IC_Avg_Matrix = pd.DataFrame(columns=IC_RWindow, index=Ret_Period)
    IC_IR_Matrix = pd.DataFrame(columns=IC_RWindow, index=Ret_Period)
    IC_Series_WinRate_Matrix = pd.DataFrame(columns=IC_RWindow, index=Ret_Period)

    for IC_RWindow_ in tqdm(IC_RWindow):
        for Ret_Period_ in Ret_Period:
            IC_Data = MacroFactor_RollingIC(MacroFactor, AssetClose, IC_RWindow=IC_RWindow_, Ret_Period=Ret_Period_,
                                            Method=Method)
            IC_Avg_Matrix.loc[Ret_Period_, IC_RWindow_] = IC_Data['IC_Series'].mean()
            # IC_IR_Matrix.loc[Ret_Period_, IC_RWindow_]=IC_Data['IC_IR'].mean()
            # IC_Series_WinRate_Matrix.loc[Ret_Period_, IC_RWindow_]=IC_Data['IC_WinRate'].mean()



            IC_IR_Matrix.loc[Ret_Period_, IC_RWindow_] = IC_Data['IC_Series'].mean()/IC_Data['IC_Series'].std()
            if IC_Data['IC_Series'].mean()>0:
                IC_Series_WinRate_Matrix.loc[Ret_Period_, IC_RWindow_] = len(IC_Data['IC_Series'][IC_Data['IC_Series']>0])/len(IC_Data['IC_Series'])
            else:
                IC_Series_WinRate_Matrix.loc[Ret_Period_, IC_RWindow_] = len(IC_Data['IC_Series'][IC_Data['IC_Series']<=0])/len(IC_Data['IC_Series'])


    if Output_To_File:

        app = xw.App(visible=False, add_book=False)
        workbook = app.books.open(Output_File_Add)
        try:
            worksheet = workbook.sheets[DataName]
            worksheet.range('B2').value = IC_Avg_Matrix
            worksheet.range('B20').value = IC_IR_Matrix
            worksheet.range('B38').value = IC_Series_WinRate_Matrix
            worksheet.range('O1').value = MacroFactor.dropna()
            worksheet.range('R1').value = IC_Data
        except:
            sheet = workbook.sheets['Mode']
            sheet.api.Copy(Before=sheet.api)
            NewSheet=workbook.sheets[0]
            NewSheet.name=DataName
            worksheet = workbook.sheets[DataName]# worksheet = workbook.sheets.add(MacroFactor.columns[0])
            worksheet.range('B2').value = IC_Avg_Matrix
            worksheet.range('B20').value = IC_IR_Matrix
            worksheet.range('B38').value = IC_Series_WinRate_Matrix
            worksheet.range('O1').value = MacroFactor.dropna()
            worksheet.range('R1').value = IC_Data
            worksheet.api.Move(After=sheet.api)

        workbook.save()
        app.kill()


    return IC_Avg_Matrix, IC_IR_Matrix, IC_Series_WinRate_Matrix



'''资产价格处理'''


def Rel_Strength(Code1, Code2, StartDate, EndDate):
    Asset1 = FetchData_AssetClose_W(Code=Code1, StartDate=StartDate, EndDate=EndDate, PriceType='CLOSE')
    Asset2 = FetchData_AssetClose_W(Code=Code2, StartDate=StartDate, EndDate=EndDate, PriceType='CLOSE')
    Rel_Strength = pd.DataFrame(Asset1 / Asset2)

    Rel_Strength.columns = ['Rel_Strength']

    return Rel_Strength

def EqualWeightBenchmark(*args,StartDate,EndDate):
    Benchmark=pd.DataFrame()
    for _ in args:
        Benchmark=pd.concat([Benchmark,FetchData_AssetClose_W(_,StartDate,EndDate)],axis=1,join='outer')
    Benchmark.columns=args
    Benchmark=Benchmark.dropna(axis=0)
    Benchmark=Benchmark.pct_change(1)
    Benchmark=Benchmark.mean(axis=1)

    EqualWeightBenchmark=pd.DataFrame(index=Benchmark.index,columns=['EqualWeightBenchmark'])
    EqualWeightBenchmark.iloc[0,0]=1

    for _ in range(1,len(Benchmark)):
        EqualWeightBenchmark.iloc[_,0]=EqualWeightBenchmark.iloc[_-1,0]*(1+Benchmark.iloc[_])

    return EqualWeightBenchmark

'''因子处理'''
def Normalization(Data,P_Window,Nsigma=3):
    '''
    :param Data: 原始数据
    :param P_Window: 滚动计算分位数窗口
    :param Nsigma: 去尾标准差
    :return: 映射正态分布后的数据
    '''
    '''去尾'''
    Data.replace(float('inf'), 10000000000,inplace=True)
    Data.replace(float('-inf'), -10000000000, inplace=True)
    Data=(Data.rolling(P_Window).rank()-1)/(P_Window-1) #使得分布在[0,1]之间
    Data=Data.apply(stats.norm.ppf)
    Data[Data>Nsigma]=Nsigma
    Data[Data<-Nsigma]=-Nsigma
    return Data

# 因子合成
def MaxICIR_Comb(*args, P_Window=36, Ret_Period=20,Nsigma=3,Method='spearman'):
    IC_Data=pd.DataFrame()
    Factors=pd.DataFrame()

    for _ in args:
        ColumnName=_.columns[0]
        Factor=Normalization(_,P_Window=P_Window,Nsigma=Nsigma)
        IC=MacroFactor_RollingIC(_,Cap_Rot,IC_RWindow=P_Window,Ret_Period=Ret_Period,Method=Method)[['IC_Series']]
        IC.columns=[ColumnName]

        Factors=pd.concat([Factors,Factor],axis=1,join='outer')
        IC_Data=pd.concat([IC_Data,IC],axis=1,join='outer')

    Factors.sort_index(ascending=True,inplace=True)
    Factors=Factors.astype('float64')
    Factors=Factors.dropna(axis=0)
    IC_Data.sort_index(ascending=True,inplace=True)
    IC_Data=IC_Data.astype('float64')
    IC_Data=IC_Data.dropna(axis=0)

    Factor_Combined=pd.DataFrame(index=Factors.index[1:],columns=['Factor_Combined'])

    for _ in range(P_Window-1,len(IC_Data)):
        # IC_Window_Data=IC_Data.iloc[_-(P_Window-1):_+1,:] #取窗口数据
        # Factor_Window_Data=Factors.iloc[_-(P_Window-1):_+1,:] #取窗口数据

        IC_Cov=np.array(IC_Data.iloc[_-(P_Window-1):_+1,:].cov())
        Inv_IC_Cov=np.linalg.inv(IC_Cov)
        IC_Vector=np.mat(IC_Data.iloc[_-(P_Window-1):_+1,:].mean())
        Weight=Inv_IC_Cov*IC_Vector.T
        Weight=Weight/np.sum(Weight)
        Factor_Combined.iloc[_,0]=np.array(Factors.iloc[_+1,:]).dot(Weight).mean() #取matrix中值


    Factor_Combined=Factor_Combined.dropna(axis=0)
    Factor_Combined=Factor_Combined.astype('float64')

        # IC_Data=pd.concat([IC_Data,_],axis=1)


    return Factor_Combined

def MaxICIR_Comb_RollingDirection(*args, P_Window=36, Ret_Period=20,Nsigma=3,Method='spearman'):
    IC_Data=pd.DataFrame()
    Factors=pd.DataFrame()

    for _ in args:
        ColumnName=_.columns[0]
        Factor=Normalization(_,P_Window=P_Window,Nsigma=Nsigma)
        IC=MacroFactor_RollingIC(_,Cap_Rot,IC_RWindow=P_Window,Ret_Period=Ret_Period,Method=Method)[['IC_Series']]
        IC.columns=[ColumnName]

        Factors=pd.concat([Factors,Factor],axis=1,join='outer')
        IC_Data=pd.concat([IC_Data,IC],axis=1,join='outer')

    Factors.sort_index(ascending=True,inplace=True)
    Factors=Factors.astype('float64')
    Factors=Factors.dropna(axis=0)
    IC_Data.sort_index(ascending=True,inplace=True)
    IC_Data=IC_Data.astype('float64')
    IC_Data=IC_Data.dropna(axis=0)

    Factor_Combined=pd.DataFrame(index=Factors.index[1:],columns=['Factor_Combined'])

    for _ in range(P_Window-1,len(IC_Data)):

        IC_Slice=IC_Data.iloc[_-(P_Window-1):_+1,:]
        Multi=IC_Slice.mean()/IC_Slice.mean().abs() #判断正负号
        IC_Slice=IC_Slice*Multi

        IC_Cov=np.array(IC_Slice.cov())
        Inv_IC_Cov=np.linalg.inv(IC_Cov)
        IC_Vector=np.mat(IC_Slice.mean())
        Weight=Inv_IC_Cov*IC_Vector.T
        Weight=Weight/np.sum(Weight)
        Factor_Combined.iloc[_,0]=np.array(Factors.iloc[_+1,:]*Multi).dot(Weight).mean() #取matrix中值

    Factor_Combined=Factor_Combined.dropna(axis=0)
    Factor_Combined=Factor_Combined.astype('float64')

        # IC_Data=pd.concat([IC_Data,_],axis=1)


    return Factor_Combined

def MaxIC_Comb(*args,Cap_Rot, P_Window=36, Ret_Period=20,Nsigma=3,Method='spearman'):
    IC_Data=pd.DataFrame()
    Factors=pd.DataFrame()

    for _ in args:
        ColumnName=_.columns[0]
        Factor=Normalization(_,P_Window=P_Window,Nsigma=Nsigma)
        IC=MacroFactor_RollingIC(_,Cap_Rot,IC_RWindow=P_Window,Ret_Period=Ret_Period,Method=Method)[['IC_Series']]
        IC.columns=[ColumnName]

        Factors=pd.concat([Factors,Factor],axis=1,join='outer')
        IC_Data=pd.concat([IC_Data,IC],axis=1,join='outer')

    Factors.sort_index(ascending=True,inplace=True)
    Factors=Factors.astype('float64')
    Factors=Factors.dropna(axis=0)
    IC_Data.sort_index(ascending=True,inplace=True)
    IC_Data=IC_Data.astype('float64')
    IC_Data=IC_Data.dropna(axis=0)

    Factor_Combined=pd.DataFrame(index=Factors.index[1:],columns=['Factor_Combined'])

    for _ in range(P_Window-1,len(IC_Data)):
        # IC_Window_Data=IC_Data.iloc[_-(P_Window-1):_+1,:] #取窗口数据
        # Factor_Window_Data=Factors.iloc[_-(P_Window-1):_+1,:] #取窗口数据


        Factor_Cov=np.array(Factors.iloc[_-(P_Window-1):_+1,:].cov())
        Inv_Factor_Cov=np.linalg.inv(Factor_Cov)
        IC_Vector=np.mat(IC_Data.iloc[_-(P_Window-1):_+1,:].mean())
        Weight=Inv_Factor_Cov*IC_Vector.T
        Weight=Weight/np.sum(Weight)
        Factor_Combined.iloc[_,0]=np.array(Factors.iloc[_+1,:]).dot(Weight).mean() #取matrix中值

    Factor_Combined=Factor_Combined.dropna(axis=0)
    Factor_Combined=Factor_Combined.astype('float64')

    return Factor_Combined

def MaxIC_Comb_RollingDirection(*args,Cap_Rot,P_Window=36, Ret_Period=20,Nsigma=3,Method='spearman'):
    IC_Data=pd.DataFrame()
    Factors=pd.DataFrame()

    for _ in args:
        ColumnName=_.columns[0]
        Factor=Normalization(_,P_Window=P_Window,Nsigma=Nsigma)
        IC=MacroFactor_RollingIC(_,Cap_Rot,IC_RWindow=P_Window,Ret_Period=Ret_Period,Method=Method)[['IC_Series']]
        IC.columns=[ColumnName]

        Factors=pd.concat([Factors,Factor],axis=1,join='outer')
        IC_Data=pd.concat([IC_Data,IC],axis=1,join='outer')

    Factors.sort_index(ascending=True,inplace=True)
    Factors=Factors.astype('float64')
    Factors=Factors.dropna(axis=0)
    IC_Data.sort_index(ascending=True,inplace=True)
    IC_Data=IC_Data.astype('float64')
    IC_Data=IC_Data.dropna(axis=0)

    Factor_Combined=pd.DataFrame(index=Factors.index[1:],columns=['Factor_Combined'])

    for _ in range(P_Window-1,len(IC_Data)):

        IC_Slice=IC_Data.iloc[_-(P_Window-1):_+1,:]
        Factor_Slice = Factors.iloc[_ - (P_Window - 1):_ + 1, :]
        Multi=IC_Slice.mean()/IC_Slice.mean().abs() #判断正负号
        Factor_Slice=Factor_Slice*Multi
        IC_Slice=IC_Slice*Multi

        Factor_Cov=np.array(Factor_Slice.cov())
        Inv_Factor_Cov=np.linalg.inv(Factor_Cov)
        IC_Vector=np.mat(IC_Slice.mean())
        Weight=Inv_Factor_Cov*IC_Vector.T
        Weight=Weight/np.sum(Weight)
        Factor_Combined.iloc[_,0]=np.array(Factors.iloc[_+1,:]*Multi).dot(Weight).mean() #取matrix中值

    Factor_Combined=Factor_Combined.dropna(axis=0)
    Factor_Combined=Factor_Combined.astype('float64')

        # IC_Data=pd.concat([IC_Data,_],axis=1)


    return Factor_Combined

def MaxIC_Comb_HalfLife(*args, Cap_Rot,P_Window=36, Ret_Period=20,Nsigma=3,H=36,Method='spearman'):
    IC_Data=pd.DataFrame()
    Factors=pd.DataFrame()

    for _ in args:
        ColumnName=_.columns[0]
        Factor=Normalization(_,P_Window=P_Window,Nsigma=Nsigma)
        IC=MacroFactor_RollingIC(_,Cap_Rot,IC_RWindow=P_Window,Ret_Period=Ret_Period,Method=Method)[['IC_Series']]
        IC.columns=[ColumnName]

        Factors=pd.concat([Factors,Factor],axis=1,join='outer')
        IC_Data=pd.concat([IC_Data,IC],axis=1,join='outer')

    Factors.sort_index(ascending=True,inplace=True)
    Factors=Factors.astype('float64')
    Factors=Factors.dropna(axis=0)
    IC_Data.sort_index(ascending=True,inplace=True)
    IC_Data=IC_Data.astype('float64')
    IC_Data=IC_Data.dropna(axis=0)
    #
    Factor_Combined=pd.DataFrame(index=Factors.index[1:],columns=['Factor_Combined'])
    #
    for _ in range(P_Window-1,len(IC_Data)):
        HalfLife_Wgt=HalfLife(H,_+1).reshape(-1,1) #生成半衰权重
        IC_Data_HalfLife_Wgted=IC_Data.iloc[:_+1,:]*HalfLife_Wgt #半衰权重乘以IC
        Factor_HalfLife_Wgted = Factors.iloc[:_ + 1, :] * HalfLife_Wgt  # 半衰权重乘以因子值

        Factor_Cov=np.array(Factor_HalfLife_Wgted.cov()) #计算协方差矩阵
        Inv_Factor_Cov=np.linalg.inv(Factor_Cov)

        IC_Vector=np.mat(IC_Data_HalfLife_Wgted.sum())
        Weight=Inv_Factor_Cov*IC_Vector.T
        Weight=Weight/np.sum(Weight)
        Factor_Combined.iloc[_,0]=np.array(Factors.iloc[_+1,:]).dot(Weight).mean() #取matrix的值
    #
    Factor_Combined=Factor_Combined.dropna(axis=0)
    Factor_Combined=Factor_Combined.astype('float64')

    # return Factor_Combined
    return Factor_Combined

def MaxIC_Comb_HalfLife_RollingDirection(*args, Cap_Rot, P_Window=36, Ret_Period=20,Nsigma=3,H=36,Method='spearman'):
    IC_Data=pd.DataFrame()
    Factors=pd.DataFrame()

    for _ in args:
        ColumnName=_.columns[0]
        Factor=Normalization(_,P_Window=P_Window,Nsigma=Nsigma)
        IC=MacroFactor_RollingIC(_,Cap_Rot,IC_RWindow=P_Window,Ret_Period=Ret_Period,Method=Method)[['IC_Series']]
        IC.columns=[ColumnName]

        Factors=pd.concat([Factors,Factor],axis=1,join='outer')
        IC_Data=pd.concat([IC_Data,IC],axis=1,join='outer')

    Factors.sort_index(ascending=True,inplace=True)
    Factors=Factors.astype('float64')
    Factors=Factors.dropna(axis=0)
    IC_Data.sort_index(ascending=True,inplace=True)
    IC_Data=IC_Data.astype('float64')
    IC_Data=IC_Data.dropna(axis=0)

    Factor_Combined=pd.DataFrame(index=Factors.index[1:],columns=['Factor_Combined'])

    for _ in range(P_Window-1,len(IC_Data)):
        HalfLife_Wgt = HalfLife(H, _ + 1).reshape(-1, 1)  # 生成半衰权重
        IC_Data_HalfLife_Wgted = IC_Data.iloc[:_ + 1, :] * HalfLife_Wgt  # 半衰权重乘以IC
        Factor_HalfLife_Wgted = Factors.iloc[:_ + 1, :]* HalfLife_Wgt # 半衰权重乘以因子值

        Multi = IC_Data_HalfLife_Wgted.mean() / IC_Data_HalfLife_Wgted.mean().abs()  # 判断正负号
        Factor_HalfLife_Wgted=Factor_HalfLife_Wgted*Multi #转换正负号
        IC_Data_HalfLife_Wgted=IC_Data_HalfLife_Wgted*Multi #转换正负号

        Factor_Cov=np.array(Factor_HalfLife_Wgted.cov()) #计算协方差矩阵
        Inv_Factor_Cov=np.linalg.inv(Factor_Cov)

        IC_Vector=np.mat(IC_Data_HalfLife_Wgted.sum())
        Weight=Inv_Factor_Cov*IC_Vector.T
        Weight=Weight/np.sum(Weight)
        Factor_Combined.iloc[_,0]=np.array(Factors.iloc[_+1,:]*Multi).dot(Weight).mean() #取matrix中值

    Factor_Combined=Factor_Combined.dropna(axis=0)
    Factor_Combined=Factor_Combined.astype('float64')

    return Factor_Combined

'''回测框架'''
def Asset_Timing(MacroFactor,Code1,Code2,RWindow=36,UpperBound=0.8,LowerBound=0.2,StartDate='2005-01-01',EndDate='2023-10-09'):
    MacroFactor=MacroFactor[MacroFactor.index >= StartDate][MacroFactor.index <= EndDate]
    Asset1 = FetchData_AssetClose_W(Code=Code1, StartDate=StartDate, EndDate=EndDate, PriceType='CLOSE').pct_change(1).dropna()
    Asset1.index = Asset1.index.astype('datetime64[ns]')
    Asset2 = FetchData_AssetClose_W(Code=Code2, StartDate=StartDate, EndDate=EndDate, PriceType='CLOSE').pct_change(1).dropna()
    Asset2.index = Asset2.index.astype('datetime64[ns]')

    MacroFactor = ((MacroFactor.rolling(RWindow).rank() - 1) / (RWindow - 1)).dropna()

    Strategy_StartDate_Pos = (Asset1.index<MacroFactor.index[0]).sum()
    Strategy=pd.DataFrame(index=Asset1.index[Strategy_StartDate_Pos-1:],columns=['Asset1','Asset2','Signal','StrategyNAV','Benchmark','Benchmark1','Benchmark2']) #策略从第一个因子值前1个交易日开始运行
    #初始化策略净值
    Strategy['StrategyNAV'].iloc[0]=1
    Strategy['Asset1'].iloc[0]=0.5
    Strategy['Asset2'].iloc[0]=0.5
    Strategy['Signal'].iloc[0]=0
    Strategy['Benchmark'].iloc[0]=1
    Strategy['Benchmark1'].iloc[0]=1
    Strategy['Benchmark2'].iloc[0]=1

    BenchmarkAsset1=0.5
    BenchmarkAsset2=0.5

    SignalDatePos=0
    for _ in range(1,len(Strategy)):
        BenchmarkAsset1*=(1+Asset1.loc[Strategy.index[_]])
        BenchmarkAsset2*=(1+Asset2.loc[Strategy.index[_]])
        Strategy['Benchmark'].iloc[_]=BenchmarkAsset1+BenchmarkAsset2
        Strategy['Benchmark1'].iloc[_]=BenchmarkAsset1*2
        Strategy['Benchmark2'].iloc[_]=BenchmarkAsset2*2
        #第一步：确定当天信号
        if SignalDatePos <= len(MacroFactor):
            if Strategy.index[_] >= MacroFactor.index[SignalDatePos]:
                if MacroFactor.iloc[SignalDatePos,0] > UpperBound:
                    Strategy['Signal'].iloc[_] = 1
                elif MacroFactor.iloc[SignalDatePos,0] < LowerBound:
                    Strategy['Signal'].iloc[_] = -1
                else:
                    Strategy['Signal'].iloc[_] = 0
                SignalDatePos+=1
            else:
                Strategy['Signal'].iloc[_] = Strategy['Signal'].iloc[_-1]
        else:
            pass

        #第二步：组合收获收益
        NAV_Today=Strategy['Asset1'].iloc[_-1]*(1+Asset1.loc[Strategy.index[_]])+Strategy['Asset2'].iloc[_-1]*(1+Asset2.loc[Strategy.index[_]])
        Strategy['StrategyNAV'].iloc[_]=NAV_Today

        #第三步：收盘前调仓
        if Strategy['Signal'].iloc[_] == 0:
            Strategy['Asset1'].iloc[_]=NAV_Today/2
            Strategy['Asset2'].iloc[_]=NAV_Today/2
        elif Strategy['Signal'].iloc[_] == 1:
            Strategy['Asset1'].iloc[_]=NAV_Today
            Strategy['Asset2'].iloc[_]=0
        elif Strategy['Signal'].iloc[_] == -1:
            Strategy['Asset1'].iloc[_]=0
            Strategy['Asset2'].iloc[_]=NAV_Today



    return MacroFactor,Asset1,Asset2,Strategy

def Asset_Timing_Rel(MacroFactor,AssetPrice,RWindow=36,UpperBound=0.8,LowerBound=0.2,StartDate='2005-01-01',EndDate='2023-10-09'):
    MacroFactor=MacroFactor[MacroFactor.index >= StartDate][MacroFactor.index <= EndDate]
    Asset = AssetPrice.copy()
    Asset =Asset.pct_change(1).dropna()

    MacroFactor = ((MacroFactor.rolling(RWindow).rank() - 1) / (RWindow - 1)).dropna()

    Strategy_StartDate_Pos = (Asset.index<MacroFactor.index[0]).sum()
    Strategy=pd.DataFrame(index=Asset.index[Strategy_StartDate_Pos-1:],columns=['Signal','StrategyNAV','Benchmark']) #策略从第一个因子值前1个交易日开始运行
    #初始化策略净值
    Strategy['StrategyNAV'].iloc[0]=1

    Strategy['Signal'].iloc[0]=0
    Strategy['Benchmark'].iloc[0]=1

    SignalDatePos=0
    for _ in range(1,len(Strategy)):
        Strategy['Benchmark'].iloc[_]=Strategy['Benchmark'].iloc[_-1]*(1+Asset.loc[Strategy.index[_]])
        #第一步：确定当天信号
        if SignalDatePos <= len(MacroFactor):
            if Strategy.index[_] >= MacroFactor.index[SignalDatePos]:
                if MacroFactor.iloc[SignalDatePos,0] > UpperBound:
                    Strategy['Signal'].iloc[_] = 1
                elif MacroFactor.iloc[SignalDatePos,0] < LowerBound:
                    Strategy['Signal'].iloc[_] = -1
                else:
                    Strategy['Signal'].iloc[_] = 0
                SignalDatePos+=1
            else:
                Strategy['Signal'].iloc[_] = Strategy['Signal'].iloc[_-1]
        else:
            pass

        #第三步：收盘前调仓
        if Strategy['Signal'].iloc[_] == 0:
            Strategy['StrategyNAV'].iloc[_]=Strategy['StrategyNAV'].iloc[_-1]*1
        elif Strategy['Signal'].iloc[_] == 1:
            Strategy['StrategyNAV'].iloc[_]=Strategy['StrategyNAV'].iloc[_-1]*(1+Asset.loc[Strategy.index[_]])
        elif Strategy['Signal'].iloc[_] == -1:
            Strategy['StrategyNAV'].iloc[_] = Strategy['StrategyNAV'].iloc[_ - 1] * (1 - Asset.loc[Strategy.index[_]])
    Strategy=Strategy.astype('float64')

    '''按信号计算决策数量'''
    Long_Win=0
    Long=0
    Short_Win=0
    Short=0


    for _ in range(len(Strategy) - 1):
        if Strategy['Signal'].iloc[_] != Strategy['Signal'].iloc[_ + 1]:
            if Strategy['Signal'].iloc[_+1] == 1:
                Long += 1
                NxtSht = np.where(Strategy['Signal'].iloc[_ + 1:].values != 1)[0]
                if len(NxtSht) > 0:
                    if Strategy['StrategyNAV'].iloc[_ + 1 + NxtSht[0] - 1] > Strategy['StrategyNAV'].iloc[_]:
                        Long_Win += 1
                else:
                    if Strategy['StrategyNAV'].iloc[-1] > Strategy['StrategyNAV'].iloc[_]:
                        Long_Win += 1
            elif Strategy['Signal'].iloc[_+1] == -1:
                Short += 1
                NxtLng = np.where(Strategy['Signal'].iloc[_ + 1:].values != -1)[0]
                if len(NxtLng) > 0:
                    if Strategy['StrategyNAV'].iloc[_ + 1 + NxtLng[0] - 1] < Strategy['StrategyNAV'].iloc[_]:
                        Short_Win += 1
                else:
                    if Strategy['StrategyNAV'].iloc[-1] < Strategy['StrategyNAV'].iloc[_]:
                        Short_Win += 1

    Long_WinRatio = Long_Win / Long
    Short_WinRatio = Short_Win / Short

    '''月度计算决策数量'''
    Long_Win2=0
    Long2=0
    Short_Win2=0
    Short2=0

    for _ in range(len(MacroFactor)-1):
        StartDate=max(np.where(Strategy.index<MacroFactor.index[_])[0])
        EndDate=max(np.where(Strategy.index<MacroFactor.index[_+1])[0])
        if Strategy['Signal'].iloc[EndDate] == 1:
            Long2 += 1
            if Strategy['StrategyNAV'].iloc[EndDate] > Strategy['StrategyNAV'].iloc[StartDate]:
                Long_Win2 += 1
        elif Strategy['Signal'].iloc[EndDate] == -1:
            Short2 += 1
            if Strategy['StrategyNAV'].iloc[EndDate] < Strategy['StrategyNAV'].iloc[StartDate]:
                Short_Win2 += 1

    LastStartDate=max(np.where(Strategy.index<MacroFactor.index[-1])[0])
    if Strategy['Signal'].iloc[-1] == 1:
        Long2 += 1
        if Strategy['StrategyNAV'].iloc[-1] > Strategy['StrategyNAV'].iloc[LastStartDate]:
            Long_Win2 += 1
    elif Strategy['Signal'].iloc[-1] == -1:
        Short2 += 1
        if Strategy['StrategyNAV'].iloc[-1] < Strategy['StrategyNAV'].iloc[LastStartDate]:
            Short_Win2 += 1

    Long_WinRatio2 = Long_Win2 / Long2
    Short_WinRatio2 = Short_Win2 / Short2


    return MacroFactor,Strategy,Long_WinRatio2,Short_WinRatio2

'''PMI组'''


def YOY(Data):
    Data = Data.pct_change(12).dropna()
    Data.columns = [Data.columns[0] + '_YOY']
    Data.replace(float('inf'), 10000000000, inplace=True)
    Data.replace(float('-inf'), -10000000000, inplace=True)
    return Data

def MOM(Data):
    Data = Data.pct_change(1).dropna()
    Data.columns = [Data.columns[0] + '_MOM']
    Data.replace(float('inf'), 10000000000, inplace=True)
    Data.replace(float('-inf'), -10000000000, inplace=True)
    return Data

def Diff(Data, Period):
    Data = Data.diff(Period).dropna()
    Data.columns = [Data.columns[0] + '_Diff{}'.format(Period)]
    return Data

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

    return pd.DataFrame(Cycle.diff(12).dropna(axis=0))
    # return PMI_Index

def PMI(DB_Add, StartDate, EndDate):
    PMI = FetchData_MacroFactor_LocalDataBase(DB_Add, '中国-官方PMI', '中国:制造业PMI', StartDate=StartDate,
                                              EndDate=EndDate)
    PMI.columns = ['PMI']
    return PMI

def PMI_Momentum(DB_Add, StartDate, EndDate):
    PMI_NO = FetchData_MacroFactor_LocalDataBase(DB_Add, '中国-官方PMI', '中国:制造业PMI:新订单', StartDate=StartDate,
                                                 EndDate=EndDate)
    PMI_P = FetchData_MacroFactor_LocalDataBase(DB_Add, '中国-官方PMI', '中国:制造业PMI:生产', StartDate=StartDate,
                                                EndDate=EndDate)
    PMI_Momentum = pd.concat([PMI_NO, PMI_P], axis=1).dropna()
    PMI_Momentum['PMI_Momentum'] = PMI_Momentum['中国:制造业PMI:新订单'] - PMI_Momentum['中国:制造业PMI:生产']
    return PMI_Momentum[['PMI_Momentum']]

def PMI_Momentum_Cycle(DB_Add, StartDate, EndDate):
    PMI_NO = PMI_To_YOY(FetchData_MacroFactor_LocalDataBase(DB_Add, '中国-官方PMI', '中国:制造业PMI:新订单', StartDate=StartDate,
                                                 EndDate=EndDate),Oneside=True)
    PMI_P = PMI_To_YOY(FetchData_MacroFactor_LocalDataBase(DB_Add, '中国-官方PMI', '中国:制造业PMI:生产', StartDate=StartDate,
                                                EndDate=EndDate),Oneside=True)
    PMI_Momentum = pd.concat([PMI_NO, PMI_P], axis=1).dropna()
    PMI_Momentum.columns=['中国:制造业PMI:新订单','中国:制造业PMI:生产']
    PMI_Momentum['PMI_Momentum'] = PMI_Momentum['中国:制造业PMI:新订单'] - PMI_Momentum['中国:制造业PMI:生产']
    return PMI_Momentum[['PMI_Momentum']]

def PMI_DomDemand(DB_Add, StartDate, EndDate):
    PMI_NO = FetchData_MacroFactor_LocalDataBase(DB_Add, '中国-官方PMI', '中国:制造业PMI:新订单', StartDate=StartDate,
                                                 EndDate=EndDate)
    PMI_NEO = FetchData_MacroFactor_LocalDataBase(DB_Add, '中国-官方PMI', '中国:制造业PMI:新出口订单',
                                                  StartDate=StartDate, EndDate=EndDate)
    PMI_DomDemand = pd.concat([PMI_NO, PMI_NEO], axis=1).dropna()
    PMI_DomDemand['PMI_DomDemand'] = PMI_DomDemand['中国:制造业PMI:新订单'] - PMI_DomDemand['中国:制造业PMI:新出口订单']
    return PMI_DomDemand[['PMI_DomDemand']]

def PMI_DomDemand_DivPMI(DB_Add, StartDate, EndDate):
    PMI_NEO = FetchData_MacroFactor_LocalDataBase(DB_Add, '中国-官方PMI', '中国:制造业PMI:新出口订单',
                                                  StartDate=StartDate, EndDate=EndDate)
    PMI_NO = FetchData_MacroFactor_LocalDataBase(DB_Add, '中国-官方PMI', '中国:制造业PMI:新订单', StartDate=StartDate,
                                                 EndDate=EndDate)
    PMI_DomDemand_DivPMI = pd.concat([PMI_NEO, PMI_NO], axis=1).dropna()
    PMI_DomDemand_DivPMI['PMI_DomDemand_DivPMI'] = 1 - PMI_DomDemand_DivPMI['中国:制造业PMI:新出口订单'] / PMI_DomDemand_DivPMI['中国:制造业PMI:新订单']
    return PMI_DomDemand_DivPMI[['PMI_DomDemand_DivPMI']]

def PMI_Profit(DB_Add, StartDate, EndDate):
    PMI_Price = FetchData_MacroFactor_LocalDataBase(DB_Add, '中国-官方PMI', '中国:制造业PMI:出厂价格',
                                                    StartDate=StartDate, EndDate=EndDate)  # PMI出厂价格
    PMI_RawPrice = FetchData_MacroFactor_LocalDataBase(DB_Add, '中国-官方PMI', '中国:制造业PMI:主要原材料购进价格',
                                                       StartDate=StartDate, EndDate=EndDate)  # PMI原材料采购价格
    PMI_Profit = pd.concat([PMI_Price, PMI_RawPrice], axis=1).dropna()
    PMI_Profit['PMI_Profit'] = PMI_Profit['中国:制造业PMI:出厂价格'] - PMI_Profit['中国:制造业PMI:主要原材料购进价格']
    return PMI_Profit[['PMI_Profit']]

def PMI_Inventory(DB_Add, StartDate, EndDate):
    PMI_Inv = FetchData_MacroFactor_LocalDataBase(DB_Add, '中国-官方PMI', '中国:制造业PMI:产成品库存',
                                                  StartDate=StartDate, EndDate=EndDate)  # PMI产成品库存
    PMI_RawInv = FetchData_MacroFactor_LocalDataBase(DB_Add, '中国-官方PMI', '中国:制造业PMI:原材料库存',
                                                     StartDate=StartDate, EndDate=EndDate)  # PMI原材料库存
    PMI_Inventory = pd.concat([PMI_Inv, PMI_RawInv], axis=1).dropna()
    PMI_Inventory['PMI_Inventory'] = PMI_Inventory['中国:制造业PMI:产成品库存'] + PMI_Inventory[
        '中国:制造业PMI:原材料库存']
    return PMI_Inventory[['PMI_Inventory']]


'''Test'''
# 文件位置
DB_Add = 'D:\Python\MacroFactorModel\MacroDB\宏观经济数据.xlsx'
Output_File_Add = 'D:\Python\MacroFactorModel\IC-BacktestResult-PMI.xlsx'

# 上证50和中证1000相对强度
Cap_Rot = Rel_Strength(Code1="H00300.CSI", Code2="H00852.SH", StartDate="2005-01-01", EndDate="2023-10-09")

# 因子构建
PMI = PMI(DB_Add, StartDate="2005-01-01", EndDate="2023-10-09")  # PMI
PMI_yoy = YOY(PMI) # PMI同比
PMI_mom = MOM(PMI)  # PMI环比
PMI_Diff = Diff(PMI, Period=1)  # PMI环比
PMI_Normed=Normalization(PMI,P_Window=36,Nsigma=3) #PMI正态化
PMI_yoy_Normed=Normalization(PMI_yoy,P_Window=36,Nsigma=3) #PMI同比正态化
PMI_mom_Normed=Normalization(PMI_mom,P_Window=36,Nsigma=3) #PMI环比正态化
PMI_Diff_Normed=Normalization(PMI_Diff,P_Window=36,Nsigma=3) #PMI环比正态化
PMI_Cycle_Normed=Normalization(PMI_To_YOY(PMI,Oneside=True,Cycle_Term=14400),P_Window=36,Nsigma=3) #PMI循环项正态化

PMI_Momentum = PMI_Momentum(DB_Add, StartDate="2005-01-01", EndDate="2023-10-09") # PMI动量
PMI_Momentum_yoy = YOY(PMI_Momentum)# PMI动量同比
PMI_Momentum_mom = MOM(PMI_Momentum)  # PMI动量环比
PMI_Momentum_Diff = Diff(PMI_Momentum, Period=1)  # PMI动量环比
PMI_Momentum_Normed=Normalization(PMI_Momentum,P_Window=36,Nsigma=3) #PMI动量正态化
PMI_Momentum_yoy_Normed=Normalization(PMI_Momentum_yoy,P_Window=36,Nsigma=3) #PMI动量同比正态化
PMI_Momentum_mom_Normed=Normalization(PMI_Momentum_mom,P_Window=36,Nsigma=3) #PMI动量环比正态化
PMI_Momentum_Diff_Normed=Normalization(PMI_Momentum_Diff,P_Window=36,Nsigma=3) #PMI动量环比正态化


PMI_DomDemand = PMI_DomDemand(DB_Add, StartDate="2005-01-01", EndDate="2023-10-09")  # PMI国内需求
PMI_DomDemand_yoy = YOY(PMI_DomDemand)  # PMI国内需求同比
PMI_DomDemand_mom = MOM(PMI_DomDemand)  # PMI国内需求环比
PMI_DomDemand_Diff = Diff(PMI_DomDemand, Period=1)  # PMI国内需求环比
PMI_DomDemand_Normed=Normalization(PMI_DomDemand,P_Window=36,Nsigma=3) #PMI国内需求正态化
PMI_DomDemand_yoy_Normed=Normalization(PMI_DomDemand_yoy,P_Window=36,Nsigma=3) #PMI国内需求同比正态化
PMI_DomDemand_mom_Normed=Normalization(PMI_DomDemand_mom,P_Window=36,Nsigma=3) #PMI国内需求环比正态化
PMI_DomDemand_Diff_Normed=Normalization(PMI_DomDemand_Diff,P_Window=36,Nsigma=3) #PMI国内需求环比正态化


PMI_Profit = PMI_Profit(DB_Add, StartDate="2005-01-01", EndDate="2023-10-09")  # PMI利润
PMI_Profit_yoy = YOY(PMI_Profit)  # PMI利润同比
PMI_Profit_mom = MOM(PMI_Profit)  # PMI利润环比
PMI_Profit_Diff = Diff(PMI_Profit, Period=1) # PMI利润环比
PMI_Profit_Normed=Normalization(PMI_Profit,P_Window=36,Nsigma=3) #PMI利润正态化
PMI_Profit_yoy_Normed=Normalization(PMI_Profit_yoy,P_Window=36,Nsigma=3) #PMI利润同比正态化
PMI_Profit_mom_Normed=Normalization(PMI_Profit_mom,P_Window=36,Nsigma=3) #PMI利润环比正态化
PMI_Profit_Diff_Normed=Normalization(PMI_Profit_Diff,P_Window=36,Nsigma=3) #PMI利润环比正态化


PMI_Inventory = PMI_Inventory(DB_Add, StartDate="2005-01-01", EndDate="2023-10-09")  # PMI库存
PMI_Inventory_yoy = YOY(PMI_Inventory)  # PMI库存同比
PMI_Inventory_mom = MOM(PMI_Inventory)  # PMI库存环比
PMI_Inventory_Diff = Diff(PMI_Inventory, Period=1)  # PMI库存环比
PMI_Inventory_Normed=Normalization(PMI_Inventory,P_Window=36,Nsigma=3) #PMI库存正态化
PMI_Inventory_yoy_Normed=Normalization(PMI_Inventory_yoy,P_Window=36,Nsigma=3) #PMI库存同比正态化
PMI_Inventory_mom_Normed=Normalization(PMI_Inventory_mom,P_Window=36,Nsigma=3) #PMI库存环比正态化
PMI_Inventory_Diff_Normed=Normalization(PMI_Inventory_Diff,P_Window=36,Nsigma=3) #PMI库存环比正态化

PMI_DomDemand_DivPMI=PMI_DomDemand_DivPMI(DB_Add, StartDate="2005-01-01", EndDate="2023-10-09")
PMI_DomDemand_DivPMI_yoy = YOY(PMI_DomDemand_DivPMI)  # PMI国内需求同比
PMI_DomDemand_DivPMI_mom = MOM(PMI_DomDemand_DivPMI)  # PMI国内需求环比
PMI_DomDemand_DivPMI_Diff = Diff(PMI_DomDemand_DivPMI, Period=1)  # PMI国内需求环比
PMI_DomDemand_DivPMI_Normed=Normalization(PMI_DomDemand_DivPMI,P_Window=36,Nsigma=3) #PMI国内需求正态化
PMI_DomDemand_DivPMI_yoy_Normed=Normalization(PMI_DomDemand_DivPMI_yoy,P_Window=36,Nsigma=3) #PMI国内需求同比正态化
PMI_DomDemand_DivPMI_mom_Normed=Normalization(PMI_DomDemand_DivPMI_mom,P_Window=36,Nsigma=3) #PMI国内需求环比正态化
PMI_DomDemand_DivPMI_Diff_Normed=Normalization(PMI_DomDemand_DivPMI_Diff,P_Window=36,Nsigma=3) #PMI国内需求环比正态化

PMI_To_YOY_Data=PMI_To_YOY(PMI,Oneside=True,Cycle_Term=14400)

# FactorSet=pd.concat([PMI_Normed,PMI_yoy_Normed,PMI_mom_Normed,PMI_Diff_Normed,
#                      PMI_Momentum_Normed,PMI_Momentum_yoy_Normed,PMI_Momentum_mom_Normed,PMI_Momentum_Diff_Normed,
#                      PMI_DomDemand_Normed,PMI_DomDemand_yoy_Normed,PMI_DomDemand_mom_Normed,PMI_DomDemand_Diff_Normed,
#                      PMI_Profit_Normed,PMI_Profit_yoy_Normed,PMI_Profit_mom_Normed,PMI_Profit_Diff_Normed,
#                      PMI_Inventory_Normed,PMI_Inventory_yoy_Normed,PMI_Inventory_mom_Normed,PMI_Inventory_Diff_Normed,
#                      PMI_DomDemand_DivPMI_Normed,PMI_DomDemand_DivPMI_yoy_Normed,PMI_DomDemand_DivPMI_mom_Normed,PMI_DomDemand_DivPMI_Diff_Normed],axis=1)

# Corr_Matrix=FactorSet.corr(method='pearson')
# Corr_Matrix.to_excel('D:\Python\MacroFactorModel\Corr_Matrix.xlsx')

'''单因子'''
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Normed,Cap_Rot,'PMI-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_mom_Normed,Cap_Rot,'PMI_mom-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Diff_Normed,Cap_Rot,'PMI_Diff-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_yoy_Normed,Cap_Rot,'PMI_yoy-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)
#
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Momentum_Normed,Cap_Rot,'PMI_Momentum-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Momentum_mom_Normed,Cap_Rot,'PMI_Momentum_mom-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Momentum_Diff_Normed,Cap_Rot,'PMI_Momentum_Diff-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Momentum_yoy_Normed,Cap_Rot,'PMI_Momentum_yoy-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)
#
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_DomDemand_Normed,Cap_Rot,'PMI_DomDemand-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_DomDemand_mom_Normed,Cap_Rot,'PMI_DomDemand_mom-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_DomDemand_Diff_Normed,Cap_Rot,'PMI_DomDemand_Diff-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_DomDemand_yoy_Normed,Cap_Rot,'PMI_DomDemand_yoy-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)
#
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Profit_Normed,Cap_Rot,'PMI_Profit-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Profit_mom_Normed,Cap_Rot,'PMI_Profit_mom-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Profit_Diff_Normed,Cap_Rot,'PMI_Profit_Diff-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Profit_yoy_Normed,Cap_Rot,'PMI_Profit_yoy-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)
#
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Inventory_Normed,Cap_Rot,'PMI_Inventory-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Inventory_mom_Normed,Cap_Rot,'PMI_Inventory_mom-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Inventory_Diff_Normed,Cap_Rot,'PMI_Inventory_Diff-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Inventory_yoy_Normed,Cap_Rot,'PMI_Inventory_yoy-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)

# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_DomDemand_DivPMI_Normed,Cap_Rot,'PMI_DomDivPMI-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_DomDemand_DivPMI_mom_Normed,Cap_Rot,'PMI_DomDivPMI_mom-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_DomDemand_DivPMI_Diff_Normed,Cap_Rot,'PMI_DomDivPMI_Diff-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_DomDemand_DivPMI_yoy_Normed,Cap_Rot,'PMI_DomDivPMI_yoy-Normed-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)


# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Normed,Cap_Rot,'PMI-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_mom_Normed,Cap_Rot,'PMI_mom-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Diff_Normed,Cap_Rot,'PMI_Diff-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_yoy_Normed,Cap_Rot,'PMI_yoy-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
#
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Momentum_Normed,Cap_Rot,'PMI_Momentum-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Momentum_mom_Normed,Cap_Rot,'PMI_Momentum_mom-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Momentum_Diff_Normed,Cap_Rot,'PMI_Momentum_Diff-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Momentum_yoy_Normed,Cap_Rot,'PMI_Momentum_yoy-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
#
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_DomDemand_Normed,Cap_Rot,'PMI_DomDemand-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_DomDemand_mom_Normed,Cap_Rot,'PMI_DomDemand_mom-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_DomDemand_Diff_Normed,Cap_Rot,'PMI_Dom_Diff-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_DomDemand_yoy_Normed,Cap_Rot,'PMI_Dom_yoy-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)

# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Profit_Normed,Cap_Rot,'PMI_Profit-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Profit_mom_Normed,Cap_Rot,'PMI_Profit_mom-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Profit_Diff_Normed,Cap_Rot,'PMI_Profit_Diff-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Profit_yoy_Normed,Cap_Rot,'PMI_Profit_yoy-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
#
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Inventory_Normed,Cap_Rot,'PMI_Inventory-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Inventory_mom_Normed,Cap_Rot,'PMI_Inventory_mom-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Inventory_Diff_Normed,Cap_Rot,'PMI_Inv_Diff-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_Inventory_yoy_Normed,Cap_Rot,'PMI_Inv_yoy-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)

# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_DomDemand_DivPMI_Normed,Cap_Rot,'PMI_DomDivPMI-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_DomDemand_DivPMI_mom_Normed,Cap_Rot,'PMI_DomDivPMI_mom-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_DomDemand_DivPMI_Diff_Normed,Cap_Rot,'PMI_DDivPMI_Diff-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(PMI_DomDemand_DivPMI_yoy_Normed,Cap_Rot,'PMI_DDivPMI_yoy-Normed-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)

'''多因子合成'''
Factor_Combined=MaxIC_Comb_HalfLife_RollingDirection(PMI_Normed,PMI_DomDemand_DivPMI_Normed,PMI_Momentum_Diff_Normed,Cap_Rot=Cap_Rot)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(Factor_Combined,Cap_Rot,'PMI-3F-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(Factor_Combined,Cap_Rot,'PMI-3F-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)

# Factor_Combined=MaxIC_Comb_HalfLife_RollingDirection(PMI_Normed,PMI_DomDemand_DivPMI_Normed,PMI_Momentum_Diff_Normed,PMI_Inventory_Diff_Normed,Cap_Rot=Cap_Rot)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(Factor_Combined,Cap_Rot,'PMI-4F-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(Factor_Combined,Cap_Rot,'PMI-4F-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)

# Factor_Combined=MaxIC_Comb_HalfLife_RollingDirection(PMI_Normed,PMI_DomDemand_DivPMI_Normed,Cap_Rot=Cap_Rot)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(Factor_Combined,Cap_Rot,'PMI-2F-RankIC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='spearman',Output_To_File=True,Output_File_Add=Output_File_Add)
# IC_Avg_Matrix,IC_IR_Matrix,IC_WinRate_Matrix=MacroFactor_RollingIC_WindowsBacktest(Factor_Combined,Cap_Rot,'PMI-2F-IC',IC_RWindow=range(6,37,3),Ret_Period=range(5,21,1),Method='pearson',Output_To_File=True,Output_File_Add=Output_File_Add)




# Factor_Combined=MaxIC_Comb_HalfLife_RollingDirection(PMI_Normed,PMI_yoy_Normed,
#                                                      PMI_mom_Normed,PMI_Diff_Normed,PMI_Momentum_Normed,PMI_Momentum_yoy_Normed,
#                                                      PMI_Momentum_mom_Normed,PMI_Momentum_Diff_Normed,PMI_DomDemand_Normed,
#                                                      PMI_DomDemand_yoy_Normed,PMI_DomDemand_mom_Normed,PMI_DomDemand_Diff_Normed,
#                                                      PMI_Inventory_Normed,PMI_Inventory_yoy_Normed,PMI_Inventory_mom_Normed,PMI_Inventory_Diff_Normed,
#                                                      PMI_DomDemand_DivPMI_Normed,PMI_DomDemand_DivPMI_yoy_Normed,PMI_DomDemand_DivPMI_mom_Normed,PMI_DomDemand_DivPMI_Diff_Normed,Cap_Rot=Cap_Rot)

'''IC_COMB_HL Test'''
# P_Window=36
# Ret_Period=20
# Nsigma=3
# H=36
# Method='spearman'
#
#
# Factor_Combined=MaxIC_Comb_HalfLife(PMI_mom,PMI_Diff,Cap_Rot=Cap_Rot)
# Factor_Combined_Rolling=MaxIC_Comb_HalfLife_RollingDirection(PMI_mom,PMI_Diff,Cap_Rot=Cap_Rot)
# a=MacroFactor_RollingIC(PMI_mom, Cap_Rot, IC_RWindow=36, Ret_Period=20, Method='spearman')



'''Benchmark表现'''
# Benchmark=EqualWeightBenchmark("H00300.CSI", "H00852.SH", StartDate="2005-01-01", EndDate="2023-10-09")
# MacroFactor,Asset1,Asset2,Strategy=Asset_Timing(Factor_Combined,Code1="H00300.CSI",Code2="H00852.SH",RWindow=36,
#                          UpperBound=0.6,LowerBound=0.4,
#                          StartDate="2005-01-01",EndDate="2023-10-09")

#
# NAV=Strategy['StrategyNAV'].dropna()
# Benchmark=Strategy['Benchmark'].dropna()
# Benchmark1=Strategy['Benchmark1'].dropna()
# Benchmark2=Strategy['Benchmark2'].dropna()
#
#
# plt.plot(NAV,label='Strategy',color='red')
# plt.plot(Benchmark,label='Benchmark',color='black')
# plt.plot(Benchmark1,label='Benchmark1',color='grey',linestyle='--')
# plt.plot(Benchmark2,label='Benchmark2',color='grey',linestyle='--')
# plt.legend()
# plt.show()

'''Benchmark2'''
MacroFactor,Strategy,Long_WinRatio,Short_WinRatio=Asset_Timing_Rel(Factor_Combined,AssetPrice=Cap_Rot,
                                                                   RWindow=36,UpperBound=0.6,
                                                                   LowerBound=0.4,
                                                                   StartDate="2005-01-01",EndDate="2023-10-09")

# NAV=Strategy['StrategyNAV'].dropna()
# Benchmark=Strategy['Benchmark'].dropna()
#
# plt.plot(NAV,label='Strategy',color='red')
# plt.plot(Benchmark,label='Benchmark',color='black')
#
# plt.legend()
# plt.show()

# app = xw.App(visible=False, add_book=False)
# workbook = app.books.open('D:\Python\MacroFactorModel\Strategy.xlsx')
# worksheet = workbook.sheets['Strategy']
# worksheet.range('A1').value = Strategy
# workbook.save()
# app.kill()


'''计算月度收益'''
# Long_Win=0
# Long=0
# Short_Win=0
# Short=0
#
#
# for _ in range(len(Strategy) - 1):
#     if Strategy['Signal'].iloc[_] != Strategy['Signal'].iloc[_ + 1]:
#         if Strategy['Signal'].iloc[_+1] == 1:
#             Long += 1
#             NxtSht = np.where(Strategy['Signal'].iloc[_ + 1:].values != 1)[0]
#             if len(NxtSht) > 0:
#                 if Strategy['StrategyNAV'].iloc[_ + 1 + NxtSht[0] - 1] > Strategy['StrategyNAV'].iloc[_]:
#                     Long_Win += 1
#             else:
#                 if Strategy['StrategyNAV'].iloc[-1] > Strategy['StrategyNAV'].iloc[_]:
#                     Long_Win += 1
#         elif Strategy['Signal'].iloc[_+1] == -1:
#             Short += 1
#             NxtLng = np.where(Strategy['Signal'].iloc[_ + 1:].values != -1)[0]
#             if len(NxtLng) > 0:
#                 if Strategy['StrategyNAV'].iloc[_ + 1 + NxtLng[0] - 1] < Strategy['StrategyNAV'].iloc[_]:
#                     Short_Win += 1
#             else:
#                 if Strategy['StrategyNAV'].iloc[-1] < Strategy['StrategyNAV'].iloc[_]:
#                     Short_Win += 1
#
# Long_WinRatio = Long_Win / Long
# Short_WinRatio = Short_Win / Short