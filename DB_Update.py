from WindPy import w

import datetime
import pandas as pd
import numpy as np

from tqdm import tqdm
import xlwings as xw

def LocalDataBase_Update(DB_Add,Data,StartDate='2023-01-01',EndDate='2023-09-01',Global=False):
    Orig_Data=pd.read_excel(DB_Add,sheet_name=Data,index_col=0,header=0)
    Update_StartDate=Orig_Data.index[(Orig_Data.index>=StartDate) & (Orig_Data.index<=EndDate)][0]
    Update_EndDate=Orig_Data.index[(Orig_Data.index>=StartDate) & (Orig_Data.index<=EndDate)][-1]

    w.start()
    for column in tqdm(range(1, Orig_Data.shape[1])):
        for row in range(max(1,list(Orig_Data.index).index(Update_StartDate)),min(Orig_Data.shape[0],list(Orig_Data.index).index(Update_EndDate)+1)):
            Time=w.edb(Orig_Data.columns[column], Orig_Data.index[row], Orig_Data.index[row]).Times[0]
            if Global:
                error_code, edb_data = w.edb(Orig_Data.columns[column], Orig_Data.index[row], Orig_Data.index[row],  usedf=True)
                if error_code == 0:
                    pass
                else:
                    print("Error Code:", error_code)
                    print("Error Message:", edb_data.iloc[0, 0])
                if Time<=Orig_Data.index[row].date():
                    Orig_Data.iloc[row,column]=edb_data.iloc[0,0]
            else:
                if Orig_Data.iloc[row,column]!=Orig_Data.iloc[row,column]:
                    error_code, edb_data = w.edb(Orig_Data.columns[column], Orig_Data.index[row], Orig_Data.index[row], usedf=True)
                    if error_code == 0:
                        pass
                    else:
                        print("Error Code:", error_code)
                        print("Error Message:", edb_data.iloc[0, 0])

                    if Time <= Orig_Data.index[row].date():
                        Orig_Data.iloc[row, column] = edb_data.iloc[0, 0]
    w.stop()

    app = xw.App(visible=False, add_book=False)
    workbook = app.books.open(DB_Add)
    worksheet = workbook.sheets[Data]
    worksheet.range('A1').value = Orig_Data
    workbook.save()
    app.kill()

    return Orig_Data

'''Test'''
DB_Add='D:\Python\MacroFactorModel\MacroDB\宏观经济数据.xlsx'
Data='中国-官方PMI'

Output=LocalDataBase_Update(DB_Add,Data,StartDate='2000-08-01',EndDate='2023-12-01',Global=True)
