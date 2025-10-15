# GET OHLCV USING KIS
# 현재 지정일의 100일 까지만 조회. 추후 업데이트 예정이라 함. 그에따라 수정해야 할 
# REST API call limit : 10 per sec. --> 0.17 sec for each call
import kis_auth as ka
import kis_ovrseastk as kb

import pandas as pd
import matplotlib.pyplot as plt
# from quant.turtle.analysis import *
import numpy as np
import mplfinance as mpf
import sys
import datetime as dt
import time




# KIS token
ka.auth()


# Ticker
code = 'NAS'    # 나스닥 / 뉴욕 / 아멕스 ['NAS', 'NYS', 'AMS']
tick = 'TQQQ'
# tick = 'SQQQ'
# tick = 'QQQ'
# tick = 'TSLL'
# tick = 'NVDL'
# tick = 'JEPQ'

# AMS
# code = 'AMS'
# tick = 'SPY'
# tick = 'PSQ'
# tick = 'FAS'
# tick = 'FAZ'
# tick = 'TMF'
# tick = 'TMV'
# tick = 'DRN'
# tick = 'DRV'
# tick = 'UPRO'
# tick = 'SPXU'
# tick = 'SOXL'
# tick = 'SOXS'
# tick = 'QLD'
# tick = 'NUGT'
# tick = 'JEPI'
# tick = 'BIL'
# tick = 'SCHD'
# tick = 'VTV'
# tick = 'DIVO'


# NYS
# code = 'NYS'
# tick = 'BRKB'


# Date of today
EOD = dt.datetime.today().strftime('%Y%m%d')


# Init. DF
df = pd.DataFrame()


# Loop for data extraction
start_time_total = time.time()
while True:
    start_time = time.time()
    
    # Get KIS ohlcv (from input date to last 100 days)
    rt_data = kb.get_overseas_price_quot_dailyprice(excd=code, itm_no=tick, gubn='0', bymd=EOD, modp='1')
    
    # Check data
    if rt_data.empty:
        break
    else:
        # Get ohlcv
        df_sub = pd.DataFrame()
        df_sub['Open'] = rt_data['open'].astype(float)
        df_sub['High'] = rt_data['high'].astype(float)
        df_sub['Low'] = rt_data['low'].astype(float)
        df_sub['Close'] = rt_data['clos'].astype(float)
        df_sub['Volume'] = rt_data['tvol'].astype(float)
        df_sub.index = pd.DatetimeIndex(rt_data['xymd'].values)
        
        # Update DF
        df = pd.concat([df, df_sub])
        
        # Update EOD
        EOD = dt.datetime.strftime(df_sub.index[-1] + dt.timedelta(-1), '%Y%m%d')
        
        # Time delay for REST API call limit
        time.sleep(0.07)
        
        # Status
        print(f'EOD = {EOD}, runtime = {time.time()-start_time}')
        
        
# Flip data in ascending order
df = df.iloc[::-1]
df.index.name = 'Date'
print(f'total runtime = {time.time()-start_time_total}')

# Save file
df.to_csv(tick + '.csv')

# Plot
mpf.plot(df,type='candle',style='charles')


