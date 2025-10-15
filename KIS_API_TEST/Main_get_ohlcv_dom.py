# GET OHLCV USING KIS
# 현재 지정일의 100일 까지만 조회. 추후 업데이트 예정이라 함. 그에따라 수정해야 할 
# REST API call limit : 10 per sec. --> 0.17 sec for each call
import kis_auth as ka
import kis_domstk as kb

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


# Code
# code = '069500'    # KODEX 200
code = '114800'    # KODEX 인버스


# Date of today
EOD = dt.datetime.today().strftime('%Y%m%d')
BOD = []

# Init. DF
df = pd.DataFrame()


# Loop for data extraction
ind = 1
start_time_total = time.time()
while True:
    start_time = time.time()
    
    # Get KIS ohlcv (from input date to last 100 days)
    # rt_data = kb.get_overseas_price_quot_dailyprice(excd=code, itm_no=tick, gubn='0', bymd=EOD, modp='1')
    if ind == 1:
        rt_data = kb.get_inquire_daily_itemchartprice(output_dv="2", inqr_end_dt=EOD, itm_no=code)
        ind = 0
    else:
        rt_data = kb.get_inquire_daily_itemchartprice(output_dv="2", inqr_strt_dt=BOD, inqr_end_dt=EOD, itm_no=code)
        
    # Check data
    if rt_data.empty:
        break
    else:
        # Get ohlcv
        df_sub = pd.DataFrame()
        df_sub['Open'] = rt_data['stck_oprc'].astype(float)
        df_sub['High'] = rt_data['stck_hgpr'].astype(float)
        df_sub['Low'] = rt_data['stck_lwpr'].astype(float)
        df_sub['Close'] = rt_data['stck_clpr'].astype(float)
        df_sub['Volume'] = rt_data['acml_vol'].astype(float)
        df_sub.index = pd.DatetimeIndex(rt_data['stck_bsop_date'].values)
        
        # Update DF
        df = pd.concat([df, df_sub])
        
        # Update EOD
        EOD = dt.datetime.strftime(df_sub.index[-1] + dt.timedelta(-1), '%Y%m%d')
        BOD = dt.datetime.strftime(df_sub.index[-1] + dt.timedelta(-28), '%Y%m%d')
        # Time delay for REST API call limit
        time.sleep(0.07)
        
        # Status
        print(f'EOD = {EOD}, runtime = {time.time()-start_time}')
        
        
# Flip data in ascending order
df = df.iloc[::-1]
df.index.name = 'Date'
print(f'total runtime = {time.time()-start_time_total}')

# Save file
df.to_csv(code + '.csv')

# Plot
mpf.plot(df,type='candle',style='charles')


