# R02
# - 전략 amnt 수식형태(문자열) 변경
# - 후 처리 코드 개선


# PATH FOR MY MODULE
import sys
sys.path.append('/Users/ku/Library/CloudStorage/OneDrive-개인/QUANT/gu/')


# MODULES
from quant import turtle
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path
import numpy as np


# DF DISPLAY OPTIONS
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# pd.reset_option('max_columns')
# pd.reset_option('max_rows')

# Trading days
# Test #1 (for post-processing coding)
# BOD = '2024-06-03'
# EOD = '2025-02-24'

# Down -> Up
# BOD = '2021-12-01'
# EOD = '2025-03-11'

# Test period [recent 5-year]
# BOD = '2015-05-08'
# BOD = '2020-05-08'
# EOD = '2025-09-04'

# For real trading
BOD = '2025-01-03'
EOD = '2025-10-10'

# INIT. STOCK
s1 = turtle.stock(code='NAS', tick='TQQQ')
s2 = turtle.stock(code='NAS', tick='SQQQ')


# INIT. PORTFOLIO
pf = turtle.portfolio()
pf.add_stock(stock_cls=s1, tag='TQ', rmin=0.0, amax=1)
pf.add_stock(stock_cls=s2, tag='SQ', rmin=0.0, amax=0.001)

            
# INIT. TRADING ACCOUNT
ac = turtle.account(pf, amount=10000)


# INIT. STRATEGY
st = turtle.strategy(pf)


### TQ
# Set stock tag for strategy generation
st.set_stock_tag('TQ')


# Buy
st.buy_lim(tag='buy#1', sgnl='pric > High20', cond=['fac == 1','ecnt[TQ,buy]==0'], amnt='0.25', lamt=1, repc=1)
st.buy_lim(tag='buy#2', sgnl='pric > High20', cond=['fac == 2','ecnt[TQ,buy]==0'], amnt='0.25', lamt=1, repc=1)
st.buy_lim(tag='buy#3', sgnl='pric > High20', cond=['fac == 6','ecnt[TQ,buy]==0'], amnt='0.25', lamt=1, repc=1)
st.gen_set(tag='buy', tlst=['buy#1','buy#2','buy#3'], ceky=-1, ropt=0)
st.buy_lim(tag='add#1', sgnl='pric > Pavg + 0.5*ATR', cond=['dcnt[TQ,buy]==0','fac == 1'], amnt='0.25', repc=0, repd=1)
st.buy_lim(tag='add#2', sgnl='pric > Pavg + 0.5*ATR', cond=['dcnt[TQ,buy]==0','fac == 2'], amnt='0.25', repc=0, repd=1)
st.buy_lim(tag='add#3', sgnl='pric > Pavg + 0.5*ATR', cond=['dcnt[TQ,buy]==0','fac == 6'], amnt='0.25', repc=0, repd=1)
st.gen_set(tag='add', tlst=['add#1','add#2','add#3'], ceky=-1, ropt=0)
st.gen_set(tag='BUY', tlst=['buy', 'add'], ceky=1, ropt=0)      # 매수전략 셋팅

# st.buy_lim(tag='buy#1', sgnl='pric > High20', amnt='0.2', lamt=1, repc=1)
# st.buy_lim(tag='add#1', sgnl='pric > Pavg + 0.5*ATR', pric='pric', amnt='0.2', repc=0, repd=1)
# st.gen_set(tag='BUY', tlst=['buy#1', 'add#1'], ceky=1, ropt=0)      # 매수전략 셋팅

# Sel
st.sel_lim(tag='sel#1', sgnl='pric < Low10', amnt='1', repc=1)
# st.sel_lim(tag='sel#1', sgnl='pric < Low20', amnt='1', repc=1)
# st.sel_lim(tag='sel#1', sgnl='pric < E20', amnt='1', repc=1)
st.sel_lim(tag='sls#1', sgnl='pric < Pavg - 2*ATR', amnt='1')
st.gen_set(tag='SEL', tlst=['sel#1', 'sls#1'])   # 매도전략 셋팅

# Total
st.gen_set(tag='tstg', tlst=['SEL','BUY'], ceky=-1)  # 전체전략 셋팅(매도우선)
# st.gen_set(tag='tstg', tlst=['BUY','SEL'], ceky=1)  # 전체전략 셋팅(매도우선)



### SQ
# Set stock tag for strategy generation
st.set_stock_tag('SQ')

# buy
st.buy_lim(tag='buy#1', sgnl='pric > High20', amnt='0.2', lamt=1, repc=1)
st.buy_lim(tag='add#1', sgnl='pric > Pavg + 0.5*ATR', pric='pric', amnt='0.2', repc=0, repd=1)
st.gen_set(tag='BUY', tlst=['buy#1', 'add#1'], ceky=1, ropt=0)      # 매수전략 셋팅

# sel
st.sel_lim(tag='sel#1', sgnl='pric < Low10', pric='pric', amnt='1', repc=1)
st.sel_lim(tag='sls#1', sgnl='pric < Pavg - 2*ATR', pric='pric', amnt='1')
st.gen_set(tag='SEL', tlst=['sel#1', 'sls#1'])   # 매도전략 셋팅

# total
st.gen_set(tag='tstg', tlst=['SEL','BUY'], ceky=-1)      # 전체전략 셋팅(매도우선)

# st.TQ1.display()
# st.TQ2.display()


# INIT. BACKTESTING
bt = turtle.backtesting(ac, st, BOD, EOD)

# RUN BACKTESTING
bt.run()


# =============================================================================
# POST-PROCESS
# =============================================================================
pt = turtle.postprocessing(bt)


# Trading history
pt.eval_trading_history()


### Plot history
pt.plot_his('TQ', cycl='all', viwl=1)
pt.plot_his('SQ', cycl='all', viwl=1)


### Plot cumul. return & ddwn
pt.plot_cuml_and_mdd()




# Plot hitory with add plot
H55 = mpf.make_addplot(pf.TQ.df['High55'], label='55H', width=.5, color='g', panel=0, secondary_y=False)
H20 = mpf.make_addplot(pf.TQ.df['High20'], label='20H', width=.5, color='g', panel=0, linestyle='--', secondary_y=False)
L55 = mpf.make_addplot(pf.TQ.df['Low55'], label='55L', width=.5, color='c', panel=0, secondary_y=False)
L20 = mpf.make_addplot(pf.TQ.df['Low20'], label='20L', width=.5, color='c', panel=0, linestyle='--', secondary_y=False)

E20 = mpf.make_addplot(pf.TQ.df['Close'].ewm(span=4*5).mean(), label='4w', width=.5, color='r', panel=0, secondary_y=False)
E50 = mpf.make_addplot(pf.TQ.df['Close'].ewm(span=10*5).mean(), label='10w', width=.5, color='b', panel=0, secondary_y=False)
E100 = mpf.make_addplot(pf.TQ.df['Close'].ewm(span=20*5).mean(), label='20w', width=.5, color='g', panel=0, secondary_y=False)
apdict = [E20, E50, E100, H55, H20, L55, L20]
pt.plot_his('TQ', cycl='all', viwl=0.1, sgpl='on', adpl=apdict)





# MACD plot
EMA = mpf.make_addplot(pf.TQ.df.loc[BOD:EOD,'E10'], label='EMA', width=.5, color='r')
EMA1 = mpf.make_addplot(pf.TQ.df.loc[BOD:EOD,'E20'], label='EMA', width=.5, color='g')
EMA2 = mpf.make_addplot(pf.TQ.df.loc[BOD:EOD,'E40'], label='EMA', width=.5, color='b')
MACD1_TOP = mpf.make_addplot(pf.TQ.df.loc[BOD:EOD,'MACD1_TOP'], width=.5, color='fuchsia',
                        panel=1, secondary_y=True)
MACD_SIG_TOP = mpf.make_addplot(pf.TQ.df.loc[BOD:EOD,'MACD_TOP'], label='signal', width=.5, color='b',
                        panel=1, secondary_y=True)
MACD_dSIG_TOP = mpf.make_addplot(np.gradient(pf.TQ.df.loc[BOD:EOD,'MACD1_TOP']), type='bar', width=0.7,
                        panel=1, color='dimgray', alpha=.5, secondary_y=False)

MACD1_MID = mpf.make_addplot(pf.TQ.df.loc[BOD:EOD,'MACD1_MID'], label='MACD', width=.5, color='fuchsia',
                        panel=2, secondary_y=True)
MACD_SIG_MID = mpf.make_addplot(pf.TQ.df.loc[BOD:EOD,'MACD_MID'], label='signal', width=.5, color='b',
                        panel=2, secondary_y=True)
MACD_dSIG_MID = mpf.make_addplot(np.gradient(pf.TQ.df.loc[BOD:EOD,'MACD1_MID']), type='bar', width=0.7,
                        panel=2, color='dimgray', alpha=.5, secondary_y=False)

MACD1_BOT = mpf.make_addplot(pf.TQ.df.loc[BOD:EOD,'MACD1_BOT'], label='MACD', width=.5, color='fuchsia',
                        panel=3, secondary_y=True)
MACD_SIG_BOT = mpf.make_addplot(pf.TQ.df.loc[BOD:EOD,'MACD_BOT'], label='signal', width=.5, color='b',
                        panel=3, secondary_y=True)
MACD_dSIG_BOT = mpf.make_addplot(np.gradient(pf.TQ.df.loc[BOD:EOD,'MACD1_BOT']), type='bar', width=0.7,
                        panel=3, color='dimgray', alpha=.5, secondary_y=False)

apdict = [EMA, EMA1, EMA2, MACD1_TOP, MACD_SIG_TOP, MACD_dSIG_TOP, MACD1_MID, MACD_SIG_MID, MACD_dSIG_MID, MACD1_BOT, MACD_SIG_BOT, MACD_dSIG_BOT]
mpf.plot(pf.TQ.df.loc[BOD:EOD,:],
         type = 'candle', 
         style = 'charles',
         volume = False,
         volume_panel = 2,
         show_nontrading = False, 
         addplot = apdict,
         title = 'History',
         ylabel = 'Price [₩]',
         # figsize = (20,10),
         figratio=(8,5),
         figscale=0.7,
         scale_padding = 0.1,
         ylim = (0.95*pf.TQ.df.loc[BOD:EOD,'Low'].min(),
                 1.05*pf.TQ.df.loc[BOD:EOD,'High'].max()),
         )












# import gspread
# from google.oauth2.service_account import Credentials

# # 1. Authorize (adjust the scope and JSON file path as needed)
# scope = ['https://www.googleapis.com/auth/spreadsheets']
# creds = Credentials.from_service_account_file("knj-invest-dc83a0b7fb30.json", scopes=scope)
# client = gspread.authorize(creds)

# # 2. Open the spreadsheet by name or ID
# spreadsheet_url = "https://docs.google.com/spreadsheets/d/1ReQJiMrpd1_ZJoM50SFjrmlQQa2ZdT40_MnyHL2-JhA/edit?gid=961839948#gid=961839948"
# spreadsheet = client.open_by_url(spreadsheet_url)

# # 3. Add a new worksheet
# new_sheet = spreadsheet.add_worksheet(title='NewSheetName', rows=100, cols=20)

# # 4. (Optional) Write data to it
# new_sheet.update('A1', [['Hello', 'World']])


# # Assume pt.his['TQ'] is a DataFrame
# data = pt.his['TQ']

# # Convert to list with headers
# values = [data.columns.tolist()] + data.values.tolist()

# # Update the sheet starting at cell A1
# new_sheet.update('A20', values)


