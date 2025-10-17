################# POST-PROCESSING MODULE ######################################
import matplotlib.pyplot as plt
import pandas as pd
import mplfinance as mpf
import numpy as np
import quant.turtle.subr as subr

# =============================================================================        
# POST PROCESS CLASS
# =============================================================================
class postprocessing:
    
    def __init__(self, bt_cls):
        '''
        Core results DFs
        '''
        # Duplicate DF
        self._bt = bt_cls
        
        # Init. core variable
        # self.log = self._bt
        # self.res = self._bt.res.copy()

        # Cumulative return (w.r.t entire account value)
        self._bt.res['PF']['cuml'] = (1 + self._bt.res['PF']['RoRa']).cumprod()
        
        # Drawdown [%]
        self._bt.res['PF']['ddwn'] = 100*(self._bt.res['PF']['cuml'] - self._bt.res['PF']['cuml'].cummax()) / self._bt.res['PF']['cuml'].cummax()
                
        


    def eval_trading_history(self):
        '''
        Display & save trading history
        
        개별 및 전체에 대한 히스토리 생성할 것!!
        '''
        # GET HISTORY FOR STOCKS
        his = {}
        for tag in self._bt._pf.tag:
            his[tag] = self._get_his_st(tag)
        
        # GET HISTORY FOR PF
        his['PF'] = self._get_his_pf(his)
        
        # SAVE
        self.his = his
        
        # tb column list
        # tnum | BOTD | EOTD | 투자 ticker tuple | days | MTAR | RoR | MDD | Cumul |
        
        print('\n')
        print(self.his['PF'].to_string())
    
    
    
    def _get_his_st(self, tag):
        '''
        History DF for individual stocks
        Items:
            cycl(index), BOTD, EOTD, MTRA, RoR,
            cumP(누적수익), cumL(누적손실), win(누적승률), loss(누적패율),
            avgP(누적평균수익), avgL(누적평균손실), RR(평균손익비),
            TE(trading edge), E(PL)[기대손익] 
        '''
        if sum(self._bt.log[tag].any()):
            # Init. DF
            res = self._bt.res[tag]
            df = self._bt.log[tag]
            bal = self._bt.bal[tag]
            
            # Add Date column
            df['Date'] = df.index.strftime('%Y-%m-%d')
            
            # Generate empty history DF
            idx = df['cycl'].unique()   
            his = pd.DataFrame({}, index = idx)
            his.index.name = 'cycl'
            
            # Trading period
            df = df.groupby('cycl', group_keys=False)
            bal = bal.groupby('cycl', group_keys=False)
            his['BOTD'] = df['Date'].apply(lambda x: x[0])
            his['EOTD'] = df['Date'].apply(lambda x: x[-1])
            
            # Trading days for each period
            for i in range(0,len(his)):
                his.loc[his.index[i], 'days'] = str(len(res.loc[his['BOTD'].iloc[i]:his['EOTD'].iloc[i],:]))
            
            # Maximum ratio of trading amount
            his['MTRA[%]'] = 100 * bal['aloc'].apply(lambda x: x.max())
            
            # RoR (w.r.t trading period)
            PanL = df['PanL'].apply(lambda x: sum(x))
            amnt_max = bal['amnt'].apply(lambda x: x.max())
            his['RoR[%]'] = 100 * PanL/amnt_max
            
            # Cumulative Profit & Loss
            tmp = df['PanL'].apply(lambda x: x[-1])
            his['cumP'] = tmp.transform(lambda x: x.where(x > 0, 0).cumsum())
            his['cumL'] = tmp.transform(lambda x: x.where(x < 0, 0).cumsum())
            
            # Cumulative win & loss rate
            his['win[%]'] = 100 * (his['RoR[%]'] > 0).cumsum() / (abs(his['RoR[%]'])>0).cumsum()
            his['loss[%]'] = 100 * (his['RoR[%]'] < 0).cumsum() / (abs(his['RoR[%]'])>0).cumsum()
            
            # Average profit & loss
            his['avgP'] = (his['cumP'] / (his['win[%]']  * his.index / 100)).fillna(0)
            his['avgL'] = (his['cumL'] / (his['loss[%]'] * his.index / 100)).fillna(0)
            
            # Average profilt / loss ratio
            his['RR'] = his['avgP'] / abs(his['avgL'])
            his['RR'].replace(float('inf'), 0, inplace=True)
            
            # Trading Edge
            his['TE'] = (his['win[%]']/100) * his['avgP'] + (his['loss[%]']/100) * his['avgL'] 
            
            # Expected Profit & Loss
            his['E(PnL)'] = his['TE'] * his.index
            return his
    
    
    
    
    def _get_his_pf(self, his0):
        '''
        History DF for portfolio
        
        - his0 : history DF for individual stocks
    
        Items:
            tag  : 주식종목 태그 ID)
            cuml : cumulative return - 초기자금 대비 누적 수익률
            MDD  : MDD - 개별종목별 매입금액 기준 (총 금액 아님)
            cycl : 총 투자 사이클 수
            
            <수익 거래 시 계산 항목>
            +RoR[%] : 평균 수익률
            +TRA[%] : 평균 투자비율
            +d/cycl : 평균 투자기간 (싸이클 당)
            
            <손실 거래 시 계산 항목>
            -RoR[%] : 평균 수익률
            -TRA[%] : 평균 투자비율
            -d/cycl : 평균 투자기간 (싸이클 당)
            
            win[%] : 승률
            PF : profit factor [win RoR / loss RoR]
            TE : trading edge (win rate * win RoR + loss rate * loss RoR)
            E(TR) : 평균 총 기대 수익률
        '''
        # Init. DF
        df = self._bt.res['PF']
        
        # Add Date column
        his = pd.DataFrame({}, index = self._bt._pf.tag)
        his.index.name = 'tag'
        
        # Loop over stock tags
        for i in self._bt._pf.tag:
            
            # Cumulative return w.r.t initial amount
            amnt0 = df['vala'].iloc[0]      # Initial amount
            his.loc[i,'cuml'] = 1 + (his0[i]['cumP'].iloc[-1] + his0[i]['cumL'].iloc[-1]) / amnt0
            
            # MDD
            res = self._bt.res[i][self._bt.res[i]['cycl'] > 0].groupby('cycl')
            ddwn = his0['TQ']['RoR[%]'] - res['RoR'].apply(lambda x: max(x.cummax()))
            his.loc[i,'MDD[%]'] = ddwn.min()
            
            # Cycle
            his.loc[i,'cycl'] = his0[i].index[-1]
            
            # FOR PROFIT [average RoR, TRA, days per cycl.]
            his.loc[i,'+RoR[%]'] = his0[i]['RoR[%]'][his0[i]['RoR[%]']>0].mean()
            his.loc[i,'+TRA[%]'] = his0[i]['MTRA[%]'][his0[i]['RoR[%]']>0].mean()
            his.loc[i,'+d/cycl'] = his0[i]['days'][his0[i]['RoR[%]']>0].astype(int).sum() / his.loc[i,'cycl']
            his.replace(float('NaN'), 0, inplace=True)
            
            # FOR LOSS [average RoR, TRA, days per cycl.]
            his.loc[i,'-RoR[%]'] = his0[i]['RoR[%]'][his0[i]['RoR[%]']<0].mean()
            his.loc[i,'-TRA[%]'] = his0[i]['MTRA[%]'][his0[i]['RoR[%]']<0].mean()
            his.loc[i,'-d/cycl'] = his0[i]['days'][his0[i]['RoR[%]']<0].astype(int).sum() / his.loc[i,'cycl']
            his.replace(float('NaN'), 0, inplace=True)
            
            # Win rate (cumulative)
            his.loc[i,'win[%]'] = his0[i]['win[%]'].iloc[-1]
            his.loc[i,'PF'] = ((his.loc[i, '+RoR[%]'] * his.loc[i, '+TRA[%]']) / 
                               abs(his.loc[i, '-RoR[%]'] * his.loc[i, '-TRA[%]']))
            his.replace(float('NaN'), 0, inplace=True)
            
            # Trading Edge
            his.loc[i,'TE'] = ((his.loc[i,'win[%]']/100 *
                                  (his.loc[i, '+RoR[%]'] * his.loc[i, '+TRA[%]']/100)) + 
                                  ((1-his.loc[i,'win[%]']/100) * 
                                  (his.loc[i, '-RoR[%]'] * his.loc[i, '-TRA[%]']/100)))
            
            # Expected RoR
            his.loc[i,'E(TR)'] = his.loc[i,'TE'] * his.loc[i,'cycl']
        return his




#!!! PLOT METHODS #############################################################
    def plot_cuml_and_mdd(self):
        df = self._bt.res['PF']
    
        fig, ax1 = plt.subplots()
    
        # Left Y-axis (Cumulative Return)
        color1 = 'tab:blue'
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Cumulative Return', color=color1, fontsize=12)
        ax1.plot(df.index, df['cuml'], label='Cumulative Return', color=color1, linewidth=1, linestyle='-')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(alpha=0.3)
    
        # Right Y-axis (Drawdown)
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Drawdown [%]', color=color2, fontsize=12)
        ax2.bar(df.index, df['ddwn'], label='Drawdown', color=color2, alpha=0.3)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.grid(False)
        ax2.set_ylim(-100, 0)
        
        # X-axis formatting
        # plt.xticks(rotation=45)
        fig.autofmt_xdate()  # << Key line to rotate x-axis labels properly
        plt.title('Cumulative Return and Drawdown', fontsize=14, fontweight='bold')
    
        # Combine legends from both axes
        # lines_1, labels_1 = ax1.get_legend_handles_labels()
        # lines_2, labels_2 = ax2.get_legend_handles_labels()
        # ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=10)
    
        # fig.tight_layout()
        plt.show()
        


        
    def plot_his(self, tag, cycl='all', viwl=1.0e5, sgpl='on', adpl=None):
        '''
        Plot trading history
        
        - tag  : tag for stock (not for 'PF')
        - cycl : cycle for plot [1,2,..N] [list] 
               : 'all' for all cycles (default)
        - viwl : view level of cycle. [0 to 1]
               : 0 for close-up view
               : very large value for all trading day (default)
        - sgpl : signal plot 'on' (default)
               : buy (red), add (?), sel(blue), sls(?) 
        - adpl : add plot in mplfinance plot [dict]
               : should be equal lengh to ohlcv
               
        '''
        ### tick
        # PLOT TARGET
        log = self._bt.log[tag]
        # res = self.res[tag]
        his = self.his[tag]
        ohlcv = self._bt._pf.__dict__[tag].df       # Full history
        inds0 = ohlcv.index.get_loc(self._bt.res['PF'].index[0])
        inde0 = ohlcv.index.get_loc(self._bt.res['PF'].index[-1])
        ohlcv = ohlcv[inds0:inde0+1]                  # Desired range
        
        
        ### cycl
        # PLOT (CYCLE) RANGE
        if cycl == 'all':
            cyc = his.index.to_list()
        else:
            cyc = cycl
        date_s = his.loc[cyc,'BOTD'].min()
        date_e = his.loc[cyc,'EOTD'].max()
            
        # DATA SLICING FOR PLOT
        inds = ohlcv.index.get_loc(date_s)
        inde = ohlcv.index.get_loc(date_e)
        
        
        ### viwl
        # DATA SLICING WITH VIEW LEVEL
        inds = int(max(0, np.ceil(inds*(1-viwl))))
        inde = int(min(len(ohlcv), np.ceil(inde*(1+viwl))))
        
        
        ### sgpl
        # INDICES FOR [BUY/SEL] SIGNAL
        df = pd.Series([False]*len(ohlcv), index=ohlcv.index)
        isig = {tag: df.copy() for tag in ['BUY','SEL']}
        for sname in log['strg'].unique():
            # Find super name [BUY / SEL]
            tags_super = self._bt._st.__dict__[tag].find_all_super_tags(sname)
            sig = tags_super[2]
            
            # Indices for strategy
            strg = log['strg'].apply(lambda x: x == sname)
            for i in range(0,len(strg)):
                if strg.iloc[i] == True:
                    isig[sig].loc[strg.index[i]] = strg.iloc[i]
            
        # CYCLE FILTERING [True for cycl > 0]
        fltr = pd.Series([False]*len(ohlcv), index=ohlcv.index)
        for icyc in cyc:
            # Set filter
            ii = ohlcv.index.get_loc(his.loc[icyc, 'BOTD'])
            jj = ohlcv.index.get_loc(his.loc[icyc, 'EOTD'])
            fltr[ii:jj+1] = True
        
        # X-DATA FOR CYCLE PLOT
        xdf = pd.Series([False]*len(ohlcv), index=ohlcv.index)
        for icyc in cyc:
            idx_s = his.loc[icyc, 'BOTD']
            idx_e = his.loc[icyc, 'EOTD']
            xdf.iloc[xdf.index.get_loc(idx_s):xdf.index.get_loc(idx_e)+1] = True
            
        # DATA FILTERING
        df = pd.Series([0]*len(ohlcv), index=ohlcv.index)
        data_tr_mark = {}                                   # Data for trading marker [BUY or SEL]
        data_tr_pric = {tag: df.copy() for tag in ['BUY','SEL']}   # Date for trading price
        for sname in log['strg'].unique():
            # Find super name [BUY / SEL]
            tags_super = self._bt._st.__dict__[tag].find_all_super_tags(sname)
            sig = tags_super[2]
            
            # Data for buy/sell marker
            if sig == 'BUY':
                data_tr_mark[sig] = 0.99 * ohlcv['Low'].copy()
            elif sig == 'SEL':
                data_tr_mark[sig] = 1.01 * ohlcv['High'].copy()
            
            # Data for trading price
            data = log['pric'][log['strg'] == sname]
            data_tr_pric[sig].loc[data.index] = data
            
            # Set NaN outside the data range (for empty plot)
            data_tr_mark[sig][~(isig[sig] * fltr)] = np.nan
            data_tr_pric[sig][~(isig[sig] * fltr)] = np.nan
            
        # GENERATE ADD PLOT DATA
        apdict = []
        if sgpl == 'on':
            if sum(data_tr_mark['BUY'] > 0) > 0:
                apdict.append(mpf.make_addplot(data_tr_mark['BUY'].iloc[inds:inde+1], type='scatter', markersize=10,
                                                marker='^', color='darkred', panel=0, secondary_y=False))
               
                apdict.append(mpf.make_addplot(data_tr_pric['BUY'].iloc[inds:inde+1], type='scatter', markersize=10,
                                                marker='o', color='dimgrey', panel=0, secondary_y=False))
                

            if sum(data_tr_mark['SEL'] > 0) > 0:
                apdict.append(mpf.make_addplot(data_tr_mark['SEL'].iloc[inds:inde+1], type='scatter', markersize=10,
                                                marker='v', color='b', panel=0, secondary_y=False))
               
                apdict.append(mpf.make_addplot(data_tr_pric['SEL'].iloc[inds:inde+1], type='scatter', markersize=10,
                                                marker='o', color='dimgrey', panel=0, secondary_y=False))
        
        
        ### adpl
        if adpl != None:
            # Data slicing
            for i in range(0,len(adpl)):
                # Nan filtering
                adpl[i]['data'] = adpl[i]['data'].apply(lambda x: np.nan if x == 0 else x)
                # Truncation from full history
                adpl[i]['data'] = adpl[i]['data'].iloc[inds0:inde0+1]
                # Data slicing for plot
                adpl[i]['data'] = adpl[i]['data'].iloc[inds:inde+1]

                apdict.append(adpl[i])


        ### Williams Accumulation/Distribution (WAD) panel
        wad_plots = []
        wad_panel = None
        if 'WAD' in ohlcv.columns:
            def _panel_index(addplot):
                panel_idx = None
                if isinstance(addplot, dict):
                    panel_idx = addplot.get('panel', None)
                else:
                    panel_idx = getattr(addplot, 'panel', None)
                return 0 if panel_idx is None else panel_idx

            existing_panels = {0}
            for addplot in apdict:
                existing_panels.add(_panel_index(addplot))

            wad_panel = max(existing_panels) + 1

            wad_plots.append(
                mpf.make_addplot(
                    ohlcv['WAD'].iloc[inds:inde+1],
                    panel=wad_panel,
                    color='tab:purple',
                    width=1,
                    secondary_y=False,
                )
            )

            if 'WAD_EMA' in ohlcv.columns:
                wad_plots.append(
                    mpf.make_addplot(
                        ohlcv['WAD_EMA'].iloc[inds:inde+1],
                        panel=wad_panel,
                        color='tab:orange',
                        width=1,
                        secondary_y=False,
                    )
                )

        apdict.extend(wad_plots)

        panel_ratios = None
        if wad_plots:
            panel_count = max(_panel_index(plot) for plot in apdict) + 1
            ratios = [1] * panel_count
            ratios[0] = 3
            ratios[wad_panel] = 1
            panel_ratios = tuple(ratios)


        ### plot
        # mc = mpf.make_marketcolors(base_mpf_style='yahoo', up="gray", down="k", edge="none", wick="black", volume="i", ohlc="i")
        mc = mpf.make_marketcolors(up='r', down='b', inherit=True)
        mc = mpf.make_marketcolors(up='lightcoral', down='cornflowerblue', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle=":", y_on_right=False)
        wconfig={}

        plt.figure()
        fig, axlist = mpf.plot(
        # mpf.plot(
            ohlcv.iloc[inds:inde+1],    
            # type = 'candle',
            # type = 'bar',
            # style = 'classic',
            style = s,
            # mav = (20,55),
            # volume = True,
            show_nontrading = False,
            addplot = apdict,
            title = 'PRICE HISTORY',
            ylabel = '$',
            # yscale = 'log',
            # vlines = dict(vlines=['2023-07-17','2023-07-18'], linewidths=(0.01,0.01)),
            # vlines = dict(vlines=bb, linewidths = 1, alpha=0.6),
            fill_between =
                dict(y1 = 1.1*ohlcv['High'].iloc[inds:inde+1].max(),
                      y2 = 0,
                      where=xdf[inds:inde+1],
                      alpha=0.1,
                      color='b'),
            ylim = (0.95*ohlcv['Low'].iloc[inds:inde+1].min(),
                    1.05*ohlcv['High'].iloc[inds:inde+1].max()),
            axisoff = False,
            figsize = (5,3),
            figratio=(8,5),
            figscale=1.1,
            scale_padding = 0.5,
            # update_width_config=dict(
            #     candle_linewidth=1.0, candle_width=0.8, volume_linewidth=1.0
            returnfig=True,
            update_width_config=dict(
                candle_linewidth=0.5, candle_width=0.5, volume_linewidth=0.0),
            return_width_config=wconfig,
            panel_ratios=panel_ratios,
            )
        # print(wconfig)
        plt.savefig('test.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        
        # mpf.plot(ohlcv.iloc[inds:inde+1], type='candle', style='yahoo', volume=True)
        # fig, axlist = mpf.plot(
        # ohlcv.iloc[inds:inde+1],
        # type="candle",
        # mav=(20, 50),
        # volume=True,
        # # title=f"\n{s_ticker} - Last 6 months",
        # title = 'price history',
        # # addplot=ap0,
        # xrotation=10,
        # style=s,
        # figratio=(13, 7),
        # figscale=1.10,
        # figsize=(10,3),
        # # scale_padding = 0.7,
        # update_width_config=dict(
        #     candle_linewidth=1.0, candle_width=0.8, volume_linewidth=1.0
        # ),
        # # tight_layout=True,
        # returnfig=True
        # )





