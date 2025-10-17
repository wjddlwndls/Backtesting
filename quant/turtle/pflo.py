'''
Modules for portfolio

List of modules
 - stock            : 개별종목 데이터 ohlcv 및 기술적 지표        
 - portfolio        : 주식 포트폴리오 구성
'''

import pandas as pd
# import datetime as dt
# import sys


# =============================================================================
# !!! STOCK CLASS
# =============================================================================
class stock:
    def __init__(self, code, tick):
        
        # Ticker code for KIS access [NAS / NYS / AMS]
        self.code = code
        
        # Ticker for stock
        self.tick = tick
        
        # INIT. stock historical data
        self.df = []
        self._init_stock_his()
        
        
    
    def _init_stock_his(self):
        '''
        Init. stock historical data
        '''
        # OHLCV
        self._get_ohlcv()
        
        # ATR
        self._get_ATR(column='ATR', N=20)
        
        # High
        self._get_HL(column='High200', N=200, HL='High')
        self._get_HL(column='High100', N=100, HL='High')
        self._get_HL(column='High55', N=55, HL='High')
        self._get_HL(column='High20', N=20, HL='High')
        self._get_HL(column='High15', N=15, HL='High')
        self._get_HL(column='High10', N=10, HL='High')
        self._get_HL(column='High5', N=5, HL='High')
        self._get_HL(column='High3', N=3, HL='High')
        self._get_HL(column='High2', N=2, HL='High')
        self._get_HL(column='High1', N=1, HL='High')

        
        # Low
        self._get_HL(column='Low200', N=200, HL='Low')
        self._get_HL(column='Low100', N=100, HL='Low')
        self._get_HL(column='Low55', N=55, HL='Low')
        self._get_HL(column='Low20', N=20, HL='Low')
        self._get_HL(column='Low15', N=15, HL='Low')
        self._get_HL(column='Low10', N=10, HL='Low')
        self._get_HL(column='Low7', N=7, HL='Low')
        self._get_HL(column='Low5', N=5, HL='Low')
        self._get_HL(column='Low3', N=3, HL='Low')
        self._get_HL(column='Low2', N=2, HL='Low')
        self._get_HL(column='Low1', N=1, HL='Low')
        
        # High/Low 55 filter
        self.df['F1'] = 0*self.df['Close'].copy()
        for i in range(0, len(self.df)):
            # Default value
            self.df['F1'].iloc[i] = self.df['F1'].shift(1).fillna(0).iloc[i]
            
            # Uptrend breakthrough
            if self.df['Close'].iloc[i] > self.df['High55'].iloc[i]:
                self.df['F1'].iloc[i] = 1
                
            # Downtrend breakthrough
            if self.df['Close'].iloc[i] < self.df['Low55'].iloc[i]:
                self.df['F1'].iloc[i] = -1
                
        
        # Low_sell limit
        self.df['L3'] = self.df['Low'].rolling(window=3).mean()
        self.df['L5'] = self.df['Low'].rolling(window=5).mean()
        self.df['L7'] = self.df['Low'].rolling(window=7).mean()
        self.df['L10'] = self.df['Low'].rolling(window=10).mean()
        
        # MACD
        self.df['MACD1_TOP'] = self.df['Close'].ewm(span=5).mean() - self.df['Close'].ewm(span=20).mean()
        self.df['MACD_TOP'] = self.df['MACD1_TOP'].ewm(span=9).mean()
        self.df['MACD1_MID'] = self.df['Close'].ewm(span=5).mean() - self.df['Close'].ewm(span=40).mean()
        self.df['MACD_MID'] = self.df['MACD1_MID'].ewm(span=9).mean()
        self.df['MACD1_BOT'] = self.df['Close'].ewm(span=20).mean() - self.df['Close'].ewm(span=40).mean()
        self.df['MACD_BOT'] = self.df['MACD1_BOT'].ewm(span=9).mean()
        
        self.df['MACD_TOP_SIG'] = self.df['MACD1_TOP'] - self.df['MACD_TOP']
        self.df['MACD_MID_SIG'] = self.df['MACD1_MID'] - self.df['MACD_MID']
        self.df['MACD_BOT_SIG'] = self.df['MACD1_BOT'] - self.df['MACD_BOT']
        
        # MACD diff (9-days EMA)
        self.df['dMACD_TOP'] = self.df['MACD_TOP'].diff(1)
        self.df['dMACD_MID'] = self.df['MACD_MID'].diff(1)
        self.df['dMACD_BOT'] = self.df['MACD_BOT'].diff(1)


        # EMA(Week)
        self.df['E10'] = self.df['Close'].ewm(span=4*5).mean()
        self.df['E20'] = self.df['Close'].ewm(span=10*5).mean()
        self.df['E40'] = self.df['Close'].ewm(span=20*5).mean()
        # EMA(week-version2 : 본주와 이평선이 유사하도록 평균일수 수정)
        # self.df['E10'] = self.df['Close'].ewm(span=3*5).mean()
        # self.df['E20'] = self.df['Close'].ewm(span=6*5).mean()
        # self.df['E40'] = self.df['Close'].ewm(span=12*5).mean()
        # EMA(days)
        # self.df['E10'] = self.df['Close'].ewm(span=5).mean()
        # self.df['E20'] = self.df['Close'].ewm(span=20).mean()
        # self.df['E40'] = self.df['Close'].ewm(span=40).mean()
        self.df['E200'] = self.df['Close'].ewm(span=200).mean()
        
        # Face 1
        ind = (self.df['E10'] > self.df['E20']) & (self.df['E20'] > self.df['E40'])
        fac = ind * 1
        
        # Face 2 & 6
        ind2 = (self.df['E20'] > self.df['E10']) & (self.df['E10'] > self.df['E40'])
        ind6 = (self.df['E10'] > self.df['E40']) & (self.df['E40'] > self.df['E20'])
        # fac[ind2 + ind6] = 2/3
        fac[ind2] = 2
        fac[ind6] = 6
        
        # Face 3 & 5
        ind3 = (self.df['E20'] > self.df['E40']) & (self.df['E40'] > self.df['E10'])
        ind5 = (self.df['E40'] > self.df['E10']) & (self.df['E10'] > self.df['E20'])
        fac[ind3] = 3
        fac[ind5] = 5
        
        # Face 4
        ind4 = (self.df['E40'] > self.df['E20']) & (self.df['E20'] > self.df['E10'])
        fac[ind4] = 4
        self.df['fac'] = fac
    
        
        # EMA 200 기준선
        ind_up = self.df['Low'] > self.df['E200']
        ind_dn = self.df['High'] < self.df['E200']
        tmp = 1 * ind_up
        tmp[ind_dn] = -1
        self.df['trend'] = tmp

        # Williams Accumulation/Distribution (WAD) line
        prev_close = self.df['Close'].shift(1)
        close = self.df['Close']
        high = self.df['High']
        low = self.df['Low']

        min_low_prev = pd.concat([low, prev_close], axis=1).min(axis=1)
        max_high_prev = pd.concat([high, prev_close], axis=1).max(axis=1)

        wad_step = pd.Series(0.0, index=self.df.index)
        cond_up = close > prev_close
        cond_down = close < prev_close

        wad_step[cond_up] = close[cond_up] - min_low_prev[cond_up]
        wad_step[cond_down] = close[cond_down] - max_high_prev[cond_down]

        self.df['WAD'] = wad_step.cumsum().fillna(0.0)
        self.df['WAD_EMA'] = self.df['WAD'].ewm(span=10).mean()
    
    def _get_ohlcv(self):
        '''
        Get stock price using KIS (later)
        '''
        # Get data from KIS
        self.df = pd.read_csv('../KIS_API_TEST/' + self.tick + '.csv', index_col='Date')
        
        # Set index type
        self.df.index = pd.DatetimeIndex(self.df.index)
        
        
        
    def _get_ATR(self, column='ATR', N=20):
        '''
        get ATR
        Args:
            - column : column name for DF
            - N      : # days for calculation
        '''
        # Loop over stocks
        HL = self.df['High'] - self.df['Low']
        HC = self.df['High']  - self.df['Close'].shift(1)
        LC = self.df['Low']  - self.df['Close'].shift(1)
        TR = pd.DataFrame({'HL' : HL, 'HC' : HC, 'LC' : LC})
        TR = TR.max(axis=1)
        
        # Update
        # self.df[column] = TR.rolling(window=N).mean().shift(1)
        self.df[column] = TR.ewm(span=N).mean().shift(1)
        
        
        
    def _get_HL(self, column='High20', N=20, HL='High'):
        '''
        get High / Low
        Args:
            - column : column name for DF
            - N      : # days for calculation
            - HL     : 'High' or 'Low'
        '''
        # High
        if HL == 'High':
            HL_N = self.df[HL].rolling(window=N).max().shift(1)
            
        # Low
        elif HL == 'Low':
            HL_N = self.df[HL].rolling(window=N).min().shift(1)
        
        # Update
        self.df[column] = HL_N
    
    


# =============================================================================
# !!!PORTFOLIO CLASS
# =============================================================================
class portfolio:
    def __init__(self):

        # Tags for portfolio stocks
        self._tag = []
        
        # Stock allocation ratio
        self._aloc = {}
        
        # Stock historical data
        self._df = {}
        
        # Ticks for portfolio stocks
        self._tick = {}
        
        
        
    def add_stock(self, stock_cls, tag=[], rmin=0, amax=1):
        '''
        Add stock for portfolio with dynamic allocation setting
        '''
        # Tag for stock
        if not tag:
            tag = stock_cls.tick
        self._tag.append(tag)
        
        # Set stock class
        setattr(self, tag, stock_cls)
        
        # Set allocation in stock class
        # setattr(stock_cls, 'aloc', aloc)
    
        # Set allocation threshold
        self._set_aloc_thresold(tag, rmin, amax)
        
        

    def _set_aloc_thresold(self, tag, rmin=0, amax=1):
        '''
        Set allocation thresholds (동적배분율 임계치 설정)
        각 주식별로 포트폴리오에서 차지하는 최소 요구 / 최대 허용 할당 비를 설정함
        
        Args
            - rmin : required minimum allocation  [최소 요구 할당 비]
                   : 주식 보유비가 rmin에 도달하면 매도 불가능 (default = 0)
                   
            - amax : allowable maximum allocation [최대 허용 할당 비]
                   : 주식 보유비가 amax에 도달하면 매수 불가능 (default = 1)
                   
        * 초기 설정값 의미
          : n개의 주식이 포트폴리오에 있다면 단일 종목을 100% 까지 보유 가능
          
        * 제약조건
          : n개의 주식 -> 'sum of rmin <= 1' 만족해야 함.
            
        * 사용예시   
            ex1) amin=0, amax=0.5
                 -> 매수: 50% 까지 가능 / 매도: 전량 가능
                 
            ex2) amin=0.2, amax=0.8
                 -> 매수: 최대 80% 까지 가능
                 -> 매도: 20% 초과분부터 가능 (0 -> 20% 되기까지 매도 불가능)
        '''
        # Set allocation in stock class
        setattr(self.__dict__[tag], 'aloc', {'rmin': rmin, 'amax': amax})
        
        
        
    # def _set_dynamic_aloc(self, tag, amin=0, amax=1):
    #     '''
    #     Set dynamic allocations (동적 배분율 설정)
        
    #     Args
    #         - amin : minimum allocation (default = 0)
    #         - amax : maximum allocation (default = 1)
            
    #         ex) amin=0.5, amax=0.5 -> 고정 할당(0.5) - 항
    #             amin=0.5, amax=1   -> 최소 할당 0.5로 제한
    #             amin=0,   amax=0.5 -> 최대 할당 0.5로 제한
                
    #     -> n개의 주식 : 'sum of amin <= 1' 만족해야 함
    #     -> 티폴트 셋팅 의미 : n개의 주식이 포트폴리오에 있다면 상황에 따라 1종목을 전체매수 할 수 있음
    #     '''
    #     # Set allocation in stock class
    #     setattr(self.__dict__[tag], 'aloc', [amin, amax])
        
        
        
    # def _update_aloc(self):
    #     '''
    #     Update allocations based on trading strategy & [amin, amax]
    #     '''
        
        
        
        
    @property
    def tag(self) -> list:
        '''
        Get lisf of tags
        '''
        return self._tag
    
    
        
    @property
    def aloc(self) -> dict:
        '''
        Get allocation dict. {tag:aloc}
        '''
        # Get dict. for allocation
        for tag in self.tag:
            # Get allocation for stock
            self._aloc[tag] = getattr(self, tag).aloc
        return self._aloc
        
    

    @property
    def df(self) -> dict:
        '''
        Get stock historical data dict.
        '''
        # Get dict. for stock data
        for tag in self.tag:
            # Get historical data
            self._df[tag] = getattr(self, tag).df
        return self._df
        
    
    
    @property
    def tick(self) -> dict:
        '''
        Get ticks for portfolio
        '''
        for tag in self.tag:
            # Get data
            self._tick[tag] = getattr(self, tag).tick
        return self._tick
    



