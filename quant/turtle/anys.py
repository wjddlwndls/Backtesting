########################## ANALYSIS MODULE ####################################
import pandas as pd
# import datetime as dt
import numpy as np
# from pathlib import Path
# import yfinance as yf
import quant.turtle.subr as subr
import quant.turtle.trad as trad
# import quant.turtle.balance as bal
# import matplotlib.pyplot as plt




    
    
    
# =============================================================================        
# !!!BACKTESTING CLASS
# =============================================================================
### DECORATORS ################################################################
# STOCK TAG LOOP
def loop_over_stock_tag(func):
    def wrapper(*args, **kwargs):
        # Loop over stock tags
        for stock_tag in args[0]._pf.tag:
            
            # Set stock tag to current 'self' object
            args[0].stag = stock_tag
            
            # Run inner function
            func(*args, **kwargs)
    return wrapper


# ALL INDIVIDUAL STRATEGY LOOP
def loop_over_idv_strg_tag(func):
    def wrapper(*args, **kwargs):
        # Loop over stock tags
        for stock_tag in args[0]._pf.tag:
            
            # Set stock tag to current 'self' object
            args[0].stag = stock_tag
            
            # Loop over all individual strategies
            for tag in args[0].iST.get_idv_strg_tag():
                
                # Set strategy tag to current 'self' object
                args[0].tag = tag
                
                # Run inner function
                func(*args, **kwargs)
    return wrapper
###############################################################################    




### MAIN CLASS ################################################################
class backtesting:
    '''
    class for backtesting
    '''
    def __init__(self, ac_cls, st_cls, BOD, EOD):

        # Duplicate data
        self._pf = st_cls._pf   # Portfolio
        self._ac = ac_cls       # Account
        self._st = st_cls       # Stratefy
        
        # Trading date info
        self.trday_info = trad.trday_info(self._pf, BOD, EOD)
        
        # Init. trading history DFs
        self._init_tr_history_DFs()
        
        # Variables for property
        self._tag  = {}
        self._tidv = {}

        
        
    
    # INIT. TRADING HISTORY DFs
    def _init_tr_history_DFs(self):
        '''
        Initialize trading history DFs
            [거래가 기준] - 개별주식
            log  : Trading log (매수/매도 거래금액에 대한 손익 계산)
            bal  : Account balance after trading (거래 후 잔고에 대한 계산)
            
            [종가 기준] - 개별주식 / 전체 포트폴리오
            res  : Daily backtesting results based-on Close-price (종가기준 백테스팅 결과)
        '''
        # Column names
        cols = {}
        cols['log'] = [
                      'cycl',     # 전략 수행 싸이클
                      'strg',     # 전략 tag
                      'shar',     # 거래 주식 수
                      'pric',     # 거래가
                      'Pavg',     # 매입단가
                      'amnt',     # 거래금액
                      'PanL',     # 거래손익 (매도만 해당)
                      'RoR' ,     # 거래수익률 (매도만 해당)
                      ]
        cols['bal'] = [
                      'cycl',     # 전략 수행 싸이클
                      'strg',     # 실행된 전략 tag
                      'shar',     # 주식 잔고
                      'pric',     # (전략 실행 시)거래 가
                      'Pavg',     # 잔고 매입단가(평단가)
                      'amnt',     # 잔고 매입금
                      'valu',     # 잔고 평가금
                      'PanL',     # 잔고 평가손익
                      'RoR' ,     # 잔고 수익률 (pirc 기준)
                      'cash',     # 계좌 예수금
                      'aloc',     # 잔고 매입 비율
                      ]
        cols['res'] = [
                      'cycl',     # 전략 수행 싸이클
                      'strg',     # 실행된 전략 tag
                      'TRoA',     # 투자 금액 비율(Trading Ratio Of Amount), 최대 허용 할당 비(amax) 기준
                      'Pavg',     # 잔고 매입단가(평단가)
                      'amnt',     # 잔고 매입금
                      'valu',     # 잔고 평가금
                      'PanL',     # 잔고 평가손익
                      'RoR' ,     # 잔고 수익률 (매입금 기준)
                      ]
        col_pf      = [
                      'tnum',     # 거래 자산 수
                      'TRoA',     # 투자 금액 비율(Trading Ratio Of Amount), 최대 허용 할당 비(amax) 기준
                      'amnt',     # 계좌 총 매입금
                      'valu',     # 계좌 총 평가금
                      'PanL',     # 계좌 총 평가손익
                      'RoR' ,     # 계좌 총 수익률 (매입금 기준)
                      'cash',     # 계좌 예수금 (단일 자산의 경우 vala 계산을 위한 베이스 값. PF 의 경우 실제 예수금)
                      'vala',     # 계좌 평가금 (valu + cash)
                      'RoRa',     # 계좌 수익률 (전일대비 - 매입기준이 아님. cumul. 계산하기 위함)
                      ]
                
        # Generate DF for idv. stocks
        log = {}
        bal = {}
        res = {}
        for stag in self._pf.tag:
            # df
            idf_log = pd.DataFrame(columns = cols['log'])
            idf_bal = pd.DataFrame(columns = cols['bal'])
            idf_res = pd.DataFrame(columns = cols['res'])
            # DF dict.
            log[stag] = idf_log
            bal[stag] = idf_bal
            res[stag] = idf_res
            
        # Generate DF for portfolio
        idf_pf = pd.DataFrame(columns = col_pf)
        res['PF'] = idf_pf 
            
        # Update class
        setattr(self, 'log', log)
        setattr(self, 'bal', bal)
        setattr(self, 'res', res)
        
        
               
                 
    '''
    !!! Property
    '''
    # ACTIVE STOCK TAG
    @property
    def stag(self):
        return self._stag
    @stag.setter
    def stag(self, stag):
        self._stag = stag


    # ACTIVE STRATEGY TAG
    @property
    def tag(self):
        return self._tag[self.stag]
    @tag.setter
    def tag(self, tag):
        self._tag[self.stag] = tag

    
    # INDIVIDUAL STRATEGY FOR TRADING
    @property
    def tidv(self) -> dict:
        # Get individual strategy for trading
        return self._tidv
    @tidv.setter
    def tidv(self, data):
        # Set trading idv. strg. list for active stock tag
        self._tidv[self.stag] = data
    
    
    # SUB-CLASS
    @property
    def iPF(self):
        # Portfolio class for current tag
        return self._pf.__dict__[self.stag]
    @property
    def iST(self):
        # Strategy class for current tag
        return self._st.__dict__[self.stag]
    @property
    def iBL(self):
        # Balance class for current tag
        return self._ac.__dict__[self.stag]



    
    '''
    !!! Main code
    '''
    def run(self):
        '''
        Flowchart
        
        [거래 수행]
        Loop over trading date
          - Get strategy for trading (return: true 상태 개별신호 리스트)
          - Loop over strategies
            * Update balance
            * Update account status
          - Update result DF (based on Close price)
          
        [로그 저장] - 데코레이터 활용?
        
        [후 처리 자료 저장]
        '''
        # LOOP OVER TRADING DAYS: RUN BACKTEST
        for index in self.trday_info.days:

            # INIT. STRATEGIES
            self.init_strg_res()
            
            
            # CHECK STRATEGY & UPDATE ACCOUNT
            self.update_account_with_strg_filtering(index)
                        
            
            # UPDATE CLOSE-PRICE BACKTESTING RESULTS
            self.update_backtesting_res(index)
            
            
            print(f'updating {index} ...')
            
            
        # SAVE TRADING LOG
                
    
    

    '''
    !!! METHODS FOR MAIN CODE
    '''
    # INITIALIZE STRATEGY SIGNAL RESULTS (SET FALSE)
    @loop_over_stock_tag
    def init_strg_res(self):
        # Init. trading day
        # Set false to all strategies
        self.iST.set_res_false()
        # Set zero dcnt
        self.iST.set_dcnt_zero()
        # Set empty trading tag
        self.tidv = []
        
        # Init. trading cycle
        '''
        rmin 존재할 경우 shar==0 만족안됨. 이 경우 싸이클을 어떻게 정의할 지 고민해야 함.
        '''
        if (self.iBL.shar == 0) & (self.iBL.Pavg != 0):
            # Set zero Pavg (for zero # shares)
            self.iBL.Pavg = 0
            
            # Set zero ecnt
            self.iST.set_ecnt_zero()
    
    
    
    
    # CHECK STRATEGY AND UPDATE ACCOUNT
    @loop_over_idv_strg_tag
    def update_account_with_strg_filtering(self, index):
        '''
        check strg. & update account
        '''
        # Repeat strategy : 전략 반복(retr=0)을 디폴트로 설정함
        flag = True
        while flag:
        
            # Init. strategy signal results (set false)
            self.iST.set_res_false()
            
            # Get sub-class for current strategy
            isub = self.iST.find_sub_by_tag(self.tag)
            
            # Get sgnl & pric for current strategy
            self.get_sgnl_pric(isub, index)
            
            # Check strategy executability
            flag = self.check_strg_executability(isub, index)
            
            # Update account
            if flag:
                # Update
                self.update_account(index)
                
                # Set false to all strategies
                self.iST.set_res_false()
                
                # Save trading tags
                self.tidv[self.stag] = self.tidv[self.stag] + [self.tag]
        
        
        
        
    def check_strg_executability(self, isub, index):
        '''
        Check executability for current strategy (check idv/set options)
        Args
            - isub  : sub-class for current idv. strg.
            - index : date index
        Return
            - flag  : True/False flag for account updating
        '''
        # Check strategy options
        self.iST.update_strg_result_by_options()


        # Display strategy (for debugging)
        # self.iST.display()
        # print(self.iST.find_all_super_tags(self.tag))
        
        
        # Get result signal applying strategy options
        # [tstg].strg=True 이면 해당 전략은 유효.
        flag = self.iST.find_sub_by_tag('tstg').strg
        
        # Check cash limit for buy [예수금 한도 확인]
        # 최대 허용 할당비 고려하여 수정
        if isub.arg['stus'] == 'buy':
            flag &= self._ac.fcsh[self.stag] > 0
            
            # # 이평선 정렬 확인(상승장 확인)
            # # Face 1 or 2 or 6
            # tmp = ((self.iPF.df.loc[index,'fac'] == 4)
            #      + (self.iPF.df.loc[index,'fac'] == 5)
            #      + (self.iPF.df.loc[index,'fac'] == 6))
            
            # tmp = tmp * self.iPF.df.loc[index,'dMACD_TOP'] > 0
            # tmp = tmp * self.iPF.df.loc[index,'dMACD_MID'] > 0
            # tmp = tmp * self.iPF.df.loc[index,'dMACD_BOT'] > 0
            # flag = flag * tmp
            
        # Check shares limit for sel [보유 수량 한도 확인]
        # 최소 요구 할당비 고려하여 수정
        if isub.arg['stus'] == 'sel':
            flag &= self._ac.famt[self.stag] > 0
        return flag
        
    
    

    # GET STRATEGY SGNL & PRIC        
    def get_sgnl_pric(self, isub, index):
        '''
        Get sgnl & pric in isub (strategy tree class)
        '''
        # Init. price - array type: [Low, High] 
        # 전략 반복실행 및 external stag 접근에 대비하여 pric 초기화
        for stag in self._pf.tag:
            L = self._pf.df[stag].loc[index, 'Low']
            H = self._pf.df[stag].loc[index, 'High']
            pric = np.linspace(L,H,2)
            self._ac.__dict__[stag].set_pric(pric)
        
        
        ### sgnl ##############################################################
        # Decomposition of sgnl (LHS, RHS, operator)
        lhs, rhs, oper = subr.split_expression(isub.arg['sgnl'])

        # Objective function
        obj = lhs + '-' + '(' + rhs + ')'
        
        # Change to regular form
        if oper == '>':
            obj = '-(' + obj + ')'
            oper = '<'
            
        # Modify expression & eval obj.
        obj_expr, stag_eval =  subr.expr_regen(self, self.stag, obj, isub.arg['kwargs'], index)
        f = eval(obj_expr)

        # Check condition
        fval = f < 0
        if fval.any():
            isub.strg = True
        #######################################################################    
            
        
        ## cond ###############################################################
        # Check input
        if isub.arg['cond']:
            
            # Loop over conditions
            for icnd in range(0,len(isub.arg['cond'])):
                
                # Modify expression
                expr_cond = subr.expr_regen_cond(self, self.stag, self.tag, isub.arg['cond'][icnd], index)
                
                # Get result
                g = eval(expr_cond)
                
                # Update signal
                isub.strg *= g
        #######################################################################
        
        
        ### pric ##############################################################
        # Check pric='pric' --> Root findindg
        if isub.arg['pric'].find('pric') >= 0:
            
            # f 바운드 찾기: low/high 무관한 값이면 단일 값. pric 관련되면 자동으로 array 로 출력될것
            f_L = f[0]
            f_U = f[-1]
            
            # f 'oper' 0 --> f > 0 만족여부 확인
            # 목적함수 부호 스위칭 확인 (부등식 방향 무관하게 +/- 값이 나와야 해 존재)
            if f_L * f_U <= 0:
                # Root finding by bisection method
                pres = self.solve_pric_bisec(stag_eval, obj, isub, index)
                # Rounding
                pres = np.round(pres,4)
                
            # 해가 존재하지 않는 경우
            elif f_L * f_U > 0:
                pres = 0
                
                # 갭 상승/하락 확인 (시가 거래)
                if (f_L < 0) & (f_U < 0):
                    pres = self.iPF.df.loc[index, 'Open']
                        
                        
        # For other cases --> eval. expression
        elif isub.arg['pric'].find('pric') < 0:
            pres_expr,_ = subr.expr_regen(self, self.stag, isub.arg['pric'], isub.arg['kwargs'], index)
            pres = eval(pres_expr)
        
        # Update pres
        isub.res['pric'] = pres
        #######################################################################


       

    # ROOT FINDING BY BISECTION METHOD
    # Newton-raphson 으로 바꿀 것 (FDM이용해도 1st order 이면 exact)
    def solve_pric_bisec(self, stag_eval, obj, isub, index, tol=1.0e-6):
        '''
        root finding by bisection method
        전략 조건(sgnl)에 pric가 없다면, pric와 직접적인 관련된 변수들만 유효(valu, PanL, RoR)
        '''
        # Set default for stock tag (default=current stag)
        if not stag_eval:
            stag_eval = self.stag
                
        # Function evaluation sub-routine
        def fx(xval):
            self._ac.__dict__[stag_eval].set_pric(xval)
            res = eval(subr.expr_regen(self, self.stag, obj, isub.arg['kwargs'], index)[0]) 
            return res
        
        # Bounds
        lb = self._ac.__dict__[stag_eval].pric[0]
        ub = self._ac.__dict__[stag_eval].pric[-1]
        
        # Get mid point
        mid = 0.5 * (lb + ub)
        
        # Check convergence
        mid_change = 1
        mid_old = mid
        # while abs(lb - ub) > tol:
        while mid_change > tol:
            
            # Check root
            if fx(mid) == 0:
                break
            
            # Update lower bound
            if np.sign(fx(mid)) == np.sign(fx(lb)):
                lb = mid
                
            # Update upper bound
            if np.sign(fx(mid)) == np.sign(fx(ub)):
                ub = mid
            
            # Update mid
            mid_old = mid
            mid = 0.5 * (lb + ub)
            
            # Update tol.
            mid_change = abs(mid - mid_old) / mid_change
        return mid
        
       
        
       
    # UPDATE ACCOUNT BALANCE
    def update_account(self, index):
        '''
        Update account balance for active strategy [active stag, tag]
       
        args
            - index : Date index
        '''
        # # Trading ratio
        # ratio = self._st.get_tree(self.stag, self.tag, 'arg')['amnt']
        
        # # Trading price
        # price = self._st.get_tree(self.stag, self.tag, 'res')['pric']
        
        # # Execute trading (using account class)
        # attr_name = self._st.get_tree(self.stag, self.tag, 'arg')['stus'] + '_amnt'
        # getattr(self._ac, attr_name)(self.stag, price, ratio)
        
        # # Update execution count (ecnt, dcnt) for all super-class
        # for tag in self.iST.find_all_super_tags(self.tag)[1:]:
        #     self.iST.update_ecnt_dcnt_by_tag(tag)
        
        # # Write trading logs & balance
        # self.update_trading_log_and_bal(price, index)
        
        
        # Trading ratio
        # amnt_expr, _ = subr.expr_regen(self, self.stag, self._st.get_tree(self.stag, self.tag, 'arg')['amnt'], {}, index)
        ratio = eval(self._st.get_tree(self.stag, self.tag, 'arg')['amnt'])
        # ratio = amnt / (self._ac.amnt + self._ac.cash)
        
        # Trading price
        price = self._st.get_tree(self.stag, self.tag, 'res')['pric']
        
        # Execute trading (using account class)
        attr_name = self._st.get_tree(self.stag, self.tag, 'arg')['stus'] + '_amnt'
        getattr(self._ac, attr_name)(self.stag, price, ratio)
        
        # Update execution count (ecnt, dcnt) for all super-class
        for tag in self.iST.find_all_super_tags(self.tag)[1:]:
            self.iST.update_ecnt_dcnt_by_tag(tag)
        
        # Write trading logs & balance
        self.update_trading_log_and_bal(price, index)
        

        
        
        
        
    # UPDATE TRADING LOG & BALANCE
    def update_trading_log_and_bal(self, price, index):
        '''
        Update trading log & bal DF
        '''
        ### LOG
        # Get recent trading log
        log = {**{'strg': self.tag}, **self._ac.tx[self.stag]}
        
        # Update DF
        self.log[self.stag] = subr.init_new_DF_row(self.log[self.stag], index)
        self.log[self.stag].iloc[-1,1:] = log
        

        ### BALANCE
        # Set base price for evaluation
        self.iBL.set_pric(price)
        
        # Get trading balance
        bal = {
              'cycl' : 0,
              'strg' : self.tag,
              'shar' : self.iBL.shar,
              'pric' : price,
              'Pavg' : self.iBL.Pavg,
              'amnt' : self.iBL.amnt,
              'valu' : self.iBL.valu,
              'PanL' : self.iBL.PanL,
              'RoR'  : self.iBL.RoR,
              'cash' : self._ac.cash,
              'aloc' : self._ac.aloc[self.stag],
              }        
        
        # Update DF
        self.bal[self.stag] = subr.init_new_DF_row(self.bal[self.stag], index)
        self.bal[self.stag].iloc[-1,1:] = bal                
               
        # [cycl]
        # Trading continue (default)
        BAL = self.bal[self.stag]
        BAL.iloc[-1,0] = BAL.shift(1).fillna(0).iloc[-1,0]
        
        # Check trading start / end (check 'shar')
        if ((BAL.shift(1).iloc[-1,2] == 0) |
            (BAL.shift(1).iloc[-1,2] == None)):
            BAL.iloc[-1,0] = max(BAL['cycl']) + 1
        
        # Update for log
        self.log[self.stag].iloc[-1,0] = BAL.iloc[-1,0]
            
                
    # UPDATE BACKTESTING RESULTS
    def update_backtesting_res(self, index):
        '''
        Update close-price backtesting results
        '''
        self.update_bt_res_idv(index)
        
        self.update_bt_res_pf(index)
        

        
        # # Set close-price for evaluation
        # for stag in self._pf.tag:
        #     self._ac.__dict__[stag].set_pric(self._pf.df[stag].loc[index, 'Close'])
        
        # # Get backtesting results for portfolio
        # res = {
        #       'tnum' : 0,
        #       'TRoA' : self._ac.amnt / (self._ac.amnt + self._ac.cash),
        #       'amnt' : self._ac.amnt,
        #       'valu' : self._ac.valu,
        #       'PanL' : self._ac.PanL,
        #       'RoR'  : self._ac.RoR,
        #       'cash' : self._ac.cash,
        #       'vala' : self._ac.valu + self._ac.cash,
        #       'RoRa' : 0,
        #       }
        
        # # Update DF (Partial Update)
        # self.res = subr.init_new_DF_row(self.res, index)
        # self.res.iloc[-1,:] = res
        # RES = self.res
        
        # # [tnum]
        # for stag in self._pf.tag:
        #     RES.loc[index,'tnum'] += min(self._ac.__dict__[stag].shar, 1)
        
        # # [RoRa] for compute 'cuml'
        # if not RES.shift(1).loc[index,'vala']:
        #     # Initial RoRa [vala / initial amount (cash + amnt)]
        #     den = RES.loc[index,'amnt'] + RES.loc[index,'cash']
        # else:
        #     # 2nd to end RoRa [ vala(current) / vala(past) ]
        #     den = RES.shift(1).loc[index,'vala']
        # RES.loc[index, 'RoRa'] = (RES.loc[index,'vala'] / den) - 1
            
        
    
    
    # UPDATE BACKTESTING RESULTS FOR INDIVIDUAL STOCKS
    @loop_over_stock_tag
    def update_bt_res_idv(self, index):
        '''
        Update close-price backtesting results - idv. stocks
        '''
        # Set close-price for evaluation
        self.iBL.set_pric(self.iPF.df.loc[index, 'Close'])
        
        # Get backtesting results
        res = {
              'cycl' : 0,
              'tnum' : 0,
              'strg' : ','.join(self.tidv[self.stag]),
              'TRoA' : self._ac.aloc[self.stag],
              'Pavg' : self.iBL.Pavg,
              'amnt' : self.iBL.amnt,
              'valu' : self.iBL.valu,
              'PanL' : self.iBL.PanL,
              'RoR'  : self.iBL.RoR,
              }
        
        
        
        # Update DF (Partial update)
        self.res[self.stag] = subr.init_new_DF_row(self.res[self.stag], index)
        self.res[self.stag].iloc[-1,:] = res        
        RES = self.res[self.stag]

        # [cycl]
        # Trading continue (default)
        RES.loc[index, 'cycl'] = RES.shift(1).fillna(0).loc[index, 'cycl']
        # RES.loc[index, 'cycl'] = self.log[self.stag].iloc[-1,0]
        
        # Check trading start / end
        if ((RES.shift(1).loc[index, 'amnt'] == 0) |
            (RES.shift(1).loc[index, 'amnt'] == None)):
            # Trading start
            if res['amnt'] > 0:
                RES.loc[index, 'cycl'] = max(RES['cycl']) + 1
            
            # Trading end
            if res['Pavg'] == 0:
                RES.loc[index, 'cycl'] = 0
        
        
        
        
    # UPDATE BACKTESTING RESULTS FOR PORTFOLIO
    def update_bt_res_pf(self, index):
        '''
        Update close-price backtesting results - portfolio
        '''
        # Get backtesting results for portfolio
        res = {
              'tnum' : 0,
              'TRoA' : self._ac.amnt / (self._ac.amnt + self._ac.cash),
              'amnt' : self._ac.amnt,
              'valu' : self._ac.valu,
              'PanL' : self._ac.PanL,
              'RoR'  : self._ac.RoR,
              'cash' : self._ac.cash,
              'vala' : self._ac.valu + self._ac.cash,
              'RoRa' : 0,
              }
        
        # Update DF (Partial Update)
        self.res['PF'] = subr.init_new_DF_row(self.res['PF'], index)
        self.res['PF'].iloc[-1,:] = res
        RES = self.res['PF']
        
        # [tnum]
        for stag in self._pf.tag:
            RES.loc[index,'tnum'] += min(self.res[stag].loc[index,'cycl'], 1)
        
        # [RoRa] for compute 'cuml'
        if not RES.shift(1).loc[index,'vala']:
            # Initial RoRa [vala / initial amount (cash + amnt)]
            den = RES.loc[index,'amnt'] + RES.loc[index,'cash']
        else:
            # 2nd to end RoRa [ vala(current) / vala(past) ]
            den = RES.shift(1).loc[index,'vala']
        RES.loc[index, 'RoRa'] = (RES.loc[index,'vala'] / den) - 1
            
        
            
            
            
            
            
            
            
