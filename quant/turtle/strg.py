########################## STRATEGY MODULE ####################################
# import quant.turtle.anys as anys
# import quant.turtle.subr as subr
from colorama import Fore, Style
# import pandas as pd
import numpy as np
# from rich import print

# import quant.turtle.IB as IB

# =============================================================================        
#!!! STRATEGY CLASS
# =============================================================================
class strategy:
    def __init__(self, pf_cls):
        
        # DUPLICATE DATA
        self._pf = pf_cls
        
        # PROPERTY
        # self._stag = []         # Tag for stock to set strategy
        
        # INIT. STRATEGY TREE CLASS
        # self.tree = {tag: strg_tree(tag=f'Trading_strategy[{tag}]', styp='set') for tag in pf_cls.tag}
        for tag in pf_cls.tag:
            setattr(self, tag, strg_tree(tag=f'Trading_strategy[{tag}]', styp='set'))
        
        
    @property
    def stock_tag(self):
        # Get activated stock tag
        return self._stock_tag
    # Set stock tag for strategy
    def set_stock_tag(self, tag):
        self._stock_tag = tag



    # LIMIT BUY
    def buy_lim(self, tag, sgnl, cond=None, pric='pric', amnt=1, lamt=1, repc=1, repd=1, **kwargs):
        # Prepare dynamic arguments (excluding 'self')
        args = {k: v for k, v in locals().items() if k != 'self'}  # Exclude 'self'

        # Explicitly add required arguments
        args['stag'] = self.stock_tag
        args['stus'] = 'buy'
        args['otyp'] = 'lim'
        
        # Pass arguments dynamically
        self.set_sig_idv(**args)

        
    # LIMIT SEL
    def sel_lim(self, tag, sgnl, cond=None, pric='pric', amnt=1, lamt=1, repc=1, repd=1, **kwargs):
        # Prepare dynamic arguments (excluding 'self')
        args = {k: v for k, v in locals().items() if k != 'self'}  # Exclude 'self'

        # Explicitly add required arguments
        args['stag'] = self.stock_tag
        args['stus'] = 'sel'
        args['otyp'] = 'lim'
        
        # Pass arguments dynamically
        self.set_sig_idv(**args)
        
        
    # LOC BUY
    def buy_loc(self, tag, bprc, cond=None, amnt=1, lamt=1, repc=1, repd=1, **kwargs):
        '''
        bprc : LOC 주문 기준가
             : 종가 < 기준가 시 종가 매수
             
           ex. bprc = 2*Open --> Close < 2*Open 시 매수
        '''
        # Prepare dynamic arguments (excluding 'self')
        args = {k: v for k, v in locals().items() if k != 'self'}  # Exclude 'self'

        # Explicitly add required arguments
        args['stag'] = self.stock_tag
        args['stus'] = 'buy'
        args['otyp'] = 'loc'
        args['pric'] = 'Close'
        
        # Add buy condition for LOC order
        args['sgnl'] = 'Close < ' + bprc
        
        # Remove bprc key
        del args['bprc']
        
        # Pass arguments dynamically
        self.set_sig_idv(**args)
        
        
    # LOC SEL
    def sel_loc(self, tag, bprc, cond=None, amnt=1, lamt=1, repc=1, repd=1, **kwargs):
        '''
        sgnl : LOC 주문 기준가
             : 종가 > 기준가 시 종가 매도
             
           ex. bprc = Open --> Close > Open 시 매도
        '''
        # Prepare dynamic arguments (excluding 'self')
        args = {k: v for k, v in locals().items() if k != 'self'}  # Exclude 'self'

        # Explicitly add required arguments
        args['stag'] = self.stock_tag
        args['stus'] = 'sel'
        args['otyp'] = 'loc'
        args['pric'] = 'Close'
        
        # Add buy condition for LOC order
        args['sgnl'] = 'Close > ' + bprc
        
        # Remove bprc key
        del args['bprc']
        
        # Pass arguments dynamically
        self.set_sig_idv(**args)
    
    
    
    
    # INDIVIDUAL STRATEGY            
    def set_sig_idv(self, stag, stus, tag, sgnl, cond, pric, otyp, amnt, lamt, repc, repd, **kwargs):
        '''
        Set individual strategy
             stag   : tags for stock
             
             stus   : 'buy' / 'sel'
            
             tag    : strategy tag (buy,add,TP,sl,etc..)
                    : all tags should be unique in each tick list

             sgnl   : mathematical expression for signal (ex. 'High > High20')
                    : 호출가능 변수
                    :  1) pf 클래스 DF의 columns
                    :  2) ac 클래스의 property
                    : 다른 티커 호출 가능. ex) High[ticker1] > High[ticker2]
                    : 수식에서 'pric'의 경우 현재가를 의미. 
                    
             cond   : 논리연산자 (구속)조건.
                    : 호출가능 변수
                    :  1) sgnl 에서 사용된 변수들 (pf, ac 클래스) 
                    :  2) ecnt, dcnt (in strg_tree class) - 전략실행 횟수와 관련된 변수
                    :     ex) ecnt[stock tag, strategy tag] == n
                    :         dcnt[stock tag, strategy tag] == n
                    :         -> 다른티커 전략의 실행횟수 호출 가능
                    :         -> self 호출의 경우 괄호 안 tag 생략 가능
                    : 문자열(1개) / 리스트(n개), 결과는 True/False 단일
                    : 리스트 입력의 경우 모든 조건이 만족해야 True (and 조건만 수행)
                    : (다수의 조건에 대하여 or 연산을 수행하고자 한다면 전략 셋을 만들면 됨.)
                    : cond = None [default] / 조건 입력문 (string or list) 
                        
             pric   : Trading price
                    : pric='pric' [default] -> sgnl=True 일 때의 가격을 의미
                    
             otyp   : order type
                    : lim (limit) - 지정가 
                    : loc (Limit On Close)
            
             amnt   : ratio of amount for trading unit [0,1]
                    : 거래유닛 = (매입금+현금) * amnt
                    :          계좌 총액(매입금+현금)기준 비율(amnt)로 계산됨 
                    : amnt = lamt -> 할당된 현금 전액 매수 의미
                    : amnt > lamt 의 경우 원하는 수량만큼 거래가 안됨(유의). 
                    : amnt = 1 [default - 전액]
                    
             lamt   : limit of amount [default = 1 * amax(maximum allocation)]
                    : lamt = 1 * amax(최대 허용 할당비) [default]
                
             repc   : Number of repeated tradings in a 'cycle'
                    : 1싸이클 당 전략의 반복 실행횟수
                    : repc = 0 - (조건 충족 시) 반복 실행
                    : repc = 1 [default - 1회]
                        
             repd   : Number of repeated tradings in a 'day'
                    : 1거래일 당 전략의 반복 실행횟수
                    : repd = 0 - (조건 충족 시) 반복 실행
                    : repd = 1 [default - 1회]
                   
             **kwargs : variables used in signal
                      : format: A=a, B=b, ...
        '''
        # Check input argument format: cond
        if cond:
            if isinstance(cond, str):
                cond = [cond]  # 리스트 형태로 변환
        
        # Init. tree class
        tree = strg_tree(tag=tag, styp='idv', stus=stus, sgnl=sgnl, cond=cond, pric=pric, otyp=otyp,
                         amnt=amnt, lamt=lamt * self._pf.aloc[stag]['amax'], repc=repc, repd=repd, **kwargs)
        
        # Update to strategy tree
        getattr(self, stag).add_sub(tree)
        
       
       
        
    # STRATEGY SET
    def gen_set(self, tag, tlst, ceky=0, ropt=0):
        '''
        Generate strategy set by 'tag'
        전략은 tlst 순서대로 sgnl을 확인하고 실행함.
            stag  : tags for stock
            
            tag   : tags for strategy set
            
            tlst  : tag list for set.
                  : [list] or string (for single input)
        
            ceky  : conditional excution key for strategy
                  : 조건부 실행 옵션 키
                  :  0 -> 조건없음(기본) : tlst 목록 순차적으로 실행됨
                  : -1 -> True 시 해당 전략 셋 종료 (전략 종료 조건)
                  : +1 -> 앞선전략 True 시 다음 전략 실행 (후 순위 전략 실행조건)
                  
                  : 적용예시
                  : [strg1, strg2, strg3]
                  : ceky= 0 -> strg1->2->3 순서대로 sgnl/price 확인
                  : ceky=-1 -> strg1=False, strg2=True -> 전략 종료 (strg3 확인 x)
                  : ceky= 1 -> strg1,2가 True 이어야 strg3 실행가능
                  
                      
            ropt : return options for logical operations in the tag list
                  : 0 for 'OR'  (return True if at least one element in tlst is True)
                  : 1 for 'AND' (return True if all component in tlst are True)
        '''
        # Get stock tag
        stag = self.stock_tag
        
        # Check input argument: tlst
        if isinstance(tlst, str):
            tlst = [tlst]  # 리스트로 변환

        # Init. tree class
        tree = strg_tree(tag=tag, styp='set', ceky=ceky, ropt=ropt)
        
        # Update to strategy tree
        getattr(self, stag).add_sub(tree)
        
        # Move elements to its node
        getattr(self, stag).move_sub(tag, tlst)




    # GET SUB-CLASS (strg_tree) ARGUMENTS
    def get_tree(self, stock_tag, tag, var_name):
        '''
        Get class variables in strategy tree class
        Args
            - stock_tag : stock tag for evaluation
            - tag       : strategy tag for evaluation
            - var_name  : variable / property name in strg_tree class to get
        '''
        # Strategy tree class
        strg_tree = getattr(self, stock_tag)
        
        # Get sub-class by tag
        sub = strg_tree.find_sub_by_tag(tag)
        
        # Get var / prop
        res = getattr(sub, var_name)
        return res
    
    
    
    
    # SET SUB-CLASS (strg_tree) ARGUMENTS
    def set_tree(self, stock_tag, tag, var_name, val):
        '''
        Set class variables in strategy tree class (only valid for property and 'res')
        Args
            - stock_tag : stock tag for evaluation
            - tag       : strategy tag for evaluation
            - var_name  : variable / property name in strg_tree class to get
            - val       : values to set
        '''
        # Strategy tree class
        strg_tree = getattr(self, stock_tag)
        
        # Get sub-class by tag
        sub = strg_tree.find_sub_by_tag(tag)
        
        # Get var / prop
        setattr(sub, var_name, val)

        






# =============================================================================        
#!!! STRATEGY TREE CLASS [Attribute for strategy class]
# =============================================================================
class strg_tree:
    # def __init__(self, tag, ceky=0, ropt=0, ecnt=0, res=False):
    def __init__(self, tag=[], styp=[], **kwargs):
        
        # Sub-tree location
        self.sub = []
        
        # Tag
        self.tag = tag
        
        # Strategy type
        self.styp = styp    # [idv / set]
        
        # Input argument
        self.arg = kwargs
        
        # Evaluation results
        # ecnt : 1투자 싸이클당 총 실행횟수 -> repc 옵션 확인에 사용
        # dcnt : 1거래일당 ecnt변화(1거래일당 실행횟수) -> ceky / repd 옵션 확인에 사용
        self.res = {
                    'ecnt': 0,      # Execution count [idv] per cycle 
                    'dcnt': 0,      # Change of execution count per day 
                    'sxpr': None,   # Re-generated expression for sgnl [idv]
                    'pxpr': None,   # Re-generated expression for pric [idv]
                    'pric': 0,      # Evaluation results for pric [idv]
                    'strg': False,  # Strategy signal results [idv/set]
                    }
        
        
        
    @property
    def ecnt(self):
        return self.res['ecnt']
    @ecnt.setter
    def ecnt(self, val):
        self.res['ecnt'] = val

    @property
    def dcnt(self):
        return self.res['dcnt']
    @dcnt.setter
    def dcnt(self, val):
        self.res['dcnt'] = val
        
    @property
    def pric(self):
        return self.res['pric']
    @pric.setter
    def pric(self, val):
        self.res['pric'] = val
        
    @property
    def strg(self):
        return self.res['strg']
    @strg.setter
    def strg(self, flag):
        # Set True/False
        self.res['strg'] = flag



    '''
    !!! MISC. METHODS
    '''
    # ADD TREE NODE        
    def add_sub(self, sub_strgy):
        self.sub.append(sub_strgy)
        
                
    # MOVE TARGET TREE NODE TO DESTINATION
    def move_sub(self, dtag, ttag_list):
        # move target tag to destination tag (ttag -> dtag)
        for ttag in ttag_list:
            
            # Che sub tree to move
            for idx, sub in enumerate(self.sub):
                # Check tag
                if sub.tag == ttag:
                    # Sub tree to move
                    tsub = self.sub.pop(idx)
                    
                    # Move to destination sub tree
                    dsub = self.find_sub_by_tag(dtag)
                    dsub.add_sub(tsub)

        
    # PLOT HIERARCHY    
    def display(self, level=0):
        # Indent setting
        indent = " " * (level * 4)  # 4 spaces per level
        indent2 = "-" * ((10-len(self.tag)) + 20 - 4*level)
        # For starting point
        if level == 0:
            if self.sub[0].strg == True:
                # For True
                print(f"{Fore.CYAN}{indent}{self.tag}{Style.RESET_ALL}")
            else:
                # For False
                print(f"{indent}{self.tag}")
                
        # For inside the tree
        else:
            # Individual strategies
            if self.styp == 'idv':
                if self.strg == True:
                    # For True
                    print(f'{Fore.CYAN}{indent}[{self.tag}] {indent2} ecnt={self.ecnt},dcnt={self.dcnt},repc={self.arg["repc"]},repd={self.arg["repd"]}{Style.RESET_ALL}')
                else:
                    # For False
                    print(f'{indent}[{self.tag}] {indent2} ecnt={self.ecnt},dcnt={self.dcnt},repc={self.arg["repc"]},repd={self.arg["repd"]}')

            # Set strategies
            elif self.styp == 'set':
                if self.strg == True:
                    # For True
                    print(f'{Fore.CYAN}{indent}[{self.tag}] {indent2} ecnt={self.ecnt},dcnt={self.dcnt},ropt={self.arg["ropt"]},ceky={self.arg["ceky"]}{Style.RESET_ALL}')
                else:
                    # For False
                    print(f'{indent}[{self.tag}] {indent2} ecnt={self.ecnt},dcnt={self.dcnt},ropt={self.arg["ropt"]},ceky={self.arg["ceky"]}')

        for child in self.sub:
            child.display(level + 1)



    '''
    !!! GET METHODS
    '''
    # GET MAXIMUM DEPTH OF TREE STRUCTURE
    def get_max_depth(self):
        if not self.sub:  # If no children, depth is 1 (current node)
            return 1

        # Recursively find the depth of each child
        child_depths = [subs.get_max_depth() for subs in self.sub]
        
        # Return 1 (current node) + maximum depth among children
        return 1 + max(child_depths)


    # GET INDIVIDUAL STRATEGY SUB-CLASS
    def get_idv_strg_sub(self) -> list:
        res = []
        
        # Check styp
        if self.styp == 'idv':
            res.append(self)
            
        # Recursively check for children
        for sub in self.sub:
            res.extend(sub.get_idv_strg_sub())
        return res
    
    
    # GET ALL INDIVIDUAL STRATEGY TAGS
    def get_idv_strg_tag(self) -> list:
        res = []
        
        # Check tree end point
        if not self.sub:
            res.append(self.tag)
        
        # Go to tree end
        for sub in self.sub:
            res.extend(sub.get_idv_strg_tag())
        return res
    
        
    # GET INDIVIDUAL STRATEGY TAGS FOR TRADING
    def get_idv_strg_tag_for_trading(self) -> list:
        res = []
        
        # Check tree end point
        if not self.sub:
            if self.strg == True:
                res.append(self.tag)
                
        # Go to tree end
        for sub in self.sub:
            res.extend(sub.get_idv_strg_tag_for_trading())

        return res
    



    '''
    !!! FIND METHODS
    '''
    # FIND SUB-CLASS BY TAG IN TREE
    def find_sub_by_tag(self, tag):
        # Find node with the given tag in the tree
        if self.tag == tag:
            return self
        
        # Recursively check tag
        for sub in self.sub:
            sub_found = sub.find_sub_by_tag(tag)
            if sub_found:
                return sub_found
        return None
    
    
    # FIND ALL SUPER CLASS BY TAG (INCLUDING ITSELF)
    def find_all_super_by_tags(self, tag, path=None):
        # Initialize path on the first call
        if path is None:
            path = []

        # If the current node matches the target tag, return the path including this node
        if self.tag == tag:
            return path + [self]

        # Recursively search in child nodes
        for sub in self.sub:
            result = sub.find_all_super_by_tags(tag, path + [self])
            if result:  # If a match is found, return the result
                return result

        return None  # Return None if the target is not found
    
    
    # FIND ALL SUPER TAGS FOR INPUT TAG (INCLUDING ITSELF)
    def find_all_super_tags(self, tag, path=None):
        # Initialize path on the first call
        if path is None:
            path = []

        # If the current node matches the target tag, return the path including this node
        if self.tag == tag:
            return path + [self.tag]

        # Recursively search in child nodes
        for sub in self.sub:
            result = sub.find_all_super_tags(tag, path + [self.tag])
            if result:  # If a match is found, return the result
                return result

        return None  # Return None if the target is not found
        
    
    
    '''
    !!! UPDATE METHODS
    '''
    # UPDATE EXECUTION COUNT
    def update_ecnt_dcnt_by_tag(self, tag):
        '''
        Update 'ecnt' & 'dcnt' of entire tree structure by input strg. tag
        '''
        # Dive into tree ends
        for sub in self.sub:
            sub.update_ecnt_dcnt_by_tag(tag)
        
        # Check input tag
        if self.tag == tag:
            self.ecnt += 1
            self.dcnt += 1
            
    
    # UPDATE STRATEGY RESULTS BY OPTIONS IN SET STRATEGY
    def update_strg_result_by_options(self):
        '''
        전략 클래스의 'res' 항목 업데이트 (실행 조건들 확인)
        '''
        
        # Dive into tree ends
        for sub in self.sub:
            sub.update_strg_result_by_options()
            
            
        # Check repc (cent) - for idv. strg.
        if self.styp == 'idv':
            
            ## Check repd [거래일당 실행횟수 확인]
            flag = True
            if self.arg['repd'] > 0:
                # 실행횟수 n 회 제한 있는 경우
                flag = not (self.dcnt == self.arg['repd'])
                
            ## Check repc [싸이클당 실행횟수 확인]
            if self.arg['repc'] > 0:
                # 실행횟수 n 회 제한 있는 경우
                flag = not (self.ecnt == self.arg['repc'])
                
            # Update result
            self.strg = (self.strg & flag)        
                
            
        # Check ceky - for set strg.
        if (self.styp == 'set') and self.arg:
            
            # Get array for input parameters
            res = np.array([], dtype=bool)
            ecnt = np.array([])
            dcnt = np.array([])
            for sub in self.sub:
                res = np.append(res, sub.strg)
                ecnt = np.append(ecnt, sub.ecnt)
                dcnt = np.append(dcnt, sub.dcnt)
            
            # Shift elements to right
            # 첫 번째 전략은 제약조건이 없으므로 ecnt=1, dcnt=0 으로 간주
            ecnt_sh = np.insert(ecnt[:-1], 0, 1)  
            dcnt_sh = np.insert(dcnt[:-1], 0, 0)
            
            
            # For 'ceky=1'
            if self.arg['ceky'] == 1:
                # Get result signal
                res = res * (ecnt_sh > 0)
            
            # For 'ceky=-1'
            elif self.arg['ceky'] == -1:
                # Get result signal
                res = res * (dcnt_sh == 0)
                
                # Index of first True
                idx = np.where(res)[0]
                if len(idx) > 0:
                    # Set False behind the 'True'
                    res[idx[0]+1:] = False
                
            # Set filtered 'strg'
            for i in range(0,len(self.sub)):
                self.sub[i].strg = res[i]
            
                
            # Check 'ropt' and get final 'strg'
            if self.arg['ropt'] == 0:
                self.strg = any(res)
            elif self.arg['ropt'] == 1:
                self.strg = all(res)
                
        return None
    
    


    '''
    !!! SET METHODS
    '''
    # SET FALSE TO STRATEGY RESULTS
    def set_res_false(self):
        # Dive into tree ends
        for sub in self.sub:
            sub.set_res_false()
        
        # Set false
        self.res['pric'] = 0
        self.res['strg'] = False
        
        
    # SET ZERO EXECUTIOIN COUNT [ecnt]
    def set_ecnt_zero(self):
        # Dive into tree ends
        for sub in self.sub:
            sub.set_ecnt_zero()
            
        # Set ecnt=0
        self.res['ecnt'] = 0


    # SET ZERO EXECUTIOIN COUNT CHANGE [dcnt]
    def set_dcnt_zero(self):
        # Dive into tree ends
        for sub in self.sub:
            sub.set_dcnt_zero()
            
        # Set ecnt=0
        self.res['dcnt'] = 0


    
    
    

    
    
    