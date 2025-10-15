import pandas as pd
import re
# import datetime as dt
# import numpy as np
# import yfinance as yf


def init_new_DF_row(DF, index_name):
    '''
    INIT DF ROWS (NEW ROWS WITH 'index_name')
    '''
    # SERIES WITH ZEROS
    df_series = pd.Series([0]*len(DF.columns), index = DF.columns)
    
    # DATAFRAME WITH ZEROS
    iDF = pd.DataFrame([df_series], index=[index_name])
    
    # APPEND NEW ROW
    DF = pd.concat([DF, iDF])
    return DF
    
    
    
def init_new_DF_col(DF, col_name_list):
    # INIT DF COLUMNS (NEW COLS. WITH 'col_name_list')
    # new_cols = list(set(col_name_list) - set(DF.columns))
    new_cols = col_name_list
    
    # CHECK AND ADD NEW COLS
    if new_cols:
        
        # NEW DATAFRAME WITH ZEROS
        iDF = pd.concat((pd.DataFrame([0]*len(DF.index), columns = [icol]) for icol in new_cols), axis=1)
        iDF.index = DF.index
        
        # ADD NEW COLUMN
        DF = pd.concat([DF, iDF], axis=1)
    return DF




def expr_regen(bt, stag, expr, var, idx):
    '''
    re-generation of input expression suits for DF indexing
    
    Input 
        - bt      : input backtesting class (including DFs)
        - stag    : stock tag
        - expr    : mathematical expression for signal
        - var     : input variables used in 'expr'
        - idx     : Date index
    Out
        - re-generated expression with stag (optional)
    '''
    # Sub-routine : expression split using regular expr.
    def split_expr(expression):
        # Regular expression pattern to capture identifiers with brackets as part of them
        pattern = r'[a-zA-Z_][\w\[\]]*|[\d.]+|[+\-*/<>=!&|()]'
        # [a-zA-Z_][\w\[\]]* : variable names and identifiers
        # [\d.]+ : numbers
        # [+\-*/<>=!&|()] : operators, logical symbols, and parentheses
        
        # Find all matches in the expression
        tokens = re.findall(pattern, expression)
        return tokens
    
    
    # Sub-routine : return tokens without brackets corresponding ac/ib/tb
    def get_tokens_for_regen(bt, tag, tokens_wo_brac):
        
        # Sub-routine : find list of name for class property
        def get_acnt_cls_item(obj):
            return set([attr for attr in dir(obj) if isinstance(getattr(type(obj), attr, None), property)])

        # Find items in stock class
        stok_cls_item = set(bt._pf.__dict__[tag].df.columns.to_list())
        
        # Find items in account class
        acnt_cls_item = get_acnt_cls_item(bt._ac.__dict__[tag])
        
        # Find tokens corresponding each class
        tokens_in_scls = stok_cls_item.intersection(tokens_wo_brac)
        tokens_in_acls = acnt_cls_item.intersection(tokens_wo_brac)
        return tokens_in_scls, tokens_in_acls
    
    
    # Sub-routine : check external tag call for expression evaluation
    def check_external_tag_call(tag, itokn_regn):
        # Default return
        res = tag
        
        # Check external tag call
        if '[' in itokn_regn:
            iparen = itokn_regn.find('[') 
            res = itokn_regn[iparen+1:-1]
        return res
    
    
    # Step 1: Tokenize the expression
    # expr = 'Close * 100 + dN + (Pavg*(Low10[TQQQ2]/Pavg[TQQQ2])+High20) / (100 -1) < -Close + 0.1'
    # expr = 'Open + RoRb[TQQQ2] + tnum > 0'
    # expr = 'RoR[TQ2]-0.1 + High'
    tokn = split_expr(expr)
    
    # Step 2: Create tokens without brackets for comparison
    tokn0 = [re.sub(r'\[.*?\]', '', itok) for itok in tokn]
    
    # Step 3: Identify tokens corresponding DF & variabels input
    tokens_in_scls, tokens_in_acls = get_tokens_for_regen(bt, stag, tokn0)
    tokn_var = set(var.keys())        # for variable input
    
    # Step 4: Change tokens for DF indexing (AC/IB/TB) or variable replacement (var)
    tokn_regn = tokn.copy()
    stag_eval = []      # Stock tag for root finding sub-routine.
    for i in range(0,len(tokn0)):
        
        # Input args. for expression evaluation in class
        item = tokn0[i]
        stock_tag = check_external_tag_call(stag, tokn_regn[i])
            
        # Re-name if item belongs stock class DFs
        if item in tokens_in_scls:
            tokn_regn[i] = f'self._pf.{stock_tag}.df.loc["{idx}","{item}"]'
        
        # Re-name if item belongs account class property
        if item in tokens_in_acls:
            tokn_regn[i] = f'self._ac.{stock_tag}.{item}'
        
        # Re-name if item belongs kwargs
        if item in tokn_var:
            tokn_regn[i] = str(var[item])
        
        # Save stock tag for root finding sub-routine
        if stock_tag != stag:
            stag_eval = stock_tag
    # Step 5: Re-construction tokens
    tokn_regn = ' '.join(tokn_regn)
    
    # Return external stock tag if exist
    return tokn_regn, stag_eval




def split_expression(expr):
    """
    Splits an expression into (left-hand side, right-hand side, operator).

    Supports the following operators: <, >, <=, >=, ==, !=

    Parameters:
        expr (str): The input mathematical or logical expression.

    Returns:
        tuple: (lhs, rhs, operator) where:
                - lhs: Left-hand side of the operator
                - rhs: Right-hand side of the operator
                - operator: The comparison operator
    """
    # Regex pattern to match comparison operators: <, >, <=, >=, ==, !=
    pattern = r'(<=|>=|!=|==|<|>)'
    
    # Search for the first occurrence of a comparison operator
    match = re.search(pattern, expr)
    
    if match:
        # Extract operator
        operator = match.group(0)
        
        # Split the expression at the operator
        lhs = expr[:match.start()].strip()  # Left-hand side
        rhs = expr[match.end():].strip()    # Right-hand side
        
        return lhs, rhs, operator
    else:
        # If no comparison operator is found, return None
        return None, None, None



    

