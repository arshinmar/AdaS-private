import pandas as pd

def get_ranks(path = '', epoch_number = -1):
    '''
    - Read from .adas-output excel file
    - Get Final epoch ranks
    OR - get max output rank for each layer
    '''
    #sheet = pd.read_excel(GLOBALS.EXCEL_PATH,index_col=0)
    sheet = pd.read_excel(path,index_col=0)
    out_rank_col = [col for col in sheet if col.startswith('out_rank')]
    in_rank_col = [col for col in sheet if col.startswith('in_rank')]

    out_ranks = sheet[out_rank_col]
    in_ranks = sheet[in_rank_col]

    last_rank_col_out = out_ranks.iloc[:,epoch_number]
    last_rank_col_in = in_ranks.iloc[:,epoch_number]

    last_rank_col_in = last_rank_col_in.tolist()
    last_rank_col_out = last_rank_col_out.tolist()

    return last_rank_col_in, last_rank_col_out


if __name__ == '__main__':
    a,b = get_ranks(path='AdaS_adapt_trial=5_net=AdaptiveNet_0.1_dataset=CIFAR10.xlsx',epoch_number=3)
    print(a,'LAST RANK COLUMN IN')
    print(b,'LAST RANK COLUMN OUT')
