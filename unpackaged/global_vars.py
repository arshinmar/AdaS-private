global NET, PERFORMANCE_STATISTICS, OPTIMIZER ,CRITERION, BEST_ACC, METRICS, ADAS, \
    CHECKPOINT_PATH, EARLY_STOP,CONFIG,EXCEL_PATH,THRESHOLD,FULL_TRAIN_MODE,FULL_TRAIN,OUTPUT_PATH, super1_idx,super2_idx,super3_idx,super4_idx,FIRST_INIT,BLOCK_TYPE
NET = None
NET_RAW = None
PERFORMANCE_STATISTICS = None
CRITERION = None
BEST_ACC = 0
METRICS = None
ADAS = None
FIRST_INIT = True
CHECKPOINT_PATH = None
EARLY_STOP = None
CONFIG = None
OPTIMIZER = None
EXCEL_PATH = 'path'
THRESHOLD = 0
FULL_TRAIN = False
FULL_TRAIN_MODE = ''
BLOCK_TYPE = ''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MAHDI~~~~~~~~~~~~~~~~~~~~~~~~
# For DASNet34
'''
super1_idx = [16,16,16,16,16,16,16]
super2_idx = [16,16,16,16,16,16,16,16]
super3_idx = [16,16,16,16,16,16,16,16,16,16,16,16]
super4_idx = [16,16,16,16,16,16]

'''
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ANDRE~~~~~~~~~~~~~~~~~~~~~~~~
super1_idx = [64,64,64,64,64,64,64]
super2_idx = [64,64,64,64,64,64,64,64]
super3_idx = [64,64,64,64,64,64,64,64,64,64,64,64]
super4_idx = [64,64,64,64,64,64]


'''
# For DASNet34
super1_idx = [16,16,16,16,16,16,16]
super2_idx = [16,16,16,16,16,16,16,16]
super3_idx = [16,16,16,16,16,16,16,16,16,16,16,16]
super4_idx = [16,16,16,16,16,16]
'''
'''
# For DASNet50
super1_idx = [16,16,16,16,16,16,16,16,16,16]
super2_idx = [16,16,16,16,16,16,16,16,16,16,16,16]
super3_idx = [16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
super4_idx = [16,16,16,16,16,16,16,16,16]
'''

OUTPUT_PATH = ''
index = None
index_used = super1_idx + super2_idx + super3_idx + super4_idx
