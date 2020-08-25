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
'''
# For DASNet50
super1_idx = [32,32,32,32,32,32,32,32,32,32]
super2_idx = [32,32,32,32,32,32,32,32,32,32,32,32]
super3_idx = [32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32]
super4_idx = [32,32,32,32,32,32,32,32,32]

super1_kernel_idx=[1,3,1,1,3,1,1,3,1]
super2_kernel_idx=[1,3,1,1,3,1,1,3,1,1,3,1]
super3_kernel_idx=[1,3,1,1,3,1,1,3,1,1,3,1,1,3,1,1,3,1]
super4_kernel_idx=[1,3,1,1,3,1,1,3,1]
'''
# For DASNet34
super1_idx = [96,96,96,96,96,96,96]
super2_idx = [96,96,96,96,96,96,96,96]
super3_idx = [96,96,96,96,96,96,96,96,96,96,96,96]
super4_idx = [96,96,96,96,96,96]

super1_kernel_idx=[9,9,9,9,9,9,9]
super2_kernel_idx=[9,9,9,9,9,9,9,9]
super3_kernel_idx=[9,9,9,9,9,9,9,9,9,9,9,9]
super4_kernel_idx=[9,9,9,9,9,9]

OUTPUT_PATH = ''
index = None
index_used = super1_idx + super2_idx + super3_idx + super4_idx
