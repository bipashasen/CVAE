# alpha = 0.5 # values for the previous run.
# beta = 0.1 # values for the previous run
alpha = 0.1
beta = 0.2
gamma = 1

MAX_SEQ_LEN_Q = 35
MAX_SEQ_LEN_A = 65

n_gpu = 4
n_nodes = 1
dl_num_workers = 10
batch_size = 100

task = 'train'

## For this run (after the first update from the author), 
## I have changed the matching function and the loss to mse loss with target as 0.
#store_path = 'models/model_updated.pth'
#store_path = 'models/model-337965.pth'
#store_path = 'models/model-dummy.pth'
store_path = 'models/model-v2.pth'

load_model_from_path = False

mode = 'prod'