
import os
import logging
#######################  网络参数设置　　############################################################
channel = 20
unit = 6
ssim_loss = True
########################################################################################
aug_data = False # Set as False for fair comparison

patch_size = 224
pic_is_pair = True  #input picture is pair or single

lr = 0.0005

data_dir = '/data/wangcong/dataset/rain100H'
if pic_is_pair is False:
    data_dir = '/data/wangcong/dataset/real-world-images'
log_dir = '../logdir'
show_dir = '../showdir'
model_dir = '../models'
show_dir_feature = '../showdir_feature'

log_level = 'info'
model_path = os.path.join(model_dir, 'latest_net')
save_steps = 400

num_workers = 8

num_GPU = 2

device_id = '3,2'

real_epoch = 20
updating_epoch = 1
value_cl = 0.00008
training_real_dir = '../training_real_imgs'
syn_data_dir = '../syn'
syn_training_data_dir = '/data/wangcong/dataset/rain100H/train'
syn_test_data_dir = '/data/wangcong/dataset/rain100H/test'
real_data_dir = '../real-world'

batch_size = 1
total_step_initial = int((500 * 1800)/12)
epoch = 500 + real_epoch

mat_files_training = os.listdir(syn_training_data_dir)
num_training_datasets = len(mat_files_training)
l1 = int(3/5 * epoch * num_training_datasets / batch_size)
l2 = int(4/5 * epoch * num_training_datasets / batch_size)
l3 = total_step_initial

mat_files_real = os.listdir(real_data_dir)
num_real_datasets = len(mat_files_real)
one_epoch_real = int(num_real_datasets/batch_size)
total_step_real = int((epoch * num_real_datasets)/batch_size)

total_step = total_step_real  + total_step_initial

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


