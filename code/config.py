"""
Configuration file for main scripts
"""

#%%
import os
import matplotlib
import torch
import time
import sys

# set path to project folder 
# (to be adapted to your path inside Docker container where project folder is mounted)
path_project = '/project/med/Hassan_Ghavidel/transformer_target_localization_phantom'

# path to data folder
path_project_data = os.path.join(path_project, 'data')

# path to model building code folder
path_project_code = os.path.join(path_project, 'code')

# path to results folder
path_project_results = os.path.join(path_project, 'results')

# add project auxiliary and models folder to Python path to be able to import self-written modules from anywhere
sys.path.append(os.path.join(path_project_code, 'auxiliary'))
sys.path.append(os.path.join(path_project_code, 'models'))
# import utils

#%%
# SET GENERAL SETTINGS 

print('-------------------------------------------------')
# GPU settings
gpu_iden = 1  # None=CPU, 0,1,...=GPU (explicitly set GPU ID)
if gpu_iden is not None:
    if torch.cuda.is_available():  
        print('GPUs are available!')
        GPU_num = torch.cuda.device_count()
        for GPU_idx in range(GPU_num):
            GPU_name = torch.cuda.get_device_name(GPU_idx)
            print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
        
        device = torch.device(f'cuda:{gpu_iden}') 
        # set device nr to standard GPU
        torch.cuda.set_device(gpu_iden)  
        print('Currently using: ' + torch.cuda.get_device_name(gpu_iden)) 
    else:  
        print('No GPU available! Running on CPU...')
        device = torch.device('cpu') 
else:
    print('Running on CPU!')
    device = torch.device('cpu')
print('--------------------------------------------------')

# plot settings
matplotlib.rcParams.update({'font.size': 22})  # increase fontsize of all plots
plot = True # whether to plot results when running scripts
     
#%%
# SET PARAMETERS FOR SCRIPTS

#start_time_string = time.strftime("%Y-%m-%d-%H:%M:%S")
custom_name = 'female89_100p_LNCC'  # Set this to None if you want to use the current timestamp
start_time_string = custom_name if custom_name is not None else time.strftime("%Y-%m-%d-%H:%M:%S")

# moving image (either or explicit frame name or None to pick the first one)
moving_id = None

# choose dataset
# dataset = '2023_05_10_test'
dataset = 'images'

# other data params
wandb_usage = False
os.environ['WANDB_API_KEY'] = '19df5b248f9a8b81b296738bc6b12c2d49cb9a6c'    # wandb API key for your user
train_val_split = 0.80     # percentage of unsupervised training samples, rest is validation
unsupervised_validation = True     # whether to use validation set without segmentations
supervised_validation = False     # whether to use validation set with segmentations
patient_specific_inference = False       # whether to use model trained on first frames of each patient in main_infer script
inference = 'validation'   # validation, testing

# choose patients for labeled fine-tuning, validation, and testing set
observer_training = ''
patients_training = ['female89_100p']

observer_validation = ''
patients_validation = ['female89_100p']

# observer_testing = 'contoured_ground_truth_VG'  #  contoured_ground_truth_LV, contoured_ground_truth_VG
# patients_testing = ['liver_patient0006', 'liver_patient0027', 'abdomen_patient0008', 'abdomen_patient0017', \
#                    'prostate_patient0006', 'prostate_patient0013', 'lung_patient0041', 'lung_patient0046', \
#                    'pancreas_patient0002', 'pancreas_patient0011', \
#                    'heart_patient0001', 'heart_patient0002', 'mediastinum_patient0002']


# choose model for main script
model_name = 'TransMorph2D'  # TransMorph2D, BSpline, NoReg, InterObserver

if model_name == 'TransMorph2D':
    model_variant = 'TransMorph'    # see dict_model_variants in models.TranMorph
    load_state = False      # continue training with current best model/optimizer for epoch_nr epochs or load model for inference
    if load_state: 
        # specify model 
        start_time_string = '2023-09-14-07:20:12'   # unsup --> this one has uploaded weights!

    batch_size = 2     # 2, 16, 64, 128, 192
    lr = 0.00001    # learning rate
    lr_scheduler = None   # None, WarmupCosine
    epoch_nr = 1000    # nr of training epochs   30, 50, 100, 150, 300, 500
    loss_name = 'LNCC'   # MSE, SSIM, LNCC, Dice, MSE-Diffusion, MSE-Bending, LNCC-Diffusion, LNCC-Bending, Dice-Diffusion, Dice-Bending, MSE-Dice-Diffusion, MSE-Dice-Bending
    loss_weights = [1, 0.01]   # weights for combined loss,  0th element is for image and 1st for displacement part
    # loss_weights = [0.05, 1, 0.005]   # weights for combined loss,  0th element is for image and 1st for segmentation and 2nd for displacement part
    prob_randaffine = 0.75
    prob_randelasticdef = 1.0  # must be 1.0
    prob_randmotion = 0.75
    prob_randbiasfield = 0.75
    prob_randgibbsnoise = 0.75
    prob_randgaussiansmooth = 0.75
    len_ps_train = 8         # nr of frames used for patient specific training
    
elif model_name == 'BSpline':
    device = torch.device('cpu')  # always run on CPU
    print('Running on CPU!')
    lambda_value = '5.0'  # regularization parameter
    init_grid_spac = '90'  # grid spacing in mm of first B-Spline stage
    metric = 'mse'   # mse, mi, gm
    max_its = '500'
    impl = 'plastimatch'  # plastimatch, itk
            
elif model_name == 'NoReg':
    device = torch.device('cpu')  # always run on CPU
    print('Running on CPU!')

elif model_name == 'InterObserver':
    device = torch.device('cpu')  # always run on CPU
    print('Running on CPU!')    
    inference = 'testing' 
    
else:
    raise ValueError('Unknown model_name specified!')  
