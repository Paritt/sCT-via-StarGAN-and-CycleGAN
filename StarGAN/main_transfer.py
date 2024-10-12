import os
import argparse
from solver_val import Solver
from data_loader import get_loader, get_loader_class
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.val_result_dir):
        os.makedirs(config.val_result_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    if not os.path.exists(config.report_dir):
        os.makedirs(config.report_dir)
        
    # Data loader.
    image_loader = None
    class_loader = None
    
    image_loader, val_data_loader = get_loader(config.train_dir, config.test_dir, config.validate_dir, config.image_size, 
                                               config.batch_size, config.mode, config.num_workers, config.augment)
    class_loader = get_loader_class(config.train_dir, config.image_size, config.batch_size,
                                 config.mode, config.num_workers)    

    # Solver for training and testing StarGAN.
    solver = Solver(image_loader, class_loader, val_data_loader, config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
       
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=3, help='dimension of domain labels dataset')
    parser.add_argument('--image_size', type=int, default=(512,512), choices=[(256,256), (512,512)], help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--c_conv_dim', type=int, default=64, help='number of conv filters in the first layer of C') 
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--c_repeat_num', type=int, default=6, help='number of strided conv layers in C')     
    parser.add_argument('--lambda_cls', type=float, default=0.25, help='weight for domain classification loss')      # default=0.25
    parser.add_argument('--lambda_rec', type=float, default=1.3, help='weight for reconstruction loss')             # default=1.3    
    parser.add_argument('--lambda_gp', type=float, default=1, help='weight for gradient penalty')                   # default=1
    
    # Training configuration.
    parser.add_argument('--augment', type=bool, default=True, choices=[True, False], help='Augmentation')
    parser.add_argument('--batch_size', type=int, default=4, help='mini-batch size')
    parser.add_argument('--num_epochs', type=int, default=500, help='number of total epochs for training D')
    parser.add_argument('--num_epochs_decay', type=int, default=0, help='number of epochs for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--c_lr', type=float, default=0.00012, help='learning rate for C')      
    parser.add_argument('--n_critic', type=int, default=1, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--c_beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    
    parser.add_argument('--resume_epoch', type=int, default=250, help='resume training from this epoch')
    
    # Test configuration.
    parser.add_argument('--test_epochs', type=int, default=None, help='test model from this step')
    parser.add_argument('--test_MRI', type=bool, default=True, help='test MRI or not')
    parser.add_argument('--test_CBCT', type=bool, default=False, help='test CBCT or not')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    new_name = 'paper_StarGAN'
    parser.add_argument('--train_dir', type=str, default=f'/home/chanon/projects/Boo_Thesis/Data/SuperStarGAN/train_transfer')
    parser.add_argument('--test_dir', type=str, default=f'/home/chanon/projects/Boo_Thesis/Data/SuperStarGAN/test_on_train_MRI')
    parser.add_argument('--validate_dir', type=str, default=f'/home/chanon/projects/Boo_Thesis/Data/SuperStarGAN/validation_transfer')
    parser.add_argument('--validate_case_dir', type=str, default=f'/home/chanon/projects/Boo_Thesis/Data/SuperStarGAN/validation_case_transfer')
    parser.add_argument('--log_dir', type=str, default=f'/home/chanon/projects/Boo_Thesis/My_SuperStarGAN/{new_name}/transfer/logs')
    parser.add_argument('--model_save_dir', type=str, default=f'/home/chanon/projects/Boo_Thesis/My_SuperStarGAN/{new_name}/models')
    parser.add_argument('--sample_dir', type=str, default=f'/home/chanon/projects/Boo_Thesis/My_SuperStarGAN/{new_name}/transfer/samples')
    parser.add_argument('--result_dir', type=str, default=f'/home/chanon/projects/Boo_Thesis/My_SuperStarGAN/{new_name}/transfer/results/test_on_train_MRI')
    parser.add_argument('--val_result_dir', type=str, default=f'/home/chanon/projects/Boo_Thesis/My_SuperStarGAN/{new_name}/transfer/val')
    parser.add_argument('--report_dir', type=str, default=f'/home/chanon/projects/Boo_Thesis/My_SuperStarGAN/{new_name}/transfer/report/test_on_train_MRI')
    
    # Step size.
    parser.add_argument('--log_step', type=int, default=10) #default=10
    parser.add_argument('--sample_step_per_epoch', type=int, default=2) 
    parser.add_argument('--lr_update_step', type=int, default=2) #default=2

    config = parser.parse_args()  
    print(config)
    config_txt_path = f"/home/chanon/projects/Boo_Thesis/My_SuperStarGAN/{new_name}/transfer/config"
    if not os.path.exists(config_txt_path):
        os.makedirs(config_txt_path)
    config_file = os.path.join(config_txt_path, 'config.txt')
    with open(config_file, "w") as file:
        print(config, file=file)
    main(config)


    
