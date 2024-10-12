from model import Generator
from model import Discriminator 
from CycleGAN import get_disc_loss, get_gen_loss
#from CycleGAN_newloss import get_disc_loss, get_gen_loss
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms as T   
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import subprocess
import pydicom
import cv2
from skimage.measure import label 
import csv
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import torch, torch.nn   as nn
import SimpleITK as sitk
import random
import string

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, image_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.image_loader = image_loader
        
        # Item definition
        self.itemA = config.itemA

        # Load Validation data
        self.CT_val = []
        self.itemA_val = []
        self.pixel_spacing = []
        self.pt_case = []

        for pt in os.listdir(config.val_dir):
            if pt != ".DS_Store":
                case_pt = os.path.join(config.val_dir, pt)
                pt_CT_val = self.read_dicom_series(os.path.join(case_pt, 'CT'))
                pt_itemA_val = self.read_dicom_series(os.path.join(case_pt, config.itemA))
                pixel_spacing_case = self.get_pixel_spacing(os.path.join(case_pt, 'CT'))
                self.CT_val.append(pt_CT_val)
                self.itemA_val.append(pt_itemA_val)
                self.pixel_spacing.append(pixel_spacing_case)
                self.pt_case.append(case_pt[-2:])

        self.CT_val = np.stack(self.CT_val, axis=0)
        self.itemA_val = np.stack(self.itemA_val, axis=0)

        # Model configurations.
        self.c_dim = config.c_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim 
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num   
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr  
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_epoch = config.resume_epoch

        # Test configurations.
        self.test_epochs = config.test_epochs

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir
        self.val_result_dir = config.val_result_dir
        self.test_dir = config.test_dir
        self.report_dir = config.report_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step_per_epoch = config.sample_step_per_epoch
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G_AB = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        self.G_BA = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        self.D_A = Discriminator(self.image_size[0], self.d_conv_dim, self.c_dim, self.d_repeat_num)
        self.D_B = Discriminator(self.image_size[0], self.d_conv_dim, self.c_dim, self.d_repeat_num)  
        
        self.g_optimizer = torch.optim.AdamW(list(self.G_AB.parameters()) + list(self.G_BA.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_A_optimizer = torch.optim.Adam(self.D_A.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.d_B_optimizer = torch.optim.Adam(self.D_B.parameters(), self.d_lr, [self.beta1, self.beta2])

        self.print_network(self.G_AB, 'G_AB')
        self.print_network(self.G_BA, 'G_BA')
        self.print_network(self.D_A, 'D_A')  
        self.print_network(self.D_B, 'D_B')  
            
        self.G_AB.to(self.device)
        self.G_BA.to(self.device)
        self.D_A.to(self.device)
        self.D_B.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_epoch):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from epoch {}...'.format(resume_epoch))
        G_AB_path = os.path.join(self.model_save_dir, '{}-G_AB.ckpt'.format(resume_epoch))
        G_BA_path = os.path.join(self.model_save_dir, '{}-G_BA.ckpt'.format(resume_epoch))
        D_A_path = os.path.join(self.model_save_dir, '{}-D_A.ckpt'.format(resume_epoch))
        D_B_path = os.path.join(self.model_save_dir, '{}-D_B.ckpt'.format(resume_epoch))
        self.G_AB.load_state_dict(torch.load(G_AB_path, map_location=lambda storage, loc: storage))
        self.G_BA.load_state_dict(torch.load(G_BA_path, map_location=lambda storage, loc: storage))
        self.D_A.load_state_dict(torch.load(D_A_path, map_location=lambda storage, loc: storage))
        self.D_B.load_state_dict(torch.load(D_B_path, map_location=lambda storage, loc: storage))    

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):          
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_A_optimizer.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.d_B_optimizer.param_groups:
            param_group['lr'] = d_lr              

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out
    
    def renormalize(self,ary):
        """Convert the range from [0, 1] to [-1000, 1000]."""
        re_ary = (ary*(1000. + 1000.)) - 1000.
        return re_ary
    

    def _preprocess_cbct_ct(self, dicom):
        hu_data = dicom.pixel_array.astype(np.float32) * dicom.RescaleSlope + dicom.RescaleIntercept
        clip_hu = np.clip(hu_data, -1000, 1000)
        nor_hu = (clip_hu + 1000.) / (1000. + 1000.)
        return nor_hu

    def _preprocess_mri(self, dicom):
        pixel_array = dicom.pixel_array.astype(np.float32) * dicom.RescaleSlope + dicom.RescaleIntercept
        clip_in = np.clip(pixel_array, 0, 1500)
        nor_hu = clip_in / (1500)
        return nor_hu

    def read_dicom_series(self,dicom_directory):
        dicom_files = [os.path.join(dicom_directory, filename) for filename in os.listdir(dicom_directory) if filename.endswith('.dcm')]
        dicom_files.sort()
        dicom_slices = [pydicom.read_file(file_path) for file_path in dicom_files]
        if 'CT' in dicom_directory:
            image_array = np.stack([self._preprocess_cbct_ct(slice) for slice in dicom_slices])
            return image_array
        if 'MRI' in dicom_directory:
            image_array = np.stack([self._preprocess_mri(slice) for slice in dicom_slices])
            return image_array

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)
    
    def calculate_dsc(self, image_gt, image_pred):

        # Define thresholds for segmentation
        bone_threshold = 250
        soft_tissue_lower = -200
        soft_tissue_upper = 250
        air_threshold = -200
        
        # Perform segmentation based on thresholds
        seg_bone_gt = (image_gt >= bone_threshold).astype(int)
        seg_bone_pred = (image_pred >= bone_threshold).astype(int)
        
        seg_soft_tissue_gt = ((image_gt >= soft_tissue_lower) & (image_gt < soft_tissue_upper)).astype(int)
        seg_soft_tissue_pred = ((image_pred >= soft_tissue_lower) & (image_pred < soft_tissue_upper)).astype(int)
        
        seg_body_gt = (image_gt > air_threshold).astype(int)
        seg_body_pred = (image_pred > air_threshold).astype(int)
        
        # Calculate DSC for each segment
        def calculate_single_dsc(seg_gt, seg_pred):
            intersection = np.logical_and(seg_gt, seg_pred).sum()
            union = seg_gt.sum() + seg_pred.sum()
            dsc = (2 * intersection) / union if union != 0 else 1.0
            return dsc
        
        dsc_bone = calculate_single_dsc(seg_bone_gt, seg_bone_pred)
        dsc_soft_tissue = calculate_single_dsc(seg_soft_tissue_gt, seg_soft_tissue_pred)
        dsc_body = calculate_single_dsc(seg_body_gt, seg_body_pred)
        
        return dsc_bone, dsc_soft_tissue, dsc_body
    
    def segment_bone_soft_body(self, image_pred):
        # Define thresholds for segmentation
        bone_threshold = 250
        soft_tissue_lower = -200
        soft_tissue_upper = 250
        air_threshold = -200
        # Create seg mask
        seg_bone_mask = (image_pred >= bone_threshold).astype(int)
        seg_soft_tissue_mask = ((image_pred >= soft_tissue_lower) & (image_pred < soft_tissue_upper)).astype(int)
        seg_body_mask = (image_pred > air_threshold).astype(int)
        return seg_bone_mask, seg_soft_tissue_mask, seg_body_mask

    def generate_random_name(self, length):
        characters = string.ascii_letters + string.digits  # Alphanumeric characters
        return ''.join(random.choice(characters) for _ in range(length))

    def generate_random_hn(self, length):
        characters = string.ascii_letters.upper() + string.digits  # Uppercase letters and digits
        return ''.join(random.choice(characters) for _ in range(length))
    
    def save_dicom(self, slice, i, slice_thickness, image_type, modality, manufacturer,
                    series_description, patient_name, patient_id, study_instance_uid, series_instance_uid, fram_of_reference_uid,
                    img_orien, pixel_spacing, window_center, window_width, rescale_intercept, rescale_slope, case, origin_mod, output_folder): 
        
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        image = sitk.GetImageFromArray(slice)
        image = sitk.Cast(image, sitk.sitkUInt16)
        slice_location = i * slice_thickness
        position = f"-263.8\\317\\{slice_location}"

        # Set DICOM tags
        image.SetMetaData("0008|0008", image_type)
        image.SetMetaData("0008|0060", modality)                # Modality
        image.SetMetaData("0008|0070", manufacturer)            # Manufacturer
        image.SetMetaData("0008|103e", series_description)      # Series Description
        image.SetMetaData("0010|0010", patient_name)            # Patient Name
        image.SetMetaData("0010|0020", patient_id)              # Patient ID
        image.SetMetaData("0018|0050", str(slice_thickness))    # Slice_thickness
        image.SetMetaData("0018|5100", 'HFS')
        image.SetMetaData("0020|000d", study_instance_uid)      # Study Instance UID
        image.SetMetaData("0020|000e", series_instance_uid)     # Series Instance UID
        image.SetMetaData("0020|0052", fram_of_reference_uid)   # Fram of Reference UID
        image.SetMetaData("0020|0032", position)                # Image Position (Patient)
        image.SetMetaData("0020|0037", img_orien)               # Image Orientation (Patient)
        image.SetMetaData("0020|1040", 'IC')                    # Position Reference Indicator
        image.SetMetaData("0020|1041", str(slice_location))     # Slice Location 
        image.SetMetaData("0028|0002", '1')                     # Samples per Pixel
        #image.SetMetaData("0028|0030", pixel_spacing)           # Pixel Spacing
        image.SetSpacing(pixel_spacing)
        image.SetMetaData("0028|1050", str(window_center))      # Window center
        image.SetMetaData("0028|1051", str(window_width))       # Window width
        image.SetMetaData("0028|1052", str(rescale_intercept))  # Rescale Intercept
        image.SetMetaData("0028|1053", str(rescale_slope))      # Rescale Slope
        image.SetMetaData("0040|1054", "HU")                    # Rescale Type
        
        output_file_template = os.path.join(output_folder, f"CT_{i}.dcm")
        writer.SetFileName(output_file_template)
        writer.Execute(image)

        print(f"Saving case {case} {origin_mod} slice {i + 1} at {output_file_template}...")

    def get_pixel_spacing(self, path):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        spacing = image.GetSpacing()
        return spacing

    def train(self):
        """Train StarGAN within a single dataset."""
        iter_per_epoch = len(self.image_loader)

        # Create a CSV file for recording CBCT and MRI MAE values in validation
        self.csv_file = os.path.join(self.report_dir, 'Validation.csv')
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'time', 'MAE'])

        # Set data loader.
        data_loader = self.image_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)        
        x_fixed = x_fixed.to(self.device)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Criterion
        adv_criterion = nn.MSELoss() 
        recon_criterion = nn.L1Loss() 

        # Start training from scratch or resume training.
        start_epoch = 0
        if self.resume_epoch:
            start_epoch = self.resume_epoch // iter_per_epoch
            self.restore_model(self.resume_epoch)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for epoch in range(start_epoch, self.num_epochs):
            for i, (realA, realCT) in enumerate(data_loader):
                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                # Fetch real images and labels.
                try:
                    realA, realCT = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    realA, realCT = next(data_iter)
                    
                realA = realA.to(self.device)           
                realCT = realCT.to(self.device)  
                
                # =================================================================================== #
                #                             2. Update discriminator A                               #
                # =================================================================================== #

                # Zero out the gradient before backpropagation
                self.d_A_optimizer.zero_grad()

                with torch.no_grad():
                    fakeA = self.G_BA(realCT)
                disc_A_loss = get_disc_loss(realA, fakeA, self.D_A, adv_criterion)
                disc_A_loss.backward(retain_graph=True) # Update gradients
                self.d_A_optimizer.step() # Update optimizer

                # Logging.
                loss = {}
                loss['d_A/loss'] = disc_A_loss.item()           
                
                # =================================================================================== #
                #                             3. Update discriminator B                              #
                # =================================================================================== #

                # Zero out the gradient before backpropagation
                self.d_B_optimizer.zero_grad()

                with torch.no_grad():
                    fakeB = self.G_AB(realA)
                disc_B_loss = get_disc_loss(realCT, fakeB, self.D_B, adv_criterion)
                disc_B_loss.backward(retain_graph=True) # Update gradients
                self.d_B_optimizer.step() # Update optimizer

                # Logging.
                loss['d_B/loss'] = disc_B_loss.item()  

                # Compute loss for gradient penalty.
                # alpha = torch.rand(realA.size(0), 1, 1, 1).to(self.device)
                # x_hat = (alpha * realA.data + (1 - alpha) * fakeB.data).requires_grad_(True)
                # out_src = self.D(x_hat)
                # d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                # d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp
                # self.reset_grad()
                # d_loss.backward()
                # self.d_optimizer.step()
                            
                # =================================================================================== #
                #                               4. Update generator                                   # 
                # =================================================================================== #
                self.g_optimizer.zero_grad()

                gen_loss, fake_A, fake_B = get_gen_loss( realA, realCT, self.G_AB, self.G_BA, 
                                                        self.D_A, self.D_B, adv_criterion, 
                                                        recon_criterion, recon_criterion)
                gen_loss.backward() # Update gradients
                self.g_optimizer.step() # Update optimizer

                # Logging.
                loss['g/loss'] = gen_loss.item() 
                    
                # =================================================================================== #
                #                                 5. Miscellaneous                                    #
                # =================================================================================== #

                # Print out training information.
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Epoch [{}/{}], Iteration [{}/{}]".format(et, epoch+1, self.num_epochs, i+1, iter_per_epoch)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, i+1)
                        if i==0 and epoch==0:
                            command = f"tensorboard --logdir={self.log_dir} --port=0"
                            subprocess.Popen(command, shell=True)
                            
                # Translate fixed images for debugging.
                if (i+1) % round(iter_per_epoch/self.sample_step_per_epoch) == 0:
                    with torch.no_grad():
                        x_fake_list = [x_fixed]
                        x_fake_list.append(self.G_AB(x_fixed))
                        x_concat = torch.cat(x_fake_list, dim=3)
                        sample_path = os.path.join(self.sample_dir, 'Epoch-{}-Iter-{}-images.jpg'.format(epoch+1, i+1))
                        save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                        print('Saved real and fake images into {}...'.format(sample_path))
                        
            # Save model checkpoints.
            G_AB_path = os.path.join(self.model_save_dir, '{}-G_AB.ckpt'.format(epoch+1))
            G_BA_path = os.path.join(self.model_save_dir, '{}-G_BA.ckpt'.format(epoch+1))
            D_A_path = os.path.join(self.model_save_dir, '{}-D_A.ckpt'.format(epoch+1))
            D_B_path = os.path.join(self.model_save_dir, '{}-D_B.ckpt'.format(epoch+1))      
    
            torch.save(self.G_AB.state_dict(), G_AB_path)
            torch.save(self.G_BA.state_dict(), G_BA_path)
            torch.save(self.D_A.state_dict(), D_A_path)
            torch.save(self.D_B.state_dict(), D_B_path)       
            print('Saved model checkpoints into {}...'.format(self.model_save_dir))
            
            # =================================================================================== #
            #                                      Validate                                       #
            # =================================================================================== #
            with torch.no_grad():
                all_mae_body = []

                transform = []
                transform.append(T.Resize(self.image_size))
                transform.append(T.ToTensor())
                transform.append(T.Lambda(lambda x: x.expand(3, -1, -1)))
                transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
                transform = T.Compose(transform)

                if not os.path.exists(os.path.join(self.val_result_dir, f'Epoch-{epoch+1}')):
                    os.makedirs(os.path.join(self.val_result_dir, f'Epoch-{epoch+1}'))
                
                for c in range(self.itemA_val.shape[0]):
                    CT_c = []
                    sCT_c = []

                    file_path = os.path.join(self.val_result_dir, f'Epoch-{epoch+1}')

                    if not os.path.exists(os.path.join(file_path, 'Case_{}'.format(self.pt_case[c]))):
                        os.makedirs(os.path.join(file_path, 'Case_{}'.format(self.pt_case[c])))

                    for s in range(len(self.itemA_val[c])):
                        #=========================================================================================#
                        # Preprocess image.
                        #=========================================================================================#
                        itemA_PIL_array = Image.fromarray(self.itemA_val[c][s])
                        itemA_pixel_array = transform(itemA_PIL_array)
                        itemA_image_sample = torch.tensor(itemA_pixel_array, dtype=torch.float32)

                        CT_PIL_array = Image.fromarray(self.CT_val[c][s])
                        CT_pixel_array = transform(CT_PIL_array)
                        CT_image_sample = torch.tensor(CT_pixel_array, dtype=torch.float32)

                        #=========================================================================================#
                        # Generate image.
                        #=========================================================================================#
                        sCT = self.G_AB(itemA_image_sample.to(self.device))
                        sCT_np = (sCT.cpu()).numpy()
                        #=========================================================================================#
                        # Denorm and Renorm.
                        #=========================================================================================#
                        sCT_de = self.denorm(sCT_np)
                        sCT_itemA_hu = self.renormalize(sCT_de)[0]
                        
                        tensor_A_de = self.denorm(np.array(itemA_image_sample))
                        tensor_A_hu = self.renormalize(tensor_A_de)[0]
                        
                        CT_de = self.denorm(np.array(CT_image_sample))
                        CT_hu = self.renormalize(CT_de)[0]
                        
                        #=========================================================================================#
                        # Save the translated images.
                        #=========================================================================================#
                        CT_fake_list = [torch.unsqueeze(torch.tensor(tensor_A_de[0]),0)]
                        CT_fake_list.append(torch.unsqueeze(torch.tensor(sCT_de[0]),0))
                        CT_fake_list.append(torch.unsqueeze(torch.tensor(CT_de[0]),0))
                        
                        CT_concat = torch.cat(CT_fake_list, dim=2)

                        case_path = os.path.join(file_path, 'Case_{}'.format(self.pt_case[c]))
                        result_path = os.path.join(case_path, 'Slice_{}.jpg'.format(s+1))
                        save_image(CT_concat.data.cpu(), result_path, nrow=1, padding=0)
                        #print('Saved real and fake images into {}...'.format(result_path))
                        #=========================================================================================#
                        # Append.
                        #=========================================================================================#
                        CT_c.append(CT_hu)
                        sCT_c.append(sCT_itemA_hu)
                    
                    #=========================================================================================#
                    # Calculate.
                    #=========================================================================================#
                    CT_c = np.array(CT_c)
                    sCT_c = np.array(sCT_c)
                    seg_bone_CT_mask, seg_soft_tissue_CT_mask, seg_body_CT_mask = self.segment_bone_soft_body(CT_c)
                    seg_body_sCT = sCT_c * seg_body_CT_mask
                    seg_body_CT = CT_c * seg_body_CT_mask
                    CT_body_n = np.count_nonzero(seg_body_CT_mask)
                    mae_body = np.sum(np.abs(seg_body_CT - seg_body_sCT))/CT_body_n
                    all_mae_body.append(mae_body)
                    #=========================================================================================#
                print(f'Epoch-{epoch+1}: MAE body of {self.itemA} sCT = {round(np.mean(all_mae_body),2)} HU')

                # Write CBCT and MRI MAE values to the CSV file
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                with open(self.csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch+1, et, np.mean(all_mae_body)])

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
        
        et = time.time() - start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        print('Training time: {}'.format(et))
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['End', et, '-'])

                
                
    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_epochs)
        
        data_loader = self.image_loader

        # Load Test data
        CT_test = []
        itemA_test = []
        pixel_spacing = []
        pt_case = []

        for pt in os.listdir(self.test_dir):
            if pt != ".DS_Store":
                case_pt = os.path.join(self.test_dir, pt)
                pt_CT_test = self.read_dicom_series(os.path.join(case_pt, 'CT'))
                pt_itemA_test = self.read_dicom_series(os.path.join(case_pt, self.itemA))
                pixel_spacing_case = self.get_pixel_spacing(os.path.join(case_pt, 'CT'))
                CT_test.append(pt_CT_test)
                itemA_test.append(pt_itemA_test)
                pixel_spacing.append(pixel_spacing_case)
                pt_case.append(case_pt[-2:])

        CT_test = np.stack(CT_test, axis=0)
        itemA_test = np.stack(itemA_test, axis=0)

        # Create a CSV file for recording CBCT and MRI MAE values in test
        self.csv_file_s = os.path.join(self.report_dir, 'Test_summary_{}.csv').format(self.itemA)
        self.csv_file_c = os.path.join(self.report_dir, 'Test_case_result_{}.csv'.format(self.itemA))
        with open(self.csv_file_c, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['model','Case', 'MAE', 'SSIM', 'PSNR', 'MAE Soft tissue', 'MAE Bone', 'MAE body', 'DSC Bone', 'DSC Soft tissue', 'DSC body'])
        
        with torch.no_grad():
            transform = []
            transform.append(T.Resize(self.image_size))
            transform.append(T.ToTensor())
            transform.append(T.Lambda(lambda x: x.expand(3, -1, -1)))
            transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
            transform = T.Compose(transform)

            all_mae = []
            all_mae_bone = []
            all_mae_soft_tissue = []
            all_mae_body = []
            all_psnr = []
            all_ssim = []
            all_dsc_bone = []
            all_dsc_soft_tissue = []
            all_dsc_body = []
            if not os.path.exists(os.path.join(self.result_dir, 'itemA')):
                os.makedirs(os.path.join(self.result_dir, 'itemA'))
            if not os.path.exists(os.path.join(self.result_dir, 'itemA_DICOM')):
                os.makedirs(os.path.join(self.result_dir, 'itemA_DICOM'))
            for c in range(itemA_test.shape[0]):

                CT_c = []
                sCT_c = []

                file_path = os.path.join(self.result_dir, 'itemA')
                dicom_file_path = os.path.join(self.result_dir, 'itemA_DICOM')
                
                # Define DICOM metadata
                series_description = "boothesis_sCT_CycleCBCT"
                manufacturer = "boothesis"
                modality = "sCT"
                patient_name = self.generate_random_name(20)
                patient_id = self.generate_random_hn(25)
                study_instance_uid = pydicom.uid.generate_uid()
                series_instance_uid = pydicom.uid.generate_uid()
                fram_of_reference_uid = pydicom.uid.generate_uid()
                rescale_intercept = -1024.0  
                rescale_slope = 1.0
                image_type = "DERIVED\\SECONDARY\\AXIAL"
                slice_thickness = 2.5
                pixel_spacing_c = pixel_spacing[c]
                img_orien = "1\\0\\0\\0\\1\\0"
                window_center = 40.0
                window_width = 350.0

                if not os.path.exists(os.path.join(file_path, 'Case_{}'.format(pt_case[c]))):
                    os.makedirs(os.path.join(file_path, 'Case_{}'.format(pt_case[c])))
                if not os.path.exists(os.path.join(dicom_file_path, 'Case_{}'.format(pt_case[c]))):
                    os.makedirs(os.path.join(dicom_file_path, 'Case_{}'.format(pt_case[c])))
                for s in range(itemA_test.shape[1]):
                    #=========================================================================================#
                    # Preprocess image.
                    #=========================================================================================#
                    itemA_PIL_array = Image.fromarray(itemA_test[c][s])
                    itemA_pixel_array = transform(itemA_PIL_array)
                    itemA_image_sample = torch.tensor(itemA_pixel_array, dtype=torch.float32)

                    CT_PIL_array = Image.fromarray(CT_test[c][s])
                    CT_pixel_array = transform(CT_PIL_array)
                    CT_image_sample = torch.tensor(CT_pixel_array, dtype=torch.float32)
                    #=========================================================================================#
                    # Generate image.
                    #=========================================================================================#
                    sCT = self.G_AB(itemA_image_sample.to(self.device))
                    sCT_np = (sCT.cpu()).numpy()
                    #=========================================================================================#
                    # Denorm and Renorm.
                    #=========================================================================================#
                    sCT_de = self.denorm(sCT_np)
                    sCT_itemA_hu = self.renormalize(sCT_de)[0]
                    
                    tensor_A_de = self.denorm(np.array(itemA_image_sample))
                    tensor_A_hu = self.renormalize(tensor_A_de)[0]
                    
                    CT_de = self.denorm(np.array(CT_image_sample))
                    CT_hu = self.renormalize(CT_de)[0]
                    #=========================================================================================#
                    # Save the generated images.
                    #=========================================================================================#
                    CT_fake_list = [torch.unsqueeze(torch.tensor(tensor_A_de[0]),0)]
                    CT_fake_list.append(torch.unsqueeze(torch.tensor(sCT_de[0]),0))
                    CT_fake_list.append(torch.unsqueeze(torch.tensor(CT_de[0]),0))
                    CT_fake_list = [torch.unsqueeze(torch.tensor(tensor_A_de[0]),0)]
                    CT_fake_list.append(torch.unsqueeze(torch.tensor(sCT_de[0]),0))
                    CT_fake_list.append(torch.unsqueeze(torch.tensor(CT_de[0]),0))
                    CT_concat = torch.cat(CT_fake_list, dim=2)
                    file_path = os.path.join(self.result_dir, 'itemA')
                    case_path = os.path.join(file_path, 'Case_{}'.format(pt_case[c]))
                    result_path = os.path.join(case_path, 'Slice_{}.jpg'.format(s+1))
                    save_image(CT_concat.data.cpu(), result_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(result_path))
                    #=========================================================================================#
                    # Save DICOM
                    #=========================================================================================#
                    

                    dicom_file_path = os.path.join(self.result_dir, 'itemA_DICOM')
                    dicom_case_path = os.path.join(dicom_file_path, 'Case_{}'.format(pt_case[c]))
                    save = True
                    if save:
                        self.save_dicom(sCT_itemA_hu, s, slice_thickness, image_type, 'CT', manufacturer,
                                        series_description, patient_name, patient_id, study_instance_uid, series_instance_uid, fram_of_reference_uid,
                                        img_orien, pixel_spacing_c, window_center, window_width, rescale_intercept, rescale_slope, 
                                        pt_case[c], 'CBCT', dicom_case_path)
                    
                    #=========================================================================================#
                    # Append.
                    #=========================================================================================#
                    CT_c.append(CT_hu)
                    sCT_c.append(sCT_itemA_hu)

                #=========================================================================================#
                # Calculate.
                #=========================================================================================#
                CT_c = np.array(CT_c)
                sCT_c = np.array(sCT_c)
                mae = np.mean(np.abs(CT_c - sCT_c))
                psnr_value = psnr(CT_c , sCT_c, data_range= CT_c.max() - CT_c.min())
                ssim_value = ssim(CT_c , sCT_c, data_range= CT_c.max() - CT_c.min())
                dsc_bone, dsc_soft_tissue, dsc_body = self.calculate_dsc(CT_c , sCT_c)
                
                seg_bone_CT_mask, seg_soft_tissue_CT_mask, seg_body_CT_mask = self.segment_bone_soft_body(CT_c)

                seg_bone_sCT = sCT_c * seg_bone_CT_mask
                seg_soft_tissue_sCT = sCT_c * seg_soft_tissue_CT_mask
                seg_body_sCT = sCT_c * seg_body_CT_mask

                seg_bone_CT = CT_c * seg_bone_CT_mask
                seg_soft_tissue_CT = CT_c * seg_soft_tissue_CT_mask
                seg_body_CT = CT_c * seg_body_CT_mask

                CT_bone_n = np.count_nonzero(seg_bone_CT_mask)
                CT_sf_n = np.count_nonzero(seg_soft_tissue_CT_mask)
                CT_body_n = np.count_nonzero(seg_body_CT)
                
                mae_bone = np.sum(np.abs(seg_bone_CT - seg_bone_sCT))/CT_bone_n
                mae_soft_tissue = np.sum(np.abs(seg_soft_tissue_CT - seg_soft_tissue_sCT))/CT_sf_n
                mae_body = np.sum(np.abs(seg_body_CT - seg_body_sCT))/CT_body_n
                #=========================================================================================#
                all_dsc_bone.append(dsc_bone)
                all_dsc_soft_tissue.append(dsc_soft_tissue)
                all_dsc_body.append(dsc_body)
                all_psnr.append(psnr_value)
                all_ssim.append(ssim_value)
                all_mae.append(mae)
                all_mae_bone.append(mae_bone)
                all_mae_soft_tissue.append(mae_soft_tissue)
                all_mae_body.append(mae_body)
                #=========================================================================================#
                #=========================================================================================#
                with open(self.csv_file_c, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([f'Case_{pt_case[c]}', f'Cycle{self.itemA}', f'{round(np.mean(mae),2)}', f'{round(np.mean(ssim_value),2)}', f'{round(np.mean(psnr_value),2)}', 
                                f'{round(np.mean(mae_soft_tissue),2)}', f'{round(np.mean(mae_bone),2)}', f'{round(np.mean(mae_body),2)}',
                                f'{round(np.mean(dsc_soft_tissue),2)}', f'{round(np.mean(dsc_bone),2)}', f'{round(np.mean(dsc_body),2)}'])
                #=========================================================================================#
            print('MAE = {}({}) HU'.format(round(np.mean(all_mae),2), round(np.std(all_mae),2)))
            print('SSIM = {}({}) A.U.'.format(round(np.mean(all_ssim),2), round(np.std(all_ssim),2)))
            print('PSNR = {}({}) dB'.format(round(np.mean(all_psnr),2), round(np.std(all_psnr),2)))
            print('MAE Soft tissue = {}({}) HU'.format(round(np.mean(all_mae_soft_tissue),2), round(np.std(all_mae_soft_tissue),2)))
            print('MAE Bone = {}({}) HU'.format(round(np.mean(all_mae_bone),2), round(np.std(all_mae_bone),2)))
            print('MAE Body = {}({}) HU'.format(round(np.mean(all_mae_body),2), round(np.std(all_mae_body),2)))
            print('DSC Soft tissue = {}({})'.format(round(np.mean(all_dsc_soft_tissue),2), round(np.std(all_dsc_soft_tissue),2)))
            print('DSC Bone = {}({})'.format(round(np.mean(all_dsc_bone),2), round(np.std(all_dsc_bone),2)))
            print('DSC Body = {}({})'.format(round(np.mean(all_dsc_body),2), round(np.std(all_dsc_body),2)))
            
            # Write CBCT and MRI MAE values to the CSV file
            with open(self.csv_file_s, mode='a', newline='') as file:
                writer = csv.writer(file)
                metic = ['MAE', 'SSIM', 'PSNR',
                         'MAE Soft tissue', 'MAE Bone', 'MAE Body',
                         'DSC Soft tissue', 'DSC Bone','DSC Body']
                value = ['{} + {}'.format(round(np.mean(all_mae),2), round(np.std(all_mae),2)),
                        '{} + {}'.format(round(np.mean(all_ssim),2), round(np.std(all_ssim),2)),
                        '{} + {}'.format(round(np.mean(all_psnr),2), round(np.std(all_psnr),2)),
                        '{} + {}'.format(round(np.mean(all_mae_soft_tissue),2), round(np.std(all_mae_soft_tissue),2)),
                        '{} + {}'.format(round(np.mean(all_mae_bone),2), round(np.std(all_mae_bone),2)),
                        '{} + {}'.format(round(np.mean(all_mae_body),2), round(np.std(all_mae_body),2)),
                        '{} + {}'.format(round(np.mean(all_dsc_soft_tissue),2), round(np.std(all_dsc_soft_tissue),2)),
                        '{} + {}'.format(round(np.mean(all_dsc_bone),2), round(np.std(all_dsc_bone),2)),
                        '{} + {}'.format(round(np.mean(all_dsc_body),2), round(np.std(all_dsc_body),2)),
                        ]
                
                #for i in range(len(metic)):
                    #writer.writerow([metic[i], value[i]])
                writer.writerow([metic[0], metic[1], metic[2]])
                writer.writerow([value[0], value[1], value[2]])
                writer.writerow([metic[3], metic[4], metic[5]])
                writer.writerow([value[3], value[4], value[5]])
                writer.writerow([metic[6], metic[7], metic[8]])
                writer.writerow([value[6], value[7], value[8]])

            

class HingeLoss(torch.nn.Module):

    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, output, target):

        hinge_loss = 1. - torch.mul(output, target)
        return torch.mean(F.relu(hinge_loss))