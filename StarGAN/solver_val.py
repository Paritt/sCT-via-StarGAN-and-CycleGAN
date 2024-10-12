from model import Generator
from model import Discriminator
from model import Classifier      
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
import SimpleITK as sitk
import random
import string

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, image_loader, class_loader, val_data_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.image_loader = image_loader
        self.class_loader = class_loader
        self.val_data_loader = val_data_loader

        # Load Validation data
        self.CT_val = []
        self.CBCT_val = []
        self.MRI_val = []
        self.pixel_spacing = []
        self.pt_case = []

        for pt in os.listdir(config.validate_case_dir):
            if pt != ".DS_Store":
                case_pt = os.path.join(config.validate_case_dir, pt)
                pt_CT_val = self.read_dicom_series(os.path.join(case_pt, 'CT'))
                pt_CBCT_val = self.read_dicom_series(os.path.join(case_pt, 'CBCT'))
                pt_MRI_val = self.read_dicom_series(os.path.join(case_pt, 'MRI'))
                pixel_spacing_case = self.get_pixel_spacing(os.path.join(case_pt, 'CT'))
                self.CT_val.append(pt_CT_val)
                self.CBCT_val.append(pt_CBCT_val)
                self.MRI_val.append(pt_MRI_val)
                self.pixel_spacing.append(pixel_spacing_case)
                self.pt_case.append(case_pt[-2:])

        self.CT_val = np.stack(self.CT_val, axis=0)
        self.CBCT_val = np.stack(self.CBCT_val, axis=0)
        self.MRI_val = np.stack(self.MRI_val, axis=0)


        # Model configurations.
        self.c_dim = config.c_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.c_conv_dim = config.c_conv_dim   
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.c_repeat_num = config.c_repeat_num    
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.c_lr = config.c_lr    
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.c_beta1 = config.c_beta1
        self.resume_epoch = config.resume_epoch

        # Test configurations.
        self.test_epochs = config.test_epochs
        self.test_MRI = config.test_MRI
        self.test_CBCT = config.test_CBCT

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
        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        self.D = Discriminator(self.image_size[0], self.d_conv_dim, self.c_dim, self.d_repeat_num) 
        self.C = Classifier(self.image_size[0], self.c_conv_dim, self.c_dim, self.c_repeat_num )  
        
        self.g_optimizer = torch.optim.AdamW(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.c_optimizer = torch.optim.Adam(self.C.parameters(), self.c_lr, [self.c_beta1, self.beta2])       
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        self.print_network(self.C, 'C')     
            
        self.G.to(self.device)
        self.D.to(self.device)
        self.C.to(self.device)      

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
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_epoch))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_epoch))
        C_path = os.path.join(self.model_save_dir, '{}-C.ckpt'.format(resume_epoch)) 
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.C.load_state_dict(torch.load(C_path, map_location=lambda storage, loc: storage))    

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr, c_lr):          
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.c_optimizer.param_groups:
            param_group['lr'] = c_lr               
            
    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.c_optimizer.zero_grad() 

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

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=3):
        """Generate target domain labels for debugging and testing."""
        c_trg_list = []
        for i in range(c_dim):
            c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)    
            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.cross_entropy(logit, target)
    
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
        # image.SetMetaData("0028|0030", pixel_spacing)           # Pixel Spacing
        image.SetSpacing(pixel_spacing)
        image.SetMetaData("0028|1050", str(window_center))      # Window center
        image.SetMetaData("0028|1051", str(window_width))       # Window width
        image.SetMetaData("0028|1052", str(rescale_intercept))  # Rescale Intercept
        image.SetMetaData("0028|1053", str(rescale_slope))      # Rescale Slope
        image.SetMetaData("0040|1054", "HU")                    # Rescale Type
        
        output_file_template = os.path.join(output_folder, f"CT_{i}.dcm")
        writer.SetFileName(output_file_template)
        writer.Execute(image)

        #print(f"Saving case {case} {origin_mod} slice {i + 1} at {output_file_template}...")

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
            writer.writerow(['Epoch', 'time', 'CBCT MAE', 'MRI MAE'])

        # Set data loader.
        data_loader = self.image_loader
        data_loader_class = self.class_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)        
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim)  
        data_iter_class = iter(data_loader_class)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr
        c_lr = self.c_lr       

        # Start training from scratch or resume training.
        start_epoch = 0
        if self.resume_epoch:
            start_epoch = self.resume_epoch
            self.restore_model(self.resume_epoch)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for epoch in range(start_epoch, self.num_epochs):
            for i, (x_real, label_org) in enumerate(data_loader):
                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                # Fetch real images and labels.
                try:
                    x_real, label_org = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    x_real, label_org = next(data_iter)
                    
                try:
                    x_real_class, label_org_class = next(data_iter_class)
                except:
                    data_iter_class = iter(data_loader_class)
                    x_real_class, label_org_class = next(data_iter_class)

                # Generate target domain labels randomly.
                rand_idx = torch.randperm(label_org.size(0)) 
                label_trg = label_org[rand_idx]

                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)
                    
                x_real = x_real.to(self.device)           # Input images.
                x_real_class = x_real_class.to(self.device)  
                
                c_org = c_org.to(self.device)             # Original domain labels.
                c_trg = c_trg.to(self.device)             # Target domain labels.
                label_org = label_org.to(self.device)     # Labels for computing classification loss.
                label_trg = label_trg.to(self.device)     # Labels for computing classification loss.
                label_org_class = label_org_class.to(self.device)
                
                # =================================================================================== #
                #                             2-0. Train the Classifier                               #
                # =================================================================================== #

                # Compute loss with real images.
                out_cls = self.C(x_real_class)
                
                c_loss = self.classification_loss(out_cls, label_org_class)
                        
                self.reset_grad()
                c_loss.backward(retain_graph=True)
                self.c_optimizer.step()

                # Logging.
                loss = {}
                loss['C/loss'] = c_loss.item()           
                
                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.
                out_src = self.D(x_real)
                #d_loss_real = - torch.mean(out_src)
                d_loss_real = torch.mean(F.relu(1. - torch.mul(out_src, 1.0)))
                
                # Compute loss with fake images.
                x_fake = self.G(x_real, c_trg)                      
                out_src = self.D(x_fake.detach())
                d_loss_fake = torch.mean(F.relu(1. - torch.mul(out_src, -1.0)))

                # # Compute loss for gradient penalty.
                # alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                # x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                # out_src = self.D(x_hat)
                # d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                # d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp
                d_loss = d_loss_real + d_loss_fake
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                # loss['D/loss_gp'] = d_loss_gp.item()
                            
                # =================================================================================== #
                #                               3. Train the generator                                # 
                # =================================================================================== #
                
                if (i+1) % self.n_critic == 0:          
                    # Original-to-target domain.
                    x_fake = self.G(x_real, c_trg)
                    out_src = self.D(x_fake)
                    g_loss_fake = - torch.mean(out_src)
                    out_cls_f = self.C(x_fake)
                    c_loss_f = self.classification_loss(out_cls_f, c_trg)

                    # Target-to-original domain.
                    x_reconst = self.G(x_fake, c_org)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * c_loss_f
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['G/loss_adv'] = g_loss_fake.item()
                    loss['G/loss_rec'] = self.lambda_rec * g_loss_rec.item()
                    loss['G/loss_cls'] = self.lambda_cls * c_loss_f.item()
                    
                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
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
                        for c_fixed in c_fixed_list:
                            x_fake_list.append(self.G(x_fixed, c_fixed))
                        x_concat = torch.cat(x_fake_list, dim=3)
                        sample_path = os.path.join(self.sample_dir, 'Epoch-{}-Iter-{}-images.jpg'.format(epoch+1, i+1))
                        save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                        print('Saved real and fake images into {}...'.format(sample_path))
                        
            # Save model checkpoints.
            G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(epoch+1))
            D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(epoch+1))
            C_path = os.path.join(self.model_save_dir, '{}-C.ckpt'.format(epoch+1))   
    
            torch.save(self.G.state_dict(), G_path)
            torch.save(self.D.state_dict(), D_path)
            torch.save(self.C.state_dict(), C_path)     
            print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # =================================================================================== #
            #                                      Validate                                       #
            # =================================================================================== #
            with torch.no_grad():
                label_trg = torch.tensor([0]) # 0=CT, 1=MRI, 2=CBCT
                c_trg = self.label2onehot(label_trg, self.c_dim)
                transform = []
                transform.append(T.Resize(self.image_size))
                transform.append(T.ToTensor())
                transform.append(T.Lambda(lambda x: x.expand(3, -1, -1)))
                transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
                transform = T.Compose(transform)

                #=========================================================================================#
                # Validate MR.
                #=========================================================================================#

                all_mae_MRI = []

                for c in range(self.MRI_val.shape[0]):
                    CT_c = []
                    sCT_c = []

                    for s in range(len(self.MRI_val[c])):
                        #=========================================================================================#
                        # Preprocess image.
                        #=========================================================================================#
                        MRI_PIL_array = Image.fromarray(self.MRI_val[c][s])
                        MRI_pixel_array = transform(MRI_PIL_array)
                        MRI_image_sample = torch.tensor(MRI_pixel_array, dtype=torch.float32)

                        CT_PIL_array = Image.fromarray(self.CT_val[c][s])
                        CT_pixel_array = transform(CT_PIL_array)
                        CT_image_sample = torch.tensor(CT_pixel_array, dtype=torch.float32)
                        #=========================================================================================#
                        # Generate image.
                        #=========================================================================================#
                        sCT_from_MRI = self.G(MRI_image_sample.to(self.device).unsqueeze(0),c_trg.to(self.device))
                        sCT_MRI_np = np.squeeze((sCT_from_MRI.cpu()).numpy())
                        #=========================================================================================#
                        # Denorm and Renorm.
                        #=========================================================================================#
                        sCT_MRI_de = self.denorm(sCT_MRI_np)
                        sCT_MRI_hu = self.renormalize(sCT_MRI_de)[0]
                    
                        tensor_A_de = self.denorm(np.array(MRI_image_sample))
                        tensor_A_hu = self.renormalize(tensor_A_de)[0]
                    
                        CT_de = self.denorm(np.array(CT_image_sample))
                        CT_hu = self.renormalize(CT_de)[0]
                        #=========================================================================================#
                        # Append.
                        #=========================================================================================#
                        CT_c.append(CT_hu)
                        sCT_c.append(sCT_MRI_hu)
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
                    all_mae_MRI.append(mae_body)
                    #=========================================================================================#
                print(f'Epoch-{epoch+1}: MAE body of MRI sCT = {round(np.mean(all_mae_MRI),2)} HU')

                #=========================================================================================#
                # Validate CBCT.
                #=========================================================================================#

                all_mae_CBCT = []

                for c in range(self.CBCT_val.shape[0]):
                    CT_c = []
                    sCT_c = []

                    for s in range(len(self.CBCT_val[c])):
                        #=========================================================================================#
                        # Preprocess image.
                        #=========================================================================================#
                        CBCT_PIL_array = Image.fromarray(self.CBCT_val[c][s])
                        CBCT_pixel_array = transform(CBCT_PIL_array)
                        CBCT_image_sample = torch.tensor(CBCT_pixel_array, dtype=torch.float32)

                        CT_PIL_array = Image.fromarray(self.CT_val[c][s])
                        CT_pixel_array = transform(CT_PIL_array)
                        CT_image_sample = torch.tensor(CT_pixel_array, dtype=torch.float32)
                        #=========================================================================================#
                        # Generate image.
                        #=========================================================================================#
                        sCT_from_CBCT = self.G(CBCT_image_sample.to(self.device).unsqueeze(0),c_trg.to(self.device))
                        sCT_CBCT_np = np.squeeze((sCT_from_CBCT.cpu()).numpy())
                        #=========================================================================================#
                        # Denorm and Renorm.
                        #=========================================================================================#
                        sCT_CBCT_de = self.denorm(sCT_CBCT_np)
                        sCT_CBCT_hu = self.renormalize(sCT_CBCT_de)[0]

                        tensor_A_de = self.denorm(np.array(CBCT_image_sample))
                        tensor_A_hu = self.renormalize(tensor_A_de)[0]

                        CT_de = self.denorm(np.array(CT_image_sample))
                        CT_hu = self.renormalize(CT_de)[0]
                        #=========================================================================================#
                        # Append.
                        #=========================================================================================#
                        CT_c.append(CT_hu)
                        sCT_c.append(sCT_CBCT_hu)

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
                    all_mae_CBCT.append(mae_body)
                    #=========================================================================================#
                print(f'Epoch-{epoch+1}: MAE body of CBCT sCT = {round(np.mean(all_mae_CBCT),2)} HU')

                # Write CBCT and MRI MAE values to the CSV file
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                with open(self.csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch+1, et, np.mean(all_mae_CBCT), np.mean(all_mae_MRI)])

                for i, (x_real, c_org) in enumerate(self.val_data_loader):
                    # Prepare input images and target domain labels.
                    x_real = x_real.to(self.device)
                    c_trg_list = self.create_labels(c_org, self.c_dim)
                    # Translate images.
                    x_fake_list = [x_real] 
                    for c_trg in c_trg_list:
                        x_fake_list.append(self.G(x_real, c_trg))
                    # Save the translated images.
                    x_concat = torch.cat(x_fake_list, dim=3)
                    if not os.path.exists(os.path.join(self.val_result_dir, 'Epoch-{}'.format(epoch+1))):
                        os.makedirs(os.path.join(self.val_result_dir, 'Epoch-{}'.format(epoch+1)))
                    val_path = os.path.join(self.val_result_dir, 'Epoch-{}/{}-images.jpg'.format(epoch+1, i+1))
                    save_image(self.denorm(x_concat.data.cpu()), val_path, nrow=1, padding=0)
                    print('Saved validate real and fake images into {}...'.format(val_path))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                c_lr -= (self.c_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr, c_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}, c_lr: {}.'.format(g_lr, d_lr, c_lr))
        
        et = time.time() - start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        print('Training time: {}'.format(et))
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['End', et, '-', '-'])
        
    
    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_epochs)
        
        data_loader = self.image_loader

        # Load Test data
        CT_test = []
        CBCT_test = []
        MRI_test = []
        pixel_spacing = []
        pt_case = []

        test_MR = self.test_MRI
        test_CBCT = self.test_CBCT

        for pt in os.listdir(self.test_dir):
            if pt != ".DS_Store":
                case_pt = os.path.join(self.test_dir, pt)
                pt_CT_test = self.read_dicom_series(os.path.join(case_pt, 'CT'))
                pixel_spacing_case = self.get_pixel_spacing(os.path.join(case_pt, 'CT'))
                if test_CBCT:
                    pt_CBCT_test = self.read_dicom_series(os.path.join(case_pt, 'CBCT'))
                    CBCT_test.append(pt_CBCT_test)
                if test_MR:
                    pt_MRI_test = self.read_dicom_series(os.path.join(case_pt, 'MRI'))
                    MRI_test.append(pt_MRI_test)
                CT_test.append(pt_CT_test)
                pixel_spacing.append(pixel_spacing_case)
                pt_case.append(case_pt[-2:])


        CT_test = np.stack(CT_test, axis=0)
        if test_CBCT:
            CBCT_test = np.stack(CBCT_test, axis=0)
        if test_MR:
            MRI_test = np.stack(MRI_test, axis=0)

        # Create a CSV file for recording CBCT and MRI MAE values in test
        self.csv_file_s = os.path.join(self.report_dir, 'Test_summary.csv')
        self.csv_file_c = os.path.join(self.report_dir, 'Test_case_result.csv')
        with open(self.csv_file_c, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['type','Case', 'MAE', 'SSIM', 'PSNR', 'MAE Soft tissue', 'MAE Bone', 'MAE body', 'DSC Bone', 'DSC Soft tissue', 'DSC body'])
        
        with torch.no_grad():
            label_trg = torch.tensor([0]) # 0=CT, 1=MRI, 2=CBCT
            c_trg = self.label2onehot(label_trg, self.c_dim)
            transform = []
            transform.append(T.Resize(self.image_size))
            transform.append(T.ToTensor())
            transform.append(T.Lambda(lambda x: x.expand(3, -1, -1)))
            transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
            transform = T.Compose(transform)

            all_mae_MRI = []
            all_mae_bone_MRI = []
            all_mae_soft_tissue_MRI = []
            all_mae_body_MRI = []
            all_psnr_MRI = []
            all_ssim_MRI = []
            all_dsc_bone_MRI = []
            all_dsc_soft_tissue_MRI = []
            all_dsc_body_MRI = []

            if test_MR:

                if not os.path.exists(os.path.join(self.result_dir, 'MRI')):
                    os.makedirs(os.path.join(self.result_dir, 'MRI'))
                if not os.path.exists(os.path.join(self.result_dir, 'MRI_DICOM')):
                    os.makedirs(os.path.join(self.result_dir, 'MRI_DICOM'))
                for c in range(MRI_test.shape[0]):
                    
                    CT_c = []
                    sCT_c = []

                    file_path = os.path.join(self.result_dir, 'MRI')
                    dicom_file_path = os.path.join(self.result_dir, 'MRI_DICOM')
                    
                    # Define DICOM metadata
                    series_description = "boothesis_sCT_StarMRI"
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
                    for s in range(MRI_test.shape[1]):
                        #=========================================================================================#
                        # Preprocess image.
                        #=========================================================================================#
                        MRI_PIL_array = Image.fromarray(MRI_test[c][s])
                        MRI_pixel_array = transform(MRI_PIL_array)
                        MRI_image_sample = torch.tensor(MRI_pixel_array, dtype=torch.float32)

                        CT_PIL_array = Image.fromarray(CT_test[c][s])
                        CT_pixel_array = transform(CT_PIL_array)
                        CT_image_sample = torch.tensor(CT_pixel_array, dtype=torch.float32)
                        #=========================================================================================#
                        # Generate image.
                        #=========================================================================================#
                        sCT_from_MRI = self.G(MRI_image_sample.to(self.device).unsqueeze(0),c_trg.to(self.device))
                        sCT_MRI_np = np.squeeze((sCT_from_MRI.cpu()).numpy())
                        #=========================================================================================#
                        # Denorm and Renorm.
                        #=========================================================================================#
                        sCT_MRI_de = self.denorm(sCT_MRI_np)
                        sCT_MRI_hu = self.renormalize(sCT_MRI_de)[0]
                        
                        MRI_de = self.denorm(np.array(MRI_image_sample))
                        MRI_hu = self.renormalize(MRI_de)[0]
                        
                        CT_de = self.denorm(np.array(CT_image_sample))
                        CT_hu = self.renormalize(CT_de)[0]
                        #=========================================================================================#
                        # Save the generated images.
                        #=========================================================================================#
                        CT_fake_list = [torch.unsqueeze(torch.tensor(MRI_de[0]),0)]
                        CT_fake_list.append(torch.unsqueeze(torch.tensor(sCT_MRI_de[0]),0))
                        CT_fake_list.append(torch.unsqueeze(torch.tensor(CT_de[0]),0))
                        CT_fake_list = [torch.unsqueeze(torch.tensor(MRI_de[0]),0)]
                        CT_fake_list.append(torch.unsqueeze(torch.tensor(sCT_MRI_de[0]),0))
                        CT_fake_list.append(torch.unsqueeze(torch.tensor(CT_de[0]),0))
                        CT_concat = torch.cat(CT_fake_list, dim=2)
                        file_path = os.path.join(self.result_dir, 'MRI')
                        case_path = os.path.join(file_path, 'Case_{}'.format(pt_case[c]))
                        result_path = os.path.join(case_path, 'Slice_{}.jpg'.format(s+1))
                        save_image(CT_concat.data.cpu(), result_path, nrow=1, padding=0)
                        print('Saved real and fake images into {}...'.format(result_path))
                        #=========================================================================================#
                        # Save DICOM
                        #=========================================================================================#
                        
                        dicom_file_path = os.path.join(self.result_dir, 'MRI_DICOM')
                        dicom_case_path = os.path.join(dicom_file_path, 'Case_{}'.format(pt_case[c]))
                        self.save_dicom(sCT_MRI_hu, s, slice_thickness, image_type, 'CT', manufacturer,
                                        series_description, patient_name, patient_id, study_instance_uid, series_instance_uid, fram_of_reference_uid,
                                        img_orien, pixel_spacing_c, window_center, window_width, rescale_intercept, rescale_slope, 
                                        pt_case[c], 'MRI', dicom_case_path)
                        
                        #=========================================================================================#
                        # Append.
                        #=========================================================================================#
                        CT_c.append(CT_hu)
                        sCT_c.append(sCT_MRI_hu)

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
                    all_dsc_bone_MRI.append(dsc_bone)
                    all_dsc_soft_tissue_MRI.append(dsc_soft_tissue)
                    all_dsc_body_MRI.append(dsc_body)
                    all_psnr_MRI.append(psnr_value)
                    all_ssim_MRI.append(ssim_value)
                    all_mae_MRI.append(mae)
                    all_mae_bone_MRI.append(mae_bone)
                    all_mae_soft_tissue_MRI.append(mae_soft_tissue)
                    all_mae_body_MRI.append(mae_body)
                    #=========================================================================================#
                    with open(self.csv_file_c, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([f'Case_{pt_case[c]}', 'StarMRI', f'{round(np.mean(mae),2)}', f'{round(np.mean(ssim_value),2)}', f'{round(np.mean(psnr_value),2)}', 
                                f'{round(np.mean(mae_soft_tissue),2)}', f'{round(np.mean(mae_bone),2)}', f'{round(np.mean(mae_body),2)}',
                                f'{round(np.mean(dsc_soft_tissue),2)}', f'{round(np.mean(dsc_bone),2)}', f'{round(np.mean(dsc_body),2)}'])
                    #=========================================================================================#
                    #=========================================================================================#
                print('MAE of MRI = {}({}) HU'.format(round(np.mean(all_mae_MRI),2), round(np.std(all_mae_MRI),2)))
                print('SSIM of MRI = {}({}) A.U.'.format(round(np.mean(all_ssim_MRI),2), round(np.std(all_ssim_MRI),2)))
                print('PSNR of MRI = {}({}) dB'.format(round(np.mean(all_psnr_MRI),2), round(np.std(all_psnr_MRI),2)))
                print('MAE Soft tissue MRI = {} ({})'.format(round(np.mean(all_mae_soft_tissue_MRI),2), round(np.std(all_mae_soft_tissue_MRI),2)))
                print('MAE Bone MRI = {} ({})'.format(round(np.mean(all_mae_bone_MRI),2), round(np.std(all_mae_bone_MRI),2)))
                print('MAE body MRI = {} ({})'.format(round(np.mean(all_mae_body_MRI),2), round(np.std(all_mae_body_MRI),2)))
                print('DSC Soft tissue of MRI = {}({})'.format(round(np.mean(all_dsc_soft_tissue_MRI),2), round(np.std(all_dsc_soft_tissue_MRI),2)))
                print('DSC Bone of MRI = {}({})'.format(round(np.mean(all_dsc_bone_MRI),2), round(np.std(all_dsc_bone_MRI),2)))
                print('DSC body of MRI = {}({})'.format(round(np.mean(all_dsc_body_MRI),2), round(np.std(all_dsc_body_MRI),2)))
                

            all_mae_CBCT = []
            all_mae_bone_CBCT = []
            all_mae_soft_tissue_CBCT = []
            all_mae_body_CBCT = []
            all_psnr_CBCT = []
            all_ssim_CBCT = []
            all_dsc_bone_CBCT = []
            all_dsc_soft_tissue_CBCT = []
            all_dsc_body_CBCT = []

            if test_CBCT:

                if not os.path.exists(os.path.join(self.result_dir, 'CBCT')):
                    os.makedirs(os.path.join(self.result_dir, 'CBCT'))
                if not os.path.exists(os.path.join(self.result_dir, 'CBCT_DICOM')):
                    os.makedirs(os.path.join(self.result_dir, 'CBCT_DICOM'))
                for c in range(CBCT_test.shape[0]):
                    
                    CT_c = []
                    sCT_c = []

                    file_path = os.path.join(self.result_dir, 'CBCT')
                    dicom_file_path = os.path.join(self.result_dir, 'CBCT_DICOM')

                    # Define DICOM metadata
                    series_description = "boothesis_sCT_StarCBCT"
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
                    for s in range(CBCT_test.shape[1]):
                        #=========================================================================================#
                        # Preprocess image.
                        #=========================================================================================#
                        CBCT_PIL_array = Image.fromarray(CBCT_test[c][s])
                        CBCT_pixel_array = transform(CBCT_PIL_array)
                        CBCT_image_sample = torch.tensor(CBCT_pixel_array, dtype=torch.float32)

                        CT_PIL_array = Image.fromarray(CT_test[c][s])
                        CT_pixel_array = transform(CT_PIL_array)
                        CT_image_sample = torch.tensor(CT_pixel_array, dtype=torch.float32)
                        #=========================================================================================#
                        # Generate image.
                        #=========================================================================================#
                        sCT_from_CBCT = self.G(CBCT_image_sample.to(self.device).unsqueeze(0),c_trg.to(self.device))
                        sCT_CBCT_np = np.squeeze((sCT_from_CBCT.cpu()).numpy())
                        #=========================================================================================#
                        # Denorm and Renorm.
                        #=========================================================================================#
                        sCT_CBCT_de = self.denorm(sCT_CBCT_np)
                        sCT_CBCT_hu = self.renormalize(sCT_CBCT_de)[0]
                        
                        CBCT_de = self.denorm(np.array(CBCT_image_sample))
                        CBCT_hu = self.renormalize(CBCT_de)[0]
                        
                        CT_de = self.denorm(np.array(CT_image_sample))
                        CT_hu = self.renormalize(CT_de)[0]
                        #=========================================================================================#
                        # Save the generated images.
                        #=========================================================================================#
                        CT_fake_list = [torch.unsqueeze(torch.tensor(CBCT_de[0]),0)]
                        CT_fake_list.append(torch.unsqueeze(torch.tensor(sCT_CBCT_de[0]),0))
                        CT_fake_list.append(torch.unsqueeze(torch.tensor(CT_de[0]),0))
                        CT_fake_list = [torch.unsqueeze(torch.tensor(CBCT_de[0]),0)]
                        CT_fake_list.append(torch.unsqueeze(torch.tensor(sCT_CBCT_de[0]),0))
                        CT_fake_list.append(torch.unsqueeze(torch.tensor(CT_de[0]),0))
                        CT_concat = torch.cat(CT_fake_list, dim=2)
                        file_path = os.path.join(self.result_dir, 'CBCT')
                        case_path = os.path.join(file_path, 'Case_{}'.format(pt_case[c]))
                        result_path = os.path.join(case_path, 'Slice_{}.jpg'.format(s+1))
                        save_image(CT_concat.data.cpu(), result_path, nrow=1, padding=0)
                        print('Saved real and fake images into {}...'.format(result_path))
                        #=========================================================================================#
                        # Save DICOM
                        #=========================================================================================#
                        
                        dicom_file_path = os.path.join(self.result_dir, 'CBCT_DICOM')
                        dicom_case_path = os.path.join(dicom_file_path, 'Case_{}'.format(pt_case[c]))
                        self.save_dicom(sCT_CBCT_hu, s, slice_thickness, image_type, 'CT', manufacturer,
                                        series_description, patient_name, patient_id, study_instance_uid, series_instance_uid, fram_of_reference_uid,
                                        img_orien, pixel_spacing_c, window_center, window_width, rescale_intercept, rescale_slope, 
                                        pt_case[c], 'CBCT', dicom_case_path)
                        
                        #=========================================================================================#
                        # Append.
                        #=========================================================================================#
                        CT_c.append(CT_hu)
                        sCT_c.append(sCT_CBCT_hu)

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
                    all_dsc_bone_CBCT.append(dsc_bone)
                    all_dsc_soft_tissue_CBCT.append(dsc_soft_tissue)
                    all_dsc_body_CBCT.append(dsc_body)
                    all_psnr_CBCT.append(psnr_value)
                    all_ssim_CBCT.append(ssim_value)
                    all_mae_CBCT.append(mae)
                    all_mae_bone_CBCT.append(mae_bone)
                    all_mae_soft_tissue_CBCT.append(mae_soft_tissue)
                    all_mae_body_CBCT.append(mae_body)
                    #=========================================================================================#
                    with open(self.csv_file_c, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([f'Case_{pt_case[c]}', 'StarCBCT', f'{round(np.mean(mae),2)}', f'{round(np.mean(ssim_value),2)}', f'{round(np.mean(psnr_value),2)}', 
                                f'{round(np.mean(mae_soft_tissue),2)}', f'{round(np.mean(mae_bone),2)}', f'{round(np.mean(mae_body),2)}',
                                f'{round(np.mean(dsc_soft_tissue),2)}', f'{round(np.mean(dsc_bone),2)}', f'{round(np.mean(dsc_body),2)}'])
                    #=========================================================================================#
                    #=========================================================================================#
                print('MAE of CBCT = {}({}) HU'.format(round(np.mean(all_mae_CBCT),2), round(np.std(all_mae_CBCT),2)))
                print('SSIM of CBCT = {}({}) A.U.'.format(round(np.mean(all_ssim_CBCT),2), round(np.std(all_ssim_CBCT),2)))
                print('PSNR of CBCT = {}({}) dB'.format(round(np.mean(all_psnr_CBCT),2), round(np.std(all_psnr_CBCT),2)))
                print('MAE Soft tissue CBCT = {} ({})'.format(round(np.mean(all_mae_soft_tissue_CBCT),2), round(np.std(all_mae_soft_tissue_CBCT),2)))
                print('MAE Bone CBCT = {} ({})'.format(round(np.mean(all_mae_bone_CBCT),2), round(np.std(all_mae_bone_CBCT),2)))
                print('MAE body CBCT = {} ({})'.format(round(np.mean(all_mae_body_CBCT),2), round(np.std(all_mae_body_CBCT),2)))
                print('DSC Soft tissue of CBCT = {}({})'.format(round(np.mean(all_dsc_soft_tissue_CBCT),2), round(np.std(all_dsc_soft_tissue_CBCT),2)))
                print('DSC Bone of CBCT = {}({})'.format(round(np.mean(all_dsc_bone_CBCT),2), round(np.std(all_dsc_bone_CBCT),2)))
                print('DSC body of CBCT = {}({})'.format(round(np.mean(all_dsc_body_CBCT),2), round(np.std(all_dsc_body_CBCT),2)))

            if test_MR:
                print('Test MRI!!')
                with open(self.csv_file_s, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    metic = ['MAE', 'SSIM', 'PSNR',
                            'MAE Soft tissue', 'MAE Bone', 'MAE body',
                            'DSC Soft tissue', 'DSC Bone','DSC body']
                    
                    value_CBCT = ['{} + {}'.format(round(np.mean(all_mae_CBCT),2), round(np.std(all_mae_CBCT),2)),
                                  '{} + {}'.format(round(np.mean(all_ssim_CBCT),2), round(np.std(all_ssim_CBCT),2)),
                                  '{} + {}'.format(round(np.mean(all_psnr_CBCT),2), round(np.std(all_psnr_CBCT),2)),
                                  '{} + {}'.format(round(np.mean(all_mae_soft_tissue_CBCT),2), round(np.std(all_mae_soft_tissue_CBCT),2)),
                                  '{} + {}'.format(round(np.mean(all_mae_bone_CBCT),2), round(np.std(all_mae_bone_CBCT),2)),
                                  '{} + {}'.format(round(np.mean(all_mae_body_CBCT),2), round(np.std(all_mae_body_CBCT),2)),
                                  '{} + {}'.format(round(np.mean(all_dsc_soft_tissue_CBCT),2), round(np.std(all_dsc_soft_tissue_CBCT),2)),
                                  '{} + {}'.format(round(np.mean(all_dsc_bone_CBCT),2), round(np.std(all_dsc_bone_CBCT),2)),
                                  '{} + {}'.format(round(np.mean(all_dsc_body_CBCT),2), round(np.std(all_dsc_body_CBCT),2)),
                                  ]
                    
                    value_MRI = ['{} + {}'.format(round(np.mean(all_mae_MRI),2), round(np.std(all_mae_MRI),2)),
                                  '{} + {}'.format(round(np.mean(all_ssim_MRI),2), round(np.std(all_ssim_MRI),2)),
                                  '{} + {}'.format(round(np.mean(all_psnr_MRI),2), round(np.std(all_psnr_MRI),2)),
                                  '{} + {}'.format(round(np.mean(all_mae_soft_tissue_MRI),2), round(np.std(all_mae_soft_tissue_MRI),2)),
                                  '{} + {}'.format(round(np.mean(all_mae_bone_MRI),2), round(np.std(all_mae_bone_MRI),2)),
                                  '{} + {}'.format(round(np.mean(all_mae_body_MRI),2), round(np.std(all_mae_body_MRI),2)),
                                  '{} + {}'.format(round(np.mean(all_dsc_soft_tissue_MRI),2), round(np.std(all_dsc_soft_tissue_MRI),2)),
                                  '{} + {}'.format(round(np.mean(all_dsc_bone_MRI),2), round(np.std(all_dsc_bone_MRI),2)),
                                  '{} + {}'.format(round(np.mean(all_dsc_body_MRI),2), round(np.std(all_dsc_body_MRI),2)),
                                  ]
                    
                    #for i in range(len(metic)):
                        #writer.writerow([metic[i], value[i]])
                    writer.writerow(['', metic[0], metic[1], metic[2]])
                    writer.writerow(['CBCT', value_CBCT[0], value_CBCT[1], value_CBCT[2]])
                    writer.writerow(['MRI', value_MRI[0], value_MRI[1], value_MRI[2]])
                    writer.writerow(['',metic[3], metic[4], metic[5]])
                    writer.writerow(['CBCT',value_CBCT[3], value_CBCT[4], value_CBCT[5]])
                    writer.writerow(['MRI', value_MRI[3], value_MRI[4], value_MRI[5]])
                    writer.writerow(['',metic[6], metic[7], metic[8]])
                    writer.writerow(['CBCT',value_CBCT[6], value_CBCT[7], value_CBCT[8]])
                    writer.writerow(['MRI', value_MRI[6], value_MRI[7], value_MRI[8]])
            if test_CBCT:
                print('Test CBCT!!')
                with open(self.csv_file_s, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    metic = ['MAE', 'SSIM', 'PSNR',
                            'MAE Soft tissue', 'MAE Bone', 'MAE body',
                            'DSC Soft tissue', 'DSC Bone','DSC body']
                    
                    value_CBCT = ['{} + {}'.format(round(np.mean(all_mae_CBCT),2), round(np.std(all_mae_CBCT),2)),
                                  '{} + {}'.format(round(np.mean(all_ssim_CBCT),2), round(np.std(all_ssim_CBCT),2)),
                                  '{} + {}'.format(round(np.mean(all_psnr_CBCT),2), round(np.std(all_psnr_CBCT),2)),
                                  '{} + {}'.format(round(np.mean(all_mae_soft_tissue_CBCT),2), round(np.std(all_mae_soft_tissue_CBCT),2)),
                                  '{} + {}'.format(round(np.mean(all_mae_bone_CBCT),2), round(np.std(all_mae_bone_CBCT),2)),
                                  '{} + {}'.format(round(np.mean(all_mae_body_CBCT),2), round(np.std(all_mae_body_CBCT),2)),
                                  '{} + {}'.format(round(np.mean(all_dsc_soft_tissue_CBCT),2), round(np.std(all_dsc_soft_tissue_CBCT),2)),
                                  '{} + {}'.format(round(np.mean(all_dsc_bone_CBCT),2), round(np.std(all_dsc_bone_CBCT),2)),
                                  '{} + {}'.format(round(np.mean(all_dsc_body_CBCT),2), round(np.std(all_dsc_body_CBCT),2)),
                                  ]
                    
                    writer.writerow(['', metic[0], metic[1], metic[2]])
                    writer.writerow(['CBCT', value_CBCT[0], value_CBCT[1], value_CBCT[2]])
                    writer.writerow(['',metic[3], metic[4], metic[5]])
                    writer.writerow(['CBCT',value_CBCT[3], value_CBCT[4], value_CBCT[5]])
                    writer.writerow(['',metic[6], metic[7], metic[8]])
                    writer.writerow(['CBCT',value_CBCT[6], value_CBCT[7], value_CBCT[8]])

class HingeLoss(torch.nn.Module):

    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, output, target):

        hinge_loss = 1. - torch.mul(output, target)
        return torch.mean(F.relu(hinge_loss))