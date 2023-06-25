
import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
import glob
import scipy.io
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from random import randint
import random
import time
import re
from timm.models.layers import DropPath
from einops import rearrange
from scipy import ndimage
from skimage import io
from skimage import transform
from natsort import natsorted
from skimage.transform import rotate, AffineTransform
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    RandAffined,
    RandCropByLabelClassesd,
    SpatialPadd,
    RandAdjustContrastd,
    RandShiftIntensityd,
    ScaleIntensityd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    ScaleIntensityRangePercentilesd,
    Resized,
    Transposed,
    ResizeWithPadOrCropd
)
from monai.transforms import (CastToTyped,
                              Compose, CropForegroundd, EnsureChannelFirstd, LoadImaged,
                              NormalizeIntensity, RandCropByPosNegLabeld,
                              RandFlipd, RandGaussianNoised,
                              RandGaussianSmoothd, RandScaleIntensityd,
                              RandZoomd, SpatialCrop, SpatialPadd, EnsureTyped)
from monai.transforms.compose import MapTransform
from monai.config import print_config
from monai.metrics import DiceMetric
from skimage.transform import resize
import scipy.io
import matplotlib.pyplot as plt

import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F


from Create_diffusion import *
from resampler import *
from torchvision.utils import save_image




BATCH_SIZE_TRAIN = 8*1
image_size = 256
img_size = (image_size,image_size)
spacing = (1,1)
channels = 1


class CustomDataset(Dataset):
    def __init__(self,imgs_path):
        self.imgs_path = imgs_path

        file_list = natsorted(glob.glob(self.imgs_path + "*"), key=lambda y: y.lower())
        # print(file_list)
        self.data = []
        # self.loader = LoadImaged(keys= ['image','label'],reader='nibabelreader')
        # self.loader = LoadImaged(keys= ['image','label'],reader='PILReader')
        for img_path in file_list:
            class_name = img_path.split("/")[-1]
            self.data.append([img_path, class_name])
        self.train_transforms = Compose(
                [
                    LoadImaged(keys=["image"],reader='PILreader'),
                    AddChanneld(keys=["image"]),
                    # Orientationd(keys=["image"], axcodes="RAS"),
                    # Spacingd(
                    #     keys=["image"],
                    #     pixdim=spacing,
                    #     mode=("bilinear"),
                    # ),
                    ScaleIntensityd(keys=["image"], minv=0, maxv=1.0),
                    # ScaleIntensityRanged(
                    #     keys=["image"],
                    #     a_min=-0,
                    #     a_max=2500,
                    #     b_min=-1,
                    #     b_max=1.0,
                    #     clip=True,
                    # ),
                    ResizeWithPadOrCropd(
                        keys=["image"],
                        spatial_size=(256,256),
                        constant_values = 0,
                    ),
                    # Resized(
                    #     keys=["image"],
                    #     spatial_size=(64,64),
                    # ),
                    ToTensord(keys=["image"]),
                ]
            )
  
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx,):

        img_path, class_name = self.data[idx]
        cao = {"image":img_path}
        affined_data_dict = self.train_transforms(cao)                    
        img_tensor =affined_data_dict['image'].to(torch.float)
        
        return img_tensor



# MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
# DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --use_scale_shift_norm False"
# TRAIN_FLAGS="--lr 2e-5 --batch_size 128"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# num_channels=128
num_res_blocks=2
num_heads_upsample=-1

dropout=0
learn_sigma=True
sigma_small=False
class_cond=False
diffusion_steps=1000
noise_schedule='linear'
timestep_respacing=[50]
use_kl=False
predict_xstart=False
rescale_timesteps=True
rescale_learned_sigmas=True
use_checkpoint=False
use_scale_shift_norm=True
resblock_updown = False

diffusion = create_gaussian_diffusion(
    steps=diffusion_steps,
    learn_sigma=learn_sigma,
    sigma_small=sigma_small,
    noise_schedule=noise_schedule,
    use_kl=use_kl,
    predict_xstart=predict_xstart,
    rescale_timesteps=rescale_timesteps,
    rescale_learned_sigmas=rescale_learned_sigmas,
    timestep_respacing=timestep_respacing,
)
schedule_sampler = UniformSampler(diffusion) 


if image_size == 512:
    channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
elif image_size == 256:
    num_channels=128
    channel_mult = (1, 1, 2, 2, 4, 4)
    attention_resolutions="64,32,16,8"
    num_heads=[4,4,4,8,16,16]
    window_size = [[4,4],[4,4],[4,4],[8,8],[8,8],[4,4]]
    num_res_blocks = [2,2,1,1,1,1]
    sample_kernel=([2,2],[2,2],[2,2],[2,2],[2,2]),
elif image_size == 128:
    channel_mult = (1, 1, 2, 3, 4,)
    num_heads=[4,4,4,8,16]
    window_size = [[4,4],[4,4],[4,4],[8,8],[4,4]]
    num_res_blocks = [2,2,2,2,2]
    sample_kernel=([2,2],[2,2],[2,2],[2,2]),
elif image_size == 64:
    channel_mult = (1, 2, 3, 4, )
    num_heads=[4,4, 8,16]
    window_size = [[4,4],[4,4],[8,8],[4,4]]
    num_res_blocks = [2,2,2,2]
    sample_kernel=([2,2],[2,2],[2,2]),
class_cond = False
attention_ds = []
for res in attention_resolutions.split(","):
    attention_ds.append(int(res))
# from Diffusion_model_Unet import *
# model = UNetModel(
#         image_size=image_size,
#         in_channels=1,
#         model_channels=num_channels,
#         out_channels=2,
#         num_res_blocks=num_res_blocks[0],
#         attention_resolutions=tuple(attention_ds),
#         dropout=0.,
#         sample_kernel=sample_kernel,
#         channel_mult=channel_mult,
#         num_classes=(NUM_CLASSES if class_cond else None),
#         use_checkpoint=False,
#         use_fp16=False,
#         num_heads=4,
#         num_head_channels=64,
#         num_heads_upsample=-1,
#         use_scale_shift_norm=use_scale_shift_norm,
#         resblock_updown=False,
#         use_new_attention_order=False,
#     ).to(device)

from Diffusion_model_transformer import *
model = SwinVITModel(
        image_size=(image_size,image_size),
        in_channels=1,
        model_channels=num_channels,
        out_channels=2,
        sample_kernel=sample_kernel,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=0,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=False,
        use_fp16=False,
        num_heads=num_heads,
        window_size = window_size,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=False,
    ).to(device)


pytorch_total_params = sum(p.numel() for p in model.parameters())
print('parameter number is '+str(pytorch_total_params))
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5,weight_decay = 1e-4)
params = {'batch_size': BATCH_SIZE_TRAIN,
          'shuffle': True,
          'pin_memory': True,
          'drop_last': False}
training_set1 = CustomDataset('C:\Pan research\Diffusion model\ACDC_2D/train/')
train_loader1 = torch.utils.data.DataLoader(training_set1, **params)


def train(model, optimizer,data_loader1, loss_history):
    model.train()
    total_samples = len(data_loader1.dataset)
    loss_sum = []
    alpha = 0.4
    total_time = 0
    for i, x1 in enumerate(data_loader1):

        traindata = x1.to(device)
        t, weights = schedule_sampler.sample(traindata.shape[0], device)

        aa = time.time()
        
        optimizer.zero_grad()
        all_loss = diffusion.training_losses(model,traindata,t=t)
        loss = (all_loss["loss"] * weights).mean()
        loss.backward()
        loss_sum.append(loss.detach().cpu().numpy())
        optimizer.step()
        total_time += time.time()-aa
        if i % 30 == 0:
            print('optimization time: '+ str(time.time()-aa))
            print('[' +  '{:5}'.format(i * BATCH_SIZE_TRAIN) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader1)) + '%)]  Loss: ' +
                  '{:6.7f}'.format(np.nanmean(loss_sum)))
        if i % 150 == 0:
            data = {"img":traindata.cpu().numpy()}
            scipy.io.savemat(path+ 'train_example_epoch'+str(epoch)+'.mat',data)
        
    average_loss = np.nanmean(loss_sum) 
    loss_history.append(average_loss)
    print("Total time per sample is: "+str(total_time))
    print('Averaged loss is: '+ str(average_loss))
    return average_loss

         
def evaluate(model,epoch,path):
    model.eval()
    aa = time.time()
    # sampler = GaussianDiffusionSampler(
    #         model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device) 
    prediction = []
    true = []
    img = []
    loss_all = []
    with torch.no_grad():
        for ind in range(1):
            x_clean = diffusion.p_sample_loop(model,(10//1, 1, image_size, image_size),clip_denoised=True)
            # noisyImage = torch.randn(
            #     size=[BATCH_SIZE_TRAIN, 1, image_size, image_size], device=device)
            # x_clean = sampler(noisyImage)
            img.append(x_clean.cpu().numpy())
        data = {"img":img}
        print(str(time.time()-aa))
        scipy.io.savemat(path+ 'test_example_epoch'+str(epoch)+'.mat',data)
        # scipy.io.savemat(path+ 'final_test_play.mat',data)



N_EPOCHS = 550
path ="C:/Pan research/Diffusion model/result/ACDC/"
PATH = path+'ViTRes1.pt' # Use your own path
PATH1 = path+'ViTRes1.pt' # Use your own pathwaa
best_loss = 1
if not os.path.exists(path):
  os.makedirs(path) 
train_loss_history, test_loss_history = [], []
# model.load_state_dict(torch.load(PATH),strict=True) 
for epoch in range(0, N_EPOCHS):
    print('Epoch:', epoch)
    start_time = time.time() 
    average_loss = train(model, optimizer, train_loader1, train_loss_history)
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
    if epoch % 5 == 0:
        evaluate(model,epoch,path)
        # if average_loss < best_loss:
        print('Save the latest best model')
        torch.save(model.state_dict(), PATH1)
        # best_loss = average_loss
print('Execution time')