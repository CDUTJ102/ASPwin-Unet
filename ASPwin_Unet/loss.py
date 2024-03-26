import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt

class Loss(nn.Module):
    def __init__(self, loss_configs=None):
        super().__init__()

        if loss_configs is not None:
            if loss_configs['type'] == 'MAE':
                self.loss = nn.L1Loss(reduction='mean')
                print('Loss function: MAE')
            elif loss_configs['type'] == 'Huber':
                self.loss = nn.HuberLoss(reduction='mean',
                                         delta=loss_configs['delta'])  # equal to SmoothL1 when delta=1.0
                print('Loss function: Huber with delta {}'.format(loss_configs['delta']))
            elif loss_configs['type'] == 'MSE':
                self.loss = nn.MSELoss()
                print('Loss function: MSE')
        else:
            self.loss = nn.L1Loss(reduction='mean')  # MAE

        # contour loss
        self.sobel_x_filter = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, requires_grad=False).view(1, 1, 3, 3).to('cuda')
        self.sobel_y_filter = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, requires_grad=False).view(1, 1, 3, 3).to('cuda')
        self.loss_contour = nn.L1Loss(reduction='mean')

    def forward(self, contour, x, gt):
        pred_dose = x
        pd_contour = contour
        gt_dose = gt[0]
        possible_dose_mask = gt[1]

        gt_contour = self.apply_sobel_operator(gt_dose)

        pred_dose = pred_dose[possible_dose_mask > 0]
        pd_contour = pd_contour[possible_dose_mask > 0]
        gt_dose = gt_dose[possible_dose_mask > 0]
        gt_contour = gt_contour[possible_dose_mask > 0]

        pre_loss = self.loss(pred_dose, gt_dose)
        contour_loss = self.loss_contour(pd_contour, gt_contour)

        return pre_loss, contour_loss

    def apply_sobel_operator(self, input_image):
        input_image = rearrange(input_image, 'b c d h w -> (b d) c h w')
        contour_x = F.conv2d(input_image, self.sobel_x_filter, padding=1)
        contour_x = rearrange(contour_x, '(b d) c h w -> b c d h w', d=128)
        contour_y = F.conv2d(input_image, self.sobel_y_filter, padding=1)
        contour_y = rearrange(contour_y, '(b d) c h w -> b c d h w', d=128)

        input_image_shift = rearrange(input_image, '(b d) c h w -> (b h) c d w', d=128)
        contour_z = F.conv2d(input_image_shift, self.sobel_y_filter, padding=1)
        contour_z = rearrange(contour_z, '(b h) c d w -> b c d h w', h=128)

        return contour_x + contour_y + contour_z


