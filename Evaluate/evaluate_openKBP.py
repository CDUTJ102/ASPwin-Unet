import os
# import Evaluate.SSIM_3D as SSIM_3D
import SimpleITK as sitk
import numpy as np
import torch
from tqdm import tqdm
import re

"""
These codes are modified from https://github.com/ababier/open-kbp
"""


def Paddick_Conformity_Index(dose, mask, prescription):
    PTV = np.sum(mask > 0)
    PIV = np.sum(dose >= prescription)
    PTV_AND_PIV = np.sum((dose >= prescription) * mask > 0)
    PCI = PTV_AND_PIV ** 2 / (PTV * PIV)
    return PCI

def Homogeneity_Index(dose, mask):
    roi_dose = dose[mask > 0]
    D2 = np.percentile(roi_dose, 98)
    D98 = np.percentile(roi_dose, 2)
    D50 = np.percentile(roi_dose, 50)
    HI = (D2 - D98) / D50
    return HI

# def Gradient_Index(dose, prescription):
#     PIV = np.sum(dose >= prescription)
#     PIV_half = np.sum(dose >= (prescription / 2))
#     GI = PIV_half / PIV
#     return GI


def get_3D_Dose_dif(pred, gt, possible_dose_mask=None):
    if possible_dose_mask is not None:
        pred = pred[possible_dose_mask > 0]
        gt = gt[possible_dose_mask > 0]

    assert isinstance(pred, np.ndarray) or isinstance(pred, torch.Tensor), 'pred and gt must both be np.ndarray or ' \
                                                                           'torch.Tensor! '
    if isinstance(pred, np.ndarray):
        dif = np.mean(np.abs(pred - gt))
    else:
        dif = torch.mean(torch.abs(pred - gt))

    return dif


def get_DVH_metrics(_dose, _mask, mode, spacing=None):
    output = {}

    if mode == 'target':
        _roi_dose = _dose[_mask > 0]
        # D1
        output['D1'] = np.percentile(_roi_dose, 99)
        # D95
        output['D95'] = np.percentile(_roi_dose, 5)
        # D99
        output['D99'] = np.percentile(_roi_dose, 1)

    elif mode == 'OAR':
        if spacing is None:
            raise Exception('calculate OAR metrics need spacing')

        _roi_dose = _dose[_mask > 0]
        _roi_size = len(_roi_dose)
        _voxel_size = np.prod(spacing)
        voxels_in_tenth_of_cc = np.maximum(1, np.round(100 / _voxel_size))
        # D_0.1_cc
        fractional_volume_to_evaluate = 100 - voxels_in_tenth_of_cc / _roi_size * 100
        output['D_0.1_cc'] = np.percentile(_roi_dose, fractional_volume_to_evaluate)
        # Dmean
        output['Dmean'] = np.mean(_roi_dose)
    else:
        raise Exception('Unknown mode!')

    return output


def get_Dose_score_and_DVH_score(prediction_dir, gt_dir):
    list_dose_dif = []
    list_dose_struc = {}
    list_dose_struc['Brainstem'] = []
    list_dose_struc['SpinalCord'] = []
    list_dose_struc['RightParotid'] = []
    list_dose_struc['LeftParotid'] = []
    list_dose_struc['Esophagus'] = []
    list_dose_struc['Larynx'] = []
    list_dose_struc['Mandible'] = []
    list_dose_struc['PTV70'] = []
    list_dose_struc['PTV63'] = []
    list_dose_struc['PTV56'] = []
    list_DVH_dif = []
    list_SSIM_dif = []
    list_DVH_struc = {}
    list_DVH_struc['D_0.1_cc'] = []
    list_DVH_struc['Dmean'] = []
    list_DVH_struc['D1'] = []
    list_DVH_struc['D95'] = []
    list_DVH_struc['D99'] = []
    list_PCI_dif = []
    list_HI_dif = []

    list_patient_ids = tqdm(os.listdir(prediction_dir))  # (pt_242, pt_340)
    for patient_id in list_patient_ids:
        pred_nii = sitk.ReadImage(prediction_dir + '/' + patient_id + '/dose.nii.gz')
        pred = sitk.GetArrayFromImage(pred_nii)

        gt_nii = sitk.ReadImage(gt_dir + '/' + patient_id + '/dose.nii.gz')
        gt = sitk.GetArrayFromImage(gt_nii)

        # pred_ = torch.tensor(pred).unsqueeze(0).unsqueeze(0)
        # gt_ = torch.tensor(gt).unsqueeze(0).unsqueeze(0)
        # list_SSIM_dif.append(SSIM_3D.ssim3D(pred_, gt_).numpy())

        # Dose dif
        possible_dose_mask_nii = sitk.ReadImage(gt_dir + '/' + patient_id + '/possible_dose_mask.nii.gz')
        possible_dose_mask = sitk.GetArrayFromImage(possible_dose_mask_nii)
        list_dose_dif.append(get_3D_Dose_dif(pred, gt, possible_dose_mask))

        # DVH dif
        for structure_name in ['Brainstem',
                               'SpinalCord',
                               'RightParotid',
                               'LeftParotid',
                               'Esophagus',
                               'Larynx',
                               'Mandible',

                               'PTV70',
                               'PTV63',
                               'PTV56']:
            structure_file = gt_dir + '/' + patient_id + '/' + structure_name + '.nii.gz'

            # If the structure has been delineated
            if os.path.exists(structure_file):
                structure_nii = sitk.ReadImage(structure_file, sitk.sitkUInt8)
                structure = sitk.GetArrayFromImage(structure_nii)

                # Dose dif
                list_dose_struc[f'{structure_name}'].append(get_3D_Dose_dif(pred, gt, structure))

                spacing = structure_nii.GetSpacing()
                if structure_name.find('PTV') > -1:
                    mode = 'target'
                else:
                    mode = 'OAR'

                if structure_name == 'PTV70':
                    prescription = 70
                    if np.sum(pred >= prescription) > 0:
                        list_PCI_dif.append(Paddick_Conformity_Index(pred, structure, prescription))
                        # list_GI_dif.append(Gradient_Index(pred, prescription))
                    list_HI_dif.append(Homogeneity_Index(pred, structure))

                pred_DVH = get_DVH_metrics(pred, structure, mode=mode, spacing=spacing)
                gt_DVH = get_DVH_metrics(gt, structure, mode=mode, spacing=spacing)

                for metric in gt_DVH.keys():
                    # list_DVH_struc[f'{structure_name}'][f'{metric}'].append(abs(gt_DVH[metric] - pred_DVH[metric]))
                    list_DVH_struc[f'{metric}'].append(abs(gt_DVH[metric] - pred_DVH[metric]))
                    list_DVH_dif.append(abs(gt_DVH[metric] - pred_DVH[metric]))

    for key in list_DVH_struc.keys():
        # for metric in list_DVH_struc[key].keys():
        #     list_DVH_struc[key][metric] = np.mean(list_DVH_struc[key][metric])
        list_DVH_struc[key] = np.mean(list_DVH_struc[key])


    for key in list_dose_struc.keys():
        list_dose_struc[key] = np.mean(list_dose_struc[key])


    return np.mean(list_dose_dif), list_dose_struc, np.mean(list_DVH_dif), list_DVH_struc, np.mean(list_PCI_dif), np.mean(list_HI_dif)
