# -*- encoding: utf-8 -*-
import argparse
import os
import sys
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import torch
import yaml
from tqdm import tqdm

if os.path.abspath('..') not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))

from Evaluate.evaluate_openKBP import get_Dose_score_and_DVH_score
from model import DoseformerV2
from NetworkTrainer.network_trainer import NetworkTrainer


def load_config(path, config_name):
    with open(os.path.join(path, config_name)) as file:
        config = yaml.safe_load(file)
        # cfg = AttrDict(config)
        # print(cfg.project_name)

    return config


def read_data(patient_dir):
    dict_images = {}
    list_structures = ['CT',
                       'PTV70',
                       'PTV63',
                       'PTV56',
                       'possible_dose_mask',
                       'Brainstem',
                       'SpinalCord',
                       'RightParotid',
                       'LeftParotid',
                       'Esophagus',
                       'Larynx',
                       'Mandible']

    for structure_name in list_structures:
        structure_file = patient_dir + '/' + structure_name + '.nii.gz'

        if structure_name == 'CT':
            dtype = sitk.sitkInt16
        else:
            dtype = sitk.sitkUInt8

        if os.path.exists(structure_file):
            dict_images[structure_name] = sitk.ReadImage(structure_file, dtype)
            dict_images[structure_name] = sitk.GetArrayFromImage(dict_images[structure_name])[np.newaxis, :, :, :]
        else:
            dict_images[structure_name] = np.zeros((1, 128, 128, 128), np.uint8)

    return dict_images


def pre_processing(dict_images):
    # PTVs
    PTVs = 70.0 / 70. * dict_images['PTV70'] \
           + 63.0 / 70. * dict_images['PTV63'] \
           + 56.0 / 70. * dict_images['PTV56']

    # OARs
    list_OAR_names = ['Brainstem',
                      'SpinalCord',
                      'RightParotid',
                      'LeftParotid',
                      'Esophagus',
                      'Larynx',
                      'Mandible'
                      ]
    # OAR_all = np.concatenate([dict_images[OAR_name] for OAR_name in list_OAR_names], axis=0)
    OAR_all = np.zeros((1, 128, 128, 128), np.uint8)
    for OAR_i in range(7):
        OAR = dict_images[list_OAR_names[OAR_i]]
        OAR_all[OAR > 0] = OAR_i + 1

    # CT image
    CT = dict_images['CT']
    CT = np.clip(CT, a_min=-1024, a_max=1500)
    CT = CT.astype(np.float32) / 1000.

    # Possible mask
    possible_dose_mask = dict_images['possible_dose_mask']

    list_images = [np.concatenate((PTVs, OAR_all, CT), axis=0),  # Input
                   possible_dose_mask]
    return list_images


def copy_sitk_imageinfo(image1, image2):
    image2.SetSpacing(image1.GetSpacing())
    image2.SetDirection(image1.GetDirection())
    image2.SetOrigin(image1.GetOrigin())

    return image2


# Input is C*Z*H*W
def flip_3d(input_, list_axes):
    if 'Z' in list_axes:
        input_ = input_[:, ::-1, :, :]
    if 'W' in list_axes:
        input_ = input_[:, :, :, ::-1]

    return input_


def test_time_augmentation(trainer, input_, TTA_mode):
    list_prediction = []

    for list_flip_axes in TTA_mode:
        # Do Augmentation before forward
        augmented_input = flip_3d(input_.copy(), list_flip_axes)
        augmented_input = torch.from_numpy(augmented_input.astype(np.float32))
        augmented_input = augmented_input.unsqueeze(0).to(trainer.setting.device)
        contour, pred = trainer.setting.network(augmented_input)
        # pred = trainer.setting.network(augmented_input)

        # Aug back to original order
        pred_dose = pred
        pred_dose = flip_3d(np.array(pred_dose.cpu().data[0, :, :, :, :]), list_flip_axes)

        list_prediction.append(pred_dose[0, :, :, :])

    return np.mean(list_prediction, axis=0)


def inference(trainer, list_patient_dirs, save_path, do_TTA=True):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with torch.no_grad():
        trainer.setting.network.eval()
        for patient_dir in tqdm(list_patient_dirs):
            patient_id = patient_dir.split('/')[-1]

            dict_images = read_data(patient_dir)
            list_images = pre_processing(dict_images)

            input_ = list_images[0]
            possible_dose_mask = list_images[1]

            # Test-time augmentation
            if do_TTA:
                TTA_mode = [[], ['Z'], ['W'], ['Z', 'W']]
            else:
                TTA_mode = [[]]
            prediction = test_time_augmentation(trainer, input_, TTA_mode)

            # Pose-processing
            prediction[np.logical_or(possible_dose_mask[0, :, :, :] < 1, prediction < 0)] = 0
            prediction = 70. * prediction

            # Save prediction to nii image
            templete_nii = sitk.ReadImage(os.path.join(patient_dir, 'possible_dose_mask.nii.gz'))
            prediction_nii = sitk.GetImageFromArray(prediction)
            prediction_nii = copy_sitk_imageinfo(templete_nii, prediction_nii)
            if not os.path.exists(save_path + '/' + patient_id):
                os.mkdir(save_path + '/' + patient_id)
            sitk.WriteImage(prediction_nii, os.path.join(save_path, patient_id, 'dose.nii.gz'))


def main(configs):
    if not os.path.exists('../Data/OpenKBP_C3D'):
        raise Exception('OpenKBP_C3D should be prepared before testing, please run prepare_OpenKBP_C3D.py')

    trainer = NetworkTrainer(phase='test')
    trainer.setting.project_name = configs['project_name']
    trainer.setting.output_dir = configs['output_dir']

    trainer.setting.network = DoseformerV2(configs)

    # Load model weights
    trainer.init_trainer(ckpt_file=os.path.join(trainer.setting.output_dir, 'best_val_evaluation_index.pkl'),
                         list_GPU_ids=configs['list_GPU_ids'],
                         only_network=True)

    # Start inference
    print('\n\n# Start inference !')
    list_patient_dirs = ['../Data/OpenKBP_C3D/pt_' + str(i) for i in range(241, 341)]
    inference(trainer, list_patient_dirs, save_path=os.path.join(trainer.setting.output_dir, 'Prediction'),
              do_TTA=configs['testing']['TTA'])

    # Evaluation
    print('\n\n# Start evaluation !')
    Dose_score, Dose_score_list, DVH_score, DVH_score_list, PCI, HI = get_Dose_score_and_DVH_score(
        prediction_dir=os.path.join(trainer.setting.output_dir, 'Prediction'),
        gt_dir='../Data/OpenKBP_C3D')

    # trainer.print_log_to_file('\nSSIM is: {}\n'.format(SSIM), 'a')
    trainer.print_log_to_file('\nDose score is: {}\n'.format(Dose_score), 'a')
    trainer.print_log_to_file('DVH score is: {}\n'.format(DVH_score), 'a')
    # print('------------Dose_score------------')
    # print(f"Brainstem： {Dose_score_list['Brainstem']}")
    # print(f"SpinalCord： {Dose_score_list['SpinalCord']}")
    # print(f"RightParotid： {Dose_score_list['RightParotid']}")
    # print(f"LeftParotid： {Dose_score_list['LeftParotid']}")
    # print(f"Esophagus： {Dose_score_list['Esophagus']}")
    # print(f"Larynx： {Dose_score_list['Larynx']}")
    # print(f"Mandible： {Dose_score_list['Mandible']}")
    # print(f"PTV70： {Dose_score_list['PTV70']}")
    # print(f"PTV63： {Dose_score_list['PTV63']}")
    # print(f"PTV56： {Dose_score_list['PTV56']}")
    # print('------------DVH_score------------')
    # print(f"D_0.1_cc： {DVH_score_list['D_0.1_cc']}")
    # print(f"Dmean： {DVH_score_list['Dmean']}")
    # print(f"D1： {DVH_score_list['D1']}")
    # print(f"D95： {DVH_score_list['D95']}")
    # print(f"D99： {DVH_score_list['D99']}")
    # print('------------PCI------------')
    trainer.print_log_to_file('PCI is: {}\n'.format(PCI), 'a')
    # print('------------HI------------')
    trainer.print_log_to_file('HI is: {}\n'.format(HI), 'a')
    # print('------------GI------------')
    # print(f"PTV70： GI:{GI_list['PTV70']}")
    # print(f"PTV63： GI:{GI_list['PTV63']}")
    # print(f"PTV56： GI:{GI_list['PTV56']}")
    # print(f"GI: {GI}")



if __name__ == "__main__":
    CONFIG_PATH = '../Configs'
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default='default_config.yaml',
    #                     help='config name')
    args = parser.parse_args()

    cfgs = load_config(CONFIG_PATH, config_name='ASPwin_Unet.yaml')
    main(cfgs)
