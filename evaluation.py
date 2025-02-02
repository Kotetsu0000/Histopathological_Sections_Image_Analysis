from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG, Logger

import argparse
import configparser
import os
import re

import cv2
import numpy as np
import pandas as pd
from VitLib import get_file_paths, create_directory, evaluate_membrane_prediction_range, evaluate_nuclear_prediction_range

class Evaluation:
    """評価を行うクラス

    Attributes:
        path_folder (str): 評価する画像が保存されているフォルダのパス
        ans_folder_path (str): 正解画像が保存されているフォルダのパス
        ans_list (list): 正解画像が保存されているフォルダのパスのリスト -> get_ans_img_folder_pathで取得
    """
    def __init__(self, path_folder:str, ans_folder_path:str, experiment_subject:str, experiment_param:dict):
        self.path_folder = path_folder
        self.ans_folder_path = ans_folder_path
        self.experiment_subject = experiment_subject
        self.experiment_param = experiment_param

        self.ans_list = get_ans_img_folder_path(ans_folder_path)

        self.logger = getLogger(__name__)
        handler = StreamHandler()
        handler.setLevel(DEBUG)
        self.logger.setLevel(DEBUG)
        self.logger.addHandler(handler)
        self.logger.propagate = False
        formatter = Formatter('%(asctime)s : %(levelname)7s - %(message)s')
        handler.setFormatter(formatter)
        file_handler = FileHandler(self.path_folder + '/log/exp.log', encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def sparse_evaluation(self):
        """sparse_evaluationを行う"""
        self.logger.info('Start Sparse Evaluation(疎探索評価)')
        sparse_evaluation = SparseEvaluation(self.path_folder, self.ans_list, self.experiment_subject, self.experiment_param, self.logger)
        sparse_evaluation.evaluate()

class SparseEvaluation:
    def __init__(self, path_folder:str, ans_list:list, experiment_subject:str, experiment_param:dict, logger:Logger):
        self.path_folder = path_folder
        self.ans_list = ans_list
        self.experiment_subject = experiment_subject
        self.experiment_param = experiment_param
        self.logger = logger

    def evaluate(self):
        """sparse_evaluationを行う"""
        path_list = self.get_png_flies_without_csv(self.path_folder)
        path_list_length = len(path_list)
        self.logger.info(f'length of path_list: {path_list_length}')
        for i, path in enumerate(path_list):
            self.evaluate_img(path)
            if i % 100 == 0:
                self.logger.info(f'Processing {i+1}/{path_list_length}')

    def evaluate_img(self, img_path:str):
        """画像単位の評価を行う"""
        self.logger.info(f'Processing {img_path}')
        if 'eval_data_membrane' in img_path:
            subject = 'membrane'
        elif 'eval_data_nuclear' in img_path:
            subject = 'nuclear'

        # 画像の情報を取得
        exp_num = self.get_int_number(r"exp(\d+)", img_path)
        val_num = self.get_int_number(r"(?:val|test)(\d+)", img_path)
        epoch_num = self.get_int_number(r"epoch(\d+)", img_path)
        pred_name = os.path.splitext(os.path.basename(img_path))[0]
        ans_path = self.select_ans_img_folder_path(pred_name, self.ans_list) + f'/y_{subject}/ans.png'

        # 画像の読み込み
        pred_img = imread(img_path)
        ans_img = imread(ans_path)

        assert pred_img is not None, f'Failed to read {img_path}'
        assert ans_img is not None, f'Failed to read {ans_path}'

        if subject == 'membrane':
            results = evaluate_membrane_prediction_range(
                pred_img, ans_img,
                radius=self.experiment_param['radius_eval'],
                min_th=self.experiment_param['membrane_sparse_threshold_min'],
                max_th=self.experiment_param['membrane_sparse_threshold_max'],
                step_th=self.experiment_param['membrane_sparse_threshold_step'],
                min_area=self.experiment_param['membrane_sparse_del_area_min'],
                max_area=self.experiment_param['membrane_sparse_del_area_max'],
                step_area=self.experiment_param['membrane_sparse_del_area_step'],
                symmetric=True,
                verbose=True,
            )
            result_dicts = []
            for result in results:
                threshold = result[0]
                del_area = result[1]
                precision = result[2]
                recall = result[3]
                fmeasure = result[4]
                membrane_length = result[5]
                tip_length = result[6]
                miss_length = result[7]
                result_dict = {
                    'exp_num': exp_num,
                    'val_num': val_num,
                    'epoch_num': epoch_num,
                    'img_name': pred_name,
                    'threshold': threshold,
                    'deleted_area': del_area,
                    'precision': precision,
                    'recall': recall,
                    'fmeasure': fmeasure,
                    'membrane_length': membrane_length,
                    'tip_length': tip_length,
                    'miss_length': miss_length,
                }
                result_dicts.append(result_dict)
        elif subject == 'nuclear':
            results = evaluate_nuclear_prediction_range(
                pred_img, ans_img,
                care_rate=self.experiment_param['care_rate'],
                lower_ratio=self.experiment_param['lower_ratio'],
                higher_ratio=self.experiment_param['higher_ratio'],
                min_th=self.experiment_param['nuclear_sparse_threshold_min'],
                max_th=self.experiment_param['nuclear_sparse_threshold_max'],
                step_th=self.experiment_param['nuclear_sparse_threshold_step'],
                min_area=self.experiment_param['nuclear_sparse_del_area_min'],
                max_area=self.experiment_param['nuclear_sparse_del_area_max'],
                step_area=self.experiment_param['nuclear_sparse_del_area_step'],
                eval_mode=self.experiment_param['nuclear_eval_mode'],
                distance=self.experiment_param['nuclear_eval_distance'],
                verbose=True,
            )
            result_dicts = []
            for result in results:
                threshold = result[0]
                del_area = result[1]
                precision = result[2]
                recall = result[3]
                fmeasure = result[4]
                correct_num = result[5]
                conformity_bottom = result[6]
                care_num = result[7]
                result_dict = {
                    'exp_num': exp_num,
                    'val_num': val_num,
                    'epoch_num': epoch_num,
                    'img_name': pred_name,
                    'threshold': threshold,
                    'deleted_area': del_area,
                    'precision': precision,
                    'recall': recall,
                    'fmeasure': fmeasure,
                    'correct_num': correct_num,
                    'conformity_bottom': conformity_bottom,
                    'care_num': care_num,
                }
                result_dicts.append(result_dict)

        csv_path = img_path.replace('.png', '_sparse.csv').replace('eval_data_membrane', 'log_eval_membrane').replace('eval_data_nuclear', 'log_eval_nuclear')
        folder_path = os.path.dirname(csv_path)
        create_directory(folder_path)
        
        df = pd.DataFrame(result_dicts)
        df.to_csv(csv_path, index=False)
        self.logger.info(f'Saved {csv_path}')

    def get_png_flies_without_csv(self, root_path:str) -> tuple:
        '''CSVのないPNGファイルを取得する。
            
            Args:
                root_path (str): モニタリングするフォルダのパス

            Returns:
                list: CSVのないPNGファイルのパスのリスト
        '''
        path_list = []
        for root, dirs, files in os.walk(root_path):
            for file in files:
                if file.endswith('.png') and 'train_data' not in root and 'test' not in root:
                    if not os.path.exists(os.path.join(root, file).replace('.png', '_sparse.csv').replace('eval_data_membrane', 'log_eval_membrane').replace('eval_data_nuclear', 'log_eval_nuclear')):
                        path_list.append(os.path.join(root, file).replace('\\', '/'))
                        if len(path_list) % 100 == 0:
                            self.logger.info(f'length of path_list: {len(path_list)}')
        return path_list

    def get_int_number(self, match_str:str, target:str) -> int:
        '''正規表現でマッチした文字列から数字を取得する。
        
        Args:
            match_str (str): 正規表現でマッチした文字列
            target (str): マッチしたい文字列
            
        Returns:
            int: マッチした数字
        '''
        match_ = re.search(match_str, target)
        assert match_ is not None, f'Not found {match_str} in {target}'
        return int(match_.group(1))

    def select_ans_img_folder_path(self, search_name:str, path_list:list) -> str:
        """指定した名前を含むパスを取得する。

        Args:
            search_name (str): 検索する名前
            path_list (list): 検索するパスのリスト

        Returns:
            str: 検索したパス
        """
        for path in path_list:
            if search_name in path:
                return path



def get_ans_img_folder_path(path:str) -> list:
    """指定したパス以下のansフォルダのパスを取得する。
    
    Args:
        path (str): 検索するパス

    Returns:
        list: ansフォルダのパスのリスト
    """
    divide_list = get_file_paths(path)
    img_folder_list = []
    for divide in divide_list:
        img_folder_list.extend(get_file_paths(divide))
    return img_folder_list

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    """画像を読み込む関数(日本語ファイル名に対応)
    
    Args:
        filename (str): ファイル名
        flags (int): cv2.imreadのflags
        dtype (numpy.dtype): cv2.imreadのdtype
    """
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

def imwrite(filename, img, params=None):
    """画像を保存する関数(日本語ファイル名に対応)

    Args:
        filename (str): ファイル名
        img (numpy.ndarray): 画像
        params (list): cv2.imwriteのparams
    """
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument('--config', '-c', type=str, default=None, help='config file path')
    args = arg.parse_args()

    if args.config is not None:
        perser = configparser.ConfigParser()
        perser.read(args.config, encoding='utf-8')
        EXPERIMENT_PATH = perser['EXPERIMENT_PATH']
        EXPERIMENT_PARAM = perser['EXPERIMENT_PARAM']
        EVALIATION_PARAM = perser['EVALIATION_PARAM']
    else:
        EXPERIMENT_PATH = {}
        EXPERIMENT_PARAM = {}
        EVALIATION_PARAM = {}

    # 結果の保存されているPATH
    path_folder = EXPERIMENT_PATH.get('default_path', './result')

    # ansフォルダのパスを取得
    ans_folder_path = EXPERIMENT_PATH.get('img_path', './Data/master_exp_data')

    # 実験対象
    experiment_subject = EXPERIMENT_PARAM.get('experiment_subject', 'membrane')

    # 評価パラメータ
    experiment_param = {
        'care_rate': float(EXPERIMENT_PATH.get('care_rate', 75)),
        'lower_ratio': float(EXPERIMENT_PATH.get('lower_ratio', 17)),
        'higher_ratio': float(EXPERIMENT_PATH.get('higher_ratio', 0)),
        'nuclear_eval_mode': EVALIATION_PARAM.get('nuclear_eval_mode', 'inclusion'),
        'nuclear_eval_distance': int(EVALIATION_PARAM.get('nuclear_eval_distance', 5)),
        'nuclear_sparse_epoch_start': int(EVALIATION_PARAM.get('nuclear_sparse_epoch_start', 1)),
        'nuclear_sparse_epoch_step': int(EVALIATION_PARAM.get('nuclear_sparse_epoch_step', 5)),
        'nuclear_sparse_threshold_min': int(EVALIATION_PARAM.get('nuclear_sparse_threshold_min', 127)),
        'nuclear_sparse_threshold_max': int(EVALIATION_PARAM.get('nuclear_sparse_threshold_max', 255)),
        'nuclear_sparse_threshold_step': int(EVALIATION_PARAM.get('nuclear_sparse_threshold_step', 1)),
        'nuclear_sparse_del_area_min': int(EVALIATION_PARAM.get('nuclear_sparse_del_area_min', 0)),
        'nuclear_sparse_del_area_max': int(EVALIATION_PARAM.get('nuclear_sparse_del_area_max', 0)) if EVALIATION_PARAM.get('nuclear_sparse_del_area_max', 'None') != 'None' else None,
        'nuclear_sparse_del_area_step': int(EVALIATION_PARAM.get('nuclear_sparse_del_area_step', 5)),

        'radius_eval': int(EVALIATION_PARAM.get('radius_eval', 3)),
        'membrane_sparse_epoch_start': int(EVALIATION_PARAM.get('membrane_sparse_epoch_start', 1)),
        'membrane_sparse_epoch_step': int(EVALIATION_PARAM.get('membrane_sparse_epoch_step', 5)),
        'membrane_sparse_threshold_min': int(EVALIATION_PARAM.get('membrane_sparse_threshold_min', 127)),
        'membrane_sparse_threshold_max': int(EVALIATION_PARAM.get('membrane_sparse_threshold_max', 255)),
        'membrane_sparse_threshold_step': int(EVALIATION_PARAM.get('membrane_sparse_threshold_step', 1)),
        'membrane_sparse_del_area_min': int(EVALIATION_PARAM.get('membrane_sparse_del_area_min', 0)),
        'membrane_sparse_del_area_max': int(EVALIATION_PARAM.get('membrane_sparse_del_area_max', 0)) if EVALIATION_PARAM.get('membrane_sparse_del_area_max', 'None') != 'None' else None,
        'membrane_sparse_del_area_step': int(EVALIATION_PARAM.get('membrane_sparse_del_area_step', 5)),
    }
    print(experiment_param)
