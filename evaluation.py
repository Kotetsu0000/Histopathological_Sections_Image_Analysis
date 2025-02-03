from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG, Logger

import argparse
import configparser
import os
import re

import cv2
import numpy as np
import pandas as pd
from VitLib import get_file_paths, get_file_stems, create_directory, evaluate_membrane_prediction_range, evaluate_nuclear_prediction_range, evaluate_membrane_prediction, evaluate_nuclear_prediction

class Evaluation:
    """評価を行うクラス

    Attributes:
        path_folder (str): 評価する画像が保存されているフォルダのパス
        ans_folder_path (str): 正解画像が保存されているフォルダのパス
        ans_list (list): 正解画像が保存されているフォルダのパスのリスト -> get_ans_img_folder_pathで取得
    """
    def __init__(self, path_folder:str, ans_folder_path:str, experiment_param:dict):
        self.path_folder = path_folder
        self.ans_folder_path = ans_folder_path
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

        self.sparse_evaluation_end = False
        self.dense_evaluation_end = False

    def sparse_evaluation(self):
        """sparse_evaluationを行う"""
        self.logger.info('Start Sparse Evaluation(疎探索評価)')
        sparse_evaluation = SparseEvaluation(self.path_folder, self.ans_list, self.experiment_param, self.logger)
        sparse_evaluation.evaluate()
        self.sparse_evaluation_end = True

    def dense_evaluation(self):
        """dense_evaluationを行う"""
        assert self.sparse_evaluation_end, 'Sparse Evaluation is not finished'
        self.logger.info('Start Dense Evaluation(密探索評価)')
        dense_evaluation = DenseEvaluation(self.path_folder, self.ans_list, self.experiment_param, self.logger)
        dense_evaluation.evaluate()
        self.dense_evaluation_end = True

    def aggregation(self):
        """evaluationの結果を集計する"""
        assert self.dense_evaluation_end, 'Dense Evaluation is not finished'
        self.logger.info('Start Evaluation Aggregation')
        evaluation_aggregation = EvaluationAggregation(self.path_folder, self.ans_list, self.experiment_param, self.logger)
        evaluation_aggregation.aggregate_validation()
        evaluation_aggregation.aggregate_test()

class SparseEvaluation:
    def __init__(self, path_folder:str, ans_list:list, experiment_param:dict, logger:Logger):
        self.path_folder = path_folder
        self.ans_list = ans_list
        self.experiment_param = experiment_param
        self.logger = logger

    def evaluate(self):
        """sparse_evaluationを行う"""
        path_list = [path for path in self.get_png_flies_without_csv(self.path_folder) if self.is_evaluation_epoch(path)]
        path_list_length = len(path_list)
        self.logger.info(f'length of path_list: {path_list_length}')
        for i, path in enumerate(path_list):
            self.logger.info(f'Processing {i+1}/{path_list_length}')
            self.evaluate_img(path)
    
    def is_evaluation_epoch(self, img_path:str) -> bool:
        """評価を行うエポックかの判定を行う
        
        Args:
            img_path (str): 画像のパス

        Returns:
            bool: 評価を行うエポックかの判定
        """
        if 'eval_data_membrane' in img_path:
            subject = 'membrane'
        elif 'eval_data_nuclear' in img_path:
            subject = 'nuclear'

        epoch_num = self.get_int_number(r"epoch(\d+)", img_path)
        is_evaluation_epoch_list = list(range(self.experiment_param[f'{subject}_sparse_epoch_start'], self.experiment_param['num_epochs'], self.experiment_param[f'{subject}_sparse_epoch_step']))
        return epoch_num in is_evaluation_epoch_list

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
        pred_img = imread(img_path, cv2.IMREAD_GRAYSCALE)
        ans_img = imread(ans_path, cv2.IMREAD_GRAYSCALE)

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

class DenseEvaluation:
    def __init__(self, path_folder:str, ans_list:list, experiment_param:dict, logger:Logger):
        self.path_folder = path_folder
        self.ans_list = ans_list
        self.experiment_param = experiment_param
        self.logger = logger

    def evaluate(self):
        """dense_evaluationを行う"""
        path_list = self.get_png_flies_without_csv(self.path_folder)
        path_list_length = len(path_list)
        self.logger.info(f'length of path_list: {path_list_length}')
        for i, path in enumerate(path_list):
            self.logger.info(f'Processing {i+1}/{path_list_length}')
            self.evaluate_img(path)

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
        pred_img = imread(img_path, cv2.IMREAD_GRAYSCALE)
        ans_img = imread(ans_path, cv2.IMREAD_GRAYSCALE)

        assert pred_img is not None, f'Failed to read {img_path}'
        assert ans_img is not None, f'Failed to read {ans_path}'

        sparse_csv_path = img_path.replace('.png', '_sparse.csv').replace('eval_data_membrane', 'log_eval_membrane').replace('eval_data_nuclear', 'log_eval_nuclear')
        assert os.path.exists(sparse_csv_path), f'Not found {sparse_csv_path}'
        sparse_df = pd.read_csv(sparse_csv_path)

        best_fmeasure = sparse_df['fmeasure'].max()
        best_row = sparse_df[sparse_df['fmeasure'] == best_fmeasure]
        
        best_threshold_max = best_row['threshold'].max()
        best_threshold_min = best_row['threshold'].min()
        max_th = sparse_df[sparse_df['threshold'] > best_threshold_max]['threshold'].min()
        min_th = sparse_df[sparse_df['threshold'] < best_threshold_min]['threshold'].max()

        try:
            max_th = int(max_th)+1
        except:
            max_th = int(best_threshold_max)+1
        try:
            min_th = int(min_th)
        except:
            min_th = int(best_threshold_min)

        best_del_area_max = best_row['deleted_area'].max()
        best_del_area_min = best_row['deleted_area'].min()
        max_area = sparse_df[sparse_df['deleted_area'] > best_del_area_max]['deleted_area'].min()
        min_area = sparse_df[sparse_df['deleted_area'] < best_del_area_min]['deleted_area'].max()

        try:
            max_area = int(max_area)+1
        except:
            max_area = int(best_del_area_max)+1
        try:
            min_area = int(min_area)
        except:
            min_area = int(best_del_area_min)

        self.logger.info(f'Min Threshold: {min_th}, Max Threshold: {max_th}, Min Area: {min_area}, Max Area: {max_area}')

        if subject == 'membrane':
            results = evaluate_membrane_prediction_range(
                pred_img, ans_img,
                radius=self.experiment_param['radius_eval'],
                min_th=min_th,
                max_th=max_th,
                step_th=1,
                min_area=min_area,
                max_area=max_area,
                step_area=1,
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
                min_th=min_th,
                max_th=max_th,
                step_th=1,
                min_area=min_area,
                max_area=max_area,
                step_area=1,
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

        csv_path = img_path.replace('.png', '.csv').replace('eval_data_membrane', 'log_eval_membrane').replace('eval_data_nuclear', 'log_eval_nuclear')
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
                if file.endswith('.png') and 'train_data' not in root:
                    if (os.path.exists(os.path.join(root, file).replace('.png', '_sparse.csv').replace('eval_data_membrane', 'log_eval_membrane').replace('eval_data_nuclear', 'log_eval_nuclear')) and
                        not os.path.exists(os.path.join(root, file).replace('.png', '.csv').replace('eval_data_membrane', 'log_eval_membrane').replace('eval_data_nuclear', 'log_eval_nuclear')) ):
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

class EvaluationAggregation:
    def __init__(self, path_folder:str, ans_list:list, experiment_param:dict, logger:Logger):
        self.path_folder = path_folder
        self.ans_list = ans_list
        self.experiment_param = experiment_param
        self.logger = logger

        self.aggregate_validation_end = False

    def aggregate_validation(self):
        """validationの結果を集計する"""
        self.logger.info('Start Validation Aggregation')
        membrane_log_eval_folder_path = f'{self.path_folder}/log_eval_membrane'
        nuclear_log_eval_folder_path = f'{self.path_folder}/log_eval_nuclear'

        # 細胞膜の評価が行われている場合
        if os.path.exists(membrane_log_eval_folder_path):
            self.logger.info(f'{membrane_log_eval_folder_path} exists')
            
            # 既にテストデータの集計が行われている場合
            if os.path.exists(f'{membrane_log_eval_folder_path}/test/all_aggregate.csv'):
                self.logger.warning(f'{membrane_log_eval_folder_path}/test/all_aggregate.csv exists. So Test Aggregation has already been done.')
            # 既にバリデーションデータの集計が全て行われている場合
            elif all([os.path.exists(f'{exp}/exp_aggregate.csv') for exp in get_file_paths(f'{membrane_log_eval_folder_path}/val')]):
                self.logger.warning(f'All Validation Aggregation has already been done.')
            # バリデーションデータの集計を行う
            else:
                self.logger.info('Start Validation Aggregation for Membrane')
                for exp in get_file_paths(f'{membrane_log_eval_folder_path}/val'):
                    for val_num in get_file_paths(exp):
                        for epoch in get_file_paths(val_num):
                            self.logger.info(f'Processing {epoch}')
                            self.aggregate_validation_epoch(epoch, 'membrane')
                        self.aggregate_validation_val_num(val_num, 'membrane')
                    self.aggregate_validation_exp_num(exp, 'membrane')

        # 細胞核の評価が行われている場合
        if os.path.exists(nuclear_log_eval_folder_path):
            self.logger.info(f'{nuclear_log_eval_folder_path} exists')
            
            # 既にテストデータの集計が行われている場合
            if os.path.exists(f'{nuclear_log_eval_folder_path}/test/all_aggregate.csv'):
                self.logger.warning(f'{nuclear_log_eval_folder_path}/test/all_aggregate.csv exists. So Test Aggregation has already been done.')
            # 既にバリデーションデータの集計が全て行われている場合
            elif all([os.path.exists(f'{exp}/exp_aggregate.csv') for exp in get_file_paths(f'{nuclear_log_eval_folder_path}/val')]):
                self.logger.warning(f'All Validation Aggregation has already been done.')
            # バリデーションデータの集計を行う
            else:
                self.logger.info('Start Validation Aggregation for Nuclear')
                for exp in get_file_paths(f'{nuclear_log_eval_folder_path}/val'):
                    for val_num in get_file_paths(exp):
                        for epoch in get_file_paths(val_num):
                            self.logger.info(f'Processing {epoch}')
                            self.aggregate_validation_epoch(epoch, 'nuclear')
                        self.aggregate_validation_val_num(val_num, 'nuclear')
                    self.aggregate_validation_exp_num(exp, 'nuclear')

        self.aggregate_validation_end = True

    def aggregate_validation_epoch(self, epoch_path:str, subject:str):
        """epoch毎の結果を集計する。
    
        Args:
            epoch_path (str): epochのパス
            subject (str): 対象('membrane' or 'nuclear')
        """
        if '_aggregate' in epoch_path:
            return
        self.logger.info(f'Aggregate Validation Epoch for {subject} in {epoch_path}')
        csv_file_list = [i for i in get_file_paths(epoch_path) if os.path.isfile(i) and i.endswith('.csv') and not i.endswith('_sparse.csv') and not i.endswith('_aggregate.csv')]
        if len(csv_file_list) == 0:
            raise Exception(f'Not found csv files: {epoch_path}')
        
        result_dicts = []
        for csv_file in csv_file_list:
            df = pd.read_csv(csv_file)

            exp_num = int(df['exp_num'].unique()[0])
            val_num = int(df['val_num'].unique()[0])
            epoch_num = int(df['epoch_num'].unique()[0])
            img_name = df['img_name'].unique()[0]

            self.logger.info(f'Agregate Exp: {exp_num}, Val: {val_num}, Epoch: {epoch_num}, Img: {img_name}')
            fmeasure_max = df['fmeasure'].max()
            fmeasure_max_df = df[df['fmeasure'] == fmeasure_max]

            fmeasure = fmeasure_max_df['fmeasure'].mean()
            precision = fmeasure_max_df['precision'].mean()
            recall = fmeasure_max_df['recall'].mean()

            threshold = int(fmeasure_max_df['threshold'].mean() + 0.5)
            deleted_area = int(fmeasure_max_df['deleted_area'].mean() + 0.5)
            if subject == 'membrane':
                membrane_length = int(fmeasure_max_df['membrane_length'].mean() + 0.5)
                tip_length = int(fmeasure_max_df['tip_length'].mean() + 0.5)
                miss_length = int(fmeasure_max_df['miss_length'].mean() + 0.5)

                self.logger.info(f'Max F-Measure: {fmeasure*100:.2f}%, Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, Threshold: {threshold}, Deleted Area: {deleted_area}, Membrane Length: {membrane_length}, Tip Length: {tip_length}, Miss Length: {miss_length}')
                
                result_dict = {
                    'exp_num': exp_num,
                    'val_num': val_num,
                    'epoch_num': epoch_num,
                    'img_name': img_name,
                    'threshold': threshold,
                    'deleted_area': deleted_area,
                    'precision': precision,
                    'recall': recall,
                    'fmeasure': fmeasure,
                    'membrane_length': membrane_length,
                    'tip_length': tip_length,
                    'miss_length': miss_length,
                }
            elif subject == 'nuclear':
                correct_num = int(fmeasure_max_df['correct_num'].mean() + 0.5)
                conformity_bottom = int(fmeasure_max_df['conformity_bottom'].mean() + 0.5)
                care_num = int(fmeasure_max_df['care_num'].mean() + 0.5)
                self.logger.info(f'Max F-Measure: {fmeasure*100:.2f}%, Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, Threshold: {threshold}, Deleted Area: {deleted_area}, Correct Num: {correct_num}, Conformity Bottom: {conformity_bottom}, Care Num: {care_num}')
                result_dict = {
                    'exp_num': exp_num,
                    'val_num': val_num,
                    'epoch_num': epoch_num,
                    'img_name': img_name,
                    'threshold': threshold,
                    'deleted_area': deleted_area,
                    'precision': precision,
                    'recall': recall,
                    'fmeasure': fmeasure,
                    'correct_num': correct_num,
                    'conformity_bottom': conformity_bottom,
                    'care_num': care_num,
                }
            result_dicts.append(result_dict)
        result_df = pd.DataFrame(result_dicts)
        result_df.to_csv(epoch_path + '/epoch_aggregate.csv', index=False)
            
    def aggregate_validation_val_num(self, val_num_path:str, subject:str):
        """val_num毎の結果を集計する。

        Args:
            val_num_path (str): val_numのパス
            subject (str): 対象('membrane' or 'nuclear')
        """
        if '_aggregate' in val_num_path:
            return
        self.logger.info(f'Aggregate Validation Val Num for {subject} in {val_num_path}')
        csv_file_list = [f'{i}/epoch_aggregate.csv' for i in get_file_paths(val_num_path) if os.path.isfile(f'{i}/epoch_aggregate.csv')]
        if len(csv_file_list) == 0:
            raise Exception(f'Not found csv files: {val_num_path}')
        
        result_dicts = []
        for csv_file in csv_file_list:
            df = pd.read_csv(csv_file)

            exp_num = int(df['exp_num'].unique()[0])
            val_num = int(df['val_num'].unique()[0])
            epoch_num = int(df['epoch_num'].unique()[0])

            self.logger.info(f'Agregate Exp: {exp_num}, Val: {val_num}, Epoch: {epoch_num}')
            
            fmeasure = df['fmeasure'].mean()
            precision = df['precision'].mean()
            recall = df['recall'].mean()

            threshold = int(df['threshold'].mean() + 0.5)
            deleted_area = int(df['deleted_area'].mean() + 0.5)

            if subject == 'membrane':
                membrane_length = int(df['membrane_length'].mean() + 0.5)
                tip_length = int(df['tip_length'].mean() + 0.5)
                miss_length = int(df['miss_length'].mean() + 0.5)

                self.logger.info(f'F-Measure: {fmeasure*100:.2f}%, Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, Threshold: {threshold}, Deleted Area: {deleted_area}, Membrane Length: {membrane_length}, Tip Length: {tip_length}, Miss Length: {miss_length}')
                result_dict = {
                    'exp_num': exp_num,
                    'val_num': val_num,
                    'epoch_num': epoch_num,
                    'threshold': threshold,
                    'deleted_area': deleted_area,
                    'precision': precision,
                    'recall': recall,
                    'fmeasure': fmeasure,
                    'membrane_length': membrane_length,
                    'tip_length': tip_length,
                    'miss_length': miss_length,
                }
            elif subject == 'nuclear':
                correct_num = int(df['correct_num'].mean() + 0.5)
                conformity_bottom = int(df['conformity_bottom'].mean() + 0.5)
                care_num = int(df['care_num'].mean() + 0.5)

                self.logger.info(f'F-Measure: {fmeasure*100:.2f}%, Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, Threshold: {threshold}, Deleted Area: {deleted_area}, Correct Num: {correct_num}, Conformity Bottom: {conformity_bottom}, Care Num: {care_num}')

                result_dict = {
                    'exp_num': exp_num,
                    'val_num': val_num,
                    'epoch_num': epoch_num,
                    'threshold': threshold,
                    'deleted_area': deleted_area,
                    'precision': precision,
                    'recall': recall,
                    'fmeasure': fmeasure,
                    'correct_num': correct_num,
                    'conformity_bottom': conformity_bottom,
                    'care_num': care_num,
                }
            
            result_dicts.append(result_dict)

        result_df = pd.DataFrame(result_dicts)
        result_df.to_csv(val_num_path + '/val_aggregate.csv', index=False)

    def aggregate_validation_exp_num(self, exp_num_path:str, subject:str):
        """撮像法の組み合わせ毎の結果を集計する。

        Args:
            exp_num_path (str): exp_numのパス
            subject (str): 対象('membrane' or 'nuclear')
        """
        if '_aggregate' in exp_num_path:
            return
        self.logger.info(f'Aggregate Validation Exp Num for {subject} in {exp_num_path}')
        csv_file_list = [f'{i}/val_aggregate.csv' for i in get_file_paths(exp_num_path) if os.path.isfile(f'{i}/val_aggregate.csv')]
        if len(csv_file_list) == 0:
            raise Exception(f'Not found csv files: {exp_num_path}')
        
        result_dicts = []
        for csv_file in csv_file_list:
            df = pd.read_csv(csv_file)

            exp_num = int(df['exp_num'].unique()[0])
            val_num = int(df['val_num'].unique()[0])

            self.logger.info(f'Agregate Exp: {exp_num}, Val: {val_num}')
            
            fmeasure_max = df['fmeasure'].max()
            fmeasure_max_df = df[df['fmeasure'] == fmeasure_max]

            epoch_num = int(fmeasure_max_df['epoch_num'].mean() + 0.5)

            fmeasure = fmeasure_max_df['fmeasure'].mean()
            precision = fmeasure_max_df['precision'].mean()
            recall = fmeasure_max_df['recall'].mean()

            threshold = int(fmeasure_max_df['threshold'].mean() + 0.5)
            deleted_area = int(fmeasure_max_df['deleted_area'].mean() + 0.5)

            if subject == 'membrane':
                membrane_length = int(fmeasure_max_df['membrane_length'].mean() + 0.5)
                tip_length = int(fmeasure_max_df['tip_length'].mean() + 0.5)
                miss_length = int(fmeasure_max_df['miss_length'].mean() + 0.5)

                self.logger.info(f'Max F-Measure: {fmeasure*100:.2f}%, Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, Threshold: {threshold}, Deleted Area: {deleted_area}, Membrane Length: {membrane_length}, Tip Length: {tip_length}, Miss Length: {miss_length}')

                result_dict = {
                    'exp_num': exp_num,
                    'val_num': val_num,
                    'epoch_num': epoch_num,
                    'threshold': threshold,
                    'deleted_area': deleted_area,
                    'precision': precision,
                    'recall': recall,
                    'fmeasure': fmeasure,
                    'membrane_length': membrane_length,
                    'tip_length': tip_length,
                    'miss_length': miss_length,
                }
            elif subject == 'nuclear':
                correct_num = int(fmeasure_max_df['correct_num'].mean() + 0.5)
                conformity_bottom = int(fmeasure_max_df['conformity_bottom'].mean() + 0.5)
                care_num = int(fmeasure_max_df['care_num'].mean() + 0.5)

                self.logger.info(f'Max F-Measure: {fmeasure*100:.2f}%, Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, Threshold: {threshold}, Deleted Area: {deleted_area}, Correct Num: {correct_num}, Conformity Bottom: {conformity_bottom}, Care Num: {care_num}')

                result_dict = {
                    'exp_num': exp_num,
                    'val_num': val_num,
                    'epoch_num': epoch_num,
                    'threshold': threshold,
                    'deleted_area': deleted_area,
                    'precision': precision,
                    'recall': recall,
                    'fmeasure': fmeasure,
                    'correct_num': correct_num,
                    'conformity_bottom': conformity_bottom,
                    'care_num': care_num,
                }
            result_dicts.append(result_dict)

        result_df = pd.DataFrame(result_dicts)
        result_df.to_csv(exp_num_path + '/exp_aggregate.csv', index=False)


    def aggregate_test(self):
        """testの結果を集計する"""
        assert self.aggregate_validation_end, 'Validation Aggregation has not been done yet.'
        self.logger.info('Start Test Aggregation')
        membrane_log_eval_folder_path = f'{self.path_folder}/log_eval_membrane'
        nuclear_log_eval_folder_path = f'{self.path_folder}/log_eval_nuclear'

        # 細胞膜の評価が行われている場合
        if os.path.exists(membrane_log_eval_folder_path):
            # テストデータの集計が行われていないものが存在する場合
            if not all([os.path.exists(f'{exp}/exp_aggregate.csv') for exp in get_file_paths(f'{membrane_log_eval_folder_path}/val')]):
                self.logger.warning('Not all Validation Aggregation has been done. So Test Aggregation is skipped.')
                raise Exception('Not all Validation Aggregation has been done. So Test Aggregation is skipped.')
            # テストデータの集計が行われている場合
            elif os.path.exists(f'{membrane_log_eval_folder_path}/test/all_aggregate.csv'):
                self.logger.warning(f'{membrane_log_eval_folder_path}/test/all_aggregate.csv exists. So Test Aggregation has already been done.')
            # テストデータの集計を行う
            else:
                for exp_val in [i for i in get_file_paths(f'{membrane_log_eval_folder_path}/val') if os.path.isdir(i)]:
                    exp = exp_val.replace('/val/', '/test/')
                    val_exp_aggregate_csv_path = f'{exp_val}/exp_aggregate.csv'
                    # バリデーションデータの集計が行われている場合
                    if os.path.isfile(val_exp_aggregate_csv_path):
                        self.logger.info(f'Start Test Aggregation for Membrane in {exp}')
                        self.aggregate_test_exp_num(exp, val_exp_aggregate_csv_path, 'membrane')
                    else:
                        self.logger.warning(f'Not found {val_exp_aggregate_csv_path}. So Test Aggregation is skipped.')
                        return
                self.aggregate_test_all(f'{membrane_log_eval_folder_path}/test', 'membrane')
                    
        # 細胞核の評価が行われている場合
        if os.path.exists(nuclear_log_eval_folder_path):
            # テストデータの集計が行われていないものが存在する場合
            if not all([os.path.exists(f'{exp}/exp_aggregate.csv') for exp in get_file_paths(f'{nuclear_log_eval_folder_path}/val')]):
                self.logger.warning('Not all Validation Aggregation has been done. So Test Aggregation is skipped.')
                raise Exception('Not all Validation Aggregation has been done. So Test Aggregation is skipped.')
            # テストデータの集計が行われている場合
            elif os.path.exists(f'{nuclear_log_eval_folder_path}/test/all_aggregate.csv'):
                self.logger.warning(f'{nuclear_log_eval_folder_path}/test/all_aggregate.csv exists. So Test Aggregation has already been done.')
            # テストデータの集計を行う
            else:
                for exp_val in [i for i in get_file_paths(f'{nuclear_log_eval_folder_path}/val') if os.path.isdir(i)]:
                    exp = exp_val.replace('/val/', '/test/')
                    val_exp_aggregate_csv_path = f'{exp_val}/exp_aggregate.csv'
                    # バリデーションデータの集計が行われている場合
                    if os.path.isfile(val_exp_aggregate_csv_path):
                        self.logger.info(f'Start Test Aggregation for Nuclear in {exp}')
                        self.aggregate_test_exp_num(exp, val_exp_aggregate_csv_path, 'nuclear')
                    else:
                        self.logger.warning(f'Not found {val_exp_aggregate_csv_path}. So Test Aggregation is skipped.')
                        return
                self.aggregate_test_all(f'{nuclear_log_eval_folder_path}/test', 'nuclear')
                    
    def aggregate_test_exp_num(self, exp_num_path:str, val_exp_aggregate_csv_path:str, subject:str):
        """撮像法の組み合わせ毎の結果を集計する。
        
        Args:
            exp_num_path (str): exp_numのパス
            val_exp_aggregate_csv_path (str): バリデーションデータの集計結果のパス
            subject (str): 対象('membrane' or 'nuclear')
        """
        if '_aggregate' in exp_num_path:
            return
        self.logger.info(f'Aggregate Test Exp Num for {subject} in {exp_num_path}')
        val_exp_aggregate_df = pd.read_csv(val_exp_aggregate_csv_path)
        eval_mode = self.experiment_param['nuclear_eval_mode'] # nuclearの場合のみ使用

        result_dicts = []
        for index, row in val_exp_aggregate_df.iterrows():
            exp_num = int(row['exp_num'])
            val_num = int(row['val_num'])
            epoch_num = int(row['epoch_num'])
            threshold = int(row['threshold'])
            deleted_area = int(row['deleted_area'])

            fmeasures = []
            precisions = []
            recalls = []
            
            if subject == 'membrane':
                membrane_lengths = []
                tip_lengths = []
                miss_lengths = []

                eval_folder_path = exp_num_path.replace('/log_eval_membrane/', '/eval_data_membrane/')
                img_folder_path = f'{eval_folder_path}/test{val_num:02d}/epoch{epoch_num:02d}'
                self.logger.info(f'Processing {img_folder_path}')
                img_length = len(get_file_paths(img_folder_path))
                for i, (img_path , img_stem) in enumerate(zip(get_file_paths(img_folder_path), get_file_stems(img_folder_path))):
                    self.logger.info(f'{i+1}/{img_length} {img_stem}, {index}')
                    ans_path = f'{self.select_ans_img_folder_path(img_stem, self.ans_list)}/y_membrane/ans.png'

                    pred_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    ans_img = cv2.imread(ans_path, cv2.IMREAD_GRAYSCALE)
                    result = evaluate_membrane_prediction(pred_img, ans_img, threshold, deleted_area)
                    fmeasures.append(result['fmeasure'])
                    precisions.append(result['precision'])
                    recalls.append(result['recall'])
                    membrane_lengths.append(result['membrane_length'])
                    tip_lengths.append(result['tip_length'])
                    miss_lengths.append(result['miss_length'])
                precision = sum(precisions) / len(precisions)
                recall = sum(recalls) / len(recalls)
                fmeasure = sum(fmeasures) / len(fmeasures)
                membrane_length = sum(membrane_lengths) / len(membrane_lengths)
                tip_length = sum(tip_lengths) / len(tip_lengths)
                miss_length = sum(miss_lengths) / len(miss_lengths)
                result_dict = {
                    'exp_num': exp_num,
                    'val_num': val_num,
                    'epoch_num': epoch_num,
                    'threshold': threshold,
                    'deleted_area': deleted_area,
                    'precision': precision,
                    'recall': recall,
                    'fmeasure': fmeasure,
                    'membrane_length': int(membrane_length + 0.5),
                    'tip_length': int(tip_length + 0.5),
                    'miss_length': int(miss_length + 0.5),
                }
                result_dicts.append(result_dict)
            elif subject == 'nuclear':
                correct_nums = []
                conformity_bottoms = []
                care_nums = []

                eval_folder_path = exp_num_path.replace('/log_eval_nuclear/', '/eval_data_nuclear/')
                img_folder_path = f'{eval_folder_path}/test{val_num:02d}/epoch{epoch_num:02d}'
                self.logger.info(f'Processing {img_folder_path}')
                img_length = len(get_file_paths(img_folder_path))
                for i, (img_path , img_stem) in enumerate(zip(get_file_paths(img_folder_path), get_file_stems(img_folder_path))):
                    self.logger.info(f'{i+1}/{img_length} {img_stem}, {index}')
                    ans_path = f'{self.select_ans_img_folder_path(img_stem, self.ans_list)}/y_nuclear/ans.png'

                    pred_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    ans_img = cv2.imread(ans_path, cv2.IMREAD_GRAYSCALE)
                    result = evaluate_nuclear_prediction(pred_img, ans_img, threshold=threshold, del_area=deleted_area, eval_mode=eval_mode)
                    fmeasures.append(result['fmeasure'])
                    precisions.append(result['precision'])
                    recalls.append(result['recall'])
                    correct_nums.append(result['correct_num'])
                    conformity_bottoms.append(result['conformity_bottom'])
                    care_nums.append(result['care_num'])
                precision = sum(precisions) / len(precisions)
                recall = sum(recalls) / len(recalls)
                fmeasure = sum(fmeasures) / len(fmeasures)
                correct_num = sum(correct_nums) / len(correct_nums)
                conformity_bottom = sum(conformity_bottoms) / len(conformity_bottoms)
                care_num = sum(care_nums) / len(care_nums)
                result_dict = {
                    'exp_num': exp_num,
                    'val_num': val_num,
                    'epoch_num': epoch_num,
                    'threshold': threshold,
                    'deleted_area': deleted_area,
                    'precision': precision,
                    'recall': recall,
                    'fmeasure': fmeasure,
                    'correct_num': int(correct_num + 0.5),
                    'conformity_bottom': int(conformity_bottom + 0.5),
                    'care_num': int(care_num + 0.5),
                }
                result_dicts.append(result_dict)

        result_df = pd.DataFrame(result_dicts)
        create_directory(exp_num_path)
        result_df.to_csv(exp_num_path + '/test_aggregate.csv', index=False)

    def aggregate_test_all(self, test_path:str, subject:str):
        """全てのテストデータを集計する
        
        Args:
            test_path (str): テストデータのパス
            subject (str): 対象('membrane' or 'nuclear')
        """
        self.logger.info(f'Aggregate Test All for {subject} in {test_path}')

        result_dicts = []
        for exp in get_file_paths(test_path):
            df = pd.read_csv(f'{exp}/test_aggregate.csv')
            exp_num = int(df['exp_num'].unique()[0])
            epoch_num = df['epoch_num'].mean()
            threshold = df['threshold'].mean()
            deleted_area = df['deleted_area'].mean()

            fmeasure = df['fmeasure'].mean()
            precision = df['precision'].mean()
            recall = df['recall'].mean()

            if subject == 'membrane':
                membrane_length = df['membrane_length'].mean()
                tip_length = df['tip_length'].mean()
                miss_length = df['miss_length'].mean()
                result_dict = {
                    'exp_num': exp_num,
                    'epoch_num': int(epoch_num + 0.5),
                    'threshold': int(threshold + 0.5),
                    'deleted_area': int(deleted_area + 0.5),
                    'precision': precision,
                    'recall': recall,
                    'fmeasure': fmeasure,
                    'membrane_length': int(membrane_length + 0.5),
                    'tip_length': int(tip_length + 0.5),
                    'miss_length': int(miss_length + 0.5),
                }
            elif subject == 'nuclear':
                correct_num = df['correct_num'].mean()
                conformity_bottom = df['conformity_bottom'].mean()
                care_num = df['care_num'].mean()
                result_dict = {
                    'exp_num': exp_num,
                    'epoch_num': int(epoch_num + 0.5),
                    'threshold': int(threshold + 0.5),
                    'deleted_area': int(deleted_area + 0.5),
                    'precision': precision,
                    'recall': recall,
                    'fmeasure': fmeasure,
                    'correct_num': int(correct_num + 0.5),
                    'conformity_bottom': int(conformity_bottom + 0.5),
                    'care_num': int(care_num + 0.5),
                }
            result_dicts.append(result_dict)

        result_df = pd.DataFrame(result_dicts)
        create_directory(test_path)
        result_df.to_csv(test_path + '/all_aggregate.csv', index=False)

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
        'num_epochs': int(EXPERIMENT_PARAM.get('num_epochs', 40)),
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
    
    evaluation = Evaluation(path_folder, ans_folder_path, experiment_param)
    evaluation.sparse_evaluation()
    evaluation.dense_evaluation()
    evaluation.aggregation()
