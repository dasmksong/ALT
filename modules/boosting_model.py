import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from modules.common.utils.decorator import TryDecorator
from modules.common.utils.file_handler import chk_and_make_dir
from modules.config import Config

model_logger = logging.getLogger("model")


class Boosting:
    """confusionmatrix class

    Attributes:
        _input_data (pd.DataFrame) : input data 전체
        _train_input (np.ndarray) : train_input
        _train_target (np.ndarray) : train_target
        _test_input (np.ndarray) : test_input
        _test_target (np.ndarray) : test_target
        _model (keras.models.Model) : 사용할 모델
        _config (dict) : 세부 설정 dictionary
        _model_path (str) : 모델 path



    """

    _input_data: pd.DataFrame
    _train_input: np.ndarray
    _train_target: np.ndarray
    _test_input: np.ndarray
    _test_target: np.ndarray
    __pred_array: np.ndarray
    _dtest: xgb.DMatrix
    _model: xgb
    _ols_model_y: str
    _trans_proteinuria: pd.DataFrame
    _config: dict
    _model_path: str

    def __init__(self, preprocessed_data: pd.DataFrame) -> None:
        """Model class init

        Args:
            self : self
            _preprocessed_data (DataFrame) : 전처리된 데이터
            _model_logger(Logger.manager.getLogger(name)) : "Model"이름을 가진 로거

        """
        self._model_logger = logging.getLogger("Model")
        self._preprocessed_data = preprocessed_data
        self._config = Config.instance().config
        self._ols_model_y = self._config["target"]["ols"]

    @TryDecorator(logger=model_logger)
    def _split_data(self) -> None:
        """X_train,X_test,y_train,y_test 데이터를 나눠줌
        이후 각 class 변수에 할당
        """

        (
            self._train_input,
            self._test_input,
            self._train_target,
            self._test_target,
        ) = train_test_split(
            self._trans_proteinuria.drop(columns=["요단백"]),
            self._trans_proteinuria["요단백"],
            test_size=0.3,
            stratify=self._trans_proteinuria["요단백"],
            random_state=17,
        )

        model_logger.info("Model Data Count-------------------------------------")
        model_logger.info(
            "preprocessed dataset  : " + str(len(self._preprocessed_data))
        )
        model_logger.info("train dataset      : " + str(len(self._train_input)))
        model_logger.info("test dataset       : " + str(len(self._test_input)))
        model_logger.info("-----------------------------------------------------")

    def _do_trans_proteinuria(self):
        data = self._preprocessed_data.copy()
        data.loc[data["요단백"] == 0, "요단백"] = 0
        data.loc[data["요단백"] != 0, "요단백"] = 1
        self._trans_proteinuria = data

    def _build_model(self) -> None:
        """Sequential 모델 빌드 함수
        이 함수에서 모델 수정 가능
        """

        dtrain = xgb.DMatrix(data=self._train_input, label=self._train_target)
        dtest = xgb.DMatrix(data=self._test_input, label=self._test_target)
        wlist = [(dtrain, "train"), (dtest, "eval")]
        self._dtest = dtest

        params = self._config["boosting"]
        print(params)
        boosting_model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self._config["boosting_round"]["num"],
            evals=wlist,
        )
        self._model = boosting_model

    def predict(self, input_data) -> list:
        """모델 predict 함수

        Args :
            input_data (Any) : 입력 데이터

        return :
            list : 모델이 입력 데이터를 기반으로 예측한 데이터
        """
        pred = self._model.predict(input_data)
        pred = [1 if x > 0.5 else 0 for x in pred]
        self._pred_array = pred
        return pred

    def evaluate_model(self) -> float:
        """모델 evaluate 함수"""
        model_logger.info(classification_report(self._pred_array, self._test_target))
        print(classification_report(self._pred_array, self._test_target))

    def ols(self):

        exog = self._preprocessed_data.drop(columns=["(혈청지오티)ALT"])
        if "const" not in self._preprocessed_data.columns:
            exog = sm.add_constant(exog)
        endog = self._preprocessed_data["(혈청지오티)ALT"]
        model = sm.OLS(endog=endog, exog=exog)  # modeling
        results = model.fit()
        model_logger.info("ols result------------------------------------------")
        model_logger.info(results.summary())
        model_logger.info("-----------------------------------------------------")
        return results, model, pd.concat([exog, endog], axis=1)

    def do_ols(self):
        self.ols()

    def fit_and_predict_test(self):
        """데이터 분리, 모델 빌드, 모델 적용 및 evaluation, 모델 저장, test_input에 대한 predict, evaluate 수행"""
        self._do_trans_proteinuria()
        self._split_data()
        self._build_model()
        self.predict(self._dtest)
        self.evaluate_model()
