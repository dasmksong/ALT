import logging
import os
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler

from modules.config import Config

preprocess_logger = logging.getLogger("Preprocess")


def drop_col(
    df: pd.DataFrame,
    col_list: list,
) -> pd.DataFrame:

    """DataFrame에서 col_list의 column를 drop

    Args:
        df(pd.DataFrame) : 원본 데이터프레임
        col_list (list) : drop 될 column

    return:
        pd.DataFrame : col이 drop된 df
    """
    return df.drop(col_list, axis=1)


def drop_na(
    df: pd.DataFrame,
) -> pd.DataFrame:

    """DataFrame에서 nan이 포함된 행을 drop

    Args:
        df(pd.DataFrame) : 원본 데이터프레임

    return:
        pd.DataFrame : drop_na 한 데이터 프레임
    """
    return df.dropna()


def log_trans(df: pd.DataFrame, method: str, columns: list):
    """method를 통해 log 변환 방식을 선택하고 해당 columns들을 로그변환

    Args:
        df (pd.DataFrame) : 변환할 DataFrame
        method (str) : log변환 혹은 log1p변환
        columns (list) : 선택 columns

    return:
        pd.DataFrame : 변환 적용된 pd.DataFrame
    """

    dataset = df.copy()
    if method == "log":
        for i in columns:
            dataset[i] = np.log(dataset[i])
    elif method == "log1p":
        for i in columns:
            dataset[i] = np.log1p(dataset[i])
    else:
        raise Exception("method를 log 혹은 log1p로 입력해주세요")
    return dataset


def scale_Robust(df: pd.DataFrame):
    """df에 Robustscaler를 적용한 pd.DataFrame 반환

    Args:
        df (pd.DataFrame) : 변환할 DataFrame

    return:
        pd.DataFrame : 변환 적용된 pd.DataFrame
    """
    data = df.copy()
    columns = df.columns
    scaler_ = RobustScaler()
    scaler_.fit(data)
    scaled = pd.DataFrame(scaler_.transform(data))
    scaled.columns = columns
    return scaled


class Preprocess:
    """Preprocess class

    Attribute:
        _data (pd.DataFrame) : 원본 데이터 프레임
        _preprocessed_data (pd.DataFrame) : 전처리 후 데이터 프레임
        _train_input (np.ndarray) : train 데이터 프레임 input
        _train_target (np.ndarray) : train 데이터 프레임 target
        _test_input (np.ndarray) : test 데이터 프레임 input
        _test_target (np.ndarray) : test 데이터 프레임 target

    """

    _data: pd.DataFrame
    _preprocessed_data: pd.DataFrame
    _train_input: np.ndarray
    _train_target: np.ndarray
    _test_input: np.ndarray
    _test_target: np.ndarray
    _config: dict

    @property
    def raw_data(self):
        return self._data

    @raw_data.setter
    def raw_data(self, new_df):
        """원본 데이터 프레임 변경

        Args:
            new_df (pd.DataFrame) : 새로 변경할 데이터 프레임

        """
        self._data = new_df

    def __init__(self) -> None:
        """Preprocess init"""
        self._config = Config.instance().config

    def load_data(self):
        """self._data 에 원본 데이터 불러오는 함수"""
        self._data = pd.read_csv(
            os.path.join(self._config["path"]["data"], "data.csv"), encoding="cp949"
        )
        pass

    def preprocess(self) -> None:
        """self._preprocessed 에 _raw_data를 할당하고, raw dataset과 preprocessed dataset의 길이를 info level에서 출력

        Args:
            self (Preprocess) :

        """
        raw_df = self._data
        drop_col_data = drop_col(raw_df, self._config["drop_col"]["drop_col_list"])
        drop_na_data = drop_na(drop_col_data)
        log_trans_data = log_trans(
            drop_na_data, "log", self._config["log_trans"]["log_trans_col_list"]
        )
        log1p_trans_data = log_trans(
            log_trans_data, "log1p", self._config["log_trans"]["log1p_trans_col_list"]
        )
        scaled_data = scale_Robust(log1p_trans_data)

        self._preprocessed_data = scaled_data
        preprocess_logger.info("Model Data Count-------------------------------------")
        preprocess_logger.info("raw dataset        : " + str(len(self._data)))
        preprocess_logger.info(
            "preprocessed dataset  : " + str(len(self._preprocessed_data))
        )
        preprocess_logger.info("-----------------------------------------------------")
        pass

    def run(self) -> Any:
        self.load_data()
        self.preprocess()
        return self._preprocessed_data
