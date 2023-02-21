from multiprocessing import cpu_count
from typing import Union

import numpy as np
import pandas as pd


def string_to_boolean(arg_str: str) -> bool or np.NaN:
    """명시된 str 예시를 bool 값으로 바꿔주는 함수

    Args:
        arg_str (str) : bool 값으로 바꿀 str

    return:
        bool : 변환된 bool 값

    """

    list_true = ["true", "1", "yes"]
    list_false = ["false", "0", "no"]

    if arg_str.lower() in list_true:
        return True
    if arg_str.lower() in list_false:
        return False
    return np.NaN


def list_chunk(lst: list, n_item: int) -> list:
    """주어진 리스트를 n개씩 분할해주는 함수

    Args:
        lst(list) : 분할할 리스트
        n_item (int) : 한 리스트에 들어갈 n 수

    return:
        list : n개씩 분할한 리스트가 담긴 리스트
    """
    return [lst[i : i + n_item] for i in range(0, len(lst), n_item)]


def unnesting(df: pd.DataFrame, explode_columns: list) -> pd.DataFrame:
    """컬럼 내부의 list를 explode하는 함수

    Args:
        df (pd.DataFrame): explode할 데이터프레임
        explode_columns (list): explode할 컬럼

    Returns:
        pd.DataFrame: explode된 데이터프레임
    """

    list_columns = df.columns
    idx = df.index.repeat(df[explode_columns[0]].str.len())
    df1 = pd.concat(
        [pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode_columns],
        axis=1,
    )
    df1.index = idx
    result = df1.join(df.drop(explode_columns, 1), how="left").reset_index(drop=True)
    result = result[list_columns]
    return result


def trim_df(df: pd.DataFrame) -> pd.DataFrame:
    """dtype을 object로 바꾸고, 양 측의 공백 제거

    Args:
        df (pd.DataFrame): 적용할 데이터프레임

    Returns:
        pd.DataFrame: trim이 적용된 데이터프레임
    """
    df_obj = df.select_dtypes(["object"])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    return df


def is_number(value: str):
    """number인지 판별해주는 함수, False면 number 아님, True 면 숫자임

    Args:
        value (str) : 적용 대상

    Returns:
        bool : 숫자 여부
    """
    try:
        judge = str(float(value))
        return False if (judge == "nan" or judge == "inf" or judge == "-inf") else True
    except ValueError:
        return False


def str_to_number(value: str):
    """str을 int or float으로 반환해주는 함수,
    변환이 가능하면 int or float을 return 하고, 불가능하면 그대로 value를 return함

    Args:
        value (str) : 적용 대상

    Returns:
        Union[int, float, str] : return
    """
    if is_number(value):
        try:
            return int(value)
        except ValueError:
            return float(value)
    return value
