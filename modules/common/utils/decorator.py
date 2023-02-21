import logging
import os
import re
import sys
import time
import traceback
from functools import wraps
from typing import Any


class RunningTimeDecorator:
    """RunningTimeDecorator class

    Attributes:

        self.__param (None)= logger
        self.__show_section (bool) = Section 표기 여부
        self.__show_pid (bool) = PID 표기 여부
    """

    def __init__(self, logger=None, show_section: bool = True, show_pid: bool = True):
        self.__param = logger
        self.__show_section = show_section
        self.__show_pid = show_pid

    def __call__(self, func):
        def printLog(str_arg: str):
            """logger 설정 시, log.info를, 미설정시 str_arg을 출력하는 함수

            Args:
                str_arg (None)= 해당 출력문

            """
            if isinstance(self.__param, logging.Logger):
                logger = logging.getLogger(self.__param.name)
                logger.info(str_arg)
            else:
                print(str_arg)

        @wraps(func)
        def decorator(*args, **kwargs):
            """함수 시작 시간과 경과 시간, PID 값을 보여주는 함수

            Args:
                *args : func에 인자로 들어가는 args
                **kwargs : func에 인자로 들어가는 kwargs

            returns:
                Any : func의 result 값

            """
            str_current_pid = ""

            if self.__show_section:
                if self.__show_pid:
                    str_current_pid = "(PID:" + str(os.getpid()) + ")"
                str_start = "{0}{1} Started.".format(func.__name__, str_current_pid)
                printLog(str_start)

            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            if self.__show_section:
                if self.__show_pid:
                    str_current_pid = "(PID:" + str(os.getpid()) + ")"
                str_finish = "{0}{1} Finished.".format(func.__name__, str_current_pid)
                printLog(str_finish)
            str_log = "{0}{1} Elapsed Time : {2:.2f} seconds".format(
                func.__name__, str_current_pid, end_time - start_time
            )
            printLog(str_log)
            return result

        return decorator


class TryDecorator:
    """TryDecorator class

    Attributes:
        self.__param(logger) = logger
        self.__exit(bool) = 종료 가능 여부

    """

    def __init__(self, logger, exit=True):
        self.__param = logger
        self.__exit = exit

    def __call__(self, func):
        def printLog(str_arg: str):
            if isinstance(self.__param, logging.Logger):
                logger = logging.getLogger(self.__param.name)
                logger.error(str_arg)
            else:
                print(str_arg)

        @wraps(func)
        def decorator(*args, **kwargs):
            """func 정상 시행 및 종료 시 func result 반환, 오류 발생 시 오류 log 출력 및 Exit code 1

            Args:
                *args : func에 인자로 들어가는 args
                **kwargs : func에 인자로 들어가는 kwargs

            returns:
                Any : func의 result 값

            """
            try:
                result = func(*args, **kwargs)
                return result

            except Exception:
                formatted_lines = traceback.format_exc().splitlines()
                num = [
                    idx
                    for idx, i in enumerate(formatted_lines)
                    if re.search(" line ", i) is not None
                ][-1]

                printLog("=" * 100)
                for formatted_line in formatted_lines[num:]:
                    printLog(formatted_line)
                printLog("=" * 100)
                if self.__exit:
                    sys.exit(1)

        return decorator
