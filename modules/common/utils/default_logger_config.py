import logging
import os
from datetime import datetime

from modules.common.utils.file_handler import chk_and_make_dir
from modules.common.utils.logging_format import CustomFormatter


class DefaultLogger:
    def __init__(self) -> None:
        pass

    def setDefaultLogger(
        self, logger_name: str, output_directory_path: str
    ) -> logging.Logger:
        """로거 이름과 output_directory_path, logger format 등을 로거에 할당 후 반환

        Args:
            logger_name (str): 로거 이름
            output_directory_path (str): 로그를 저장할 경로

        returns:
            logging.Logger : logger_name 가진 로거 반환

        """
        # 로거 설정
        log_file_name = os.path.join(
            output_directory_path,
            "log_current.log",
        )
        self._output_directory_path = output_directory_path
        chk_and_make_dir(output_directory_path)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(CustomFormatter())
        file_handler = logging.FileHandler(log_file_name)

        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s:%(name)s:%(asctime)s] %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            handlers=[stream_handler, file_handler],
        )

        return logging.getLogger(logger_name)

    def rename_log_file(self):
        """'log_current.log' 의 이름을 가진 로거의 이름을 'log_생성시각.log'로 바꿈"""
        logging.shutdown()
        c_time = os.path.getctime(
            os.path.join(
                self._output_directory_path,
                "log_current.log",
            )
        )
        c_time = datetime.fromtimestamp(c_time).strftime("%Y%m%d_%H%M%S")
        os.rename(
            os.path.join(
                self._output_directory_path,
                "log_current.log",
            ),
            os.path.join(
                self._output_directory_path,
                f"log_{c_time}.log",
            ),
        )
