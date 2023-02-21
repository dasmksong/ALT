import atexit
import os

from modules.bagging_model import Bagging
from modules.boosting_model import Boosting
from modules.common.utils.default_logger_config import DefaultLogger
from modules.common.utils.file_handler import chk_and_make_dir
from modules.config import Config

# from modules.model import Model
from modules.preprocess import Preprocess

if __name__ == "__main__":
    # config 설정
    config_path = os.path.join("config", "config.ini")
    config = Config.instance(config_path).config

    # logger 세팅
    default_logger = DefaultLogger()
    main_logger = default_logger.setDefaultLogger("main", config["log"]["path"])
    atexit.register(default_logger.rename_log_file)

    # output 디렉토리 세팅
    for path in config["path"]:
        chk_and_make_dir(config["path"][path])

    main_logger.info("Program Start")
    main_logger.info(config)
    preprocess = Preprocess()
    preprocessed_data = preprocess.run()
    model = Bagging(preprocessed_data)

    model.do_ols()
    model.fit_and_predict_test()

    model = Boosting(preprocessed_data)
