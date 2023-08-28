from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.quick_start import run_recbole, quick_start
from recbole.trainer import Trainer



def EASE():
    model = 'EASE'
    dataset = 'ml-1m'
    config_dict = {
        "model": model,
        "dataset": dataset,
        "config_file_list": ["ml-1m.yaml"],  # 데이터셋 설정 파일
        "reg_weight": 667,  # 초기 값 설정
    }
        # RecBole 실행
    return run_recbole(model=model, dataset=dataset, config_dict=config_dict)