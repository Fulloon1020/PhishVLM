from datetime import datetime, date, timedelta
from scripts.phishintention.configs import load_config
from scripts.pipeline.test_llm import *
from scripts.utils.PhishIntentionWrapper import LogoDetector, LogoEncoder, LayoutDetector
import argparse
from tqdm import tqdm
import yaml
import openai
import logging
from selenium.common.exceptions import *

# (确保导入 os)
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read().strip()
logging.getLogger("httpcore").setLevel(logging.WARNING)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="./datasets/field_study/2023-09-02/")
    parser.add_argument("--config", default='./param_dict.yaml', help="Config .yaml path")
    args = parser.parse_args()

    PhishLLMLogger.set_debug_on()
    PhishLLMLogger.set_verbose(True)

    # load hyperparameters
    with open(args.config) as file:
        param_dict = yaml.load(file, Loader=yaml.FullLoader)

    AWL_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE = load_config()
    logo_extractor = LogoDetector(AWL_MODEL)
    logo_encoder = LogoEncoder(SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE)
    layout_extractor = LayoutDetector(AWL_MODEL)

    # PhishLLM
    llm_cls = TestVLM(param_dict=param_dict,
                      logo_encoder=logo_encoder,
                      logo_extractor=logo_extractor,
                      layout_extractor=layout_extractor)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = os.getenv("http_proxy", "")  # set openai proxy

    driver = boot_driver()

    day = date.today().strftime("%Y-%m-%d")
    result_txt = '{}_phishllm.txt'.format(day)

    if not os.path.exists(result_txt):
        with open(result_txt, "w+") as f:
            f.write("folder" + "\t")
            f.write("phish_prediction" + "\t")

            # --- [修改 1/3] ---
            # 更新表头以匹配您要求的信号名称
            f.write("F_brand_flag" + "\t")  # 信号 B: 品牌意图不符标志
            f.write("F_intent_flag" + "\t")  # 信号 C: 凭证窃取意图标志
            # ---------------------

            f.write("brand_recog_time" + "\t")
            f.write("crp_prediction_time" + "\t")
            f.write("crp_transition_time" + "\n")

    for ct, folder in tqdm(enumerate(os.listdir(args.folder))):
        if folder in [x.split('\t')[0] for x in open(result_txt, encoding='ISO-8859-1').readlines()]:
            continue

        info_path = os.path.join(args.folder, folder, 'info.txt')
        html_path = os.path.join(args.folder, folder, 'html.txt')
        shot_path = os.path.join(args.folder, folder, 'shot.png')
        predict_path = os.path.join(args.folder, folder, 'predict.png')
        if not os.path.exists(shot_path):
            continue

        try:
            if len(open(info_path, encoding='ISO-8859-1').read()) > 0:
                url = open(info_path, encoding='ISO-8859-1').read()
            else:
                url = 'https://' + folder
        except FileNotFoundError:
            url = 'https://' + folder

        logo_box, reference_logo = llm_cls.detect_logo(shot_path)
        try:

            # --- [修改 2/3] ---
            # 解包 llm_cls.test() 返回的 7 个值, 捕获两个新标志
            pred, F_brand, F_intent, brand_recog_time, crp_prediction_time, crp_transition_time, plotvis = llm_cls.test(
                url=url,
                reference_logo=reference_logo,
                logo_box=logo_box,
                shot_path=shot_path,
                html_path=html_path,
                driver=driver,
                )
            # ---------------------

            driver.delete_all_cookies()
        except (WebDriverException) as e:
            print(f"Driver crashed or encountered an error: {e}. Restarting driver.")
            driver = restart_driver(driver)
            continue

        try:
            with open(result_txt, "a+", encoding='ISO-8859-1') as f:

                # --- [修改 3/3] ---
                # 将捕获到的两个标志写入结果文件
                f.write(
                    f"{folder}\t{pred}\t{F_brand}\t{F_intent}\t{brand_recog_time}\t{crp_prediction_time}\t{crp_transition_time}\n")
                # ---------------------

            if pred == 'phish':
                plotvis.save(predict_path)
        except UnicodeEncodeError:
            continue

    driver.quit()
