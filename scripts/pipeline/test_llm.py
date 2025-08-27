from openai import OpenAI
import torch
from scripts.utils.utils import *
from scripts.utils.web_utils import *
from scripts.utils.draw_utils import draw_annotated_image_box
import os
from typing import List, Tuple, Optional, Union, Dict
import PIL
import json
from tldextract import tldextract
import urllib3
from urllib3.exceptions import MaxRetryError
urllib3.disable_warnings()
http = urllib3.PoolManager(maxsize=10)  # Increase the maxsize to a larger value, e.g., 10

os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read().strip()
os.environ['CURL_CA_BUNDLE'] = ''


class TestVLM():

    def __init__(self, logo_encoder, logo_extractor, layout_extractor, param_dict: Dict, proxies:Union[float, Dict] = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.proxies = proxies
        self.logo_encoder   = logo_encoder
        self.logo_extractor = logo_extractor
        self.layout_extractor = layout_extractor

        ## LLM
        self.VLM_model = param_dict["VLM_model"]
        self.brand_prompt = param_dict['brand_recog']['prompt_path']
        self.crp_prompt = param_dict['crp_pred']['prompt_path']
        self.rank_prompt = param_dict['rank']['prompt_path']
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        # Load the Google API key and SEARCH_ENGINE_ID once during initialization
        self.API_KEY, self.SEARCH_ENGINE_ID = [x.strip() for x in open('./datasets/google_api_key.txt').readlines()]

        ## Load hyperparameters

        self.brand_recog_temperature, self.brand_recog_max_tokens = param_dict['brand_recog']['temperature'], param_dict['brand_recog']['max_tokens']
        self.brand_recog_sleep = param_dict['brand_recog']['sleep_time']
        self.do_brand_validation = param_dict['brand_valid']['activate']
        self.brand_valid_k, self.brand_valid_siamese_thre = param_dict['brand_valid']['k'], param_dict['brand_valid']['siamese_thre']

        self.crp_temperature, self.crp_max_tokens = param_dict['crp_pred']['temperature'], param_dict['crp_pred']['max_tokens']
        self.crp_sleep = param_dict['crp_pred']['sleep_time']

        self.rank_max_uis = param_dict['rank']['max_uis_process']
        self.rank_temperature, self.rank_max_tokens = param_dict['rank']['temperature'], param_dict['rank']['max_tokens']
        self.rank_driver_sleep = param_dict['rank']['driver_sleep_time']
        self.rank_driver_script_timeout = param_dict['rank']['script_timeout']
        self.rank_driver_page_load_timeout = param_dict['rank']['page_load_timeout']
        self.interaction_limit = param_dict['rank']['depth_limit']

        # webhosting domains as blacklist
        self.webhosting_domains = [x.strip() for x in open('./datasets/hosting_blacklists.txt').readlines()]


    def detect_logo(self, save_shot_path: str) -> Tuple[Optional[List[float]], Optional[Image.Image]]:
        '''
            Logo detection
            :param save_shot_path:
            :return:
        '''
        reference_logo = None
        logo_box = None

        try:
            screenshot_img = Image.open(save_shot_path).convert("RGB")
            logo_boxes = self.logo_extractor(save_shot_path)
            if len(logo_boxes) > 0:
                logo_box = logo_boxes[0]  # get coordinate for logo
                reference_logo = screenshot_img.crop((int(logo_box[0]), int(logo_box[1]),
                                                      int(logo_box[2]), int(logo_box[3])))
        except PIL.UnidentifiedImageError:
            pass

        return logo_box, reference_logo


    def brand_recognition_llm(self, reference_logo: Optional[Image.Image]) -> Tuple[Optional[str], Optional[Image.Image], float]:
        '''
            Brand Recognition Model
            :param reference_logo:
            :param webpage_text:
            :param logo_caption:
            :param logo_ocr:
            :param announcer:
            :return:
        '''
        company_domain, company_logo = None, None
        brand_llm_pred_time = 0

        if not reference_logo:
            return company_domain, company_logo, brand_llm_pred_time

        with open(self.brand_prompt, 'r') as file:
            system_prompt = json.load(file)
        question = vlm_question_template_brand(reference_logo)
        system_prompt.append(question)

        inference_done = False
        while not inference_done:
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.VLM_model,
                    messages=system_prompt,
                    temperature=self.brand_recog_temperature,
                    max_tokens=self.brand_recog_max_tokens,
                )
                brand_llm_pred_time = time.time() - start_time
                inference_done = True
            except Exception as e:
                PhishLLMLogger.spit('LLM Exception {}'.format(e), debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
                system_prompt[-1]['content'] = system_prompt[-1]['content'][:len(system_prompt[-1]['content']) // 2]  # maybe the prompt is too long, cut by half
                time.sleep(self.brand_recog_sleep)  # retry

        answer = ''.join([choice.message.content for choice in response.choices])

        PhishLLMLogger.spit(f"Time taken for LLM brand prediction: {brand_llm_pred_time}\tDetected brand: {answer}",
                            debug=True,
                            caller_prefix=PhishLLMLogger._caller_prefix)

        # check the validity of the returned domain, i.e. liveness
        if len(answer) > 0 and is_valid_domain(answer):
            company_logo = reference_logo
            company_domain = answer

        return company_domain, company_logo, brand_llm_pred_time

    def popularity_validation(self, company_domain: str) -> Tuple[bool, float]:
        '''
            Brand recognition model : result validation
            :param company_domain:
            :return:
        '''
        validation_success = False

        start_time = time.time()
        returned_urls = query2url(query=company_domain,
                                  SEARCH_ENGINE_ID=self.SEARCH_ENGINE_ID,
                                  SEARCH_ENGINE_API=self.API_KEY,
                                  num=self.brand_valid_k,
                                  proxies=self.proxies)
        searching_time = time.time() - start_time

        returned_domains = ['.'.join(part for part in tldextract.extract(url) if part).split('www.')[-1] for url in returned_urls]
        if company_domain in returned_domains:
            validation_success = True

        return validation_success, searching_time

    def brand_validation(
        self,
        company_domain: str,
        reference_logo: Image.Image
    ) -> Tuple[bool, float, float]:
        '''
            Brand recognition model : result validation
            :param company_domain:
            :param reference_logo:
            :return:
        '''
        logo_searching_time, logo_matching_time = 0, 0
        validation_success = False

        if not reference_logo:
            return True, logo_searching_time, logo_matching_time

        start_time = time.time()
        returned_urls = query2image(query='Brand: ' + company_domain + ' logo',
                                    SEARCH_ENGINE_ID=self.SEARCH_ENGINE_ID, SEARCH_ENGINE_API=self.API_KEY,
                                    num=self.brand_valid_k,
                                    proxies=self.proxies)
        logo_searching_time = time.time() - start_time
        logos = get_images(returned_urls, proxies=self.proxies)
        msg = f'Number of logos found on google images {len(logos)}'
        print(msg)
        PhishLLMLogger.spit(msg, debug=True, caller_prefix=PhishLLMLogger._caller_prefix)

        if len(logos) > 0:
            reference_logo_feat = self.logo_encoder(reference_logo)
            start_time = time.time()
            sim_list = []
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.logo_encoder, logo) for logo in logos]
                for future in futures:
                    logo_feat = future.result()
                    matched_sim = reference_logo_feat @ logo_feat
                    sim_list.append(matched_sim)

            if any([x > self.brand_valid_siamese_thre for x in sim_list]):
                validation_success = True

            logo_matching_time = time.time() - start_time

        return validation_success, logo_searching_time, logo_matching_time

    def crp_prediction_llm(self, webpage_screenshot: Image.Image) -> Tuple[bool, float]:
        '''
            Use LLM to classify credential-requiring page v.s. non-credential-requiring page
            :param webpage_screenshot:
            :return:
        '''
        crp_llm_pred_time = 0

        with open(self.crp_prompt, 'r') as file:
            system_prompt = json.load(file)
        question = vlm_question_template_prediction(webpage_screenshot)
        system_prompt.append(question)

        # example token count from the OpenAI API
        inference_done = False
        while not inference_done:
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.VLM_model,
                    messages=system_prompt,
                    temperature=self.crp_temperature,
                    max_tokens=self.crp_max_tokens,
                )
                crp_llm_pred_time = time.time() - start_time
                inference_done = True
            except Exception as e:
                PhishLLMLogger.spit('LLM Exception {}'.format(e), debug=True,
                                    caller_prefix=PhishLLMLogger._caller_prefix)
                system_prompt[-1]['content'] = system_prompt[-1]['content'][:len(system_prompt[-1]['content']) // 2]  # maybe the prompt is too long, cut by half
                time.sleep(self.crp_sleep)

        answer = ''.join([choice.message.content for choice in response.choices])

        PhishLLMLogger.spit(f'Time taken for LLM CRP classification: {crp_llm_pred_time} \t CRP prediction: {answer}',
                            debug=True,
                            caller_prefix=PhishLLMLogger._caller_prefix)
        if 'A.' in answer:
            return True, crp_llm_pred_time  # CRP
        else:
            return False, crp_llm_pred_time

    def ranking_model(self, url: str, driver: WebDriver, ranking_model_refresh_page: bool) -> \
                                Tuple[Union[List, str], List[torch.Tensor], WebDriver, float]:
        transition_pred_time = 0
        if ranking_model_refresh_page:
            try:
                driver.get(url)
                time.sleep(self.rank_driver_sleep)
            except Exception as e:
                PhishLLMLogger.spit(e, debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
                driver = restart_driver(driver)
                return [], [], driver, transition_pred_time

        try:
            (btns, btns_dom),  \
                (links, links_dom), \
                (images, images_dom), \
                (others, others_dom) = get_all_clickable_elements(driver)
        except Exception as e:
            PhishLLMLogger.spit(e, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
            return [], [], driver, transition_pred_time

        all_clickable = btns + links + images + others
        all_clickable_dom = btns_dom + links_dom + images_dom + others_dom

        # element screenshot
        candidate_uis = []
        candidate_uis_imgs = []
        candidate_uis_text = []

        for it in range(min(self.rank_max_uis, len(all_clickable))):
            try:
                candidate_ui, candidate_ui_img, candidate_ui_text = screenshot_element(all_clickable[it],
                                                                                       all_clickable_dom[it],
                                                                                       driver)
            except (MaxRetryError, WebDriverException, TimeoutException) as e:
                PhishLLMLogger.spit(e, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
                driver = restart_driver(driver)
                continue

            if (candidate_ui is not None) and (candidate_ui_img is not None) and (candidate_ui_text is not None):
                candidate_uis.append(candidate_ui)
                candidate_uis_imgs.append(candidate_ui_img)
                candidate_uis_text.append(candidate_ui_text)

        # rank them
        if len(candidate_uis_imgs):
            msg = f'Find {len(candidate_uis_imgs)} candidate UIs'
            PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)

            # Simple heuristic is to find login-related text
            regex_match = [
                bool(re.search(Regexes.CREDENTIAL_TAKING_KEYWORDS, text, re.IGNORECASE | re.VERBOSE)) if text else False
                for text in candidate_uis_text]
            indices = []
            for i, is_match in enumerate(regex_match):
                if is_match:
                    indices.append(i)

            if len(indices) > 0:
                candidate_uis_selected = [candidate_uis[ind] for ind in indices]
                candidate_imgs_selected = [candidate_uis_imgs[ind] for ind in indices]
                return candidate_uis_selected, candidate_imgs_selected, driver, transition_pred_time

            ## fixme
            with open(self.rank_prompt, 'r') as file:
                system_prompt = json.load(file)
            question = vlm_question_template_transition(candidate_uis_imgs, candidate_uis_text)
            system_prompt.append(question)

            inference_done = False
            while not inference_done:
                try:
                    start_time = time.time()
                    response = self.client.chat.completions.create(
                        model=self.VLM_model,
                        messages=system_prompt,
                        temperature=self.rank_temperature,
                        max_tokens=self.rank_max_tokens,
                    )
                    transition_pred_time = time.time() - start_time
                    inference_done = True
                except Exception as e:
                    PhishLLMLogger.spit('LLM Exception {}'.format(e), debug=True,
                                        caller_prefix=PhishLLMLogger._caller_prefix)
                    system_prompt[-1]['content'] = system_prompt[-1]['content'][:len(
                        system_prompt[-1]['content']) // 2]  # maybe the prompt is too long, cut by half
                    time.sleep(self.crp_sleep)

            answer = ''.join([choice.message.content for choice in response.choices])
            indices = eval(answer) # get the login UI index

            if len(indices) > 0:
                candidate_uis_selected = [candidate_uis[ind] for ind in indices]
                candidate_imgs_selected = [candidate_uis_imgs[ind] for ind in indices]
                return candidate_uis_selected, candidate_imgs_selected, driver, transition_pred_time

            return [], [], driver, transition_pred_time


    def test(self, url: str,
             reference_logo: Optional[Image.Image],
             logo_box: Optional[List[float]],
             shot_path: str,
             html_path: str,
             driver: Union[WebDriver, float]=None,
             limit: int=0,
             brand_recog_time: float=0, crp_prediction_time: float=0, clip_prediction_time: float=0,
             ranking_model_refresh_page: bool=True,
             skip_brand_recognition: bool=False,
             company_domain: Optional[str]=None, company_logo: Optional[Image.Image]=None,
             ):
        '''
            PhishLLM
            :param url:
            :param reference_logo:
            :param logo_box:
            :param shot_path:
            :param html_path:
            :param driver:
            :param limit:
            :param brand_recog_time:
            :param crp_prediction_time:
            :param clip_prediction_time:
            :param ranking_model_refresh_page:
            :param skip_brand_recognition:
            :param company_domain:
            :param company_logo:
            :param announcer:
            :return:
        '''
        ## Run OCR to extract text
        plotvis = Image.open(shot_path)

        ## Brand recognition model
        if not skip_brand_recognition:
            company_domain, company_logo, brand_recog_time = self.brand_recognition_llm(reference_logo=reference_logo)
            time.sleep(self.brand_recog_sleep) # fixme: allow the openai api to rest, not sure whether this help
        # check domain-brand inconsistency
        domain_brand_inconsistent = False

        if company_domain:
            if company_domain in self.webhosting_domains:
                msg = '[\U00002705] Benign, since it is a brand providing cloud services'
                PhishLLMLogger.spit(msg)
                return 'benign', 'None', brand_recog_time, crp_prediction_time, clip_prediction_time, plotvis

            domain4pred, suffix4pred = tldextract.extract(company_domain).domain, tldextract.extract(company_domain).suffix
            domain4url, suffix4url = tldextract.extract(url).domain, tldextract.extract(url).suffix
            domain_brand_inconsistent = (domain4pred != domain4url) or (suffix4pred != suffix4url)

        phish_condition = domain_brand_inconsistent

        # Brand prediction results validation
        if phish_condition and (not skip_brand_recognition):
            if self.do_brand_validation: # we can check the validity by comparing the logo on the webpage with the logos for the predicted brand
                validation_success, logo_searching_time, logo_matching_time = self.brand_validation(company_domain=company_domain,
                                                                                                    reference_logo=reference_logo)
                brand_recog_time += logo_searching_time
                brand_recog_time += logo_matching_time
                phish_condition = validation_success
                msg = f"Time taken for brand validation (logo matching with Google Image search results): {logo_searching_time+logo_matching_time}<br>Domain {company_domain} is relevant and valid? {validation_success}"
                PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
            else: # alternatively, we can check the aliveness of the predicted brand
                validation_success = is_alive_domain(domain4pred + '.' + suffix4pred , self.proxies) # fixme
                phish_condition = validation_success
                msg = f"Brand Validation: Domain {company_domain} is alive? {validation_success}"
                PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)

        if phish_condition:
            # CRP prediction model
            crp_cls, crp_prediction_time = self.crp_prediction_llm(webpage_screenshot=plotvis)
            time.sleep(self.crp_sleep)

            if crp_cls: # CRP page is detected
                plotvis = draw_annotated_image_box(plotvis, company_domain, logo_box)
                msg = f'[\u2757\uFE0F] Phishing discovered, phishing target is {company_domain}'
                PhishLLMLogger.spit(msg)
                return 'phish', company_domain, brand_recog_time, crp_prediction_time, clip_prediction_time, plotvis
            else:
                # Not a CRP page => CRP transition
                if limit >= self.interaction_limit:  # reach interaction limit -> just return
                    msg = '[\U00002705] Benign, reached interaction limit ...'
                    PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
                    return 'benign', 'None', brand_recog_time, crp_prediction_time, clip_prediction_time, plotvis

                # Ranking model
                candidate_elements, _, driver, clip_prediction_time = self.ranking_model(url=url, driver=driver, ranking_model_refresh_page=ranking_model_refresh_page)

                if len(candidate_elements):
                    save_html_path = re.sub("index[0-9]?.html", f"index{limit}.html", html_path)
                    save_shot_path = re.sub("shot[0-9]?.png", f"shot{limit}.png", shot_path)

                    if not ranking_model_refresh_page: # if previous click didnt refresh the page select the lower ranked element to click
                        msg = f"Since previously the URL has not changed, trying to click the Top-{min(len(candidate_elements), limit+1)} login button instead: "
                        candidate_ele = candidate_elements[min(len(candidate_elements)-1, limit)]
                    else: # else, just click the top-1 element
                        msg = "Trying to click the Top-1 login button: "
                        candidate_ele = candidate_elements[0]

                    # record the webpage elements before clicking the button
                    screenshot_path = "tmp.png"
                    driver.get_screenshot_as_file(screenshot_path)
                    _, prev_screenshot_elements = self.layout_extractor(screenshot_path)
                    os.remove(screenshot_path)

                    element_text, current_url, *_ = page_transition(driver=driver,
                                                                    dom=candidate_ele,
                                                                    save_html_path=save_html_path,
                                                                    save_shot_path=save_shot_path)
                    msg += f'{element_text}'
                    PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
                    if current_url: # click success
                        screenshot_path = "tmp.png"
                        driver.get_screenshot_as_file(screenshot_path)
                        _, curr_screenshot_elements = self.layout_extractor(screenshot_path)
                        os.remove(screenshot_path)
                        ranking_model_refresh_page = has_page_content_changed(curr_screenshot_elements=curr_screenshot_elements,
                                                                              prev_screenshot_elements=prev_screenshot_elements)
                        msg = f"Has the webpage changed? {ranking_model_refresh_page}"
                        PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)

                        # logo detection on new webpage
                        logo_box, reference_logo = self.detect_logo(save_shot_path)
                        return self.test(current_url, reference_logo, logo_box,
                                         save_shot_path, save_html_path, driver, limit + 1,
                                         brand_recog_time, crp_prediction_time, clip_prediction_time,
                                         ranking_model_refresh_page=ranking_model_refresh_page,
                                         skip_brand_recognition=True,
                                         company_domain=company_domain,
                                         company_logo=company_logo)
                else:
                    msg = '[\U00002705] Benign'
                    PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
                    return 'benign', 'None', brand_recog_time, crp_prediction_time, clip_prediction_time, plotvis

        msg = '[\U00002705] Benign'
        PhishLLMLogger.spit(msg)
        return 'benign', 'None', brand_recog_time, crp_prediction_time, clip_prediction_time, plotvis




