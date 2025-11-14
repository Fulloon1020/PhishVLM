
# PhishVLM

This version of the PhishVLM codebase has been modified from the original to serve as a signal extraction component for a Computer Networks project.

The primary goal was to adapt this tool to provide raw risk signals to a separate "Central Decision Engine" (插件), rather than using its own built-in decision logic.

Key modifications were made to scripts/pipeline/test_llm.py and scripts/infer/test.py to achieve this. The script now calculates and directly outputs the two core risk signals:

Signal B: F_brand_flag (品牌意图不符标志): test_llm.py now explicitly performs the cross-validation between the LLM-detected brand and the current URL's domain. It returns a True/False flag (None if the check fails/is skipped).

Signal C: F_intent_flag (凭证窃取意图标志): test_llm.py now directly returns the True/False result (None if skipped) from the crp_prediction_llm model.

<p align="center">

  • <a href="">Read Paper</a> •

  • <a href="https://sites.google.com/view/phishllm">Visit Website</a> •

  • <a href="https://sites.google.com/view/phishllm/experimental-setup-datasets?authuser=0#h.r0fy4h1fw7mq">Download Datasets</a>  •

</p>

## Introduction
Existing reference-based phishing detection:

- :x: Relies on a pre-defined reference list, which is lack of comprehensiveness and incurs high maintenance cost 
- :x: Does not fully make use of the textual semantics present on the webpage

In our PhishVLM, we build a reference-based phishing detection framework:

- ✅ **Without the pre-defined reference list**: Modern VLMs have encoded far more extensive brand-domain information than any predefined list
- ✅ **Chain-of-thought credential-taking prediction**: Reasoning the credential-taking status in a step-by-step way by looking at the screenshot

## Framework

Input: a URL and its screenshot, Output: Phish/Benign, Phishing target

* **Step 1: Brand recognition model**
    * Input: Logo Screenshot
    * Output: VLM's predicted brand
* **Step 2: Credential-Requiring-Page classification model**
    * Input: Webpage Screenshot
    * Output: VLM chooses from A. Credential-Taking Page or B. Non-Credential-Taking Page
    * *Go to step 4 if VLM chooses 'A', otherwise go to step 3.*
* **Step 3: Credential-Requiring-Page transition model (activates if VLM chooses 'B' from the last step)**
    * Input: All clickable UI elements screenshots
    * Intermediate Output: Top-1 most likely login UI
    * Output: Webpage after clicking that UI, go back to Step 1 with the updated webpage and URL
* **Step 4: Output step**
    * *Case 1: If the domain is from a web hosting domain: it is flagged as phishing if (i) VLM predicts a targeted brand inconsistent with the webpage's domain and (ii) VLM chooses 'A' from Step 2*
    * *Case 2: If the domain is not from a web hosting domain: it is flagged as phishing if (i) VLM predicts a targeted brand inconsistent with the webpage's domain (ii) VLM chooses 'A' from Step 2 and (iii) the domain is not a popular domain indexed by Google*
    * *Otherwise: reported as benign*

**Project structure**
```
scripts/ 
├── infer/
│   └──test.py             # inference script
├── pipeline/             
│   └──test_llm.py # TestVLM class
└── utils/ # other utitiles such as web interaction utility functions 

prompts/ 
├── brand_recog_prompt.json 
└── crp_pred_prompt.json
└── crp_trans_prompt.json
```

## Setup

**Step 1: Install Requirements**

Tested on Ubuntu, CUDA 11. A new conda environment "phishllm" will be created after this step.

```bash
conda create -n phishllm python=3.10
conda activate phishllm
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url [https://download.pytorch.org/whl/cu113](https://download.pytorch.org/whl/cu113)
pip install --no-build-isolation git+[https://github.com/facebookresearch/detectron2.git](https://github.com/facebookresearch/detectron2.git)
cd scripts/phishintention
chmod +x setup.sh
./setup.sh
```
**Step 2: Install Chrome**

```bash

sudo apt install ./google-chrome-stable_current_amd64.deb
```
**Step 3: Register Two API Keys**
- 🔑 **OpenAI API key**, [See Tutorial here](https://platform.openai.com/docs/quickstart). Paste the API key to ``./datasets/openai_key.txt``.

- 🔑 **Google Programmable Search API Key**, [See Tutorial here](https://meta.discourse.org/t/google-search-for-discourse-ai-programmable-search-engine-and-custom-search-api/307107). 
Paste your API Key (in the first line) and Search Engine ID (in the second line) to ``./datasets/google_api_key.txt``:
     ```text 
      [API_KEY]
      [SEARCH_ENGINE_ID]
     ```
     
**Prepare the Dataset**
To test on your own dataset, you need to prepare the dataset in the following structure:
```
testing_dir/
├── [aaa.com/](https://aaa.com/)
│   ├── shot.png  # save the webpage screenshot
│   ├── info.txt  # save the webpage URL
│   └── html.txt  # save the webpage HTML source
├── [bbb.com/](https://bbb.com/)
│   ├── shot.png  # save the webpage screenshot
│   ├── info.txt  # save the webpage URL
│   └── html.txt  # save the webpage HTML source
├── [ccc.com/](https://ccc.com/)
│   ├── shot.png  # save the webpage screenshot
│   ├── info.txt  # save the webpage URL
│   └── html.txt  # save the webpage HTML source
```
## Run PhishLLM
我使用 -m 标志来运行脚本，这可以确保 Python 正确解析模块路径，避免 ModuleNotFoundError。

```Bash

conda activate phishllm
python -m scripts.infer.test --folder [folder to test, e.g., ./datasets/test_sites/dynapd]
```
## Understand the Output
You will see the console is printing logs like the following: (Sample log hidden)

Meanwhile, a txt file named "[today's date]_phishllm.txt" is being created. This file now contains the raw signals needed for downstream analysis (e.g., a central decision engine plugin).

It has the following columns:
```

"folder": name of the folder (e.g., 00a237...)

"phish_prediction": The script's final decision (phish | benign)

"F_brand_flag" (Signal B): True if a brand was detected AND it did not match the site's domain. False if it matched. None if no brand was detected or the check failed.

"F_intent_flag" (Signal C): True if the page was classified as having credential-stealing intent. False if not. None if the check was skipped (e.g., F_brand_flag was not True).

"brand_recog_time": time taken for brand recognition

"crp_prediction_time": time taken for CRP prediction

"crp_transition_time": time taken for CRP transition
```
