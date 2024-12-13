# MetabizDesign Geogeni

## Usage Guide
### Environment Setup
- Ubuntu
```bash
'''python version => 3.12.7'''

python -m venv venv
. venv/bin/activate
python -m pip install -r requirements.txt
```

- Window
```bash
'''python version => 3.12.7'''

python -m venv venv
. venv\Scripts\activate
python -m pip install -r requirements-window.txt
```

### Model Download
보내준 zip 파일 다운로드
- report_llm/llm/report_1_2/merged_model_2
- report_llm/llm/report_1_2/Qwen2.5-7B-Instruct_1epoch_1batch_53963
- report_llm/llm/report_3/qwen_merged_part3


### Dataset Download
- dataset => `A ~ Y_Z 데이터셋 폴더 추가`
- qa_llm/dataset => `기초설계.pdf ~ 흙이 분류와 다짐_2.pdf 파일 추가`

### Running Demos
```bash
streamlit run app/app.py
```