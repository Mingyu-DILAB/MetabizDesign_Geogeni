# MetabizDesign Geogeni

## Usage Guide
### Environment Setup
- 위에꺼는 기존 서버에서 실행한 환경이고, 아래꺼는 새로운 서버에서 실행한 환경이에요.
```bash
'''python version => 3.8.19'''

python -m venv venv
. venv/bin/activate
python -m pip install -r requirements.txt
```

```bash
'''python version => 3.12.7'''
'''cuda version => 12.1'''

python -m venv venv
. venv/bin/activate
python -m pip install -r requirements2.txt
```

### Dataset Download
- dataset => `A ~ Y_Z 데이터셋 폴더 추가`

### Running Demos
```bash
streamlit run app/app.py
```