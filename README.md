# MetabizDesign Geogeni

## Usage Guide
### Environment Setup
- 위에꺼는 기존 서버에서 실행한 환경이고, 아래꺼는 새로운 서버에서 실행한 환경이에요.
```bash
'''python version => 3.8.19'''

python -m venv venv
. venv/bin/activate
pip install -r requirements308.txt
```

```bash
'''python version => 3.11.0'''
'''cuda version => 12.1'''

conda create -n mbd_geogeni -y python==3.11
conda activate mbd_geogeni
pip install -r requirements311.txt
```

### Dataset Download
- dataset => `A ~ Y_Z 데이터셋 폴더 추가`

### Running Demos
```bash
streamlit run app/app.py
```