import subprocess
import sys

def install_and_import(package_list):
    for package in package_list:
        try:
            __import__(package.split('-')[0]) 
        except ImportError:
            print(f"[{package}] 설치 필요. 설치 중.")
            # 터미널 명령어를 파이썬 내부에서 실행
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"[{package}] 설치 완료")

# install_and_import(["scikit-learn","statsmodels"])

install_and_import(["jinja2"])