# src/visualize.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .dgp import generate_dgp  # 패키지 내부 참조

def plot_noise_scenarios(n=1000, p=10, rho=0.8):
    """3가지 노이즈 데이터 분포 시각화"""
    np.random.seed(42)
    
    # 데이터 배출
    X_gauss, y_gauss, _ = generate_dgp(n=n, p=p, rho=rho, noise_type='gaussian')
    X_cauchy, y_cauchy, _ = generate_dgp(n=n, p=p, rho=rho, noise_type='cauchy')
    X_exp, y_exp, _ = generate_dgp(n=n, p=p, rho=rho, noise_type='exponential')
    
    # 캔버스 오픈 및 렌더링
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.set_theme(style="whitegrid")
    
    # Scenario A
    sns.histplot(y_gauss.flatten(), kde=True, ax=axes[0], color='royalblue', stat="density")
    axes[0].set_title("Scenario A: Gaussian Baseline\n(Thin-tailed, Symmetric)", fontsize=12, fontweight='bold')
    
    # Scenario B
    # 범위(-20 ~ 20) 안의 데이터만 넘파이 불리언 인덱싱으로 필터링
    cauchy_filtered = y_cauchy[(y_cauchy >= -20) & (y_cauchy <= 20)]
    # bins=50을 명시해서 -20~20 사이를 50개의 촘촘한 막대로 쪼개도록 강제합니다.
    sns.histplot(cauchy_filtered, kde=True, ax=axes[1], color='crimson', stat="density", bins=200)
    axes[1].set_xlim(-20, 20)  # 아웃라이어 방어장치
    axes[1].set_title("Scenario B: Heavy-tailed Cauchy\n(Infinite Variance & Outliers)", fontsize=12, fontweight='bold')
    
    # Scenario C
    sns.histplot(y_exp.flatten(), kde=True, ax=axes[2], color='darkorange', stat="density")
    axes[2].set_title("Scenario C: Asymmetric Exponential\n(Skewed/Non-Gaussian)", fontsize=12, fontweight='bold')
    
    plt.suptitle("Stress Testing Sandbox: Target Distribution (y) across Contamination Scenarios", 
                 fontsize=15, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    # 노트북 화면에 그래프를 띄우기 위해 plt.show() 호출
    plt.show()