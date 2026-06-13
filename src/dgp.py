import numpy as np
from typing import Literal

def generate_toeplitz_x(n, p, rho):
    # distance_matrix[i, j] = |i - j|
    idx = np.arange(p)
    distance_matrix = np.abs(idx[:, None] - idx)
    
    # 토플리츠 공분산 행렬 Sigma 생성 (rho^|i-j|)
    Sigma = rho ** distance_matrix
    
    # 3. 평균이 0이고, 공분산이 Sigma인 n x p 행렬을 한 방에 샘플링!
    mean = np.zeros(p)
    X = np.random.multivariate_normal(mean, Sigma, size=n)
    
    return X

def generate_noise(n, noise_type:Literal['gaussian', 'cauchy', 'exponential'] = 'gaussian'):
    allowed_noises = ['gaussian', 'cauchy', 'exponential']
    
    # 3. 사용자가 오타를 내거나 이상한 값을 넣었을 때 에러와 함께 목록 명시
    if noise_type not in allowed_noises:
        raise ValueError(
            f"부적절한 분포: '{noise_type}'. "
            f"다음 분포 중 선택: {allowed_noises}"
        )
    
    if noise_type == 'gaussian':
        return np.random.normal(loc=0.0, scale=1.0, size=(n, 1))
    elif noise_type == 'cauchy':
        return np.random.standard_cauchy(size=(n, 1))
    elif noise_type == 'exponential':
        return np.random.exponential(scale=1.0, size=(n, 1)) - 1.0
    
    return

def generate_dgp(n, p, rho, noise_type='gaussian'):
    X = generate_toeplitz_x(n,p,rho)
    
    # 정답 모수(beta_true), X1,X2,X5
    beta_true = np.zeros(p)
    beta_true = beta_true.reshape(-1, 1)
    
    beta_true[0] = 3.0    
    beta_true[1] = 1.5    
    beta_true[4] = -2.0   
    # beta_true[2], [3], [5]~[p] 전부 0, 가짜 노이즈 변수들
    
    # noise_type에 따라 오차항(epsilon) 생성
    epsilon = generate_noise(n,noise_type)
    
    y = X @ beta_true + epsilon

    # print(f"‼️‼️‼️‼️‼️‼️‼️‼️‼️데이터 생성중 y:{y.shape} X:{X.shape}, beta:{beta_true.shape}, epsilon:{epsilon.shape}‼️‼️‼️‼️‼️‼️‼️‼️‼️")

    return X, y, beta_true

# 추천 모수 n=100, p=20,100,500, rho=0.8, noise_type=gaussian, cauchy, expontetial