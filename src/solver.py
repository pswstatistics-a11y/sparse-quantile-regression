# src/solver.py
import time
import numpy as np

def sparse_quantile_reg_subgradient(X, y, tau=0.5, lam=0.1, lr=1.5, max_iter=2000, clip_val=5.0, tol=1e-4):
    """
    서브그래디언트 경사하강법 기반의 L1-벌점화 분위수 회귀분석 솔버 Engine
    
    Parameters:
    - X : (n, p) 독립변수 행렬
    - y : (n, 1) 타깃 변수 세로 벡터
    - tau : 타깃 분위수 (중앙값 추정은 0.5)
    - lam : L1 정규화 강도 (Lasso 페널티 파라미터)
    - lr : 초기 학습률
    - max_iter : 최적화 루프 최대 반복 횟수
    - clip_val : 코시 노이즈 폭발 방지용 그래디언트 클리핑 임계치
    """
    n, p = X.shape

    # y를 (150, 1)로 확실히 변환
    y = y.reshape(-1, 1)
    beta = np.zeros((p, 1)) # Shape: (p, 1) 세로 벡터
    
    start_time = time.perf_counter()
    converged_iter = max_iter

    for it in range(max_iter):

        # 2. 이번 루프의 실시간 잔차(Residual) 연산 -> Shape: (n, 1)
        residual = y - X @ beta
        
        # 3. 체크 손실 함수 서브그래디언트 유도
        # 잔차가 0보다 작으면(True=1) 1-tau, 0보다 크면(False=0) -tau 
        psi = np.where(residual < 0, 1.0 - tau, -tau)  # Shape: (n, 1)
        loss_subgrad = (X.T @ psi) / n            # Shape: (p, 1)
        loss_subgrad = np.clip(loss_subgrad, -clip_val, clip_val) 
        
        # 수렴 보조 
        current_lr = lr / np.sqrt(it*0.005 + 1)

        beta_temp = beta - current_lr * loss_subgrad 
        beta_next = np.sign(beta_temp) * np.maximum(0, np.abs(beta_temp) - current_lr * lam) 

        if np.any(np.isnan(beta_next)) or np.any(np.isinf(beta_next)):
            beta = np.ones(p) * np.inf # 수치 파산을 기록
            elapsed_time = time.perf_counter() - start_time
            return np.ones(p) * np.inf, elapsed_time
        
        if np.linalg.norm(beta_next - beta) < tol:
            beta = beta_next
            break

        beta = beta_next
        
    elapsed_time = time.perf_counter() - start_time
    return beta, elapsed_time
