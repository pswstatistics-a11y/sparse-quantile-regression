import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
import statsmodels.api as sm
import warnings
import time

from src import generate_dgp
from src import sparse_quantile_reg_subgradient

"""
# 데이터 시나리오 시각화
from src import plot_noise_scenarios
plot_noise_scenarios(n=1000, p=10, rho=0.8)
"""
np.random.seed(42) 


# Statsmodels의 고차원 수렴 경고 로그 제거 (콘솔 시각성 확보)
warnings.filterwarnings('ignore')

# 1. 몬테카를로 성능 평가 지표 연산 장치(Metric Calculator)
def calculate_metrics(beta_true, beta_hat, threshold=1e-2):
    """
    수리통계학적 추정량의 성질을 계량화하는 함수
    - MSE : 효율성(Efficiency) 트래킹
    - TPR / FDR : 고차원 변수 선택의 일치성(Oracle Property) 트래킹
    """
    if np.any(np.isnan(beta_hat)) or np.any(np.isinf(beta_hat)):
        # 모델이 고장 났으므로 TPR, FDR, MSE 모두 깔끔하게 nan으로 통일해서 즉시 반환
        return np.nan, np.nan, np.nan
    
    beta_true = beta_true.flatten()
    beta_hat = beta_hat.flatten()
    
    # 참 변수(Signal)와 가짜 변수(Noise)의 인덱스 마스킹
    true_nonzero = (beta_true != 0)
    true_zero = (beta_true == 0)
    
    # 솔버의 수치적 근사치를 고려한 임계치 필터링 (Hard Thresholding)
    est_nonzero = (np.abs(beta_hat) > threshold)
    
    # TPR (참 변수 복원력): 진짜 중요한 변수 중 모델이 살려낸 비율
    tp = np.sum(true_nonzero & est_nonzero)
    tpr = tp / np.sum(true_nonzero) if np.sum(true_nonzero) > 0 else 0.0
    
    # FDR (허위 발견율): 모델이 중요하다고 뱉은 것 중 낀 가짜 변수의 비율
    fp = np.sum(true_zero & est_nonzero)
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # MSE (예측 및 추정 오차)
    mse = np.mean((beta_true - beta_hat) ** 2)
    
    return mse, tpr, fdr



# 2. [글로벌 무작위성 통제] 시드 (재현성 보장)
np.random.seed(42)

# 3. 몬테카를로 평행우주 아우터 루프 작동
def montecarlo(N, P, RHO, REPLICATIONS, SCENARIOS):
    results = []
    print(f" [N={N}, P={P}] 몬테카를로 스트레스 테스트 시뮬레이션 시작... (Loop: {REPLICATIONS}회)")

    for rep in range(REPLICATIONS):
        if (rep + 1) % 20 == 0:
            print(f"{rep + 1}/{REPLICATIONS}회차 연산 중...")
            
        for scenario in SCENARIOS:
            # [Step A] 데이터 공장에서 다양한 환경의 데이터를 독립 샘플링
            X, y, beta_true = generate_dgp(n=N, p=P, rho=RHO, noise_type=scenario)
            
            # ----------------------------------------------------
            # 1번: Custom Solver (Quantile + L1)
            # ----------------------------------------------------
            c, tau = 1.0, 0.5
            current_lambda = c * np.sqrt(tau * (1 - tau) * np.log(P) / N)
            beta_mine, mine_time = sparse_quantile_reg_subgradient(X, y, tau, lam=current_lambda, max_iter=1500)
            mse_m, tpr_m, fdr_m = calculate_metrics(beta_true, beta_mine)
            results.append({'P':P, 'Scenario': scenario, 'Model': '1. Custom_Sparse_Quantile', 'MSE': mse_m, 'TPR': tpr_m, 'FDR': fdr_m, 'time':mine_time})
            
            # ----------------------------------------------------
            # 2번: 고전 대조군 사이킷런 Lasso (L2 + L1)
            # ----------------------------------------------------
            lasso_model = Lasso(alpha=0.1, fit_intercept=False)
            start_t = time.perf_counter()
            try:
                lasso_model.fit(X, y)
                beta_lasso = lasso_model.coef_.reshape(-1, 1)
                mse_l, tpr_l, fdr_l = calculate_metrics(beta_true, beta_lasso)
                lasso_time = time.perf_counter() - start_t
            except:
                mse_l, tpr_l, fdr_l = np.nan, np.nan, np.nan
                lasso_time = np.nan

            results.append({'P':P, 'Scenario': scenario, 'Model': '2. Scikit_Learn_Lasso', 'MSE': mse_l, 'TPR': tpr_l, 'FDR': fdr_l, 'time':lasso_time})
            
            # ----------------------------------------------------
            # 3번: 정규화 없는 분위수 회귀 Statsmodels QuantReg (Quantile + No Penalty)
            # ----------------------------------------------------
            start_t = time.perf_counter()
            try:
                # 고차원의 저주 대비 예외 처리 스펙 적용
                quant_reg = sm.QuantReg(y, X).fit(q=0.5)
                beta_qr = quant_reg.params.reshape(-1, 1)
                mse_q, tpr_q, fdr_q = calculate_metrics(beta_true, beta_qr)
                quant_time = time.perf_counter() - start_t
            except:
                # 고차원 링(P>=N)에서 싱귤래리티 파산 시 성적표 오염 방지
                mse_q, tpr_q, fdr_q = np.nan, np.nan, np.nan
                quant_time = np.nan
                
            results.append({'P':P, 'Scenario': scenario, 'Model': '3. Unpenalized_QuantReg', 'MSE': mse_q, 'TPR': tpr_q, 'FDR': fdr_q, 'time':quant_time})

    print(" [System Machine] 전체 연산 종료. 결과 집계")

    return results

# 4. 글로벌 시뮬레이션 하이퍼파라미터 셋업
n = 100
rho = 0.8
replications = 100
scenarios = ['gaussian', 'cauchy', 'exponential']

# 성적표 원시 데이터(Raw Log) 저장소 
results_log = []

p = 20
results_log.extend(montecarlo(n, p, rho, replications, scenarios))
p = 100
results_log.extend(montecarlo(n, p, rho, replications, scenarios))
p = 200
results_log.extend(montecarlo(n, p, rho, replications, scenarios))



# 5. 대수의 법칙(LLN) 100개 성적의 표본평균 일괄 컴파일
df_raw = pd.DataFrame(results_log)
final_report = (
    df_raw.groupby(['P', 'Scenario', 'Model'])
    .agg(['mean', 'count'])
    .xs('mean', axis=1, level=1)
    .round(4)
)


# ==========
import os

# 1. 결과물을 저장할 폴더(outputs) 자동 생성
output_dir = "src/outputs"
os.makedirs(output_dir, exist_ok=True)

# 2. 판다스 데이터프레임 파일 내보내기 (기본 백업)
csv_path = os.path.join(output_dir, "monte_carlo_results.csv")
# xlsx_path = os.path.join(output_dir, "monte_carlo_results.xlsx")

# 인덱스(Scenario, Model)포함 저장
final_report.to_csv(csv_path, index=True)
# final_report.to_excel(xlsx_path, index=True)

print(f"📁 [System] CSV 파일 내보내기 완료: {csv_path}")
# print(f"📁 [System] 엑셀 파일 내보내기 완료: {xlsx_path}")

# 3. 데이터프레임을 LaTeX 논문용 표 수식으로 자동 변환
# 소수점 4자리까지 포맷팅해서 콘솔에 텍스트로 출력
# print(final_report.to_latex(float_format="%.4f"))

print("="*50)