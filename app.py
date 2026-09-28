import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler

import docx
from docx.shared import RGBColor
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 페이지 기본 설정
st.set_page_config(page_title="FDC AI Control & 8D Export System", layout="wide")

# GPU 가속 연산 설정 및 시드 고정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 1. 데이터 전처리 & 모델 학습 (캐싱으로 고속 로딩)
# ==============================================================================
@st.cache_data
def load_and_train_model():
    df = pd.read_csv('semiconductor_wafer_defect_dataset.csv')
    sensor_cols = ['temperature_c', 'pressure_torr', 'gas_flow_sccm', 'etch_rate_nm_min', 'voltage_v', 'current_ma']

    df_encoded = pd.get_dummies(df, columns=['process_step'], drop_first=False)
    train_df = df_encoded[df_encoded['defect_label'] == 0]
    y_test = df_encoded['defect_label'].values

    scaler = RobustScaler()
    X_train_sensor = torch.FloatTensor(scaler.fit_transform(train_df[sensor_cols]))
    X_test_sensor = torch.FloatTensor(scaler.transform(df_encoded[sensor_cols]))

    X_train_3d = X_train_sensor.unsqueeze(2)
    X_test_3d = X_test_sensor.unsqueeze(2)

    train_dataset = TensorDataset(X_train_3d)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

    class Conv1DAutoencoder(nn.Module):
        def __init__(self, in_channels):
            super(Conv1DAutoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels, 16, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(16, 8, kernel_size=1),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Conv1d(8, 16, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(16, in_channels, kernel_size=1)
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = Conv1DAutoencoder(in_channels=len(sensor_cols)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    loss_history = []
    epochs = 40
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            s_batch = batch[0].to(device)
            optimizer.zero_grad()
            output = model(s_batch)
            loss = criterion(output, s_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * s_batch.size(0)

        avg_loss = epoch_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_3d.to(device))
        train_mae = torch.mean(torch.abs(train_pred - X_train_3d.to(device)), dim=(1, 2)).cpu().numpy()

        q25, q75 = np.percentile(train_mae, [25, 75])
        iqr = q75 - q25

        upper_threshold = q75 + 1.0 * iqr
        lower_threshold = max(0.0, q25 - 1.0 * iqr)

        test_pred = model(X_test_3d.to(device))
        feature_mae = torch.abs(test_pred - X_test_3d.to(device)).squeeze(2).cpu().numpy()
        test_mae = np.mean(feature_mae, axis=1)

    y_pred = ((test_mae > upper_threshold) | (test_mae < lower_threshold)).astype(int)
    detected_anomaly_indices = np.where(y_pred == 1)[0]
    anomaly_wafer_list = [f"Wafer #{df.iloc[idx]['wafer_id']} (Index: {idx+1})" for idx in detected_anomaly_indices]

    return df, sensor_cols, test_mae, feature_mae, upper_threshold, lower_threshold, y_pred, anomaly_wafer_list, y_test

df, sensor_cols, test_mae, feature_mae, upper_threshold, lower_threshold, y_pred, anomaly_wafer_list, y_test = load_and_train_model()

# ==============================================================================
# 2. 리포트 생성 및 파일 내보내기 모듈
# ==============================================================================
def generate_8d_text(wafer_id, process_step, anomaly_score, upper_th, lower_th, top_sensor, top_sensor_ratio, actual_label):
    is_anomaly = (anomaly_score > upper_th) or (anomaly_score < lower_th)
    status_str = "CRITICAL ANOMALY DETECTED" if is_anomaly else "NORMAL PROCESS"

    return f"""================================================================================
                         8D PROBLEM SOLVING REPORT
================================================================================

[D0. 문제 발생 기본 정보 (General Information)]
- CAR 번호        : CAR-FDC-2026-0927
- 대상 제품 / Lot : Semiconductor Wafer (Wafer ID: #{wafer_id})
- 공정 단계        : {process_step}
- 발생 불량 (Defect): FDC Sensor Anomaly ({status_str} / Actual: {actual_label})
- 수량 및 스코어  : 1 Wafer (Score: {anomaly_score:.4f} | IQR Band: [{lower_th:.4f} ~ {upper_th:.4f}])

[D1. 문제 해결 팀 구성 (Team Approach)]
- 팀장 (Leader)   : FDC AI 시스템 모니터링 파트장
- 구성원          : Process Engineer (PE), Equipment Engineer (EE), Quality Engineer (QE)

[D2. 정확한 문제 파악 (Problem Description)]
- WHAT           : {process_step} 공정 중 주요 센서({top_sensor})의 이상 오차 감지
- WHY            : Anomaly Score({anomaly_score:.4f})가 IQR 신뢰 구간 이탈
- HOW            : FDC 센서 오차 기여도 분석 결과 [{top_sensor}] 센서가 {top_sensor_ratio:.1f}% 기여

[D3. 즉각적인 봉쇄 조치 (Problem Containing)]
- 해당 Lot/Wafer 즉시 Hold 처리 및 설비 진행 일시 중단 (Interlock 유발)

[D4. 근본 원인 규명 (Root Cause Analysis)]
- AI 자동 RCA 결과 : 주요 원인 센서 -> [{top_sensor}] ({top_sensor_ratio:.1f}% 기여)

[D5. 시정 조치 수립 및 시행 (Corrective Action)]
- 설비 조치 : {process_step} 단계의 {top_sensor} 하드웨어 점검 및 교체

[D6. 시정 조치 유효성 검증 (Validation)]
- Anomaly Score가 동적 IQR 신뢰 구간 내로 수렴함을 재검증

[D7. 재발 방지 대책 수립 (Recurrence Prevention)]
- 표준 문서(SOP) 반영 : {top_sensor} 점검 주기 정례화

[D8. 결과 전달 및 승인 (Approvals)]
- 최종 승인 : Process Engineer (PE) / Quality Engineer (QE)
================================================================================
"""

def export_8d_docx(report_text, wafer_id):
    filename = f"8D_Report_Wafer_{wafer_id}.docx"
    doc = docx.Document()
    title = doc.add_heading('8D PROBLEM SOLVING REPORT', level=0)
    for line in report_text.split('\n'):
        if line.startswith('[D'):
            doc.add_heading(line, level=2)
        elif not line.startswith('='):
            doc.add_paragraph(line)
    doc.save(filename)
    with open(filename, "rb") as f:
        return f.read()

# ==============================================================================
# 3. Streamlit UI 화면 구성
# ==============================================================================
st.title("🏭 반도체 FDC 공정 이상 감지 & 8D Report 자동 내보내기")

tab1, tab2 = st.tabs(["🔍 이상 웨이퍼 선택 & 8D 리포트", "🖥️ AI 시스템 모니터링"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_wafer_str = st.selectbox("🚨 감지된 이상 웨이퍼 선택", options=anomaly_wafer_list)
    
    if selected_wafer_str:
        wafer_id = int(selected_wafer_str.split('#')[1].split(' ')[0])
        sample_idx = wafer_id - 1

        wafer_info = df.iloc[sample_idx]
        score = test_mae[sample_idx]
        is_anomaly = (score > upper_threshold) or (score < lower_threshold)
        actual_label = "Defect" if y_test[sample_idx] == 1 else "Normal"

        s_mae = feature_mae[sample_idx]
        s_ratio = (s_mae / np.sum(s_mae)) * 100
        cause_df = pd.DataFrame({'Sensor': sensor_cols, 'Ratio(%)': s_ratio}).sort_values(by='Ratio(%)', ascending=True)
        top_sensor = cause_df.iloc[-1]

        report_8d = generate_8d_text(
            wafer_id=wafer_id,
            process_step=wafer_info['process_step'],
            anomaly_score=score,
            upper_th=upper_threshold,
            lower_th=lower_threshold,
            top_sensor=top_sensor['Sensor'],
            top_sensor_ratio=top_sensor['Ratio(%)'],
            actual_label=actual_label
        )

        # 시각화 차트
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.scatter(range(len(test_mae)), test_mae, color='skyblue', s=4, alpha=0.5)
        ax1.axhline(upper_threshold, color='red', linestyle='--')
        ax1.axhline(lower_threshold, color='orange', linestyle=':')
        ax1.scatter(sample_idx, score, color='crimson' if is_anomaly else 'darkgreen', s=100)
        ax1.set_title(f"Data Distribution (Wafer #{wafer_id})")

        ax2.barh(cause_df['Sensor'], cause_df['Ratio(%)'], color='crimson' if is_anomaly else 'skyblue')
        ax2.set_title("Root Cause Breakdown (%)")
        plt.tight_layout()
        st.pyplot(fig)

        st.text_area("📄 8D Report 미리보기", report_8d, height=300)

        # 문서 다운로드 버튼
        st.subheader("📥 8D Report 문서 다운로드")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                label="📄 TXT 리포트 다운로드",
                data=report_8d,
                file_name=f"8D_Report_Wafer_{wafer_id}.txt",
                mime="text/plain"
            )
        with col_d2:
            docx_data = export_8d_docx(report_8d, wafer_id)
            st.download_button(
                label="📝 Word 문서 (.docx) 다운로드",
                data=docx_data,
                file_name=f"8D_Report_Wafer_{wafer_id}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

with tab2:
    st.markdown(f"### 🟢 AI 시스템 상태 정보")
    st.write(f"- **총 웨이퍼 샘플 수:** {len(test_mae):,} 개")
    st.write(f"- **감지된 이상 웨이퍼 수:** {len(anomaly_wafer_list)} 개")
    st.write(f"- **동적 IQR Threshold:** Upper ({upper_threshold:.4f}) | Lower ({lower_threshold:.4f})")