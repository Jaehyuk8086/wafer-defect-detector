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

# Word / PDF 생성 모듈
import docx
from docx.shared import RGBColor
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 기본 페이지 레이아웃 설정
st.set_page_config(page_title="FDC AI Control & 8D Export System", layout="wide")

# GPU 가속 연산 설정 및 시드 고정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 1. 데이터 전처리 및 Conv1D Autoencoder 모델 학습 (캐싱)
# ==============================================================================
@st.cache_resource
def train_and_evaluate_model():
    df = pd.read_csv('semiconductor_wafer_defect_dataset.csv')
    sensor_cols = ['temperature_c', 'pressure_torr', 'gas_flow_sccm', 'etch_rate_nm_min', 'voltage_v', 'current_ma']

    df_encoded = pd.get_dummies(df, columns=['process_step'], drop_first=False)
    train_df = df_encoded[df_encoded['defect_label'] == 0]
    y_test = df_encoded['defect_label'].values

    scaler = RobustScaler()
    X_train_sensor = torch.FloatTensor(scaler.fit_transform(train_df[sensor_cols]))
    X_test_sensor = torch.FloatTensor(scaler.transform(df_encoded[sensor_cols]))

    X_train_3d = X_train_sensor.unsqueeze(2)  # [N, 6, 1]
    X_test_3d = X_test_sensor.unsqueeze(2)    # [N, 6, 1]

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

    if not anomaly_wafer_list:
        anomaly_wafer_list = ["감지된 이상 웨이퍼가 없습니다."]

    return df, sensor_cols, test_mae, feature_mae, upper_threshold, lower_threshold, y_pred, anomaly_wafer_list, y_test, loss_history

df, sensor_cols, test_mae, feature_mae, upper_threshold, lower_threshold, y_pred, anomaly_wafer_list, y_test, loss_history = train_and_evaluate_model()

# ==============================================================================
# 2. 8D 리포트 텍스트 및 PDF/DOCX 생성 함수
# ==============================================================================
def generate_8d_text(wafer_id, process_step, anomaly_score, upper_th, lower_th, top_sensor, top_sensor_ratio, actual_label):
    is_anomaly = (anomaly_score > upper_th) or (anomaly_score < lower_th)
    status_str = "CRITICAL ANOMALY DETECTED" if is_anomaly else "NORMAL PROCESS"

    return f"""================================================================================
                         8D PROBLEM SOLVING REPORT
================================================================================

[D0. 문제 발생 기본 정보 (General Information)]
- CAR 번호        : CAR-FDC-2026-0927
- 고객사 (Customer): SEC / Global Foundry
- 대상 제품 / Lot : Semiconductor Wafer (Wafer ID: #{wafer_id})
- 공정 단계        : {process_step}
- 발생 불량 (Defect): FDC Sensor Anomaly ({status_str} / Actual: {actual_label})
- 수량 및 스코어  : 1 Wafer (Score: {anomaly_score:.4f} | IQR Band: [{lower_th:.4f} ~ {upper_th:.4f}])
- 검출 영역       : In-FAB FDC Real-time Sensor Database

[D1. 문제 해결 팀 구성 (Team Approach)]
- 팀장 (Leader)   : FDC AI 시스템 모니터링 파트장
- 공정 엔지니어   : Process Engineer (PE)
- 설비 엔지니어   : Equipment Engineer (EE)
- 품질 엔지니어   : Quality Engineer (QE)

[D2. 정확한 문제 파악 (Problem Description - 5W 2H & IS/IS-NOT)]
- WHAT (무엇이)    : {process_step} 공정 중 주요 센서({top_sensor})의 이상 오차 감지
- WHERE (어디서)  : FDC 실시간 센서 모니터링 DB (In-FAB)
- WHEN (언제)     : Wafer #{wafer_id} 공정 진행 과정
- WHO (누가)      : Conv1D Autoencoder AI 이상 감지 모니터링 시스템
- WHY (왜)        : Anomaly Score({anomaly_score:.4f})가 IQR 신뢰 구간({lower_th:.4f} ~ {upper_th:.4f}) 이탈
- HOW (어떻게)    : FDC 센서 오차 기여도 분석 결과 [{top_sensor}] 센서가 {top_sensor_ratio:.1f}% 기여
- HOW MANY (얼마나): FDC Interlock 레벨 이상의 센서 변동 발생

[D3. 즉각적인 봉쇄 조치 (Problem Containing / Emergency Action)]
- 해당 Lot/Wafer 즉시 Hold 처리 및 설비 진행 일시 중단 (Interlock 유발)
- 후속 공정 영향도 평가 및 샘플 웨이퍼 비파괴/파괴 검사 진행

[D4. 근본 원인 규명 (Root Cause Analysis - 4M & AI RCA)]
- AI 자동 RCA 결과 : 주요 원인 센서 -> [{top_sensor}] (전체 이상 오차의 {top_sensor_ratio:.1f}% 기여)
- 4M 관점 상세 분석 :
  1) Machine (설비) : {top_sensor} 하드웨어 밸브/유량 제어 인터페이스 열화 또는 Calibration 오차
  2) Method (방법)  : 센서 Parameter 입력 범위 제한 미비 및 고정 Spec 적용에 따른 감지 지연
  3) Material (자재): 이전 공정 Wafer 상태 변동 가능성
  4) Men (사람)     : 작업자 파라미터 오적용 또는 알람 인식 미흡

[D5. 시정 조치 수립 및 시행 (Corrective Action Implementation)]
- 설비 조치 : {process_step} 단계의 {top_sensor} 하드웨어, 밸브 및 센서 인터페이스 점검 및 교체
- 시스템 개선: Parameter 오적용 예방을 위해 각 Parameter별 입력 가능 범위 제한 설정
- 인식률 개선: 이상 발생 시 시각/청각 Display 표시 및 인식 방법 보완 (Fool Proof 적용)

[D6. 시정 조치 유효성 검증 (Corrective Action Validation)]
- 조치 후 Sample Wafer 재실행 및 FDC 센서 Real-time 수치 모니터링
- Anomaly Score가 동적 IQR 신뢰 구간 내로 수렴함을 재검증

[D7. 재발 방지 대책 수립 (Recurrence Prevention)]
- 표준 문서(SOP) 반영 : {top_sensor} 점검 주기 정례화 및 작업 지침서 업데이트
- System 반영 : Soft Spec (6-Sigma 및 동적 IQR 기반) FDC 모니터링 시스템 자동화 고도화
- 타 라인 전파 : 동일 설비 및 유사 공정 챔버 대상 점검 항목 전파 및 공유

[D8. 결과 전달 및 승인 (Communication Result & Approvals)]
- 결과 전파 : FDC 시스템 등록 및 전 부서원 공유
- 최종 승인 : Process Engineer (PE) / Quality Engineer (QE) / QA Part Manager
================================================================================
"""

def export_8d_docx(report_text, wafer_id):
    filename = f"8D_Report_Wafer_{wafer_id}.docx"
    doc = docx.Document()
    title = doc.add_heading('8D PROBLEM SOLVING REPORT', level=0)
    title.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER

    for line in report_text.split('\n'):
        if line.startswith('[D'):
            h = doc.add_heading(line, level=2)
            for r in h.runs:
                r.font.color.rgb = RGBColor(0, 0, 0)
        elif not line.startswith('='):
            p = doc.add_paragraph(line)
            for r in p.runs:
                r.font.color.rgb = RGBColor(0, 0, 0)

    doc.save(filename)
    with open(filename, "rb") as f:
        return f.read()

def export_8d_pdf(report_text, wafer_id):
    filename = f"8D_Report_Wafer_{wafer_id}.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()

    body_style = ParagraphStyle(
        'ReportBodyText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=RGBColor(0, 0, 0)
    )
    heading_style = ParagraphStyle(
        'ReportHeadingText',
        parent=styles['Heading2'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=RGBColor(0, 0, 0)
    )
    title_style = ParagraphStyle(
        'ReportTitleText',
        parent=styles['Heading1'],
        fontName='Helvetica',
        fontSize=14,
        leading=18,
        textColor=RGBColor(0, 0, 0),
        alignment=1
    )

    story = [Paragraph("<b>8D PROBLEM SOLVING REPORT</b>", title_style)]
    story.append(HRFlowable(width="100%", thickness=1.5, color=RGBColor(0, 0, 0), spaceAfter=10))

    for line in report_text.split('\n'):
        line_clean = line.replace('<', '&lt;').replace('>', '&gt;')
        if line_clean.startswith('[D'):
            story.append(Spacer(1, 6))
            story.append(Paragraph(f"<b>{line_clean}</b>", heading_style))
        elif not (line_clean.startswith('=') or line_clean.startswith('-'*10)) and line_clean.strip():
            story.append(Paragraph(line_clean, body_style))

    doc.build(story)
    with open(filename, "rb") as f:
        return f.read()

# ==============================================================================
# 3. Streamlit 대시보드 UI
# ==============================================================================
st.title("🏭 반도체 FDC 공정 이상 감지 & 8D Report 자동 내보내기 시스템")

tab1, tab2 = st.tabs(["🔍 이상 웨이퍼 선택 & 표준 8D 리포트", "🖥️ AI 시스템 모니터링 (System Health)"])

with tab1:
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        selected_wafer_str = st.selectbox(
            "🚨 감지된 이상 웨이퍼 선택 (이상 발생 건만 표시)",
            options=anomaly_wafer_list
        )

    if selected_wafer_str and "없습니다" not in selected_wafer_str:
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

        # Matplotlib 시각화 차트
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

        ax1.scatter(range(len(test_mae)), test_mae, color='skyblue', s=4, alpha=0.5, label='All Wafers Scatter')
        ax1.axhline(upper_threshold, color='red', linestyle='--', linewidth=1.5, label=f'Upper IQR Thresh ({upper_threshold:.3f})')
        ax1.axhline(lower_threshold, color='orange', linestyle=':', linewidth=1.5, label=f'Lower IQR Thresh ({lower_threshold:.3f})')
        ax1.axhspan(lower_threshold, upper_threshold, color='green', alpha=0.06, label='Normal Band')

        dot_color = 'crimson' if is_anomaly else 'darkgreen'
        ax1.scatter(sample_idx, score, color=dot_color, s=120, zorder=6, edgecolors='black', linewidth=1.5)

        ax1.annotate(
            f'#{wafer_id}',
            (sample_idx, score),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=10,
            fontweight='bold',
            color='crimson' if is_anomaly else 'darkgreen',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=dot_color, lw=1)
        )

        ax1.set_title(f"Data Distribution & Dynamic Threshold (Wafer #{wafer_id})")
        ax1.set_xlabel("Wafer Index (Total 5,000)")
        ax1.set_ylabel("Anomaly Score (MAE)")
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        bars = ax2.barh(cause_df['Sensor'], cause_df['Ratio(%)'], color='crimson' if is_anomaly else 'skyblue')
        ax2.set_title(f"Root Cause Breakdown (%) - Wafer #{wafer_id}")
        ax2.set_xlabel("Contribution Ratio (%)")
        for bar in bars:
            w = bar.get_width()
            ax2.text(w + 0.5, bar.get_y() + bar.get_height()/2, f'{w:.1f}%', va='center', fontsize=9, color='black')
        ax2.set_xlim(0, max(cause_df['Ratio(%)']) + 15)
        ax2.grid(True, axis='x', linestyle='--', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("📄 표준 8D Report 미리보기")
        st.text_area("8D Report Text", report_8d, height=350)

        st.subheader("📥 문서 내보내기 설정 (Export Options)")
        col_d1, col_d2, col_d3 = st.columns(3)

        with col_d1:
            st.download_button(
                label="📄 TXT 파일 다운로드",
                data=report_8d,
                file_name=f"8D_Report_Wafer_{wafer_id}.txt",
                mime="text/plain"
            )
        with col_d2:
            docx_bytes = export_8d_docx(report_8d, wafer_id)
            st.download_button(
                label="📝 Word (.docx) 다운로드",
                data=docx_bytes,
                file_name=f"8D_Report_Wafer_{wafer_id}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        with col_d3:
            pdf_bytes = export_8d_pdf(report_8d, wafer_id)
            st.download_button(
                label="📕 PDF (.pdf) 다운로드",
                data=pdf_bytes,
                file_name=f"8D_Report_Wafer_{wafer_id}.pdf",
                mime="application/pdf"
            )

with tab2:
    total_processed = len(test_mae)
    detected_anomalies = int(np.sum(y_pred))
    anomaly_rate = (detected_anomalies / total_processed) * 100
    final_loss = loss_history[-1]

    health_status = "🟢 HEALTHY (정상 작동 중)" if final_loss < 0.01 and anomaly_rate < 5.0 else "⚠️ WARNING (점검 필요)"

    st.markdown(f"### 🟢 AI 시스템 헬스 상태: `{health_status}`")
    st.markdown(f"- **최종 모델 Loss (MSE):** `{final_loss:.5f}`")
    st.markdown(f"- **전처리 알고리즘:** `RobustScaler (Median / IQR 기반 아웃라이어 강건화)`")
    st.markdown(f"- **총 처리된 웨이퍼 샘플 수:** `{total_processed:,} 개`")
    st.markdown(f"- **동적 IQR Band (강화 기준):** `Upper ({upper_threshold:.4f})` | `Lower ({lower_threshold:.4f})`")
    st.markdown(f"- **AI 감지 이상 웨이퍼 수:** `{detected_anomalies} 개` (전체의 **{anomaly_rate:.2f}%**)")

    fig_health, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(loss_history, color='black', linewidth=2, marker='o', markersize=3)
    ax1.set_title("1) AI Model Loss Convergence")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE)")
    ax1.grid(True, alpha=0.3)

    df_temp = df.copy()
    df_temp['is_anomaly'] = y_pred
    step_anomaly_counts = df_temp.groupby('process_step')['is_anomaly'].sum()

    bars = ax2.bar(step_anomaly_counts.index, step_anomaly_counts.values, color='coral', alpha=0.85)
    ax2.set_title("2) Anomalies Count by Process Step")
    ax2.set_xlabel("Process Step")
    ax2.set_ylabel("Detected Anomalies")
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.2, f'{int(h)}', ha='center', va='bottom', fontsize=9, color='black')
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig_health)
