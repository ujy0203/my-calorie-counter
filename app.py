from __future__ import annotations

import os
import fitz  # PyMuPDF
import gradio as gr
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def extract_text_from_pdf(pdf_file) -> str:
    if pdf_file is None:
        return ""

    doc = fitz.open(pdf_file.name)
    text = ""

    for page in doc:
        text += page.get_text()

    doc.close()

    return text[:12000]


def analyze_paper(pdf_file):
    if pdf_file is None:
        return "PDF 논문 파일을 업로드해 주세요."

    paper_text = extract_text_from_pdf(pdf_file)

    if not paper_text.strip():
        return "PDF에서 텍스트를 추출하지 못했습니다. 스캔본 PDF일 가능성이 있습니다."

    if not GEMINI_API_KEY:
        return (
            "GEMINI_API_KEY가 설정되지 않았습니다.\n\n"
            "우선 PDF 텍스트 추출은 성공했습니다.\n\n"
            "추출된 텍스트 미리보기:\n\n"
            f"{paper_text[:2000]}"
        )

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
너는 대학생의 논문 발표를 도와주는 AI 발표 도우미다.
아래 논문 내용을 바탕으로 한국어로 정리하라.

반드시 다음 형식으로 답변하라.

# 1. 논문 핵심 요약
- 논문의 주제를 5문장 이내로 요약

# 2. 연구 목적
- 이 연구가 무엇을 밝히고자 했는지 설명

# 3. 연구 방법
- 연구 대상, 자료, 분석 방법을 쉽게 설명

# 4. 주요 결과
- 핵심 결과를 번호로 정리

# 5. 연구의 한계점
- 논문에서 보이는 한계 또는 주의할 점 정리

# 6. 발표 목차 추천
- 5분 발표용 목차를 5개 내외로 제안

# 7. 5분 발표 대본
- 실제 발표자가 읽을 수 있는 자연스러운 대본 작성

# 8. 예상 질문과 답변
- 발표 후 받을 수 있는 질문 3개와 답변 작성

논문 내용:
{paper_text}
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 분석 중 오류가 발생했습니다.\n\n오류 내용: {type(e).__name__}: {str(e)}"


CUSTOM_CSS = """
.gradio-container {
    background: #eef2f7 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif !important;
}

.main-wrap {
    max-width: 1120px;
    margin: 0 auto;
}

.hero-card {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
    color: white;
    border-radius: 26px;
    padding: 44px;
    margin: 28px 0 22px;
    box-shadow: 0 20px 48px rgba(30, 58, 138, 0.25);
}

.hero-badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 700;
    background: rgba(255,255,255,0.18);
    margin-bottom: 14px;
}

.hero-title {
    font-size: 40px;
    font-weight: 800;
    letter-spacing: -0.04em;
    margin: 0 0 12px;
}

.hero-desc {
    font-size: 17px;
    line-height: 1.75;
    opacity: 0.95;
    max-width: 760px;
    margin: 0;
}

.guide-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 18px 0 24px;
}

.guide-box {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 18px;
    min-height: 112px;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
}

.guide-box strong {
    color: #1e3a8a;
    display: block;
    margin-bottom: 8px;
    font-size: 15px;
}

.guide-box span {
    color: #64748b;
    font-size: 14px;
    line-height: 1.55;
}

.upload-panel {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 24px;
    padding: 26px;
    box-shadow: 0 16px 36px rgba(15, 23, 42, 0.08);
    margin-bottom: 24px;
}

.result-panel {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 24px;
    padding: 26px;
    box-shadow: 0 16px 36px rgba(15, 23, 42, 0.08);
}

button.primary {
    background: #2563eb !important;
    border-radius: 14px !important;
    font-weight: 800 !important;
}

.footer-note {
    color: #64748b;
    text-align: center;
    font-size: 13px;
    margin: 28px 0 10px;
}

@media (max-width: 900px) {
    .guide-grid {
        grid-template-columns: 1fr 1fr;
    }
    .hero-title {
        font-size: 32px;
    }
}

@media (max-width: 640px) {
    .guide-grid {
        grid-template-columns: 1fr;
    }
    .hero-card {
        padding: 30px;
    }
}
"""


def build_ui():
    with gr.Blocks(css=CUSTOM_CSS, title="AI 논문 발표 도우미") as demo:
        with gr.Column(elem_classes=["main-wrap"]):
            gr.HTML(
                """
                <section class="hero-card">
                    <div class="hero-badge">📄 Paper Presentation Assistant</div>
                    <h1 class="hero-title">AI 논문 발표 도우미</h1>
                    <p class="hero-desc">
                        논문 PDF를 업로드하면 AI가 연구 목적, 연구 방법, 주요 결과, 한계점을 정리하고
                        발표 목차, 5분 발표 대본, 예상 질문과 답변까지 생성합니다.
                    </p>
                </section>

                <section class="guide-grid">
                    <div class="guide-box">
                        <strong>📌 핵심 요약</strong>
                        <span>논문의 주제와 전체 흐름을 빠르게 파악할 수 있게 정리합니다.</span>
                    </div>
                    <div class="guide-box">
                        <strong>🎯 연구 목적</strong>
                        <span>이 연구가 무엇을 밝히고자 했는지 발표용 문장으로 정리합니다.</span>
                    </div>
                    <div class="guide-box">
                        <strong>🧪 연구 방법</strong>
                        <span>연구 대상, 자료, 분석 방법을 쉽게 설명합니다.</span>
                    </div>
                    <div class="guide-box">
                        <strong>🎤 발표 대본</strong>
                        <span>실제 발표자가 읽을 수 있는 5분 분량의 대본을 생성합니다.</span>
                    </div>
                </section>
                """
            )

            with gr.Group(elem_classes=["upload-panel"]):
                gr.Markdown("## 논문 PDF 업로드")
                gr.Markdown("분석할 논문 PDF 파일을 업로드한 뒤 아래 버튼을 눌러 주세요.")
                pdf_input = gr.File(
                    label="PDF 파일 선택",
                    file_types=[".pdf"]
                )
                submit_btn = gr.Button("논문 분석하기", variant="primary")

            with gr.Group(elem_classes=["result-panel"]):
                gr.Markdown("## AI 분석 결과")
                output = gr.Markdown(
                    value="PDF를 업로드하고 분석 버튼을 누르면 결과가 여기에 표시됩니다."
                )

            submit_btn.click(
                fn=analyze_paper,
                inputs=pdf_input,
                outputs=output,
            )

            gr.HTML(
                """
                <p class="footer-note">
                    ※ 결과는 발표 준비를 돕기 위한 AI 생성 초안입니다. 실제 발표 전 논문 원문과 반드시 대조해 주세요.
                </p>
                """
            )

    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        show_api=False,
        ssr_mode=False,
    )