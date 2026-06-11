from __future__ import annotations

import html
import os

import fitz  # PyMuPDF
import google.generativeai as genai
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def get_pdf_path(pdf_file) -> str:
    if pdf_file is None:
        return ""
    return getattr(pdf_file, "name", str(pdf_file))


def extract_text_from_pdf(pdf_file) -> str:
    pdf_path = get_pdf_path(pdf_file)
    if not pdf_path:
        return ""

    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text()

    doc.close()
    return text[:12000]


def get_file_info(pdf_file):
    if pdf_file is None:
        return """
        <div class="empty-file">
          <div class="empty-icon">📄</div>
          <strong>아직 업로드된 PDF가 없습니다.</strong>
          <span>분석할 논문 PDF 파일을 업로드해 주세요.</span>
        </div>
        """

    pdf_path = get_pdf_path(pdf_file)
    file_name = html.escape(os.path.basename(pdf_path))
    file_size = os.path.getsize(pdf_path) / (1024 * 1024)

    return f"""
    <div class="file-preview-custom">
      <div class="file-icon-custom">PDF</div>
      <div class="file-info-custom">
        <strong>{file_name}</strong>
        <span>{file_size:.1f} MB · 업로드 완료</span>
      </div>
    </div>
    """


def analyze_paper(pdf_file):
    if pdf_file is None:
        return (
            "<span class='status-warn'>PDF 필요</span>",
            """
            <div class="result-empty">
              <div class="big">📄</div>
              <strong>PDF 논문 파일을 업로드해 주세요.</strong>
              <p>업로드 후 분석 버튼을 누르면 결과가 여기에 표시됩니다.</p>
            </div>
            """
        )

    paper_text = extract_text_from_pdf(pdf_file)

    if not paper_text.strip():
        return (
            "<span class='status-error'>추출 실패</span>",
            """
            <div class="result-empty">
              <div class="big">⚠️</div>
              <strong>PDF에서 텍스트를 추출하지 못했습니다.</strong>
              <p>스캔본 PDF이거나 이미지 기반 문서일 가능성이 있습니다.</p>
            </div>
            """
        )

    if not GEMINI_API_KEY:
        preview = html.escape(paper_text[:2000])
        return (
            "<span class='status-error'>API 키 없음</span>",
            f"""
            <div class="result-empty">
              <div class="big">🔑</div>
              <strong>GEMINI_API_KEY가 설정되지 않았습니다.</strong>
              <p>PDF 텍스트 추출은 성공했습니다. 아래는 추출된 텍스트 미리보기입니다.</p>
              <pre class="preview-box">{preview}</pre>
            </div>
            """
        )

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
너는 대학생의 논문 발표를 도와주는 AI 발표 도우미다.
아래 논문 내용을 바탕으로 한국어로 정리하라.

반드시 다음 형식으로 답변하라.
마크다운 형식으로 보기 좋게 작성하라.
중요한 내용은 발표자가 바로 사용할 수 있도록 구체적으로 작성하라.

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
        return "<span class='status-ok'>분석 완료</span>", response.text
    except Exception as e:
        return (
            "<span class='status-error'>분석 오류</span>",
            f"""
            <div class="result-empty">
              <div class="big">⚠️</div>
              <strong>AI 분석 중 오류가 발생했습니다.</strong>
              <p>{type(e).__name__}: {html.escape(str(e))}</p>
            </div>
            """
        )


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
    border-radius: 28px;
    padding: 42px 46px;
    margin: 28px 0 22px;
    box-shadow: 0 22px 46px rgba(30, 58, 138, 0.25);
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.24);
    padding: 7px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 700;
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
    line-height: 1.7;
    opacity: 0.95;
    max-width: 760px;
    margin: 0;
}

.guide-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 18px 0 28px;
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

.main-grid {
    display: grid;
    grid-template-columns: 390px 1fr;
    gap: 22px;
    align-items: start;
}

.clean-card {
    background: white !important;
    border: 1px solid #dbe4ef !important;
    border-radius: 26px !important;
    padding: 26px !important;
    box-shadow: 0 16px 36px rgba(15, 23, 42, 0.08) !important;
}

.clean-card h2 {
    margin: 0 0 8px;
    font-size: 23px;
    color: #0f172a;
}

.card-desc {
    margin: 0 0 22px;
    color: #64748b;
    font-size: 14px;
    line-height: 1.65;
}

/* 실제 Gradio File 컴포넌트를 업로드 박스처럼 보이게 수정 */
#paper-upload {
    border: 2px dashed #93c5fd !important;
    background: #f8fbff !important;
    border-radius: 22px !important;
    padding: 24px !important;
    min-height: 210px !important;
}

#paper-upload:hover {
    border-color: #2563eb !important;
    background: #eff6ff !important;
}

#paper-upload button,
#paper-upload .button {
    background: white !important;
    color: #1d4ed8 !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
}

#paper-upload * {
    color: #334155 !important;
}

#paper-upload [class*="file"],
#paper-upload [data-testid*="file"] {
    background: transparent !important;
}

#paper-upload [class*="preview"],
#paper-upload [class*="File"],
#paper-upload [class*="file-preview"] {
    background: #f8fafc !important;
    color: #0f172a !important;
    border-color: #e2e8f0 !important;
}

.file-preview-custom {
    margin-top: 16px;
    padding: 16px;
    border-radius: 18px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    display: flex;
    gap: 14px;
    align-items: center;
}

.file-icon-custom {
    width: 46px;
    height: 46px;
    border-radius: 14px;
    background: #fee2e2;
    color: #dc2626;
    display: grid;
    place-items: center;
    font-size: 13px;
    font-weight: 900;
    flex: 0 0 auto;
}

.file-info-custom {
    min-width: 0;
    flex: 1;
}

.file-info-custom strong {
    display: block;
    font-size: 14px;
    color: #0f172a;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.file-info-custom span {
    font-size: 13px;
    color: #64748b;
}

.empty-file {
    margin-top: 16px;
    padding: 22px;
    border-radius: 18px;
    background: #f8fafc;
    border: 1px dashed #cbd5e1;
    text-align: center;
    color: #64748b;
}

.empty-file .empty-icon {
    font-size: 30px;
    margin-bottom: 8px;
}

.empty-file strong {
    display: block;
    color: #334155;
    margin-bottom: 4px;
}

button.primary {
    background: linear-gradient(135deg, #2563eb, #1e40af) !important;
    color: white !important;
    border-radius: 16px !important;
    font-weight: 800 !important;
    font-size: 16px !important;
    padding: 14px 18px !important;
    box-shadow: 0 12px 26px rgba(37, 99, 235, 0.25) !important;
}

.result-card {
    min-height: 540px;
}

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: start;
    gap: 12px;
    margin-bottom: 20px;
}

.status-ok,
.status-warn,
.status-error,
.status-ready,
.status-loading {
    display: inline-block;
    padding: 6px 11px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 800;
    white-space: nowrap;
}

.status-ok {
    background: #ecfdf5;
    color: #047857;
}

.status-ready {
    background: #eff6ff;
    color: #1d4ed8;
}

.status-loading {
    background: #eff6ff;
    color: #1d4ed8;
}

.status-warn {
    background: #fffbeb;
    color: #b45309;
}

.status-error {
    background: #fef2f2;
    color: #dc2626;
}

.result-empty {
    border: 1px dashed #cbd5e1;
    background: #f8fafc;
    border-radius: 22px;
    padding: 42px 24px;
    text-align: center;
    color: #64748b;
}

.result-empty .big {
    font-size: 40px;
    margin-bottom: 10px;
}

.result-empty strong {
    display: block;
    color: #334155;
    font-size: 17px;
    margin-bottom: 6px;
}

.preview-box {
    margin-top: 16px;
    text-align: left;
    background: #0f172a;
    color: #e2e8f0;
    border-radius: 14px;
    padding: 16px;
    max-height: 260px;
    overflow: auto;
    white-space: pre-wrap;
}

.loading-card {
    border: 1px solid #bfdbfe;
    background: #eff6ff;
    border-radius: 22px;
    padding: 28px;
    display: flex;
    gap: 18px;
    align-items: center;
}

.spinner {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    border: 5px solid #bfdbfe;
    border-top-color: #2563eb;
    animation: spin 1s linear infinite;
    flex: 0 0 auto;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.loading-card strong {
    display: block;
    color: #1e3a8a;
    font-size: 17px;
    margin-bottom: 4px;
}

.loading-card span {
    color: #475569;
    font-size: 14px;
}

.progress {
    margin-top: 14px;
    height: 9px;
    background: #dbeafe;
    border-radius: 999px;
    overflow: hidden;
}

.progress div {
    height: 100%;
    background: linear-gradient(90deg, #2563eb, #60a5fa);
    border-radius: inherit;
    animation: progress 1.6s ease-in-out infinite alternate;
}

@keyframes progress {
    from { width: 32%; }
    to { width: 86%; }
}

/* 분석 결과 가독성 개선 */
#analysis-output,
#analysis-output * {
    color: #1f2937 !important;
}

#analysis-output h1,
#analysis-output h2,
#analysis-output h3 {
    color: #0f172a !important;
    font-weight: 800 !important;
    margin-top: 1.2em !important;
}

#analysis-output p,
#analysis-output li {
    color: #334155 !important;
    line-height: 1.8 !important;
    font-size: 15px !important;
}

#analysis-output ul,
#analysis-output ol {
    color: #334155 !important;
}

.footer-note {
    color: #64748b;
    text-align: center;
    font-size: 13px;
    margin: 28px 0 10px;
}

@media (max-width: 960px) {
    .guide-grid,
    .main-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 760px) {
    .guide-grid {
        grid-template-columns: 1fr 1fr;
    }
    .hero-card {
        padding: 32px;
    }
    .hero-title {
        font-size: 32px;
    }
}

@media (max-width: 560px) {
    .guide-grid {
        grid-template-columns: 1fr;
    }
}
"""


LOADING_HTML = """
<div class="loading-card">
  <div class="spinner"></div>
  <div>
    <strong>논문을 분석하는 중입니다</strong>
    <span>PDF 텍스트를 추출하고 Gemini가 발표 자료를 생성하고 있어요. 잠시만 기다려 주세요.</span>
    <div class="progress"><div></div></div>
  </div>
</div>
"""

EMPTY_RESULT_HTML = """
<div class="result-empty">
  <div class="big">📄</div>
  <strong>분석 결과가 아직 없습니다.</strong>
  <p>PDF를 업로드하고 분석 버튼을 누르면 결과가 여기에 표시됩니다.</p>
</div>
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
                        <span>논문의 주제와 전체 흐름을 빠르게 정리합니다.</span>
                    </div>
                    <div class="guide-box">
                        <strong>🎯 연구 목적</strong>
                        <span>발표자가 설명하기 쉬운 문장으로 정리합니다.</span>
                    </div>
                    <div class="guide-box">
                        <strong>🧪 연구 방법</strong>
                        <span>대상, 자료, 분석 방법을 쉽게 풀어냅니다.</span>
                    </div>
                    <div class="guide-box">
                        <strong>🎤 발표 대본</strong>
                        <span>5분 발표용 자연스러운 대본을 생성합니다.</span>
                    </div>
                </section>
                """
            )

            with gr.Row(elem_classes=["main-grid"]):
                with gr.Column(elem_classes=["clean-card"]):
                    gr.HTML(
                        """
                        <h2>논문 PDF 업로드</h2>
                        <p class="card-desc">분석할 논문 PDF 파일을 업로드한 뒤 분석 버튼을 눌러 주세요.</p>
                        """
                    )

                    pdf_input = gr.File(
                        label="PDF 파일 선택",
                        file_types=[".pdf"],
                        elem_id="paper-upload",
                    )

                    file_info = gr.HTML(value=get_file_info(None))

                    submit_btn = gr.Button(
                        "논문 분석하기",
                        variant="primary",
                    )

                with gr.Column(elem_classes=["clean-card", "result-card"]):
                    with gr.Row(elem_classes=["result-header"]):
                        gr.HTML(
                            """
                            <div>
                                <h2>AI 분석 결과</h2>
                                <p class="card-desc" style="margin-bottom:0;">
                                    분석이 완료되면 발표 준비용 결과가 아래에 표시됩니다.
                                </p>
                            </div>
                            """
                        )
                        status = gr.HTML(value="<span class='status-ready'>Ready</span>")

                    output = gr.Markdown(value=EMPTY_RESULT_HTML, elem_id="analysis-output")

            pdf_input.change(
                fn=get_file_info,
                inputs=pdf_input,
                outputs=file_info,
            )

            submit_btn.click(
                fn=lambda: ("<span class='status-loading'>분석 중</span>", LOADING_HTML),
                inputs=None,
                outputs=[status, output],
                queue=False,
            ).then(
                fn=analyze_paper,
                inputs=pdf_input,
                outputs=[status, output],
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