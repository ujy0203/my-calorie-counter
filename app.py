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
      <div class="file-icon-cust석 도우미</h1>
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
