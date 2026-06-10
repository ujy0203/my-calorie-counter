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

    response = model.generate_content(prompt)
    return response.text


def build_ui():
    return gr.Interface(
        fn=analyze_paper,
        inputs=gr.File(
            label="논문 PDF 업로드",
            file_types=[".pdf"]
        ),
        outputs=gr.Markdown(label="AI 분석 결과"),
        title="📄 논문 발표 도우미",
        description=(
            "논문 PDF를 업로드하면 AI가 연구 목적, 연구 방법, 주요 결과, "
            "발표 대본, 예상 질문을 자동으로 생성합니다."
        ),
        flagging_mode="never",
    )


demo = build_ui()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        show_api=False,
        ssr_mode=False,
    )