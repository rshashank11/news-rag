import argparse
import logging
import re
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.agents.workflow import chat_graph
from main import build_initial_state
from schemas import ChatMessage, ChatResponse


DEFAULT_PDF_PATH = Path("/Users/shashank/Downloads/bnb_newsbot_poc_v2.pptx.pdf")
DEFAULT_REPORT_PATH = Path("eval_runs/bnb_newsbot_poc_v2_results.md")
LOGGER = logging.getLogger(__name__)


@dataclass
class PdfQuestion:
    persona: str
    scenario: str
    question_id: str
    question_type: str
    question: str


def stream_body(object_body: bytes) -> bytes | None:
    if b"stream" not in object_body:
        return None

    start = object_body.find(b"stream") + len(b"stream")

    if object_body[start:start + 2] == b"\r\n":
        start += 2
    elif object_body[start:start + 1] == b"\n":
        start += 1

    end = object_body.find(b"endstream", start)
    raw_stream = object_body[start:end].strip(b"\r\n")

    if b"FlateDecode" not in object_body:
        return raw_stream

    return zlib.decompress(raw_stream)


def parse_pdf_objects(pdf_bytes: bytes) -> dict[int, bytes]:
    return {
        int(match.group(1)): match.group(2)
        for match in re.finditer(rb"(\d+)\s+0\s+obj(.*?)endobj", pdf_bytes, re.S)
    }


def parse_cmap(cmap_stream: bytes) -> dict[int, str]:
    cmap_text = cmap_stream.decode("latin1", errors="ignore")
    mapping: dict[int, str] = {}

    for block in re.finditer(r"beginbfchar(.*?)endbfchar", cmap_text, re.S):
        for source, target in re.findall(
            r"<([0-9A-Fa-f]+)>\s+<([0-9A-Fa-f]+)>",
            block.group(1),
        ):
            mapping[int(source, 16)] = chr(int(target, 16))

    for block in re.finditer(r"beginbfrange(.*?)endbfrange", cmap_text, re.S):
        for source_start, source_end, target_start in re.findall(
            r"<([0-9A-Fa-f]+)>\s+<([0-9A-Fa-f]+)>\s+<([0-9A-Fa-f]+)>",
            block.group(1),
        ):
            start_code = int(source_start, 16)
            end_code = int(source_end, 16)
            target_code = int(target_start, 16)

            for code in range(start_code, end_code + 1):
                mapping[code] = chr(target_code + code - start_code)

    return mapping


def decode_pdf_literal(literal: bytes, cmap: dict[int, str]) -> str:
    value = literal[1:-1]
    value = (
        value
        .replace(b"\\(", b"(")
        .replace(b"\\)", b")")
        .replace(b"\\\\", b"\\")
    )
    decoded = []

    for index in range(0, len(value) - 1, 2):
        code = (value[index] << 8) | value[index + 1]
        decoded.append(cmap.get(code, chr(code) if 32 <= code < 0x110000 else ""))

    return "".join(decoded)


def page_texts_from_pdf(pdf_path: Path) -> list[str]:
    objects = parse_pdf_objects(pdf_path.read_bytes())
    cmap_by_object = {
        object_id: parse_cmap(stream)
        for object_id, object_body in objects.items()
        if (stream := stream_body(object_body)) and b"begincmap" in stream
    }
    font_to_cmap = {}

    for object_id, object_body in objects.items():
        match = re.search(rb"/ToUnicode\s+(\d+)\s+0\s+R", object_body)
        if match:
            font_to_cmap[object_id] = cmap_by_object.get(int(match.group(1)), {})

    font_reference_pattern = re.compile(rb"/(Font\d+)\s+(\d+)\s+0\s+R")
    text_token_pattern = re.compile(
        rb"/(Font\d+)\s+[0-9.]+\s+Tf|"
        rb"(\((?:\\.|[^\\)])*\))\s*Tj|"
        rb"\[(.*?)\]\s*TJ",
        re.S,
    )
    page_object_ids = [
        object_id
        for object_id, object_body in sorted(objects.items())
        if re.search(rb"/Type\s*\n?/Page\b", object_body)
    ]
    page_texts = []

    for page_object_id in page_object_ids:
        page_body = objects[page_object_id]
        content_match = re.search(rb"/Contents\s+(\d+)\s+0\s+R", page_body)
        resources_match = re.search(rb"/Resources\s+(\d+)\s+0\s+R", page_body)

        if not content_match or not resources_match:
            page_texts.append("")
            continue

        content_object_id = int(content_match.group(1))
        resources_object_id = int(resources_match.group(1))
        font_map = {
            font_name.decode("latin1"): font_to_cmap.get(int(font_object_id), {})
            for font_name, font_object_id in font_reference_pattern.findall(
                objects[resources_object_id]
            )
        }
        content_stream = stream_body(objects[content_object_id]) or b""
        current_cmap: dict[int, str] = {}
        parts = []

        for token in text_token_pattern.finditer(content_stream):
            if token.group(1):
                current_cmap = font_map.get(token.group(1).decode("latin1"), {})
            elif token.group(2):
                parts.append(decode_pdf_literal(token.group(2), current_cmap))
            elif token.group(3):
                for literal in re.findall(rb"\((?:\\.|[^\\)])*\)", token.group(3)):
                    parts.append(decode_pdf_literal(literal, current_cmap))

        page_text = re.sub(r"\s+", " ", " ".join(part for part in parts if part.strip()))
        page_texts.append(page_text)

    return page_texts


def extract_quoted_after(label: str, text: str) -> str | None:
    pattern = rf"{re.escape(label)}\s+\"(.*?)\""
    match = re.search(pattern, text)
    if not match:
        return None

    return clean_extracted_question(match.group(1))


def clean_extracted_question(question: str) -> str:
    return (
        question
        .replace("NCL T", "NCLT")
        .replace("NCLA T", "NCLAT")
        .replace("UAP A", "UAPA")
        .replace("PML A", "PMLA")
        .replace("T est", "Test")
        .strip()
    )


def extract_questions(page_texts: list[str]) -> list[PdfQuestion]:
    questions: list[PdfQuestion] = []

    for page_text in page_texts:
        if "Parent question" not in page_text or "Follow-up" not in page_text:
            continue

        persona_match = re.search(r"^\d+\s+(.+?)\s+Scenario", page_text)
        scenario_match = re.search(r"Scenario\s+(\d+\s+of\s+\d+)", page_text)
        question_id_match = re.search(r"\b(Q\d+)\s+Parent question", page_text)

        parent_question = extract_quoted_after("Parent question", page_text)
        follow_up_1 = extract_quoted_after("Follow-up 1 · Memory test", page_text)
        follow_up_2 = extract_quoted_after("Follow-up 2 · Memory test", page_text)

        if not parent_question or not question_id_match or not scenario_match:
            continue

        persona = persona_match.group(1).strip() if persona_match else "Unknown persona"
        scenario = scenario_match.group(1).strip()
        question_id = question_id_match.group(1)

        questions.append(
            PdfQuestion(
                persona=persona,
                scenario=scenario,
                question_id=question_id,
                question_type="parent",
                question=parent_question,
            )
        )

        if follow_up_1:
            questions.append(
                PdfQuestion(
                    persona=persona,
                    scenario=scenario,
                    question_id=question_id,
                    question_type="follow_up_1",
                    question=follow_up_1,
                )
            )

        if follow_up_2:
            questions.append(
                PdfQuestion(
                    persona=persona,
                    scenario=scenario,
                    question_id=question_id,
                    question_type="follow_up_2",
                    question=follow_up_2,
                )
            )

    return questions


def source_summary(response: ChatResponse) -> str:
    if not response.sources:
        return "None"

    return "; ".join(
        f"[{source.source_number}] {source.published_at or 'Unknown'} | {source.headline}"
        for source in response.sources
    )


def step_summary(result: dict[str, Any]) -> str:
    steps = result.get("steps") or []
    if not steps:
        return "No trace steps returned."

    return " / ".join(
        f"{step.name}: {step.detail}"
        for step in steps
    )


def response_excerpt(response: ChatResponse, limit: int = 900) -> str:
    text = response.message.replace("\n", " ").strip()
    if len(text) <= limit:
        return text

    return f"{text[:limit].rstrip()}..."


def run_question(question: PdfQuestion, history: list[ChatMessage]) -> tuple[dict[str, Any], ChatResponse, float]:
    started_at = time.perf_counter()
    result = chat_graph.invoke(build_initial_state(question.question, history))
    elapsed_seconds = round(time.perf_counter() - started_at, 2)
    response = result.get("response")

    if response is None:
        raise RuntimeError("Workflow finished without a response.")

    return result, response, elapsed_seconds


def append_history(history: list[ChatMessage], question: str, response: ChatResponse) -> None:
    history.append(ChatMessage(role="user", content=question))
    history.append(ChatMessage(role="assistant", content=response.message))


def build_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# BNB Newsbot PDF Question Evaluation",
        "",
        f"Total questions tested: {len(results)}",
        "",
    ]
    counts: dict[str, int] = {}

    for item in results:
        counts[item["response_type"]] = counts.get(item["response_type"], 0) + 1

    lines.append("## Response Type Summary")
    for response_type, count in sorted(counts.items()):
        lines.append(f"- `{response_type}`: {count}")

    lines.append("")
    lines.append("## Detailed Results")

    for index, item in enumerate(results, start=1):
        lines.extend(
            [
                "",
                f"### {index}. {item['persona']} / {item['scenario']} / {item['question_id']} / {item['question_type']}",
                "",
                f"**Question:** {item['question']}",
                "",
                f"**Response type:** `{item['response_type']}`",
                f"**Elapsed:** {item['elapsed_seconds']}s",
                f"**Planner query:** {item['planner_query']}",
                f"**Current query:** {item['current_query']}",
                f"**Sources:** {item['sources']}",
                "",
                f"**Trace:** {item['trace']}",
                "",
                f"**Answer excerpt:** {item['answer_excerpt']}",
            ]
        )

    return "\n".join(lines) + "\n"


def evaluate_questions(pdf_path: Path, output_path: Path, limit: int | None) -> list[dict[str, Any]]:
    page_texts = page_texts_from_pdf(pdf_path)
    questions = extract_questions(page_texts)

    if limit is not None:
        questions = questions[:limit]

    results: list[dict[str, Any]] = []
    history_by_group: dict[tuple[str, str, str], list[ChatMessage]] = {}

    for index, question in enumerate(questions, start=1):
        group_key = (question.persona, question.scenario, question.question_id)
        history = history_by_group.setdefault(group_key, [])
        LOGGER.info(
            "running benchmark question",
            extra={
                "index": index,
                "total": len(questions),
                "question_id": question.question_id,
                "question_type": question.question_type,
                "question": question.question,
            },
        )

        try:
            result, response, elapsed_seconds = run_question(question, history)
            append_history(history, question.question, response)
            analysis = result.get("analysis")

            results.append(
                {
                    "persona": question.persona,
                    "scenario": question.scenario,
                    "question_id": question.question_id,
                    "question_type": question.question_type,
                    "question": question.question,
                    "response_type": response.type,
                    "elapsed_seconds": elapsed_seconds,
                    "planner_query": analysis.search_query if analysis else "None",
                    "current_query": result.get("current_query") or "None",
                    "sources": source_summary(response),
                    "trace": step_summary(result),
                    "answer_excerpt": response_excerpt(response),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "persona": question.persona,
                    "scenario": question.scenario,
                    "question_id": question.question_id,
                    "question_type": question.question_type,
                    "question": question.question,
                    "response_type": "error",
                    "elapsed_seconds": 0,
                    "planner_query": "Error before planner result",
                    "current_query": "Error before current query",
                    "sources": "None",
                    "trace": f"Exception: {exc}",
                    "answer_excerpt": "No answer generated.",
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report(results), encoding="utf-8")
    LOGGER.info("wrote benchmark report", extra={"output_path": str(output_path)})

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PDF benchmark questions against the chat graph.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    evaluate_questions(
        pdf_path=args.pdf,
        output_path=args.output,
        limit=args.limit,
    )
