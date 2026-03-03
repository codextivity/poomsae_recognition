"""Prompt templates (Korean-only)."""

from __future__ import annotations

from typing import Optional


def _amount_to_qual_ko(amount: int | None) -> str:
    if amount is None:
        return "약간"
    if amount <= 5:
        return "아주 조금"
    if amount <= 10:
        return "조금"
    if amount <= 20:
        return "다소"
    if amount <= 30:
        return "꽤"
    return "많이"


def _metric_label(metric: str) -> str:
    mapping = {
        "stance_width_norm": "보폭",
        "wrist_height_left": "왼손 손목",
        "wrist_height_right": "오른손 손목",
        "knee_angle_left": "왼쪽 무릎",
        "knee_angle_right": "오른쪽 무릎",
        "tempo": "속도",
        "power": "힘",
    }
    return mapping.get(metric, metric)


def _direction_label(metric: str, direction: str) -> str:
    if metric == "stance_width_norm":
        return "넓혀" if direction == "widen" else "좁혀"
    if metric.startswith("wrist_height"):
        return "올려" if direction == "raise" else "내려"
    if metric.startswith("knee_angle"):
        return "더 굽혀" if direction == "bend_more" else "펴"
    if metric == "tempo":
        return "빠르게" if direction == "faster" else "느리게"
    if metric == "power":
        return "강하게" if direction == "stronger" else "가볍게"
    return direction


def _severity_label(severity: str) -> str:
    mapping = {"minor": "경미", "medium": "중간", "high": "높음"}
    return mapping.get(severity, severity)


def _instruction_block(instructions: list) -> str:
    if not instructions:
        return "없음"
    lines = []
    for inst in instructions:
        metric = inst.get("metric", "")
        direction = inst.get("direction", "")
        amount = (
            inst.get("amount_deg")
            if "amount_deg" in inst
            else inst.get("amount_percent_torso")
            if "amount_percent_torso" in inst
            else inst.get("amount_percent")
        )
        severity = inst.get("severity", "")
        parts = [
            f"항목={_metric_label(metric)}",
            f"방향={_direction_label(metric, direction)}",
            f"정도={_amount_to_qual_ko(amount)}",
        ]
        if severity:
            parts.append(f"중요도={_severity_label(severity)}")
        lines.append("- " + ", ".join(parts))
    return "\n".join(lines)


def build_prompt():
    PROMPT_TEMPLATE = """[역할]
    당신은 태권도 품새 코치입니다.

    [출력 형식 고정]
    반드시 한 문단으로만 작성하십시오.
    반드시 두 문장으로만 작성하십시오.
    줄바꿈을 절대 사용하지 마십시오.
    피드백 문장 외의 어떠한 설명도 추가하지 마십시오.

    [언어 고정 규칙]
    출력은 완전한 한글 문장만 허용됩니다.
    한자, 영어, 로마자, 숫자, 특수문자가 하나라도 포함되면 실패입니다.

    [출력 제한 규칙]
    숫자, 기호, 백분율, 각도 표현을 사용하지 마십시오.
    인사, 칭찬, 점수, 평가성 표현을 쓰지 마십시오.
    기존에 없는 기술명이나 동작명을 임의로 추가하지 마십시오.
    선택지를 제시하는 표현을 쓰지 마십시오.
    입력에 없는 내용은 절대 추가하지 마십시오.
    입력된 방향 값은 반대로 바꾸지 마십시오.
    입력 정보에 속도나 힘 조절 지시가 없으면 관련 내용을 언급하지 마십시오.

    [의미 재구성 규칙]
    입력된 용어를 그대로 반복하지 말고 의미를 바꾸어 표현하십시오.
    예시: 보폭은 디딤 간격처럼 바꾸어 표현할 수 있습니다.
    단, 새로운 기술명이나 동작명은 만들지 마십시오.

    [작성 원칙]
    중요도가 높은 요소부터 우선 반영하십시오.
    한 번에 최대 두 가지 요소만 다루십시오.
    정도 표현은 약간, 조금, 다소, 충분히 등 자연스러운 부사로 완화하십시오.
    간결하고 구체적인 코칭 어조로 서술하십시오.

    [입력]
    - 항목=보폭, 방향=좁혀, 정도=다소, 중요도=중간
    - 항목=속도, 방향=빠르게, 정도=다소, 중요도=경미

    [최종 지시]
    출력 전에 모든 규칙 충족 여부를 스스로 점검하십시오.
    규칙을 모두 충족하는 경우에만 최종 피드백 두 문장을 한 문단으로 출력하십시오.
    """

    # 사용 예시:
        # instruction_tokens = \"\"\"Instruction tokens:
        # - 항목=보폭, 방향=좁혀, 정도=다소, 중요도=중간
        # - 항목=속도, 방향=빠르게, 정도=다소, 중요도=경미\"\"\"
        # prompt = PROMPT_TEMPLATE.format(instruction_tokens=instruction_tokens)


    return PROMPT_TEMPLATE

