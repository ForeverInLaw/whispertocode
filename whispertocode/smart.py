import sys
from typing import Any, Callable, List, Optional, Tuple

from .utils import _coerce_stream_text

def build_smart_messages(raw_text: str) -> List[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert editor for speech-to-text transcripts. "
                "Your task is to transform the raw dictated text enclosed in <transcript> tags into clean, naturally typed text.\n\n"
                "CORE RULES:\n"
                "- Language & Tone: Keep the exact original language (do not translate). Preserve the speaker's original voice and tone. Do not over-formalize casual speech.\n"
                "- Accuracy: Preserve all original meaning, facts, names, numbers, links, and technical terms. Do not add any new information.\n"
                "- Cleanup: Remove filler words (um, uh, you know), stutters, redundancies, and false starts.\n"
                "- Grammar & Correction: Fix punctuation, capitalization, sentence boundaries, and paragraph breaks. Correct obvious speech-to-text phonetic mishearings (homophones) based on context.\n\n"
                "OUTPUT CONSTRAINTS:\n"
                "- Return ONLY the final corrected text.\n"
                "- Do NOT include greetings, explanations, or confirmation messages (e.g., 'Here is the text').\n"
                "- Do NOT wrap the output in markdown code blocks (```) or quotes.\n"
                "- If the input contains no meaningful words (only noise/fillers), return an empty string."
            ),
        },
        {
            "role": "user", 
            "content": f"<transcript>\n{raw_text}\n</transcript>"
        },
    ]



def ensure_nemotron_client(current_client: Any, base_url: str, api_key: str) -> Any:
    if current_client is not None:
        return current_client
    from openai import OpenAI

    return OpenAI(
        base_url=base_url,
        api_key=api_key,
    )


def rewrite_text_streaming(
    *,
    raw_text: str,
    get_client: Callable[[], Any],
    model: str,
    messages: List[dict],
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning_budget: int,
    enable_thinking: bool,
    reasoning_print_limit: int,
    type_char: Callable[[str], None],
) -> Tuple[bool, Optional[Exception]]:
    typed_any = False
    reasoning_printed = False
    try:
        client = get_client()
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body={
                "reasoning_budget": reasoning_budget,
                "chat_template_kwargs": {
                    "enable_thinking": enable_thinking,
                },
            },
            stream=True,
        )

        for chunk in completion:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue

            reasoning_text = _coerce_stream_text(
                getattr(delta, "reasoning_content", None)
            )
            if reasoning_text:
                print(reasoning_text, end="", flush=True)
                reasoning_printed = True

            content_text = _coerce_stream_text(getattr(delta, "content", None))
            if content_text:
                for char in content_text:
                    type_char(char)
                typed_any = True

        return typed_any, None
    except Exception as exc:
        return typed_any, exc
    finally:
        if reasoning_printed:
            print()
