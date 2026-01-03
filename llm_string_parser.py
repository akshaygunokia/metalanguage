import re
import json
from typing import Any, Dict, List, Optional


qwen3_schema = {
    "x-regex": r"^(?:<think>\n?(?P<reasoning_content>.+?)\n?</think>\s*)?(?P<content>.*?)(?=(?:<tool_call>|<\|im_end\|>|$))(?P<tool_calls>(?:<tool_call>.+?</tool_call>\s*)+)?(?P<trailing>.*)$",
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string"},
        "reasoning_content": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "x-regex-iterator": r"<tool_call>\s*(.+?)\s*</tool_call>",
            "items": {
                "x-parser": "json",
                "x-parser-args": {"transform": "{type: 'function', function: @}"},
                "type": "object",
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {
                                "type": "object",
                                "additionalProperties": {},
                            },
                        },
                    },
                },
            },
        },
    },
}

def parse_with_schema(text: str, schema: Dict[str, Any] = qwen3_schema) -> Dict[str, Any]:
    tc_schema = schema["properties"]["tool_calls"]
    tool_call_iter = re.compile(tc_schema["x-regex-iterator"], flags=re.DOTALL)

    tool_resp_re = re.compile(r"<tool_response>\s*(.+?)\s*</tool_response>", flags=re.DOTALL)

    tool_calls: List[Dict[str, Any]] = []
    matches = list(tool_call_iter.finditer(text))

    for i, tm in enumerate(matches):
        raw = tm.group(1).strip()

        # parse tool call json
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            tool_calls.append({
                "type": "function",
                "function": {"name": None, "arguments": {}},
                "raw": raw,
                "parse_error": f"json_decode_error: {e}",
                "tool_response": None,
            })
            continue

        transformed = {"type": "function", "function": parsed}

        # enforce minimal constraints
        fn = transformed["function"]
        if transformed["type"] != "function":
            raise ValueError(f"Invalid tool_call.type: {transformed.get('type')}")

        name = fn.get("name")
        if not isinstance(name, str):
            raise ValueError(f"Invalid tool_call.function.name: {name!r}")

        args = fn.get("arguments", {})
        if not isinstance(args, dict):
            args = {} if args is None else {"_raw": args}
            fn["arguments"] = args

        # ---- extract tool_response ONLY if it follows this tool_call ----
        after_call = tm.end()
        before_next_call = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        window = text[after_call:before_next_call]

        # allows junk like "user\n" or "<|im_start|>user" between them
        m_resp = tool_resp_re.search(window)
        tool_response = None
        tool_response_json = None

        if m_resp:
            raw_resp = m_resp.group(1).strip()
            tool_response = raw_resp
            try:
                tool_response_json = json.loads(raw_resp)
            except Exception:
                tool_response_json = None

        transformed["tool_response"] = tool_response
        transformed["tool_response_json"] = tool_response_json
        tool_calls.append(transformed)

    return {
        "tool_calls": tool_calls,
    }
