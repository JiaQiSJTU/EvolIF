# encoding = "utf-8"

"""
Synthesize query_x.jsonl from snapshots_x.jsonl under the state directory.

Rules:
- Each query consists of the topic_query (only when the topic changes) plus a description of instruction changes.
- Only describe changes to instructions (add/remove/modify); do not repeat the full set.
- Generate deterministic natural-language text suitable for direct use as multi-turn user inputs.
"""

import argparse
import json
import os
from typing import Dict, Any, Optional
# from openai import OpenAI
from tqdm import tqdm
from instruction import *
import random
from data_utils.utils import LLM_backend, load_jsonl, write_jsonl

from data_utils.query_synthesis_prompts import (
    INIT_QUERY_SYNTHESIS_WO_STYLE,
    TOPIC_CHANGE_QUERY_SYNTHESIS_WO_STYLE,
    TOPIC_CONTINUE_QUERY_SYNTHESIS_WO_STYLE,
    INIT_QUERY_SYNTHESIS_W_STYLE,
    TOPIC_CHANGE_QUERY_SYNTHESIS_W_STYLE,
    TOPIC_CONTINUE_QUERY_SYNTHESIS_W_STYLE,
)


def query_synthesis(topic_turn_idx, topic_query, cur_operation, topic_name, prev_topic_name, api_key, base_url, style=None, use_json_mode=True, model_name="gpt-4.1"):

    if topic_turn_idx == 0:
        if style == None:
            prompt = INIT_QUERY_SYNTHESIS_WO_STYLE.format(
                topic_query=topic_query, instructions=cur_operation["new_description"])
        else:
            prompt = INIT_QUERY_SYNTHESIS_W_STYLE.format(
                topic_query=topic_query, instructions=cur_operation["new_description"], style=style)

    else:
        instruction = ""
        if cur_operation["operation_type"] == "add":
            instruction = "Add: " + cur_operation["new_description"]
        elif cur_operation["operation_type"] == "modify":
            instruction = f"Modify: from \" {cur_operation['original_description']} \" to \" {cur_operation['new_description']} \"."
        elif cur_operation["operation_type"] == "remove":
            instruction = "Remove: " + cur_operation["new_description"]

        if topic_name != prev_topic_name:
            if style == None:
                prompt = TOPIC_CHANGE_QUERY_SYNTHESIS_WO_STYLE.format(
                    topic_query=topic_query, instructions=instruction)
            else:
                prompt = TOPIC_CHANGE_QUERY_SYNTHESIS_W_STYLE.format(
                    topic_query=topic_query, instructions=instruction, style=", ".join(style["styles"][:3]))

            # print(prompt)
            # exit(0)
        else:
            if style == None:
                prompt = TOPIC_CONTINUE_QUERY_SYNTHESIS_WO_STYLE.format(
                    instructions=instruction, turn_idx=topic_turn_idx)
            else:
                prompt = TOPIC_CONTINUE_QUERY_SYNTHESIS_W_STYLE.format(
                    instructions=instruction, style=", ".join(style["styles"][:3]))

    # print(prompt)
    # exit(0)

    messages = [{"role": "user", "content": prompt}]
    if not use_json_mode:
        response, prompt_tokens, completion_tokens = LLM_backend(
            api_key, messages, model_name, base_url, temperature=1.2, use_json_mode=False)
    else:
        response, prompt_tokens, completion_tokens = LLM_backend(
            api_key, messages, model_name, base_url, temperature=1.2)

    try:
        response = json.loads(response)

        if response["user_query"].startswith("Actually"):
            response["user_query"] = response["user_query"].replace(
                "Actually, ", "", 1).strip().capitalize()
        if response["user_query"].startswith("Add:"):
            response["user_query"] = response["user_query"].replace(
                "Add:", "", 1).strip()

        return response["user_query"], prompt
    except Exception as e:
        # print(f"Error in query_synthesis: {e}")
        # print(response)
        response = response.split(
            "user_query\": \"")[-1].split("\"\n}\n```")[0]
        # print(response)
        # exit(0)
        return response, prompt


def load_style_candidates():

    styles = []
    with open("./data/persona_language_styles_500.jsonl") as f:
        for line in f:
            styles.append(json.loads(line.strip()))
    return styles


def query_checker(final_query: str, cur_op: Dict[str, Any]) -> bool:
    """
    Check whether final_query satisfies the instruction-specific
    """
    if not isinstance(cur_op, dict):
        return True

    op_type = cur_op.get("operation_type")
    cls_name = cur_op.get("instruction_class")
    prev_args = cur_op.get("args_before")
    cur_args = cur_op.get("args_after")
    # Removal operations don't impose new completeness constraints
    if op_type == "remove":
        return True
    # Resolve class by name directly from current namespace
    cls = globals().get(cls_name)
    if cls is None:
        return True

    try:
        return bool(cls.check_query_completeness(final_query, prev_args, cur_args))
    except Exception:
        return False


def topic_checker(final_query, topic_query, use_json_mode, api_key, base_url, model_name="gpt-4.1"):
    """
    Use LLM to judge whether final_query is semantically related to topic_query.
    Rules:
    - Purely generic references like "Regarding the earlier topic" are NOT related.
    - If final_query mentions content clearly tied to topic_query (e.g., "about a joke" for topic "tell me a joke about animals"),
      it IS related even if phrased differently.
    Return True if related, else False.
    """
    try:
        if not isinstance(final_query, str) or not isinstance(topic_query, str):
            return False
        if topic_query.strip() == "":
            return True
        instruction = (
            "You will be given two sentences. Decide if Sentence-1 is semantically related to Sentence-2.\n\n"
            "Rules:\n"
            "- Generic references without concrete content in Sentence-1 (e.g., 'the earlier topic') -> not related.\n"
            "- If Sentence-1 contains a concrete clue pointing to the Sentence-2 (paraphrase/keywords) -> related.\n\n"
            "Sentence-1: {final_query}\n"
            "Sentence-2: {topic_query}\n\n"
            "Answer JSON only: {{\"related\": true|false}}\n"
        )
        
        prompt = instruction.format(
            final_query=final_query, topic_query=topic_query)
        messages = [{"role": "user", "content": prompt}]
        if not use_json_mode:
            resp, _, _ = LLM_backend(
                api_key, messages, model_name, base_url, use_json_mode=False)
        else:
            resp, _, _ = LLM_backend(api_key, messages, model_name, base_url)
        try:
            data = json.loads(resp)
            val = data.get("related")
            return bool(val) is True
        except Exception:
            # Best-effort fallback: look for a bare true/false in response
            text = str(resp).strip().lower()
            if "true" in text:
                return True
            if "false" in text:
                return False
            # Last resort: conservative -> not related
            return False
    except Exception:
        return False


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_style:
        style_candidates = load_style_candidates()

    for file_id in tqdm(range(args.start_id, args.end_id + 1)):

        snapshots_path = os.path.join(
            args.input_dir, f"snapshots_{file_id}.jsonl")
        if not os.path.exists(snapshots_path):
            continue

        out_path = os.path.join(args.output_dir, f"dialog_{file_id}.jsonl")

        turn_idx = 0
        topic_turn_idx = {}
        prev_topic_name: Optional[str] = None
        cur_style = None

        # If an out_path file already exists, read it and obtain the last turn_idx and prev_topic_name
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    turn_idx = item.get("turn")
                    prev_topic_name = item.get("active_topic")
                    topic_turn_idx[prev_topic_name] = topic_turn_idx.get(
                        prev_topic_name, 0) + 1
                    cur_style = item.get("style", None)

        output_file = open(out_path, "a+", encoding="utf-8")

        if args.use_style and cur_style == None:
            cur_style = random.choice(style_candidates)

        raw_turns = load_jsonl(snapshots_path)
        for item in tqdm(raw_turns[turn_idx:turn_idx+10]):

            cur_turn_idx = item.get("turn")
            topic_name = item.get("active_topic")
            topic_query = item.get("topic_query")
            cur_op = item.get("cur_operation")
            instructions = item.get("instructions")

            cur_topic_turn_idx = topic_turn_idx.get(topic_name, 0)

            # Try up to 3 times to satisfy both checks
            attempts = 0
            while True:
                final_query, prompt = query_synthesis(cur_topic_turn_idx, topic_query, cur_op, topic_name,
                                                      prev_topic_name, args.api_key, args.base_url, cur_style, args.use_json_mode, args.model_name)
                # print("final_query", final_query)
                instruction_success = query_checker(final_query, cur_op)
                if cur_turn_idx != 0 and topic_name != prev_topic_name:
                    topic_success = topic_checker(
                        final_query, topic_query, args.use_json_mode, args.api_key, args.base_url, args.model_name)
                else:
                    topic_success = True

                if instruction_success and topic_success:
                    break
                attempts += 1
                if attempts >= 3:
                    print(
                        f"Failed to synthesize query for turn {cur_turn_idx} of topic {topic_name}")
                    break

            # exit(0)
            topic_turn_idx[topic_name] = cur_topic_turn_idx + 1
            prev_topic_name = topic_name

            if instruction_success and topic_success:
                user_query_verified = final_query
            else:
                user_query_verified = ""

            output_file.write(json.dumps({
                "turn": cur_turn_idx,
                "active_topic": topic_name,
                "user_query_verified": user_query_verified,
                "user_query": final_query,
                "instructions": instructions,
                "style": cur_style,
                "instruction_success": instruction_success,
                "topic_success": topic_success,
                # "prompt": prompt
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Synthesize user queries from snapshots")
    parser.add_argument("--input_dir", type=str,
                        default="./state", help="Directory containing snapshots_*.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="./dialog", help="Output directory for query_*.jsonl")
    parser.add_argument("--start_id", type=int, default=69, help="Starting ID")
    parser.add_argument("--end_id", type=int, default=132, help="Ending ID (inclusive)")
    parser.add_argument("--api_key", type=str, default="", help="API key")
    parser.add_argument("--base_url", type=str, default="", help="Base URL")
    parser.add_argument("--use_json_mode",
                        action="store_true", help="Enable JSON mode")
    parser.add_argument("--use_style", action="store_true",
                        help="incorporate random style for each dialogue")
    parser.add_argument("--model_name", type=str,
                        default="gpt-4.1", help="Model name")
    args = parser.parse_args()

    # python3 src/query_synthesis.py --use_json_mode --use_style --api_key $API_KEY --base_url $BASE_URL --start_id 1 --end_id 68
    # python3 src/query_synthesis.py --use_style --api_key $API_KEY --base_url $BASE_URL --model_name gemini-2.5-flash --start_id 69 --end_id 132
    # python3 src/query_synthesis.py --use_style --api_key $API_KEY --base_url $BASE_URL --model_name deepseek-ai/DeepSeek-V3.1 --start_id 133 --end_id 205

    main(args)
