# encoding = "utf-8"
from dataclasses import dataclass, field
from instruction import *
import random
from typing import List, Optional
from instruction.instruction_utils import get_topic_list, TOPIC_QUERY_DICT
import json
import copy


def _rng_state_to_jsonable(state):
    """Convert random.getstate() tuple into JSON-serializable nested lists."""
    def convert(obj):
        if isinstance(obj, tuple):
            return [convert(x) for x in obj]
        return obj
    return convert(state)


def _jsonable_to_rng_state(obj):
    """Convert nested lists back into tuples acceptable by random.setstate."""
    def convert(o):
        if isinstance(o, list):
            return tuple(convert(x) for x in o)
        return o
    return convert(obj)


INSTRUCTION_DICT = {
    "startwith": StartWithInstruction,
    "endwith": EndWithInstruction,
    # "language": LanguageInstruction,
    "format": FormatInstruction,
    "countableItems": CountableItemsInstruction,
    "length": LengthInstruction,
    "existence": ExistenceInstruction,
    "forbidden": ForbiddenInstruction,
    "case": ChangeCaseInstruction,
    "punctuation": PunctuationInstruction,
    "emotion": EmotionInstruction,
    "reader_age": ReaderAgeInstruction,
    "style": StyleInstruction,
}

INSTRUCTION_WEIGHT_DICT = {
    "startwith": 1,
    "endwith": 1,
    "format": 1,
    "countableItems": 1,
    "length": 1,
    "existence": 1,
    "forbidden": 1,
    "case": 1,
    "punctuation": 1,
    "emotion": 1,
    "reader_age": 1,
    "style": 1,
}

INSTRUCTION_REMOVE_DICT = {
    "startwith": "Regardless of the instructions regarding how to start your response.",
    "endwith": "Regardless of the instructions regarding how to end your response.",
    # "language": "Regardless of the instructions regarding language requirements.",
    "format": "Regardless of the instructions regarding response format, such as CSV, JSON, XML, HTML, Markdown, etc.",
    "countableItems": "Regardless of the instructions regarding included bullet points.",
    "length": "Regardless of the instructions regarding length requirements of the response, such as word count, paragraph count, character count, sentence count, etc.",
    "existence": "Regardless of the instructions regarding required keyword.",
    "forbidden": "Regardless of the instructions regarding forbidden words.",
    "case": "Regardless of the instructions regarding letter case.",
    "punctuation": "Regardless of the instructions regarding punctuation requirements.",
    "emotion": "Regardless of the instructions regarding emotional requirements.",
    "reader_age": "Regardless of the instructions regarding reader age requirements.",
    "style": "Regardless of the instructions regarding style requirements.",
}


@dataclass
class TopicManager:
    topic_name: str
    topic_query: str
    # Key: instruction_id; Value: instruction arguments
    instructions: dict = field(default_factory=dict)
    # cur_topic_turn: total number of turns executed under this topic
    cur_topic_turn: int = 0  # turns already executed for the current topic

    # # --------------- Basic helper methods ---------------
    def _collect_forbidden_keywords(self) -> List[str]:
        """Collect the forbidden keyword list from ForbiddenInstruction under the current topic."""
        if "forbidden" in self.instructions:
            args = self.instructions["forbidden"]
            if isinstance(args, list):
                return args
        return []

    def _collect_required_keywords(self) -> List[str]:
        """Collect required keywords from StartWith/EndWith (keyword mode only) and Existence under the current topic."""
        result: List[str] = []

        if "startwith" in self.instructions:
            sw_args = self.instructions["startwith"]
            if sw_args["mode"] == "keyword" and isinstance(sw_args, dict):
                result.append(sw_args.get("value", ""))
        if "endwith" in self.instructions:
            ew_args = self.instructions["endwith"]
            if ew_args["mode"] == "keyword" and isinstance(ew_args, dict):
                result.append(ew_args.get("value", ""))
        if "existence" in self.instructions:
            ex_args = self.instructions["existence"]
            # args is directly a dict: {keyword: count}
            if isinstance(ex_args, dict):
                result.extend(ex_args.keys())
        return result

    # --------------- Random operations ---------------
    def add_random_instruction(self) -> bool:
        """Randomly add one instruction. 
        """
        all_keys = [k for k in INSTRUCTION_DICT if k not in self.instructions]
        weights = [INSTRUCTION_WEIGHT_DICT[k] for k in all_keys]
        for _ in range(10):
            key = random.choices(all_keys, weights=weights, k=1)[0]

            cls = INSTRUCTION_DICT[key]
            try:
                # Pass topic_name / forbidden_keywords depending on instruction type
                if key in {"startwith", "endwith", "existence"}:
                    inst = cls().initialization(topic_name=self.topic_name,
                                                forbidden_keywords=self._collect_forbidden_keywords())
                elif key == "forbidden":
                    inst = cls().initialization(
                        topic_name=self.topic_name,
                        forbidden_keywords=self._collect_required_keywords(),
                    )
                else:
                    inst = cls().initialization()
                _ = inst.build_description()
            except Exception:
                continue

            args = inst.args

            # Store as an args dict keyed by instruction id
            inst_id = getattr(inst, "id", key)
            self.instructions[inst_id] = args
            self.cur_topic_turn += 1
            # print("add instruction", inst._description)
            # Tuple format: (operation_type, original_description, new_description, instruction_id, args_before, args_after)
            return ("add", None, inst._description, inst_id, None, copy.deepcopy(args))
        # Failed to add successfully
        print("add random instruction failed")
        return ("failed", None, None)

    def remove_random_instruction(self) -> bool:
        """Randomly remove one instruction."""
        if not self.instructions:
            return ("failed", None, None)
        key_list = list(self.instructions.keys())
        idx = random.randrange(0, len(key_list))
        rem_key = key_list[idx]
        old_args = copy.deepcopy(self.instructions.get(rem_key))
        self.instructions.pop(rem_key, None)
        self.cur_topic_turn += 1
        # Tuple format: (operation_type, original_description, new_description, instruction_id, args_before, args_after)
        return ("remove", None, INSTRUCTION_REMOVE_DICT[rem_key], rem_key, old_args, None)

    def modify_random_instruction(self) -> bool:
        """Randomly modify one instruction.
        """
        if not self.instructions:
            print("modify random instrauction failed")
            return ("failed", None, None)
        key_list = list(self.instructions.keys())

        idx = random.randrange(0, len(key_list))
        target_id = key_list[idx]

        cls = INSTRUCTION_DICT.get(target_id)
        if cls is None:
            print(f"instruction class does not exist, target_id: {target_id}")
            return ("failed", None, None)

        # Rebuild the instruction object using existing args, then modify it
        args = self.instructions.get(target_id)
        try:
            # When rebuilding, pass topic_name and the corresponding mask to avoid get_keywords(None)
            if target_id in {"startwith", "endwith", "existence"}:
                inst = cls().initialization(
                    topic_name=self.topic_name,
                    forbidden_keywords=self._collect_forbidden_keywords(),
                    args=args,
                )
            elif target_id == "forbidden":
                inst = cls().initialization(
                    topic_name=self.topic_name,
                    forbidden_keywords=self._collect_required_keywords(),
                    args=args,
                )
            else:
                inst = cls().initialization(args=args)
        except Exception as e:
            print(e)
            print("modify random instruction failed")
            return ("failed", None, None)

        try:
            if target_id in {"startwith", "endwith", "existence"}:
                original_description, description = inst.modification(
                    topic_name=self.topic_name, forbidden_keywords=self._collect_forbidden_keywords())
            elif target_id == "forbidden":
                original_description, description = inst.modification(
                    topic_name=self.topic_name,
                    forbidden_keywords=self._collect_required_keywords(),
                )
            else:
                original_description, description = inst.modification()
        except Exception as e:
            print(e)
            print("modify random instruction failed")
            return ("failed", None, None)

        try:
            new_args = inst.args
        except Exception:
            new_args = {}

        # Apply changes
        self.instructions[target_id] = new_args
        self.cur_topic_turn += 1
        # print("{} -> prev:{}, after:{}".format(target_id, args, new_args))
        # exit(0)
        # Tuple format: (operation_type, original_description, new_description, instruction_id, args_before, args_after)
        return ("modify", original_description, description, target_id, copy.deepcopy(args), copy.deepcopy(new_args))

    def random_mutate(self) -> tuple:
        """Perform one random mutation (add/remove/modify) for the current topic.
        op can be specified as 'add'|'remove'|'modify'; randomly chosen by default.
        """

        # If no op is specified, choose based on current state
        if len(self.instructions) == 0:
            # Empty instruction set: can only add
            op = "add"
        elif len(self.instructions) == 1:
            # Only one instruction: can only modify (or add)
            op = random.choice(["modify", "add"])
        elif len(self.instructions) == 12:  # <fix>
            op = random.choice(["modify", "remove"])
        else:
            # With existing instructions, randomly choose add/remove/modify
            op = random.choices(["add", "remove", "modify"], weights=[
                                5, 1, 4], k=1)[0]  # run1: 4，2，4

        if op == "add":
            return self.add_random_instruction()
        elif op == "remove":
            return self.remove_random_instruction()
        elif op == "modify":
            return self.modify_random_instruction()

        print("mutate random instruction failed")
        return ("failed", None, None, None, None, None)

    # --------------- Serialization ---------------
    def to_dict(self) -> dict:  # dump to record
        data = {
            "topic_name": self.topic_name,
            "topic_query": self.topic_query,
            "cur_topic_turn": self.cur_topic_turn,
            "instructions": [],
        }
        for inst_id, args in self.instructions.items():
            try:
                data["instructions"].append({
                    "id": inst_id,
                    "args": args,  # keep original type (dict or list)
                })
            except Exception:
                data["instructions"].append({"id": inst_id, "args": {}})
        return data

    @staticmethod
    def from_dict(data: dict) -> "TopicManager":  # load from record
        name = data.get("topic_name")
        topic_query = data.get("topic_query")
        tm = TopicManager(topic_name=name, topic_query=topic_query)
        tm.cur_topic_turn = int(data.get("cur_topic_turn", 0))
        items = data.get("instructions", []) or []
        # Handle duplicate ids: later entries overwrite earlier ones
        for item in items:
            inst_id = item.get("id")
            args = item.get("args")
            if isinstance(inst_id, str) and args is not None:
                # Keep original type (dict or list)
                tm.instructions[inst_id] = args
        return tm


@dataclass
class StateManager:

    # topic_list: list of topics tracked by the StateManager
    topic_list: list[TopicManager] = field(default_factory=list)
    # Track topic names that have been used/covered
    topic_name_list: list[str] = field(default_factory=list)
    # cur_activate_topic: index of the currently active topic
    cur_activate_topic: int = -1
    # last_activation_turn: turn index when the active topic last changed; used with cur_turn to decide switching
    last_activation_turn: int = 0
    # cur_turn: current global turn index
    cur_turn: int = 0  # total turns

    # 每轮最终需遵循的指令快照
    round_instruction_history: list = field(default_factory=list)

    def _find_topic(self, topic_idx: Optional[int]) -> Optional[TopicManager]:
        # Get topic by topic_idx
        if topic_idx is None:
            return None
        if not isinstance(topic_idx, int):
            return None
        if topic_idx < 0 or topic_idx >= len(self.topic_list):
            return None
        return self.topic_list[topic_idx]

    def _create_new_topic(self, topic_name: Optional[str] = None) -> Optional[TopicManager]:
        if topic_name is None:
            candidates = get_topic_list()
            available = [
                n for n in candidates if n not in self.topic_name_list]
            if available:
                topic_name = random.choice(available)
            else:
                # No new topic available; return None
                return None
        tm = TopicManager(topic_name=topic_name,
                          topic_query=TOPIC_QUERY_DICT[topic_name])
        self.topic_name_list.append(topic_name)
        self.topic_list.append(tm)

        # Initialize one instruction to ensure the topic has at least one instruction
        result = tm.add_random_instruction()
        if result[0] == "failed":
            # If adding fails, retry a few times
            for _ in range(5):
                result = tm.add_random_instruction()
                if result[0] != "failed":
                    break

        return tm, result

    def _ensure_active_topic(self) -> tuple[Optional[TopicManager], bool, Optional[tuple]]:
        """Ensure there is an active topic.
        Returns (topic_obj, whether_topic_was_created_or_switched, instruction_mutation_result).
        """
        tm = self._find_topic(self.cur_activate_topic)
        topic_changed = False
        instruction_result = None
        if tm is None:
            created_result = self._create_new_topic()
            if created_result is not None:
                created, instruction_result = created_result
                tm = created
                # Activate the newly created topic
                self.cur_activate_topic = len(self.topic_list) - 1
                self.last_activation_turn = self.cur_turn
                topic_changed = True
            else:
                # If no new topic is available, switch to an existing topic (if any)
                if len(self.topic_list) > 0:
                    self.cur_activate_topic = random.randrange(
                        0, len(self.topic_list))
                    tm = self.topic_list[self.cur_activate_topic]
                    self.last_activation_turn = self.cur_turn
                    topic_changed = True
                    # When switching to an existing topic, mutate its instructions once
                    instruction_result = tm.random_mutate()
                else:
                    return None, False, None
        return tm, topic_changed, instruction_result

    def _switch_topic(self) -> tuple[bool, Optional[str], Optional[tuple]]:
        """Switch topics by randomly choosing between "new" and "backtrack".
        Returns (switched_successfully, new_topic_name, instruction_mutation_result).
        """

        action = random.choices(["new", "backtrack"], weights=[
                                0.4, 0.6], k=1)[0]  # run1:  random

        if action == "backtrack" and len(self.topic_list) > 1:
            candidates = [i for i in range(
                len(self.topic_list)) if i != self.cur_activate_topic]
            if candidates:
                target_idx = random.choice(candidates)
                self.cur_activate_topic = target_idx
                self.last_activation_turn = self.cur_turn
                new_topic = self.topic_list[target_idx]
                # When switching to an existing topic, mutate its instructions once
                instruction_result = new_topic.random_mutate()
                return True, new_topic.topic_name, instruction_result
        # Try creating a new topic
        created_result = self._create_new_topic()
        if created_result is not None:
            created, instruction_result = created_result
            self.cur_activate_topic = len(self.topic_list) - 1
            self.last_activation_turn = self.cur_turn
            return True, created.topic_name, instruction_result
        # No new topic available; backtrack switch
        candidates = [i for i in range(
            len(self.topic_list)) if i != self.cur_activate_topic]
        if candidates:
            target_idx = random.choice(candidates)
            self.cur_activate_topic = target_idx
            self.last_activation_turn = self.cur_turn
            new_topic = self.topic_list[target_idx]
            # When switching to an existing topic, mutate its instructions once
            instruction_result = new_topic.random_mutate()
            return True, new_topic.topic_name, instruction_result
        return False, None, None

    def _snapshot_active_instructions(self, operation_info: Optional[tuple] = None) -> dict:
        active = self._find_topic(self.cur_activate_topic)
        items = []
        if active is not None:
            for inst_id, args in active.instructions.items():
                try:
                    # Rebuild instruction objects to generate descriptions
                    # cls = INSTRUCTION_DICT.get(inst_id)
                    # if cls is not None:
                    #     try:
                    #         inst = cls().initialization(args=args)
                    #         _ = inst.build_description() or ""
                    #     except Exception:
                    #         pass

                    # items.append({
                    #     "id": inst_id,
                    #     "args": args,
                    #     "description": inst._description,
                    # })
                    # fix incorrect description
                    cls = INSTRUCTION_DICT.get(inst_id)
                    description = ""
                    inst = None
                    if cls is not None:
                        if inst_id in {"startwith", "endwith", "existence"}:
                            inst = cls().initialization(
                                topic_name=active.topic_name,
                                forbidden_keywords=active._collect_forbidden_keywords(),
                                args=args,
                            )
                        elif inst_id == "forbidden":
                            inst = cls().initialization(
                                topic_name=active.topic_name,
                                forbidden_keywords=active._collect_required_keywords(),
                                args=args,
                            )
                        else:
                            inst = cls().initialization(args=args)
                        description = inst.build_description() or ""
                    
                    items.append({
                        "id": inst_id,
                        "args": args,
                        "description": description,
                    })
                except Exception:
                    items.append(
                        {"id": inst_id, "args": args, "description": ""})

        snapshot = {
            "turn": self.cur_turn,
            # Record topic name in the snapshot for readability
            "active_topic": active.topic_name if active is not None else None,
            "topic_query": active.topic_query if active is not None else None,
            "instructions": items,
        }

        # If operation info is provided, attach it to the snapshot
        if operation_info is not None:
            # operation_info tuple format:
            # (operation_type, original_description, new_description, instruction_id, args_before, args_after)
            op_type = operation_info[0] if len(operation_info) >= 1 else None
            orig_desc = operation_info[1] if len(operation_info) >= 2 else None
            new_desc = operation_info[2] if len(operation_info) >= 3 else None
            inst_id = operation_info[3] if len(operation_info) >= 4 else None
            args_before = operation_info[4] if len(
                operation_info) >= 5 else None
            args_after = operation_info[5] if len(
                operation_info) >= 6 else None
            try:
                inst_cls_name = INSTRUCTION_DICT[inst_id].__name__ if inst_id in INSTRUCTION_DICT else None
            except Exception:
                inst_cls_name = None
            snapshot["cur_operation"] = {
                "operation_type": op_type,
                "original_description": orig_desc,
                "new_description": new_desc,
                "instruction_id": inst_id,
                "instruction_class": inst_cls_name,
                "args_before": args_before,
                "args_after": args_after,
            }

        return snapshot

    def step(self) -> tuple:
        """Execute one step.
        Each step performs exactly one change: either switch topics or mutate instructions.

        Returns: (operation_type, original_description, new_description)
        """
        # Ensure an active topic exists
        active, topic_changed, instruction_result = self._ensure_active_topic()
        # print(active, topic_changed)
        if active is None:
            print("no activate state -> failed")
            return ("failed", None, None)

        # If we just created/switched topics, record the instruction mutation
        if topic_changed:
            self.cur_turn += 1
            # Record instruction mutation info into the snapshot
            self.round_instruction_history.append(
                self._snapshot_active_instructions(instruction_result))
            return instruction_result

        # Decide this turn's action: if >3 turns have passed and probability condition is met, switch topics; otherwise mutate instructions
        should_try_switch = (self.cur_turn - self.last_activation_turn) > 3

        if should_try_switch and random.random() < 0.8:  # run1: 80% probability to switch topics
            # Switch topics and mutate instructions under the new topic
            old_topic = active.topic_name
            switched, new_topic_name, instruction_result = self._switch_topic()
            if switched:
                # Switch succeeded; record instruction mutation result
                self.cur_turn += 1
                # Record instruction mutation info into the snapshot
                self.round_instruction_history.append(
                    self._snapshot_active_instructions(instruction_result))
                return instruction_result
            else:
                # Switch failed; continue mutating instructions for the current topic
                pass

        # Perform one random mutation on the active topic
        ok = active.random_mutate()
        # Turn +1
        self.cur_turn += 1
        # Record snapshot (final state after this turn), and include operation info in the snapshot
        self.round_instruction_history.append(
            self._snapshot_active_instructions(ok))
        return ok

    # --------------- Serialization ---------------
    def to_dict(self) -> dict:
        return {
            "topic_list": [t.to_dict() for t in self.topic_list],
            # Save index directly
            "cur_activate_topic": int(self.cur_activate_topic),
            "last_activation_turn": self.last_activation_turn,
            "cur_turn": self.cur_turn,
            "topic_name_list": list(self.topic_name_list),
            "round_instruction_history": list(self.round_instruction_history),
            # Save RNG state to ensure determinism for --resume
            "rng_state": _rng_state_to_jsonable(random.getstate()),
        }

    @staticmethod
    def from_dict(data: dict) -> "StateManager":
        sm = StateManager()
        # Read as index; default -1
        sm.cur_activate_topic = int(data.get("cur_activate_topic", -1))
        sm.last_activation_turn = int(data.get("last_activation_turn", 0))
        sm.cur_turn = int(data.get("cur_turn", 0))
        sm.round_instruction_history = data.get(
            "round_instruction_history", []) or []
        tlist = data.get("topic_list", []) or []
        for item in tlist:
            sm.topic_list.append(TopicManager.from_dict(item))
        # Read topic name list (if missing, recover from topic_list)
        sm.topic_name_list = data.get("topic_name_list", []) or [
            t.topic_name for t in sm.topic_list]
        # If the index is invalid but topics exist, point to 0
        if (sm.cur_activate_topic < 0 or sm.cur_activate_topic >= len(sm.topic_list)) and len(sm.topic_list) > 0:
            sm.cur_activate_topic = 0
        # Restore RNG state (if present)
        try:
            rng_state = data.get("rng_state")
            if rng_state is not None:
                random.setstate(_jsonable_to_rng_state(rng_state))
        except Exception:
            pass
        return sm

    def save_to_file(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_from_file(filepath: str) -> "StateManager":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return StateManager.from_dict(data)
