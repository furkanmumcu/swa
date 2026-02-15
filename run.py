import json
from typing import List, Dict, Optional
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---- Config ----
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

NUM_AGENTS = 5
CAPACITY = 20.0
BETA = 1.6

LAMBDA_VAL = 0.1
NUM_STEPS = 20

K_CANDIDATES = 7
GRID_STEP = 0.25

EMA_ALPHA = 0.2
SAFETY_MARGIN = 2.0

USE_CENTERING = True
GAMMA = 0.02

ANCHORS = [0.0, 2.0, 3.0, 4.0, 4.5]
SEED = 0


SYSTEM_PROMPT = (
    "You are an agent in a shared resource allocation environment.\n"
    "Each turn you must request a real number between 0 and 10.\n"
    "There are 5 agents. System capacity is 20.\n"
    "If total demand exceeds capacity, everyone is penalized.\n"
    "You will be given a table of candidate actions and computed scores.\n"
    "You MUST choose exactly one candidate from the table.\n"
    "Return ONLY valid JSON.\n"
)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def chat_generate(
    tokenizer,
    model,
    user_content: str,
    temperature: float = 0.2,
    do_sample: bool = False,
    max_new_tokens: int = 200,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    attention_mask = torch.ones_like(input_ids, device=model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = out[0, input_ids.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def env_step(demands: List[float]) -> Dict[str, float]:
    total = float(np.sum(demands))
    excess = max(0.0, total - CAPACITY)
    penalty = (BETA / NUM_AGENTS) * excess
    welfare = total - BETA * excess
    status = "OVERLOADED" if excess > 0 else "SAFE"
    return {
        "total": total,
        "excess": excess,
        "penalty": penalty,
        "welfare": welfare,
        "status": status,
    }


def predicted_welfare(total_hat: float) -> float:
    eff_capacity = CAPACITY - SAFETY_MARGIN
    excess_hat = max(0.0, total_hat - eff_capacity)
    w = total_hat - BETA * excess_hat
    if USE_CENTERING:
        w -= GAMMA * (total_hat - eff_capacity) ** 2
    return w


def group_score(mu_est: float, candidate: float) -> float:
    total_hat = candidate + (NUM_AGENTS - 1) * mu_est
    w_hat = predicted_welfare(total_hat)

    eff_capacity = CAPACITY - SAFETY_MARGIN
    w_target = predicted_welfare(eff_capacity)

    s = w_hat / (w_target + 1e-8)
    return float(np.clip(s, 0.0, 1.0))


def self_score(candidate: float) -> float:
    return float(np.clip(candidate / 10.0, 0.0, 1.0))


def swa_utility(s_self: float, s_group: float, lam: float) -> float:
    return (1.0 - lam) * s_self + lam * s_group


def llm_propose_candidates(tokenizer, model, public_board: str, k: int) -> List[float]:
    prompt = (
        f"{public_board}\n\n"
        f"Propose {k} different candidate requests between 0 and 10.\n"
        f"Return ONLY JSON like: {{\"candidates\": [1.0, 3.5, 7.25]}}\n"
        f"Rules: candidates must be numbers, must be distinct, 0<=x<=10.\n"
    )
    txt = chat_generate(
        tokenizer, model, prompt, temperature=0.7, do_sample=True, max_new_tokens=120
    )

    try:
        obj = json.loads(txt)
        cands = obj.get("candidates", [])
    except Exception:
        cands = []

    out: List[float] = []
    for c in cands:
        try:
            v = float(c)
            if 0.0 <= v <= 10.0 and v not in out:
                out.append(v)
        except Exception:
            pass

    for a in ANCHORS:
        if float(a) not in out:
            out.append(float(a))

    if len(out) < 3:
        grid = np.round(np.arange(0.0, 10.0 + 1e-9, GRID_STEP), 2)
        out = list(
            np.random.choice(grid, size=max(k, len(ANCHORS)), replace=False).astype(float)
        )
        for a in ANCHORS:
            if float(a) not in out:
                out.append(float(a))

    return out[: max(k, len(ANCHORS))]


def llm_choose_from_table(
    tokenizer,
    model,
    agent_id: int,
    public_board: str,
    lam: float,
    candidates: List[float],
    scores: List[Dict[str, float]],
) -> float:
    table = "\n".join(
        [
            f"a={r['a']:.2f} | s_self={r['s_self']:.3f} | s_group={r['s_group']:.3f} | U={r['utility']:.3f}"
            for r in scores
        ]
    )

    prompt = (
        f"{public_board}\n\n"
        f"You are Agent {agent_id}.\n"
        f"Lambda = {lam:.2f}.\n"
        f"Candidate actions and scores:\n"
        f"{table}\n\n"
        f"Choose the action with the HIGHEST U.\n"
        f"Return ONLY JSON like: {{\"action\": 3.50}} (must match one listed a=...)\n"
    )

    txt = chat_generate(
        tokenizer, model, prompt, temperature=0.0, do_sample=False, max_new_tokens=80
    )

    try:
        obj = json.loads(txt)
        a = float(obj["action"])
    except Exception:
        a = None

    if a is None:
        return float(candidates[np.argmax([r["utility"] for r in scores])])

    return float(min(candidates, key=lambda x: abs(x - a)))


def run_experiment():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    tokenizer, model = load_model()

    mu_est = CAPACITY / NUM_AGENTS
    last_demands: Optional[List[float]] = None

    totals: List[float] = []
    overloads = 0

    print("--- Experiment ---")
    print(
        f"n={NUM_AGENTS} C={CAPACITY} beta={BETA} lambda={LAMBDA_VAL} "
        f"steps={NUM_STEPS} k={K_CANDIDATES}"
    )

    for t in range(NUM_STEPS):
        if last_demands is not None:
            mu_est = (1.0 - EMA_ALPHA) * mu_est + EMA_ALPHA * float(np.mean(last_demands))

        public_board = (
            f"Time step: {t}\n"
            f"Capacity: {CAPACITY}\n"
            f"Penalty model: if total > C, penalty per agent = (beta/n)*(total-C), beta={BETA}\n"
            f"Belief mean (mu_est): {mu_est:.2f}\n"
        )

        if last_demands is None:
            public_board += "Last turn: none\n"
        else:
            prev = env_step(last_demands)
            public_board += (
                f"Last demands: {last_demands}\n"
                f"Last total: {prev['total']:.2f} ({prev['status']})\n"
            )

        step_demands: List[float] = []
        print(f"\n--- t={t} --- mu_est={mu_est:.2f}")

        for i in range(1, NUM_AGENTS + 1):
            cands = llm_propose_candidates(tokenizer, model, public_board, K_CANDIDATES)

            rows = []
            for a in cands:
                s_s = self_score(a)
                s_g = group_score(mu_est, a)
                u = swa_utility(s_s, s_g, LAMBDA_VAL)
                rows.append({"a": a, "s_self": s_s, "s_group": s_g, "utility": u})

            action = llm_choose_from_table(
                tokenizer, model, i, public_board, LAMBDA_VAL, cands, rows
            )
            step_demands.append(action)
            print(f"Agent {i} -> {action:.2f} (cands={sorted([round(x, 2) for x in cands])})")

        env = env_step(step_demands)
        totals.append(env["total"])
        overloads += int(env["excess"] > 0)

        print(f"Total={env['total']:.1f} ({env['status']}), welfare={env['welfare']:.2f}")
        if env["excess"] > 0:
            print(f"  Excess={env['excess']:.1f}, penalty/agent={env['penalty']:.2f}")

        last_demands = step_demands

    print("\n--- Summary ---")
    print(f"Average total load: {float(np.mean(totals)):.2f}")
    print(f"Overloaded steps: {overloads}/{NUM_STEPS}")


if __name__ == "__main__":
    run_experiment()
