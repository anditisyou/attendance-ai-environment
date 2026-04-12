import os
import sys
from attendance_env import AttendanceEnv, Action
from openai import OpenAI

# ✅ Use exactly what the validator injects — no fallback defaults
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY      = os.environ["API_KEY"]
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

def log_start():
    print(f"[START] task=attendance env=attendance_env model={MODEL_NAME}", flush=True)

def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True
    )

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True
    )

def run():
    log_start()

    # Initialize OpenAI client pointing to validator's proxy
    try:
        client = OpenAI(
            base_url=API_BASE_URL,   # ✅ validator's LiteLLM proxy
            api_key=API_KEY,         # ✅ validator's injected key
            timeout=30.0,
            max_retries=2
        )
    except Exception as e:
        print(f"[LLM] Client initialization error: {str(e)}", flush=True)
        log_end(False, 0, 0.0, [])
        return

    # Test LLM connection
    try:
        test_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "OK"}],
            max_tokens=2,
            timeout=10.0
        )
        print("[LLM] success", flush=True)
    except Exception as e:
        print(f"[LLM] error: {str(e)}", flush=True)
        log_end(False, 0, 0.0, [])
        return

    # Initialize environment
    try:
        env = AttendanceEnv()
    except Exception as e:
        print(f"[ERROR] Environment init error: {str(e)}", flush=True)
        log_end(False, 0, 0.0, [])
        return

    rewards = []
    steps   = 0
    success = False

    try:
        for step_num, difficulty in enumerate(["easy", "medium", "hard"], start=1):
            try:
                state = env.reset(difficulty)
            except Exception as e:
                print(f"[ERROR] Reset failed: {str(e)}", flush=True)
                break

            location     = state.get("location", "")
            ambiguity    = state.get("ambiguity", 0)
            fraud_signal = state.get("fraud_signal", False)
            student_id   = state.get("student_id", "")

            prompt = f"""Student: {student_id}
Location: {location}
Ambiguity: {ambiguity}
Fraud signal: {fraud_signal}

Choose action (0=present, 1=absent, 2=suspicious):"""

            # LLM decision
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "Reply with only 0, 1, or 2"},
                        {"role": "user",   "content": prompt}
                    ],
                    max_tokens=5,
                    temperature=0.0,
                    timeout=10.0
                )
                action = int(response.choices[0].message.content.strip()[0])
                if action not in [0, 1, 2]:
                    action = 2
            except Exception as e:
                print(f"[LLM] Step {step_num} error: {str(e)}", flush=True)
                # Fallback rule-based logic
                if fraud_signal or ambiguity > 0.6:
                    action = 2
                elif "classroom" not in location.lower() and "lab" not in location.lower():
                    action = 1
                else:
                    action = 0

            # Execute action
            try:
                _, reward, done, info = env.step(action)
                error = None
            except Exception as e:
                reward = 0.0
                done   = False
                error  = str(e)

            action_map = {0: "MARK_PRESENT", 1: "MARK_ABSENT", 2: "FLAG_SUSPICIOUS"}
            action_str = action_map[action]

            rewards.append(reward)
            steps = step_num

            log_step(step_num, action_str, reward, done, error)

            if done:
                break

        score   = sum(rewards) / len(rewards) if rewards else 0.0
        score   = max(0.0, min(1.0, score))
        success = score > 0.5

    except Exception as e:
        print(f"[ERROR] Unexpected: {str(e)}", flush=True)
        score   = 0.0
        success = False
    finally:
        try:
            env.close()
        except Exception:
            pass
        log_end(success, steps, score, rewards)

if __name__ == "__main__":
    run()
