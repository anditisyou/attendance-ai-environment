import os
from attendance_env import AttendanceEnv, Action
from openai import OpenAI

# ✅ STRICT: Use only injected environment variables
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

# ---------------- LOGGING ---------------- #

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

# ---------------- MAIN ---------------- #

def run():
    log_start()

    # ✅ Initialize client (NO extra params)
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY
        )
    except Exception as e:
        print(f"[LLM INIT ERROR] {str(e)}", flush=True)
        log_end(False, 0, 0.0, [])
        return

    # ✅ FORCE API CALL (VERY IMPORTANT)
    try:
        test = client.responses.create(
            model=MODEL_NAME,
            input="Hello",
            max_output_tokens=2
        )
        print("[LLM CALL MADE]", flush=True)
    except Exception as e:
        print(f"[LLM ERROR] {str(e)}", flush=True)
        log_end(False, 0, 0.0, [])
        return

    # ✅ Init environment
    try:
        env = AttendanceEnv()
    except Exception as e:
        print(f"[ENV ERROR] {str(e)}", flush=True)
        log_end(False, 0, 0.0, [])
        return

    rewards = []
    steps = 0
    success = False

    try:
        for step_num, difficulty in enumerate(["easy", "medium", "hard"], start=1):

            try:
                state = env.reset(difficulty)
            except Exception as e:
                print(f"[RESET ERROR] {str(e)}", flush=True)
                break

            location = state.get("location", "")
            ambiguity = state.get("ambiguity", 0)
            fraud_signal = state.get("fraud_signal", False)
            student_id = state.get("student_id", "")

            prompt = f"""
Student: {student_id}
Location: {location}
Ambiguity: {ambiguity}
Fraud signal: {fraud_signal}

Choose action:
0 = present
1 = absent
2 = suspicious
Only reply with 0, 1, or 2.
"""

            # ✅ LLM decision (proxy-safe)
            try:
                response = client.responses.create(
                    model=MODEL_NAME,
                    input=prompt,
                    max_output_tokens=5,
                    temperature=0.0
                )

                output_text = response.output[0].content[0].text.strip()
                action = int(output_text[0])

                if action not in [0, 1, 2]:
                    action = 2

            except Exception as e:
                print(f"[LLM STEP ERROR] {str(e)}", flush=True)

                # ✅ fallback logic
                if fraud_signal or ambiguity > 0.6:
                    action = 2
                elif "classroom" not in location.lower() and "lab" not in location.lower():
                    action = 1
                else:
                    action = 0

            # ✅ Execute action
            try:
                _, reward, done, info = env.step(action)
                error = None
            except Exception as e:
                reward = 0.0
                done = False
                error = str(e)

            action_map = {
                0: "MARK_PRESENT",
                1: "MARK_ABSENT",
                2: "FLAG_SUSPICIOUS"
            }

            rewards.append(reward)
            steps = step_num

            log_step(step_num, action_map[action], reward, done, error)

            if done:
                continue  # environment ends each step, move to next difficulty

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(0.0, min(1.0, score))
        success = score > 0.5

    except Exception as e:
        print(f"[UNEXPECTED ERROR] {str(e)}", flush=True)
        score = 0.0
        success = False

    finally:
        try:
            env.close()
        except Exception:
            pass

        log_end(success, steps, score, rewards)


if __name__ == "__main__":
    run()
