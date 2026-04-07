import os
from attendance_env import AttendanceEnv, Action

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost")
MODEL_NAME = os.getenv("MODEL_NAME", "optimized-agent")
HF_TOKEN = os.getenv("HF_TOKEN")


def get_action(state):
    try:
        location = state.get("location", "").lower()
        ambiguity = state.get("ambiguity", 0)
        fraud_signal = state.get("fraud_signal", False)
        student_id = state.get("student_id", "").lower()
        confidence = state.get("confidence", 1.0)

        if fraud_signal or ambiguity > 0.6:
            return Action.FLAG_SUSPICIOUS.value

        if "unknown" in student_id or "corridor" in location:
            return Action.FLAG_SUSPICIOUS.value

        if "classroom" not in location and "lab" not in location:
            return Action.MARK_ABSENT.value

        if confidence < 0.5:
            return Action.FLAG_SUSPICIOUS.value

        return Action.MARK_PRESENT.value

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        return Action.FLAG_SUSPICIOUS.value


def run():
    env = AttendanceEnv()

    total_reward = 0
    steps = 0

    print("[START] task=attendance_validation", flush=True)

    for i, difficulty in enumerate(["easy", "medium", "hard"], start=1):
        try:
            state = env.reset(difficulty)
            action = int(get_action(state))

            _, reward, _, info = env.step(action)

            total_reward += reward
            steps += 1

            print(f"[STEP] step={i} difficulty={difficulty} reward={reward}", flush=True)

        except Exception as e:
            print(f"[STEP] step={i} error={e}", flush=True)

    score = total_reward / steps if steps > 0 else 0

    print(f"[END] task=attendance_validation score={score} steps={steps}", flush=True)


if __name__ == "__main__":
    run()
