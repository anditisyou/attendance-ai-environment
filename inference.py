import os
from attendance_env import AttendanceEnv, Action

# ✅ Required env variables (even if unused)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost")
MODEL_NAME = os.getenv("MODEL_NAME", "baseline-model")
HF_TOKEN = os.getenv("HF_TOKEN")


def get_action(state):
    try:
        location = state.get("location", "").lower()
        ambiguity = state.get("ambiguity", 0)
        fraud_signal = state.get("fraud_signal", False)
        student_id = state.get("student_id", "").lower()

        if fraud_signal or ambiguity > 0.6:
            return Action.FLAG_SUSPICIOUS.value

        if "corridor" in location or "unknown" in student_id:
            return Action.FLAG_SUSPICIOUS.value

        if "classroom" not in location and "lab" not in location:
            return Action.MARK_ABSENT.value

        if ambiguity < 0.2:
            return Action.MARK_PRESENT.value

        return Action.FLAG_SUSPICIOUS.value

    except Exception as e:
        print("ERROR in agent:", e)
        return Action.FLAG_SUSPICIOUS.value


def run():
    print("START")  # ✅ REQUIRED FORMAT

    env = AttendanceEnv()

    for difficulty in ["easy", "medium", "hard"]:
        print(f"STEP difficulty={difficulty}")  # ✅ REQUIRED FORMAT

        try:
            state = env.reset(difficulty)
            action = int(get_action(state))

            _, reward, _, info = env.step(action)

            print(f"ACTION {action}")
            print(f"REWARD {reward}")
            print(f"CORRECT {info.get('ground_truth')}")

        except Exception as e:
            print("ERROR:", e)

    print("END")  # ✅ REQUIRED FORMAT


if __name__ == "__main__":
    run()
