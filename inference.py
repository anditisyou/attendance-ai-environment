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

        # 🔥 1. FRAUD DETECTION (TOP PRIORITY)
        if fraud_signal:
            return Action.FLAG_SUSPICIOUS.value

        if "unknown" in student_id:
            return Action.FLAG_SUSPICIOUS.value

        if "corridor" in location:
            return Action.FLAG_SUSPICIOUS.value

        # 🔥 2. HIGH AMBIGUITY → PLAY SAFE
        if ambiguity > 0.6:
            return Action.FLAG_SUSPICIOUS.value

        # ⚠️ 3. MEDIUM AMBIGUITY → CAUTIOUS
        if 0.3 < ambiguity <= 0.6:
            return Action.FLAG_SUSPICIOUS.value

        # ❌ 4. INVALID LOCATION
        if "classroom" not in location and "lab" not in location:
            return Action.MARK_ABSENT.value

        # ⚠️ 5. LOW CONFIDENCE → SUSPICIOUS
        if confidence < 0.5:
            return Action.FLAG_SUSPICIOUS.value

        # ✅ 6. CLEAN CASE → PRESENT
        return Action.MARK_PRESENT.value

    except Exception as e:
        print("ERROR in agent:", e)
        return Action.FLAG_SUSPICIOUS.value


def run():
    print("START")

    env = AttendanceEnv()

    for difficulty in ["easy", "medium", "hard"]:
        print(f"STEP difficulty={difficulty}")

        try:
            state = env.reset(difficulty)
            action = int(get_action(state))

            _, reward, _, info = env.step(action)

            print(f"ACTION {action}")
            print(f"REWARD {reward}")
            print(f"CORRECT {info.get('ground_truth')}")

        except Exception as e:
            print("ERROR:", e)

    print("END")


if __name__ == "__main__":
    run()
