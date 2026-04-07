from attendance_env import AttendanceEnv, Action
import random

def simple_agent(state):
    """
    Improved agent with:
    - Correct action format (int)
    - Better logic using env signals
    """

    try:
        # ✅ Priority 1: fraud detection
        if state.get("fraud_signal", False):
            return Action.FLAG_SUSPICIOUS.value

        # ✅ Priority 2: invalid location
        if not state["location"].startswith("classroom"):
            return Action.MARK_ABSENT.value

        # ✅ Priority 3: time check (extract from timestamp)
        timestamp = state["timestamp"]  # ISO format
        time_str = timestamp[11:16]     # HH:MM

        if time_str > "10:10":
            return Action.MARK_ABSENT.value

        # ✅ Random cautious behavior
        if random.random() < 0.2:
            return Action.FLAG_SUSPICIOUS.value

        return Action.MARK_PRESENT.value

    except Exception as e:
        print("Agent error:", e)
        return Action.FLAG_SUSPICIOUS.value  # safe fallback


def run_task(env, difficulty):
    total_reward = 0
    episodes = 5

    for _ in range(episodes):
        state = env.reset(difficulty)

        try:
            action = simple_agent(state)
            action = int(action)  # ensure correct type

            print(f"[{difficulty}] Action being sent:", action)

            _, reward, _, info = env.step(action)

        except Exception as e:
            print("ERROR during env.step:", e)
            reward = 0
            info = {}

        print(f"[{difficulty}] State:", state)
        print(f"[{difficulty}] Action:", action)
        print(f"[{difficulty}] Correct:", info.get("ground_truth"))
        print(f"[{difficulty}] Reward:", reward)
        print("-----")

        total_reward += reward

    return total_reward / episodes


if __name__ == "__main__":
    env = AttendanceEnv()

    print("Running improved agent...\n")

    for level in ["easy", "medium", "hard"]:
        score = run_task(env, level)
        print(f"{level.upper()} score: {score:.2f}")
