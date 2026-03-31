from attendance_env import AttendanceEnv

import random

def simple_agent(state):
    # introduce randomness (realistic agent behavior)
    
    if random.random() < 0.3:
        return "flag_suspicious"

    if state["location"] != "classroom":
        return "mark_absent"

    if state["time"] > "10:10":
        return "mark_absent"

    return "mark_present"

def run_task(env, difficulty):
    total_reward = 0
    episodes = 5

    for _ in range(episodes):
        state = env.reset(difficulty)
        action = simple_agent(state)
        _, reward, _, info = env.step(action)

        print(f"[{difficulty}] State:", state)
        print(f"[{difficulty}] Action:", action)
        print(f"[{difficulty}] Correct:", info["correct_action"])
        print(f"[{difficulty}] Reward:", reward)
        print("-----")

        total_reward += reward

    avg_score = total_reward / episodes
    return avg_score


if __name__ == "__main__":
    env = AttendanceEnv()

    print("Running baseline agent...\n")

    for level in ["easy", "medium", "hard"]:
        score = run_task(env, level)
        print(f"{level.upper()} score: {score:.2f}")