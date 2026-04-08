import os
from attendance_env import AttendanceEnv, Action
from openai import OpenAI

# ✅ Use the provided environment variables (DO NOT hardcode)
API_BASE_URL = os.getenv("API_BASE_URL")  # NO default - must come from validator
MODEL_NAME = os.getenv("MODEL_NAME")      # NO default - must come from validator
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")  # Use HF_TOKEN or API_KEY

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

def get_llm_action(state, client):
    """Call the LLM to decide the action"""
    
    # Build prompt from state
    location = state.get("location", "unknown")
    ambiguity = state.get("ambiguity", 0)
    fraud_signal = state.get("fraud_signal", False)
    student_id = state.get("student_id", "unknown")
    
    prompt = f"""You are an attendance monitoring system. Based on the following state, decide the appropriate action.

State:
- Student ID: {student_id}
- Location: {location}
- Ambiguity score: {ambiguity}
- Fraud signal: {fraud_signal}

Available actions:
- 0: MARK_PRESENT (student is present)
- 1: MARK_ABSENT (student is absent)
- 2: FLAG_SUSPICIOUS (suspicious activity detected)

Rules:
- If fraud_signal is True or ambiguity > 0.6, use FLAG_SUSPICIOUS
- If location is not classroom or lab, use MARK_ABSENT
- Otherwise use MARK_PRESENT

Respond with ONLY the action number (0, 1, or 2)."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an attendance monitoring assistant. Reply only with the action number."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.0
        )
        
        action_num = int(response.choices[0].message.content.strip())
        # Ensure valid action
        if action_num not in [0, 1, 2]:
            return 2  # FLAG_SUSPICIOUS as fallback
        return action_num
        
    except Exception as e:
        print(f"[LLM ERROR] {e}", flush=True)
        # Fallback to rule-based logic
        if fraud_signal or ambiguity > 0.6:
            return Action.FLAG_SUSPICIOUS.value
        if "classroom" not in location.lower() and "lab" not in location.lower():
            return Action.MARK_ABSENT.value
        return Action.MARK_PRESENT.value

def run():
    log_start()
    
    # ✅ Initialize OpenAI client with PROVIDED credentials
    if not API_BASE_URL or not API_KEY:
        print("[ERROR] API_BASE_URL and HF_TOKEN/API_KEY must be set", flush=True)
        log_end(False, 0, 0.0, [])
        return
    
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL
    )
    
    # Test the connection (required by validator)
    try:
        test_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=5
        )
        print("[LLM] success", flush=True)
    except Exception as e:
        print(f"[LLM] error: {str(e)}", flush=True)
        log_end(False, 0, 0.0, [])
        return
    
    env = AttendanceEnv()
    rewards = []
    steps = 0
    success = False
    
    try:
        for step_num, difficulty in enumerate(["easy", "medium", "hard"], start=1):
            state = env.reset(difficulty)
            action = get_llm_action(state, client)
            
            try:
                _, reward, done, info = env.step(int(action))
                error = None
            except Exception as e:
                reward = 0.0
                done = False
                error = "step_failed"
            
            rewards.append(reward)
            steps = step_num
            
            # Convert action number to string representation
            action_str = ["MARK_PRESENT", "MARK_ABSENT", "FLAG_SUSPICIOUS"][action]
            log_step(step_num, action_str, reward, done, error)
            
            if done:
                break
        
        score = sum(rewards) / len(rewards) if rewards else 0
        success = score > 0
        
    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        score = 0
        
    finally:
        log_end(success, steps, score, rewards)

if __name__ == "__main__":
    run()
