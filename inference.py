"jaaganath"
import os
from attendance_env import AttendanceEnv, Action
from openai import OpenAI

# ✅ Defaults set ONLY for API_BASE_URL and MODEL_NAME (as required)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
# ✅ NO default for HF_TOKEN (as required)
HF_TOKEN = os.getenv("HF_TOKEN")
# Optional - only if using from_docker_image()
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

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
    
    # HF_TOKEN must be provided by validator (no default)
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN environment variable not set", flush=True)
        log_end(False, 0, 0.0, [])
        return
    
    # ✅ All LLM calls use OpenAI client configured via these variables
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN
    )
    
    # Required LLM ping test
    try:
        test_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=2
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
            
            # Extract state info
            location = state.get("location", "")
            ambiguity = state.get("ambiguity", 0)
            fraud_signal = state.get("fraud_signal", False)
            student_id = state.get("student_id", "")
            
            # ✅ LLM call through their proxy
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an attendance system. Reply with only 0, 1, or 2."},
                    {"role": "user", "content": f"Student: {student_id}, Location: {location}, Ambiguity: {ambiguity}, Fraud: {fraud_signal}. Choose: 0=present, 1=absent, 2=suspicious"}
                ],
                max_tokens=5,
                temperature=0.0
            )
            
            action = int(response.choices[0].message.content.strip())
            if action not in [0, 1, 2]:
                action = 2
            
            _, reward, done, _ = env.step(action)
            
            action_map = {0: "MARK_PRESENT", 1: "MARK_ABSENT", 2: "FLAG_SUSPICIOUS"}
            action_str = action_map[action]
            
            rewards.append(reward)
            steps = step_num
            
            log_step(step_num, action_str, reward, done, None)
            
            if done:
                break
        
        score = sum(rewards) / len(rewards) if rewards else 0
        score = max(0.0, min(1.0, score))
        success = score > 0.5
        
    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)
        score = 0.0
        success = False
    finally:
        env.close()
        log_end(success, steps, score, rewards)

if __name__ == "__main__":
    run()
