import os
from attendance_env import AttendanceEnv, Action
from openai import OpenAI

def log_start():
    print(f"[START] task=attendance env=attendance_env model={os.getenv('MODEL_NAME')}", flush=True)

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
    
    # CRITICAL: Get environment variables - NO FALLBACKS, NO ALTERNATIVES
    api_base = os.getenv("API_BASE_URL")
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    model_name = os.getenv("MODEL_NAME")
    
    # MUST exist - fail if not provided
    if not api_base:
        print("[ERROR] API_BASE_URL not set", flush=True)
        log_end(False, 0, 0.0, [])
        return
    
    if not api_key:
        print("[ERROR] HF_TOKEN or API_KEY not set", flush=True)
        log_end(False, 0, 0.0, [])
        return
    
    if not model_name:
        print("[ERROR] MODEL_NAME not set", flush=True)
        log_end(False, 0, 0.0, [])
        return
    
    # Initialize client with EXACTLY the provided values - NO ALTERNATIVES
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=api_base  # Use EXACTLY what was provided
        )
    except Exception as e:
        print(f"[LLM] Client init error: {str(e)}", flush=True)
        log_end(False, 0, 0.0, [])
        return
    
    # REQUIRED: Test LLM connection through THEIR proxy
    try:
        test_response = client.chat.completions.create(
            model=model_name,  # Use EXACT model name provided
            messages=[{"role": "user", "content": "OK"}],
            max_tokens=2
        )
        print("[LLM] success", flush=True)
    except Exception as e:
        print(f"[LLM] error: {str(e)}", flush=True)
        log_end(False, 0, 0.0, [])
        return
    
    # Initialize environment
    env = AttendanceEnv()
    rewards = []
    steps = 0
    success = False
    
    try:
        for step_num, difficulty in enumerate(["easy", "medium", "hard"], start=1):
            state = env.reset(difficulty)
            
            # Get action from LLM through THEIR proxy
            location = state.get("location", "")
            ambiguity = state.get("ambiguity", 0)
            fraud_signal = state.get("fraud_signal", False)
            student_id = state.get("student_id", "")
            
            prompt = f"""Student: {student_id}
Location: {location}
Ambiguity: {ambiguity}
Fraud signal: {fraud_signal}

Choose action (0=present, 1=absent, 2=suspicious):"""
            
            try:
                response = client.chat.completions.create(
                    model=model_name,  # Use EXACT model name
                    messages=[
                        {"role": "system", "content": "Reply with only 0, 1, or 2"},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=5,
                    temperature=0.0
                )
                action = int(response.choices[0].message.content.strip())
                if action not in [0, 1, 2]:
                    action = 2
            except Exception as e:
                # Fallback only if LLM call fails
                print(f"[LLM] Step {step_num} error: {str(e)}", flush=True)
                if fraud_signal or ambiguity > 0.6:
                    action = 2
                elif "classroom" not in location.lower() and "lab" not in location.lower():
                    action = 1
                else:
                    action = 0
            
            # Execute action
            _, reward, done, info = env.step(action)
            
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
