import os
import sys
from attendance_env import AttendanceEnv, Action
from openai import OpenAI

def log_start():
    print(f"[START] task=attendance env=attendance_env model={os.getenv('MODEL_NAME', 'unknown')}", flush=True)

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
    """Call the LLM to decide the action with proper error handling"""
    
    try:
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

Available actions (respond with ONLY the number):
0 = MARK_PRESENT
1 = MARK_ABSENT  
2 = FLAG_SUSPICIOUS

Rules:
- If fraud_signal is True or ambiguity > 0.6, use 2 (FLAG_SUSPICIOUS)
- If location does not contain 'classroom' or 'lab', use 1 (MARK_ABSENT)
- Otherwise use 0 (MARK_PRESENT)

Action number:"""

        response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": "You are an attendance assistant. Reply with ONLY a single number (0, 1, or 2)."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5,
            temperature=0.0,
            timeout=10.0
        )
        
        action_num = int(response.choices[0].message.content.strip())
        if action_num not in [0, 1, 2]:
            return 2
        return action_num
        
    except Exception as e:
        print(f"[LLM ERROR] {str(e)}", flush=True)
        # Fallback to rule-based logic
        if fraud_signal or ambiguity > 0.6:
            return Action.FLAG_SUSPICIOUS.value
        if "classroom" not in location.lower() and "lab" not in location.lower():
            return Action.MARK_ABSENT.value
        return Action.MARK_PRESENT.value

def run():
    log_start()
    
    # Get environment variables with defaults for local testing
    api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    
    # For local testing, you can set defaults
    if not api_base:
        api_base = "https://api.openai.com/v1"  # Default for local testing
        print("[WARN] API_BASE_URL not set, using default", flush=True)
    
    if not api_key:
        print("[ERROR] HF_TOKEN or API_KEY environment variable not set", flush=True)
        log_end(False, 0, 0.0, [])
        return
    
    # Initialize OpenAI client with error handling
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=30.0,
            max_retries=2
        )
        print("[LLM] Client initialized successfully", flush=True)
    except Exception as e:
        print(f"[LLM] Client initialization error: {str(e)}", flush=True)
        log_end(False, 0, 0.0, [])
        return
    
    # Test LLM connection (required by validator)
    try:
        test_response = client.chat.completions.create(
            model=model_name or "gpt-3.5-turbo",
            messages=[{"role": "user", "content": "OK"}],
            max_tokens=2,
            timeout=5.0
        )
        print("[LLM] success", flush=True)
    except Exception as e:
        print(f"[LLM] Connection test failed: {str(e)}", flush=True)
        # Continue anyway - maybe the test fails but actual calls work
        print("[LLM] Continuing despite test failure", flush=True)
    
    # Initialize environment
    try:
        env = AttendanceEnv()
    except Exception as e:
        print(f"[ERROR] Failed to initialize environment: {str(e)}", flush=True)
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
                print(f"[ERROR] Reset failed for {difficulty}: {str(e)}", flush=True)
                break
            
            # Get action from LLM
            action = get_llm_action(state, client)
            
            # Execute action
            try:
                _, reward, done, info = env.step(int(action))
                error = None
            except Exception as e:
                reward = 0.0
                done = False
                error = f"step_failed: {str(e)}"
            
            rewards.append(reward)
            steps = step_num
            
            # Convert action number to string representation
            action_map = {0: "MARK_PRESENT", 1: "MARK_ABSENT", 2: "FLAG_SUSPICIOUS"}
            action_str = action_map.get(action, "FLAG_SUSPICIOUS")
            
            log_step(step_num, action_str, reward, done, error)
            
            if done:
                break
        
        # Calculate score (normalized to 0-1)
        if rewards:
            score = sum(rewards) / len(rewards)
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            success = score > 0.5  # Success if average reward > 0.5
        else:
            score = 0.0
            success = False
        
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}", flush=True)
        score = 0.0
        success = False
        
    finally:
        try:
            env.close()
        except:
            pass
        log_end(success, steps, score, rewards)

if __name__ == "__main__":
    run()
