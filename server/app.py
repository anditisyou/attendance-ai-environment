from fastapi import FastAPI
from attendance_env import AttendanceEnv, Action

app = FastAPI()

env = AttendanceEnv()


def get_action(state):
    location = state.get("location", "").lower()
    ambiguity = state.get("ambiguity", 0)
    fraud_signal = state.get("fraud_signal", False)
    student_id = state.get("student_id", "").lower()

    if fraud_signal or ambiguity > 0.6:
        return Action.FLAG_SUSPICIOUS.value

    if "unknown" in student_id or "corridor" in location:
        return Action.FLAG_SUSPICIOUS.value

    if "classroom" not in location and "lab" not in location:
        return Action.MARK_ABSENT.value

    return Action.MARK_PRESENT.value


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/reset")
def reset(difficulty: str = "easy"):
    state = env.reset(difficulty)
    return state


@app.post("/step")
def step(action: int):
    _, reward, done, info = env.step(action)
    return {
        "reward": reward,
        "done": done,
        "info": info
    }
