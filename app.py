"""
FastAPI Deployment with OpenEnv + Dashboard (FINAL FIXED)
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from enhanced_inference import comprehensive_evaluation
from visualization import AttendanceDashboard
from attendance_env import AttendanceEnv

app = FastAPI(
    title="Attendance Validation Environment - Hackathon Edition",
    description="OpenEnv Hackathon | Scaler x Meta",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 🔥 OpenEnv Environment Instance
# -------------------------------
env = AttendanceEnv()

# -------------------------------
# 🚀 BACKGROUND STARTUP TASK
# -------------------------------
@app.on_event("startup")
async def startup_event():
    print("⚡ Running evaluation in background...")
    try:
        app.cached_results, _ = comprehensive_evaluation(num_episodes_per_difficulty=3)
        print("✅ Evaluation ready")
    except Exception as e:
        print("❌ Error:", e)
        app.cached_results = {}

# -------------------------------
# 🧠 REQUIRED OPENENV ENDPOINTS
# -------------------------------

@app.post("/reset")
async def reset_env():
    state = env.reset()
    return {"state": state}


@app.post("/step")
async def step_env(action: dict):
    act = action.get("action")

    state, reward, done, info = env.step(act)

    return {
        "state": state,
        "reward": reward,
        "done": done,
        "info": info
    }

# -------------------------------
# 🌐 DASHBOARD ROUTE
# -------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():

    if not hasattr(app, "cached_results") or not app.cached_results:
        return HTMLResponse("""
        <html>
        <body style="background:#0f172a; color:white; text-align:center; padding-top:120px;">
            <h1>🚀 Initializing AI Environment...</h1>
            <p>Please wait a few seconds and refresh.</p>
        </body>
        </html>
        """)

    dashboard = AttendanceDashboard()

    agent_results = app.cached_results.get("Basic Stochastic", {})

    dashboard_results = {
        "easy": agent_results.get("easy", {}),
        "medium": agent_results.get("medium", {}),
        "hard": agent_results.get("hard", {})
    }

    html_report = dashboard.generate_html_report(
        dashboard_results,
        "Enhanced Stochastic Agent"
    )

    return HTMLResponse(content=html_report)

# -------------------------------
# 📊 LEADERBOARD
# -------------------------------
@app.get("/api/leaderboard")
async def get_leaderboard():

    if not hasattr(app, "cached_results") or not app.cached_results:
        return {"status": "loading"}

    leaderboard = []

    for agent_name, agent_results in app.cached_results.items():
        overall_score = (
            agent_results["easy"]["accuracy"] * 0.2 +
            agent_results["medium"]["accuracy"] * 0.3 +
            agent_results["hard"]["accuracy"] * 0.3 +
            agent_results["hard"]["fraud_detection_rate"] * 0.2
        ) * 100

        leaderboard.append({
            "agent": agent_name,
            "score": round(overall_score, 2)
        })

    return sorted(leaderboard, key=lambda x: x["score"], reverse=True)

# -------------------------------
# 📄 REPORT
# -------------------------------
@app.get("/api/report")
async def get_report():

    if not hasattr(app, "cached_results") or not app.cached_results:
        return {"status": "loading"}

    return app.cached_results

# -------------------------------
# 🏁 RUN
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
