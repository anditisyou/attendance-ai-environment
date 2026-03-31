"""
FastAPI Deployment with Interactive Dashboard (Optimized for Fast Startup)
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from enhanced_inference import comprehensive_evaluation
from visualization import AttendanceDashboard

app = FastAPI(
    title="Attendance Validation Environment - Hackathon Edition",
    description="OpenEnv Hackathon | Scaler x Meta",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 🚀 BACKGROUND STARTUP TASK
# -------------------------------
@app.on_event("startup")
async def startup_event():
    print("⚡ Running evaluation in background...")
    try:
        # Reduced episodes for fast startup
        app.cached_results, _ = comprehensive_evaluation(num_episodes_per_difficulty=3)
        print("✅ Evaluation ready")
    except Exception as e:
        print("❌ Error during evaluation:", e)
        app.cached_results = {}


# -------------------------------
# 🌐 MAIN ROUTE
# -------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    """Main dashboard"""

    # If results not ready → show loading screen
    if not hasattr(app, "cached_results") or not app.cached_results:
        return HTMLResponse("""
        <html>
        <head>
            <title>Loading AI Environment</title>
        </head>
        <body style="
            background:#0f172a;
            color:white;
            font-family:sans-serif;
            text-align:center;
            padding-top:120px;">
            
            <h1>🚀 Initializing AI Environment...</h1>
            <p>Running evaluation and preparing dashboard</p>
            <p>⏳ Please wait a few seconds and refresh</p>
            
        </body>
        </html>
        """)

    # Generate dashboard
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
# 📊 LEADERBOARD API
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
            "score": round(overall_score, 2),
            "easy_accuracy": round(agent_results["easy"]["accuracy"] * 100, 1),
            "medium_accuracy": round(agent_results["medium"]["accuracy"] * 100, 1),
            "hard_accuracy": round(agent_results["hard"]["accuracy"] * 100, 1),
            "fraud_detection": round(agent_results["hard"]["fraud_detection_rate"] * 100, 1)
        })

    return sorted(leaderboard, key=lambda x: x["score"], reverse=True)


# -------------------------------
# 📄 FULL REPORT API
# -------------------------------
@app.get("/api/report")
async def get_report():

    if not hasattr(app, "cached_results") or not app.cached_results:
        return {"status": "loading"}

    return app.cached_results


# -------------------------------
# 🏁 RUN SERVER
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)