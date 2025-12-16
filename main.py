from fastapi import FastAPI

app = FastAPI(title="FastAPI on Render")

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "FastAPI",
        "platform": "Render"
    }

@app.get("/health")
def health():
    return {"healthy": True}
