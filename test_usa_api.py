from fastapi import FastAPI
import uvicorn

app = FastAPI(title='Test USA Agriculture API', version='1.0.0')

@app.get("/")
async def root():
    return {"message": "USA Agriculture API Test", "status": "working"}

@app.get("/test")
async def test():
    return {"test": "success"}

if __name__ == "__main__":
    print("Starting test USA Agriculture API...")
    uvicorn.run(app, host='0.0.0.0', port=8003, log_level="info")