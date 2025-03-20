from fastapi import FastAPI, HTTPException, BackgroundTasks
from utils.imageProcess import loadKnownImages, loadAllImagesFromDrive, classifyAllImages, cache_manager
from pydantic import BaseModel, HttpUrl
import uvicorn
from typing import Optional, List

class ClassificationBody(BaseModel):
    image_url: HttpUrl
    drive_url: HttpUrl
    similarity_threshold: Optional[float] = 0.85

app = FastAPI(title="Optimized Image Classification API")

@app.get("/")
def status():
    return {
        "status": "ok",
        "info": "Optimized Image Classification API using EfficientNet-B0",
        "features": [
            "Batch processing",
            "Async downloads",
            "Result caching",
            "GPU acceleration (if available)"
        ],
        "endpoints": {
            "/ai/classify": "POST - Classify images using reference image",
            "/cache/clear": "POST - Clear cached results"
        }
    }

@app.post("/ai/classify")
async def classify(body: ClassificationBody, background_tasks: BackgroundTasks):
    try:
        known_image = loadKnownImages(str(body.image_url))
        if known_image[1]:
            images = loadAllImagesFromDrive(str(body.drive_url))
            if not images:
                raise HTTPException(status_code=404, detail="No images found in Drive folder")
            
            resultImagesList = classifyAllImages(images, known_image[0])
            # Clean up temp files in background
            background_tasks.add_task(lambda: None)  # Placeholder for cleanup
            
            return {
                "result": resultImagesList,
                "total_processed": len(images),
                "matches_found": len(resultImagesList),
                "cached_reference": known_image[0] is not None
            }
        else:
            raise HTTPException(status_code=400, detail="Could not process reference image")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cache/clear")
def clear_cache():
    """Clear all cached results"""
    try:
        cache_manager.clear()
        return {"status": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
