import torch
import timm
import numpy as np
from PIL import Image
import requests
import aiohttp
import asyncio
from torchvision import transforms
from utils.drive import getAllImagesListFromDrive, downloadImageFromDrive
from utils.cache_manager import CacheManager
import os
from tqdm import tqdm

# Initialize model and transform globally for reuse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
model = model.to(device)
model.eval()

# Initialize cache manager
cache_manager = CacheManager()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(image_tensor)
        return features.cpu().numpy()[0], True
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None, False

def extract_features_batch(image_paths, batch_size=16):
    """Process images in batches for better performance"""
    all_features = []
    all_success = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []
        
        for path in batch_paths:
            try:
                image = Image.open(path).convert('RGB')
                tensor = transform(image)
                batch_tensors.append(tensor)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        
        if not batch_tensors:
            continue
            
        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            features = model(batch)
        
        features = features.cpu().numpy()
        all_features.extend(features)
        all_success.extend([True] * len(batch_tensors))
    
    return all_features, all_success

def cosine_similarity(features1, features2):
    return np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))

async def download_image_async(session, url, filename):
    """Download image asynchronously"""
    try:
        async with session.get(url) as response:
            if response.status == 200:
                with open(filename, 'wb') as f:
                    f.write(await response.read())
                return filename
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return None

def loadKnownImages(url: str):
    """Load and cache reference image features"""
    # Check cache first
    cached_features = cache_manager.get_features(url)
    if cached_features is not None:
        return [cached_features, True]

    try:
        response = requests.get(url, stream=True)
        with open("known.jpg", 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        
        features, success = extract_features("known.jpg")
        if success:
            # Cache the features
            cache_manager.set_features(url, features)
        
        if os.path.exists("known.jpg"):
            os.remove("known.jpg")
        
        return [features, success]
    except Exception as e:
        if os.path.exists("known.jpg"):
            os.remove("known.jpg")
        return [str(e), False]

def loadAllImagesFromDrive(url: str):
    images = getAllImagesListFromDrive(url)
    if images is None:
        print("Error: no images found")
    return images

async def download_all_images_async(image_ids):
    """Download multiple images concurrently"""
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for img_id in image_ids:
            url = f"https://drive.google.com/uc?id={img_id.id}"
            filename = os.path.join(temp_dir, f"{img_id.id}.jpg")
            task = download_image_async(session, url, filename)
            tasks.append(task)
        
        downloaded_paths = []
        for result in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading images"):
            path = await result
            if path:
                downloaded_paths.append(path)
    
    return downloaded_paths

def process_images_batch(image_paths, known_features, similarity_threshold=0.85):
    """Process a batch of images and return matching ones"""
    features_list, success_list = extract_features_batch(image_paths)
    matching_indices = []
    
    for i, (features, success) in enumerate(zip(features_list, success_list)):
        if success and cosine_similarity(known_features, features) > similarity_threshold:
            matching_indices.append(i)
    
    return matching_indices

def classifyAllImages(images: list[str], known_features: any):
    """Classify images using batch processing and async downloads"""
    if not images:
        return []
    
    # Create event loop for async operations
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Download images asynchronously
    downloaded_paths = loop.run_until_complete(download_all_images_async(images))
    
    # Process images in batches
    matching_indices = process_images_batch(downloaded_paths, known_features)
    
    # Create result list
    resultImagesList = [
        f"https://drive.google.com/uc?id={images[i].id}"
        for i in matching_indices
    ]
    
    # Cleanup
    for path in downloaded_paths:
        try:
            os.remove(path)
        except:
            pass
    try:
        os.rmdir("temp_images")
    except:
        pass
    
    return resultImagesList
