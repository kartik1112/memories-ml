from PIL import Image
import os

def create_test_images():
    """Create test images for API testing"""
    os.makedirs("test_data", exist_ok=True)
    
    # Create reference image
    ref_img = Image.new('RGB', (224, 224), color='blue')
    ref_img.save("test_data/reference.jpg")
    
    # Create similar images
    for i in range(3):
        # Slightly different blue
        color = (0, 0, 200 + i * 20)
        img = Image.new('RGB', (224, 224), color=color)
        img.save(f"test_data/similar_{i}.jpg")
    
    # Create different images
    colors = ['red', 'green', 'yellow']
    for i, color in enumerate(colors):
        img = Image.new('RGB', (224, 224), color=color)
        img.save(f"test_data/different_{i}.jpg")

if __name__ == "__main__":
    create_test_images()
