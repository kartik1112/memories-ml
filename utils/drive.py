import gdown


def getAllImagesListFromDrive(url:str):
    ok = gdown.download_folder(url, quiet=True, use_cookies=False, skip_download=True)
    return ok

def downloadImageFromDrive(url:str):
    image = gdown.download(url, quiet=True, use_cookies=False)
    return image    