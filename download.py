import gdown


def download_from_drive():
    url = 'https://drive.google.com/uc?id=1aLwstGpuovhwLKd4QtPN4KEE6iXyrzk0'
    output = 'model_state.pth'
    gdown.download(url, output, quiet=False)
