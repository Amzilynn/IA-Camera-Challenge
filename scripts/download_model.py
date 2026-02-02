import urllib.request
import os
import sys

URL = "https://github.com/WildChlamydia/MiVOLO/releases/download/v1.0/model_imdb_cross_person_4.24_99.46.pth.tar"
DEST = "models/mivolo_imbd.pth.tar"

def download_file():
    print(f"Downloading {URL} to {DEST}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    req = urllib.request.Request(URL, headers=headers)
    
    try:
        with urllib.request.urlopen(req) as response:
            with open(DEST, 'wb') as f:
                f.write(response.read())
        print(f"\nSuccessfully downloaded to {DEST} ({os.path.getsize(DEST)} bytes)")
    except Exception as e:
        print(f"\nError downloading file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_file()
