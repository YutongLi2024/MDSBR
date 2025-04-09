import os
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv

asin2item_df = pd.read_csv('asin2itemID.csv')
item_final_df = pd.read_csv('item_final.csv')

asin_list = np.load('asinlist.npy', allow_pickle=True)

filtered_item_final_df = item_final_df[item_final_df['asin'].isin(asin_list)]

asin_to_url = pd.Series(filtered_item_final_df.image.values, index=filtered_item_final_df.asin).to_dict()

image_path = 'Clothes_images/'

if not os.path.exists(image_path):
    os.makedirs(image_path)

print(f"Total ASINs in asin_list: {len(asin_list)}")
print(f"Total ASINs in filtered_item_final_df: {len(filtered_item_final_df)}")

if len(asin_to_url) != len(filtered_item_final_df):
    print("Warning: There are duplicate ASINs in the dataset.")

failed_asins = []

def download_image(asin, url):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            file_path = os.path.join(image_path, f"{asin}.jpg")
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
        else:
            print(f"Failed to download image for ASIN: {asin}, URL: {url}")
            return asin
    except Exception as e:
        print(f"Error downloading image for ASIN: {asin}, URL: {url}, Error: {e}")
        return asin
    return None

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(download_image, asin, url): asin for asin, url in asin_to_url.items()}

    for future in tqdm(as_completed(futures), total=len(futures)):
        asin = futures[future]
        if future.result():
            failed_asins.append(asin)

print (failed_asins)
with open("failed_asins.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Failed ASINs"])  
    for asin in failed_asins:
        writer.writerow([asin])

print("Failed ASINs have been written to 'failed_asins.csv'.")



downloaded_images = len([name for name in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, name))])
print(f"Total images downloaded successfully: {downloaded_images}")
print(f"Total failed downloads: {len(failed_asins)}")


