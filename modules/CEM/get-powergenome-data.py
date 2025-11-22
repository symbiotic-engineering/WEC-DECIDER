#!/usr/bin/env python3
"""
Python script to download PowerGenome data using the gdown package.
This script must be run from the WEC-DECIDER/modules/CEM folder.
"""

import os
import zipfile
import gdown
import urllib.request
import time
from pathlib import Path


def retry_gdown_download(file_id, output_path, max_retries=3, delay=5):
    """
    Retry gdown download with exponential backoff on FileURLRetrievalError.
    
    Args:
        file_id: Google Drive file ID
        output_path: Path to save the file
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
    
    Returns:
        bool: True if successful, False otherwise
    """
    for attempt in range(max_retries + 1):
        try:
            gdown.download(id=file_id, output=str(output_path), quiet=True)
            return True
        except gdown.exceptions.FileURLRetrievalError as e:
            if attempt < max_retries:
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                print(f"    Download failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                print(f"    Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"    Download failed after {max_retries + 1} attempts: {e}")
                return False
        except Exception as e:
            print(f"    Unexpected error during download: {e}")
            return False
    return False


def retry_gdown_folder_download(folder_id, output_path, max_retries=3, delay=5):
    """
    Retry gdown folder download with exponential backoff on FileURLRetrievalError.
    
    Args:
        folder_id: Google Drive folder ID
        output_path: Path to save the folder
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
    
    Returns:
        bool: True if successful, False otherwise
    """
    for attempt in range(max_retries + 1):
        try:
            gdown.download_folder(id=folder_id, output=str(output_path), quiet=True)
            return True
        except gdown.exceptions.FileURLRetrievalError as e:
            if attempt < max_retries:
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                print(f"    Folder download failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                print(f"    Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"    Folder download failed after {max_retries + 1} attempts: {e}")
                return False
        except Exception as e:
            print(f"    Unexpected error during folder download: {e}")
            return False
    return False


def main():
    # Get script directory (equivalent to $script_dir in bash)
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    # Create data directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)
    
    print("Starting PowerGenome data download...")
    
    # Download and extract zip files
    zip_downloads = [
        {
            "id": "1nbhWwOsNeOtcUew9Mn4QGuAtCsZo0VZ2",
            "name": "cambium",
            "output_dir": data_dir / "cambium"
        },
        {
            "id": "1dWA35bQpPksnSb6auybMbrIqyaBG6wBM", 
            "name": "efs",
            "output_dir": data_dir / "efs"
        },
        {
            "id": "1AT7vsfxLsKuf9N2JXBTlrt2-4I8Rg_hI",
            "name": "pg", 
            "output_dir": data_dir / "pg"
        },
        {
            "id": "1tJipxJYxP_dcAnopJrdXdcZh7K3SlI1-",
            "name": "pudl",
            "output_dir": data_dir / "pudl"
        }
    ]
    
    for download in zip_downloads:
        output_dir = download["output_dir"]
        if not output_dir.exists():
            print(f"Downloading and extracting {download['name']}...")
            zip_path = data_dir / f"{download['name']}.zip"
            
            # Download zip file with retry logic
            if retry_gdown_download(download["id"], zip_path):
                # Extract zip file
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(output_dir)
                    
                    # Remove zip file after extraction
                    zip_path.unlink()
                    print(f"Completed {download['name']}")
                except zipfile.BadZipFile as e:
                    print(f"Error extracting {download['name']}: {e}")
                    # Clean up bad zip file
                    if zip_path.exists():
                        zip_path.unlink()
            else:
                print(f"Failed to download {download['name']} after all retry attempts")
        else:
            print(f"Directory {download['name']} already exists, skipping...")
    
    # Download folders (not zips)
    folder_downloads = [
        {
            "id": "1ZYxnl4U_3HXlYPxm8qlmqyWB8NyC3PpG",
            "name": "resource_profiles",
            "output_dir": data_dir / "resource_profiles"
        },
        {
            "id": "1Svkz6fKgc1m9ewUMPjVHJJV5TWDYKdmw",
            "name": "resource_groups", 
            "output_dir": data_dir / "resource_groups"
        },
        {
            "id": "16bnl3VSUMP8UNEhA881VGpFqCkmeadcm",
            "name": "network_costs",
            "output_dir": data_dir / "network_costs"
        },
        {
            "id": "1dQt1Drk8wkWU-T3BO8zUlg4yUf1euYJx",
            "name": "extra_inputs",
            "output_dir": data_dir / "extra_inputs"
        }
    ]
    
    for download in folder_downloads:
        output_dir = download["output_dir"]
        if not output_dir.exists():
            print(f"Downloading folder {download['name']}...")
            if retry_gdown_folder_download(download["id"], output_dir):
                print(f"Completed {download['name']}")
            else:
                print(f"Failed to download folder {download['name']} after all retry attempts")
        else:
            print(f"Directory {download['name']} already exists, skipping...")
    
    # Download files from PowerGenome GitHub repository using wget equivalent
    # Create data_east directory for ISONE files
    data_east_dir = script_dir / "data_east"
    data_east_dir.mkdir(exist_ok=True)
    
    isone_urls = [
        "https://raw.githubusercontent.com/PowerGenome/PowerGenome/refs/heads/master/example_systems/ISONE/extra_inputs/misc_gen_inputs.csv",
        "https://raw.githubusercontent.com/PowerGenome/PowerGenome/refs/heads/master/example_systems/ISONE/extra_inputs/demand_segments_voll.csv", 
        "https://raw.githubusercontent.com/PowerGenome/PowerGenome/refs/heads/master/example_systems/ISONE/extra_inputs/resource_capacity_spur.csv",
        "https://raw.githubusercontent.com/PowerGenome/PowerGenome/refs/heads/master/example_systems/ISONE/extra_inputs/emission_policies.csv",
        "https://raw.githubusercontent.com/PowerGenome/PowerGenome/refs/heads/master/example_systems/ISONE/extra_inputs/Reserves.csv"
    ]
    
    print("Downloading ISONE files...")
    for url in isone_urls:
        filename = url.split('/')[-1]
        output_path = data_east_dir / filename
        
        # Only download if file doesn't exist (equivalent to wget -nc)
        if not output_path.exists():
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(url, output_path)
        else:
            print(f"  File {filename} already exists, skipping...")
    
    # Create data_CA directory for CA_AZ files  
    data_ca_dir = script_dir / "data_CA"
    data_ca_dir.mkdir(exist_ok=True)
    
    ca_az_urls = [
        "https://raw.githubusercontent.com/PowerGenome/PowerGenome/refs/heads/master/example_systems/CA_AZ/extra_inputs/test_demand_segments_voll.csv",
        "https://raw.githubusercontent.com/PowerGenome/PowerGenome/refs/heads/master/example_systems/CA_AZ/extra_inputs/test_misc_gen_inputs.csv"
    ]
    
    print("Downloading CA_AZ files...")
    for url in ca_az_urls:
        filename = url.split('/')[-1] 
        output_path = data_ca_dir / filename
        
        # Only download if file doesn't exist (equivalent to wget -nc)
        if not output_path.exists():
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(url, output_path)
        else:
            print(f"  File {filename} already exists, skipping...")
    
    print("All downloads completed successfully!")


if __name__ == "__main__":
    main()
