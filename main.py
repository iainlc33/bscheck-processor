def extract_tiktok_video(url):
    """Download TikTok video using apidojo/tiktok-scraper"""
    try:
        logger.info(f"Trying to download video from {url}")
        file_id = str(int(time.time()))
        output_path = f"{TEMP_DIR}/{file_id}.mp4"
        
        # Format based on apidojo's official documentation
        # Using the most basic format possible
        run_input = {
            "search": url,  # Try using search parameter directly
            "resultsPerPage": 1,
            "shouldDownloadVideos": True,
            "proxy": {
                "useApifyProxy": True
            }
        }
        
        # Make API request to Apify
        run_url = "https://api.apify.com/v2/acts/apidojo~tiktok-scraper/runs"
        
        headers = {
            "Authorization": f"Bearer {APIFY_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Start the scraper run
        response = requests.post(run_url, headers=headers, json={"runInput": run_input})
        
        if response.status_code != 201:
            logger.error(f"Apify run creation failed: {response.text}")
            raise Exception(f"Failed to start Apify scraper: {response.status_code}")
        
        run_data = response.json()
        run_id = run_data["data"]["id"]
        
        # Log the run URL for manual checking
        logger.info(f"Apify run created: https://console.apify.com/view/runs/{run_id}")
        
        # Wait for the run to finish
        run_status_url = f"https://api.apify.com/v2/actor-runs/{run_id}"
        
        # Poll for completion (with timeout)
        start_time = time.time()
        timeout = 60  # seconds
        
        while time.time() - start_time < timeout:
            status_response = requests.get(run_status_url, headers={"Authorization": f"Bearer {APIFY_API_TOKEN}"})
            status_data = status_response.json()
            
            if status_data["data"]["status"] == "SUCCEEDED":
                break
                
            if status_data["data"]["status"] in ["FAILED", "ABORTED", "TIMED-OUT"]:
                logger.error(f"Apify run failed with status: {status_data['data']['status']}")
                logger.error(f"Status message: {status_data['data'].get('statusMessage', 'No status message')}")
                logger.error(f"Apify run details: {status_data}")
                raise Exception(f"Apify run failed with status: {status_data['data']['status']}")
                
            # Wait before checking again
            time.sleep(2)
        
        # Get the results from the dataset
        dataset_id = status_data["data"]["defaultDatasetId"]
        dataset_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
        
        results_response = requests.get(dataset_url, headers={"Authorization": f"Bearer {APIFY_API_TOKEN}"})
        results = results_response.json()
        
        if not results or len(results) == 0:
            raise Exception("No results found from Apify scraper")
        
        logger.info(f"Apify result structure: {results[0].keys()}")
        
        # Find the video URL (may be in different location based on this specific actor)
        video_item = results[0]
        
        # The video might be available directly or as a download URL
        video_url = None
        
        # Check possible locations based on apidojo's format
        if "videoUrl" in video_item:
            video_url = video_item["videoUrl"]
        elif "video" in video_item and "downloadAddr" in video_item["video"]:
            video_url = video_item["video"]["downloadAddr"]
        elif "downloadUrl" in video_item:
            video_url = video_item["downloadUrl"]
        elif "video" in video_item and "playAddr" in video_item["video"]:
            video_url = video_item["video"]["playAddr"]
        elif "url" in video_item:
            video_url = video_item["url"]
        
        if not video_url:
            # Try to find in the Key Store first
            key_value_store_id = status_data["data"]["defaultKeyValueStoreId"]
            store_url = f"https://api.apify.com/v2/key-value-stores/{key_value_store_id}/records/OUTPUT"
            
            store_response = requests.get(store_url, headers={"Authorization": f"Bearer {APIFY_API_TOKEN}"})
            
            if store_response.status_code == 200:
                store_data = store_response.json()
                if "videoUrl" in store_data:
                    video_url = store_data["videoUrl"]
            
            if not video_url:
                # Log the full response to help debug the structure
                logger.error(f"Could not find video URL in response. Response structure: {results[0]}")
                raise Exception("No video URL found in Apify results")
        
        # Download the video
        logger.info(f"Downloading video from URL: {video_url}")
        video_response = requests.get(video_url, stream=True)
        
        if video_response.status_code != 200:
            raise Exception(f"Failed to download video: Status code {video_response.status_code}")
            
        with open(output_path, 'wb') as f:
            for chunk in video_response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        
        # Verify file was downloaded
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Video downloaded successfully to {output_path}")
            return output_path
        else:
            raise Exception("Video file is empty or doesn't exist after download")
        
    except Exception as e:
        logger.error(f"Error extracting TikTok: {str(e)}")
        return None
