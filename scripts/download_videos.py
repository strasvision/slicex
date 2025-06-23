#!/usr/bin/env python3
"""
Instagram Video Downloader Script

This script downloads videos from Instagram URLs listed in a CSV file and saves them
to a local directory. It includes error handling and rate limiting.

Usage:
    python download_videos.py --input "path/to/this videoUrl.csv" --output videos/
"""

import os
import csv
import re
import json
import time
import random
import argparse
from datetime import datetime
from urllib.parse import urlparse, unquote
from typing import List, Dict, Optional, Tuple, Any
import requests
from requests.adapters import HTTPAdapter, Retry
import instaloader

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('video_download.log')
    ]
)
logger = logging.getLogger(__name__)

def sanitize_filename(url: str) -> str:
    """Create a safe filename from URL."""
    # Extract the last part of the URL as filename
    path = urlparse(url).path.strip('/')
    filename = os.path.basename(path.split('?')[0])  # Remove query parameters
    
    # If no extension or too short, use a hash of the URL
    if not filename or len(filename) < 5:
        import hashlib
        return f"video_{hashlib.md5(url.encode()).hexdigest()[:8]}.mp4"
    
    # Ensure the filename has an extension
    if '.' not in filename[-5:]:
        filename += '.mp4'
        
    return filename

def download_instagram_video(
    shortcode: str,
    output_dir: str,
    max_retries: int = 3,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Download a video from Instagram using instaloader.
    
    Args:
        shortcode: The Instagram post shortcode (from URL like instagram.com/p/SHORTCODE/)
        output_dir: Directory to save the downloaded video
        max_retries: Maximum number of retry attempts
        timeout: Timeout in seconds for each request
        
    Returns:
        Dict containing download result and metadata
    """
    result = {
        'shortcode': shortcode,
        'status': 'failed',
        'filename': None,
        'error': None,
        'attempts': 0
    }
    
    # Initialize Instaloader with rate limiting
    L = instaloader.Instaloader(
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        max_connection_attempts=1,
        request_timeout=timeout,
        rate_controller=lambda x: time.sleep(request_delay)
    )
    
    # Try to load session or login if credentials provided
    session_loaded = False
    if os.path.exists(session_file):
        try:
            L.load_session_from_file(session_file)
            logger.info(f"Loaded session from {session_file}")
            session_loaded = True
        except Exception as e:
            logger.warning(f"Failed to load session from {session_file}: {e}")
    
    if not session_loaded and username:
        try:
            if not password:
                import getpass
                password = getpass.getpass(f"Enter Instagram password for {username}: ")
            
            logger.info(f"Logging in as {username}...")
            L.login(username, password)
            L.save_session_to_file(session_file)
            logger.info("Successfully logged in and saved session")
            session_loaded = True
        except Exception as e:
            logger.error(f"Failed to login: {e}")
            if not input("Continue without login? (y/n): ").lower().startswith('y'):
                logger.info("Exiting...")
                return []
    
    if not session_loaded:
        logger.warning("No active Instagram session. Some content may not be accessible.")
    
    for attempt in range(1, max_retries + 1):
        result['attempts'] = attempt
        try:
            logger.info(f"Downloading Instagram post {shortcode} (attempt {attempt}/{max_retries})")
            
            # Get the post
            post = instaloader.Post.from_shortcode(L.context, shortcode)
            
            # Check if the post has a video
            if not post.is_video:
                result['error'] = "Post does not contain a video"
                logger.warning(f"Post {shortcode} does not contain a video")
                return result
            
            # Create output filename
            filename = f"{post.owner_username}_{post.date_utc.strftime('%Y%m%d_%H%M%S')}.mp4"
            filename = "".join(c if c.isalnum() or c in '._-' else '_' for c in filename)
            output_path = os.path.join(output_dir, filename)
            
            # Download the video
            L.download_post(post, target=output_dir)
            
            # Find the downloaded video file
            downloaded_files = [f for f in os.listdir(output_dir) if f.startswith(f"{post.owner_username}_{post.date_utc.strftime('%Y%m%d')}") and f.endswith('.mp4')]
            
            if not downloaded_files:
                raise Exception("Video file not found after download")
            
            # Update result with success
            result.update({
                'status': 'success',
                'filename': downloaded_files[0],
                'url': f"https://www.instagram.com/p/{shortcode}/"
            })
            
            logger.info(f"Successfully downloaded to {downloaded_files[0]}")
            return result
            
        except instaloader.exceptions.InstaloaderException as e:
            error_msg = str(e)
            result['error'] = error_msg
            logger.error(f"Instaloader error: {error_msg}")
            
            if "login_required" in error_msg.lower():
                logger.error("Login required to access this content. Please log in using 'instaloader --login'")
                break
                
        except Exception as e:
            error_msg = str(e)
            result['error'] = error_msg
            logger.error(f"Error downloading {shortcode}: {error_msg}")
        
        if attempt < max_retries:
            wait_time = 2 ** attempt + random.uniform(0, 1)  # Exponential backoff with jitter
            logger.info(f"Waiting {wait_time:.1f}s before retry...")
            time.sleep(wait_time)
    
    logger.error(f"Failed to download {shortcode} after {max_retries} attempts")
    return result

def extract_instagram_shortcode(url: str) -> Optional[str]:
    """Extract Instagram post shortcode from URL or return None if not found."""
    if not url or not isinstance(url, str):
        return None
        
    # Handle different Instagram URL formats
    patterns = [
        r'instagram\.com/p/([^/?#&]+)',  # Standard post URL
        r'instagram\.com/reel/([^/?#&]+)',  # Reel URL
        r'instagram\.com/tv/([^/?#&]+)',  # IG TV URL
        r'instagr\.am/p/([^/?#&]+)',  # Short URL format
        r'([a-zA-Z0-9_-]{10,})'  # Just the shortcode itself
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None

def process_instagram_urls(
    csv_path: str, 
    output_dir: str, 
    url_column: str = 'videoUrl',
    limit: Optional[int] = None,
    skip_existing: bool = True,
    username: Optional[str] = None,
    password: Optional[str] = None,
    session_file: str = 'instagram_session',
    request_delay: float = 10.0,
    max_retries: int = 3
) -> List[Dict[str, Any]]:
    """
    Process a CSV file containing Instagram URLs and download the videos.
    
    Args:
        csv_path: Path to the CSV file
        output_dir: Directory to save downloaded videos
        url_column: Name of the column containing Instagram URLs
        limit: Maximum number of videos to download
        skip_existing: Skip videos that already exist in the output directory
        
    Returns:
        List of download results with metadata
    """
    results = []
    os.makedirs(output_dir, exist_ok=True)
    
    # Track downloaded files to avoid duplicates
    downloaded_shortcodes = set()
    
    # First, try to read the file as a CSV with proper dialect detection
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
            # Try to detect the dialect (e.g., delimiter, quotechar)
            sample = f.read(1024)
            f.seek(0)
            
            # Use csv.Sniffer to detect the dialect
            try:
                dialect = csv.Sniffer().sniff(sample)
                reader = csv.DictReader(f, dialect=dialect)
                
                # Process each row
                for i, row in enumerate(reader):
                    if limit and len(results) >= limit:
                        logger.info(f"Reached download limit of {limit} videos")
                        break
                        
                    # Try to find Instagram post URLs in any column
                    url = None
                    processed_urls = set()  # Track processed URLs to avoid duplicates
                    
                    # First, try to find a URL in any column that looks like an Instagram post
                    for col, value in row.items():
                        if not value or not isinstance(value, str):
                            continue
                            
                        # Look for Instagram post URLs in the cell
                        urls = re.findall(r'https?://[^\s"]+', value)
                        for potential_url in urls:
                            if any(domain in potential_url for domain in ['instagram.com/p/', 'instagram.com/reel/', 'instagram.com/tv/']):
                                shortcode = extract_instagram_shortcode(potential_url)
                                if shortcode and shortcode not in processed_urls:
                                    url = f"https://www.instagram.com/p/{shortcode}/"
                                    processed_urls.add(shortcode)
                                    logger.info(f"Found Instagram URL in row {i+1}, column '{col}': {url}")
                                    break
                        if url:
                            break
                    
                    # If no post URL found, look for any Instagram URL that might contain video content
                    if not url:
                        for col, value in row.items():
                            if not value or not isinstance(value, str):
                                continue
                                
                            urls = re.findall(r'https?://[^\s"]+', value)
                            for potential_url in urls:
                                if 'instagram.com' in potential_url or 'instagr.am' in potential_url:
                                    shortcode = extract_instagram_shortcode(potential_url)
                                    if shortcode and shortcode not in processed_urls:
                                        url = f"https://www.instagram.com/p/{shortcode}/"
                                        processed_urls.add(shortcode)
                                        logger.info(f"Found potential Instagram URL in row {i+1}, column '{col}': {url}")
                                        break
                            if url:
                                break
                    
                    # If still no URL found, look for shortcodes that might be in the text
                    if not url:
                        for col, value in row.items():
                            if not value or not isinstance(value, str):
                                continue
                                
                            # Look for shortcode patterns (11-12 alphanumeric chars, typically)
                            shortcode_match = re.search(r'[a-zA-Z0-9_-]{10,20}', value)
                            if shortcode_match:
                                shortcode = shortcode_match.group(0)
                                # Additional validation for shortcodes
                                if (shortcode not in processed_urls and 
                                    10 <= len(shortcode) <= 30 and  # Reasonable length for shortcodes
                                    not any(word in shortcode.lower() for word in ['instagram', 'http', 'www', 'com', 'reel', 'tv', 'p'])):  # Exclude common false positives
                                    
                                    url = f"https://www.instagram.com/p/{shortcode}/"
                                    logger.info(f"Found potential shortcode in row {i+1}, column '{col}': {url}")
                                    processed_urls.add(shortcode)
                                    break
                    
                    if not url:
                        # Log the first few columns to help with debugging
                        sample_cols = {k: v for i, (k, v) in enumerate(row.items()) if i < 3}
                        logger.warning(f"No Instagram URL found in row {i+1}. Sample columns: {sample_cols}")
                        results.append({
                            'row': i+1,
                            'status': 'skipped',
                            'reason': 'no_url',
                            'url': None,
                            'sample_columns': sample_cols
                        })
                        continue
                    
                    # Extract Instagram shortcode
                    try:
                        shortcode = extract_instagram_shortcode(url)
                        if not shortcode:
                            raise ValueError("Could not extract shortcode from URL")
                            
                        # Skip if we've already processed this shortcode
                        if shortcode in downloaded_shortcodes:
                            logger.info(f"Skipping duplicate shortcode: {shortcode}")
                            results.append({
                                'row': i+1,
                                'status': 'skipped',
                                'reason': 'duplicate',
                                'shortcode': shortcode,
                                'url': url
                            })
                            continue
                        
                        # Check if we already have this video downloaded
                        if skip_existing:
                            existing_files = [f for f in os.listdir(output_dir) 
                                           if f.endswith('.mp4') and f'_{shortcode}.' in f]
                            if existing_files:
                                logger.info(f"Skipping existing video for shortcode: {shortcode}")
                                results.append({
                                    'row': i+1,
                                    'status': 'skipped',
                                    'reason': 'file_exists',
                                    'shortcode': shortcode,
                                    'filename': existing_files[0],
                                    'url': url
                                })
                                downloaded_shortcodes.add(shortcode)
                                continue
                        
                        # Download the video
                        logger.info(f"Processing {len(results) + 1}/{limit if limit else 'all'}: {shortcode}")
                        result = download_instagram_video(shortcode, output_dir)
                        
                        # Add metadata
                        result.update({
                            'row': i+1,
                            'timestamp': datetime.now().isoformat(),
                            'url': url
                        })
                        
                        if result['status'] == 'success':
                            downloaded_shortcodes.add(shortcode)
                        
                        results.append(result)
                        
                        # Add a small delay between downloads to avoid rate limiting
                        time.sleep(2)
                        
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"Error processing URL {url}: {error_msg}")
                        results.append({
                            'row': i+1,
                            'status': 'failed',
                            'url': url,
                            'error': error_msg,
                            'timestamp': datetime.now().isoformat()
                        })
                    
            except csv.Error as e:
                logger.error(f"CSV parsing error on row {i+1}: {e}")
                if row:  # Only try to log row content if it exists
                    logger.error(f"Row content: {row}")
                # Don't use continue here as it's not in a loop context
                # Just log the error and let the loop continue naturally
                
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        logger.error("Please check the CSV file format and try again.")
        return []  # Return empty list instead of raising to allow graceful exit
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Download Instagram videos from a CSV file containing post URLs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input CSV file path containing Instagram post URLs'
    )
    parser.add_argument(
        '--output', '-o',
        default='../data/raw_videos',
        help='Output directory for downloaded videos'
    )
    parser.add_argument(
        '--url-column',
        default='videoUrl',
        help='Name of the column containing Instagram post URLs'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=5,
        help='Maximum number of videos to download (set to 0 for no limit)'
    )
    parser.add_argument(
        '--force',
        action='store_false',
        dest='skip_existing',
        help='Overwrite existing files'
    )
    parser.add_argument(
        '--session',
        help='Path to Instagram session file (created with instabot --login)'
    )
    
    args = parser.parse_args()
    
    # Validate limit
    if args.limit == 0:
        args.limit = None
    
    # Set up logging level
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Process the CSV and download videos
    try:
        results = process_instagram_urls(
            csv_path=args.input,
            output_dir=args.output,
            url_column=args.url_column,
            limit=args.limit,
            skip_existing=not args.force,
            username=args.username,
            password=args.password,
            session_file=args.session_file,
            request_delay=args.request_delay,
            max_retries=args.max_retries
        )
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=args.debug)
        return 1
        
    return 0
    
    # Print summary
    success = sum(1 for r in results if r.get('status') == 'success')
    skipped = sum(1 for r in results if r.get('status') in ['skipped', 'file_exists', 'duplicate'])
    failed = len(results) - success - skipped
    
    print(f"\n{'='*50}")
    print(f"Download Summary:")
    print(f"{'='*50}")
    print(f"Total processed: {len(results)}")
    print(f"✅ Success: {success}")
    print(f"⏩ Skipped: {skipped} (already downloaded or duplicate)")
    print(f"❌ Failed: {failed}")
    
    # Save detailed results to a JSON file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'download_results_{timestamp}.json'
    
    # Convert results to a serializable format
    serializable_results = []
    for result in results:
        # Create a new dict with only serializable values
        serialized = {}
        for k, v in result.items():
            try:
                json.dumps({k: v})  # Test if value is JSON serializable
                serialized[k] = v
            except (TypeError, OverflowError):
                serialized[k] = str(v)  # Convert non-serializable values to strings
        serializable_results.append(serialized)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {os.path.abspath(results_file)}")
    
    # Print failed downloads for easy reference
    failed_downloads = [r for r in results if r.get('status') == 'failed']
    if failed_downloads:
        print(f"\n❌ Failed Downloads ({len(failed_downloads)}):")
        for i, item in enumerate(failed_downloads, 1):
            print(f"{i}. Row {item.get('row', '?')}: {item.get('url', '')}")
            if 'shortcode' in item:
                print(f"   Shortcode: {item['shortcode']}")
            if 'error' in item:
                print(f"   Error: {item['error']}")
    
    # Print successful downloads
    if success > 0:
        print(f"\n✅ Successfully Downloaded ({success}):")
        for item in [r for r in results if r.get('status') == 'success']:
            print(f"- {item.get('filename')} (from {item.get('url')})")
    
    # Add a final newline for better readability
    print()

if __name__ == "__main__":
    main()
