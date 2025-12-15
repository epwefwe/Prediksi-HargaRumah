import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
import time
import os
import re

# --- Configurations ---
BASE_URL = "https://www.rumah123.com/jual/jakarta-selatan/rumah/"
OUTPUT_FILE = "data/raw/DATA RUMAH.xlsx"
HEADERS_LIST = [
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"},
     {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"}
]

def scrape_data(pages=1):
    all_data = []
    
    for page in range(1, pages + 1):
        url = f"{BASE_URL}?page={page}"
        print(f"Scraping page {page}: {url}")
        
        try:
            session = requests.Session()
            response = session.get(url, headers=random.choice(HEADERS_LIST), timeout=30)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                
                # Robust selector found via debugging
                listings = soup.select('div[data-test-id^="srp-listing-card-"]')
                print(f"Found {len(listings)} listings on page {page}")

                for item in listings:
                    try:
                        # Full text content of the card
                        full_text = item.get_text(" ", strip=True) 

                        # 1. Title (H2 is usually reliable)
                        title_tag = item.find("h2")
                        nama_rumah = title_tag.get_text(strip=True) if title_tag else "N/A"
                        
                        # 2. Price (Regex)
                        # Matches: Rp 7,5 Miliar, Rp 7.5 M, Rp 500 Juta, etc.
                        price_match = re.search(r'Rp\s*([\d\.,]+)\s+(Miliar|M|Juta|Jt)', full_text, re.IGNORECASE)
                        price = 0
                        if price_match:
                            val_str = price_match.group(1)
                            unit = price_match.group(2).lower()
                            val = float(val_str.replace(",", "."))
                            
                            if 'm' in unit:
                                price = int(val * 1_000_000_000)
                            elif 'j' in unit:
                                price = int(val * 1_000_000)
                        
                        # 3. Specs (Regex)
                        # LT: 60 m2, LB: 60 m2
                        lt, lb, kt, km = 0, 0, 0, 0
                        
                        # LT
                        lt_match = re.search(r'LT\s*:\s*(\d+)', full_text, re.IGNORECASE)
                        if lt_match: lt = int(lt_match.group(1))
                        
                        # LB
                        lb_match = re.search(r'LB\s*:\s*(\d+)', full_text, re.IGNORECASE)
                        if lb_match: lb = int(lb_match.group(1))
                        
                        # KT/KM (Regex Heuristic)
                        # Pattern found: "LT: 76 m² LB: 375 m² 3 3"
                        # We look for digits after LB and m2.
                        
                        # Find the substring after "LB"
                        if "LB" in full_text:
                            after_lb = full_text.split("LB")[1]
                            # Find all numbers in the remainder
                            specs_numbers = re.findall(r'\b(\d+)\b', after_lb)
                            # specs_numbers[0] is LB value (already parsed)
                            # specs_numbers[1] is likely KT
                            # specs_numbers[2] is likely KM
                            
                            if len(specs_numbers) >= 2:
                                # Skip the first one as it is the LB value
                                # Note: Sometimes 'm2' is adjacent, so ensure we aren't picking up '2' from m2.
                                # But re.findall(\b\d+\b) handles 'm2' as 'm' and '2'. 
                                # Better: use specific regex for sequence
                                pass

                        # Extraction Strategy 1: Look for specific SVG icons (Most Robust)
                        kt, km, grs = 0, 0, 0
                        
                        # Helper to extract text from span next to specific icon
                        def get_spec_by_icon(soup_item, icon_id):
                            try:
                                icon = soup_item.find("use", href=lambda x: x and icon_id in x)
                                if not icon:
                                    icon = soup_item.find("use", attrs={"xlink:href": lambda x: x and icon_id in x})
                                if icon:
                                    svg = icon.find_parent("svg")
                                    if svg:
                                        span = svg.find_parent("span")
                                        if span:
                                            txt = span.get_text(strip=True)
                                            # Clean non-digit chars if any
                                            clean_txt = re.sub(r'\D', '', txt)
                                            if clean_txt:
                                                return int(clean_txt)
                            except:
                                pass
                            return 0

                        kt = get_spec_by_icon(item, "#bedroom-icon")
                        km = get_spec_by_icon(item, "#bathroom-icon")
                        grs = get_spec_by_icon(item, "#carports-icon")

                        # Extraction Strategy 2: Regex on text (Fallback)
                        if kt == 0 or km == 0:
                            # Robust regex for "LB: X m² KT KM GRS?"
                            # matches "375 m² 3 3 1" or "375 m2 3 3"
                            specs_match = re.search(r'LB\s*:\s*\d+\s*m[²2]?\s+(\d+)\s+(\d+)(?:\s+(\d+))?', full_text, re.IGNORECASE)
                            if specs_match:
                                if kt == 0: kt = int(specs_match.group(1))
                                if km == 0: km = int(specs_match.group(2))
                                # Only split-second guess GRS if not found by icon
                                if grs == 0 and specs_match.group(3):
                                    grs = int(specs_match.group(3))
                            else:
                                # Fallback: Try to find ANY two small integers (1-9) close to end of string if above fails
                                loose_match = re.findall(r'\s(\d)\s+(\d)\s', full_text)
                                if loose_match:
                                    if kt == 0: kt = int(loose_match[0][0])
                                    if km == 0: km = int(loose_match[0][1])
                        
                        # Defaults
                        if kt == 0: kt = 2
                        if km == 0: km = 1
                        if grs == 0: grs = 1 # Default to 1 if genuinely not found (most houses have 1)
                        
                        if price > 0:
                            all_data.append({
                                "NAMA RUMAH": nama_rumah,
                                "HARGA": price,
                                "LB": lb if lb > 0 else 60,
                                "LT": lt if lt > 0 else 60,
                                "KT": kt,
                                "KM": km,
                                "GRS": grs
                            })
                            
                    except Exception as e:
                        print(f"Error parsing item: {e}")
                        continue
            else:
                print(f"Failed to fetch page {page}: {response.status_code}")
                
            time.sleep(random.uniform(2, 5))
            
        except Exception as e:
            print(f"Request failed: {e}")

    # ... Scrape logic ends ...

    # Save Data (Append Mode)
    if all_data:
        new_df = pd.DataFrame(all_data)
        print(f"Scraped {len(new_df)} new records.")
        
        if os.path.exists(OUTPUT_FILE):
            print(f"Found existing dataset at {OUTPUT_FILE}. Appending...")
            existing_df = pd.read_excel(OUTPUT_FILE)
            
            # Combine
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Remove Duplicates
            # We assume if (NAMA, MRG) are same, it's same listing. 
            # Or use all columns except NO/Index.
            # Ideally duplicates are subset=['NAMA RUMAH', 'HARGA', 'LB', 'LT']
            combined_df.drop_duplicates(subset=['NAMA RUMAH', 'LB', 'LT', 'KT', 'KM'], keep='last', inplace=True)
            
            # Re-index NO column
            combined_df.reset_index(drop=True, inplace=True)
            if 'NO' in combined_df.columns:
                combined_df['NO'] = combined_df.index + 1
            else:
                combined_df.insert(0, 'NO', combined_df.index + 1)
                
            final_df = combined_df
        else:
            print("No existing dataset found. Creating new one.")
            new_df.reset_index(inplace=True)
            new_df.rename(columns={'index': 'NO'}, inplace=True)
            new_df['NO'] = new_df['NO'] + 1
            final_df = new_df
        
        # Save
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        final_df.to_excel(OUTPUT_FILE, index=False)
        print(f"Successfully saved {len(final_df)} records to {OUTPUT_FILE}")
    else:
        print("No data scraped. Check selectors or anti-scraping blocking.")

def clean_price(price_text):
    # Legacy wrapper if needed, but regex handles it inside loop now
    return 0

def parse_int(text):
    try:
         return int(''.join(filter(str.isdigit, text)))
    except:
        return 0

if __name__ == "__main__":
    scrape_data(pages=2) # Try 2 pages
