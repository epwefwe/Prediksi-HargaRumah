from bs4 import BeautifulSoup
import re

with open("debug_page.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")
listings = soup.select('div[data-test-id^="srp-listing-card-"]')

print(f"Found {len(listings)} listings.")

for i, item in enumerate(listings):
    print(f"\n--- Listing {i} ---")
    full_text = item.get_text(" ", strip=True)
    
    # Try to isolate the part after LB
    if "LB" in full_text:
        try:
            # Find LB and print next 50 chars
            lb_index = full_text.find("LB")
            print(f"Context after LB: '{full_text[lb_index:lb_index+50]}...'")
            
            # Existing regex
            specs_match = re.search(r'LB\s*:\s*\d+\s*m[²2]?\s+(\d+)\s+(\d+)', full_text, re.IGNORECASE)
            if specs_match:
                print(f"Current Match: KT={specs_match.group(1)}, KM={specs_match.group(2)}")
            else:
                print("No Current Match")
                
            # New proposed regex validation
            # Look for 3 integers?
            extended_match = re.search(r'LB\s*:\s*\d+\s*m[²2]?\s+(\d+)\s+(\d+)\s+(\d+)', full_text, re.IGNORECASE)
            if extended_match:
                print(f"Extended Match: KT={extended_match.group(1)}, KM={extended_match.group(2)}, GRS={extended_match.group(3)}")
            else:
                print("No Extended Match (GRS might be missing)")
                
        except Exception as e:
            print(e)
