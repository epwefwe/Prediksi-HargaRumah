from bs4 import BeautifulSoup

with open("debug_page.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")
listings = soup.select('div[data-test-id^="srp-listing-card-"]')

print(f"Found {len(listings)} listings.")

for i, item in enumerate(listings):
    grs = "0"
    carport_icon = item.find("use", href=lambda x: x and "#carports-icon" in x)
    if not carport_icon:
        carport_icon = item.find("use", attrs={"xlink:href": lambda x: x and "#carports-icon" in x})
    
    if carport_icon:
        svg = carport_icon.find_parent("svg")
        if svg:
            span = svg.find_parent("span")
            if span:
                grs = span.get_text(strip=True)
    
    if grs != "0" and grs != "1":
         print(f"Listing {i}: GRS={grs}")
    elif grs == "1" and i < 3: # Only print a few 1s to verify
         print(f"Listing {i}: GRS={grs}")
