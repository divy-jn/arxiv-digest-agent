import requests
from bs4 import BeautifulSoup
from app.pdf.extractor import ExtractedBlock

def extract_html(url: str) -> list[ExtractedBlock] | None:
    response = requests.get(url, timeout=20)
    if response.status_code != 200:
        return None
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # We want to extract semantic blocks: title, abstract, sections.
    blocks = []
    
    # Simple extraction strategy:
    # Look for abstract
    abstract = soup.find(id="abstract") or soup.find(class_="abstract")
    if abstract:
        text = abstract.get_text(separator=" ", strip=True)
        if text:
            blocks.append(ExtractedBlock(text=text, page=1, section="Abstract", subsection=""))
    
    # Main sections: looking for <section> or heading elements
    # Since arXiv html structure can vary, we will iterate over headings and paragraphs
    current_section = "Main"
    current_subsection = ""
    
    article = soup.find("article") or soup.find("main") or soup.find("body")
    if not article:
        return None
    
    for element in article.find_all(["h1", "h2", "h3", "h4", "p", "div"]):
        if element.name in ["h1", "h2"]:
            current_section = element.get_text(strip=True)
            current_subsection = ""
        elif element.name in ["h3", "h4"]:
            current_subsection = element.get_text(strip=True)
        elif element.name in ["p"]:
            text = element.get_text(separator=" ", strip=True)
            # Only add meaningful paragraphs
            if len(text) > 30:
                blocks.append(ExtractedBlock(
                    text=text,
                    page=1, # safe convention, html has no pages
                    section=current_section,
                    subsection=current_subsection
                ))
    
    if len(blocks) < 3: # If we extracted too little, probably a failure
        return None
        
    return blocks
