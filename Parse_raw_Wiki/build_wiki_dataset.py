import bz2
import xml.etree.ElementTree as ET
import re
import os
import time

# --- CONFIGURATION (UPDATED FOR RUNPOD) ---
INPUT_FILE = '/workspace/Data/enwiki-latest-pages-articles-multistream.xml.bz2'

# Output: The cleaned text file
OUTPUT_FILE = '/workspace/Data/wiki_data_cleaned.txt'

MIN_ARTICLE_LENGTH = 200  
DEBUG_LIMIT = None  # <--- CHANGED: Process EVERYTHING (Takes ~3-5 hours)

def remove_nested_brackets(text, prefixes_to_delete):
    """Robustly removes [[...]] blocks with specific prefixes, keeping others."""
    result = []
    i = 0
    n = len(text)
    
    while i < n:
        if text[i:i+2] == '[[':
            depth = 0
            start_pos = i
            end_pos = -1
            
            for j in range(i, n):
                if text[j:j+2] == '[[':
                    depth += 1
                elif text[j:j+2] == ']]':
                    depth -= 1
                    if depth == 0:
                        end_pos = j + 2
                        break
            
            if end_pos != -1:
                full_block = text[start_pos:end_pos]
                inner_content = full_block[2:-2]
                processed_inner = remove_nested_brackets(inner_content, prefixes_to_delete)
                
                lower_inner = processed_inner.lower()
                should_delete = any(lower_inner.startswith(p.lower()) for p in prefixes_to_delete)
                
                if should_delete and len(full_block) > 1000: should_delete = False
                
                if not should_delete:
                    if '|' in processed_inner:
                        result.append(processed_inner.split('|')[-1])
                    else:
                        result.append(processed_inner)
                
                i = end_pos
                continue
            else:
                result.append(text[i])
                i += 1
        else:
            result.append(text[i])
            i += 1
    return "".join(result)

def clean_wikitext(text):
    # 1. Cutoff at Footer Sections
    cutoff_headers = ["See also", "References", "Notes", "Further reading", "External links", "Bibliography"]
    pattern_str = r'==\s*(' + '|'.join(cutoff_headers) + r')\s*=='
    match = re.search(pattern_str, text, re.IGNORECASE)
    if match:
        text = text[:match.start()]

    # 2. Remove Tables & Comments
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # 3. Remove References (Safe Limit)
    text = re.sub(r'<ref[^>]*?>.{0,5000}?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*?/>', '', text)

    # 4. Remove Templates
    pattern = re.compile(r'\{\{[^{}]*?\}\}')
    while True:
        new_text = pattern.sub('', text)
        if len(new_text) == len(text):
            break
        text = new_text

    # 5. Process Links/Files
    text = remove_nested_brackets(text, ['File:', 'Image:', 'Category:', 'en:'])

    # 6. Final Cleanup (HTML, Formatting, Newlines)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r"''+", '', text)
    text = re.sub(r'=+\s*(.*?)\s*=+', r'\1', text)
    
    # Flatten to one line
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_dump(input_path, output_path, limit=None):
    print(f"Processing {input_path} -> {output_path}")
    if limit:
        print(f"DEBUG MODE: Stopping after {limit} articles.")
    
    start_time = time.time()
    count = 0
    
    with bz2.open(input_path, 'rt', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        context = ET.iterparse(f_in, events=('end',))
        
        for event, elem in context:
            if elem.tag.endswith('page'):
                title = None
                text_content = None
                redirect = False
                
                for child in elem.iter():
                    if child.tag.endswith('title'):
                        title = child.text
                    elif child.tag.endswith('redirect'):
                        redirect = True
                    elif child.tag.endswith('text'):
                        text_content = child.text
                
                # Filter: Must have text, not be a redirect, and be long enough
                if text_content and not redirect and title:
                    cleaned = clean_wikitext(text_content)
                    
                    if len(cleaned) > MIN_ARTICLE_LENGTH:
                        f_out.write(cleaned + '\n')
                        count += 1
                        
                        # Print progress every 1000 articles so your terminal doesn't freeze
                        if count % 1000 == 0:
                            print(f"Processed {count} articles...", end='\r')
                        
                        if limit and count >= limit:
                            print(f"\nReached limit of {limit} articles.")
                            print(f"Done! Processed {count} articles in {time.time() - start_time:.2f} seconds.")
                            return
                
                elem.clear()
                
    print(f"\nDone! Processed {count} articles in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    process_dump(INPUT_FILE, OUTPUT_FILE, DEBUG_LIMIT)