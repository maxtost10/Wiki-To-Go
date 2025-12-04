import bz2
import xml.etree.ElementTree as ET

file_path = '/home/max-tost/Dokumente/Wiki-To-Go/Data/Debugging_Data/enwiki-20250901-pages-articles-multistream1.xml-p1p41242.bz2'

def inspect_specific_article(path, target_title):
    print(f"Searching for '{target_title}' in {path}...")
    
    with bz2.open(path, 'rt', encoding='utf-8') as f:
        # iterparse walks through the XML file. 
        # We wait for the 'end' event of an element so we know it's fully loaded.
        context = ET.iterparse(f, events=('end',))
        
        for event, elem in context:
            # We only care when a full <page> tag is closed
            if elem.tag.endswith('page'):
                
                # Initialize variables to hold what we find inside this page
                current_title = None
                current_text = None
                
                # Iterate through the children of the <page> tag (like <title>, <revision>, etc.)
                for child in elem.iter():
                    if child.tag.endswith('title'):
                        current_title = child.text
                    elif child.tag.endswith('text'):
                        current_text = child.text
                
                # Check if this is the article we want
                if current_title == target_title:
                    print(f"\n=== FOUND ARTICLE: {current_title} ===\n")
                    
                    if current_text:
                        # Print the first 2000 characters to see the structure
                        print("--- RAW CONTENT START ---")
                        print(current_text[:2000])
                        print("\n--- RAW CONTENT END (Truncated) ---")
                    else:
                        print("No text content found.")
                    
                    # We found what we wanted, so we stop the function
                    return

                # Important: Clear the element from memory after processing to keep RAM usage low
                elem.clear()

if __name__ == "__main__":
    inspect_specific_article(file_path, "Anarchism")