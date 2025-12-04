import bz2
import xml.etree.ElementTree as ET

# The path you provided
file_path = '/home/max-tost/Dokumente/Wiki-To-Go/Data/Debugging_Data/enwiki-20250901-pages-articles-multistream1.xml-p1p41242.bz2'

def show_first_articles(path, count=3):
    print(f"Reading from: {path}")
    
    # Open the bz2 file in text mode ('rt') with utf-8 encoding
    try:
        with bz2.open(path, 'rt', encoding='utf-8') as f:
            # iterparse allows us to process the XML incrementally
            context = ET.iterparse(f, events=('end',))
            
            found = 0
            for event, elem in context:
                # Check if the tag ends with 'title' (ignoring namespaces)
                if elem.tag.endswith('title'):
                    print(f"{found + 1}. {elem.text}")
                    found += 1
                    
                    # Clear the element to free memory
                    elem.clear()
                    
                    if found >= count:
                        break
                        
    except FileNotFoundError:
        print("Error: File not found. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    show_first_articles(file_path)