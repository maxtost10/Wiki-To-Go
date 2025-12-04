# Wikipedia Dump Parser & Cleaner

This project contains a set of tools designed to parse massive compressed Wikipedia XML dumps (`.bz2`), clean the raw Wikitext markup, and convert it into a machine-learning-friendly format (one plain text article per line).

## 🧠 Lessons Learned

During the development of this parser, we encountered and solved several specific challenges related to processing large-scale unstructured data:

### 1. Memory Management (The "Conveyor Belt")
**Problem:** Wikipedia dumps are massive (tens of gigabytes compressed). Loading the whole file into RAM crashes standard computers.
**Solution:** We use `xml.etree.ElementTree.iterparse` to stream the file. Crucially, we use `elem.clear()` immediately after processing each `<page>` tag. This acts like a conveyor belt: we pick up one article, process it, and then throw it in the "trash" (clear from RAM) before picking up the next one.

### 2. The Limits of Regex (The "Vacuum" Bug)
**Problem:** Simple regular expressions like `r'<ref>.*</ref>'` are "greedy." If an article has a `<ref>` at the start and another at the end, a simple regex might swallow the entire article text in between.
**Solution:** We implemented safety limits (e.g., `.{0,5000}`) to prevent regex from eating too much, and switched to iterative parsers for nested structures.

### 3. Nested Structures require Stacks, not Regex
**Problem:** Wikitext is often nested: `[[Image:File.jpg|thumb|This is a picture of [[Anarchism]]]]`. A simple regex to remove `[[Image...]]` fails because it stops at the *first* closing bracket it sees, leaving broken artifacts like `]]` behind.
**Solution:** We wrote a custom `remove_nested_brackets` function using a counter/stack approach. It counts opening `[[` and closing `]]` brackets to correctly identify the full block, allowing us to cleanly remove images while preserving the text inside valid links.

### 4. Raw Data vs. Rendered View
**Problem:** Extracted text sometimes appeared "mixed up" compared to the website.
**Discovery:** By inspecting the raw XML structure (`debug_raw_sections.py`), we learned that the raw file order often differs from the visual render. Images and infoboxes are often defined at the top of sections, even if they appear on the side or bottom of the webpage.

---

## 🛠 How the Script Works (`build_wiki_dataset.py`)

The script follows a strict "Clean & Flatten" pipeline for every article found in the stream:

1.  **Filter:** Skips "Redirect" pages and very short articles (stubs).
2.  **Truncate:** Detects footer headers like `== See also ==` or `== References ==` and cuts off the rest of the article.
3.  **Remove Noise:**
    *   **Tables:** Strips `{| ... |}` blocks.
    *   **References:** Removes `<ref>...</ref>` tags.
    *   **Comments:** Removes `<!-- ... -->`.
4.  **Recursive Cleaning:**
    *   **Templates:** Loops to remove nested `{{...}}` tags (infoboxes, citations).
    *   **Links/Files:** Uses the stack-parser to remove `[[File:...]]` and `[[Category:...]]` blocks entirely, while converting `[[Link|Text]]` into just `Text`.
5.  **Flatten:** Removes all newlines and extra spaces, resulting in a single long string of text per article.

## 🚀 Usage

### Configuration
Open `build_wiki_dataset.py` and edit the constants at the top:
```python
INPUT_FILE = '/path/to/enwiki-latest-pages-articles.xml.bz2'
OUTPUT_FILE = '/path/to/output.txt'
DEBUG_LIMIT = 5  # Set to None to process the whole file