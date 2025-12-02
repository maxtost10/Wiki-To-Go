#!/usr/bin/env python3
"""
Wikipedia Dump Parser für Tokenizer Training
Unterstützt Debugging-Modus und skaliert auf volle Wikipedia-Dumps
"""

import xml.etree.ElementTree as ET
import bz2
import re
import argparse
import sys
from pathlib import Path
from typing import Iterator, Optional
import time


class WikipediaParser:
    """Parser für Wikipedia XML-Dumps"""
    
    # XML-Namespace (kann je nach Dump-Version variieren)
    NS = '{http://www.mediawiki.org/xml/export-0.11/}'
    
    def __init__(self, input_file: str, output_file: str, debug: bool = False, max_articles: Optional[int] = None):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.debug = debug
        self.max_articles = max_articles
        
        # Statistiken
        self.stats = {
            'total_pages': 0,
            'articles_processed': 0,
            'articles_skipped': 0,
            'bytes_written': 0
        }
        
    def parse(self):
        """Hauptfunktion zum Parsen des Dumps"""
        print(f"📚 Starte Wikipedia-Parser")
        print(f"   Input:  {self.input_file}")
        print(f"   Output: {self.output_file}")
        if self.max_articles:
            print(f"   Limit:  {self.max_articles} Artikel")
        if self.debug:
            print(f"   🐛 DEBUG-MODUS aktiviert")
        print()
        
        start_time = time.time()
        
        with bz2.open(self.input_file, 'rt', encoding='utf-8') as f_in, \
             open(self.output_file, 'w', encoding='utf-8') as f_out:
            
            for text in self._iterate_articles(f_in):
                # Text in Datei schreiben (ein Artikel pro Zeile)
                f_out.write(text + '\n')
                self.stats['bytes_written'] += len(text.encode('utf-8'))
                
                # Check ob Limit erreicht
                if self.max_articles and self.stats['articles_processed'] >= self.max_articles:
                    print(f"\n✓ Limit von {self.max_articles} Artikeln erreicht")
                    break
                
                # Fortschritt anzeigen
                if self.stats['articles_processed'] % 100 == 0:
                    self._print_progress()
        
        elapsed = time.time() - start_time
        self._print_summary(elapsed)
    
    def _iterate_articles(self, file_handle) -> Iterator[str]:
        """Iteriert durch alle Artikel im XML-Dump"""
        
        # Versuche automatisch den richtigen Namespace zu erkennen
        context = ET.iterparse(file_handle, events=('start', 'end'))
        _, root = next(context)
        
        # Extrahiere Namespace aus Root-Element
        if '}' in root.tag:
            self.NS = root.tag[:root.tag.index('}')+1]
            if self.debug:
                print(f"🔍 Erkannter XML-Namespace: {self.NS}")
        
        for event, elem in context:
            if event == 'end' and elem.tag == f'{self.NS}page':
                self.stats['total_pages'] += 1
                
                # Artikel extrahieren und verarbeiten
                text = self._extract_article(elem)
                
                if text:
                    self.stats['articles_processed'] += 1
                    
                    if self.debug and self.stats['articles_processed'] <= 3:
                        self._debug_print_article(text)
                    
                    yield text
                else:
                    self.stats['articles_skipped'] += 1
                
                # Speicher freigeben (wichtig für große Dumps!)
                elem.clear()
                
                # Root auch clearen um Speicher zu sparen
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
    
    def _extract_article(self, page_elem) -> Optional[str]:
        """Extrahiert und bereinigt Text aus einem Page-Element"""
        
        # 1. Prüfe Namespace (nur Hauptartikel = 0)
        ns_elem = page_elem.find(f'{self.NS}ns')
        if ns_elem is None or ns_elem.text != '0':
            return None
        
        # 2. Hole Titel (für Debug)
        title_elem = page_elem.find(f'{self.NS}title')
        title = title_elem.text if title_elem is not None else "Unknown"
        
        # 3. Hole Text-Inhalt
        revision = page_elem.find(f'{self.NS}revision')
        if revision is None:
            return None
        
        text_elem = revision.find(f'{self.NS}text')
        if text_elem is None or not text_elem.text:
            return None
        
        raw_text = text_elem.text
        
        # 4. Bereinige Wiki-Markup
        cleaned_text = self._clean_wikitext(raw_text)
        
        # 5. Finale Normalisierung
        cleaned_text = self._normalize_text(cleaned_text)
        
        # Überspringe sehr kurze Artikel (wahrscheinlich Redirects/Stubs)
        if len(cleaned_text.strip()) < 100:
            return None
        
        return cleaned_text
    
    def _clean_wikitext(self, text: str) -> str:
        """Entfernt MediaWiki-Markup"""
        
        # Kommentare entfernen
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        # Referenzen entfernen
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        text = re.sub(r'<ref[^>]*\/>', '', text)
        
        # Templates entfernen (geschachtelt möglich, daher iterativ)
        while '{{' in text:
            text = re.sub(r'\{\{[^{}]*\}\}', '', text)
        
        # Tabellen entfernen
        text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
        
        # Datei/Bild-Links entfernen
        text = re.sub(r'\[\[(File|Image|Datei|Bild):[^\]]+\]\]', '', text, flags=re.IGNORECASE)
        
        # Interne Links: [[Link|Text]] -> Text oder [[Link]] -> Link
        text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
        
        # Externe Links: [http://... Text] -> Text
        text = re.sub(r'\[https?://[^\s\]]+ ([^\]]+)\]', r'\1', text)
        text = re.sub(r'\[https?://[^\s\]]+\]', '', text)
        
        # Überschriften: == Text == -> Text
        text = re.sub(r'={2,6}\s*(.*?)\s*={2,6}', r'\1', text)
        
        # Formatierung
        text = re.sub(r"'{2,5}", '', text)  # '''bold''', ''italic''
        
        # HTML-Tags entfernen
        text = re.sub(r'<[^>]+>', '', text)
        
        # Kategorien und andere Meta-Zeilen entfernen
        text = re.sub(r'\[\[Category:.*?\]\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*\|.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*!.*$', '', text, flags=re.MULTILINE)
        
        return text
    
    def _normalize_text(self, text: str) -> str:
        """Normalisiert den Text"""
        
        # HTML-Entities
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&nbsp;', ' ')
        
        # Mehrfache Leerzeichen
        text = re.sub(r' +', ' ', text)
        
        # Mehrfache Newlines (max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Leerzeichen am Zeilenanfang/-ende
        text = '\n'.join(line.strip() for line in text.split('\n'))
        
        # Leere Zeilen am Anfang/Ende entfernen
        text = text.strip()
        
        return text
    
    def _debug_print_article(self, text: str):
        """Gibt einen Artikel im Debug-Modus aus"""
        print("=" * 80)
        print(f"📄 Artikel #{self.stats['articles_processed']}")
        print("=" * 80)
        print(text[:500])
        if len(text) > 500:
            print(f"\n... ({len(text) - 500} weitere Zeichen) ...\n")
        print("=" * 80)
        print()
    
    def _print_progress(self):
        """Zeigt Fortschritt an"""
        print(f"⏳ Verarbeitet: {self.stats['articles_processed']} Artikel "
              f"(übersprungen: {self.stats['articles_skipped']}, "
              f"gesamt: {self.stats['total_pages']} Seiten)")
    
    def _print_summary(self, elapsed: float):
        """Zeigt Zusammenfassung am Ende"""
        print("\n" + "=" * 80)
        print("✅ FERTIG!")
        print("=" * 80)
        print(f"Verarbeitete Artikel:    {self.stats['articles_processed']}")
        print(f"Übersprungene Seiten:    {self.stats['articles_skipped']}")
        print(f"Gesamt gelesene Seiten:  {self.stats['total_pages']}")
        print(f"Geschriebene Bytes:      {self.stats['bytes_written']:,}")
        print(f"Ausgabedatei Größe:      {self.output_file.stat().st_size / (1024*1024):.2f} MB")
        print(f"Zeit:                    {elapsed:.2f} Sekunden")
        print(f"Geschwindigkeit:         {self.stats['articles_processed'] / elapsed:.2f} Artikel/Sekunde")
        print("=" * 80)


def main():
    # Default-Werte für einfaches Debugging
    DEFAULT_INPUT = '/home/max-tost/Dokumente/Wiki-To-Go/Data/Debugging_Data/enwiki-20250901-pages-articles-multistream1.xml-p1p41242.bz2'
    DEFAULT_OUTPUT = '/home/max-tost/Dokumente/Wiki-To-Go/Data/Debugging_Data/output_debug.txt'
    
    parser = argparse.ArgumentParser(
        description='Wikipedia XML-Dump Parser für Tokenizer Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Debug-Modus mit nur 10 Artikeln
  python wikipedia_parser.py -i dump.xml.bz2 -o output.txt --debug --limit 10
  
  # Voller Durchlauf
  python wikipedia_parser.py -i dump.xml.bz2 -o output.txt
  
  # Erste 1000 Artikel für Testing
  python wikipedia_parser.py -i dump.xml.bz2 -o output.txt --limit 1000
  
  # Ohne Argumente (nutzt Defaults für Debugging)
  python wikipedia_parser.py
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        default=DEFAULT_INPUT,
        help=f'Input Wikipedia XML dump (kann .bz2 komprimiert sein). Default: {DEFAULT_INPUT}'
    )
    
    parser.add_argument(
        '-o', '--output',
        default=DEFAULT_OUTPUT,
        help=f'Output Textdatei (ein Artikel pro Zeile). Default: {DEFAULT_OUTPUT}'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='Debug-Modus: Zeigt erste 3 Artikel detailliert an (Default: True)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=10,  # Nur 10 Artikel standardmäßig für schnelles Testing
        help='Maximale Anzahl Artikel zu verarbeiten (für Testing). Default: 10'
    )
    
    args = parser.parse_args()
    
    # Info über verwendete Defaults
    if args.input == DEFAULT_INPUT:
        print("ℹ️  Verwende Default-Input (kein -i angegeben)")
    if args.output == DEFAULT_OUTPUT:
        print("ℹ️  Verwende Default-Output (kein -o angegeben)")
    print()
    
    # Validierung
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input-Datei nicht gefunden: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Parser starten
    wiki_parser = WikipediaParser(
        input_file=args.input,
        output_file=args.output,
        debug=args.debug,
        max_articles=args.limit
    )
    
    try:
        wiki_parser.parse()
    except KeyboardInterrupt:
        print("\n⚠️  Unterbrochen durch Benutzer")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fehler: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()