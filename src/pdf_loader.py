import fitz
from pathlib import Path

class PDFLoader:
    def load(self, path: str) -> list[dict]:
        doc = fitz.open(path)
        pages = []
        source = Path(path).stem    

        for page_number, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                pages.append({
                    "text":text,
                    "page": page_number,
                    "source": source
                    })
        return pages