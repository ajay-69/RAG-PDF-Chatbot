import fitz
class PDFLoader:
    def load(self, path: str) -> str:
        doc = fitz.open(path)
        pages = []
        for page in doc:
            pages.append(page.get_text())
        return "\n".join(pages)