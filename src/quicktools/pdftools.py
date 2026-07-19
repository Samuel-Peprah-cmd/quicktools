"""PDF utilities: extracting, merging, splitting, rotating, and creating PDFs."""
from pypdf import PdfReader, PdfWriter


def extract_text_from_pdf(path: str) -> str:
    """Extract and return all text content from a PDF file."""
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def get_pdf_page_count(path: str) -> int:
    """Return the number of pages in a PDF file."""
    return len(PdfReader(path).pages)


def merge_pdfs(paths: list[str], output_path: str) -> None:
    """Merge multiple PDF files into a single PDF, in the given order."""
    writer = PdfWriter()
    for path in paths:
        reader = PdfReader(path)
        for page in reader.pages:
            writer.add_page(page)
    with open(output_path, "wb") as f:
        writer.write(f)


def split_pdf_to_pages(path: str, output_dir: str) -> list[str]:
    """Split a PDF into individual single-page PDF files. Returns the list of output file paths."""
    import os
    reader = PdfReader(path)
    output_paths = []
    for i, page in enumerate(reader.pages):
        writer = PdfWriter()
        writer.add_page(page)
        out_path = os.path.join(output_dir, f"page_{i + 1}.pdf")
        with open(out_path, "wb") as f:
            writer.write(f)
        output_paths.append(out_path)
    return output_paths


def extract_pdf_pages(path: str, page_numbers: list[int], output_path: str) -> None:
    """Extract specific pages (1-indexed) from a PDF into a new PDF file."""
    reader = PdfReader(path)
    writer = PdfWriter()
    for num in page_numbers:
        writer.add_page(reader.pages[num - 1])
    with open(output_path, "wb") as f:
        writer.write(f)


def rotate_pdf(path: str, degrees: int, output_path: str) -> None:
    """Rotate every page in a PDF by the given degrees (must be a multiple of 90)."""
    reader = PdfReader(path)
    writer = PdfWriter()
    for page in reader.pages:
        page.rotate(degrees)
        writer.add_page(page)
    with open(output_path, "wb") as f:
        writer.write(f)


def get_pdf_metadata(path: str) -> dict:
    """Return the metadata (title, author, creation date, etc.) of a PDF file."""
    reader = PdfReader(path)
    meta = reader.metadata
    return dict(meta) if meta else {}


def create_pdf_from_text(text: str, output_path: str) -> None:
    """Create a simple PDF file containing the given text, wrapping and paginating automatically."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch

    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    margin = inch
    max_width_chars = 90
    y = height - margin
    line_height = 14

    import textwrap
    for paragraph in text.split("\n"):
        wrapped_lines = textwrap.wrap(paragraph, max_width_chars) or [""]
        for line in wrapped_lines:
            if y < margin:
                c.showPage()
                y = height - margin
            c.drawString(margin, y, line)
            y -= line_height
    c.save()


def encrypt_pdf(path: str, password: str, output_path: str) -> None:
    """Create a password-protected copy of a PDF."""
    reader = PdfReader(path)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt(password)
    with open(output_path, "wb") as f:
        writer.write(f)


def decrypt_pdf(path: str, password: str, output_path: str) -> None:
    """Remove password protection from a PDF, given the correct password."""
    reader = PdfReader(path)
    if reader.is_encrypted:
        reader.decrypt(password)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    with open(output_path, "wb") as f:
        writer.write(f)


def add_page_numbers(path: str, output_path: str) -> None:
    """Add page numbers (bottom-center) to every page of a PDF."""
    import io
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    reader = PdfReader(path)
    writer = PdfWriter()

    for i, page in enumerate(reader.pages):
        packet = io.BytesIO()
        page_width = float(page.mediabox.width)
        c = canvas.Canvas(packet, pagesize=(page_width, float(page.mediabox.height)))
        c.drawCentredString(page_width / 2, 20, str(i + 1))
        c.save()
        packet.seek(0)
        overlay_reader = PdfReader(packet)
        page.merge_page(overlay_reader.pages[0])
        writer.add_page(page)

    with open(output_path, "wb") as f:
        writer.write(f)


def pdf_to_images(path: str, output_dir: str, dpi: int = 150) -> list[str]:
    """Render every page of a PDF as a PNG image. Requires the optional 'PyMuPDF' package
    (pip install pymupdf) — raises ImportError with instructions if it's not installed."""
    import os
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "pdf_to_images() requires PyMuPDF. Install it with: pip install pymupdf"
        )
    doc = fitz.open(path)
    output_paths = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=matrix)
        out_path = os.path.join(output_dir, f"page_{i + 1}.png")
        pix.save(out_path)
        output_paths.append(out_path)
    return output_paths