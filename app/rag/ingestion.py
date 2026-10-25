from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="queryloom-knowlegde",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)


def ingest_document(file_path: str, user_id: str) -> int:
    """Read a document, split it into chunks, and store it."""

    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("A valid user_id is required.")

    path = Path(file_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    documents = []

    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": path.name,
                            "page": page_number,
                            "user_id": user_id,
                        },
                    )
                )

    elif path.suffix.lower() in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8")

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "user_id": user_id,
                        },
                )
            )

    else:
        raise ValueError("Only PDF, TXT, and MD files are supported.")

    if not documents:
        raise ValueError("No readable text found in the document.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )

    chunks = splitter.split_documents(documents)

    chunks = [
        chunk
        for chunk in chunks
        if len(chunk.page_content.strip()) >= 100
    ]

    if not chunks:
        raise ValueError("No usable chunks found in the document.")

    vector_store.delete(
        where={
            "$and": [
                {"user_id": user_id},
                {"source": path.name},
            ]
        }
    )
    vector_store.add_documents(chunks)

    return len(chunks)