import logging
import re
from typing import Literal

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, ConfigDict

from app.rag.ingestion import vector_store

load_dotenv()

logger = logging.getLogger(__name__)

UPPER_TH = 0.7
LOWER_TH = 0.3

crag_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


class ChunkScore(BaseModel):
    """Structured relevance score for one retrieved chunk."""

    model_config = ConfigDict(extra="forbid")

    index: int
    score: float
    reason: str

class ChunkScoreBatch(BaseModel):
    """Relevance scores for all retrieved chunks in one model call."""

    model_config = ConfigDict(extra="forbid")

    evaluations: list[ChunkScore]


class QueryRewrite(BaseModel):
    """A rewritten document-retrieval query."""

    model_config = ConfigDict(extra="forbid")

    query: str


evaluator_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for a document-only RAG system. "
            "Evaluate each numbered chunk against the user's question. "
            "Give each chunk a relevance score from 0.0 to 1.0. "
            "Use 1.0 only when a chunk contains direct evidence that answers "
            "the question fully or mostly. Use 0.0 when it is irrelevant. "
            "Same topic or matching keywords alone are not enough. "
            "Return exactly one evaluation for every supplied chunk, preserve "
            "the chunk indices, and include a short reason for each score. "
            "Do not answer the user's question and do not use outside knowledge.",
        ),
        (
            "human",
            "Question:\n{question}\n\nRetrieved chunks:\n{chunks}",
        ),
    ]
)

grade_chain = evaluator_prompt | crag_llm.with_structured_output(
    ChunkScoreBatch,
    method="json_schema",
    strict=True,
)

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the input as a standalone search query for an uploaded document. "
            "Preserve the user's original intent and all names or specific entities. "
            "If the input is a short fragment, expand it with useful document-search "
            "terms and likely synonyms. For example, '12th' can become '12th-grade "
            "percentage, marks, and educational qualification stated in the resume'. "
            "Do not answer the question, invent facts, or search the web. Return only "
            "the rewritten query.",
        ),
        (
            "human",
            "Selected document source: {source}\nOriginal question or query: {query}",
        ),
    ]
)

rewrite_chain = rewrite_prompt | crag_llm.with_structured_output(
    QueryRewrite,
    method="json_schema",
    strict=True,
)


_PROGRAM_MARKER_RE = re.compile(
    r"\b(?:PROGRAM|EXPERIMENT|LAB\s+EXERCISE)\s*[-:#]?\s*\d+"
    r"(?:\s*\([A-Z0-9]+\))?",
    re.IGNORECASE,
)
_OBJECTIVE_RE = re.compile(r"\b(?:objective|aim)\s*[-:–]", re.IGNORECASE)
_OVERVIEW_QUERY_RE = re.compile(
    r"\b(?:which|what|list|identify|all|every|overview|summari[sz]e|summary|"
    r"main topic|covered|contents|programs?|experiments?|algorithms?)\b",
    re.IGNORECASE,
)


def _is_document_overview_query(query: str) -> bool:
    """Recognize broad questions that need coverage across a whole document."""
    q = query.casefold()
    asks_for_coverage = any(
        phrase in q
        for phrase in (
            "covered in", "main topic", "what does this document cover",
            "what does this pdf cover", "what is this pdf about",
            "summarize", "summary of", "overview of", "contents of",
            "which algorithms", "what algorithms", "all algorithms",
            "which experiments", "what experiments", "all experiments",
            "which programs", "what programs", "all programs",
            "list the algorithms", "list the experiments", "list the programs",
        )
    )
    return asks_for_coverage or bool(
        _OVERVIEW_QUERY_RE.search(q)
        and re.search(r"\b(?:all|every|overview|covered|contents|algorithms?|experiments?|programs?)\b", q)
    )


def _overview_chunks_from_source(source: str,k: int,user_id: str,) ->list[Document]:
    """Select representative program/objective chunks across the selected PDF.

    Unlike similarity-only retrieval, this scans the selected source's already
    indexed chunk metadata and prioritizes explicit program/experiment headings.
    It makes no LLM/API call. The normal CRAG grader still decides which chunks
    are relevant before they are returned to the answer model.
    """
    stored = vector_store.get(
        where={
            "$and": [
                {"user_id": user_id},
                {"source": source},
            ]
        },
        include=["documents", "metadatas"],
    )
    contents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []

    best_by_section: dict[str, tuple[int, Document]] = {}
    for content, metadata in zip(contents, metadatas):
        if not content or not isinstance(content, str):
            continue
        metadata = metadata or {}
        program_match = _PROGRAM_MARKER_RE.search(content)
        objective_match = _OBJECTIVE_RE.search(content)

        if not program_match and not objective_match:
            continue

        score = 0
        if program_match:
            score += 4
        if objective_match:
            score += 4
        if re.search(r"\b(?:implement|implementation|algorithm|sort|search|experiment)\b", content, re.I):
            score += 2
        if not score:
            continue

        if program_match:
            section_key = re.sub(r"\s+", "", program_match.group(0)).casefold()
        else:
            section_key = f"page:{metadata.get('page', 'unknown')}"

        document = Document(page_content=content, metadata=metadata)
        current = best_by_section.get(section_key)
        if current is None or score > current[0]:
            best_by_section[section_key] = (score, document)

    candidates = [item[1] for item in best_by_section.values()]
    if not candidates:
        return []

    def page_key(doc: Document):
        page = doc.metadata.get("page", 0)
        try:
            return int(page)
        except (TypeError, ValueError):
            return 0

    candidates.sort(key=page_key)
    return candidates[:k]


def _search_documents(
    query: str,
    user_id: str,
    k: int = 4,
    source: str | None = None,
) -> list[Document]:
    """Retrieve chunks restricted to the current user and optional source."""

    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("A valid user_id is required.")

    if source and _is_document_overview_query(query):
        try:
            overview_documents = _overview_chunks_from_source(
                source,
                k,
                user_id=user_id,
            )

            if overview_documents:
                logger.info(
                    "Document-overview retrieval selected %d section chunks from %s.",
                    len(overview_documents),
                    source,
                )
                return overview_documents

            logger.info(
                "No section headings found for document overview; "
                "using similarity search."
            )

        except Exception:
            logger.exception(
                "Document-overview scan failed for %s; "
                "falling back to similarity search.",
                source,
            )

    search_filter = {"user_id": user_id}

    if source:
        search_filter = {
            "$and": [
                {"user_id": user_id},
                {"source": source},
            ]
        }

    search_kwargs = {
        "k": k,
        "filter": search_filter,
    }

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    return retriever.invoke(query)


def _grade_documents(
    query: str,
    documents: list[Document],
) -> tuple[Literal["CORRECT", "INCORRECT", "AMBIGUOUS"], list[Document], list[ChunkScore]]:
    """Score chunks in one LLM call and apply the tutorial's thresholds."""

    if not documents:
        return "INCORRECT", [], []

    chunk_text = "\n\n".join(
        f"CHUNK INDEX: {index}\n"
        f"SOURCE: {doc.metadata.get('source', 'Unknown')}\n"
        f"PAGE: {doc.metadata.get('page', 'N/A')}\n"
        f"CONTENT:\n{doc.page_content}"
        for index, doc in enumerate(documents, start=1)
    )

    response = grade_chain.invoke(
        {"question": query, "chunks": chunk_text}
    )

    scores_by_index: dict[int, ChunkScore] = {}
    for evaluation in response.evaluations:
        if evaluation.index in scores_by_index:
            raise ValueError(
                f"CRAG grader returned duplicate chunk index {evaluation.index}."
            )
        if not 1 <= evaluation.index <= len(documents):
            raise ValueError(
                f"CRAG grader returned invalid chunk index {evaluation.index}."
            )
        if not 0.0 <= evaluation.score <= 1.0:
            raise ValueError(
                f"CRAG grader returned out-of-range score {evaluation.score}."
            )
        scores_by_index[evaluation.index] = evaluation

    expected_indices = set(range(1, len(documents) + 1))
    if set(scores_by_index) != expected_indices:
        raise ValueError(
            "CRAG grader did not return exactly one score for every retrieved chunk."
        )

    evaluations = [scores_by_index[i] for i in range(1, len(documents) + 1)]
    numeric_scores = [evaluation.score for evaluation in evaluations]

    if any(score > UPPER_TH for score in numeric_scores):
        verdict: Literal["CORRECT", "INCORRECT", "AMBIGUOUS"] = "CORRECT"
        selected = [
            doc
            for index, doc in enumerate(documents, start=1)
            if scores_by_index[index].score > LOWER_TH
        ]
    elif all(score < LOWER_TH for score in numeric_scores):
        verdict = "INCORRECT"
        selected = []
    else:
        verdict = "AMBIGUOUS"
        selected = []

    return verdict, selected, evaluations


def _rewrite_query(query: str, source: str | None = None) -> str:
    """Rewrite a query for another retrieval attempt within the same source."""

    result = rewrite_chain.invoke(
        {"query": query, "source": source or "uploaded document"}
    )
    rewritten = " ".join(result.query.split()).strip(" \t\n\r\"'")
    return rewritten


def retrieve_documents(
    query: str,
    user_id: str,
    k: int = 4,
    source: str | None = None,
) -> list[Document]:
    """Run document-only CRAG with one corrective retrieval attempt."""

    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("A valid user_id is required.")

    print(f"\n[CRAG] Original query: {query}")
    print(f"[CRAG] Selected source: {source}")

    documents = _search_documents(
        query=query,
        user_id=user_id,
        k=k,
        source=source,
    )
    print(f"[CRAG] Initial chunks retrieved: {len(documents)}")

    verdict, relevant_documents, evaluations = _grade_documents(
        query, documents
    )
    print(f"[CRAG] Initial verdict: {verdict}")

    for evaluation in evaluations:
        print(
            f"[CRAG] Chunk {evaluation.index}: score={evaluation.score:.2f}; "
            f"reason={evaluation.reason}"
        )

    if verdict == "CORRECT":
        print(f"[CRAG] Returning {len(relevant_documents)} relevant chunks.")
        return relevant_documents

    if source and _is_document_overview_query(query):
        overview_relevant = [
            doc
            for doc, evaluation in zip(documents, evaluations)
            if evaluation.score > LOWER_TH
        ]
        if overview_relevant:
            print(
                f"[CRAG] Overview query: retaining {len(overview_relevant)} "
                "positively graded section chunks."
            )
            return overview_relevant

    rewritten_query = _rewrite_query(query, source)
    print(f"[CRAG] Rewritten query: {rewritten_query}")

    if not rewritten_query or rewritten_query.casefold() == query.casefold():
        print("[CRAG] Query rewrite produced no change; stopping safely.")
        return []

    retry_documents = _search_documents(
        query=rewritten_query,
        user_id=user_id,
        k=k,
        source=source,
    )
    print(f"[CRAG] Retry chunks retrieved: {len(retry_documents)}")

    retry_verdict, retry_relevant_documents, retry_evaluations = (
        _grade_documents(query, retry_documents)
    )
    print(f"[CRAG] Retry verdict: {retry_verdict}")

    for evaluation in retry_evaluations:
        print(
            f"[CRAG] Retry chunk {evaluation.index}: "
            f"score={evaluation.score:.2f}; reason={evaluation.reason}"
        )

    if retry_verdict == "CORRECT":
        print(
            f"[CRAG] Returning {len(retry_relevant_documents)} corrected chunks."
        )
        return retry_relevant_documents

    if source and _is_document_overview_query(query):
        overview_relevant = [
            doc
            for doc, evaluation in zip(retry_documents, retry_evaluations)
            if evaluation.score > LOWER_TH
        ]
        if overview_relevant:
            print(
                f"[CRAG] Overview retry: retaining {len(overview_relevant)} "
                "positively graded section chunks."
            )
            return overview_relevant

    print(
        "[CRAG] No confident document evidence after correction; "
        "returning no chunks."
    )
    return []