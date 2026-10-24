import ast
import math
import operator
import re
from langchain_core.tools import tool
import logging
import requests
import os
from langchain_core.runnables import RunnableConfig

from dotenv import load_dotenv
from tavily import TavilyClient
from ddgs import DDGS
from app.rag.retrieval import retrieve_documents

load_dotenv()

logger = logging.getLogger(__name__)


def _is_document_overview_query(query: str) -> bool:
    """Detect requests that need coverage across a document, not one fact."""
    normalized = " ".join(query.casefold().split())
    patterns = (
        r"\b(which|what|list|name|identify)\b.{0,70}\b(algorithms?|experiments?|programs?|topics?|sections?)\b",
        r"\b(main topic|overall topic|overview)\b",
        r"\bsummar(?:ize|ise|y|izing|ising)\b",
        r"\bwhat (?:is|does) this (?:pdf|document|file) (?:about|cover)\b",
        r"\b(?:all|every)\b.{0,40}\b(algorithms?|experiments?|programs?|topics?|sections?)\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)

Binary_Operators = {
    ast.Add:operator.add,
    ast.Sub:operator.sub,
    ast.Mult:operator.mul,
    ast.Div:operator.truediv,
    ast.FloorDiv:operator.floordiv,
    ast.Mod:operator.mod,
    ast.Pow:operator.pow,
}

Unary_Operators = {
    ast.UAdd:operator.pos,
    ast.USub:operator.neg,
}

def evaluate_node(node):
    """Evaluate only supported arithmetic operations."""

    if isinstance(node,ast.Expression):
        return evaluate_node(node.body)

    if isinstance(node,ast.Constant):
        value = node.value

        if (
            isinstance(value,(int,float))
            and not isinstance(value,bool)
        ):
            if isinstance(value,int) and value.bit_length()>4096:
                raise ValueError("Number is too large.")

            if isinstance(value,float) and not math.isfinite(value):
                raise ValueError("Number must be finite.")

            return value

        raise ValueError("Only numeric values are allowed")

    if isinstance(node,ast.BinOp):
        operator_type = type(node.op)

        if operator_type not in Binary_Operators:
            raise ValueError("Unsupported operator.")

        left = evaluate_node(node.left)
        right = evaluate_node(node.right)

        if operator_type is ast.Pow and abs(right)>10:
            raise ValueError("Exponent is too large.")

        result = Binary_Operators[operator_type](left,right)

        if (
            not isinstance(result,(int,float))
            or isinstance(result,bool)
        ):
            raise ValueError("Only real-number results are allowed.")

        if isinstance(result,float) and not math.isfinite(result):
            raise ValueError("Result is not finite.")

        return result

    if isinstance(node,ast.UnaryOp):
        operator_type = type(node.op)

        if operator_type not in Unary_Operators:
            raise ValueError("Unsupported unary operator.")

        return Unary_Operators[operator_type](
            evaluate_node(node.operand)
        )

    raise ValueError("Only basic arithmetic is allowed.")

@tool
def calculator(expression:str) -> str:
    """Calculate arithmetic expressions using numbers and basic operators."""

    if not isinstance(expression,str) or not expression.strip():
        return "Error: Expression must not be empty."

    if len(expression) > 100:
        return "Error: Expression is too long."

    try:
        syntax_tree = ast.parse(expression,mode = "eval")
        result = evaluate_node(syntax_tree)
        return str(result)

    except ZeroDivisionError:
        return "Error: Division by zero."

    except (SyntaxError,ValueError,OverflowError) as error:
        return f"Error: {error}"


@tool
def web_search(query: str) -> str:
    """Search the web using Tavily, with DDGS as a fallback."""

    try:
        results = TavilyClient().search(
            query=query,
            max_results=5,
        ).get("results", [])

        if results:
            return "\n\n".join(
                f"Title: {r.get('title', '')}\n"
                f"URL: {r.get('url', '')}\n"
                f"Content: {r.get('content', '')}"
                for r in results
            )

    except Exception as error:
        logger.warning("Tavily failed; trying DDGS: %s", error)

    try:
        results = DDGS().text(query, max_results=5)

        if results:
            return "\n\n".join(
                f"Title: {r.get('title', '')}\n"
                f"URL: {r.get('href', '')}\n"
                f"Content: {r.get('body', '')}"
                for r in results
            )

        return "No search results found."

    except Exception as error:
        logger.error("DDGS search failed: %s", error)
        return "Web search is currently unavailable."

@tool
def stock_price(symbol: str) -> str:
    """Get the latest available stock quote."""

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol.strip().upper(),
                "apikey": os.getenv("ALPHAVANTAGE_API_KEY"),
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        quote = data.get("Global Quote", {})

        if not quote:
            return data.get(
                "Note",
                data.get("Information", "No stock data found.")
            )

        return (
            f"Symbol: {quote['01. symbol']}\n"
            f"Price: {quote['05. price']}\n"
            f"Trading day: {quote['07. latest trading day']}"
        )

    except requests.RequestException:
        return "Unable to fetch stock data. Try again later."

@tool
def get_weather(location: str) -> str:
    """Get current weather conditions for a city."""

    try:
        place_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1},
            timeout=10,
        )
        place_response.raise_for_status()

        places = place_response.json().get("results", [])

        if not places:
            return f"Location not found: {location}"

        place = places[0]

        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": (
                    "temperature_2m,apparent_temperature,"
                    "relative_humidity_2m,wind_speed_10m"
                ),
                "timezone": "auto",
            },
            timeout=10,
        )
        response.raise_for_status()

        weather = response.json()["current"]

        return (
            f"Weather in {place['name']}, {place.get('country', '')}:\n"
            f"Temperature: {weather['temperature_2m']}°C\n"
            f"Feels like: {weather['apparent_temperature']}°C\n"
            f"Humidity: {weather['relative_humidity_2m']}%\n"
            f"Wind speed: {weather['wind_speed_10m']} km/h"
        )

    except (requests.RequestException, KeyError, IndexError) as error:
        logger.error("Weather lookup failed: %s", error)
        return "Weather data is currently unavailable."

@tool
def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
) -> str:
    """Convert an amount between two currencies."""

    source = from_currency.strip().upper()
    target = to_currency.strip().upper()

    try:
        response = requests.get(
            f"https://api.frankfurter.dev/v2/rate/{source}/{target}",
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        converted = amount * data["rate"]

        return (
            f"{amount:.2f} {source} = {converted:.2f} {target}\n"
            f"Rate date: {data['date']}"
        )

    except requests.RequestException:
        return "Unable to fetch exchange rates. Check the currency codes."

@tool
def search_knowledge_base(
    query: str,
    config: RunnableConfig,
    source: str = "",
) -> str:
    """Search uploaded documents for relevant content."""

    active_documents = (
        config.get("configurable", {}).get("active_documents", [])
    )

    # If only one document is active, use its authoritative source name.
    if len(active_documents) == 1:
        source = active_documents[0]

    elif active_documents and source:
        matches = [
            name for name in active_documents
            if name == source
            or name.endswith(source)
            or source.endswith(name)
        ]

        if len(matches) == 1:
            source = matches[0]
        else:
            return (
                "Could not uniquely identify the requested document. "
                "Please clarify which uploaded document you mean."
            )

    # Broad overview questions need more candidates so the grader can select
    # relevant chunks from different parts of the same PDF. Keep focused fact
    # lookups at the existing k=4 to avoid unnecessary token usage.
    retrieval_k = 12 if _is_document_overview_query(query) else 4
    logger.info(
        "Knowledge-base retrieval scope=%s; candidate_k=%d",
        "document overview" if retrieval_k > 4 else "focused question",
        retrieval_k,
    )

    documents = retrieve_documents(
        query,
        k=retrieval_k,
        source=source or None,
    )

    if not documents:
        return "No relevant information found in the knowledge base."

    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}, "
        f"Page: {doc.metadata.get('page', 'N/A')}\n"
        f"{doc.page_content}"
        for doc in documents
    )

tools = [
    calculator,web_search,stock_price,get_weather,convert_currency,search_knowledge_base,
]