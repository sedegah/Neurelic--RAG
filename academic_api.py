import requests

# --- arXiv API Integration ---
def fetch_arxiv_results(query, max_results=3):
    """
    Fetches papers from arXiv based on a search query.
    """
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        entries = []
        for entry in response.text.split("<entry>")[1:]:
            title = entry.split("<title>")[1].split("</title>")[0].strip()
            summary = entry.split("<summary>")[1].split("</summary>")[0].strip()
            entries.append({"title": title, "summary": summary})
        return entries
    except Exception as e:
        return [{"error": str(e)}]


# --- Wikipedia Summary API (supports multilingual) ---
def fetch_wikipedia_summary(query, lang="en"):
    """
    Fetches a summary of a topic from Wikipedia in the specified language.
    """
    try:
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("extract", "No summary available.")
        else:
            return "No Wikipedia summary found."
    except Exception as e:
        return f"Error: {e}"


# --- Crossref Metadata API ---
def fetch_crossref_metadata(query):
    """
    Fetches metadata about academic publications from Crossref.
    """
    url = f"https://api.crossref.org/works?query={query}&rows=3"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        items = data.get("message", {}).get("items", [])
        return [
            {
                "title": item.get("title", ["No title"])[0],
                "author": item.get("author", [{}])[0].get("family", "Unknown"),
                "published": item.get("published-print", {}).get("date-parts", [[None]])[0][0],
                "doi": item.get("DOI", "N/A")
            }
            for item in items
        ]
    except Exception as e:
        return [{"error": str(e)}]
