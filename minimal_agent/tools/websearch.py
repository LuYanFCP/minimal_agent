import logging
import requests
from typing import ParamSpec, List, Dict, Any, Optional
from datetime import datetime
import re
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from minimal_agent.tools.base import Tools, ToolsTypeEnum
from minimal_agent.tools.types import Arg

P = ParamSpec("P")


class SearxngWebSearch(Tools[P, List[Dict[str, Any]]]):
    def __init__(self, searx_host: str = "http://localhost:8888", count: int = 10):
        self.searx_host = searx_host
        self.count = count
        super().__init__(
            name="searxng_websearch",
            description="Perform a web search using the SearxNG search engine.",
            args=[
                Arg(
                    arg_name="query",
                    arg_desc="The search query string.",
                    arg_type="str",
                    required=True,
                ),
            ],
            func=self._inner_websearch,
        )

    @property
    def tool_type(self) -> ToolsTypeEnum:
        return ToolsTypeEnum.WEB_SEARCH

    def clean_html(
        self, html_content: str, main_content_selector: Optional[str] = None
    ) -> str:
        soup = BeautifulSoup(html_content, "html.parser")

        for tag in soup.select(
            "script, style, nav, footer, header, aside, .ads, .advertisement, .banner, .comments, iframe"
        ):
            tag.decompose()

        return str(soup)

    def html_to_markdown(self, html_content: str, **options) -> str:
        default_options = {
            "heading_style": "atx",
            "convert": [
                "b",
                "i",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "a",
                "img",
                "ul",
                "ol",
                "li",
                "p",
                "blockquote",
                "pre",
                "code",
                "table",
                "tr",
                "td",
                "th",
            ],
            "escape_asterisks": True,
            "escape_underscores": True,
        }

        default_options.update(options)

        markdown = md(html_content, **default_options)

        markdown = self.clean_markdown(markdown)

        return markdown

    def clean_markdown(self, markdown_text: str) -> str:
        markdown_text = re.sub(r"\n{3,}", "\n\n", markdown_text)

        markdown_text = "\n".join([line.strip() for line in markdown_text.split("\n")])

        markdown_text = re.sub(r"-{4,}", "---", markdown_text)

        markdown_text = re.sub(r"([^\n])(\n#{1,6} )", r"\1\n\n\2", markdown_text)
        markdown_text = re.sub(r"(#{1,6} .+?)(\n[^#\n])", r"\1\n\n\2", markdown_text)

        return markdown_text

    def _inner_format_result(
        self, output: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Format the search results into a structured list with enhanced content processing.
        """
        structured_citations = []

        for result in output[:self.count]:
            url = result.get("url")
            if not url:
                continue

            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                page_response = requests.get(url, headers=headers, timeout=10)
                page_response.raise_for_status()

                if page_response.encoding == "ISO-8859-1":
                    page_response.encoding = page_response.apparent_encoding

                cleaned_html = self.clean_html(
                    page_response.text
                )

                markdown_text = self.html_to_markdown(cleaned_html)

                citation = {
                    "source": result.get("engines", ["Unknown"])[
                        0
                    ],
                    "author": result.get("author", "Unknown"),
                    "title": result.get("title", "Unknown"),
                    "url": url,
                    "datePublished": result.get(
                        "publishedDate", datetime.now().strftime("%Y-%m-%d")
                    ),
                    "accessedDate": datetime.now().strftime("%Y-%m-%d"),
                    "markdownContent": markdown_text,
                }

                if len(markdown_text) > 500:
                    citation["summary"] = markdown_text[:500] + "..."
                else:
                    citation["summary"] = markdown_text

                structured_citations.append(citation)

            except requests.RequestException as e:
                logging.error(f"Error fetching {url}: {e}")
                structured_citations.append(
                    {
                        "source": result.get("engines", ["Unknown"])[0],
                        "title": result.get("title", "Unknown"),
                        "url": url,
                        "error": str(e),
                        "markdownContent": f"*Error fetching content: {str(e)}*",
                    }
                )
            except Exception as e:
                logging.error(f"Error processing {url}: {e}")
                structured_citations.append(
                    {
                        "source": result.get("engines", ["Unknown"])[0],
                        "title": result.get("title", "Unknown"),
                        "url": url,
                        "error": str(e),
                        "markdownContent": f"*Error processing content: {str(e)}*",
                    }
                )

        return structured_citations

    def _inner_websearch(self, query: str) -> List[Dict[str, Any]]:
        url = f"{self.searx_host}/search"

        # Set up the parameters for the GET request
        params = {
            "q": query,  # The search query
            "format": "json",  # Request the response in JSON format
            "count": self.count,  # Number of results to return
        }

        try:
            response = requests.get(url, params=params, timeout=15)

            response.raise_for_status()

            results = response.json()

            return self._inner_format_result(results["results"])
        except requests.RequestException as e:
            # Handle any errors that occur during the request
            logging.error(f"An error occurred while making the request: {e}")
            return [{"error": f"Search request failed: {str(e)}"}]
        except (ValueError, KeyError) as e:
            # Handle any errors that may occur while parsing the response
            logging.error(f"An error occurred while parsing the response: {e}")
            return [{"error": f"Failed to parse search results: {str(e)}"}]
