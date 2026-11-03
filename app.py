import logging
import os
from pathlib import Path
from urllib.parse import unquote, urlsplit

import bleach
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from markdown_it import MarkdownIt
from markdown_it.token import Token
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel

load_dotenv()

app = FastAPI()
logger = logging.getLogger(__name__)
APP_DIR = Path(__file__).resolve().parent
INDEX_FILE = APP_DIR / "templates" / "index.html"
configured_prompt_path = Path(os.getenv("SUBSIDY_PROMPT_PATH", "prompts/subsidy-office.md"))
PROMPT_FILE = configured_prompt_path if configured_prompt_path.is_absolute() else APP_DIR / configured_prompt_path
MARKDOWN = MarkdownIt("default", {"html": False})
ALLOWED_HTML_TAGS = {
    "a", "blockquote", "br", "code", "del", "em", "h1", "h2", "h3",
    "h4", "h5", "h6", "hr", "li", "ol", "p", "pre", "strong", "table",
    "tbody", "td", "th", "thead", "tr", "ul",
}


class AskRequest(BaseModel):
    question: str


def render_markdown(markdown: str, source_titles: dict[str, str]) -> str:
    tokens = MARKDOWN.parse(markdown)
    for block in tokens:
        if block.type != "inline" or not block.children:
            continue
        children = block.children
        index = 0
        while index < len(children):
            token = children[index]
            url = token.attrGet("href") if token.type == "link_open" else None
            if url not in source_titles:
                index += 1
                continue

            end = index + 1
            while end < len(children) and children[end].type != "link_close":
                end += 1
            label = Token("text", "", 0)
            label.content = source_titles[url]
            children[index + 1:end] = [label]
            index += 2

    rendered = MARKDOWN.renderer.render(tokens, MARKDOWN.options, {})
    return bleach.clean(
        rendered,
        tags=ALLOWED_HTML_TAGS,
        attributes={"a": ["href", "title"], "th": ["align"], "td": ["align"]},
        protocols={"http", "https", "mailto"},
        strip=True,
    )


def format_source_title(title: str, url: str) -> str:
    parsed_url = urlsplit(url)
    filename = unquote(Path(parsed_url.path).name)
    generic_titles = {parsed_url.hostname or "", f"www.{parsed_url.hostname}"}

    if not title or title.strip().casefold() in {value.casefold() for value in generic_titles}:
        stem = Path(filename).stem
        if "koubo" in stem.lower() and "youryou" in stem.lower():
            title = "公募要領"
        else:
            title = stem.replace("_", " ").replace("-", " ").strip() or parsed_url.hostname or "公式サイト"
    elif title.strip().casefold() == filename.casefold():
        title = Path(filename).stem.replace("_", " ").replace("-", " ").strip()

    title = title.strip()
    is_pdf = parsed_url.path.lower().endswith(".pdf") or title.lower().endswith(".pdf")
    if title.lower().endswith(".pdf"):
        title = title[:-4].rstrip()
    suffix = "（PDF）" if is_pdf else ""
    if is_pdf and title.endswith("（PDF）"):
        suffix = ""
    return f"[引用元] {title}{suffix}"


def get_sources(response, allowed_domain: str) -> list[dict[str, str]]:
    sources = []
    seen_urls = set()
    for item in getattr(response, "output", []):
        for content in getattr(item, "content", []):
            for annotation in getattr(content, "annotations", []):
                if getattr(annotation, "type", None) != "url_citation":
                    continue
                url = getattr(annotation, "url", "")
                parsed_url = urlsplit(url)
                if parsed_url.scheme != "https" or parsed_url.hostname != allowed_domain:
                    continue
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                title = format_source_title(getattr(annotation, "title", ""), url)
                sources.append({"title": title, "url": url})
    return sources


@app.get("/")
async def index():
    return FileResponse(INDEX_FILE, media_type="text/html")


@app.post("/api/ask")
async def ask(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="質問を入力してください。")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY が設定されていません。")

    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise HTTPException(status_code=500, detail="OPENAI_MODEL が設定されていません。")

    official_site_url = os.getenv("OFFICIAL_SITE_URL", "").strip().rstrip("/")
    parsed_site_url = urlsplit(official_site_url)
    if parsed_site_url.scheme != "https" or not parsed_site_url.hostname:
        raise HTTPException(status_code=500, detail="OFFICIAL_SITE_URL の設定を確認してください。")

    try:
        instructions = PROMPT_FILE.read_text(encoding="utf-8").replace(
            "{{OFFICIAL_SITE_URL}}", official_site_url
        )
    except OSError:
        logger.exception("Failed to read assistant instructions")
        raise HTTPException(status_code=500, detail="回答設定を読み込めませんでした。") from None

    try:
        async with AsyncOpenAI(api_key=api_key) as client:
            response = await client.responses.create(
                model=model,
                instructions=instructions,
                input=question,
                tools=[
                    {
                        "type": "web_search",
                        "filters": {"allowed_domains": [parsed_site_url.hostname]},
                    }
                ],
                tool_choice={"type": "web_search"},
                store=True,
            )
    except OpenAIError:
        logger.exception("OpenAI API request failed")
        raise HTTPException(status_code=502, detail="回答を取得できませんでした。") from None

    logger.info("OpenAI response id: %s", response.id)
    sources = get_sources(response, parsed_site_url.hostname)
    return {
        "answer_html": render_markdown(
            response.output_text,
            {source["url"]: source["title"] for source in sources},
        ),
        "sources": sources,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)