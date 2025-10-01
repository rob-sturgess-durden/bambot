import json
import re
import sys
from pathlib import Path
from typing import List, Dict

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://businessandmission.org/resources/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ContentFetcher/1.0)"}

OUTPUT_JSONL = "resources_full.jsonl"
OUTPUT_MD = "resources_full.md"


def fetch_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.text


def absolutize(url: str, href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("/"):
        # root-relative on same domain
        from urllib.parse import urljoin
        return urljoin(url, href)
    from urllib.parse import urljoin
    return urljoin(url, href)


def extract_resources_list(html: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    # WordPress.com themes often render posts in article tags or within main content
    # We'll look for anchors within the content area that likely correspond to resource cards
    candidates: List[Dict[str, str]] = []

    # Primary: look for headings followed by excerpts and a "Read more" link
    for section in soup.select("main, .site-content, #content, .entry-content, .wp-block-group, body"):
        # each resource block seems to have an h2/h3 and a Read more link
        for block in section.select("h2, h3"):
            title = block.get_text(strip=True)
            # find nearest "Read more" link following the header
            read_more = None
            # search siblings and next elements for a link containing 'Read more'
            for a in block.find_all_next("a", href=True):
                if a.get_text(strip=True).lower().startswith("read more"):
                    read_more = a
                    break
                # stop if we reach the next major header to avoid leaking into next section
                if a.find_previous(lambda tag: tag == block) is None and a.find_previous(["h2", "h3"]) not in (None, block):
                    break
            if read_more is None:
                # also accept the header itself as a link if it has href
                header_link = block.find("a", href=True)
                if header_link is not None:
                    read_more = header_link
            if read_more is None:
                continue
            url = absolutize(BASE_URL, read_more["href"]).split("#")[0]
            # attempt to capture the short summary in the next paragraph
            summary = None
            p = block.find_next("p")
            if p is not None:
                summary = p.get_text(" ", strip=True)
            candidates.append({"title": title, "url": url, "summary": summary or ""})

    # de-duplicate by url preserving order
    seen = set()
    unique: List[Dict[str, str]] = []
    for c in candidates:
        if c["url"] in seen:
            continue
        seen.add(c["url"])
        unique.append(c)
    return unique


def extract_article_content(url: str) -> str:
    html = fetch_html(url)
    soup = BeautifulSoup(html, "lxml")

    # Common WordPress content containers
    selectors = [
        ".entry-content",
        "article .entry-content",
        "main article",
        "main .post",
        "article",
        "#content",
        ".site-content",
    ]

    best = None
    best_len = 0
    for sel in selectors:
        for node in soup.select(sel):
            text = node.get_text("\n", strip=True)
            # heuristic: pick the longest meaningful block
            l = len(text)
            if l > best_len:
                best_len = l
                best = text
    if best is None or best_len < 80:
        # fallback to full page text
        best = soup.get_text("\n", strip=True)

    # normalize excessive blank lines
    best = re.sub(r"\n{3,}", "\n\n", best)
    return best


def write_outputs(resources: List[Dict[str, str]]) -> None:
    jsonl_path = Path(OUTPUT_JSONL)
    md_path = Path(OUTPUT_MD)

    with jsonl_path.open("w", encoding="utf-8") as jf, md_path.open("w", encoding="utf-8") as mf:
        mf.write(f"# Business & Mission — Resources (Full)\n\n")
        mf.write(f"Source: {BASE_URL}\n\n")
        for idx, res in enumerate(resources, start=1):
            rec = {
                "id": f"resource-{idx}",
                "title": res.get("title", "").strip(),
                "summary": res.get("summary", "").strip(),
                "url": res.get("url", ""),
                "source": res.get("url", ""),
                "text": res.get("content", "").strip(),
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            mf.write(f"## {rec['title']}\n")
            if rec["summary"]:
                mf.write(rec["summary"] + "\n\n")
            mf.write(f"Source: {rec['url']}\n\n")
            mf.write(rec["text"] + "\n\n")


def main() -> int:
    try:
        index_html = fetch_html(BASE_URL)
        resources = extract_resources_list(index_html)
        if not resources:
            print("No resources discovered on the page.")
            return 1
        for res in resources:
            try:
                res["content"] = extract_article_content(res["url"]) or ""
            except Exception as e:
                res["content"] = f"[Error fetching article: {e}]"
        write_outputs(resources)
        print(f"Wrote {OUTPUT_JSONL} and {OUTPUT_MD}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

