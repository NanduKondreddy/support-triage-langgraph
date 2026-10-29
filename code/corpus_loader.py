from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Document:
    domain: str
    product_area: str
    path: str
    title: str
    text: str


AREA_ALIASES = {
    "hackerrank_community": "community",
    "general-help": "general_support",
    "privacy-and-legal": "privacy",
    "pro-and-max-plans": "billing",
    "team-and-enterprise-plans": "team_enterprise",
    "claude-api-and-console": "api_console",
    "claude-mobile-apps": "mobile",
    "identity-management-sso-jit-scim": "identity_management",
    "amazon-bedrock": "api_console",
    "small-business": "small_business",
    "consumer": "consumer",
}


def clean_markdown(text: str) -> str:
    text = re.sub(r"\A---\s*\n.*?\n---\s*\n", "", text, flags=re.S)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def title_from(path: Path, text: str) -> str:
    in_frontmatter = False
    for line in text.splitlines():
        if line.strip() == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter:
            if line.lower().startswith("title:"):
                title = line.split(":", 1)[1].strip().strip('"')
                if title:
                    return title[:120]
            continue
        stripped = line.strip(" #\t")
        if stripped:
            return stripped[:120]
    return path.stem.replace("-", " ").replace("_", " ").title()


def product_area_for(path: Path, domain: str) -> str:
    parts = [p for p in path.parts]
    try:
        domain_index = parts.index(domain)
    except ValueError:
        return "general_support"
    after = parts[domain_index + 1 :]
    if not after:
        return "general_support"
    first = after[0]
    if first == "support" and len(after) > 1:
        first = after[1]
    return AREA_ALIASES.get(first, first.replace("-", "_"))


def load_corpus(data_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for domain in ("hackerrank", "claude", "visa"):
        domain_dir = data_dir / domain
        if not domain_dir.exists():
            continue
        for path in sorted(domain_dir.rglob("*.md")):
            try:
                raw = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"[Warning] Skipping {path}: {e}")
                continue
            text = clean_markdown(raw)
            if not text:
                continue
            rel = path.relative_to(data_dir).as_posix()
            docs.append(
                Document(
                    domain=domain,
                    product_area=product_area_for(path, domain),
                    path=rel,
                    title=title_from(path, raw),
                    text=text[:6000],
                )
            )
    return docs
