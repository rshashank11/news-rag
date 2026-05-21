import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

XML_FOLDER = Path("data/xml")
OUTPUT_FILE = Path("sakal.json")


def clean_text(value: str | None) -> str:
    """
    Clean one XML text value before storing it in JSON.

    Example:
    "  Pune\\n Market   Yard  " becomes "Pune Market Yard".
    """
    if value is None:
        return ""

    value = re.sub(r"\s+", " ", value)
    value = value.strip()
    value = value.strip(" ,;|")
    return value


def clean_body_text(value: str | None) -> str:
    """
    Clean the main Sakal article body.

    Sakal XML bodies can include media IDs and layout separators that are not
    article text. We remove those before embedding.

    Examples removed:
    - AKL26B15985
    - Associated Media Ids : AMG26B06812
    - -----------------------------
    """
    if value is None:
        return ""

    value = value.replace("\r\n", "\n")
    value = value.replace("\r", "\n")

    value = re.sub(
        r"(?m)^\s*[A-Z]{2,5}[0-9०-९]{2}[A-Z][0-9०-९]{5}(?:\s*,\s*[A-Z]{2,5}[0-9०-९]{2}[A-Z][0-9०-९]{5})*\s*$",
        "",
        value,
    )

    value = re.sub(
        r"\s*Associated\s+Media\s+Ids\s*:.*\Z",
        "",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )

    value = re.sub(r"-{5,}", " ", value)

    return clean_text(value)


def format_date_to_iso(value):
    """
    Convert NewsML dates into normal API date strings.

    Example:
    "20260401" becomes "2026-04-01".
    """
    value = clean_text(value)

    if not re.fullmatch(r"\d{8}", value):
        return None

    return f"{value[0:4]}-{value[4:6]}-{value[6:8]}"


def format_date_to_number(value):
    """
    Convert NewsML dates into Pinecone filter numbers.

    Example:
    "20260401" becomes 20260401.
    """
    value = clean_text(value)

    if not re.fullmatch(r"\d{8}", value):
        return None

    return int(value)


def get_text(root, path, default=""):
    """
    Read cleaned text from an XML path.

    Example:
    get_text(root, ".//HeadLine") returns the article headline.
    """
    element = root.find(path)

    if element is None or element.text is None:
        return default

    return clean_text(element.text)


def get_raw_text(root, path, default=""):
    """
    Read raw XML text without cleaning it first.

    We use this for article body text because body cleanup needs line boundaries
    to remove media ID lines correctly.
    """
    element = root.find(path)

    if element is None or element.text is None:
        return default

    return element.text


def get_property_value(root, formal_name, default=""):
    """
    Read a NewsML Property value by FormalName.

    Example:
    FormalName="Location" returns the article location metadata.
    """
    element = root.find(f".//Property[@FormalName='{formal_name}']")

    if element is None:
        return default

    value = element.attrib.get("Value")
    return clean_text(value)


def get_source(root, default=""):
    """
    Read the provider/source code from the XML.

    Example:
    This may return a source value like "sak".
    """
    element = root.find(".//Provider/Party")

    if element is None:
        return default

    source = element.attrib.get("FormalName")
    return clean_text(source)


def get_keywords(root) -> list[str]:
    """
    Convert the XML keyword line into a list.

    Example:
    "Pune, Market Yard, Vegetables" becomes
    ["Pune", "Market Yard", "Vegetables"].
    """
    keyword_text = get_text(root, ".//KeywordLine")
    raw_keywords = keyword_text.split(",")
    keywords = []

    for keyword in raw_keywords:
        cleaned_keyword = clean_text(keyword)

        if cleaned_keyword:
            keywords.append(cleaned_keyword)

    return keywords


def convert_xml_file_to_json(xml_file):
    """
    Convert one Sakal XML file into one normalized article dictionary.

    This is the shape later used by ingest_sakal.py for chunking, embedding,
    metadata, and Pinecone upload.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    published_date_raw = get_text(root, ".//DateLine")
    created_date_raw = get_text(root, ".//FirstCreated")
    slug = get_text(root, ".//SlugLine")

    article = {
        "id": get_text(root, ".//NewsItemId"),
        "headline": get_text(root, ".//HeadLine"),
        "slug": slug,
        "body": clean_body_text(get_raw_text(root, ".//DataContent")),
        "summary": slug,
        "keywords": get_keywords(root),
        "language": "mr",
        "date_published": format_date_to_iso(published_date_raw),
        "date_published_yyyymmdd": format_date_to_number(published_date_raw),
        "date_created": format_date_to_iso(created_date_raw),
        "date_created_yyyymmdd": format_date_to_number(created_date_raw),
        "location": get_property_value(root, "Location"),
        "edition": get_property_value(root, "Edition"),
        "source": get_source(root),
        "entities": {
            "places": [],
            "people": [],
            "organizations": [],
        },
        "raw": {
            "xml_id": get_text(root, ".//PublicIdentifier"),
            "source_file": str(xml_file),
        },
    }

    return article


def convert_xml_folder_to_json(xml_folder, output_file):
    """
    Convert a folder of Sakal XML files into one JSON array file.

    Example:
    data/xml/*.xml becomes sakal.json.
    """
    articles = []

    for xml_file in sorted(xml_folder.glob("*.xml")):
        article = convert_xml_file_to_json(xml_file)
        articles.append(article)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(articles, file, ensure_ascii=False, indent=2)

    return articles


if __name__ == "__main__":
    converted_articles = convert_xml_folder_to_json(XML_FOLDER, OUTPUT_FILE)
    print(f"Converted {len(converted_articles)} XML files into {OUTPUT_FILE}")
