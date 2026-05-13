import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

XML_FOLDER = Path("data/xml")
OUTPUT_FILE = Path("sakal.json")

def clean_text(value: str | None) -> str:
    if value is None: 
        return ""
    
    value = re.sub(r"\s+", " ", value)
    value = value.strip()
    value = value.strip(" ,;|")
    return value


def clean_body_text(value: str | None) -> str:
    if value is None:
        return ""

    # Keep line breaks temporarily because some junk appears on its own line.
    value = value.replace("\r\n", "\n")
    value = value.replace("\r", "\n")

    # Remove standalone media ID lines like:
    # AKL26B15985
    # AKL26B15985, AKL26B15986, AKL26B15987
    # Some IDs use Marathi/Devanagari digits, so ०-९ is included too.
    value = re.sub(
        r"(?m)^\s*[A-Z]{2,5}[0-9०-९]{2}[A-Z][0-9०-९]{5}(?:\s*,\s*[A-Z]{2,5}[0-9०-९]{2}[A-Z][0-9०-९]{5})*\s*$",
        "",
        value,
    )

    # Remove trailing media references like:
    # Associated Media Ids : AMG26B06812
    # This also removes cases where the label exists but the ID is blank.
    # Once this label appears, everything after it is media metadata, not article text.
    value = re.sub(
        r"\s*Associated\s+Media\s+Ids\s*:.*\Z",
        "",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove newspaper layout separator lines like:
    # -----------------------------
    value = re.sub(r"-{5,}", " ", value)

    # Reuse the normal text cleaner after body-specific cleanup is done.
    return clean_text(value)


def format_date_to_iso(value):
    value = clean_text(value)
    
    if not re.fullmatch(r"\d{8}", value):
        return None
    
    return f"{value[0:4]}-{value[4:6]}-{value[6:8]}"

def format_date_to_number(value):
    value = clean_text(value)
    
    if not re.fullmatch(r"\d{8}", value):
        return None
    
    return int(value)

def get_text(root, path, default=""):
    element = root.find(path)
    
    if element is None or element.text is None:
        return default
    
    return clean_text(element.text)


def get_raw_text(root, path, default=""):
    element = root.find(path)

    if element is None or element.text is None:
        return default

    return element.text

# This function gets values from XML Property tags.
def get_property_value(root, formal_name, default=""):
    # Find a Property tag with a matching FormalName.
    element = root.find(f".//Property[@FormalName='{formal_name}']")

    # If the property does not exist, return default value.
    if element is None:
        return default

    # Get the Value attribute from the Property tag.
    value = element.attrib.get("Value")

    # Clean and return the value.
    return clean_text(value)


# This function gets source name like sak.
def get_source(root, default=""):
    # Find the Party tag inside Provider.
    element = root.find(".//Provider/Party")

    # If the tag does not exist, return default value.
    if element is None:
        return default

    # Get the FormalName attribute.
    source = element.attrib.get("FormalName")

    # Clean and return the source.
    return clean_text(source)


# This function extracts keywords as a list.
def get_keywords(root):
    # Get text from KeywordLine.
    keyword_text = get_text(root, ".//KeywordLine")

    # Split keyword text by comma.
    raw_keywords = keyword_text.split(",")

    # Create an empty list for cleaned keywords.
    keywords = []

    # Go through each keyword.
    for keyword in raw_keywords:
        # Clean the keyword.
        cleaned_keyword = clean_text(keyword)

        # Add keyword only if it is not empty.
        if cleaned_keyword:
            keywords.append(cleaned_keyword)

    # Return final keyword list.
    return keywords

# This function converts one XML file into one JSON-like dictionary.
def convert_xml_file_to_json(xml_file):
    # Read and parse the XML file.
    tree = ET.parse(xml_file)

    # Get the root tag, usually NewsML.
    root = tree.getroot()

    # Get published date from DateLine.
    published_date_raw = get_text(root, ".//DateLine")

    # Get created date from FirstCreated.
    created_date_raw = get_text(root, ".//FirstCreated")

    # Get slug text once, because we use it in two places.
    slug = get_text(root, ".//SlugLine")

    # Create final article dictionary.
    article = {
        # Article ID.
        "id": get_text(root, ".//NewsItemId"),

        # Article headline.
        "headline": get_text(root, ".//HeadLine"),

        # Short internal title/description.
        "slug": slug,

        # Main article body.
        "body": clean_body_text(get_raw_text(root, ".//DataContent")),

        # Temporary summary. Later you can generate better summary using AI.
        "summary": slug,

        # Keywords as a list.
        "keywords": get_keywords(root),

        # Article language. We use mr because the article text is Marathi.
        "language": "mr",

        # Human-readable published date.
        "date_published": format_date_to_iso(published_date_raw),

        # Number date for faster vector DB filtering.
        "date_published_yyyymmdd": format_date_to_number(published_date_raw),

        # Human-readable created date.
        "date_created": format_date_to_iso(created_date_raw),

        # Number created date.
        "date_created_yyyymmdd": format_date_to_number(created_date_raw),

        # Location from Property tag.
        "location": get_property_value(root, "Location"),

        # Edition from Property tag.
        "edition": get_property_value(root, "Edition"),

        # Source/provider name.
        "source": get_source(root),

        # Empty for now. Later AI/NER can fill these.
        "entities": {
            "places": [],
            "people": [],
            "organizations": [],
        },

        # Raw XML-related information.
        "raw": {
            "xml_id": get_text(root, ".//PublicIdentifier"),
            "source_file": str(xml_file),
        },
    }

    # Return the final article.
    return article

# This function converts all XML files in a folder.
def convert_xml_folder_to_json(xml_folder, output_file):
    # Create an empty list to store all articles.
    articles = []

    # Find every .xml file in the folder, sorted by name.
    for xml_file in sorted(xml_folder.glob("*.xml")):
        # Convert one XML file into one article dictionary.
        article = convert_xml_file_to_json(xml_file)

        # Add the article to the list.
        articles.append(article)

    # Create output folder if needed.
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Open output JSON file in write mode.
    with output_file.open("w", encoding="utf-8") as file:
        # Write all articles into JSON file.
        json.dump(articles, file, ensure_ascii=False, indent=2)

    # Return articles in case we want to use them in Python.
    return articles


# This runs only when we execute this file directly.
if __name__ == "__main__":
    # Convert XML folder into JSON file.
    converted_articles = convert_xml_folder_to_json(XML_FOLDER, OUTPUT_FILE)

    # Print how many files were converted.
    print(f"Converted {len(converted_articles)} XML files into {OUTPUT_FILE}")
    
