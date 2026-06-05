import re

# ---------------------------------------------------------------------------
# Court patterns
# Each entry is (canonical_name, compiled_regex).
# Regexes use word boundaries and are case-insensitive.
# ---------------------------------------------------------------------------

_COURT_DEFS: list[tuple[str, str]] = [
    ("Supreme Court", r"\bSupreme Court(?:\s+of India)?\b"),
    ("Delhi High Court", r"\b(?:Delhi|New Delhi)\s+High Court\b|\bHigh Court of Delhi\b"),
    ("Bombay High Court", r"\bBombay High Court\b|\bHigh Court of Bombay\b"),
    ("Madras High Court", r"\bMadras High Court\b|\bHigh Court of Madras\b"),
    ("Calcutta High Court", r"\bCalcutta High Court\b|\bHigh Court of Calcutta\b"),
    ("Allahabad High Court", r"\bAllahabad High Court\b|\bHigh Court of Allahabad\b"),
    ("Kerala High Court", r"\bKerala High Court\b|\bHigh Court of Kerala\b"),
    ("Gujarat High Court", r"\bGujarat High Court\b|\bHigh Court of Gujarat\b"),
    ("Karnataka High Court", r"\bKarnataka High Court\b|\bHigh Court of Karnataka\b"),
    ("Punjab and Haryana High Court", r"\bPunjab\s+and\s+Haryana High Court\b|\bHigh Court of Punjab and Haryana\b"),
    ("Rajasthan High Court", r"\bRajasthan High Court\b|\bHigh Court of Rajasthan\b"),
    ("Madhya Pradesh High Court", r"\bMadhya Pradesh High Court\b|\bHigh Court of Madhya Pradesh\b|\bMP High Court\b"),
    ("Patna High Court", r"\bPatna High Court\b|\bHigh Court of Patna\b"),
    ("Gauhati High Court", r"\bGauhati High Court\b|\bHigh Court of Gauhati\b"),
    ("Orissa High Court", r"\bOrissa High Court\b|\bOdisha High Court\b|\bHigh Court of Orissa\b"),
    ("Jharkhand High Court", r"\bJharkhand High Court\b|\bHigh Court of Jharkhand\b"),
    ("Himachal Pradesh High Court", r"\bHimachal Pradesh High Court\b|\bHigh Court of Himachal Pradesh\b|\bHP High Court\b"),
    ("Uttarakhand High Court", r"\bUttarakhand High Court\b|\bHigh Court of Uttarakhand\b|\bNainital High Court\b"),
    ("Andhra Pradesh High Court", r"\bAndhra Pradesh High Court\b|\bHigh Court of Andhra Pradesh\b|\bAP High Court\b"),
    ("Telangana High Court", r"\bTelangana High Court\b|\bHigh Court of Telangana\b"),
    ("Chhattisgarh High Court", r"\bChhattisgarh High Court\b|\bHigh Court of Chhattisgarh\b"),
    ("NCLT", r"\bNCLT\b|\bNational Company Law Tribunal\b"),
    ("NCLAT", r"\bNCLAT\b|\bNational Company Law Appellate Tribunal\b"),
    ("NGT", r"\bNGT\b|\bNational Green Tribunal\b"),
    ("SAT", r"\bSAT\b|\bSecurities Appellate Tribunal\b"),
    ("DRT", r"\bDRT\b|\bDebt Recovery Tribunal\b"),
    ("DRAT", r"\bDRAT\b|\bDebt Recovery Appellate Tribunal\b"),
]

COURT_PATTERNS: list[tuple[str, re.Pattern]] = [
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in _COURT_DEFS
]

# Known court canonical names (lowercase) for quick membership testing
KNOWN_COURT_NAMES: set[str] = {name.lower() for name, _ in _COURT_DEFS}

# Lowercase → canonical-cased name map for building exact Pinecone filter values
CANONICAL_COURT_MAP: dict[str, str] = {name.lower(): name for name, _ in _COURT_DEFS}

# ---------------------------------------------------------------------------
# Statute abbreviations and full names
# ---------------------------------------------------------------------------

STATUTE_ABBREVIATIONS: list[str] = [
    "PMLA",
    "IBC",
    "UAPA",
    "NDPS",
    "IPC",
    "BNS",
    "BNSS",
    "BSA",
    "POCSO",
    "FEMA",
    "SEBI",
    "CPC",
    "CrPC",
    "IT Act",
    "RTI",
    "AFSPA",
    "FCRA",
    "RERA",
    "GST",
    "PCR Act",
    "SC/ST Act",
    "MTP Act",
    "Dowry Prohibition Act",
    "Contempt of Courts Act",
    "NIA Act",
    "TADA",
    "Prevention of Corruption Act",
    "PCA",
    "Money Laundering",
    "Insolvency and Bankruptcy Code",
    "Prevention of Money Laundering Act",
    "Unlawful Activities Prevention Act",
    "Narcotic Drugs and Psychotropic Substances Act",
    "Foreign Exchange Management Act",
    "Code of Criminal Procedure",
    "Bharatiya Nyaya Sanhita",
    "Bharatiya Nagarik Suraksha Sanhita",
]

_STATUTE_PATTERN = re.compile(
    "|".join(
        r"\b" + re.escape(s) + r"\b"
        for s in sorted(STATUTE_ABBREVIATIONS, key=len, reverse=True)
    ),
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Case type keywords
# Each entry is (label, list_of_trigger_phrases).
# ---------------------------------------------------------------------------

CASE_TYPE_KEYWORDS: dict[str, list[str]] = {
    "bail": ["bail", "anticipatory bail", "regular bail", "default bail", "statutory bail"],
    "PIL": ["PIL", "public interest litigation"],
    "SLP": ["SLP", "special leave petition"],
    "contempt": ["contempt of court", "contempt petition", "contempt proceedings"],
    "arbitration": ["arbitration", "arbitral award", "arbitral tribunal"],
    "writ": ["writ petition", "habeas corpus", "mandamus", "certiorari", "quo warranto"],
    "appeal": ["criminal appeal", "civil appeal", "first appeal", "letters patent appeal"],
    "insolvency": ["insolvency", "IBC", "CIRP", "corporate insolvency", "liquidation", "NCLT"],
    "review": ["review petition", "curative petition"],
    "transfer": ["transfer petition", "transfer of case"],
    "FIR": ["FIR", "first information report"],
    "chargesheet": ["charge sheet", "chargesheet", "filed charge"],
    "arrest": ["arrest", "arrested", "detention"],
    "search_seizure": ["search and seizure", "searched premises", "raid"],
    "acquittal": ["acquitted", "acquittal"],
    "conviction": ["convicted", "conviction", "sentenced"],
    "stay": ["stay order", "stay granted", "interim stay", "ad interim stay"],
    "appointment": ["judicial appointment", "judge appointment", "elevation", "collegium"],
}

_CASE_TYPE_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        label,
        re.compile(
            "|".join(r"\b" + re.escape(phrase) + r"\b" for phrase in phrases),
            re.IGNORECASE,
        ),
    )
    for label, phrases in CASE_TYPE_KEYWORDS.items()
]


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

def extract_courts(text: str) -> list[str]:
    """
    Extract court names mentioned in the text.

    Returns a deduplicated list of canonical court names in order of first
    appearance.

    Example:
        "The Supreme Court and Delhi High Court both..." →
            ["Supreme Court", "Delhi High Court"]
    """
    found: list[str] = []
    seen: set[str] = set()

    for name, pattern in COURT_PATTERNS:
        if pattern.search(text) and name not in seen:
            found.append(name)
            seen.add(name)

    return found


def extract_statutes(text: str) -> list[str]:
    """
    Extract statute names and abbreviations mentioned in the text.

    Returns deduplicated matches in their canonical casing from the source text.

    Example:
        "Charged under PMLA and IBC..." → ["PMLA", "IBC"]
    """
    matches = _STATUTE_PATTERN.findall(text)
    seen: set[str] = set()
    found: list[str] = []

    for match in matches:
        key = match.strip().lower()
        if key not in seen:
            found.append(match.strip())
            seen.add(key)

    return found


def extract_case_type(text: str) -> list[str]:
    """
    Extract case type labels from the text.

    Returns a deduplicated list of matched label strings.

    Example:
        "Filed anticipatory bail application before the Supreme Court..." →
            ["bail", "SLP"] (if SLP also present)
    """
    found: list[str] = []

    for label, pattern in _CASE_TYPE_PATTERNS:
        if pattern.search(text):
            found.append(label)

    return found
