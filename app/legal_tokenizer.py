import re

# Compound legal terms that should be treated as single BM25 tokens.
# Listed longest-first so multi-word phrases are matched before sub-phrases.
# All entries are lowercase; matching is case-insensitive.
LEGAL_COMPOUND_TERMS: list[str] = [
    # Statutes — full names
    "prevention of money laundering act",
    "insolvency and bankruptcy code",
    "unlawful activities prevention act",
    "narcotic drugs and psychotropic substances act",
    "foreign exchange management act",
    "securities and exchange board of india",
    "protection of children from sexual offences",
    "bharatiya nyaya sanhita",
    "bharatiya nagarik suraksha sanhita",
    "bharatiya sakshya adhiniyam",
    "code of criminal procedure",
    "code of civil procedure",
    "information technology act",
    "right to information act",
    "armed forces special powers act",
    "scheduled castes and scheduled tribes act",
    "maintenance and welfare of parents and senior citizens act",
    "consumer protection act",
    "competition commission of india",
    "national green tribunal",
    "national company law appellate tribunal",
    "national company law tribunal",
    "debt recovery tribunal",
    # Courts
    "supreme court of india",
    "high court of delhi",
    "high court of bombay",
    "high court of madras",
    "high court of calcutta",
    "high court of allahabad",
    "high court of kerala",
    "high court of gujarat",
    "high court of karnataka",
    "high court of punjab and haryana",
    "high court of rajasthan",
    "high court of madhya pradesh",
    "high court of patna",
    "high court of gauhati",
    "high court of orissa",
    "high court of jharkhand",
    "high court of himachal pradesh",
    "high court of uttarakhand",
    "high court of andhra pradesh",
    "high court of telangana",
    "high court of chhattisgarh",
    "supreme court",
    "high court",
    "district court",
    "sessions court",
    "magistrate court",
    "family court",
    "special court",
    "fast track court",
    # Investigative agencies / bodies
    "enforcement directorate",
    "central bureau of investigation",
    "national investigation agency",
    "central vigilance commission",
    "income tax department",
    "directorate of revenue intelligence",
    "serious fraud investigation office",
    "financial intelligence unit",
    # Legal process terms
    "anticipatory bail",
    "bail application",
    "bail order",
    "regular bail",
    "interim bail",
    "transit bail",
    "default bail",
    "statutory bail",
    "special leave petition",
    "public interest litigation",
    "writ petition",
    "criminal appeal",
    "civil appeal",
    "transfer petition",
    "review petition",
    "curative petition",
    "first information report",
    "charge sheet",
    "money laundering",
    "tax evasion",
    "insider trading",
    "contempt of court",
    "contempt petition",
    "habeas corpus",
    "mandamus petition",
    "interim order",
    "stay order",
    "injunction order",
    "ad interim stay",
    "attachment order",
    "search and seizure",
    "look out circular",
    "non-bailable warrant",
    "bailable warrant",
    "arrest warrant",
    "remand order",
    "judicial custody",
    "police custody",
    "constitution bench",
    "larger bench",
    "full bench",
    "division bench",
    "single bench",
    "principal bench",
    # Judicial appointments / HR
    "chief justice of india",
    "chief justice",
    "senior advocate",
    "solicitor general",
    "additional solicitor general",
    "attorney general",
    "advocate general",
    "amicus curiae",
    "legal aid",
    # Insolvency terms
    "resolution professional",
    "insolvency resolution process",
    "corporate insolvency resolution process",
    "liquidation proceedings",
    "committee of creditors",
    "resolution plan",
    "moratorium period",
]

# Sort longest-first to prevent partial match of shorter sub-phrases
_SORTED_TERMS = sorted(LEGAL_COMPOUND_TERMS, key=len, reverse=True)

# Build a single regex that matches any of the compound terms (case-insensitive)
_TERM_PATTERN = re.compile(
    "|".join(re.escape(term) for term in _SORTED_TERMS),
    re.IGNORECASE,
)


def apply_legal_tokenizer(text: str) -> str:
    """
    Replace spaces inside compound legal terms with underscores.

    This preserves multi-word legal phrases as single BM25 tokens so that a
    query for "Supreme Court" scores against chunks that contain exactly those
    two words together, not just either word individually.

    Example:
        "Supreme Court granted bail" → "supreme_court granted bail"
        "Enforcement Directorate filed PMLA charges" →
            "enforcement_directorate filed PMLA charges"

    Must be applied identically at both corpus-fit time and query-encode time
    so the token space is consistent.
    """
    def _replace(match: re.Match) -> str:
        return match.group(0).lower().replace(" ", "_")

    return _TERM_PATTERN.sub(_replace, text)
