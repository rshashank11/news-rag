#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import datetime as dt
import gzip
import json
import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Hardcoded test cases
# ============================================================

HARDCODED_SESSIONS = [
    # --------------------------------------------------------
    # ENGLISH SESSION 1: Market Yard vegetable price rise
    # --------------------------------------------------------
    {
        "language": "English",
        "persona": "Pune local reader",
        "need": "wants a practical local market update before shopping",
        "expected_source_ids": [
            "YAR26B03985",
            "YAR26B03996",
            "YAR26B03972",
        ],
        "expected_primary_article_id": "YAR26B03985",
        "expected_primary_headline": "मटार, टोमॅटो, पावट्याच्या भावात वाढ",
        "expected_primary_date_published": "2026-04-13",
        "expected_topic": "Agriculture",
        "expected_keywords": [
            "मटार",
            "टोमॅटो",
            "पावट्याच्या भावात वाढ",
            "मार्केटयार्ड",
            "१०० ट्रक फळभाज्यांची आवक",
        ],
        "turns": [
            {
                "label": "parent",
                "question": (
                    "In Pune Market Yard, which vegetables became costlier, and why?"
                ),
            },
            {
                "label": "follow_up_1",
                "question": (
                    "In the same update, how many truckloads arrived and what is said about peas?"
                ),
            },
            {
                "label": "follow_up_2",
                "question": (
                    "From that report, give date, place, key vegetables and price trend in short."
                ),
            },
        ],
    },

    # --------------------------------------------------------
    # ENGLISH SESSION 2: RTE admission bribery / civic issue
    # --------------------------------------------------------
    {
        "language": "English",
        "persona": "College student",
        "need": "wants a simple current-affairs explanation about education and corruption",
        "expected_source_ids": [
            "PNE26Y81513",
            "PNE26Y71858",
            "PNE26Y71883",
            "PNE26Y72186",
        ],
        "expected_primary_article_id": "PNE26Y81513",
        "expected_primary_headline": "लाचखोरीनंतर महापालिकेला जाग",
        "expected_primary_date_published": "2026-05-05",
        "expected_topic": "Civic",
        "expected_keywords": [
            "आरटीई प्रवेश",
            "लाचखोरी",
            "पारदर्शक",
            "ऑनलाइन",
            "विनामूल्य",
            "‘आरटीई’ प्रक्रिया",
        ],
        "turns": [
            {
                "label": "parent",
                "question": (
                    "In Pimpri-Chinchwad RTE admissions, what happened after the bribe allegation?"
                ),
            },
            {
                "label": "follow_up_1",
                "question": (
                    "In that case, what was said about online, free and transparent admission?"
                ),
            },
            {
                "label": "follow_up_2",
                "question": (
                    "If anyone asks for money in this RTE process, what are parents told to do?"
                ),
            },
        ],
    },

    # --------------------------------------------------------
    # ENGLISH SESSION 3: Western Ghats insect diversity
    # --------------------------------------------------------
    {
        "language": "English",
        "persona": "Journalism researcher",
        "need": "checks ecological reporting, dates, people and institutions",
        "expected_source_ids": ["PNE26D51162"],
        "expected_primary_article_id": "PNE26D51162",
        "expected_primary_headline": "किटकांच्या ज्ञात विविधतेत ३५ टक्के घट",
        "expected_primary_date_published": "2026-04-26",
        "expected_topic": "Environment",
        "expected_keywords": [
            "पायाभूत सुविधा",
            "फेब्रुवारी",
            "महाराष्ट्र",
            "ऐतिहासिक",
            "भविष्य",
            "संशोधन",
            "पुणे",
        ],
        "turns": [
            {
                "label": "parent",
                "question": (
                    "In the Western Ghats study, what was found about dragonfly and damselfly diversity?"
                ),
            },
            {
                "label": "follow_up_1",
                "question": (
                    "In that report, which threats are linked to the biodiversity decline?"
                ),
            },
            {
                "label": "follow_up_2",
                "question": (
                    "Give the survey period, states, locations and researchers from this report."
                ),
            },
        ],
    },

    # --------------------------------------------------------
    # ENGLISH SESSION 4: ST cleanliness fee
    # --------------------------------------------------------
    {
        "language": "English",
        "persona": "Civic resident",
        "need": "tracks transport services, cleanliness and passenger complaints",
        "expected_source_ids": [
            "PNE26Y80666",
            "PNE26Y80296",
            "PNE26D49029",
            "PNE26D48616",
            "PNE26U67274",
        ],
        "expected_primary_article_id": "PNE26Y80666",
        "expected_primary_headline": "स्वच्छता शुल्काची वसुली तत्काळ थांबवा",
        "expected_primary_date_published": "2026-05-04",
        "expected_topic": "Civic",
        "expected_keywords": [
            "एसटी प्रशास",
            "नागरिक",
            "प्रतिक्रिया",
            "स्वारगेट",
            "वाकडेवाडी",
            "स्वच्छता शुल्क",
        ],
        "turns": [
            {
                "label": "parent",
                "question": (
                    "Why are passengers upset about the two-rupee ST cleanliness fee?"
                ),
            },
            {
                "label": "follow_up_1",
                "question": (
                    "What problems are mentioned at bus stands like Swargate and Wakdewadi?"
                ),
            },
            {
                "label": "follow_up_2",
                "question": (
                    "What action are passengers demanding from ST authorities in this issue?"
                ),
            },
        ],
    },

    # --------------------------------------------------------
    # ENGLISH SESSION 5: Mango pulp health risk
    # --------------------------------------------------------
    {
        "language": "English",
        "persona": "Food safety reader",
        "need": "wants practical public-health advice from local reporting",
        "expected_source_ids": ["SVW26A00195"],
        "expected_primary_article_id": "SVW26A00195",
        "expected_primary_headline": "रस्त्याकडेच्या ‘मँगो पल्प’ने आरोग्याला धोका",
        "expected_primary_date_published": "2026-04-28",
        "expected_topic": "Health",
        "expected_keywords": [
            "काळेवाडी",
            "दुकाने",
            "स्वच्छता",
            "मँगो पल्प",
            "आंबा गर",
        ],
        "turns": [
            {
                "label": "parent",
                "question": (
                    "What health risk is reported in Kalewadi about fake or low-quality mango pulp?"
                ),
            },
            {
                "label": "follow_up_1",
                "question": (
                    "What illness or symptoms are linked to old pulp, additives and poor hygiene?"
                ),
            },
            {
                "label": "follow_up_2",
                "question": (
                    "Give date, place, health risks and consumer advice from that report."
                ),
            },
        ],
    },

    # --------------------------------------------------------
    # MARATHI SESSION 1: Watermelon demand in Ahilyanagar
    # --------------------------------------------------------
    {
        "language": "Marathi",
        "persona": "मराठी वाचक",
        "need": "स्थानिक बाजारभाव आणि उन्हाळ्यातील फळांच्या मागणीचा आढावा हवा आहे",
        "expected_source_ids": [
            "PNE26J26124",
            "PNE26Y72016",
        ],
        "expected_primary_article_id": "PNE26J26124",
        "expected_primary_headline": "कलिंगडाला मागणी वाढली",
        "expected_primary_date_published": "2026-04-14",
        "expected_topic": "Agriculture",
        "expected_keywords": [
            "अहिल्यानगर",
            "कलिंगड",
            "पाऊस",
            "ऊस",
            "गर",
        ],
        "turns": [
            {
                "label": "parent",
                "question": (
                    "अहिल्यानगरमध्ये कलिंगड-खरबुजाच्या मागणी आणि दरांबद्दल काय सांगितले आहे?"
                ),
            },
            {
                "label": "follow_up_1",
                "question": (
                    "मागणी असूनही दर का वाढले नाहीत, आणि सध्याचे भाव काय आहेत?"
                ),
            },
            {
                "label": "follow_up_2",
                "question": (
                    "या बातमीत तारीख, ठिकाण, प्रभावित पिके आणि शेतकऱ्यांची अडचण थोडक्यात द्या."
                ),
            },
        ],
    },

    # --------------------------------------------------------
    # MARATHI SESSION 2: ST cleanliness fee
    # --------------------------------------------------------
    {
        "language": "Marathi",
        "persona": "पालक",
        "need": "वाहतूक सेवा, स्वच्छता आणि नागरिकांच्या अडचणी समजून घ्यायच्या आहेत",
        "expected_source_ids": [
            "PNE26Y80666",
            "PNE26Y80296",
            "PNE26D49029",
            "PNE26D48616",
            "PNE26U67274",
        ],
        "expected_primary_article_id": "PNE26Y80666",
        "expected_primary_headline": "स्वच्छता शुल्काची वसुली तत्काळ थांबवा",
        "expected_primary_date_published": "2026-05-04",
        "expected_topic": "Civic",
        "expected_keywords": [
            "एसटी प्रशास",
            "नागरिक",
            "प्रतिक्रिया",
            "स्वारगेट",
            "वाकडेवाडी",
            "स्वच्छता शुल्क",
        ],
        "turns": [
            {
                "label": "parent",
                "question": (
                    "एसटीच्या दोन रुपयांच्या स्वच्छता शुल्कावर प्रवासी नाराज का आहेत?"
                ),
            },
            {
                "label": "follow_up_1",
                "question": (
                    "स्वारगेट आणि वाकडेवाडीसारख्या ठिकाणी नेमक्या कोणत्या समस्या सांगितल्या आहेत?"
                ),
            },
            {
                "label": "follow_up_2",
                "question": (
                    "या प्रकरणात प्रवासी एसटी प्रशासनाकडून नेमकी कोणती कारवाई मागत आहेत?"
                ),
            },
        ],
    },

    # --------------------------------------------------------
    # MARATHI SESSION 3: Madhuri Misal letter to police
    # --------------------------------------------------------
    {
        "language": "Marathi",
        "persona": "महाविद्यालयीन विद्यार्थी",
        "need": "राजकीय आणि प्रशासकीय घडामोडी सोप्या भाषेत समजून घ्यायच्या आहेत",
        "expected_source_ids": ["PNE26Y81432"],
        "expected_primary_article_id": "PNE26Y81432",
        "expected_primary_headline": "राज्यमंत्र्यांचे पत्र; पोलिसांची धावपळ",
        "expected_primary_date_published": "2026-05-05",
        "expected_topic": "Politics",
        "expected_keywords": [
            "पोलिसांना पत्र",
            "माधुरी मिसाळ",
            "कंपनीत तपासणी",
            "बहुराष्ट्रीय कंपनी",
            "कंपनी",
            "गैरप्रकार",
        ],
        "turns": [
            {
                "label": "parent",
                "question": (
                    "हिंजवडी कंपनी प्रकरणात राज्यमंत्र्यांच्या पत्रानंतर नेमके काय घडले?"
                ),
            },
            {
                "label": "follow_up_1",
                "question": (
                    "पत्रात पुरुष विश्रांती कक्ष, धार्मिक विधी आणि महिला कर्मचाऱ्यांबाबत काय मुद्दे होते?"
                ),
            },
            {
                "label": "follow_up_2",
                "question": (
                    "या प्रकरणात तारीख, ठिकाण, संबंधित व्यक्ती आणि तपासाचा निष्कर्ष थोडक्यात द्या."
                ),
            },
        ],
    },

    # --------------------------------------------------------
    # MARATHI SESSION 4: Gokhale Institute workshop
    # --------------------------------------------------------
    {
        "language": "Marathi",
        "persona": "पत्रकारिता संशोधक",
        "need": "शिक्षण धोरण, संस्था, व्यक्ती आणि कार्यक्रम तपशील तपासायचे आहेत",
        "expected_source_ids": ["PNE26D51245"],
        "expected_primary_article_id": "PNE26D51245",
        "expected_primary_headline": "स्वायत्त महाविद्यालयांसाठी गोखले संस्थेत कार्यशाळा",
        "expected_primary_date_published": "2026-04-26",
        "expected_topic": "Education",
        "expected_keywords": [
            "महाराष्ट्र",
            "संशोधन",
            "शिक्षण",
            "मुंबई",
            "गोखले संस्थेत स्वायत्त महाविद्यालयांसाठी कार्यशाळेचे आयोजन",
        ],
        "turns": [
            {
                "label": "parent",
                "question": (
                    "गोखले संस्थेतील स्वायत्त महाविद्यालय कार्यशाळेबद्दल मुख्य माहिती काय आहे?"
                ),
            },
            {
                "label": "follow_up_1",
                "question": (
                    "चौथे वर्ष अंमलबजावणी आणि २०२६-२७ साठी या कार्यशाळेचा उद्देश काय आहे?"
                ),
            },
            {
                "label": "follow_up_2",
                "question": (
                    "या कार्यशाळेची तारीख, वेळ, ठिकाण आणि आयोजक थोडक्यात द्या."
                ),
            },
        ],
    },

    # --------------------------------------------------------
    # MARATHI SESSION 5: Rain damage compensation report
    # --------------------------------------------------------
    {
        "language": "Marathi",
        "persona": "शेतकरी / ग्रामीण वाचक",
        "need": "अवकाळी पाऊस, पीक नुकसान आणि भरपाई अहवाल समजून घ्यायचा आहे",
        "expected_source_ids": [
            "PNE26D47949",
            "SHD26B03124",
            "PNE26Y72598",
            "MBRU26A13737",
        ],
        "expected_primary_article_id": "PNE26D47949",
        "expected_primary_headline": "पावसामुळे झालेल्या शेतीच्या नुकसानभरपाईचा अहवाल सादर",
        "expected_primary_date_published": "2026-04-18",
        "expected_topic": "Agriculture",
        "expected_keywords": [
            "पाऊस",
            "नुकसानभरपाई",
            "अहवाल",
            "शेती",
            "शेतकऱ्यांना",
        ],
        "turns": [
            {
                "label": "parent",
                "question": (
                    "अवकाळी पावसामुळे किती शेतीचे नुकसान झाले आणि किती शेतकरी बाधित झाले?"
                ),
            },
            {
                "label": "follow_up_1",
                "question": (
                    "गाव/तालुका पातळीवर नुकसान आणि पंचनाम्याबद्दल काय माहिती दिली आहे?"
                ),
            },
            {
                "label": "follow_up_2",
                "question": (
                    "या प्रकरणात मदत, भरपाई आणि पुढील प्रशासनिक कारवाईचे अपडेट काय आहे?"
                ),
            },
        ],
    },
]


# ============================================================
# Data models
# ============================================================

@dataclass
class TestTurn:
    language: str
    persona: str
    need: str
    session_index: int
    turn_index: int
    label: str
    question: str
    expected_source_ids: List[str]
    expected_primary_article_id: str
    expected_primary_headline: str
    expected_primary_date_published: str
    expected_topic: str
    expected_keywords: List[str]


@dataclass
class RetrievedSource:
    source_number: int
    raw: str
    article_id: Optional[str]
    headline: Optional[str]
    published_at: Optional[str]


@dataclass
class TestResult:
    test: TestTurn
    response_type: str
    elapsed: float
    answer: str
    status_code: Optional[int]
    error: Optional[str]
    sources: List[RetrievedSource]
    expected_article_retrieved: bool
    trace: str
    raw_response: Any


# ============================================================
# Helpers
# ============================================================

def clean_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_text(value: str) -> str:
    return clean_spaces(value).lower()


ENGLISH_TO_MARATHI_QUESTION = {
    "In Pune Market Yard, which vegetables became costlier, and why?": "पुणे मार्केटयार्डमध्ये कोणत्या भाज्या महागल्या आणि त्यामागचे कारण काय सांगितले आहे?",
    "In the same update, how many truckloads arrived and what is said about peas?": "त्याच बातमीत किती ट्रक माल आल्याचे सांगितले आहे आणि मटारबद्दल काय म्हटले आहे?",
    "From that report, give date, place, key vegetables and price trend in short.": "त्या अहवालातून तारीख, ठिकाण, मुख्य भाज्या आणि दरातील ट्रेंड थोडक्यात सांगा.",
    "In Pimpri-Chinchwad RTE admissions, what happened after the bribe allegation?": "पिंपरी-चिंचवड आरटीई प्रवेश प्रकरणात लाचखोरीच्या आरोपानंतर नेमके काय घडले?",
    "In that case, what was said about online, free and transparent admission?": "त्या प्रकरणात ऑनलाइन, विनामूल्य आणि पारदर्शक प्रवेशाबद्दल काय सांगितले आहे?",
    "If anyone asks for money in this RTE process, what are parents told to do?": "या आरटीई प्रक्रियेत कोणी पैसे मागितले तर पालकांना काय करायला सांगितले आहे?",
    "In the Western Ghats study, what was found about dragonfly and damselfly diversity?": "पश्चिम घाटातील अभ्यासात ड्रॅगनफ्लाय आणि डॅम्सेलफ्लाय विविधतेबद्दल काय निष्कर्ष नोंदवला आहे?",
    "In that report, which threats are linked to the biodiversity decline?": "त्या अहवालात जैवविविधतेतील घटीशी कोणते धोके जोडले आहेत?",
    "Give the survey period, states, locations and researchers from this report.": "या अहवालातून सर्वेक्षण कालावधी, राज्ये, ठिकाणे आणि संशोधकांची माहिती द्या.",
    "Why are passengers upset about the two-rupee ST cleanliness fee?": "एसटीच्या दोन रुपयांच्या स्वच्छता शुल्कामुळे प्रवासी नाराज का आहेत?",
    "What problems are mentioned at bus stands like Swargate and Wakdewadi?": "स्वारगेट आणि वाकडेवाडीसारख्या बसस्थानकांवर कोणत्या समस्या सांगितल्या आहेत?",
    "What action are passengers demanding from ST authorities in this issue?": "या मुद्द्यावर प्रवासी एसटी प्रशासनाकडून कोणती कारवाई मागत आहेत?",
    "What health risk is reported in Kalewadi about fake or low-quality mango pulp?": "काळेवाडीमध्ये बनावट किंवा निकृष्ट मँगो पल्पमुळे कोणता आरोग्यधोका सांगितला आहे?",
    "What illness or symptoms are linked to old pulp, additives and poor hygiene?": "जुना पल्प, भेसळयुक्त घटक आणि अस्वच्छतेमुळे कोणते आजार किंवा लक्षणे जोडली आहेत?",
    "Give date, place, health risks and consumer advice from that report.": "त्या अहवालातून तारीख, ठिकाण, आरोग्यधोके आणि ग्राहकांसाठी सल्ला द्या.",
}


def test_turn_from_session(
    session: Dict[str, Any],
    session_index: int,
    turn_index: int,
    turn: Dict[str, Any],
    language: str,
    question: str,
    persona: Optional[str] = None,
    need: Optional[str] = None,
) -> TestTurn:
    return TestTurn(
        language=language,
        persona=persona or session["persona"],
        need=need or session["need"],
        session_index=session_index,
        turn_index=turn_index,
        label=turn["label"],
        question=question,
        expected_source_ids=session["expected_source_ids"],
        expected_primary_article_id=session["expected_primary_article_id"],
        expected_primary_headline=session["expected_primary_headline"],
        expected_primary_date_published=session["expected_primary_date_published"],
        expected_topic=session["expected_topic"],
        expected_keywords=session["expected_keywords"],
    )


def build_tests(language_mode: str = "both") -> List[TestTurn]:
    tests: List[TestTurn] = []

    english_sessions = [
        (session_index, session)
        for session_index, session in enumerate(HARDCODED_SESSIONS, start=1)
        if normalize_text(session.get("language", "")) == "english"
    ]
    native_marathi_sessions = [
        (session_index, session)
        for session_index, session in enumerate(HARDCODED_SESSIONS, start=1)
        if normalize_text(session.get("language", "")) == "marathi"
    ]

    for session_index, session in english_sessions:
        for turn_index, turn in enumerate(session["turns"], start=1):
            english_question = turn["question"]
            marathi_question = ENGLISH_TO_MARATHI_QUESTION.get(english_question)

            if not marathi_question:
                raise ValueError(
                    "Missing Marathi mirror for English question: "
                    f"{english_question}"
                )

            if language_mode in {"both", "english"}:
                tests.append(
                    test_turn_from_session(
                        session=session,
                        session_index=session_index,
                        turn_index=turn_index,
                        turn=turn,
                        language="English",
                        question=english_question,
                    )
                )

            if language_mode in {"both", "marathi"}:
                tests.append(
                    test_turn_from_session(
                        session=session,
                        session_index=session_index,
                        turn_index=turn_index,
                        turn=turn,
                        language="Marathi",
                        question=marathi_question,
                        persona="मराठी वाचक",
                        need="इंग्रजी प्रश्नाचा समान अर्थ मराठीत",
                    )
                )

    if language_mode in {"both", "marathi"}:
        for session_index, session in native_marathi_sessions:
            for turn_index, turn in enumerate(session["turns"], start=1):
                tests.append(
                    test_turn_from_session(
                        session=session,
                        session_index=session_index,
                        turn_index=turn_index,
                        turn=turn,
                        language="Marathi",
                        question=turn["question"],
                    )
                )

    return tests


def assert_no_headline_leak(tests: List[TestTurn]) -> None:
    for test in tests:
        headline = normalize_text(test.expected_primary_headline)
        question = normalize_text(test.question)

        if headline and headline in question:
            raise ValueError(
                f"Headline leaked into question for {test.expected_primary_article_id}: "
                f"{test.expected_primary_headline}\nQuestion: {test.question}"
            )


# ============================================================
# Optional sakal.json.gz lookup for resolving retrieved IDs
# ============================================================

def load_source_lookup(data_path: Path) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], List[str]]]:
    headline_to_ids: Dict[str, List[str]] = collections.defaultdict(list)
    date_headline_to_ids: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)

    if not data_path.exists():
        print(f"Warning: data file not found for lookup: {data_path}")
        return {}, {}

    with gzip.open(data_path, "rt", encoding="utf-8") as f:
        raw = json.load(f)

    for item in raw:
        if not isinstance(item, dict):
            continue

        article_id = clean_spaces(str(item.get("id") or ""))
        headline = clean_spaces(str(item.get("headline") or item.get("title") or ""))
        date_published = clean_spaces(str(item.get("date_published") or ""))

        if not article_id or not headline:
            continue

        headline_key = normalize_text(headline)
        headline_to_ids[headline_key].append(article_id)

        if date_published:
            date_headline_to_ids[(date_published, headline_key)].append(article_id)

    return dict(headline_to_ids), dict(date_headline_to_ids)


def validate_source_expectations(
    data_path: Path,
    tests: List[TestTurn],
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if not data_path.exists():
        return [f"Data file not found: {data_path}"], warnings

    with gzip.open(data_path, "rt", encoding="utf-8") as f:
        raw = json.load(f)

    by_id: Dict[str, Dict[str, Any]] = {}

    for item in raw:
        if not isinstance(item, dict):
            continue

        article_id = clean_spaces(str(item.get("id") or ""))

        if article_id:
            by_id[article_id] = item

    seen_primary_ids = set()

    for test in tests:
        for expected_id in test.expected_source_ids:
            if expected_id not in by_id:
                errors.append(
                    f"Missing expected source `{expected_id}` "
                    f"for question: {test.question}"
                )

        primary_id = test.expected_primary_article_id

        if primary_id in seen_primary_ids:
            continue

        seen_primary_ids.add(primary_id)
        primary = by_id.get(primary_id)

        if not primary:
            errors.append(f"Missing primary source `{primary_id}`")
            continue

        actual_headline = clean_spaces(
            str(primary.get("headline") or primary.get("title") or "")
        )
        actual_date = clean_spaces(str(primary.get("date_published") or ""))

        if normalize_text(actual_headline) != normalize_text(test.expected_primary_headline):
            errors.append(
                f"Headline mismatch for `{primary_id}`: expected "
                f"`{test.expected_primary_headline}`, found `{actual_headline}`"
            )

        if actual_date != test.expected_primary_date_published:
            errors.append(
                f"Date mismatch for `{primary_id}`: expected "
                f"`{test.expected_primary_date_published}`, found `{actual_date}`"
            )

        article_text = normalize_text(json.dumps(primary, ensure_ascii=False))
        missing_keywords = [
            keyword
            for keyword in test.expected_keywords
            if normalize_text(keyword) not in article_text
        ]

        if missing_keywords:
            warnings.append(
                f"`{primary_id}` does not contain exact keyword(s): "
                f"{', '.join(missing_keywords)}"
            )

    return errors, warnings


# ============================================================
# API calls
# ============================================================

def build_url(base_url: str, endpoint: str) -> str:
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint

    return base_url.rstrip("/") + "/" + endpoint.lstrip("/")


def post_json(url: str, payload: Dict[str, Any], timeout: int) -> Tuple[int, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=body,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_body = resp.read().decode("utf-8", errors="replace")
            status = resp.getcode()
    except urllib.error.HTTPError as e:
        raw_body = e.read().decode("utf-8", errors="replace")
        status = e.code
    except urllib.error.URLError as e:
        raise ConnectionError(str(e)) from e

    try:
        return status, json.loads(raw_body)
    except json.JSONDecodeError:
        return status, raw_body


def response_looks_rate_limited(status_code: Optional[int], response: Any, error: Optional[str]) -> bool:
    if status_code == 429:
        return True

    text = ""

    if error:
        text += error

    if isinstance(response, dict):
        text += json.dumps(response, ensure_ascii=False)
    elif isinstance(response, str):
        text += response

    lowered = text.lower()

    return "429" in lowered or "too many requests" in lowered or "rate limit" in lowered


def post_json_with_retry(
    url: str,
    payload: Dict[str, Any],
    timeout: int,
    max_retries: int,
    retry_sleep: float,
) -> Tuple[int, Any]:
    last_status: Optional[int] = None
    last_response: Any = None

    for attempt in range(max_retries + 1):
        status, response = post_json(url=url, payload=payload, timeout=timeout)
        last_status = status
        last_response = response

        if not response_looks_rate_limited(status, response, None):
            return status, response

        if attempt < max_retries:
            sleep_for = retry_sleep * (attempt + 1)
            print(f"  Rate limit detected. Sleeping {sleep_for:.1f}s before retry...")
            time.sleep(sleep_for)

    return last_status or 500, last_response


def extract_answer(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()

    if isinstance(response, dict):
        for key in ["answer", "response", "message", "content", "text", "output", "result"]:
            value = response.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

        return json.dumps(response, ensure_ascii=False, indent=2)

    return str(response)


VALID_SCHEMA_RESPONSE_TYPES = {
    "answer",
    "clarification_needed",
    "limited_answer",
    "out_of_scope",
}


def extract_schema_response_type(response: Any) -> Optional[str]:
    if not isinstance(response, dict):
        return None

    raw_type = response.get("type")

    if not isinstance(raw_type, str):
        return None

    normalized_type = clean_spaces(raw_type).lower()

    if normalized_type in VALID_SCHEMA_RESPONSE_TYPES:
        return normalized_type

    return None


def source_from_dict(index: int, item: dict) -> RetrievedSource:
    article_id = item.get("article_id") or item.get("story_id") or item.get("id")

    headline = (
        item.get("headline")
        or item.get("title")
        or item.get("source")
        or item.get("url")
    )

    published_at = (
        item.get("published_at")
        or item.get("date_published")
        or item.get("date")
    )

    if article_id and headline:
        raw = f"{article_id} | {headline}"
    elif published_at and headline:
        raw = f"{published_at} | {headline}"
    elif headline:
        raw = str(headline)
    else:
        raw = json.dumps(item, ensure_ascii=False)[:300]

    return RetrievedSource(
        source_number=index,
        raw=raw,
        article_id=str(article_id) if article_id else None,
        headline=str(headline) if headline else None,
        published_at=str(published_at) if published_at else None,
    )


def parse_source_string(index: int, source: str) -> RetrievedSource:
    source = clean_spaces(source)

    id_match = re.match(r"^\s*([A-Z0-9]{6,})\s*\|\s*(.+)$", source)

    if id_match:
        return RetrievedSource(
            source_number=index,
            raw=source,
            article_id=id_match.group(1),
            headline=clean_spaces(id_match.group(2)),
            published_at=None,
        )

    date_match = re.match(r"^\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(.+)$", source)

    if date_match:
        return RetrievedSource(
            source_number=index,
            raw=source,
            article_id=None,
            headline=clean_spaces(date_match.group(2)),
            published_at=date_match.group(1),
        )

    return RetrievedSource(
        source_number=index,
        raw=source,
        article_id=None,
        headline=source,
        published_at=None,
    )


def extract_sources(response: Any) -> List[RetrievedSource]:
    if not isinstance(response, dict):
        return []

    possible = (
        response.get("sources")
        or response.get("citations")
        or response.get("documents")
        or []
    )

    sources: List[RetrievedSource] = []

    if isinstance(possible, list):
        for index, item in enumerate(possible, start=1):
            if isinstance(item, str):
                sources.append(parse_source_string(index, item))
            elif isinstance(item, dict):
                sources.append(source_from_dict(index, item))

    return sources[:10]


def resolve_source_ids(
    sources: List[RetrievedSource],
    headline_to_ids: Dict[str, List[str]],
    date_headline_to_ids: Dict[Tuple[str, str], List[str]],
) -> List[RetrievedSource]:
    for source in sources:
        if source.article_id:
            continue

        headline_key = normalize_text(source.headline or "")
        date_key = clean_spaces(source.published_at or "")

        matched_ids: List[str] = []

        if date_key and headline_key:
            matched_ids = date_headline_to_ids.get((date_key, headline_key), [])

        if not matched_ids and headline_key:
            matched_ids = headline_to_ids.get(headline_key, [])

        if not matched_ids and headline_key:
            for known_headline, ids in headline_to_ids.items():
                if known_headline and known_headline in headline_key:
                    matched_ids = ids
                    break

                if headline_key and headline_key in known_headline:
                    matched_ids = ids
                    break

        if matched_ids:
            source.article_id = matched_ids[0]

    return sources


def retrieved_ids(sources: List[RetrievedSource]) -> List[str]:
    ids: List[str] = []
    seen = set()

    for source in sources:
        if not source.article_id:
            continue

        if source.article_id in seen:
            continue

        seen.add(source.article_id)
        ids.append(source.article_id)

    return ids


# ============================================================
# Classification
# ============================================================

LIMITED_PATTERNS = [
    "could not find enough",
    "not enough",
    "limited",
    "insufficient",
    "माहिती उपलब्ध नाही",
    "पुरेशी माहिती",
    "सापडली नाही",
    "माहिती नाही",
]

CLARIFICATION_PATTERNS = [
    "clarify",
    "please specify",
    "which",
    "what do you mean",
    "कृपया स्पष्ट",
    "कोणती बातमी",
    "कुठली बातमी",
]

OUT_OF_SCOPE_PATTERNS = [
    "out of scope",
    "outside",
    "cannot help",
    "can't help",
    "not able to",
    "policy",
    "मर्यादेबाहेर",
]

ERROR_PATTERNS = [
    "traceback",
    "internal server error",
    "exception",
    "validation error",
    "bad request",
    "too many requests",
    "429",
]


def contains_any(text: str, patterns: List[str]) -> bool:
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def classify_response(answer: str, status_code: Optional[int], error: Optional[str]) -> str:
    if error or (status_code is not None and status_code >= 500):
        return "error"

    if status_code is not None and status_code >= 400:
        return "error"

    if not answer.strip():
        return "empty"

    if contains_any(answer, ERROR_PATTERNS):
        return "error"

    if contains_any(answer, OUT_OF_SCOPE_PATTERNS):
        return "out_of_scope"

    if contains_any(answer, LIMITED_PATTERNS):
        return "limited_answer"

    if contains_any(answer, CLARIFICATION_PATTERNS) and len(answer.split()) < 90:
        return "clarification_needed"

    return "answer"


def extract_trace(response: Any, status_code: Optional[int], error: Optional[str]) -> str:
    if error:
        return f"API error: {error}"

    if isinstance(response, dict):
        trace_parts: List[str] = []

        for key in ["intent", "planner_query", "current_query", "query", "rewritten_query"]:
            if response.get(key):
                trace_parts.append(f"{key}: {response.get(key)}")

        for key in ["trace", "debug", "metadata", "retrieval"]:
            if response.get(key):
                value = response.get(key)

                if not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False)

                trace_parts.append(f"{key}: {value}")

        if trace_parts:
            return " / ".join(trace_parts)[:3000]

    return f"HTTP status: {status_code}"


def keyword_hit_count(answer: str, expected_keywords: List[str]) -> int:
    text = answer.lower()
    count = 0

    for kw in expected_keywords:
        kw = clean_spaces(kw).lower()

        if kw and kw in text:
            count += 1

    return count


# ============================================================
# Runner
# ============================================================

def call_website(
    test: TestTurn,
    url: str,
    payload_key: str,
    timeout: int,
    strict_payload: bool,
    history: List[Dict[str, str]],
    headline_to_ids: Dict[str, List[str]],
    date_headline_to_ids: Dict[Tuple[str, str], List[str]],
    max_retries: int,
    retry_sleep: float,
) -> TestResult:
    if strict_payload:
        payload = {
            payload_key: test.question,
        }
    else:
        payload = {
            payload_key: test.question,
            "source": "sakal",
            "history": history,
        }

    started = time.perf_counter()
    status_code: Optional[int] = None
    raw_response: Any = None
    error: Optional[str] = None

    try:
        status_code, raw_response = post_json_with_retry(
            url=url,
            payload=payload,
            timeout=timeout,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )

        answer = extract_answer(raw_response)
        sources = extract_sources(raw_response)
        sources = resolve_source_ids(
            sources=sources,
            headline_to_ids=headline_to_ids,
            date_headline_to_ids=date_headline_to_ids,
        )

    except Exception as exc:
        answer = ""
        sources = []
        error = str(exc)

    elapsed = time.perf_counter() - started
    schema_response_type = extract_schema_response_type(raw_response)
    response_type = schema_response_type or classify_response(answer, status_code, error)
    trace = extract_trace(raw_response, status_code, error)
    ids = retrieved_ids(sources)
    expected_article_retrieved = any(expected_id in ids for expected_id in test.expected_source_ids)

    return TestResult(
        test=test,
        response_type=response_type,
        elapsed=elapsed,
        answer=answer,
        status_code=status_code,
        error=error,
        sources=sources,
        expected_article_retrieved=expected_article_retrieved,
        trace=trace,
        raw_response=raw_response,
    )


def run_dry_tests(tests: List[TestTurn]) -> List[TestResult]:
    results: List[TestResult] = []

    for test in tests:
        results.append(
            TestResult(
                test=test,
                response_type="dry_run",
                elapsed=0.0,
                answer=(
                    "DRY RUN ONLY. This hardcoded test case was generated but not sent "
                    f"to the API. Expected sources: {', '.join(test.expected_source_ids)} — "
                    f"{test.expected_primary_headline}"
                ),
                status_code=None,
                error=None,
                sources=[],
                expected_article_retrieved=False,
                trace="Dry run: API call skipped.",
                raw_response=None,
            )
        )

    return results


def run_live_tests(
    tests: List[TestTurn],
    url: str,
    payload_key: str,
    timeout: int,
    delay: float,
    strict_payload: bool,
    headline_to_ids: Dict[str, List[str]],
    date_headline_to_ids: Dict[Tuple[str, str], List[str]],
    max_retries: int,
    retry_sleep: float,
) -> List[TestResult]:
    results: List[TestResult] = []
    histories: Dict[Tuple[str, int], List[Dict[str, str]]] = collections.defaultdict(list)

    for idx, test in enumerate(tests, start=1):
        print(
            f"[{idx}/{len(tests)}] "
            f"{test.language} / session {test.session_index} / {test.label}"
        )

        history_key = (test.language, test.session_index)
        history = histories[history_key]

        result = call_website(
            test=test,
            url=url,
            payload_key=payload_key,
            timeout=timeout,
            strict_payload=strict_payload,
            history=history,
            headline_to_ids=headline_to_ids,
            date_headline_to_ids=date_headline_to_ids,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )

        results.append(result)

        if result.answer.strip():
            history.append({"role": "user", "content": test.question})
            history.append({"role": "assistant", "content": result.answer[:2000]})

        if delay > 0:
            time.sleep(delay)

    return results


# ============================================================
# Markdown report
# ============================================================

def truncate(text: str, max_chars: int = 900) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()

    if len(text) <= max_chars:
        return text

    return text[: max_chars - 3].rstrip() + "..."


def format_sources(sources: List[RetrievedSource]) -> str:
    if not sources:
        return "None"

    formatted = []

    for source in sources:
        article_id = source.article_id or "UNKNOWN_ID"
        date_value = source.published_at or "UNKNOWN_DATE"
        headline = source.headline or source.raw

        formatted.append(
            f"[{source.source_number}] `{article_id}` | {date_value} | {headline}"
        )

    return "; ".join(formatted)


def format_retrieved_ids(sources: List[RetrievedSource]) -> str:
    ids = retrieved_ids(sources)

    if not ids:
        return "None"

    return ", ".join(ids)


def ranked_retrieved_ids(sources: List[RetrievedSource]) -> List[str]:
    return [
        source.article_id
        for source in sources
        if source.article_id
    ]


def precision_at_k(ranked_ids: List[str], expected_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0

    top_k = ranked_ids[:k]

    if not top_k:
        return 0.0

    hits = sum(1 for article_id in top_k if article_id in expected_ids)
    return hits / k


def recall_at_k(ranked_ids: List[str], expected_ids: set[str], k: int) -> float:
    if not expected_ids:
        return 0.0

    top_k = ranked_ids[:k]
    hits = len({article_id for article_id in top_k if article_id in expected_ids})
    return hits / len(expected_ids)


def reciprocal_rank(ranked_ids: List[str], expected_ids: set[str]) -> float:
    for index, article_id in enumerate(ranked_ids, start=1):
        if article_id in expected_ids:
            return 1.0 / index

    return 0.0


def average_precision(ranked_ids: List[str], expected_ids: set[str]) -> float:
    if not expected_ids:
        return 0.0

    hits = 0
    precision_sum = 0.0
    seen_relevant = set()

    for index, article_id in enumerate(ranked_ids, start=1):
        if article_id not in expected_ids or article_id in seen_relevant:
            continue

        seen_relevant.add(article_id)
        hits += 1
        precision_sum += hits / index

    return precision_sum / len(expected_ids)


def ndcg_at_k(ranked_ids: List[str], expected_ids: set[str], k: int) -> float:
    if not expected_ids or k <= 0:
        return 0.0

    dcg = 0.0

    for index, article_id in enumerate(ranked_ids[:k], start=1):
        if article_id in expected_ids:
            dcg += 1.0 / math.log2(index + 1)

    ideal_hits = min(len(expected_ids), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def source_diversity_ratio(ranked_ids: List[str]) -> float:
    if not ranked_ids:
        return 0.0

    return len(set(ranked_ids)) / len(ranked_ids)


def retrieval_metrics(result: TestResult) -> Dict[str, float]:
    ranked_ids = ranked_retrieved_ids(result.sources)
    expected_ids = set(result.test.expected_source_ids)

    return {
        "hit@1": 1.0 if any(article_id in expected_ids for article_id in ranked_ids[:1]) else 0.0,
        "hit@3": 1.0 if any(article_id in expected_ids for article_id in ranked_ids[:3]) else 0.0,
        "hit@5": 1.0 if any(article_id in expected_ids for article_id in ranked_ids[:5]) else 0.0,
        "precision@3": precision_at_k(ranked_ids, expected_ids, 3),
        "precision@5": precision_at_k(ranked_ids, expected_ids, 5),
        "recall@5": recall_at_k(ranked_ids, expected_ids, 5),
        "mrr": reciprocal_rank(ranked_ids, expected_ids),
        "map": average_precision(ranked_ids, expected_ids),
        "ndcg@5": ndcg_at_k(ranked_ids, expected_ids, 5),
        "source_diversity": source_diversity_ratio(ranked_ids),
        "retrieved_count": float(len(ranked_ids)),
    }


def mean_metric(results: List[TestResult], metric_name: str) -> float:
    if not results:
        return 0.0

    return sum(retrieval_metrics(result)[metric_name] for result in results) / len(results)


def append_retrieval_metric_table(lines: List[str], title: str, results: List[TestResult]) -> None:
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| Group | N | Hit@1 | Hit@3 | Hit@5 | Precision@3 | Precision@5 | Recall@5 | MRR | MAP | nDCG@5 | Source diversity |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    groups: List[Tuple[str, List[TestResult]]] = [("All", results)]

    for language in ["English", "Marathi"]:
        groups.append((language, [result for result in results if result.test.language == language]))

    for group_name, group_results in groups:
        lines.append(
            f"| {group_name} | {len(group_results)} | "
            f"{mean_metric(group_results, 'hit@1'):.3f} | "
            f"{mean_metric(group_results, 'hit@3'):.3f} | "
            f"{mean_metric(group_results, 'hit@5'):.3f} | "
            f"{mean_metric(group_results, 'precision@3'):.3f} | "
            f"{mean_metric(group_results, 'precision@5'):.3f} | "
            f"{mean_metric(group_results, 'recall@5'):.3f} | "
            f"{mean_metric(group_results, 'mrr'):.3f} | "
            f"{mean_metric(group_results, 'map'):.3f} | "
            f"{mean_metric(group_results, 'ndcg@5'):.3f} | "
            f"{mean_metric(group_results, 'source_diversity'):.3f} |"
        )

    lines.append("")
    lines.append(
        "Note: MRR is Mean Reciprocal Rank. MMR usually means Maximal Marginal "
        "Relevance, which is a retrieval diversification strategy rather than a "
        "pass/fail metric; `Source diversity` is a lightweight proxy for duplicate-heavy results."
    )
    lines.append("")


def write_markdown_report(
    output_path: Path,
    results: List[TestResult],
    data_path: Path,
    url: str,
    dry_run: bool,
) -> None:
    counts = collections.Counter(r.response_type for r in results)
    language_counts = collections.Counter(r.test.language for r in results)
    retrieved_yes = sum(1 for r in results if r.expected_article_retrieved)
    retrieved_no = len(results) - retrieved_yes
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []

    lines.append("# Sakal Newsbot Hardcoded Multi-turn Evaluation")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append(f"Data file: `{data_path}`")
    lines.append(f"Target URL: `{url}`")
    lines.append(f"Dry run: `{dry_run}`")
    lines.append("")
    lines.append(f"Total questions tested: {len(results)}")
    lines.append("")

    lines.append("## Language Summary")
    lines.append("")
    for language in ["English", "Marathi"]:
        lines.append(f"- `{language}`: {language_counts.get(language, 0)}")
    lines.append("")

    lines.append("## Response Type Summary")
    lines.append("")
    for key in [
        "answer",
        "limited_answer",
        "clarification_needed",
        "out_of_scope",
        "error",
        "empty",
        "dry_run",
    ]:
        if counts.get(key):
            lines.append(f"- `{key}`: {counts[key]}")
    lines.append("")

    lines.append("## Retrieval Match Summary")
    lines.append("")
    lines.append(f"- Expected article retrieved: `{retrieved_yes}`")
    lines.append(f"- Expected article not retrieved: `{retrieved_no}`")
    lines.append("")

    append_retrieval_metric_table(lines, "Ranked Retrieval Metrics", results)

    lines.append("## Detailed Results")
    lines.append("")

    for idx, result in enumerate(results, start=1):
        test = result.test
        keyword_hits = keyword_hit_count(result.answer, test.expected_keywords)

        lines.append(
            f"### {idx}. {test.language} / {test.persona} / {test.need} / "
            f"Session {test.session_index} / Q{test.turn_index} / {test.label}"
        )
        lines.append("")
        lines.append(f"**Question:** {test.question}")
        lines.append("")
        lines.append(f"**Response type:** `{result.response_type}`")
        lines.append(f"**Elapsed:** {result.elapsed:.2f}s")
        lines.append(
            f"**HTTP status:** "
            f"{result.status_code if result.status_code is not None else 'None'}"
        )
        lines.append(
            f"**Expected sources:** `{', '.join(test.expected_source_ids)}`"
        )
        lines.append(
            f"**Primary expected source:** `{test.expected_primary_article_id}` | "
            f"{test.expected_primary_date_published or 'UNKNOWN_DATE'} | "
            f"{test.expected_primary_headline}"
        )
        lines.append(f"**Expected topic:** {test.expected_topic}")
        lines.append(
            f"**Expected keywords:** "
            f"{', '.join(test.expected_keywords) if test.expected_keywords else 'None'}"
        )
        lines.append(f"**Keyword hits in answer:** {keyword_hits}")
        lines.append(f"**Retrieved sources:** {format_sources(result.sources)}")
        lines.append(f"**Retrieved article IDs:** {format_retrieved_ids(result.sources)}")
        metrics = retrieval_metrics(result)
        lines.append(
            "**Retrieval metrics:** "
            f"Hit@1 `{metrics['hit@1']:.0f}`, "
            f"Hit@3 `{metrics['hit@3']:.0f}`, "
            f"Precision@3 `{metrics['precision@3']:.3f}`, "
            f"Recall@5 `{metrics['recall@5']:.3f}`, "
            f"MRR `{metrics['mrr']:.3f}`, "
            f"nDCG@5 `{metrics['ndcg@5']:.3f}`"
        )
        lines.append(
            f"**Expected article retrieved:** "
            f"{'YES' if result.expected_article_retrieved else 'NO'}"
        )
        lines.append("")
        lines.append(f"**Trace:** {truncate(result.trace, 1800)}")
        lines.append("")

        if result.error:
            lines.append(f"**Error:** {result.error}")
            lines.append("")

        lines.append(f"**Answer excerpt:** {truncate(result.answer, 1400)}")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def summarize_metrics(results: List[TestResult]) -> Dict[str, float]:
    total = len(results)

    if total == 0:
        return {
            "total": 0.0,
            "error_count": 0.0,
            "error_rate": 0.0,
            "retrieval_hit_count": 0.0,
            "retrieval_hit_rate": 0.0,
            "answer_count": 0.0,
            "answer_rate": 0.0,
            "hit@1": 0.0,
            "hit@3": 0.0,
            "precision@3": 0.0,
            "recall@5": 0.0,
            "mrr": 0.0,
            "ndcg@5": 0.0,
        }

    error_count = sum(1 for result in results if result.response_type == "error")
    retrieval_hit_count = sum(1 for result in results if result.expected_article_retrieved)
    answer_count = sum(1 for result in results if result.response_type == "answer")

    return {
        "total": float(total),
        "error_count": float(error_count),
        "error_rate": error_count / total,
        "retrieval_hit_count": float(retrieval_hit_count),
        "retrieval_hit_rate": retrieval_hit_count / total,
        "answer_count": float(answer_count),
        "answer_rate": answer_count / total,
        "hit@1": mean_metric(results, "hit@1"),
        "hit@3": mean_metric(results, "hit@3"),
        "precision@3": mean_metric(results, "precision@3"),
        "recall@5": mean_metric(results, "recall@5"),
        "mrr": mean_metric(results, "mrr"),
        "ndcg@5": mean_metric(results, "ndcg@5"),
    }


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Sakal hardcoded multi-turn evaluation tests."
    )

    parser.add_argument(
        "--data",
        default="sakal.json.gz",
        help="Path to sakal.json.gz. Used only to resolve retrieved source IDs.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of your running API.",
    )
    parser.add_argument(
        "--endpoint",
        default="/chat",
        help="Endpoint path or full URL.",
    )
    parser.add_argument(
        "--payload-key",
        default="question",
        help="JSON key used for the user question.",
    )
    parser.add_argument(
        "--strict-payload",
        action="store_true",
        help="Send only the question field. Do not use this for multi-turn history.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Markdown output path.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=6.0,
        help="Delay between API calls in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Number of retries when API response appears rate-limited.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=20.0,
        help="Base sleep seconds for retry backoff.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate report without calling the API.",
    )
    parser.add_argument(
        "--language-mode",
        choices=["both", "english", "marathi"],
        default="both",
        help=(
            "Question set to run: english=15 English questions, "
            "marathi=30 Marathi questions, both=45 total questions."
        ),
    )
    parser.add_argument(
        "--min-retrieval-hit-rate",
        type=float,
        default=None,
        help="Fail if retrieval hit rate is below this threshold (0.0-1.0).",
    )
    parser.add_argument(
        "--max-error-rate",
        type=float,
        default=None,
        help="Fail if error rate is above this threshold (0.0-1.0).",
    )
    parser.add_argument(
        "--min-answer-rate",
        type=float,
        default=None,
        help="Fail if answer rate is below this threshold (0.0-1.0).",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    data_path = Path(args.data)
    output_path = Path(
        args.output
        or f"eval_runs/sakal_newsbot_hardcoded_results_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )

    url = build_url(args.base_url, args.endpoint)

    tests = build_tests(language_mode=args.language_mode)
    assert_no_headline_leak(tests)

    english_count = sum(1 for test in tests if test.language == "English")
    marathi_count = sum(1 for test in tests if test.language == "Marathi")

    print(f"Loaded hardcoded test questions: {len(tests)}")
    print(f"English questions: {english_count}")
    print(f"Marathi questions: {marathi_count}")

    print(f"Loading source lookup from: {data_path}")
    headline_to_ids, date_headline_to_ids = load_source_lookup(data_path)
    print(f"Built headline lookup: {len(headline_to_ids)}")
    print(f"Built date+headline lookup: {len(date_headline_to_ids)}")

    fixture_errors, fixture_warnings = validate_source_expectations(data_path, tests)

    if fixture_warnings:
        print("\nFixture warnings:")
        for warning in fixture_warnings:
            print(f"- {warning}")

    if fixture_errors:
        print("\nFixture errors:")
        for error in fixture_errors:
            print(f"- {error}")
        raise ValueError("Hardcoded Sakal test fixtures do not match the data file.")

    print("\nGenerated questions preview:")
    for idx, test in enumerate(tests, start=1):
        print(
            f"{idx:02d}. [{test.language}] {test.expected_primary_article_id} | "
            f"{test.expected_topic} | {test.question}"
        )

    if args.dry_run:
        results = run_dry_tests(tests)
    else:
        results = run_live_tests(
            tests=tests,
            url=url,
            payload_key=args.payload_key,
            timeout=args.timeout,
            delay=args.delay,
            strict_payload=args.strict_payload,
            headline_to_ids=headline_to_ids,
            date_headline_to_ids=date_headline_to_ids,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
        )

    write_markdown_report(
        output_path=output_path,
        results=results,
        data_path=data_path,
        url=url,
        dry_run=args.dry_run,
    )

    metrics = summarize_metrics(results)

    print("\nRAG evaluation metrics:")
    print(f"- total: {int(metrics['total'])}")
    print(f"- error_rate: {metrics['error_rate']:.3f}")
    print(f"- retrieval_hit_rate: {metrics['retrieval_hit_rate']:.3f}")
    print(f"- answer_rate: {metrics['answer_rate']:.3f}")
    print(f"- hit@1: {metrics['hit@1']:.3f}")
    print(f"- hit@3: {metrics['hit@3']:.3f}")
    print(f"- precision@3: {metrics['precision@3']:.3f}")
    print(f"- recall@5: {metrics['recall@5']:.3f}")
    print(f"- mrr: {metrics['mrr']:.3f}")
    print(f"- ndcg@5: {metrics['ndcg@5']:.3f}")

    failures: List[str] = []

    if args.min_retrieval_hit_rate is not None:
        if metrics["retrieval_hit_rate"] < args.min_retrieval_hit_rate:
            failures.append(
                "retrieval_hit_rate "
                f"{metrics['retrieval_hit_rate']:.3f} < {args.min_retrieval_hit_rate:.3f}"
            )

    if args.max_error_rate is not None:
        if metrics["error_rate"] > args.max_error_rate:
            failures.append(
                "error_rate "
                f"{metrics['error_rate']:.3f} > {args.max_error_rate:.3f}"
            )

    if args.min_answer_rate is not None:
        if metrics["answer_rate"] < args.min_answer_rate:
            failures.append(
                "answer_rate "
                f"{metrics['answer_rate']:.3f} < {args.min_answer_rate:.3f}"
            )

    print(f"\nReport written to: {output_path}")

    if failures:
        print("\nRAG evaluation FAILED thresholds:")
        for failure in failures:
            print(f"- {failure}")
        return 2

    if (
        args.min_retrieval_hit_rate is not None
        or args.max_error_rate is not None
        or args.min_answer_rate is not None
    ):
        print("\nRAG evaluation PASSED thresholds.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
