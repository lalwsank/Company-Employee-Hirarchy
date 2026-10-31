"""
Org Chart Inference Engine - v4 (rewritten from scratch)
=========================================================

The employee table is the ONLY source of truth.  Everything the engine knows
(what a title means, which team belongs to which department, which skills
belong to which team) is learned from that table, then used to infer who
reports to whom.

The pipeline has two stages:

STAGE 1 - prepare()   : clean the table
    * Company identity   One real company = one e-mail domain.  Rows with a
                         typo'd / missing domain or name are attached to the
                         right company by name + industry + country evidence.
                         Industry spellings ("Auto", "Automotive Industry",
                         "Automobile", "Automotiev") collapse to ONE canonical
                         industry and are never part of the company's name.
    * Duplicates         The same person listed twice is merged into one record.
    * Seniority          Seniority Level column -> else learned from the
                         designation -> else keyword rule.
    * Department / Team  Canonical spellings; blanks are filled from the team,
                         the designation or the skills (each learned from the
                         table itself).

STAGE 2 - infer_company() : infer reporting lines for one company
    Level 10  CEO
    Level 9   department heads         -> report to the CEO
    Level 8   team managers            -> report to the head of their department
    Level <8  individual contributors  -> report to the manager of their team

    HARD RULE: only level >= 8 can have direct reports.
    A wrong line is worse than none: weak or tied cases are left unresolved.
"""

from __future__ import annotations

import difflib
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT_LEVEL = 10
HEAD_LEVEL = 9
MANAGER_LEVEL = 8
MANAGER_MIN_LEVEL = 8          # only levels >= 8 may have direct reports


@dataclass
class InferenceConfig:
    min_confidence: int = 60           # reporting line must score at least this
    min_margin: int = 5                # best candidate must beat runner-up by this
    allow_skip_level: bool = True      # team has no manager -> attach to dept head
    guess_between_ties: bool = False   # split people between equally likely managers


# ============================================================
# 2. TEXT HELPERS
# ============================================================

PLACEHOLDERS = {
    "", "n a", "na", "null", "none", "tbd", "unknown", "nan", "x", "xx", "xxx",
    "test", "tester", "asdf", "temp", "not available", "undefined", "nil",
}

FREE_MAIL = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com",
    "rediffmail.com", "aol.com", "live.com", "protonmail.com", "msn.com",
    "me.com", "mail.com", "ymail.com",
}
JUNK_DOMAINS = {"example.com", "test.com", "asdf.com", "tbd", "n/a", "na"}


def ct(value: Any) -> str:
    """Categorical clean: lower-case ASCII words (stray mojibake is dropped)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    v = str(value).encode("ascii", "ignore").decode().lower().strip()
    v = v.replace("&", " and ").replace("-", " ").replace("/", " ")
    v = re.sub(r"[^a-z0-9\s]", " ", v)
    return re.sub(r"\s+", " ", v).strip()


def fold(value: Any) -> str:
    """Accent-folding lower-case (for people's names)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    v = unicodedata.normalize("NFKD", str(value))
    v = "".join(c for c in v if not unicodedata.combining(c))
    return v.encode("ascii", "ignore").decode().lower().strip()


def is_placeholder(value: Any) -> bool:
    return ct(value) in PLACEHOLDERS


_closest_cache: Dict[Tuple[str, Tuple[str, ...], float], Optional[str]] = {}


def closest(word: str, vocab: List[str], cutoff: float) -> Optional[str]:
    key = (word, tuple(vocab), cutoff) if len(vocab) < 80 else (word, id(vocab), cutoff)  # type: ignore
    if key in _closest_cache:
        return _closest_cache[key]
    m = difflib.get_close_matches(word, vocab, n=1, cutoff=cutoff)
    _closest_cache[key] = m[0] if m else None
    return _closest_cache[key]


# ============================================================
# 3. INDUSTRY
# ============================================================

INDUSTRY_ALIASES = {
    "Healthcare": ["healthcare", "health care", "medical", "health", "healthcare services"],
    "Energy & Utilities": ["energy and utilities", "energy", "utilities", "oil and gas utilities"],
    "Logistics & Transportation": [
        "logistics and transportation", "logistics", "supply chain and logistics",
        "transportation", "transport and logistics"],
    "Education": ["education", "edtech", "edu", "higher ed", "education and training"],
    "Pharmaceuticals": [
        "pharmaceuticals", "pharmaceutical", "pharma", "pharma and biotech", "life sciences"],
    "Automotive": ["automotive", "auto", "automobile", "automotive industry"],
    "Consulting": [
        "consulting", "management consulting", "consultancy", "professional services",
        "consulting services"],
    "Telecommunications": [
        "telecommunications", "telecommunication", "telecom", "telecomms", "telco"],
    "Real Estate": ["real estate", "re", "realty", "property"],
    "Financial Services": [
        "financial services", "finance", "finserv", "bfsi", "banking and finance",
        "financial svcs", "fin services"],
    "Media & Entertainment": ["media and entertainment", "media", "entertainment", "m and e"],
    "Manufacturing": [
        "manufacturing", "mfg", "manufacturing and industrial", "industrial"],
    "Hospitality": [
        "hospitality", "travel and hospitality", "hotels and hospitality",
        "hospitality and tourism", "hotel hospitality"],
    "Information Technology": [
        "information technology", "it", "tech", "software it", "info tech", "i t",
        "information tech"],
    "Retail & E-Commerce": [
        "retail and e commerce", "retail ecom", "e commerce retail", "ecommerce", "retail"],
}
_INDUSTRY_LOOKUP = {ct(a): k for k, v in INDUSTRY_ALIASES.items() for a in v + [k]}
_INDUSTRY_KEYS = list(_INDUSTRY_LOOKUP)


def canonical_industry(value: Any) -> str:
    c = ct(value)
    if c in _INDUSTRY_LOOKUP:
        return _INDUSTRY_LOOKUP[c]
    if c in PLACEHOLDERS or not re.search(r"[a-z]{2}", c):
        return ""
    m = closest(c, _INDUSTRY_KEYS, 0.74)
    return _INDUSTRY_LOOKUP[m] if m else ""


# ============================================================
# 4. DEPARTMENT / TEAM / TITLE NORMALISATION
# ============================================================

DEPARTMENTS = ["Engineering", "Sales", "Operations", "Marketing", "Finance",
               "Human Resources", "Executive"]

DEPT_ALIASES = {
    "Engineering": [
        "engineering", "engg", "eng", "it engineering", "r and d", "dev", "tech",
        "product engineering", "software", "technology", "information technology", "it",
        "development", "research and development", "software engineering"],
    "Sales": [
        "sales", "sales dept", "revenue", "sales and bd", "biz dev", "business development",
        "bd", "account management"],
    "Operations": [
        "operations", "ops", "operation", "operations and admin", "business ops",
        "ops dept", "business operations", "supply chain", "admin"],
    "Marketing": [
        "marketing", "mktg", "brand and marketing", "marketing and comms", "marketing dept",
        "growth", "mkt", "mktng", "mtg", "comms", "communications"],
    "Finance": [
        "finance", "finance dept", "finance and accounting", "fin", "f and a",
        "accounting", "finance and admin"],
    "Human Resources": [
        "human resources", "hr", "people and culture", "h r", "human resource",
        "people ops", "people", "talent", "hr dept"],
    "Executive": ["executive", "exec", "leadership", "management", "c suite"],
}
_HYBRID_DEPTS = {"sales and marketing": ("Sales", "Marketing")}
_DEPT_LOOKUP = {ct(a): k for k, v in DEPT_ALIASES.items() for a in v + [k]}
_DEPT_KEYS = list(_DEPT_LOOKUP) + list(_HYBRID_DEPTS)


def canonical_department(value: Any) -> Tuple[str, ...]:
    """Return () if unknown, ('Sales',) if clear, ('Sales','Marketing') if hybrid."""
    c = ct(value)
    if c in PLACEHOLDERS or not re.search(r"[a-z]{2}", c):
        return ()
    if c in _HYBRID_DEPTS:
        return _HYBRID_DEPTS[c]
    if c in _DEPT_LOOKUP:
        return (_DEPT_LOOKUP[c],)
    m = closest(c, _DEPT_KEYS, 0.76)
    if m is None:
        return ()
    return _HYBRID_DEPTS[m] if m in _HYBRID_DEPTS else (_DEPT_LOOKUP[m],)


_TEAM_WORDS = {
    "engg": "engineering", "eng": "engineering", "mktg": "marketing",
    "mktng": "marketing", "mkt": "marketing", "ops": "operations",
    "bus": "business", "biz": "business", "cust": "customer",
    "compl": "compliance", "admin": "administration", "acq": "acquisition",
    "dev": "development", "dept": "department", "partners": "partnerships",
    "analytcs": "analytics",
}
_TEAM_ALIASES = {
    "qa": "quality assurance", "q a": "quality assurance",
    "scm": "supply chain", "pr": "public relations", "p r": "public relations",
    "fp and a and analysis": "financial planning and analysis",
    "fp and a": "financial planning and analysis",
    "fpa": "financial planning and analysis",
    "management": "department leadership", "leadership": "department leadership",
    "leadership team": "department leadership", "management team": "department leadership",
    "executive office": "department leadership", "office of the ceo": "department leadership",
    "c suite": "department leadership", "dept head": "department leadership",
    "department head": "department leadership", "executive": "department leadership",
}


def team_key(value: Any) -> str:
    c = ct(value)
    if c in PLACEHOLDERS:
        return ""
    c = " ".join(_TEAM_WORDS.get(w, w) for w in c.split())
    return _TEAM_ALIASES.get(c, c)


LEADERSHIP_TEAM = "department leadership"
_ACRONYMS = {"Smb": "SMB", "Hr": "HR", "Qa": "QA", "Fp": "FP", "Pr": "PR"}


def pretty_team(key: str) -> str:
    if not key:
        return ""
    words = [w if w == "and" else _ACRONYMS.get(w.title(), w.title()) for w in key.split()]
    return " ".join(words).replace(" and ", " & ")

_TITLE_WORDS = {
    "sr": "senior", "snr": "senior", "jr": "junior", "mgr": "manager",
    "mngr": "manager", "manger": "manager", "dir": "director", "asst": "assistant",
    "assoc": "associate", "exec": "executive", "ops": "operations", "soft": "software",
    "sw": "software", "engr": "engineer", "engg": "engineer", "eng": "engineer",
    "mktg": "marketing", "acct": "account", "swe": "software engineer",
    "vp": "vice president", "hr": "hr",
}
_TITLE_NOISE = re.compile(r"\b(contractor|temp|remote|on leave|contract)\b")


def title_key(value: Any) -> str:
    c = ct(value)
    c = c.replace("v p ", "vp ").replace("s w ", "software ")
    c = _TITLE_NOISE.sub(" ", c)
    c = " ".join(_TITLE_WORDS.get(w, w) for w in c.split())
    return re.sub(r"\s+", " ", c).strip()


# ============================================================
# 5. SENIORITY
# ============================================================

_LEVEL_WORDS = {
    "mid": 3, "intermediate": 3, "mid level": 3, "entry": 1, "entry level": 1,
    "associate": 1, "junior": 1, "manager": 8, "mgr": 8, "lead": 6, "team lead": 6,
    "senior": 5, "sr": 5, "principal": 5, "director": 9, "head": 9, "vp": 9,
    "c level": 10, "c suite": 10, "ceo": 10, "executive": 10,
}


def parse_level(value: Any) -> int:
    """Seniority Level cell -> 1..10, or 0 if missing / unreadable."""
    c = ct(value)
    if c in PLACEHOLDERS:
        return 0
    m = re.search(r"\d+", c)
    if m:
        n = int(m.group())
        return n if 1 <= n <= 10 else 0
    return _LEVEL_WORDS.get(c, 0)


def rule_level(title: str) -> int:
    """Keyword fallback for a (title_key-normalised) designation. 0 = no idea."""
    t = title
    if not t:
        return 0
    if "vice" not in t and re.search(r"\bceo\b|chief executive|\bpresident\b|\bfounder\b|c suite", t):
        return 10
    if re.search(r"\bchief\b|\bvice president\b|\bdirector\b|\bhead\b|\bc[a-z]o\b", t):
        return 9
    if re.search(r"\bmanager\b|\bsupervisor\b", t):
        return 8
    if re.search(r"\blead\b", t):
        return 6
    if re.search(r"\bsenior\b|\bprincipal\b|\bstaff\b", t):
        return 5
    if re.search(r"\bjunior\b|\bassociate\b|\bintern\b|\btrainee\b|\bassistant\b|\bgraduate\b", t):
        return 1
    return 3


def head_rank(title: str) -> int:
    """Tie-break between several level-9 people in one department."""
    if re.search(r"\bchief\b|\bc[a-z]o\b", title):
        return 3
    if re.search(r"\bvice president\b|\bhead\b", title):
        return 2
    return 1


# ============================================================
# 6. COLUMN HANDLING
# ============================================================

COLUMN_ALIASES = {
    "Company Name": ["company name", "company", "organization", "organisation", "employer"],
    "Industry": ["industry", "sector"],
    "Designation": ["designation", "title", "job title", "position", "role"],
    "Seniority Level": ["seniority level", "seniority", "level", "grade"],
    "Department": ["department", "dept", "function"],
    "Team/Business Unit": ["team/business unit", "team business unit", "team", "business unit", "unit"],
    "Employee Name": ["employee name", "name", "full name", "employee"],
    "Skills/Specialization": ["skills/specialization", "skills specialization", "skills",
                              "specialization", "specialisation"],
    "City": ["city", "location"],
    "Country": ["country"],
    "Email": ["email", "e-mail", "email address", "work email"],
}
MANDATORY_COLUMNS = ["Company Name", "Employee Name"]


def _read_columns(df: pd.DataFrame) -> pd.DataFrame:
    lookup = {re.sub(r"\s+", " ", str(c).strip().lower()): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for canonical, aliases in COLUMN_ALIASES.items():
        src = None
        for a in [canonical.lower()] + aliases:
            if a in lookup:
                src = lookup[a]
                break
        if src is None:
            if canonical in MANDATORY_COLUMNS:
                raise ValueError(f"Missing required column: {canonical}")
            out[canonical] = ""
        else:
            out[canonical] = df[src].fillna("").astype(str).str.replace("\u00a0", " ").str.strip()
    return out


# ============================================================
# 7. COMPANY IDENTITY
# ============================================================

_COMPANY_SUFFIX = re.compile(
    r"\b(inc|incorporated|ltd|limited|llc|l l c|plc|corp|corporation|co|company|"
    r"group|grp|gmbh|pvt|sa|ag)\b")
MIN_DOMAIN_ROWS = 12


def company_key(value: Any) -> str:
    s = fold(value).replace("&", " and ")
    s = re.sub(r"^the\s+", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = _COMPANY_SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def email_domain(value: Any) -> str:
    s = str(value or "").strip().lower().replace(",", ".")
    m = re.search(r"@([^@\s]+)$", s)
    return m.group(1).strip(".;") if m else ""


def display_company_name(raw: str) -> str:
    s = re.sub(r"\s+", " ", str(raw).replace("\u00a0", " ")).strip()
    s = re.sub(r"^the\s+", "", s, flags=re.I)
    if s.isupper() or s.islower():
        s = s.title()
    return s


@dataclass
class _Anchor:
    domain: str
    name_key: str
    display: str
    industry: str
    country: str
    rows: int


def _build_anchors(frame: pd.DataFrame) -> Dict[str, _Anchor]:
    counts = frame["_dom"].value_counts()
    domains = [
        d for d, n in counts.items()
        if d and n >= MIN_DOMAIN_ROWS and "." in d
        and d not in FREE_MAIL and d not in JUNK_DOMAINS
    ]
    anchors: Dict[str, _Anchor] = {}
    sub = frame[frame["_dom"].isin(domains)]
    for dom, g in sub.groupby("_dom", sort=False):
        nk = g["_nk"][g["_nk"] != ""]
        ind = g["Industry (clean)"][g["Industry (clean)"] != ""]
        cty = g["_country"][g["_country"] != ""]
        raw = g["Company Name"][g["_nk"] != ""]
        anchors[dom] = _Anchor(
            domain=dom,
            name_key=nk.mode().iloc[0] if len(nk) else "",
            display=display_company_name(raw.mode().iloc[0]) if len(raw) else dom,
            industry=ind.mode().iloc[0] if len(ind) else "",
            country=cty.mode().iloc[0] if len(cty) else "",
            rows=len(g),
        )
    return anchors


def _resolve_company_rows(frame: pd.DataFrame, anchors: Dict[str, _Anchor]) -> pd.Series:
    """Company Key (= anchor domain) per row, '' when it cannot be decided."""
    by_nk: Dict[str, List[str]] = defaultdict(list)
    for a in anchors.values():
        by_nk[a.name_key].append(a.domain)
    nk_vocab = [k for k in by_nk if k]
    dom_vocab = list(anchors)

    resolved: Dict[Tuple[str, str, str, str], str] = {}

    def decide(nk: str, ind: str, country: str, dom: str) -> str:
        if dom in anchors:
            return dom
        cands: Dict[str, float] = {}
        # domain typo ("whitelewis.con", "whitelewis,com")
        if dom and dom not in FREE_MAIL and dom not in JUNK_DOMAINS and "." in dom:
            for m in difflib.get_close_matches(dom, dom_vocab, n=3, cutoff=0.86):
                cands[m] = cands.get(m, 0) + 5
        # company name (exact or typo)
        if nk:
            names = [nk] if nk in by_nk else (
                [closest(nk, nk_vocab, 0.86)] if closest(nk, nk_vocab, 0.86) else [])
            for name in names:
                bonus = 4 if name == nk else 3
                for d in by_nk[name]:
                    cands[d] = cands.get(d, 0) + bonus
        if not cands:
            return ""
        scored = []
        for d, s in cands.items():
            a = anchors[d]
            if ind and a.industry:
                s += 2 if ind == a.industry else -4
            if country and a.country and country == a.country:
                s += 1
            scored.append((s, d))
        scored.sort(reverse=True)
        best = scored[0]
        second = scored[1][0] if len(scored) > 1 else -99
        if best[0] >= 4 and best[0] - second >= 2:
            return best[1]
        return ""

    out = []
    for nk, ind, country, dom in zip(
        frame["_nk"], frame["Industry (clean)"], frame["_country"], frame["_dom"]
    ):
        key = (nk, ind, country, dom)
        if key not in resolved:
            resolved[key] = decide(*key)
        out.append(resolved[key])
    return pd.Series(out, index=frame.index)


def _company_labels(anchors: Dict[str, _Anchor]) -> Dict[str, str]:
    """Clean dropdown labels. Industry/domain is added ONLY to separate
    different companies that share the same name."""
    by_name: Dict[str, List[_Anchor]] = defaultdict(list)
    for a in anchors.values():
        by_name[a.display.lower()].append(a)
    labels: Dict[str, str] = {}
    for group in by_name.values():
        if len(group) == 1:
            labels[group[0].domain] = group[0].display
            continue
        inds = Counter(a.industry for a in group)
        for a in group:
            if a.industry and inds[a.industry] == 1:
                labels[a.domain] = f"{a.display} \u00b7 {a.industry}"
            else:
                labels[a.domain] = f"{a.display} \u00b7 {a.domain}"
    return labels


# ============================================================
# 8. PEOPLE: NAMES, SKILLS, DUPLICATES
# ============================================================

_NAME_SUFFIX = {"sr", "jr", "ii", "iii", "iv", "phd", "md", "esq"}


def clean_person_name(value: Any) -> str:
    s = re.sub(r"\s+", " ", str(value or "").replace("\u00a0", " ")).strip()
    if is_placeholder(s) or len(re.sub(r"[^A-Za-z]", "", s)) < 2:
        return ""
    if s.count(",") == 1:  # "LAST, First [Middle]"
        last, first = [p.strip() for p in s.split(",")]
        if last and first:
            s = f"{first} {last}"
    tokens = [t for t in s.split() if t.strip(".").lower() not in _NAME_SUFFIX]
    s = " ".join(tokens)
    if s.isupper() or s.islower():
        s = s.title()
    return s


def name_signature(name: str) -> str:
    tokens = re.findall(r"[a-z]+", fold(name))
    tokens = [t for t in tokens if len(t) > 1 and t not in _NAME_SUFFIX]
    return " ".join(sorted(tokens))


def split_skills(raw: Any) -> List[str]:
    parts = re.split(r"[;,|/]+", str(raw or ""))
    return [p for p in (ct(x) for x in parts) if p and p not in PLACEHOLDERS]


class _Voter:
    """feature -> label distribution, learned from the table itself."""

    def __init__(self) -> None:
        self.counts: Dict[str, Counter] = defaultdict(Counter)

    def learn(self, features: Iterable[str], label: str) -> None:
        for f in features:
            self.counts[f][label] += 1

    def predict(self, features: Iterable[str], min_share: float = 0.75) -> str:
        votes: Counter = Counter()
        n = 0
        for f in features:
            c = self.counts.get(f)
            if not c:
                continue
            total = sum(c.values())
            n += 1
            for label, k in c.items():
                votes[label] += k / total
        if not votes or n == 0:
            return ""
        label, score = votes.most_common(1)[0]
        return label if score / n >= min_share else ""


# ============================================================
# 9. STAGE 1 - PREPARE
# ============================================================

@dataclass
class PreparedData:
    employees: pd.DataFrame            # cleaned people, one row per person, with Company Key
    unassigned: pd.DataFrame           # people whose company could not be determined
    companies: pd.DataFrame            # one row per real company
    report: Dict[str, Any] = field(default_factory=dict)


EMPLOYEE_COLUMNS = [
    "Record ID", "Source Rows", "Company Key", "Company Name", "Industry",
    "Employee Name", "Designation", "Seniority Level", "Level Source",
    "Department", "Department Source", "Team/Business Unit", "Team Source",
    "Skills/Specialization", "City", "Country", "Email", "Valid", "Duplicates Merged",
]


def prepare(source, infer_from_skills: bool = True) -> PreparedData:
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    if raw.empty:
        raise ValueError("The uploaded CSV contains no employee records.")
    df = _read_columns(raw)
    df["Source Row"] = range(2, len(df) + 2)          # row number as seen in a spreadsheet
    n_raw = len(df)

    # ---------- unique-value cleaning (fast) ----------
    def mapped(col: str, fn) -> pd.Series:
        uniq = {v: fn(v) for v in df[col].unique()}
        return df[col].map(uniq)

    df["Industry (clean)"] = mapped("Industry", canonical_industry)
    df["_nk"] = mapped("Company Name", company_key)
    df["_dom"] = mapped("Email", email_domain)
    df["_country"] = mapped("Country", ct)

    # ---------- company identity ----------
    anchors = _build_anchors(df)
    df["Company Key"] = _resolve_company_rows(df, anchors)
    labels = _company_labels(anchors)

    # ---------- field cleaning ----------
    df["Employee Name (clean)"] = mapped("Employee Name", clean_person_name)
    df["_title"] = mapped("Designation", title_key)
    df["_title_ok"] = df["_title"].map(lambda t: t not in PLACEHOLDERS and bool(t))
    df["_given_level"] = mapped("Seniority Level", parse_level)
    df["_team"] = mapped("Team/Business Unit", team_key)
    df["_dept"] = mapped("Department", canonical_department)
    df["_skills"] = mapped("Skills/Specialization", lambda v: tuple(split_skills(v)))

    # canonical team vocabulary learned from the table
    team_counts = df.loc[df["_team"] != "", "_team"].value_counts()
    team_vocab = [t for t, n in team_counts.items() if n >= 150]
    team_map = {}
    for t in df["_team"].unique():
        if t == "":
            team_map[t] = ""
        elif t in team_vocab:
            team_map[t] = t
        else:
            m = closest(t, team_vocab, 0.82)
            team_map[t] = m or ""
    df["_team"] = df["_team"].map(team_map)

    # learned relationships --------------------------------------------------
    clear_dept = df["_dept"].map(len) == 1
    team_dept: Dict[str, str] = {}
    td = df[clear_dept & (df["_team"] != "")].groupby("_team")["_dept"].agg(
        lambda s: Counter(x[0] for x in s).most_common(1)[0])
    for team, (dept, n) in td.items():
        total = int(((df["_team"] == team) & clear_dept).sum())
        if total and n / total >= 0.7 and team != LEADERSHIP_TEAM:
            team_dept[team] = dept

    title_dept: Dict[str, str] = {}
    tg = df[clear_dept & df["_title_ok"]].groupby("_title")["_dept"].agg(
        lambda s: (Counter(x[0] for x in s).most_common(1)[0], len(s)))
    for t, ((dept, n), total) in tg.items():
        if total >= 3 and n / total >= 0.75:
            title_dept[t] = dept

    has_level = df["_given_level"] > 0
    title_level: Dict[str, int] = {}
    lg = df[has_level & df["_title_ok"]].groupby("_title")["_given_level"].agg(
        lambda s: (Counter(s).most_common(1)[0], len(s)))
    for t, ((lvl, n), total) in lg.items():
        if total >= 3 and n / total >= 0.6:
            title_level[t] = int(lvl)

    team_voter, dept_voter = _Voter(), _Voter()
    for sk, team, dept in zip(df["_skills"], df["_team"], df["_dept"]):
        if sk and team:
            team_voter.learn(sk, team)
        if sk and len(dept) == 1:
            dept_voter.learn(sk, dept[0])

    # ---------- per-row resolution ----------
    levels, level_src, depts, dept_src, teams, team_src = [], [], [], [], [], []
    for given, tk, ok, team, dept, sk in zip(
        df["_given_level"], df["_title"], df["_title_ok"],
        df["_team"], df["_dept"], df["_skills"],
    ):
        # level
        if given > 0:
            lvl, src = int(given), "column"
        elif ok and tk in title_level:
            lvl, src = title_level[tk], "learned from designation"
        elif ok and rule_level(tk):
            lvl, src = rule_level(tk), "designation keywords"
        else:
            lvl, src = 3, "default (no data)"
        levels.append(lvl)
        level_src.append(src)

        # team
        t_src = "column" if team else ""
        # department
        d, d_src = "", ""
        if len(dept) == 1:
            d, d_src = dept[0], "column"
        elif len(dept) == 2:
            pick = team_dept.get(team) if team else None
            if pick in dept:
                d, d_src = pick, "team"
            elif ok and title_dept.get(tk) in dept:
                d, d_src = title_dept[tk], "designation"
            else:
                d, d_src = dept[0], "column (hybrid)"
        else:
            if team and team in team_dept:
                d, d_src = team_dept[team], "team"
            elif ok and tk in title_dept:
                d, d_src = title_dept[tk], "designation"
        if not d and sk and infer_from_skills:
            guess = dept_voter.predict(sk, 0.8)
            if guess:
                d, d_src = guess, "skills"
        # skills can also recover a blank team (only if it agrees with the department)
        if not team and sk and infer_from_skills and lvl < ROOT_LEVEL:
            guess = team_voter.predict(sk, 0.8)
            if guess and (not d or team_dept.get(guess, d) == d):
                team, t_src = guess, "skills"
                if not d and guess in team_dept:
                    d, d_src = team_dept[guess], "skills"
        depts.append(d)
        dept_src.append(d_src)
        teams.append(team)
        team_src.append(t_src)

    df["Seniority Level"] = levels
    df["Level Source"] = level_src
    df["Department"] = depts
    df["Department Source"] = dept_src
    df["Team/Business Unit"] = teams
    df["Team Source"] = team_src
    df["Team/Business Unit"] = df["Team/Business Unit"].map(pretty_team)
    df["Employee Name"] = df["Employee Name (clean)"]
    df["Industry"] = df["Industry (clean)"]
    df["Skills/Specialization"] = df["_skills"].map(lambda s: "; ".join(s))
    df["Designation"] = df["Designation"].map(
        lambda v: "" if is_placeholder(v) else re.sub(r"\s+", " ", v).strip())
    df["Email"] = df["Email"].map(lambda v: "" if is_placeholder(v) else v.strip().lower())
    df["City"] = df["City"].map(lambda v: "" if is_placeholder(v) else v.title() if v.isupper() or v.islower() else v)
    df["Country"] = df["Country"].map(lambda v: "" if is_placeholder(v) else v)
    df["Valid"] = df["Employee Name"] != ""
    # test / junk rows
    junk_title = df["Designation"].map(ct).isin({"test", "tester", "xxx", "asdf"})
    df.loc[junk_title, "Valid"] = False

    unassigned = df[df["Company Key"] == ""].copy()
    emp = df[df["Company Key"] != ""].copy()
    emp["Company Name"] = emp["Company Key"].map({k: a.display for k, a in anchors.items()})

    # ---------- merge duplicate people ----------
    emp, merged = _merge_duplicates(emp)

    emp["Record ID"] = [f"R{i:06d}" for i in range(1, len(emp) + 1)]
    emp = emp[EMPLOYEE_COLUMNS + ["_title", "_skills", "_country"]].reset_index(drop=True)

    companies = pd.DataFrame([
        {
            "Company Key": a.domain, "Company Name": a.display, "Industry": a.industry,
            "Domain": a.domain, "Label": labels[a.domain],
        } for a in anchors.values()
    ])
    sizes = emp.groupby("Company Key").size()
    companies["Employees"] = companies["Company Key"].map(sizes).fillna(0).astype(int)
    companies = companies.sort_values("Label", key=lambda s: s.str.lower()).reset_index(drop=True)

    ind_map = (
        raw["Industry"].map(lambda v: re.sub(r"\s+", " ", str(v)).strip() or "(blank)")
        .value_counts().rename_axis("Spelling in file").reset_index(name="Rows")
    )
    ind_map["Cleaned to"] = ind_map["Spelling in file"].map(
        lambda v: canonical_industry(v) or "(unknown - taken from the company)")
    report = {
        "industry_map": ind_map,
        "raw_rows": n_raw,
        "people_after_merge": len(emp),
        "duplicates_merged": merged,
        "companies": len(companies),
        "unassigned_rows": len(unassigned),
        "raw_industry_spellings": int(raw["Industry"].nunique()) if "Industry" in raw else 0,
        "industries": int(companies["Industry"].replace("", pd.NA).nunique()),
        "raw_company_spellings": int(raw["Company Name"].nunique()),
        "levels_from_column": int((emp["Level Source"] == "column").sum()),
        "levels_learned": int((emp["Level Source"] == "learned from designation").sum()),
        "levels_keywords": int((emp["Level Source"] == "designation keywords").sum()),
        "levels_default": int((emp["Level Source"] == "default (no data)").sum()),
        "dept_filled": int((~emp["Department Source"].isin(["column", ""])).sum()),
        "dept_missing": int((emp["Department"] == "").sum()),
        "team_from_skills": int((emp["Team Source"] == "skills").sum()),
        "team_missing": int((emp["Team/Business Unit"] == "").sum()),
        "invalid_records": int((~emp["Valid"]).sum()),
    }
    return PreparedData(emp, unassigned[
        ["Source Row", "Company Name", "Industry", "Employee Name", "Designation", "Email"]
    ].reset_index(drop=True), companies, report)


def _merge_duplicates(emp: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Collapse rows describing the same person (same e-mail, or same name + designation)."""
    emp = emp.copy()
    emp["_email_key"] = emp["Email"].where(emp["Email"].str.contains("@", na=False), "")
    emp["_name_sig"] = emp["Employee Name"].map(name_signature)
    emp["_group"] = -1
    next_id = 0
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    idx = list(emp.index)
    for i in idx:
        parent[i] = i
    for cols in (["Company Key", "_email_key"], ["Company Key", "_name_sig", "_title"]):
        sub = emp[(emp[cols[-1]] != "") & (emp["Valid"])]
        if cols[1] == "_name_sig":
            sub = sub[sub["_name_sig"].str.count(" ") >= 1]
        sub = sub[sub.duplicated(cols, keep=False)]
        for _, g in sub.groupby(cols, sort=False):
            first = g.index[0]
            for j in g.index[1:]:
                parent[find(j)] = find(first)
    emp["_group"] = [find(i) for i in idx]

    size = emp.groupby("_group")["_group"].transform("size")
    singles = emp[size == 1].copy()
    singles["Source Rows"] = singles["Source Row"].astype(str)
    singles["Duplicates Merged"] = 0
    multi = emp[size > 1]

    rows, merged = [], 0
    filled = ["Designation", "Department", "Team/Business Unit", "Skills/Specialization",
              "City", "Country", "Email"]
    for _, g in multi.groupby("_group", sort=False):
        merged += len(g) - 1
        score = g[filled].apply(lambda c: c.astype(str).str.len() > 0).sum(axis=1)
        score = score + (g["Level Source"] == "column").astype(int)
        g = g.assign(_score=score).sort_values("_score", ascending=False)
        r = g.iloc[0].to_dict()
        for col in filled + ["_title", "_skills", "_country"]:
            if not r[col]:
                for v in g[col]:
                    if v:
                        r[col] = v
                        break
        if r["Level Source"] != "column":
            col_rows = g[g["Level Source"] == "column"]
            if len(col_rows):
                r["Seniority Level"] = col_rows.iloc[0]["Seniority Level"]
                r["Level Source"] = "column"
        for col, src in (("Department", "Department Source"), ("Team/Business Unit", "Team Source")):
            if r[src] in ("", None) and r[col]:
                r[src] = "merged duplicate"
        r["Source Rows"] = ", ".join(str(x) for x in sorted(g["Source Row"]))
        r["Duplicates Merged"] = len(g) - 1
        rows.append(r)
    out = pd.concat([singles, pd.DataFrame(rows)], ignore_index=True)
    out = out.sort_values("Source Row").reset_index(drop=True)
    return out.drop(columns=["_email_key", "_name_sig", "_group", "_score"], errors="ignore"), merged


# ============================================================
# 10. STAGE 2 - INFERENCE
# ============================================================

EDGE_COLUMNS = [
    "Company Key", "Company Name", "Employee ID", "Employee Name", "Employee Designation",
    "Employee Level", "Employee Team", "Manager ID", "Manager Name", "Manager Designation",
    "Manager Level", "Manager Team", "Confidence", "Runner-up Score", "Score Margin",
    "Decision", "Evidence",
]
UNRESOLVED_COLUMNS = [
    "Company Key", "Record ID", "Employee Name", "Designation", "Level", "Department", "Team",
    "Status", "Reason", "Best Guess Manager ID", "Best Guess Score",
]

STATUS_LABELS = {
    "root": "Company root (CEO)",
    "department_root": "Top of department (no CEO / head above)",
    "ambiguous_root": "Ambiguous root",
    "ambiguous": "Ambiguous",
    "unresolved": "Unassigned",
    "invalid_record": "Invalid record",
}


def _skill_overlap(a: Dict, b: Dict) -> float:
    sa, sb = set(a["_skills"]), set(b["_skills"])
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _soft_score(emp: Dict, cand: Dict) -> Tuple[float, List[str]]:
    """Tie-breaking evidence (0..~33): skills, title words, location."""
    pts, why = 0.0, []
    sk = _skill_overlap(emp, cand)
    if sk > 0:
        pts += sk * 20
        why.append(f"skills overlap {sk:.0%}")
    et = set(emp["_title"].split()) - {"senior", "junior", "associate", "manager", "lead", "of", "and"}
    ct_ = set(cand["_title"].split()) - {"senior", "junior", "associate", "manager", "lead", "of", "and"}
    if et and ct_ and (et & ct_):
        a = len(et & ct_) / len(et)
        pts += a * 8
        why.append(f"title overlap {a:.0%}")
    ec, cc = fold(emp["City"]), fold(cand["City"])
    if ec and ec == cc:
        pts += 4
        why.append("same city")
    elif emp["_country"] and emp["_country"] == cand["_country"]:
        pts += 1
    return pts, why


def infer_company(
    company_df: pd.DataFrame, config: Optional[InferenceConfig] = None
) -> Tuple[List[Dict], List[Dict]]:
    """Infer reporting lines for ONE company (a slice of PreparedData.employees)."""
    cfg = config or InferenceConfig()
    records = company_df.to_dict(orient="records")
    edges: List[Dict] = []
    unresolved: List[Dict] = []
    by_id = {r["Record ID"]: r for r in records}

    def level(r: Dict) -> int:
        return int(r["Seniority Level"])

    def mark(emp: Dict, status: str, reason: str, guess: Optional[Dict] = None,
             score: Optional[float] = None) -> None:
        unresolved.append({
            "Company Key": emp["Company Key"], "Record ID": emp["Record ID"],
            "Employee Name": emp["Employee Name"], "Designation": emp["Designation"],
            "Level": level(emp), "Department": emp["Department"],
            "Team": emp["Team/Business Unit"], "Status": status, "Reason": reason,
            "Best Guess Manager ID": guess["Record ID"] if guess else None,
            "Best Guess Score": round(float(score), 1) if score is not None else None,
        })

    def link(emp: Dict, mgr: Dict, conf: float, runner: float, decision: str,
             evidence: List[str]) -> None:
        conf = max(0.0, min(99.0, conf))
        edges.append({
            "Company Key": emp["Company Key"], "Company Name": emp["Company Name"],
            "Employee ID": emp["Record ID"], "Employee Name": emp["Employee Name"],
            "Employee Designation": emp["Designation"], "Employee Level": level(emp),
            "Employee Team": emp["Team/Business Unit"],
            "Manager ID": mgr["Record ID"], "Manager Name": mgr["Employee Name"],
            "Manager Designation": mgr["Designation"], "Manager Level": level(mgr),
            "Manager Team": mgr["Team/Business Unit"],
            "Confidence": round(conf, 1), "Runner-up Score": round(runner, 1),
            "Score Margin": round(conf - runner, 1), "Decision": decision,
            "Evidence": "; ".join(evidence),
        })

    valid = [r for r in records if r["Valid"]]
    for r in records:
        if not r["Valid"]:
            mark(r, "invalid_record", "Missing / placeholder name or test record.")

    # ---- roots -------------------------------------------------------------
    top = [r for r in valid if level(r) >= ROOT_LEVEL]
    root: Optional[Dict] = None
    if len(top) == 1:
        root = top[0]
    elif len(top) > 1:
        ceos = [r for r in top if re.search(r"chief executive|\bceo\b", r["_title"])]
        if len(ceos) == 1:
            root = ceos[0]
    top_ids = {r["Record ID"] for r in top}
    for r in top:
        if root is not None and r is root:
            mark(r, "root", "CEO - top of the company; no manager by design.")
        elif root is None:
            mark(r, "ambiguous_root", "Several level-10 people and no single CEO.")

    # Extra level-10 people next to a unique CEO act as executives under the CEO.
    heads = [r for r in valid if level(r) == HEAD_LEVEL or
             (level(r) >= ROOT_LEVEL and root is not None and r is not root)]
    managers = [r for r in valid if level(r) == MANAGER_LEVEL]
    people = [r for r in valid if level(r) < MANAGER_LEVEL]

    heads_by_dept: Dict[str, List[Dict]] = defaultdict(list)
    for h in heads:
        heads_by_dept[h["Department"]].append(h)

    def pick(emp: Dict, pool: List[Dict], base: float, decision: str,
             why: List[str], tie_label: str, rank=None) -> bool:
        """Choose the best of `pool`. Returns True if an edge/unresolved mark was written."""
        scored = []
        for cand in pool:
            soft, soft_why = _soft_score(emp, cand)
            r = rank(cand) if rank else 0
            scored.append((r, soft, cand, soft_why))
        scored.sort(key=lambda s: (-s[0], -s[1], s[2]["Record ID"]))
        best_rank = scored[0][0]
        scored = [s for s in scored if s[0] == best_rank]
        top_soft = scored[0][1]
        runner = scored[1][1] if len(scored) > 1 else -1
        if len(scored) == 1 or top_soft - runner >= cfg.min_margin:
            _, soft, cand, soft_why = scored[0]
            conf = base + min(soft, 20) * (0.35 if len(pool) == 1 else 0.75)
            if conf < cfg.min_confidence:
                mark(emp, "unresolved",
                     f"Best confidence {conf:.1f} is below the threshold {cfg.min_confidence}.",
                     cand, conf)
                return True
            link(emp, cand, conf, base + max(runner, 0) * 0.75 if len(scored) > 1 else 0.0,
                 decision, why + soft_why)
            return True
        tied = [s for s in scored if top_soft - s[1] < cfg.min_margin]
        if not cfg.guess_between_ties:
            mark(emp, "ambiguous",
                 f"{len(tied)} {tie_label} are equally likely (score gap < {cfg.min_margin}).",
                 tied[0][2], base + tied[0][1] * 0.75)
            return True
        k = len(tied)
        # deterministic spread: pick by position of the employee among the tied managers
        _, soft, cand, soft_why = tied[int(emp["Record ID"][1:]) % k]
        conf = base + min(soft, 20) * 0.75 - 25 * (1 - 1 / k)
        if conf < cfg.min_confidence:
            mark(emp, "ambiguous", f"{k} {tie_label} equally likely; confidence {conf:.1f} "
                 f"below the threshold.", cand, conf)
            return True
        link(emp, cand, conf, conf - 5, "tie_guess",
             why + soft_why + [f"{k} equally likely - assigned by spread, treat as a guess"])
        return True

    # ---- level 9 -> CEO ------------------------------------------------------
    for h in heads:
        if root is not None:
            link(h, root, 92, 0.0, "head_to_ceo",
                 [f"level {level(h)} reports to the CEO", f"department: {h['Department'] or 'n/a'}"])
        else:
            mark(h, "department_root", "No unique CEO in the data; top of their department.")

    # ---- level 8 -> department head -------------------------------------------
    def head_pool(dept: str) -> List[Dict]:
        return heads_by_dept.get(dept, []) if dept else []

    for m in managers:
        dept = m["Department"]
        pool = head_pool(dept)
        if pool:
            base = 85 + (5 if len(pool) == 1 else 0)
            why = [f"manager reports to the head of {dept}"]
            if m["Department Source"] not in ("column", ""):
                base -= 6
                why.append(f"department inferred from {m['Department Source']}")
            pick(m, pool, base, "manager_to_head", why, "department heads",
                 rank=lambda c: head_rank(c["_title"]))
        elif root is not None and cfg.allow_skip_level and dept:
            link(m, root, 64, 0.0, "manager_skip_to_ceo",
                 [f"no head of {dept} in the data; reports to the CEO (skip level)"])
        elif root is not None and cfg.allow_skip_level:
            mark(m, "unresolved", "Department unknown, so the head cannot be identified.")
        else:
            mark(m, "department_root", f"No head of {dept or 'department'} in the data.")

    # ---- individual contributors -> team manager ---------------------------------
    mgr_by_team: Dict[str, List[Dict]] = defaultdict(list)
    mgr_by_dept: Dict[str, List[Dict]] = defaultdict(list)
    for m in managers:
        if m["Team/Business Unit"]:
            mgr_by_team[m["Team/Business Unit"]].append(m)
        if m["Department"]:
            mgr_by_dept[m["Department"]].append(m)

    for p in people:
        dept, team = p["Department"], p["Team/Business Unit"]
        team_pool = [
            m for m in mgr_by_team.get(team, [])
            if not (dept and m["Department"] and dept != m["Department"])
        ] if team and team.lower() != LEADERSHIP_TEAM else []

        if team_pool:
            inferred = p["Team Source"] == "skills" or any(m["Team Source"] == "skills" for m in team_pool)
            base = (90 if len(team_pool) == 1 else 72) - (7 if inferred else 0)
            why = [f"same team ({team})"]
            if p["Team Source"] == "skills":
                why.append("team inferred from skills")
            pick(p, team_pool, base, "team_manager", why, "managers on this team")
            continue

        # no manager on the team (or team unknown)
        dept_pool = mgr_by_dept.get(dept, []) if dept else []
        if dept_pool and (not team or not mgr_by_team.get(team)) and not team:
            # team unknown: only accept a manager if the evidence separates them
            if len(dept_pool) == 1:
                pick(p, dept_pool, 66, "department_manager",
                     [f"team unknown; only manager in {dept}"], "managers")
                continue
        head = head_pool(dept)
        if head and cfg.allow_skip_level:
            why = [f"no manager on team {team or 'unknown'}; reports to the head of {dept} (skip level)"]
            base = 63 - (4 if p["Department Source"] not in ("column", "") else 0)
            pick(p, head, base, "skip_to_head", why, "department heads",
                 rank=lambda c: head_rank(c["_title"]))
        elif not dept:
            mark(p, "unresolved", "Department and team unknown; nothing to match on.")
        elif not cfg.allow_skip_level:
            mark(p, "unresolved", f"No manager on team {team or 'unknown'} and skip-level "
                 f"assignment is switched off.")
        else:
            mark(p, "unresolved", f"No manager on team {team or 'unknown'} and no head of {dept} "
                 f"in the data.")

    # ---- safety net ---------------------------------------------------------------
    safe: List[Dict] = []
    for e in edges:
        mgr, emp = by_id[e["Manager ID"]], by_id[e["Employee ID"]]
        if level(mgr) < MANAGER_MIN_LEVEL or level(mgr) <= level(emp):
            mark(emp, "unresolved", f"Rejected: level {level(mgr)} cannot manage level {level(emp)}.")
        else:
            safe.append(e)
    return safe, unresolved


def infer_all(
    employees: pd.DataFrame, config: Optional[InferenceConfig] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    edges: List[Dict] = []
    unresolved: List[Dict] = []
    for _, company_df in employees.groupby("Company Key", sort=False):
        e, u = infer_company(company_df, config)
        edges.extend(e)
        unresolved.extend(u)
    return (pd.DataFrame(edges, columns=EDGE_COLUMNS),
            pd.DataFrame(unresolved, columns=UNRESOLVED_COLUMNS))
