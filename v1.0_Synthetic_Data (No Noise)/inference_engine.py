"""
Org Chart Inference Engine – v3 (Seniority Level / Team / Skills aware)

Input columns
-------------
Company Name, Industry, Designation, Seniority Level, Department,
Team/Business Unit, Employee Name, Skills/Specialization, City, Country, Email

Only the original seven columns are mandatory. Industry, Seniority Level,
Team/Business Unit and Skills/Specialization are used when present, and the
engine falls back to title-based inference when they are missing.

Seniority scale (matches Generate_Data.py)
------------------------------------------
    10 CEO                      -> company root
     9 Department head (VP/Director)
     8 Manager                  -> leads a team
     6 Lead / 5 Senior / 3 Mid / 1 Junior  -> individual contributors

HARD RULE
---------
Only people at level >= 8 can ever have direct reports.  Levels 6, 5, 3 and 1
(and any other level < 8) are never chosen as a manager.  This is enforced
twice: candidates are filtered by level before scoring, and a final safety
pass removes any edge that would violate it.

Inference model
---------------
1. Reporting is strictly "up the ladder": the manager's level must be higher
   than the employee's, so cycles are impossible by construction.
2. Each person looks at the NEAREST populated tier above them
   (IC -> managers, manager -> department head, head -> CEO).  A tier is
   skipped only if nobody in it belongs to the person's department, and the
   skip is penalised.
3. Candidates in that tier are scored.  Strong ("structural") evidence:
       department match   > company-wide CEO > team match > (title/skills when
       no team data exists)
   Weak ("soft") evidence, used only as tie-breakers:
       title overlap, skills overlap, city/country, e-mail domain.
4. If several managers are structurally indistinguishable (e.g. two managers
   leading the same team), employees are spread over them with a
   span-of-control cap and the edge is reported with reduced confidence.
5. Wrong edges are worse than none: anything below MIN_CONFIDENCE, or without
   a clear structural winner, goes to the unresolved list with a best guess.
6. Same-named companies in different industries are separated automatically.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ============================================================
# 1. CONFIGURATION (tunable via the Streamlit UI)
# ============================================================

MIN_CONFIDENCE = 58
MIN_SCORE_MARGIN = 4
SECOND_PASS_MATCHED_SAMPLE_PCT = 100  # kept for UI compatibility

# --- seniority ladder -------------------------------------------------
ROOT_LEVEL = 10          # CEO
HEAD_LEVEL = 9           # department heads
MANAGER_LEVEL = 8        # team managers
MANAGER_MIN_LEVEL = 8    # ONLY levels >= 8 may have direct reports

# --- structural scoring ----------------------------------------------
BASE_SCORE = 55
SKIPPED_TIER_PENALTY = 10
COMPANY_WIDE_BONUS = 35
DEPT_POINTS = {
    "exact": 25,
    "contains": 19,
    "title_mentions_dept": 14,
    "function": 8,
}
TEAM_MATCH_BONUS = 25
TEAM_PARTIAL_BONUS = 15
TEAM_MISMATCH_PENALTY = 8

# Used as STRUCTURAL evidence only when team data is missing
TITLE_AFFINITY_BONUS = 24
SKILLS_STRUCTURAL_BONUS = 20

# --- soft tie-breakers --------------------------------------------------
SOFT_TITLE_BONUS = 4
SOFT_SKILLS_BONUS = 4
SAME_CITY_BONUS = 2
SAME_COUNTRY_BONUS = 1
SAME_DOMAIN_BONUS = 1

# --- managers that cannot be told apart --------------------------------
ALLOW_SHARED_TEAM_ASSIGNMENT = False  # True = guess between managers who share a team
SHARED_TEAM_PENALTY = 40          # confidence lost as 40 * (1 - 1/k)
SHARED_TEAM_CAPACITY_SLACK = 0    # extra reports allowed above an even split

_RANK_WORDS = {
    "senior", "junior", "assistant", "associate", "intern", "trainee",
    "staff", "principal", "lead", "leader", "manager", "management",
    "director", "head", "vice", "president", "chief", "officer",
    "executive", "general", "team", "sr", "jr", "of", "the", "and",
    "a", "an", "level", "i", "ii", "iii", "iv", "v", "vp", "svp", "evp",
}

REQUIRED_COLUMNS = [
    "Company Name",
    "Designation",
    "Department",
    "Employee Name",
    "City",
    "Country",
    "Email",
]

OPTIONAL_COLUMN_ALIASES = {
    "Seniority Level": ["seniority level", "seniority", "level"],
    "Industry": ["industry"],
    "Team/Business Unit": [
        "team/business unit", "team business unit", "team", "business unit",
    ],
    "Skills/Specialization": [
        "skills/specialization", "skills specialization", "skills",
        "specialization", "specialisation",
    ],
}

# ============================================================
# 2. TITLE / DEPARTMENT NORMALISATION
# ============================================================

TITLE_REPLACEMENTS = {
    r"\bceo\b": "chief executive officer",
    r"\bcoo\b": "chief operating officer",
    r"\bcfo\b": "chief financial officer",
    r"\bcto\b": "chief technology officer",
    r"\bciso\b": "chief information security officer",
    r"\bcio\b": "chief information officer",
    r"\bcmo\b": "chief marketing officer",
    r"\bchro\b": "chief human resources officer",
    r"\bcpo\b": "chief product officer",
    r"\bcro\b": "chief revenue officer",
    r"\bevp\b": "executive vice president",
    r"\bsvp\b": "senior vice president",
    r"\bvp\b": "vice president",
    r"\bsr\.?\b": "senior",
    r"\bjr\.?\b": "junior",
    r"\bmgr\.?\b": "manager",
    r"\beng\.?\b": "engineer",
    r"\bhr\b": "human resources",
    r"\bqa\b": "quality assurance",
    r"\bit\b": "information technology",
    r"\bops\b": "operations",
    r"\bdir\.?\b": "director",
    r"\basst\.?\b": "assistant",
    r"\bassoc\.?\b": "associate",
}

TITLE_ALIASES = {
    "chief executive": "chief executive officer",
    "chief finance officer": "chief financial officer",
    "chief technology": "chief technology officer",
    "head engineering": "head of engineering",
    "head sales": "head of sales",
    "head finance": "head of finance",
    "head marketing": "head of marketing",
    "head product": "head of product",
    "head hr": "head of human resources",
    "head human resources": "head of human resources",
}

DEPARTMENT_ALIASES = {
    "hr": "human resources",
    "people": "human resources",
    "talent": "human resources",
    "talent acquisition": "human resources",
    "people operations": "human resources",
    "accounting": "finance",
    "finance and accounting": "finance",
    "fin": "finance",
    "technology": "engineering",
    "software": "engineering",
    "software engineering": "engineering",
    "it": "information technology",
    "information systems": "information technology",
    "supply chain": "operations",
    "business operations": "operations",
    "ops": "operations",
    "account management": "sales",
    "business development": "sales",
    "bd": "sales",
    "customer success": "sales",
    "cs": "sales",
    "growth": "marketing",
    "brand": "marketing",
    "content": "marketing",
    "product management": "product",
    "pm": "product",
    "legal and compliance": "legal",
    "compliance": "legal",
    "data science": "data",
    "analytics": "data",
    "bi": "data",
    "business intelligence": "data",
}

# Labels some versions of Generate_Data.py wrote instead of numbers
LEVEL_LABELS = {
    "c-level": 10, "c level": 10, "ceo": 10,
    "director vp": 9, "director": 9, "vp": 9,
    "manager": 8,
    "lead": 6,
    "senior": 5,
    "mid level": 3, "mid-level": 3, "mid": 3,
    "entry level": 1, "entry-level": 1, "entry": 1, "junior": 1,
}

# ============================================================
# 3. FUNCTION KEYWORDS (fallback when department is unusable)
# ============================================================

FUNCTION_KEYWORDS = {
    "engineering": [
        "engineer", "engineering", "developer", "development", "software",
        "technical", "infrastructure", "devops", "sre", "architect",
        "quality assurance", "qa", "security", "platform", "backend",
        "frontend", "full stack", "fullstack",
    ],
    "finance": [
        "finance", "financial", "accountant", "accounting", "treasury",
        "tax", "audit", "controller", "fp&a", "fpa",
    ],
    "sales": [
        "sales", "account executive", "business development", "revenue",
        "account manager", "customer success", "ae ", "sdr", "bdr",
    ],
    "marketing": [
        "marketing", "brand", "content", "social media", "public relations",
        "growth", "demand generation", "product marketing",
    ],
    "human resources": [
        "human resources", "recruit", "talent", "people", "compensation",
        "benefits", "hrbp", "people operations",
    ],
    "operations": [
        "operations", "supply chain", "logistics", "process",
        "program manager", "project manager", "pmo",
    ],
    "product": ["product", "product manager", "product owner", "pm "],
    "legal": ["legal", "counsel", "compliance", "attorney", "lawyer"],
    "data": [
        "data", "analytics", "business intelligence", "data scientist",
        "data engineer", "data analyst", "bi ",
    ],
    "design": ["design", "designer", "ux", "ui", "product design", "creative"],
    "customer support": [
        "support", "customer support", "customer service", "helpdesk",
        "technical support",
    ],
}

# Layers (0 = CEO … 3 = individual contributor)
LAYER_BOUNDS = [(ROOT_LEVEL, 0), (HEAD_LEVEL, 1), (MANAGER_LEVEL, 2), (0, 3)]

# ============================================================
# 4. TEXT HELPERS
# ============================================================

def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    value = str(value).strip().lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[_/|]+", " ", value)
    value = re.sub(r"[^a-z0-9\s-]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def is_gibberish(value: Any) -> bool:
    text = clean_text(value)
    return not text or not re.search(r"[a-z]", text) or len(text) < 2


def validate_record(row: Dict) -> Tuple[bool, Optional[str]]:
    if not row["Company Name"].strip():
        return False, "Missing Company Name"
    if not row["Employee Name"].strip():
        return False, "Missing Employee Name"
    if not row["Designation"].strip():
        return False, "Missing Designation"
    if is_gibberish(row["Employee Name"]):
        return False, "Invalid Employee Name"
    if is_gibberish(row["Designation"]):
        return False, "Invalid Designation"
    return True, None


def normalize_title(value: Any) -> str:
    title = clean_text(value)
    for pattern, replacement in TITLE_REPLACEMENTS.items():
        title = re.sub(pattern, replacement, title)
    return TITLE_ALIASES.get(title, title)


def normalize_department(value: Any) -> str:
    department = clean_text(value)
    return DEPARTMENT_ALIASES.get(department, department)


def infer_function(title: str, department: str) -> str:
    matches = [
        function for function, keywords in FUNCTION_KEYWORDS.items()
        if any(kw in title for kw in keywords)
    ]
    if matches:
        return matches[0]
    return DEPARTMENT_ALIASES.get(department, department or "unknown")


# ============================================================
# 5. SENIORITY LEVEL
# ============================================================

def is_ceo_title(title: str) -> bool:
    title = clean_text(title)
    return (
        "chief executive officer" in title
        or title == "ceo"
        or title.startswith("co chief executive")
    )


def is_possible_root(title: str) -> bool:
    title = clean_text(title)
    return (
        is_ceo_title(title)
        or title == "president"
        or title.startswith("president ")
    )


def derive_level_from_title(title: str) -> int:
    """Title -> 1..10 level, using the same scale as Generate_Data.py."""
    t = clean_text(title)
    if is_ceo_title(t):
        return 10
    if re.search(r"\bpresident\b", t) and "vice president" not in t:
        return 10
    if re.search(
        r"\bchief\b|\bvice president\b|\bdirector\b|\bhead\b|\bfounder\b|\bchair", t
    ):
        return 9
    if re.search(r"\bmanager\b|\bsupervisor\b", t):
        return 8
    if re.search(r"\blead(er)?\b(?! gen)", t):
        return 6
    if re.search(r"\bsenior\b|\bprincipal\b|\bstaff\b", t):
        return 5
    if re.search(r"\bjunior\b|\bassociate\b|\bassistant\b|\bintern\b|\btrainee\b", t):
        return 1
    return 3


def get_seniority(title: str) -> int:
    """Title-based level (1..10). Used when 'Seniority Level' is missing."""
    return derive_level_from_title(normalize_title(title))


def parse_seniority_level(value: Any, title: str) -> int:
    """Use the Seniority Level column when valid; otherwise derive from title."""
    if value is not None and not (isinstance(value, float) and math.isnan(value)):
        raw = str(value).strip()
        if raw:
            try:
                level = int(float(raw))
                if 1 <= level <= 10:
                    return level
            except (ValueError, OverflowError):
                pass
            label = LEVEL_LABELS.get(clean_text(raw))
            if label is not None:
                return label
    return derive_level_from_title(title)


def get_layer(seniority: int) -> int:
    for minimum, layer in LAYER_BOUNDS:
        if seniority >= minimum:
            return layer
    return 3


def has_leadership_title(title: str) -> bool:
    """True if the title maps to a level that is allowed to manage people."""
    return get_seniority(title) >= MANAGER_MIN_LEVEL


# ============================================================
# 6. DATA LOADING
# ============================================================

def _assign_company_keys(df: pd.DataFrame) -> pd.Series:
    """
    Company Key = cleaned company name.  If the same name clearly belongs to
    several companies (several CEO records AND several industries), the
    industry is appended so their org charts are not mixed together.
    """
    base = df["Company Name"].map(clean_text)
    keys = base.copy()
    if not (df["Industry"].str.strip() != "").any():
        return keys

    industry_key = df["Industry"].map(clean_text)
    is_root = df["Seniority"] >= ROOT_LEVEL
    for _, index in base.groupby(base).groups.items():
        if is_root.loc[index].sum() < 2:
            continue
        if industry_key.loc[index].replace("", pd.NA).dropna().nunique() < 2:
            continue
        keys.loc[index] = base.loc[index] + " | " + industry_key.loc[index]
    return keys


def load_data(file_path) -> pd.DataFrame:
    df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    if df.empty:
        raise ValueError("The uploaded CSV contains no employee records.")

    lookup = {str(col).strip().lower(): col for col in df.columns}
    rename: Dict[str, str] = {}

    for col in REQUIRED_COLUMNS:
        if col.lower() not in lookup:
            raise ValueError(f"Missing required column: {col}")
        rename[lookup[col.lower()]] = col

    present_optional = []
    for canonical, aliases in OPTIONAL_COLUMN_ALIASES.items():
        for alias in [canonical.lower()] + aliases:
            source = lookup.get(alias)
            if source is not None and source not in rename:
                rename[source] = canonical
                present_optional.append(canonical)
                break

    for alias in ("employee id", "employee_id", "employeeid"):
        if alias in lookup and lookup[alias] not in rename:
            rename[lookup[alias]] = "Employee ID"
            break

    df = df.rename(columns=rename)
    keep = list(REQUIRED_COLUMNS) + present_optional
    if "Employee ID" in df.columns:
        keep.append("Employee ID")
    df = df[keep].copy()

    for canonical in OPTIONAL_COLUMN_ALIASES:
        if canonical not in df.columns:
            df[canonical] = ""

    df.insert(0, "Record ID", [f"R{i:08d}" for i in range(1, len(df) + 1)])

    text_cols = REQUIRED_COLUMNS + list(OPTIONAL_COLUMN_ALIASES)
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()

    if "Employee ID" in df.columns:
        df["Employee ID"] = (
            df["Employee ID"].fillna("").astype(str).str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
        df.loc[
            df["Employee ID"].str.lower().isin({"nan", "none", "null"}), "Employee ID"
        ] = ""

    df["Normalized Title"] = df["Designation"].map(normalize_title)
    df["Normalized Department"] = df["Department"].map(normalize_department)
    df["Normalized Team"] = df["Team/Business Unit"].map(clean_text)
    df["Normalized City"] = df["City"].map(clean_text)
    df["Normalized Country"] = df["Country"].map(clean_text)
    df["Email Domain"] = (
        df["Email"].str.lower().str.extract(r"@([^@]+)$", expand=False).fillna("")
    )
    df["Function"] = [
        infer_function(t, d)
        for t, d in zip(df["Normalized Title"], df["Normalized Department"])
    ]

    df["Seniority Level"] = [
        parse_seniority_level(v, t)
        for v, t in zip(df["Seniority Level"], df["Normalized Title"])
    ]
    df["Seniority"] = df["Seniority Level"]          # name used by older code
    df["Layer"] = df["Seniority"].map(get_layer)
    df["Company Key"] = _assign_company_keys(df)

    return df


# ============================================================
# 7. EVIDENCE HELPERS
# ============================================================

def _lvl(row: Dict) -> int:
    value = row.get("Seniority")
    try:
        return int(value)
    except (TypeError, ValueError):
        return get_seniority(row.get("Normalized Title", ""))


def _title_tokens(title: str) -> set:
    return {
        tok for tok in re.findall(r"[a-z0-9]+", title or "")
        if len(tok) > 2 and tok not in _RANK_WORDS
    }


def _dept_tokens(dept: str) -> set:
    return _title_tokens(dept)


@lru_cache(maxsize=None)
def _skill_set(raw: str) -> frozenset:
    parts = re.split(r"[;,|/]+", raw or "")
    return frozenset(p for p in (clean_text(x) for x in parts) if p)


def _skill_overlap(employee: Dict, candidate: Dict) -> float:
    a = _skill_set(employee.get("Skills/Specialization", ""))
    b = _skill_set(candidate.get("Skills/Specialization", ""))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _title_affinity(employee: Dict, candidate: Dict) -> float:
    emp_toks = _title_tokens(employee.get("Normalized Title", ""))
    mgr_toks = _title_tokens(candidate.get("Normalized Title", ""))
    dept_toks = _dept_tokens(employee.get("Normalized Department", ""))
    if dept_toks and mgr_toks & dept_toks:
        emp_toks = emp_toks | dept_toks
        mgr_toks = mgr_toks | dept_toks
    if not emp_toks or not mgr_toks:
        return 0.0
    return len(emp_toks & mgr_toks) / len(emp_toks)


def _team_relation(employee: Dict, candidate: Dict) -> str:
    """exact | partial | different | unknown"""
    a = employee.get("Normalized Team", "") or ""
    b = candidate.get("Normalized Team", "") or ""
    if not a or not b:
        return "unknown"
    if a == b:
        return "exact"
    ta, tb = _title_tokens(a), _title_tokens(b)
    if ta and tb and len(ta & tb) / len(ta | tb) >= 0.5:
        return "partial"
    return "different"


def _dept_relation(employee: Dict, candidate: Dict) -> str:
    """exact > contains > title_mentions_dept > function > none"""
    emp_dept = employee.get("Normalized Department", "") or ""
    mgr_dept = candidate.get("Normalized Department", "") or ""
    if emp_dept and mgr_dept and emp_dept == mgr_dept:
        return "exact"
    et, ct = _dept_tokens(emp_dept), _dept_tokens(mgr_dept)
    if et and ct and (et <= ct or ct <= et):
        return "contains"
    if et and (et & _title_tokens(candidate.get("Normalized Title", "")) or et & ct):
        return "title_mentions_dept"
    emp_func = employee.get("Function", "unknown") or "unknown"
    mgr_func = candidate.get("Function", "unknown") or "unknown"
    if emp_func != "unknown" and emp_func == mgr_func:
        return "function"
    return "none"


_STRONG_RELATIONS = {"exact", "contains", "title_mentions_dept"}
_WEAK_RELATIONS = {"function"}

_DEPT_REASON = {
    "exact": "same department",
    "contains": "department contains match",
    "title_mentions_dept": "manager title matches department",
    "function": "same function",
}


# ============================================================
# 8. CANDIDATE POOL + SCORING
# ============================================================

def _find_pool(
    employee: Dict, candidates: List[Dict]
) -> Optional[Tuple[List[Tuple[Dict, str]], int, str]]:
    """
    Pick the nearest tier above the employee that has someone from the same
    department.  Only level >= MANAGER_MIN_LEVEL people are ever considered.
    Returns (pool of (candidate, relation), tier penalty, note) or None.
    """
    emp_level = _lvl(employee)
    above = [
        c for c in candidates
        if c["Record ID"] != employee["Record ID"]
        and _lvl(c) >= MANAGER_MIN_LEVEL
        and _lvl(c) > emp_level
    ]
    if not above:
        return None

    tiers = sorted({_lvl(c) for c in above})
    nearest = tiers[0]

    for allowed in (_STRONG_RELATIONS, _WEAK_RELATIONS):
        for tier in tiers:
            # Individual contributors never report straight to the CEO.
            if tier >= ROOT_LEVEL and emp_level < MANAGER_MIN_LEVEL:
                continue
            pool: List[Tuple[Dict, str]] = []
            for cand in above:
                if _lvl(cand) != tier:
                    continue
                if tier >= ROOT_LEVEL:
                    pool.append((cand, "company"))
                    continue
                relation = _dept_relation(employee, cand)
                if relation in allowed:
                    pool.append((cand, relation))
            if pool:
                if tier == nearest:
                    return pool, 0, ""
                return (
                    pool,
                    SKIPPED_TIER_PENALTY,
                    f"no level-{nearest} leader in department; skipped to level {tier}",
                )
    return None


def _score_candidate(
    employee: Dict,
    candidate: Dict,
    relation: str,
    tier_penalty: int,
    tier_note: str,
) -> Dict:
    emp_level, cand_level = _lvl(employee), _lvl(candidate)
    reasons: List[str] = [f"level {cand_level} above level {emp_level}"]

    structural = float(BASE_SCORE - tier_penalty)
    if tier_note:
        reasons.append(tier_note)

    # --- department / company-wide leader -------------------------------
    if relation == "company":
        structural += COMPANY_WIDE_BONUS
        reasons.append("company-wide CEO")
    else:
        structural += DEPT_POINTS[relation]
        reasons.append(_DEPT_REASON[relation])

    # --- team (IC -> team manager) ---------------------------------------
    team_known = False
    if emp_level < MANAGER_LEVEL and cand_level == MANAGER_LEVEL:
        team_rel = _team_relation(employee, candidate)
        if team_rel == "exact":
            team_known = True
            structural += TEAM_MATCH_BONUS
            reasons.append(f"same team ({employee.get('Team/Business Unit', '')})")
        elif team_rel == "partial":
            team_known = True
            structural += TEAM_PARTIAL_BONUS
            reasons.append("similar team name")
        elif team_rel == "different":
            team_known = True
            structural -= TEAM_MISMATCH_PENALTY
            reasons.append("different team")

    affinity = _title_affinity(employee, candidate)
    skills = _skill_overlap(employee, candidate)
    soft = 0.0

    if emp_level < MANAGER_LEVEL and cand_level == MANAGER_LEVEL and not team_known:
        # No usable team labels: title + skills must separate the managers.
        structural += affinity * TITLE_AFFINITY_BONUS
        structural += skills * SKILLS_STRUCTURAL_BONUS
        if affinity > 0:
            reasons.append(f"title overlap {affinity:.0%}")
        if skills > 0:
            reasons.append(f"skills overlap {skills:.0%}")
    else:
        soft += affinity * SOFT_TITLE_BONUS + skills * SOFT_SKILLS_BONUS
        if skills >= 0.34:
            reasons.append(f"skills overlap {skills:.0%}")

    # --- location / domain tie-breakers -----------------------------------
    emp_city = employee.get("Normalized City", "") or ""
    cand_city = candidate.get("Normalized City", "") or ""
    emp_country = employee.get("Normalized Country", "") or ""
    cand_country = candidate.get("Normalized Country", "") or ""
    if emp_city and emp_city == cand_city:
        soft += SAME_CITY_BONUS
        reasons.append("same city")
    elif emp_country and emp_country == cand_country:
        soft += SAME_COUNTRY_BONUS
    if (
        employee.get("Email Domain")
        and employee.get("Email Domain") == candidate.get("Email Domain")
    ):
        soft += SAME_DOMAIN_BONUS

    return {
        "candidate": candidate,
        "structural": structural,
        "soft": soft,
        "total": max(0.0, min(100.0, structural + soft)),
        "reasons": reasons,
        "relation": relation,
    }


# ============================================================
# 9. MAIN INFERENCE
# ============================================================

def _infer_single_company(records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    edges: List[Dict] = []
    unresolved: List[Dict] = []

    def mark_unresolved(
        emp: Dict, status: str, reason: str,
        guess: Optional[Dict] = None, score: Optional[float] = None,
    ):
        unresolved.append({
            "Record ID": emp["Record ID"],
            "Company Name": emp["Company Name"],
            "Employee Name": emp["Employee Name"],
            "Designation": emp["Designation"],
            "Status": status,
            "Reason": reason,
            "Best Guess Manager ID": guess["Record ID"] if guess else None,
            "Best Guess Score": round(float(score), 1) if score is not None else None,
        })

    def add_edge(emp, mgr, conf, runner, margin, evidence, decision):
        edges.append({
            "Company Name": emp["Company Name"],
            "Employee Record ID": emp["Record ID"],
            "Employee Name": emp["Employee Name"],
            "Employee Designation": emp["Designation"],
            "Employee Level": _lvl(emp),
            "Manager Record ID": mgr["Record ID"],
            "Manager Name": mgr["Employee Name"],
            "Manager Designation": mgr["Designation"],
            "Manager Level": _lvl(mgr),
            "Manager Team": mgr.get("Team/Business Unit", ""),
            "Confidence": round(float(conf), 1),
            "Runner-up Score": round(float(runner), 1),
            "Score Margin": round(float(margin), 1),
            "Evidence": evidence,
            "Decision": decision,
        })

    # ------------------------------------------------------------------
    # Pre-validation
    # ------------------------------------------------------------------
    valid: List[Dict] = []
    for row in records:
        ok, reason = validate_record(row)
        if ok:
            valid.append(row)
        else:
            mark_unresolved(row, "invalid_record", reason)
    by_id = {r["Record ID"]: r for r in valid}

    # ------------------------------------------------------------------
    # Roots (level 10)
    # ------------------------------------------------------------------
    ceos = [r for r in valid if _lvl(r) >= ROOT_LEVEL]
    unique_ceo = ceos[0] if len(ceos) == 1 else None
    for ceo in ceos:
        if unique_ceo is not None:
            mark_unresolved(
                ceo, "possible_root",
                "CEO identified as company root; no manager assigned.",
            )
        else:
            mark_unresolved(
                ceo, "ambiguous_root",
                "Multiple CEO records; cannot safely select a single root.",
            )
    ceo_ids = {c["Record ID"] for c in ceos}

    # Only level >= 8 people can ever be a manager.
    candidates = [
        r for r in valid
        if _lvl(r) >= MANAGER_MIN_LEVEL
        and (r["Record ID"] not in ceo_ids or r is unique_ceo)
    ]

    # ------------------------------------------------------------------
    # Assign a manager to everybody else (top-down, deterministic order)
    # ------------------------------------------------------------------
    employees = sorted(
        (r for r in valid if r["Record ID"] not in ceo_ids),
        key=lambda r: (-_lvl(r), r["Record ID"]),
    )

    shared_groups: Dict[frozenset, List[Tuple[Dict, List[Dict], float]]] = defaultdict(list)

    for emp in employees:
        found = _find_pool(emp, candidates)
        if found is None:
            if _lvl(emp) >= MANAGER_MIN_LEVEL:
                mark_unresolved(
                    emp, "department_root",
                    "Top-level leader; no eligible superior found.",
                )
            else:
                mark_unresolved(
                    emp, "unresolved",
                    "No manager-level (8+) employee found in the department.",
                )
            continue

        pool, penalty, note = found
        scored = [
            _score_candidate(emp, cand, rel, penalty, note) for cand, rel in pool
        ]
        scored.sort(
            key=lambda s: (-s["structural"], -s["soft"], s["candidate"]["Record ID"])
        )
        best_structural = scored[0]["structural"]
        top = [s for s in scored if best_structural - s["structural"] < MIN_SCORE_MARGIN]
        rest = [s for s in scored if best_structural - s["structural"] >= MIN_SCORE_MARGIN]
        runner_structural = max((s["structural"] for s in rest), default=0.0)

        if len(top) == 1:
            best = top[0]
            confidence = best["total"]
            margin = best["structural"] - runner_structural
            if confidence < MIN_CONFIDENCE:
                mark_unresolved(
                    emp, "unresolved",
                    f"Best confidence {confidence:.1f} below threshold {MIN_CONFIDENCE}.",
                    best["candidate"], confidence,
                )
                continue
            decision = {
                (HEAD_LEVEL, ROOT_LEVEL): "department_head_to_ceo",
            }.get((_lvl(emp), _lvl(best["candidate"])))
            if decision is None:
                if _lvl(best["candidate"]) >= ROOT_LEVEL:
                    decision = "manager_escalated_to_ceo"
                elif "same team" in " ".join(best["reasons"]):
                    decision = "team_match"
                else:
                    decision = "department_hierarchy"
            add_edge(
                emp, best["candidate"], confidence, runner_structural, margin,
                "; ".join(best["reasons"]), decision,
            )
            continue

        # Several managers are structurally indistinguishable.
        if not ALLOW_SHARED_TEAM_ASSIGNMENT:
            mark_unresolved(
                emp, "ambiguous",
                f"Ambiguous: {len(top)} managers equally likely "
                f"(score margin < {MIN_SCORE_MARGIN}).",
                top[0]["candidate"], top[0]["total"],
            )
            continue
        key = frozenset(s["candidate"]["Record ID"] for s in top)
        shared_groups[key].append((emp, top, runner_structural))

    # ------------------------------------------------------------------
    # Shared-team groups: spread employees with a span-of-control cap
    # ------------------------------------------------------------------
    for key, items in shared_groups.items():
        managers = sorted(key)
        k = len(managers)
        capacity = math.ceil(len(items) / k) + SHARED_TEAM_CAPACITY_SLACK
        load = {mid: 0 for mid in managers}

        pairs = []
        for emp, top, _ in items:
            for s in top:
                pairs.append((
                    -s["soft"], emp["Record ID"], s["candidate"]["Record ID"], s, emp,
                ))
        pairs.sort(key=lambda p: p[:3])

        assigned: Dict[str, Dict] = {}
        for _, emp_id, mgr_id, s, emp in pairs:
            if emp_id in assigned or load[mgr_id] >= capacity:
                continue
            assigned[emp_id] = s
            load[mgr_id] += 1

        penalty = SHARED_TEAM_PENALTY * (1 - 1 / k)
        for emp, top, runner_structural in items:
            s = assigned.get(emp["Record ID"])
            if s is None:  # cannot happen with capacity >= n/k, kept as a guard
                mark_unresolved(
                    emp, "ambiguous", "Shared-team assignment failed.",
                    top[0]["candidate"], top[0]["total"],
                )
                continue
            confidence = max(0.0, s["total"] - penalty)
            if confidence < MIN_CONFIDENCE:
                mark_unresolved(
                    emp, "ambiguous",
                    f"{k} managers equally likely; confidence {confidence:.1f} "
                    f"below threshold {MIN_CONFIDENCE}.",
                    s["candidate"], confidence,
                )
                continue
            others = [o["total"] for o in top if o is not s]
            runner = max(others) if others else 0.0
            add_edge(
                emp, s["candidate"], confidence, runner,
                max(0.0, s["total"] - runner),
                "; ".join(s["reasons"])
                + f"; {k} managers share this team, assigned by balanced span of control",
                "shared_team_balanced",
            )

    # ------------------------------------------------------------------
    # Safety net: nobody below level 8 may manage, no upward/sideways edges
    # ------------------------------------------------------------------
    safe_edges: List[Dict] = []
    for edge in edges:
        mgr = by_id[edge["Manager Record ID"]]
        emp = by_id[edge["Employee Record ID"]]
        if _lvl(mgr) < MANAGER_MIN_LEVEL or _lvl(mgr) <= _lvl(emp):
            mark_unresolved(
                emp, "unresolved",
                f"Rejected: level {_lvl(mgr)} cannot manage level {_lvl(emp)}.",
            )
        else:
            safe_edges.append(edge)

    return safe_edges, unresolved


def infer_company(company_df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    """
    Infer reporting lines for one company (a dataframe from load_data).
    If the frame mixes several Company Keys (same name, different industry),
    each is processed separately so org charts never cross.
    """
    if company_df.empty:
        return [], []

    records = company_df.to_dict(orient="records")
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for row in records:
        groups[row.get("Company Key", row.get("Company Name", ""))].append(row)

    all_edges: List[Dict] = []
    all_unresolved: List[Dict] = []
    for group in groups.values():
        edges, unresolved = _infer_single_company(group)
        all_edges.extend(edges)
        all_unresolved.extend(unresolved)
    return all_edges, all_unresolved


def infer_all(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: run every company and return two DataFrames."""
    edges: List[Dict] = []
    unresolved: List[Dict] = []
    for _, company_df in df.groupby("Company Key", sort=False):
        e, u = infer_company(company_df)
        edges.extend(e)
        unresolved.extend(u)
    return pd.DataFrame(edges), pd.DataFrame(unresolved)
