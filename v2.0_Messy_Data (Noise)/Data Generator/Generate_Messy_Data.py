"""
Generate_Messy_Data.py
======================
Creates two files:

  1. Synthetic_Company_Employees.csv
       "Ground truth" file. Same columns as the original generator. Reporting
       relationships are strictly valid (every non-CEO has a manager in the same
       company, seniority always decreases down the chain, teams line up), but
       the org framework now contains real-world oddities: wacky job titles
       ("Technoking", "Glubgogabgalab"), alternate executive titles (CTO, CFO,
       Head of X), variable email conventions, remote employees in other
       countries, non-sequential employee IDs, uneven team sizes, etc.

  2. Employees_Database.csv
       Same 11 columns as the original, but MESSY like a real HR database that
       was merged from several systems: inconsistent / abbreviated / misspelled
       titles, missing fields, placeholder values (N/A, TBD, -), mixed name
       formats, mixed country spellings, personal emails, extra whitespace,
       encoding glitches, duplicate records, test records, shuffled rows.
"""

import logging
import sys
from datetime import datetime
import random
import re
import math
import unicodedata
from collections import Counter

import pandas as pd
from faker import Faker

# ==========================================
# 0. LOGGING (same behaviour as original)
# ==========================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"log_{timestamp}.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)


class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        message = message.strip()
        if message:
            self.level(message)

    def flush(self):
        pass


sys.stdout = LoggerWriter(logging.info)
sys.stderr = LoggerWriter(logging.error)

print(f"Logging started: {log_filename}")

# ==========================================
# 1. CONFIGURATION
# ==========================================


def get_input(prompt, default, minimum=None, maximum=None):
    while True:
        try:
            value = int(input(f"{prompt} [Default: {default}]: ") or default)
            if minimum is not None and value < minimum:
                print(f"Value must be >= {minimum}.")
                continue
            if maximum is not None and value > maximum:
                print(f"Value must be <= {maximum}.")
                continue
            return value
        except ValueError:
            print("Please enter a valid integer.")


N = get_input("Total Companies (>= 500)", 500, 500)
MIN_EMPLOYEES = get_input("Minimum Employees (>= 100)", 100, 100)
MAX_EMPLOYEES = get_input("Maximum Employees", 300, MIN_EMPLOYEES)
SEED = get_input("Random Seed", 42)
MESS_LEVEL = get_input("Messiness level (1=light, 2=realistic, 3=chaotic)", 2, 1, 3)

# Tunables
OFFSHORE_RATE = 0.12      # share of non-leadership staff working outside HQ country
DUP_RATE = 0.015          # share of extra duplicate rows in the messy file
TEST_RATE = 0.002         # share of junk/test rows in the messy file
MESS_MULT = {1: 0.5, 2: 1.0, 3: 1.8}[MESS_LEVEL]

rng = random.Random(SEED)          # all "clean" structure
mrng = random.Random(SEED + 1000)  # all "mess" (so mess never changes the clean data)

company_fake = Faker("en_US")
company_fake.seed_instance(SEED)

# ==========================================
# 2. COUNTRY / FAKER CONFIGURATION
# ==========================================
COUNTRY_CFG = {  # country -> (faker locale, typical TLD)
    "USA": ("en_US", ".com"),
    "India": ("en_IN", ".co.in"),
    "UK": ("en_GB", ".co.uk"),
    "Australia": ("en_AU", ".com.au"),
    "Germany": ("de_DE", ".de"),
    "Canada": ("en_CA", ".ca"),
}
COUNTRIES = list(COUNTRY_CFG.keys())

fake_generators = {c: Faker(loc) for c, (loc, _) in COUNTRY_CFG.items()}
for index, fk in enumerate(fake_generators.values()):
    fk.seed_instance(SEED + index)

# ==========================================
# 3. ORGANIZATION STRUCTURE (same as original)
# ==========================================
organization = {
    "Engineering": {
        "head": "VP Engineering",
        "managers": ["Engineering Manager", "Software Development Manager", "QA Manager", "Infrastructure Manager"],
        "employees": ["Technical Lead", "Senior Software Engineer", "Software Engineer", "Junior Software Engineer", "Associate Engineer"]},
    "Sales": {
        "head": "VP Sales",
        "managers": ["Sales Operations Manager", "Senior Sales Manager", "Sales Manager", "Regional Sales Manager"],
        "employees": ["Account Executive", "Sales Representative", "Business Development Representative", "Client Success Associate", "Sales Operations Associate"]},
    "Human Resources": {
        "head": "HR Director",
        "managers": ["HR Manager", "Talent Acquisition Manager", "HR Operations Manager", "Compensation and Benefits Manager"],
        "employees": ["HR Business Partner", "HR Executive", "Recruiter"]},
    "Finance": {
        "head": "Finance Director",
        "managers": ["Finance Manager", "Accounting Manager", "Tax Manager", "Financial Planning Manager"],
        "employees": ["Senior Financial Analyst", "Financial Analyst", "Accountant"]},
    "Marketing": {
        "head": "Marketing Director",
        "managers": ["Marketing Manager", "Digital Marketing Manager", "Brand Manager", "Content Marketing Manager", "PR Manager"],
        "employees": ["Marketing Analyst", "Marketing Executive", "Content Specialist", "Social Media Intern"]},
    "Operations": {
        "head": "Operations Director",
        "managers": ["Operations Manager", "Regional Operations Manager", "Process Manager", "Supply Chain Manager"],
        "employees": ["Operations Analyst", "Operations Executive", "Process Associate"]},
}
departments = list(organization.keys())
# Real companies are not evenly split: engineering & sales are usually biggest
DEPT_WEIGHTS = [0.34, 0.22, 0.06, 0.08, 0.12, 0.18]

# ==========================================
# 3B. INDUSTRY / TEAM / SKILL REFERENCE DATA (same as original)
# ==========================================
INDUSTRIES = [
    "Information Technology", "Financial Services", "Healthcare",
    "Manufacturing", "Retail & E-Commerce", "Telecommunications",
    "Energy & Utilities", "Education", "Logistics & Transportation",
    "Media & Entertainment", "Consulting", "Real Estate",
    "Automotive", "Pharmaceuticals", "Hospitality",
]

TEAMS = {
    "Engineering": {
        "Platform Engineering": ["Kubernetes", "Terraform", "CI/CD Pipelines", "Cloud Architecture"],
        "Backend Services": ["Java", "Python", "Microservices", "REST API Design"],
        "Frontend & Mobile": ["React", "TypeScript", "Android Development", "UI Performance"],
        "Quality Assurance": ["Test Automation", "Selenium", "Performance Testing", "API Testing"],
        "Data Engineering": ["SQL", "Apache Spark", "ETL Pipelines", "Data Modeling"],
        "Security Engineering": ["Application Security", "Threat Modeling", "IAM", "Penetration Testing"],
    },
    "Sales": {
        "Enterprise Sales": ["Solution Selling", "Contract Negotiation", "Key Account Management", "RFP Management"],
        "SMB Sales": ["Lead Conversion", "Cold Calling", "Pipeline Management", "Upselling"],
        "Inside Sales": ["Outbound Prospecting", "CRM Management", "Demo Delivery", "Lead Qualification"],
        "Channel & Partnerships": ["Partner Enablement", "Alliance Management", "Reseller Programs", "Co-selling"],
        "Sales Operations": ["Salesforce Administration", "Sales Forecasting", "Territory Planning", "Sales Analytics"],
        "Customer Success": ["Onboarding", "Renewal Management", "Churn Reduction", "Customer Advocacy"],
    },
    "Human Resources": {
        "Talent Acquisition": ["Technical Sourcing", "Interviewing", "Employer Branding", "Applicant Tracking Systems"],
        "HR Business Partnering": ["Employee Relations", "Workforce Planning", "Performance Management", "Change Management"],
        "Compensation & Benefits": ["Payroll Strategy", "Benefits Administration", "Salary Benchmarking", "Equity Plans"],
        "Learning & Development": ["Training Design", "Leadership Development", "Skills Assessment", "E-Learning"],
        "HR Operations": ["HRIS Management", "Onboarding Processes", "Compliance Reporting", "Employee Records"],
        "Culture & Engagement": ["Engagement Surveys", "Diversity & Inclusion", "Internal Communications", "Recognition Programs"],
    },
    "Finance": {
        "Accounting": ["General Ledger", "Month-End Close", "Account Reconciliation", "IFRS/GAAP"],
        "Financial Planning & Analysis": ["Budgeting", "Forecasting", "Variance Analysis", "Financial Modeling"],
        "Tax & Compliance": ["Corporate Tax", "Transfer Pricing", "Statutory Compliance", "Tax Audits"],
        "Treasury": ["Cash Management", "Liquidity Planning", "FX Risk Management", "Banking Relations"],
        "Accounts Payable & Receivable": ["Invoice Processing", "Collections", "Vendor Payments", "Credit Control"],
        "Internal Audit & Risk": ["Risk Assessment", "SOX Compliance", "Internal Controls", "Fraud Detection"],
    },
    "Marketing": {
        "Digital Marketing": ["SEO", "Google Ads", "Email Marketing", "Marketing Automation"],
        "Brand & Creative": ["Brand Strategy", "Visual Design", "Copywriting", "Creative Direction"],
        "Content Marketing": ["Content Strategy", "Blogging", "Video Production", "Editorial Planning"],
        "Product Marketing": ["Go-To-Market Strategy", "Competitive Analysis", "Positioning", "Sales Enablement"],
        "Public Relations": ["Media Relations", "Crisis Communication", "Press Releases", "Influencer Outreach"],
        "Marketing Analytics": ["Google Analytics", "Attribution Modeling", "A/B Testing", "Campaign Reporting"],
    },
    "Operations": {
        "Supply Chain": ["Demand Planning", "Inventory Management", "Procurement", "Vendor Management"],
        "Process Excellence": ["Lean Six Sigma", "Process Mapping", "Continuous Improvement", "Root Cause Analysis"],
        "Logistics & Fulfilment": ["Warehouse Management", "Route Optimization", "Shipping Coordination", "Last-Mile Delivery"],
        "Facilities & Administration": ["Facilities Management", "Vendor Contracts", "Health & Safety", "Asset Management"],
        "Business Operations": ["Operational Reporting", "KPI Tracking", "Resource Planning", "SLA Management"],
        "Quality & Compliance": ["ISO Standards", "Quality Audits", "Regulatory Compliance", "Incident Management"],
    },
}

EXECUTIVE_TEAM = "Executive Office"
LEADERSHIP_TEAM = "Department Leadership"

CEO_SKILLS = ["Corporate Strategy", "Executive Leadership", "Investor Relations",
              "M&A", "Organizational Design", "Board Governance", "P&L Ownership"]
HEAD_SKILLS = ["Strategic Planning", "Budget Ownership", "Stakeholder Management",
               "Cross-Functional Leadership", "Talent Strategy", "Executive Reporting"]
MANAGER_SKILLS = ["People Management", "Coaching & Mentoring", "Resource Allocation",
                  "Performance Reviews", "Project Delivery", "Team Planning"]

# Tiers: the TRUE position in the hierarchy, independent of what the title says
CEO, HEAD, MGR, IC = "CEO", "HEAD", "MGR", "IC"

HEAD_DESIGNATIONS = {c["head"] for c in organization.values()}
MANAGER_DESIGNATIONS = {m for c in organization.values() for m in c["managers"]}


def get_seniority(designation):
    """Seniority of a STANDARD designation (same rules as the original)."""
    if designation == "CEO":
        return 10
    if designation in HEAD_DESIGNATIONS:
        return 9
    if designation in MANAGER_DESIGNATIONS:
        return 8
    lowered = designation.lower()
    if "lead" in lowered:
        return 6
    if "senior" in lowered:
        return 5
    if any(w in lowered for w in ("junior", "associate", "intern")):
        return 1
    return 3


# ==========================================
# 3C. REAL-WORLD TITLE ODDITIES
# ==========================================
# Wacky vanity titles. Each has a TRUE hierarchy level so relationships stay valid.
WACKY_CEO = ["Technoking", "Chief Everything Officer", "Head Honcho", "Chief Visionary Officer",
             "Supreme Leader", "Chief Dreamer", "Big Cheese", "Chief Galactic Officer"]
# Legit but non-standard executive titles
ALT_CEO = ["Founder & CEO", "President & CEO", "Managing Director", "Founder and Chief Executive",
           "Chief Executive Officer", "Co-Founder & CEO"]

ALT_HEAD = {
    "Engineering": ["CTO", "Head of Engineering", "Chief Technology Officer", "SVP Engineering"],
    "Sales": ["Chief Revenue Officer", "Head of Sales", "SVP Sales", "Chief Sales Officer"],
    "Human Resources": ["Chief People Officer", "Head of People", "VP People Operations", "CHRO"],
    "Finance": ["CFO", "Chief Financial Officer", "Head of Finance", "VP Finance"],
    "Marketing": ["CMO", "Head of Growth", "Chief Marketing Officer", "VP Marketing"],
    "Operations": ["COO", "Head of Operations", "Chief Operating Officer", "VP Operations"],
}
WACKY_HEAD = {
    "Engineering": ["Chief Technology Wizard", "Grand Wizard of Code", "Lord of the Servers"],
    "Sales": ["Revenue Overlord", "Chief Deal Whisperer", "Head of Making It Rain"],
    "Human Resources": ["Chief Happiness Officer", "Minister of Vibes", "Chief Culture Shaman"],
    "Finance": ["Master of Coin", "Chief Money Wrangler", "Keeper of the Vault"],
    "Marketing": ["Chief Hype Officer", "Brand Overlord", "Head of Buzz"],
    "Operations": ["Chief of Getting Things Done", "Grand Master of Logistics", "Director of Everything Else"],
}
WACKY_MANAGER = ["Glubgogabgalab", "Herder of Cats", "Team Whisperer", "Minister of Synergy",
                 "Dungeon Master", "Chief Fire Fighter", "Head of Stuff", "Captain Chaos",
                 "Sprint Sorcerer", "Overlord of Tasks"]
# (title, true seniority level) ; None -> random junior/mid/senior
WACKY_IC_DEPT = {
    "Engineering": [("Code Ninja", 3), ("Bug Whisperer", 3), ("Full-Stack Unicorn", 5),
                    ("Rockstar Developer", 3), ("Keyboard Warrior", 1), ("Software Jedi", 5)],
    "Sales": [("Sales Rockstar", 3), ("Deal Slayer", 5), ("Revenue Ninja", 3), ("Lead Whisperer", 3), ("Handshake Guru", 1)],
    "Human Resources": [("People Ninja", 3), ("Culture Guru", 3), ("Happiness Hero", 1), ("Talent Scout", 3)],
    "Finance": [("Number Cruncher", 3), ("Spreadsheet Sorcerer", 3), ("Penny Pincher", 1), ("Bean Counter", 3)],
    "Marketing": [("Growth Hacker", 3), ("Brand Evangelist", 3), ("Content Ninja", 3), ("Social Butterfly", 1), ("Hype Specialist", 3)],
    "Operations": [("Process Wizard", 3), ("Logistics Ninja", 3), ("Fixer", 3), ("Gopher", 1)],
}
WACKY_IC_GLOBAL = [("Glubgogabgalab", None), ("Snorkelmaster", None), ("Flibbertigibbet", None),
                   ("Wizard of Whatever", None), ("Ninja", None), ("Guru", None),
                   ("Associate Vibes Curator", 1), ("Unicorn", None), ("Chief of Misc.", None),
                   ("Professional Problem Solver", None)]

# Every title that is truly weird (these are protected from abbreviation in the messy file)
WACKY_SET = set(WACKY_CEO) | set(WACKY_MANAGER)
WACKY_SET |= {t for v in WACKY_HEAD.values() for t in v}
WACKY_SET |= {t for v in WACKY_IC_DEPT.values() for t, _ in v}
WACKY_SET |= {t for t, _ in WACKY_IC_GLOBAL}


def company_quirk_rate():
    """Some companies are very conventional, a few are 'startup-y' and weird."""
    r = rng.random()
    if r < 0.35:
        return 0.005
    if r < 0.80:
        return rng.uniform(0.02, 0.08)
    if r < 0.95:
        return rng.uniform(0.10, 0.20)
    return rng.uniform(0.30, 0.50)


def pick_title(tier, dept, standard, q):
    """Returns (designation, true_seniority_level, is_wacky)."""
    r = rng.random()
    if tier == CEO:
        p = min(0.9, q * 3)
        if r < p * 0.5:
            return rng.choice(WACKY_CEO), 10, True
        if r < p:
            return rng.choice(ALT_CEO), 10, False
        return "CEO", 10, False
    if tier == HEAD:
        if r < q * 1.5:
            return rng.choice(WACKY_HEAD[dept]), 9, True
        if r < q * 1.5 + 0.15 + q:
            return rng.choice(ALT_HEAD[dept]), 9, False
        return standard, 9, False
    if tier == MGR:
        if r < q:
            return rng.choice(WACKY_MANAGER), 8, True
        return standard, 8, False
    # IC
    if r < q * 1.2:
        pool = WACKY_IC_DEPT[dept] if rng.random() < 0.65 else WACKY_IC_GLOBAL
        title, lvl = rng.choice(pool)
        if lvl is None:
            lvl = rng.choice([1, 3, 5])
        return title, lvl, True
    return standard, get_seniority(standard), False


IC_LEVEL_WEIGHT = {6: 0.08, 5: 0.22, 3: 0.40, 1: 0.30}


def get_skills(tier, dept, team):
    if tier == CEO:
        return "; ".join(rng.sample(CEO_SKILLS, 3))
    if tier == HEAD:
        return "; ".join(rng.sample(HEAD_SKILLS, 3))
    team_skills = TEAMS[dept][team]
    if tier == MGR:
        return "; ".join(rng.sample(MANAGER_SKILLS, 1) + rng.sample(team_skills, 2))
    return "; ".join(rng.sample(team_skills, rng.randint(2, 3)))


# ==========================================
# 4. HELPERS (clean data)
# ==========================================
def to_ascii(s):
    s = s.lower()
    for a, b in (("ß", "ss"), ("ä", "ae"), ("ö", "oe"), ("ü", "ue")):
        s = s.replace(a, b)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def slug(s):
    return re.sub(r"[^a-z0-9]", "", to_ascii(s))


DOMAIN_STOP = {"inc", "llc", "ltd", "plc", "group", "and", "sons"}
EMAIL_PATTERNS = ["{f}.{l}", "{f}{l}", "{f}_{l}", "{fi}{l}", "{f}{li}", "{f}", "{l}.{f}"]
EMAIL_PATTERN_W = [0.45, 0.10, 0.10, 0.20, 0.05, 0.05, 0.05]
used_domains = set()


def make_domain(name, country, ci):
    words = [w for w in re.split(r"[^a-z0-9]+", to_ascii(name)) if w and w not in DOMAIN_STOP]
    base = "".join(words)[:24] or f"company{ci}"
    tld = COUNTRY_CFG[country][1] if rng.random() < 0.6 else ".com"
    for cand in (base + tld, base + "group" + tld, base + "hq" + tld, f"{base}{ci}{tld}"):
        if cand not in used_domains:
            used_domains.add(cand)
            return cand
    cand = f"{base}-{ci}.com"
    used_domains.add(cand)
    return cand


# ==========================================
# 5. GENERATE UNIQUE COMPANY NAMES
# ==========================================
company_names = set()
while len(company_names) < N:
    company_names.add(company_fake.company())
company_names = sorted(company_names)

# ==========================================
# 6. GENERATE CLEAN COMPANIES / EMPLOYEES
# ==========================================
records = []

for ci, company in enumerate(company_names, 1):
    print(f"Generating Company {ci}/{N}: {company}")

    n_emp = rng.randint(MIN_EMPLOYEES, MAX_EMPLOYEES)
    hq = rng.choice(COUNTRIES)
    industry = rng.choice(INDUSTRIES)
    q = company_quirk_rate()
    pattern = rng.choices(EMAIL_PATTERNS, EMAIL_PATTERN_W)[0]
    domain = make_domain(company, hq, ci)

    city_pools = {}

    def city_for(country, is_leader=False):
        if country not in city_pools:
            fk = fake_generators[country]
            k = 10 if country == hq else 5
            city_pools[country] = sorted({fk.city() for _ in range(k)})
        pool = city_pools[country]
        if country == hq and rng.random() < (0.8 if is_leader else 0.35):
            return pool[0]  # HQ city
        return rng.choice(pool)

    nodes = []

    def add_node(mgr, tier, dept, std_title, team):
        title, level, wacky = pick_title(tier, dept if dept in organization else "Engineering", std_title, q)
        leader = tier in (CEO, HEAD)
        country = hq if (leader or rng.random() >= OFFSHORE_RATE) else rng.choice([c for c in COUNTRIES if c != hq])
        fk = fake_generators[country]
        nodes.append({
            "mgr": mgr, "tier": tier, "dept": dept, "team": team,
            "title": title, "level": level, "wacky": wacky,
            "country": country, "city": city_for(country, leader),
            "first": fk.first_name(), "last": fk.last_name(),
            "skills": get_skills(tier, dept, team),
        })
        return len(nodes) - 1

    ceo_idx = add_node(None, CEO, "Executive", "CEO", EXECUTIVE_TEAM)

    # Department sizes: min 8 each, rest spread by realistic weights
    extra = (n_emp - 1) - 8 * len(departments)
    sizes = [8] * len(departments)
    for d_i in rng.choices(range(len(departments)), DEPT_WEIGHTS, k=extra):
        sizes[d_i] += 1

    for dept, size in zip(departments, sizes):
        cfg = organization[dept]
        head_idx = add_node(ceo_idx, HEAD, dept, cfg["head"], LEADERSHIP_TEAM)

        manager_count = max(1, math.ceil((size - 1) / 10))
        team_order = rng.sample(list(TEAMS[dept].keys()), len(TEAMS[dept]))

        mgr_idxs, mgr_teams = [], []
        for m in range(manager_count):
            team = team_order[m % len(team_order)]
            idx = add_node(head_idx, MGR, dept, rng.choice(cfg["managers"]), team)
            mgr_idxs.append(idx)
            mgr_teams.append(team)

        ic_count = size - 1 - manager_count
        # Uneven team sizes: every manager gets >=1 report, the rest are random-weighted
        team_w = [rng.uniform(0.4, 2.0) for _ in range(manager_count)]
        assign = list(range(manager_count)) + rng.choices(range(manager_count), team_w, k=ic_count - manager_count)
        rng.shuffle(assign)

        std_w = [IC_LEVEL_WEIGHT[get_seniority(t)] for t in cfg["employees"]]
        for m_i in assign:
            std = rng.choices(cfg["employees"], std_w)[0]
            add_node(mgr_idxs[m_i], IC, dept, std, mgr_teams[m_i])

    # ---- Final IDs: CEO/founder is #1, everyone else gets non-sequential numbers (gaps = leavers)
    n = len(nodes)
    pool = rng.sample(range(2, int(n * 1.4) + 3), n - 1)
    numbers = [1] + pool
    ids = [f"C{ci:04d}-E{num:05d}" for num in numbers]

    used_local = set()
    for idx, nd in enumerate(nodes):
        f, l = slug(nd["first"]) or "user", slug(nd["last"]) or "user"
        local = pattern.format(f=f, l=l, fi=f[0], li=l[0])
        base, k = local, 1
        while local in used_local:
            k += 1
            local = f"{base}{k}"
        used_local.add(local)

        records.append({
            "Employee ID": ids[idx],
            "Manager ID": ids[nd["mgr"]] if nd["mgr"] is not None else None,
            "Company Name": company,
            "Industry": industry,
            "Designation": nd["title"],
            "Seniority Level": str(nd["level"]),
            "Department": nd["dept"],
            "Team/Business Unit": nd["team"],
            "Employee Name": f"{nd['first']} {nd['last']}",
            "Skills/Specialization": nd["skills"],
            "City": nd["city"],
            "Country": nd["country"],
            "Email": f"{local}@{domain}",
            # hidden helper columns (dropped on export)
            "_tier": nd["tier"], "_first": nd["first"], "_last": nd["last"], "_wacky": nd["wacky"],
        })

# ==========================================
# 7. CREATE CLEAN DATAFRAME
# ==========================================
PUBLIC_COLS = ["Employee ID", "Manager ID", "Company Name", "Industry", "Designation",
               "Seniority Level", "Department", "Team/Business Unit", "Employee Name",
               "Skills/Specialization", "City", "Country", "Email"]

full_df = pd.DataFrame(records).sort_values("Employee ID").reset_index(drop=True)
df = full_df[PUBLIC_COLS].copy()

# ==========================================
# 8. VALIDATE CLEAN DATA (relationships must be correct)
# ==========================================
print("\nRunning validation checks...")

company_counts = df.groupby("Company Name").size()
tier = full_df.set_index("Employee ID")["_tier"]
company_of = df.set_index("Employee ID")["Company Name"]
team_of = df.set_index("Employee ID")["Team/Business Unit"]
dept_of = df.set_index("Employee ID")["Department"]
level_of = df.set_index("Employee ID")["Seniority Level"].astype(int)
has_mgr = full_df["Manager ID"].notna()

assert df["Employee ID"].is_unique, "Duplicate Employee IDs found!"
assert df["Email"].str.lower().is_unique, "Duplicate Email addresses found!"
assert company_counts.min() >= MIN_EMPLOYEES, "Some companies have fewer employees than required!"
assert company_counts.max() <= MAX_EMPLOYEES, "Some companies exceed the maximum employee count!"
assert set(df["Manager ID"].dropna()).issubset(set(df["Employee ID"])), "Invalid Manager IDs found!"

ceo_counts = full_df[full_df["_tier"] == CEO].groupby("Company Name").size()
assert len(ceo_counts) == N and (ceo_counts == 1).all(), "Each company must have exactly one top executive!"
assert full_df.loc[full_df["_tier"] == CEO, "Manager ID"].isna().all(), "CEO should not have a Manager ID!"
assert has_mgr.sum() == len(df) - N, "Every non-CEO must have a manager!"

mgr_series = full_df.loc[has_mgr, "Manager ID"]
emp_series = full_df.loc[has_mgr, "Employee ID"]
assert (mgr_series.values != emp_series.values).all(), "Self-reporting relationship found!"
assert (mgr_series.map(company_of).values == emp_series.map(company_of).values).all(), "Cross-company manager relationship!"

# Chain of command: IC -> MGR -> HEAD -> CEO
expected_mgr_tier = {IC: MGR, MGR: HEAD, HEAD: CEO}
for t, mt in expected_mgr_tier.items():
    sub = full_df[full_df["_tier"] == t]
    assert (sub["Manager ID"].map(tier) == mt).all(), f"{t} must report to {mt}!"

# Seniority strictly decreases down the chain
assert (emp_series.map(level_of).values < mgr_series.map(level_of).values).all(), "Employee outranks manager!"
# Seniority bands per tier
bands = {CEO: (10, 10), HEAD: (9, 9), MGR: (8, 8), IC: (1, 6)}
for t, (lo, hi) in bands.items():
    lv = full_df.loc[full_df["_tier"] == t, "Seniority Level"].astype(int)
    assert lv.between(lo, hi).all(), f"Seniority out of range for {t}!"

# Same department for everyone below the CEO's direct reports; same team for ICs
nonhead = full_df[full_df["_tier"].isin([MGR, IC])]
assert (nonhead["Manager ID"].map(dept_of) == nonhead["Department"]).all(), "Department mismatch with manager!"
ics = full_df[full_df["_tier"] == IC]
assert (ics["Manager ID"].map(team_of) == ics["Team/Business Unit"]).all(), "Employee team does not match manager team!"

for col in ["Industry", "Seniority Level", "Team/Business Unit", "Skills/Specialization", "Designation"]:
    assert df[col].notna().all() and (df[col].astype(str).str.strip() != "").all(), f"Missing values in '{col}'!"
assert (df.groupby("Company Name")["Industry"].nunique() == 1).all(), "A company has more than one industry!"

print("All validation checks passed!")

output_file1 = "Synthetic_Company_Employees.csv"
df.to_csv(output_file1, index=False)

# ==========================================
# 9. THE MESSY ENGINE
# ==========================================
# Messy data is produced by "source systems" with different habits, so the
# mess is CORRELATED (like a real merged HR database) rather than uniformly random.


def chance(p):
    return mrng.random() < min(0.95, p * MESS_MULT)


PROFILES = {
    # Fairly tidy modern HRIS
    "HRIS": dict(
        weight=0.40, abbr=0.10, case=0.02, case_fns=[str.upper, str.lower], typo=0.003, ws=0.015,
        miss=0.6, sparse=0.01, null=[""], variant_country=0.10, personal_email=0.01, email_case=0.01,
        sen_plain=0.85, strip_acc=0.10, mojibake=0.0,
        name_w={"first_last": 0.80, "middle": 0.08, "last_first": 0.04, "initial_last": 0.02,
                "title": 0.02, "nick": 0.03, "upper": 0.005, "lower": 0.005},
    ),
    # Old mainframe export: ALL CAPS, abbreviations, "LAST, FIRST", encoding junk
    "LEGACY": dict(
        weight=0.30, abbr=0.60, case=0.55, case_fns=[str.upper, str.upper, str.upper, str.lower], typo=0.004,
        ws=0.03, miss=1.0, sparse=0.03, null=["", "N/A", "NULL", "N/A"], variant_country=0.60,
        personal_email=0.005, email_case=0.50, sen_plain=0.50, strip_acc=0.50, mojibake=0.35,
        name_w={"last_first_upper": 0.45, "last_first": 0.20, "upper": 0.15, "first_last": 0.10,
                "initial_last": 0.05, "first_only": 0.02, "title": 0.03},
    ),
    # Hand-maintained spreadsheet: typos, stray spaces, placeholders, personal emails
    "SHEET": dict(
        weight=0.30, abbr=0.35, case=0.18, case_fns=[str.lower, str.title, str.upper], typo=0.02, ws=0.08,
        miss=1.6, sparse=0.06, null=["", "", "n/a", "-", "?", "TBD", "unknown", "none"], variant_country=0.45,
        personal_email=0.05, email_case=0.05, sen_plain=0.60, strip_acc=0.25, mojibake=0.05,
        name_w={"first_last": 0.50, "lower": 0.10, "upper": 0.05, "nick": 0.10, "initial_last": 0.05,
                "first_initial": 0.05, "last_first": 0.08, "title": 0.03, "middle": 0.04},
    ),
}
PROFILE_NAMES = list(PROFILES.keys())
PROFILE_W = [PROFILES[p]["weight"] for p in PROFILE_NAMES]

# Base probability that each field is blank (scaled by the profile's `miss`)
FIELD_MISS = {
    "Company Name": 0.002, "Industry": 0.04, "Designation": 0.02, "Seniority Level": 0.20,
    "Department": 0.03, "Team/Business Unit": 0.12, "Employee Name": 0.003,
    "Skills/Specialization": 0.15, "City": 0.04, "Country": 0.03, "Email": 0.05,
}

# ---------- generic text noise ----------


def typo(s):
    idx = [i for i, c in enumerate(s) if c.isalpha()]
    if len(idx) < 4:
        return s
    i = mrng.choice(idx[1:-1])
    op = mrng.choice(["swap", "drop", "dup", "sub"])
    if op == "swap" and i + 1 < len(s):
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if op == "drop":
        return s[:i] + s[i + 1:]
    if op == "dup":
        return s[:i] + s[i] + s[i:]
    return s[:i] + mrng.choice("abcdefghijklmnopqrstuvwxyz") + s[i + 1:]


def ws_noise(s):
    r = mrng.random()
    if r < 0.40:
        return " " + s
    if r < 0.80:
        return s + " "
    if r < 0.90:
        return s.replace(" ", "  ", 1) if " " in s else s + " "
    return s + "\u00a0"  # non-breaking space (classic copy/paste artifact)


def noise_text(s, prof, protect_case=False):
    if not s:
        return s
    if chance(prof["typo"]):
        s = typo(s)
    if not protect_case and chance(prof["case"]):
        s = mrng.choice(prof["case_fns"])(s)
    if chance(prof["ws"]):
        s = ws_noise(s)
    return s


def strip_accents(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def mojibake(s):
    try:
        return s.encode("utf-8").decode("latin-1")
    except Exception:
        return s


# ---------- abbreviation tables ----------
ABBR = {
    "Senior": ["Sr.", "Sr", "Snr"], "Junior": ["Jr.", "Jr"], "Manager": ["Mgr", "Mgr.", "Mngr"],
    "Engineering": ["Eng.", "Eng", "Engg"], "Engineer": ["Engr", "Eng."], "Software": ["SW", "S/W", "Soft."],
    "Development": ["Dev", "Dev."], "Operations": ["Ops", "Ops."], "Business": ["Biz", "Bus."],
    "Representative": ["Rep", "Rep."], "Associate": ["Assoc.", "Assoc", "Asst."], "Executive": ["Exec", "Exec."],
    "Technical": ["Tech.", "Tech"], "Marketing": ["Mktg", "Mkt.", "Mktng"], "Financial": ["Fin.", "Fin"],
    "Human Resources": ["HR", "H.R."], "Administration": ["Admin", "Admin."], "Management": ["Mgmt", "Mgt."],
    "Infrastructure": ["Infra"], "Quality Assurance": ["QA", "Q.A."], "Acquisition": ["Acq."],
    "Compensation and Benefits": ["Comp & Ben", "C&B", "Comp and Benefits"], "Regional": ["Reg.", "Regl"],
    "Specialist": ["Spec.", "Splst"], "Customer": ["Cust.", "Cust"], "Director": ["Dir.", "Dir"],
    "Supply Chain": ["SCM", "Supply-Chain"], "Information": ["Info"], "Communications": ["Comms"],
    "Public Relations": ["PR", "P.R."], "Financial Planning": ["FP&A", "FP & A"], "Analytics": ["Analytcs"],
    "Partnerships": ["Partners"], "Compliance": ["Compl."],
}
_abbr_re = re.compile(r"\b(" + "|".join(sorted(map(re.escape, ABBR), key=len, reverse=True)) + r")\b")


def abbreviate_words(s):
    return _abbr_re.sub(lambda m: mrng.choice(ABBR[m.group(1)]) if mrng.random() < 0.7 else m.group(1), s)


# ---------- designation ----------
WHOLE_SYN = {
    "CEO": ["Chief Executive Officer", "C.E.O.", "Chief Executive", "CEO & Founder", "Ceo",
            "Chief Exec. Officer", "CEO/Founder", "President & CEO"],
    "Technical Lead": ["Tech Lead", "Technical Team Lead", "Team Lead", "Lead Engineer", "Tech Lead - Software"],
    "Senior Software Engineer": ["Software Engineer III", "Sr. Software Developer", "Sr SWE", "Senior SDE",
                                 "Software Engineer (Senior)", "Senior Developer"],
    "Software Engineer": ["Software Developer", "SWE", "SDE", "Developer", "Software Engineer II", "Programmer"],
    "Junior Software Engineer": ["Jr. Software Engineer", "Software Engineer I", "Junior Developer", "Jr Dev",
                                 "Graduate Software Engineer"],
    "Associate Engineer": ["Associate Software Engineer", "Engineer I", "Graduate Engineer", "Trainee Engineer"],
    "Account Executive": ["AE", "Acct Exec", "Account Exec", "Sales Executive"],
    "Sales Representative": ["Sales Rep", "Sales Associate", "Salesperson", "Sales Exec"],
    "Business Development Representative": ["BDR", "Biz Dev Rep", "Sales Development Representative", "SDR"],
    "Client Success Associate": ["CS Associate", "Customer Success Associate", "CSA"],
    "Sales Operations Associate": ["Sales Ops Associate", "Sales Operations Coordinator", "SalesOps Associate"],
    "HR Business Partner": ["HRBP", "HR BP", "Human Resources Business Partner", "HR Partner"],
    "HR Executive": ["HR Exec", "Human Resources Executive", "HR Generalist", "HR Officer"],
    "Recruiter": ["Talent Acquisition Specialist", "Technical Recruiter", "Recruiting Specialist", "Talent Partner"],
    "Senior Financial Analyst": ["Sr. Financial Analyst", "Sr Fin Analyst", "Financial Analyst III", "Senior Finance Analyst"],
    "Financial Analyst": ["Finance Analyst", "Fin. Analyst", "Financial Analyst II", "Analyst, Finance"],
    "Accountant": ["Staff Accountant", "Accounts Executive", "Accounting Associate"],
    "Marketing Analyst": ["Mktg Analyst", "Marketing Data Analyst", "Marketing Insights Analyst"],
    "Marketing Executive": ["Marketing Exec", "Marketing Associate", "Mktg Exec"],
    "Content Specialist": ["Content Writer", "Content Creator", "Copywriter"],
    "Social Media Intern": ["SM Intern", "Social Intern", "Intern - Social Media"],
    "Operations Analyst": ["Ops Analyst", "Operational Analyst", "Business Operations Analyst"],
    "Operations Executive": ["Ops Exec", "Ops Executive", "Operations Exec."],
    "Process Associate": ["Process Assoc.", "Process Specialist", "Process Exec"],
}
VP_TEMPLATES = ["Vice President of {x}", "Vice President, {x}", "VP of {x}", "VP, {x}", "V.P. {x}",
                "VP - {x}", "Vice-President {x}", "Vice President {x}"]
DIR_TEMPLATES = ["Director of {x}", "Director, {x}", "Director - {x}", "Dir. of {x}", "{x} Dir.",
                 "Head of {x}", "Dir, {x}"]
MGR_TEMPLATES = ["Manager, {x}", "Manager - {x}", "{x} Mgr", "{x} Mgr.", "Mgr, {x}", "Manager of {x}",
                 "{x} Team Manager", "{x} Lead", "Head of {x}"]
NOTE_SUFFIXES = [" (Contract)", " - Contractor", " (Acting)", " (Remote)", " - Interim", " (On Leave)", " (Temp)"]


def mess_designation(d, prof):
    if not d:
        return d
    if d in WACKY_SET:
        # vanity titles are kept as-is (people don't abbreviate "Technoking"), just typed sloppily
        return noise_text(d, prof)

    a = prof["abbr"]
    if d in WHOLE_SYN and chance(a * 0.7):
        d = mrng.choice(WHOLE_SYN[d])
    elif d.startswith("VP ") and chance(0.6 + a):
        d = mrng.choice(VP_TEMPLATES).format(x=d[3:])
    elif d.endswith(" Director") and chance(0.5 + a):
        x = d[: -len(" Director")]
        if x == "HR" and chance(0.4):
            x = "Human Resources"
        d = mrng.choice(DIR_TEMPLATES).format(x=x)
    elif d.endswith(" Manager") and chance(a * 0.8):
        d = mrng.choice(MGR_TEMPLATES).format(x=d[: -len(" Manager")])

    if chance(a):
        d = abbreviate_words(d)
    if " and " in d and chance(0.15):
        d = d.replace(" and ", " & ")
    elif " & " in d and chance(0.15):
        d = d.replace(" & ", " and ")
    if chance(a * 0.10):
        d = d + mrng.choice(NOTE_SUFFIXES)
    return noise_text(d, prof)


# ---------- company ----------
SUFFIX_VARIANTS = {
    "Inc": [" Inc.", ", Inc.", " Incorporated", " INC", ",Inc", ""],
    "LLC": [" L.L.C.", " Llc", ", LLC", ""],
    "Ltd": [" Ltd.", " Limited", ", Ltd", ""],
    "PLC": [" plc", " Plc.", " P.L.C."],
    "Group": [" Grp", " Grp.", " Group Inc", ""],
}
_suffix_re = re.compile(r"\s(Inc|LLC|Ltd|PLC|Group)$")


def mess_company(name, prof):
    s = name
    m = _suffix_re.search(s)
    if m and chance(0.55):
        s = s[: m.start()] + mrng.choice(SUFFIX_VARIANTS[m.group(1)])
    if " and " in s and chance(0.30):
        s = s.replace(" and ", " & ")
    elif " & " in s and chance(0.2):
        s = s.replace(" & ", " and ")
    if "," in s and chance(0.15):
        s = s.replace(",", "")
    if chance(0.015):
        s = "The " + s
    return noise_text(s, prof)


# ---------- industry ----------
INDUSTRY_VARIANTS = {
    "Information Technology": ["IT", "Info Tech", "Information tech", "Tech", "Software/IT", "I.T."],
    "Financial Services": ["Finance", "Fin Services", "Financial Svcs", "Banking & Finance", "FinServ", "BFSI"],
    "Healthcare": ["Health Care", "Health", "Healthcare Services", "Medical"],
    "Manufacturing": ["Mfg", "Manufacturing & Industrial", "Industrial"],
    "Retail & E-Commerce": ["Retail", "Retail and E-commerce", "Ecommerce", "E-Commerce / Retail", "Retail/Ecom"],
    "Telecommunications": ["Telecom", "Telco", "Telecomms", "Telecommunication"],
    "Energy & Utilities": ["Energy", "Utilities", "Energy and Utilities", "Oil & Gas / Utilities"],
    "Education": ["Edu", "Education & Training", "EdTech", "Higher Ed"],
    "Logistics & Transportation": ["Logistics", "Transportation", "Transport & Logistics", "Supply Chain & Logistics"],
    "Media & Entertainment": ["Media", "Entertainment", "M&E", "Media/Entertainment"],
    "Consulting": ["Consultancy", "Management Consulting", "Professional Services", "Consulting Services"],
    "Real Estate": ["Realty", "Property", "Real-estate", "RE"],
    "Automotive": ["Auto", "Automobile", "Automotive Industry"],
    "Pharmaceuticals": ["Pharma", "Pharmaceutical", "Life Sciences", "Pharma & Biotech"],
    "Hospitality": ["Hotels & Hospitality", "Hotel/Hospitality", "Travel & Hospitality", "Hospitality & Tourism"],
}


def mess_industry(s, prof):
    if chance(prof["abbr"] + 0.15):
        s = mrng.choice(INDUSTRY_VARIANTS.get(s, [s]))
    return noise_text(s, prof)


# ---------- department / team ----------
DEPT_VARIANTS = {
    "Engineering": ["Eng", "Engg", "R&D", "Tech", "Product Engineering", "Dev", "IT / Engineering"],
    "Sales": ["Sales & BD", "Biz Dev", "Revenue", "Sales Dept", "Sales & Marketing"],
    "Human Resources": ["HR", "H.R.", "People Ops", "People", "Human Resource", "People & Culture"],
    "Finance": ["Fin", "Finance & Accounting", "Accounting", "F&A", "Finance Dept"],
    "Marketing": ["Mktg", "Marketing & Comms", "Growth", "Marketing Dept", "Brand & Marketing"],
    "Operations": ["Ops", "Operations & Admin", "Business Ops", "Ops Dept", "Operation"],
    "Executive": ["Exec", "Executive Office", "Management", "C-Suite", "Leadership", "Corporate"],
}


def mess_department(s, prof):
    if chance(prof["abbr"] + 0.08):
        s = mrng.choice(DEPT_VARIANTS.get(s, [s]))
    return noise_text(s, prof)


TEAM_SPECIAL = {
    EXECUTIVE_TEAM: ["Exec Office", "Executive", "Executive Team", "C-Suite", "Management Team", "Office of the CEO"],
    LEADERSHIP_TEAM: ["Dept Leadership", "Dept. Leadership", "Leadership", "Leadership Team", "Management", "Dept Head"],
}


def mess_team(s, prof):
    if s in TEAM_SPECIAL:
        if chance(prof["abbr"] + 0.2):
            s = mrng.choice(TEAM_SPECIAL[s])
        return noise_text(s, prof)
    if chance(prof["abbr"]):
        s = abbreviate_words(s)
    if " & " in s and chance(0.25):
        s = s.replace(" & ", " and ")
    return noise_text(s, prof)


# ---------- country ----------
COUNTRY_VARIANTS = {
    "USA": ["US", "U.S.A.", "United States", "United States of America", "U.S.", "America"],
    "India": ["IN", "IND", "Republic of India", "INDIA"],
    "UK": ["United Kingdom", "U.K.", "GB", "Great Britain", "England"],
    "Australia": ["AU", "AUS", "Aus", "Commonwealth of Australia"],
    "Germany": ["DE", "DEU", "Deutschland", "GER"],
    "Canada": ["CA", "CAN", "Can."],
}
COUNTRY_CODE = {"USA": "US", "India": "IN", "UK": "UK", "Australia": "AU", "Germany": "DE", "Canada": "CA"}


def mess_country(s, prof):
    if chance(prof["variant_country"]):
        s = mrng.choice(COUNTRY_VARIANTS.get(s, [s]))
    return noise_text(s, prof)


# ---------- city ----------
def mess_city(city, country, prof):
    if not city:
        return city
    if chance(prof["strip_acc"] * 0.5):
        city = strip_accents(city)
    if chance(0.04):
        city = f"{city}, {COUNTRY_CODE.get(country, '')}".rstrip(", ")
    return noise_text(city, prof)


# ---------- seniority ----------
SEN_WORDS = {
    1: ["Entry", "Junior", "Entry-level", "Associate"], 3: ["Mid", "Intermediate", "Mid-level"],
    5: ["Senior", "Sr"], 6: ["Lead", "Principal", "Team Lead"], 8: ["Manager", "Mgr"],
    9: ["Director", "Head", "VP"], 10: ["Executive", "C-Level", "C-Suite"],
}


def mess_seniority(level, prof):
    lv = int(level)
    if chance(1 - prof["sen_plain"]) or True:
        r = mrng.random()
        plain = prof["sen_plain"] / MESS_MULT if MESS_MULT > 1 else prof["sen_plain"]
        if r < plain:
            return str(lv)
        r = (r - plain) / max(1e-9, 1 - plain)
        if r < 0.25:
            return f"L{lv}"
        if r < 0.45:
            return f"Level {lv}"
        if r < 0.65:
            return f"{lv}.0"
        if r < 0.75:
            return f"{lv:02d}"
        return mrng.choice(SEN_WORDS[lv])
    return str(lv)


# ---------- names ----------
NICK = {"Michael": "Mike", "William": "Bill", "Robert": "Bob", "James": "Jim", "Elizabeth": "Liz",
        "Jennifer": "Jen", "Christopher": "Chris", "Matthew": "Matt", "Katherine": "Kate", "Richard": "Rick",
        "Joseph": "Joe", "Thomas": "Tom", "Daniel": "Dan", "Anthony": "Tony", "Patricia": "Pat",
        "Jessica": "Jess", "Stephanie": "Steph", "Alexander": "Alex", "Nicholas": "Nick", "Margaret": "Maggie",
        "David": "Dave", "Steven": "Steve", "Andrew": "Andy", "Samuel": "Sam", "Rebecca": "Becky",
        "Jonathan": "Jon", "Benjamin": "Ben", "Deborah": "Debbie", "Susan": "Sue", "Timothy": "Tim"}
TITLES = ["Dr.", "Mr.", "Ms.", "Mrs.", "Prof.", "Mr", "Dr"]
SUFFIXES = ["Jr.", "Sr.", "II", "III", "PhD", "MBA", "CPA", "Jr"]


def weighted(d):
    return mrng.choices(list(d.keys()), list(d.values()))[0]


def mess_name(first, last, prof):
    style = weighted(prof["name_w"])
    base = f"{first} {last}"
    if style == "last_first":
        s = f"{last}, {first}"
    elif style == "last_first_upper":
        s = f"{last.upper()}, {first}"
    elif style == "upper":
        s = base.upper()
    elif style == "lower":
        s = base.lower()
    elif style == "initial_last":
        s = f"{first[0]}. {last}"
    elif style == "first_initial":
        s = f"{first} {last[0]}."
    elif style == "middle":
        s = f"{first} {mrng.choice('ABCDEFGHJKLMNPRSTW')}. {last}"
    elif style == "first_only":
        s = first
    elif style == "nick":
        s = f"{NICK.get(first, first)} {last}"
    elif style == "title":
        s = f"{mrng.choice(TITLES)} {base}" if mrng.random() < 0.7 else f"{base} {mrng.choice(SUFFIXES)}"
    else:
        s = base
    if chance(prof["strip_acc"] * 0.5):
        s = strip_accents(s)
    return noise_text(s, prof, protect_case=style in ("upper", "lower", "last_first_upper"))


# ---------- skills ----------
SKILL_ALIAS = {"Kubernetes": ["K8s", "kubernetes"], "CI/CD Pipelines": ["CI/CD", "CICD"],
               "Microservices": ["Micro-services", "micro services"], "Python": ["python3", "Python 3"],
               "Test Automation": ["Automation Testing", "test automation"], "SQL": ["sql", "T-SQL"],
               "Apache Spark": ["Spark", "PySpark"], "Google Analytics": ["GA4", "GA"],
               "Search Engine Optimization": ["SEO"], "React": ["ReactJS", "React.js"],
               "Salesforce Administration": ["Salesforce Admin", "SFDC"], "Lean Six Sigma": ["Six Sigma", "LSS"]}
SKILL_FILLER = ["MS Office", "Communication", "Teamwork", "Excel", "Problem Solving", "Leadership", "Time Management"]


def mess_skills(skills, prof):
    items = [x.strip() for x in skills.split(";") if x.strip()]
    if len(items) > 1 and chance(0.15):
        items.pop(mrng.randrange(len(items)))
    if chance(0.30):
        mrng.shuffle(items)
    items = [mrng.choice(SKILL_ALIAS[i]) if i in SKILL_ALIAS and chance(0.25) else i for i in items]
    if chance(0.10):
        items.append(mrng.choice(SKILL_FILLER))
    if chance(prof["case"] * 0.8):
        items = [i.lower() for i in items]
    if chance(prof["typo"] * 4):
        k = mrng.randrange(len(items))
        items[k] = typo(items[k])
    sep = mrng.choices([";", "; ", ", ", ",", " / ", " | ", "\n"], [0.30, 0.15, 0.30, 0.08, 0.07, 0.07, 0.03])[0]
    s = sep.join(items)
    if sep == "\n":
        s = ", ".join(items)  # avoid raw newlines breaking naive CSV readers
    return noise_text(s, prof, protect_case=True)


# ---------- email ----------
PERSONAL_DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com", "rediffmail.com"]


def mess_email(email, first, last, prof):
    local, domain = email.split("@")
    f, l = slug(first) or "user", slug(last) or "user"
    if chance(prof["personal_email"]):
        local = mrng.choice([f"{f}{l}", f"{f}.{l}{mrng.randint(1, 99)}", f"{f[0]}{l}{mrng.randint(70, 99)}",
                             f"{f}_{l}{mrng.randint(1, 9)}"])
        domain = mrng.choice(PERSONAL_DOMAINS)
    s = f"{local}@{domain}"
    if chance(0.012):
        s = s.replace(".com", mrng.choice([".con", ".co", ",com", ".cm"]), 1) if ".com" in s else s.replace("@", "@ ", 1)
    if chance(0.004):
        s = s.replace("@", "", 1)
    if chance(prof["email_case"]):
        s = mrng.choice([str.upper, str.title])(s)
    if chance(prof["ws"]):
        s = ws_noise(s)
    return s


# ---------- row assembly ----------
def mess_row(row, prof):
    dept = row["Department"]
    if row["_tier"] in (MGR, IC) and chance(0.004):  # person tagged to the wrong department
        dept = mrng.choice(departments)

    out = {
        "Company Name": mess_company(row["Company Name"], prof),
        "Industry": mess_industry(row["Industry"], prof),
        "Designation": mess_designation(row["Designation"], prof),
        "Seniority Level": mess_seniority(row["Seniority Level"], prof),
        "Department": mess_department(dept, prof),
        "Team/Business Unit": mess_team(row["Team/Business Unit"], prof),
        "Employee Name": mess_name(row["_first"], row["_last"], prof),
        "Skills/Specialization": mess_skills(row["Skills/Specialization"], prof),
        "City": mess_city(row["City"], row["Country"], prof),
        "Country": mess_country(row["Country"], prof),
        "Email": mess_email(row["Email"], row["_first"], row["_last"], prof),
    }

    # Missing / placeholder values (one placeholder style per row, like a real source system)
    token = mrng.choice(prof["null"])
    sparse = chance(prof["sparse"])
    blank = set()
    if sparse:  # a half-filled record
        blank = set(mrng.sample([c for c in FIELD_MISS if c != "Company Name"], mrng.randint(3, 6)))
    for col, base in FIELD_MISS.items():
        if col in blank or chance(base * prof["miss"]):
            out[col] = token

    # Encoding damage (legacy exports): "Müller" -> "MÃ¼ller"
    if prof["mojibake"]:
        for col, val in out.items():
            if isinstance(val, str) and not val.isascii() and chance(prof["mojibake"]):
                out[col] = mojibake(val)
    return out


def choose_profile():
    return mrng.choices(PROFILE_NAMES, PROFILE_W)[0]


# ==========================================
# 10. BUILD THE MESSY DATASET
# ==========================================
print("\nBuilding messy Employees_Database...")
clean_rows = full_df.to_dict("records")
company_profile = {c: choose_profile() for c in company_names}

messy_rows = []
profile_usage = Counter()
for r in clean_rows:
    pname = company_profile[r["Company Name"]] if mrng.random() < 0.7 else choose_profile()
    profile_usage[pname] += 1
    messy_rows.append(mess_row(r, PROFILES[pname]))

# Duplicate people (same person entered twice, possibly via a different system)
n_dup = int(len(clean_rows) * min(0.2, DUP_RATE * MESS_MULT))
for i in mrng.sample(range(len(clean_rows)), n_dup):
    if mrng.random() < 0.30:
        messy_rows.append(dict(messy_rows[i]))  # exact duplicate
    else:
        messy_rows.append(mess_row(clean_rows[i], PROFILES[choose_profile()]))  # re-keyed differently

# Junk / test records that nobody ever deleted
TEST_NAMES = ["Test User", "TEST", "asdf asdf", "John Doe", "Jane Doe", "Admin", "Dummy Employee",
              "xxxxx", "Test Test", "Sample Employee", "New Hire", "TBD TBD"]
n_test = max(5, int(len(clean_rows) * TEST_RATE * MESS_MULT))
for _ in range(n_test):
    junk = {c: "" for c in FIELD_MISS}
    junk["Company Name"] = mess_company(mrng.choice(company_names), PROFILES[choose_profile()])
    junk["Employee Name"] = mrng.choice(TEST_NAMES)
    junk["Designation"] = mrng.choice(["Test", "TEST", "Tester", "", "N/A", "Contractor", "Temp", "Intern", "xxx"])
    junk["Email"] = mrng.choice(["test@test.com", "", "noreply@example.com", "asdf@asdf.com", "tbd@tbd"])
    junk["Department"] = mrng.choice(["", "Test", "N/A", "Unassigned", "TBD"])
    messy_rows.append(junk)

mrng.shuffle(messy_rows)

MESSY_COLS = ["Company Name", "Industry", "Designation", "Seniority Level", "Department",
              "Team/Business Unit", "Employee Name", "Skills/Specialization", "City", "Country", "Email"]
messy_df = pd.DataFrame(messy_rows)[MESSY_COLS]
assert len(messy_df) == len(clean_rows) + n_dup + n_test

output_file2 = "Employees_Database.csv"
messy_df.to_csv(output_file2, index=False)

# ==========================================
# 11. SUMMARY
# ==========================================
wacky_counts = full_df.loc[full_df["_wacky"], "Designation"].value_counts()
placeholder_re = r"^\s*(?:|n/?a|null|none|-|\?|tbd|unknown)\s*$"

print("\n" + "=" * 50)
print("DATASET GENERATED SUCCESSFULLY!")
print("=" * 50)
print(f"Messiness level             : {MESS_LEVEL}")
print(f"Total Companies             : {df['Company Name'].nunique():,}")
print(f"Total Employees (clean)     : {len(df):,}")
print(f"Rows in Employees_Database  : {len(messy_df):,} "
      f"({n_dup:,} duplicates + {n_test:,} test rows)")
print(f"Min / Max / Avg per company : {company_counts.min()} / {company_counts.max()} / {company_counts.mean():.1f}")
print(f"Unique designations (clean) : {df['Designation'].nunique():,}")
print(f"Unique designations (messy) : {messy_df['Designation'].nunique():,}")
print(f"Wacky titles used           : {len(wacky_counts)} distinct, {int(wacky_counts.sum()):,} people")
print("\nTop wacky titles:")
print(wacky_counts.head(12))
print("\nSource-system usage:")
print(pd.Series(profile_usage))
print("\nMissing / placeholder % per column (messy file):")
miss_pct = messy_df.apply(lambda s: s.fillna("").astype(str).str.match(placeholder_re, case=False).mean() * 100)
print(miss_pct.round(1).to_string())
print("\nSample clean records:")
print(df.head(8).to_string(index=False))
print("\nSample messy records:")
print(messy_df.head(12).to_string(index=False))
print(f"\nOutput File: {output_file1}")
print(f"Output File: {output_file2}")
print("=" * 50)
