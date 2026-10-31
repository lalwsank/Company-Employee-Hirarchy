# Company Employee Hierarchy — Data Generation and Org-Chart Inference

This project does two things:

1. **Generates** large synthetic employee tables (hundreds of companies, tens of thousands of people) in which the true "who reports to whom" structure is known.
2. **Infers** that structure back from a copy of the table that has the manager information removed — and, for the messy variant, a copy that has also been deliberately corrupted the way real HR exports are (typos, abbreviations, missing values, duplicates, mixed formats).

Three Streamlit apps let you run the inference, explore the resulting org charts, see a confidence score for every reporting line, and (optionally) score the result against the known truth.

> The top-level folder in the zip is spelled `Company Employee Hirarchy`. This README keeps the folder names exactly as they appear in the zip.

---

## Table of contents

1. [Project map](#1-project-map)
2. [How the pieces fit together](#2-how-the-pieces-fit-together)
3. [Setup and running](#3-setup-and-running)
4. [Data model](#4-data-model)
5. [The data generators, in depth](#5-the-data-generators-in-depth)
6. [The inference engines, in depth](#6-the-inference-engines-in-depth)
7. [The Streamlit apps, in depth](#7-the-streamlit-apps-in-depth)
8. [Output schemas](#8-output-schemas)
9. [Tunable parameters](#9-tunable-parameters)
10. [Known limitations and quirks](#10-known-limitations-and-quirks)
11. [Appendix — verification notes](#11-appendix--verification-notes)

---

## 1. Project map

```text
Company Employee Hirarchy/
├── v1.0_Synthetic_Data (No Noise)/
│   ├── Data Generator/
│   │   └── Generate_Synthetic_Data.py
│   ├── __pycache__/
│   ├── inference_engine.py
│   └── App.py
│
├── v2.0_Messy_Data (Noise)/
│   ├── Data Generator/
│   │   └── Generate_Messy_Data.py
│   ├── __pycache__/
│   ├── inference_engine.py
│   └── App.py
│
└── v2.1_Mess_Data (Final)/
    ├── Data Generator/
    │   └── Generate_Messy_Data.py
    ├── __pycache__/
    ├── inference_engine.py
    └── App.py
```

Each version folder has its own `Data Generator` and `__pycache__` subfolder. The `__pycache__` folders contain Python bytecode and can be deleted; Python recreates them when the code is run.

| File | One-line purpose |
|---|---|
| `Generate_Synthetic_Data.py` | Builds N companies with a strictly regular hierarchy; writes a ground-truth CSV and a shuffled, ID-less, clean CSV. |
| `Generate_Messy_Data.py` | Same idea, but the hierarchy has real-world oddities (vanity titles, remote staff, uneven teams) and the output CSV is corrupted like a merged HR database. |
| `v1.0_Synthetic_Data (No Noise)/inference_engine.py` | Infers reporting lines from *clean* labels using additive scoring (department, team, skills, title, location). |
| `v1.0_Synthetic_Data (No Noise)/App.py` | UI for the above, including a side-by-side comparison with ground truth. |
| `v2.0_Messy_Data (Noise)/inference_engine.py` | Stage 1 cleans/merges/normalises a messy table (learning from the table itself); Stage 2 infers lines with a fixed rule ladder. |
| `v2.0_Messy_Data (Noise)/App.py` | UI for the above (tabs: chart, people, lines, needs review, data cleaning, downloads). |
| `v2.1_Mess_Data (Final)/inference_engine.py` | The v4 engine plus `load_truth`, `compare_to_truth`, `score_rows`. |
| `v2.1_Mess_Data (Final)/App.py` | The messy UI plus a "Ground truth" tab with precision/recall breakdowns and ✓/✗/– markers on the trees. |

---

## 2. How the pieces fit together

```
                  Generate_Synthetic_Data.py
                  ├─► Synthetic_Company_Employees.csv   GROUND TRUTH  (13 cols: has Employee ID + Manager ID)
                  └─► Employees_Database.csv            clean, shuffled, 11 cols, NO IDs
                                    │
                                    ▼
                         Synthetic Data App  ◄── (optional) ground-truth CSV for scoring


                  Generate_Messy_Data.py
                  ├─► Synthetic_Company_Employees.csv   GROUND TRUTH  (13 cols, quirky titles, valid structure)
                  └─► Employees_Database.csv            MESSY, shuffled, 11 cols, NO IDs
                                    │
                       ┌────────────┴─────────────┐
                       ▼                          ▼
                 Messy Data App        Messy Data True Compare App ◄── ground-truth CSV
```

Key idea: the inference engines **never see** `Employee ID` or `Manager ID`. They only get the 11 descriptive columns and must work out reporting lines from them. The ground-truth file is used purely to *grade* the answer.

**Both generators write files with the same names** (`Synthetic_Company_Employees.csv`, `Employees_Database.csv`) into the current working directory. Run them from different folders, or rename the outputs, so one run does not overwrite the other.

---

## 3. Setup and running

### Dependencies

No `requirements.txt` ships with the project. The code imports:

| Package | Used by |
|---|---|
| `pandas` | everything |
| `Faker` | both generators (names, cities, company names) |
| `streamlit` | the three `App.py` files |
| standard library only | `logging`, `random`, `re`, `math`, `unicodedata`, `difflib`, `dataclasses`, `collections`, `functools`, `html`, `io`, `threading`, `pathlib`, `datetime` |

```bash
pip install pandas faker streamlit
```

### Generate data

Each generator prompts for its settings (press Enter to accept the default):

| Prompt | Default | Constraint |
|---|---|---|
| Total Companies | 500 | ≥ 500 |
| Minimum Employees (per company) | 100 | ≥ 100 |
| Maximum Employees (per company) | 300 | ≥ minimum |
| Random Seed | 42 | any integer |
| Messiness level *(messy generator only)* | 2 | 1 = light, 2 = realistic, 3 = chaotic |

Run the generator from the `Data Generator` folder inside the version you want to use:

```bash
# Clean dataset
cd "Company Employee Hirarchy/v1.0_Synthetic_Data (No Noise)/Data Generator"
python Generate_Synthetic_Data.py

# Messy dataset (v2.0)
cd "Company Employee Hirarchy/v2.0_Messy_Data (Noise)/Data Generator"
python Generate_Messy_Data.py

# Final messy dataset generator (v2.1)
cd "Company Employee Hirarchy/v2.1_Mess_Data (Final)/Data Generator"
python Generate_Messy_Data.py
```

Both scripts also write a timestamped `log_YYYYMMDD_HHMMSS.txt` and mirror everything printed to the console into it (see `LoggerWriter` in §5).

### Run an app

Run from **inside** the app's folder, because each `App.py` does `import inference_engine` by plain name:

```bash
# Clean app (v1.0)
cd "Company Employee Hirarchy/v1.0_Synthetic_Data (No Noise)"
streamlit run App.py

# Messy app (v2.0)
cd "Company Employee Hirarchy/v2.0_Messy_Data (Noise)"
streamlit run App.py

# Final messy app with ground-truth comparison (v2.1)
cd "Company Employee Hirarchy/v2.1_Mess_Data (Final)"
streamlit run App.py
```

Upload `Employees_Database.csv` in the sidebar. *v2.0_Messy_Data (Noise)* and *v2.1_Mess_Data (Final)* will also auto-load an `Employees_Database.csv` that sits next to `App.py`.

### Which app for which file?

| Your CSV | Use |
|---|---|
| `Employees_Database.csv` from `Generate_Synthetic_Data.py` | **v1.0_Synthetic_Data (No Noise)** |
| `Employees_Database.csv` from `Generate_Messy_Data.py` | **v2.0_Messy_Data (Noise)** (or **v2.1_Mess_Data (Final)** if you also have the ground truth) |
| Any real-world export | **v2.0_Messy_Data (Noise)** — but read the size limits in [§10](#10-known-limitations-and-quirks) first |

---

## 4. Data model

### Input columns (what the engines see) — 11 columns

| Column | Example | Notes |
|---|---|---|
| `Company Name` | `Acosta Group` | |
| `Industry` | `Information Technology` | One of 15; one per company |
| `Designation` | `Senior Software Engineer` | Job title |
| `Seniority Level` | `5` | Integer 1–10 as text (see scale below) |
| `Department` | `Engineering` | Six real departments + `Executive` for the CEO |
| `Team/Business Unit` | `Backend Services` | 6 teams per department, plus `Executive Office` (CEO) and `Department Leadership` (heads) |
| `Employee Name` | `Douglas Acosta` | |
| `Skills/Specialization` | `Java; Microservices` | `;`-separated; 2–3 for most people |
| `City`, `Country` | `Clayton`, `USA` | Countries: USA, India, UK, Australia, Germany, Canada |
| `Email` | `douglas.acosta@acosta.com` | One domain per company |

### Ground-truth file — 13 columns

The 11 above **plus** `Employee ID` (`C0001-E00042` = company index + employee number) and `Manager ID` (blank only for the CEO).

### Seniority scale

| Level | Meaning | Can have direct reports? |
|---|---|---|
| 10 | CEO (company root) | yes |
| 9 | Department head (VP / Director) | yes |
| 8 | Team manager | yes |
| 6 | Lead | **no** |
| 5 | Senior | **no** |
| 3 | Mid-level (default) | **no** |
| 1 | Junior / Associate / Intern | **no** |

**Hard rule used everywhere:** only level ≥ 8 can ever be someone's manager.

### Organisation template (identical in both generators)

| Department | Head (L9) | Manager titles (L8) | Individual-contributor titles |
|---|---|---|---|
| Engineering | VP Engineering | Engineering Manager, Software Development Manager, QA Manager, Infrastructure Manager | Technical Lead, Senior Software Engineer, Software Engineer, Junior Software Engineer, Associate Engineer |
| Sales | VP Sales | Sales Operations Manager, Senior Sales Manager, Sales Manager, Regional Sales Manager | Account Executive, Sales Representative, Business Development Representative, Client Success Associate, Sales Operations Associate |
| Human Resources | HR Director | HR Manager, Talent Acquisition Manager, HR Operations Manager, Compensation and Benefits Manager | HR Business Partner, HR Executive, Recruiter |
| Finance | Finance Director | Finance Manager, Accounting Manager, Tax Manager, Financial Planning Manager | Senior Financial Analyst, Financial Analyst, Accountant |
| Marketing | Marketing Director | Marketing Manager, Digital Marketing Manager, Brand Manager, Content Marketing Manager, PR Manager | Marketing Analyst, Marketing Executive, Content Specialist, Social Media Intern |
| Operations | Operations Director | Operations Manager, Regional Operations Manager, Process Manager, Supply Chain Manager | Operations Analyst, Operations Executive, Process Associate |

Each department has six teams (e.g. Engineering: Platform Engineering, Backend Services, Frontend & Mobile, Quality Assurance, Data Engineering, Security Engineering), and each team has four characteristic skills (e.g. Backend Services → Java, Python, Microservices, REST API Design).

**How a title becomes a level** (`get_seniority` in the generators): `CEO` → 10; a department-head title → 9; a manager title → 8; otherwise the word `lead` → 6, `senior` → 5, `junior`/`associate`/`intern` → 1, anything else → 3. (Manager titles are checked first, which is why "Senior Sales Manager" is level 8, not 5.)

---

## 5. The data generators, in depth

### 5.1 `Generate_Synthetic_Data.py` — the clean generator

**Purpose.** Produce a perfectly regular dataset where every reporting line is known. This is the "easy mode" dataset.

#### Step-by-step

| Section in file | What it does |
|---|---|
| **Logging setup** | Creates `log_<timestamp>.txt`, configures `logging` to write to both file and console, then replaces `sys.stdout`/`sys.stderr` with a `LoggerWriter` object so every `print()` is logged. |
| **1. Configuration** | `get_input()` prompts for an integer with a default and optional minimum; asks for companies, min/max employees, seed. Seeds `random`. |
| **Separate RNG** | `attr_rng = random.Random(SEED)` is used only for the later-added attributes (industry, team order, skills). This keeps names, cities and designations identical to older runs with the same seed. |
| **2. Faker setup** | One `Faker` per country (`en_US`, `en_IN`, `en_GB`, `en_AU`, `de_DE`, `en_CA`), each seeded with `SEED + index`. |
| **3. Organisation structure** | The department/head/manager/employee template from §4. |
| **3B. Reference data** | `INDUSTRIES` (15), `TEAMS` (6 per department, 4 skills each), skill pools for CEO / heads / managers. |
| **`get_seniority()`** | Title → level string (`"10"`, `"9"`, `"8"`, `"6"`, `"5"`, `"3"`, `"1"`). |
| **`get_skills()`** | CEO: 3 CEO skills. Head: 3 head skills. Manager: 1 generic management skill + 2 skills of their team. Individual contributor: 2–3 skills of their team. Joined with `"; "`. |
| **4. Company names** | Calls `Faker.company()` until `N` unique names exist, then sorts them (so company index → name is stable). |
| **5. `create_employee()`** | Builds one record: ID `C{company:04d}-E{number:05d}`, a Faker name for the company's country, email `first.last.<number>@company<index>.example`, level, skills. |
| **6. Company loop** | The hierarchy generation described below. |
| **7–8. DataFrame + validation** | 13 assertions (listed below). |
| **9–10. Export** | `Synthetic_Company_Employees.csv` (13 columns, ground truth) and `Employees_Database.csv` (the 11 non-ID columns, rows shuffled). |
| **11. Summary** | Prints counts and distributions. |

#### How the reporting relationships are generated

For each company:

1. **Size and place.** `employee_count = randint(MIN, MAX)`; one random country; 10 cities sampled from that country's Faker; one random industry; email domain `company0001.example`, `company0002.example`, …
2. **CEO.** Employee #1: `Manager ID = None`, department `Executive`, team `Executive Office`.
3. **Split the rest across the 6 departments.**
   ```
   remaining = employee_count - 1
   department_sizes = [8, 8, 8, 8, 8, 8]            # every department gets at least 8
   repeat (remaining - 48) times:  add 1 to a uniformly random department
   ```
4. **For each department:**
   - **Head** → `Manager ID` = the CEO.
   - **Number of managers** = `max(1, ceil((dept_size - 1) / 10))`, i.e. roughly one manager per 10 people.
   - **Teams.** The six teams of the department are shuffled; manager *m* leads team `shuffled[m % 6]`. (A department can only have two managers on one team if it has 7+ managers, i.e. 62+ people.)
   - **Managers** each get a random manager title from the department's list and report to the head.
   - **Individual contributors** = `dept_size - 1 - manager_count`. Contributor *i* is assigned **round-robin**: manager `i % manager_count`, and inherits that manager's team.

**Worked example.** A 110-person company: 109 non-CEO people → 48 reserved (8 × 6) → 61 spread randomly. Suppose Engineering ends with 25 people: 1 head + `ceil(24/10) = 3` managers + 21 contributors. Contributors 0,3,6,… report to manager 0; 1,4,7,… to manager 1; 2,5,8,… to manager 2 → 7 each. Everyone in the chain shares the department; each contributor shares their manager's team.

#### The 13 validation checks

Unique Employee IDs · unique emails · every company within the min/max size · all Manager IDs exist · exactly one CEO per company · CEO has no manager · everyone else has a manager · manager and employee are in the same company · nobody reports to themselves · the new columns are never blank · one industry per company · contributors share their manager's team. The script stops with an `AssertionError` if any fails.

> **Reproducibility note.** Everything is seeded except the final shuffle of `Employees_Database.csv` (`sample(frac=1, random_state=None)`), so the row order of that file differs between runs even with the same seed. The ground-truth file is deterministic.

---

### 5.2 `Generate_Messy_Data.py` — the realistic generator

This script has two halves, deliberately driven by **two independent random generators** so the mess never changes the clean data:

```python
rng  = random.Random(SEED)          # all "clean" structure
mrng = random.Random(SEED + 1000)   # all "mess"
```

#### Half A — a clean-but-quirky ground truth

The hierarchy is still strictly valid (every non-CEO has a manager in the same company, seniority always decreases down the chain, teams line up), but it has real-world oddities:

| Feature | How it works |
|---|---|
| **Tiers vs titles** | Every node has a hidden *tier* — `CEO`, `HEAD`, `MGR`, `IC` — which fixes its true place in the hierarchy **independently of what its title says**. The chain is always IC → MGR → HEAD → CEO. |
| **Quirk rate per company** | `company_quirk_rate()` draws how "weird" a company is: 35% of companies are conventional (0.5% odd titles), 45% mildly odd (2–8%), 15% startup-ish (10–20%), 5% very weird (30–50%). |
| **Vanity titles** | `pick_title()` may replace a standard title with `Technoking`, `Chief Everything Officer`, `Glubgogabgalab`, `Code Ninja`, `Minister of Vibes`, … Each wacky title carries its **true level** (e.g. `Deal Slayer` → 5, `Handshake Guru` → 1) so relationships remain valid. |
| **Alternate executive titles** | CEO may appear as `Founder & CEO`, `Managing Director`, …; heads as `CTO`, `CFO`, `Head of Sales`, `Chief People Officer`, … |
| **Uneven department sizes** | Min 8 each, remainder spread with weights `[0.34, 0.22, 0.06, 0.08, 0.12, 0.18]` for Engineering, Sales, HR, Finance, Marketing, Operations. |
| **Uneven team sizes** | Each manager is guaranteed ≥ 1 report; the rest are assigned randomly with per-manager weights drawn from `uniform(0.4, 2.0)`. |
| **Seniority mix of contributors** | Standard contributor titles are drawn with weights by level: Lead 8%, Senior 22%, Mid 40%, Junior 30%. |
| **Remote staff** | 12% (`OFFSHORE_RATE`) of non-leadership people work in a country other than HQ. CEO and heads always sit in the HQ country; leaders are placed in the HQ city 80% of the time. |
| **Email conventions** | Each company draws one pattern from `{f}.{l}` (45%), `{fi}{l}` (20%), `{f}{l}`, `{f}_{l}`, `{f}{li}`, `{f}`, `{l}.{f}`. Domain is derived from the company name (stop-words and suffixes removed), TLD is the country's TLD 60% of the time else `.com`; clashes get `group`/`hq`/index suffixes. |
| **Non-sequential IDs** | The CEO is always `E00001`; everyone else gets distinct random numbers drawn from `2 … 1.4·n + 2`, so there are gaps (like leavers in a real system). |

The same style of validation as §5.1 runs here, plus: chain-of-command by tier, seniority strictly decreasing from manager to employee, seniority bands (CEO 10, HEAD 9, MGR 8, IC 1–6), department/team consistency.

#### Half B — turning the clean table into a "merged HR database"

The messy engine is built so that mess is **correlated**, like a real merged database, instead of uniformly random. Each row is produced by one of three *source-system profiles*:

| Profile | Share | Personality | Key settings |
|---|---|---|---|
| `HRIS` | 40% | Tidy modern system | abbreviations 10%, odd casing 2%, typos 0.3%, mostly `First Last` names, no encoding damage |
| `LEGACY` | 30% | Old mainframe export | ALL-CAPS 55%, abbreviations 60%, `LAST, FIRST` names, country variants 60%, encoding damage on 35% of values that contain accented characters ("Müller" → "MÃ¼ller"), nulls like `NULL`/`N/A` |
| `SHEET` | 30% | Hand-maintained spreadsheet | typos 2%, stray whitespace 8%, many placeholders (`n/a`, `-`, `?`, `TBD`, `unknown`), personal e-mail addresses 5% |

A company is mostly entered by one system: each row uses its company's profile with 70% probability, otherwise a random profile.

The mess itself, per column (each step is applied with probability `base × MESS_MULT`, where `MESS_MULT` = 0.5 / 1.0 / 1.8 for levels 1 / 2 / 3):

| Column | Corruptions |
|---|---|
| Company Name | Legal-suffix variants (`Inc` → `Inc.`, `, Inc.`, `INC`), `&`↔`and`, comma removed, leading "The", typos |
| Industry | Abbreviations/synonyms (`IT`, `Pharma`, `BFSI`, `Telco`), typos |
| Designation | Whole-title synonyms (`Senior Software Engineer` → `Sr SWE`), word abbreviations (`Mgr`, `Dev`, `Ops`), templates (`VP of X`, `Director, X`, `X Mgr.`), suffixes (`(Contract)`, `(Acting)`), casing, typos. Vanity titles are *not* abbreviated. |
| Seniority Level | `5`, `L5`, `Level 5`, `5.0`, `05`, or a word (`Senior`, `Mid`, `C-Level`) |
| Department / Team | Aliases (`Eng`, `R&D`, `People Ops`, `Biz Dev`), abbreviations, rare wrong-department tag (0.4%) |
| Employee Name | `Last, First`, `LAST, First`, all-upper/lower, `J. Smith`, `John S.`, middle initial, nickname (`Bill`), `Dr.`/`Jr.`/`PhD`, accent stripping |
| Skills | One skill dropped, shuffled, aliased (`K8s`, `SFDC`), a filler skill added (`MS Office`), lower-cased, mixed separators |
| City / Country | Accent stripping, `City, XX` suffix, country variants (`U.S.A.`, `Deutschland`, `GB`) |
| Email | Personal domains, `.con`/`,com` typos, missing `@`, upper-case |
| Any field | Whitespace noise (leading/trailing/double/non-breaking space), blanks and placeholders (base blank rates: Seniority 20%, Skills 15%, Team 12%, Email 5%, Industry 4%, …, scaled per profile), half-empty "sparse" rows |

Finally:

- **Duplicates.** `n_dup = int(rows × min(0.2, 0.015 × MESS_MULT))` people are entered twice — 30% as exact copies, 70% re-keyed through a different profile (so the duplicate looks different).
- **Junk/test rows.** `max(5, int(rows × 0.002 × MESS_MULT))` rows such as `Test User`, `asdf asdf`, `John Doe`, with mostly empty fields.
- **Shuffle** all rows, **drop** `Employee ID`/`Manager ID`, export 11 columns.

Output: `Synthetic_Company_Employees.csv` (clean ground truth) and `Employees_Database.csv` (messy).

---

## 6. The inference engines, in depth

### 6.1 Principles shared by both engines

1. **Reporting only goes "up the ladder".** A manager's level must be strictly higher than the employee's, so cycles are impossible by construction.
2. **Only level ≥ 8 can manage.** Enforced twice: candidates are filtered by level *before* scoring, and a final "safety net" pass deletes any edge that violates it.
3. **Nearest tier first.** Contributor → team manager (8) → department head (9) → CEO (10).
4. **Two kinds of evidence.** *Structural* evidence (department, team, company-wide CEO) decides who wins. *Soft* evidence (skills, title words, city, email domain) only breaks ties.
5. **A wrong line is worse than no line.** Anything below the confidence threshold, or with no clear winner, is *not* drawn; it goes to an "unresolved" list together with a best guess for human review.
6. **Confidence is a heuristic, not a probability.** It is a sum of weighted evidence clipped to 0–100 (or 0–99).

---

### 6.2 Synthetic engine (v3) — `v1.0_Synthetic_Data (No Noise)/inference_engine.py`

Designed for **clean labels**: it normalises text and matches by (normalised) equality, with small alias tables, but does no fuzzy matching or de-duplication.

```
CSV ──load_data()──► normalised table (Record ID, level, Company Key, …)
        │
        └─ per company ─► infer_company()  →  _infer_single_company()
                            ├─ validate_record()          → status "invalid_record"
                            ├─ roots = people at level 10  → "possible_root" / "ambiguous_root"
                            ├─ for each remaining person (highest level first):
                            │     _find_pool()             who could be my manager?
                            │     _score_candidate()       how well does each one fit?
                            │     tie / threshold checks   → reporting line  OR  unresolved
                            └─ safety net                  drop any illegal line
```

#### Step 1 — `load_data()`

- Reads everything as text. Column names are matched case-insensitively. Seven columns are **required** (`Company Name, Designation, Department, Employee Name, City, Country, Email`); `Industry`, `Seniority Level`, `Team/Business Unit`, `Skills/Specialization` are optional (aliases such as `seniority`, `level`, `team`, `skills` are accepted). A missing required column or an empty file raises `ValueError`.
- Adds `Record ID` = `R00000001`, `R00000002`, … (the input has no IDs).
- Builds normalised helper columns:
  - `clean_text()` — lower-case, `&` → `and`, `_ / |` → space, punctuation removed, whitespace collapsed.
  - `normalize_title()` — regex expansion of abbreviations (`vp` → `vice president`, `qa` → `quality assurance`, `sr` → `senior`, `mgr` → `manager`, `cfo` → `chief financial officer`, …) plus a few whole-title aliases.
  - `normalize_department()` — alias table (`hr`/`people`/`talent` → `human resources`, `accounting` → `finance`, `technology` → `engineering`, …).
  - Team, city, country (cleaned) and `Email Domain` (text after `@`).
  - `Function` — a keyword guess of the person's function (`engineering`, `finance`, `sales`, …) from title; falls back to the department. Used only as a *weak* department relation.
- **Level** = `parse_seniority_level()`: the numeric `Seniority Level` (1–10) if valid → else a label (`c-level`=10, `director`=9, `manager`=8, `lead`=6, `senior`=5, `mid`=3, `entry`/`junior`=1) → else `derive_level_from_title()`:

  | Title contains | Level |
  |---|---|
  | `chief executive officer`, `ceo`, or `president` (not *vice*) | 10 |
  | `chief`, `vice president`, `director`, `head`, `founder`, `chair…` | 9 |
  | `manager`, `supervisor` | 8 |
  | `lead`, `leader` | 6 |
  | `senior`, `principal`, `staff` | 5 |
  | `junior`, `associate`, `assistant`, `intern`, `trainee` | 1 |
  | anything else | 3 |

- `Company Key` = cleaned company name. If the same name has **≥ 2 level-10 records and ≥ 2 distinct industries**, the industry is appended (`name | industry`) so two unrelated companies sharing a name are not merged.
- `Layer` (0 = CEO … 3 = contributor) is stored for information; it is not used in scoring.

#### Step 2 — Roots

People at level ≥ 10 are roots. Exactly one → status `possible_root` ("CEO identified as company root"). Several → all are `ambiguous_root` and none is used as anyone's manager.

#### Step 3 — Candidate pool: `_find_pool(employee, candidates)`

1. A candidate must be level ≥ 8, **strictly above** the employee, and not the employee.
2. Take the tiers present above the employee in ascending order. The lowest is the **nearest** tier (contributor → 8, manager → 9, head → 10).
3. **Contributors never attach directly to the CEO** (tier 10 is skipped for level < 8).
4. In any non-CEO tier, a candidate qualifies only if it has a **department relation** to the employee, tried strongest first:

   | Relation | Meaning |
   |---|---|
   | `exact` | same normalised department |
   | `contains` | one department's word-set is a subset of the other's |
   | `title_mentions_dept` | the employee's department words overlap the manager's title or department words |
   | `function` *(weak)* | same inferred function |

   All tiers are tried with the *strong* relations first; only if nothing is found are *weak* relations tried.
5. If the first tier that yields a pool is **not** the nearest tier, the pool carries a **−10 "skipped tier" penalty** and a note (e.g. no head of that department exists, so a manager escalates to the CEO).

#### Step 4 — Scoring: `_score_candidate()`

`structural` decides who wins; `soft` only breaks ties. Final `total = clip(structural + soft, 0, 100)`.

| Evidence | Points | Kind |
|---|---|---|
| Base | **+55** | structural |
| Skipped-tier penalty | **−10** | structural |
| Company-wide CEO (head → CEO) | **+35** | structural |
| Department relation: `exact` / `contains` / `title_mentions_dept` / `function` | **+25 / +19 / +14 / +8** | structural |
| Team, contributor → manager: same / similar name (token Jaccard ≥ 0.5) / different | **+25 / +15 / −8** | structural |
| *Team unknown on either side:* title affinity × **24** and skills overlap × **20** | up to +24 and +20 | structural |
| Title affinity × 4 and skills overlap × 4 *(when the team is known, or for non-contributors)* | up to +4 each | soft |
| Same city +2, else same country +1 | | soft |
| Same email domain | +1 | soft |

Definitions:

- **Skills overlap** = Jaccard index `|A ∩ B| / |A ∪ B|` over the two people's skill sets.
- **Title affinity** = `|employee words ∩ manager words| / |employee words|`, after removing rank words (`senior`, `manager`, `director`, `lead`, `of`, `the`, …). If the manager's title mentions the employee's department, the department words are added to both sides.

#### Step 5 — Choosing a manager

```
sort candidates by (−structural, −soft, Record ID)
top  = candidates whose structural is within MIN_SCORE_MARGIN (4) of the best
```

- **Exactly one in `top`** → `confidence = total`. If `confidence < MIN_CONFIDENCE (58)` → unresolved, with the best guess recorded. Otherwise a reporting line is written.
  - `Runner-up Score` = best structural score among candidates at least 4 points behind (0 if there are none); `Score Margin` = best structural − runner-up.
  - `Decision` label: `department_head_to_ceo` (head → CEO), `manager_escalated_to_ceo`, `team_match` (the evidence says "same team"), otherwise `department_hierarchy`.
- **Several in `top`** (managers can't be told apart) → status `ambiguous`, *unless* shared-team guessing is on (Step 6).

#### Step 6 — Optional "guess between managers who share a team"

Employees tied between the *same set of k managers* are grouped. Each manager may take at most `ceil(n / k)` of them; (employee, manager) pairs are filled greedily in order of soft score. Confidence = `total − 40 × (1 − 1/k)`; below 58 → `ambiguous`. Lines carry the decision `shared_team_balanced` and the evidence "assigned by balanced span of control". Treat these as guesses.

#### Step 7 — Safety net

Any line whose manager is below level 8 or not strictly senior to the employee is removed and the employee is moved to `unresolved`.

#### Worked example (real engine output, 10-person company "Acme Inc")

| Employee → chosen manager | Structural | Soft | Confidence |
|---|---|---|---|
| Victor VP (9) → Alice CEO (10) | 55 + 35 (company-wide CEO) = **90** | same city 2 + same domain 1 = 3 | **93.0** |
| Mia Manager (8) → Victor VP (9) | 55 + 25 (same department) = **80** | title 1.0 × 4 = 4, city 2, domain 1 = 7 | **87.0** |
| Quinn QA (8, other city) → Victor VP | **80** | title 0.33 × 4 = 1.3, same country 1, domain 1 = 3.3 | **83.3** |
| Sam Dev (3) → Mia Manager (8) | 55 + 25 + 25 (same team) = **105** | skills 0.25 × 4 = 1, title 1.3, city 2, domain 1 = 5.3 | 110.3 → clipped to **100** |

For Sam, the runner-up is Quinn QA on a *different* team: 55 + 25 − 8 = 72, so `Score Margin = 105 − 72 = 33`. The 33-point gap (25 for a team match + 8 penalty for a mismatch) is why team labels almost always decide contributor assignments.

**Tie case.** If two managers both lead *Backend Services*, every contributor on that team scores 105 against both → within the 4-point margin → `ambiguous`. With shared-team guessing on, k = 2, capacity = `ceil(4/2) = 2`, penalty = `40 × (1 − ½) = 20`, so confidence = `100 − 20 = 80`.

**No team column.** If `Team/Business Unit` is blank, title affinity (× 24) and skills overlap (× 20) become *structural* evidence. In a quick test this made contributors cluster on whichever manager's title/skills happened to match best (see [§11](#11-appendix--verification-notes) for measured accuracy).

#### Unresolved statuses

| Status | Meaning |
|---|---|
| `possible_root` | The single CEO: no manager by design. |
| `ambiguous_root` | Several level-10 people; no safe single root. |
| `department_root` | A level ≥ 8 person with no eligible superior. |
| `ambiguous` | Two or more managers equally likely. |
| `unresolved` | No eligible manager, confidence below threshold, or edge rejected by the safety net. |
| `invalid_record` | Missing/gibberish name, designation, or company. |

---

### 6.3 Messy engine (v4) — `v2.0_Messy_Data (Noise)/inference_engine.py`

A full rewrite built on one principle: **the table is the only source of truth**. Anything the engine "knows" (what a title means, which team belongs to which department, which skills belong to which team) is **learned from the table itself**.

```
CSV ─► STAGE 1  prepare()  ─► PreparedData(employees, unassigned, companies, report)
                │               clean · identify companies · merge duplicates · fill blanks
                ▼
       STAGE 2  infer_company() per Company Key ─► (edges, unresolved)
                                  CEO ◄─ heads ◄─ managers ◄─ contributors   (rule ladder)
```

#### Stage 1 — `prepare(source, infer_from_skills=True)`

**1. Columns.** `_read_columns()` accepts many aliases (`company`, `employer`, `title`, `job title`, `grade`, `dept`, `full name`, `e-mail`, …). Only `Company Name` and `Employee Name` are mandatory; others default to blank. Non-breaking spaces are replaced. `Source Row` records the row number as seen in a spreadsheet (starts at 2).

**2. Text helpers.** `ct()` = ASCII-lower-case with `& → and`, `- / → space`, punctuation stripped. `fold()` = accent-folding for names. `PLACEHOLDERS` (`""`, `n a`, `null`, `none`, `tbd`, `unknown`, `test`, `asdf`, …) are treated as blank. `closest()` wraps `difflib.get_close_matches` with a cache; all fuzzy matching uses it, each with its own cut-off:

| Fuzzy match of … | Cut-off |
|---|---|
| industry spelling | 0.74 |
| department spelling | 0.76 |
| team name (to the team vocabulary) | 0.82 |
| company name / email domain (to a known company) | 0.86 |

**3. Industry.** `canonical_industry()` maps every spelling (`Auto`, `Automotive Industry`, `Automobile`, typos) to one of 15 canonical industries via an alias table, then fuzzy matching. (In the verification run, 1,171 distinct spellings collapsed to 15.)

**4. Company identity — "one real company = one email domain".**
- `company_key()` normalises a name: fold accents, `&` → `and`, drop leading "the", strip legal suffixes (`inc, ltd, llc, plc, corp, co, company, group, grp, gmbh, pvt, sa, ag`).
- `_build_anchors()` creates an **anchor** for every email domain that appears on **≥ 12 rows** (`MIN_DOMAIN_ROWS`), contains a dot, and is not free-mail (`gmail.com`, …) or junk (`test.com`, …). The anchor stores the most common name key, display name, industry and country on that domain.
- `_resolve_company_rows()` assigns every row to an anchor:
  - If the row's domain *is* an anchor → done.
  - Otherwise score candidate anchors: domain typo (`whitelewis.con`; difflib ≥ 0.86 against anchors) **+5**; company-name key exact **+4** or typo-close **+3**; industry agrees **+2** / conflicts **−4**; same country **+1**.
  - Accept only if the best score is **≥ 4** *and* leads the runner-up by **≥ 2**; otherwise the row goes to `unassigned` ("left out rather than guessed").

**5. Field cleaning.**
- Names (`clean_person_name`): `LAST, First` → `First Last`, suffixes (`Jr`, `PhD`, …) removed, all-upper/lower → title-case.
- Titles (`title_key`): expand `sr`, `mgr`, `dir`, `eng`, `swe`, `vp`, …; drop noise words (`contractor`, `temp`, `remote`).
- Level (`parse_level`): first number 1–10 in the cell, or a word (`senior`, `c-level`, …); 0 if unreadable.
- Department (`canonical_department`): alias table → fuzzy; returns `()` (unknown), `('Sales',)` (clear), or two departments for hybrids such as "Sales and Marketing".
- Team (`team_key`): expand abbreviations; map `leadership`/`management`/`executive office`/`c suite` → `department leadership`.

**6. Team vocabulary (learned).** A team label is part of the vocabulary only if it occurs **≥ 150 times** in the table. Other labels are fuzzy-mapped (cut-off 0.82) to a vocabulary label, or **blanked** if nothing is close enough.

**7. Learned lookup tables.** Built only from rows where the answer is clear:

| Learned map | Rule |
|---|---|
| `team_dept` — team → department | team appears with one department in **≥ 70%** of rows |
| `title_dept` — title → department | title seen **≥ 3×** and **≥ 75%** agree |
| `title_level` — title → level | title seen **≥ 3×** with a level, **≥ 60%** agree |
| skill → team, skill → department (`_Voter`) | each skill's team distribution; prediction is accepted when the vote share is **≥ 80%** |

`_Voter.predict(features)` adds up, for each skill, the probability of each label; the top label wins if `top_score / number_of_known_skills ≥ min_share`.

**8. Per-row resolution** (each value remembers its source: `Level Source`, `Department Source`, `Team Source`):

| Field | Order of attempts |
|---|---|
| **Level** | the `Seniority Level` column → learned from the designation → designation keywords (`rule_level`) → default **3** |
| **Department** | the column → *(hybrid: pick via team, else via title, else first)* → learned from team → learned from designation → learned from skills |
| **Team** | the column → *(if blank, level < 10, skills enabled)* predicted from skills, accepted only if it agrees with the department |

**9. Validity and duplicates.** A row is `Valid` only if it has a clean name and its designation is not `test/tester/xxx/asdf`. `_merge_duplicates()` merges rows (union-find) that share **company + email**, or **company + name signature + title**; the most complete row survives and fills its blanks from the others (`Source Rows` lists the merged spreadsheet rows).

Finally, `Record ID` = `R000001…`, a `companies` table (one row per anchor, with a label that adds industry/domain only to separate same-named companies), and a `report` dict with all the cleaning statistics.

#### Stage 2 — `infer_company(company_df, config)`

Settings are in `InferenceConfig`:

| Field | Default | Meaning |
|---|---|---|
| `min_confidence` | 60 | Lines scoring below this are not drawn |
| `min_margin` | 5 | Best candidate must beat the runner-up's soft score by this much |
| `allow_skip_level` | True | If a team has no manager, attach to the department head (and a manager with no head to the CEO) |
| `guess_between_ties` | False | Spread people across equally likely managers (a guess) |

**Roots.** One person at level 10 → the root. Several → the root is the single one whose title contains "chief executive"/"ceo"; if that is not unique → all are `ambiguous_root`. Level-10 people other than a unique root are treated like department heads (but see the quirk in §10).

**The rule ladder** (people are grouped into `heads` = level 9, `managers` = level 8, `people` = level < 8):

| Who | Candidate pool | Base score | Adjustments | `Decision` |
|---|---|---|---|---|
| Head (level 9) | the CEO | **92** (fixed) | — | `head_to_ceo` |
| Manager (8) | heads of the **same department** | **85** (+5 if only one head) | −6 if the department was inferred, not given | `manager_to_head` |
| Manager, no head in department, CEO exists, skip-level on, department known | the CEO | **64** (fixed) | — | `manager_skip_to_ceo` |
| Contributor with a team | managers with the **same team** (not contradicting department) | **90** if one manager, **72** if several | −7 if the contributor's or any candidate's team came from skills | `team_manager` |
| Contributor, team blank, exactly one manager in the department | that manager | **66** | — | `department_manager` |
| Contributor, no manager on the team, skip-level on | heads of the department | **63** | −4 if department was inferred | `skip_to_head` |

**`pick()` — choosing among candidates** (used by every row of the ladder):

1. Rank candidates; when the pool is heads, the primary sort key is `head_rank()` (Chief/C?O = 3, VP/Head = 2, other = 1) and only the top rank stays in contention.
2. Compute each remaining candidate's **soft score** (below). Sort by (−rank, −soft, Record ID).
3. If there is one candidate, or `best_soft − runner_soft ≥ min_margin`:
   ```
   confidence = base + min(soft, 20) × (0.35 if pool has one candidate else 0.75)     # clipped to 0–99
   ```
   If `confidence < min_confidence` → `unresolved`; else write the line. `Runner-up Score = base + runner_soft × 0.75` (0 if single), `Score Margin = confidence − runner-up`.
4. Otherwise it is a tie → status `ambiguous`. If `guess_between_ties` is on, the tied manager is chosen deterministically as `tied[int(record_digits) % k]` and `confidence = base + min(soft,20) × 0.75 − 25 × (1 − 1/k)`, decision `tie_guess`.

**Soft score** (`_soft_score`, ranges from 0 up to about 32):

```
skills overlap (Jaccard) × 20
+ title-word overlap × 8        # share of the employee's title words found in the candidate's title (stop-words removed)
+ 4 if same city, else +1 if same country
```

**Safety net.** As in v3: any line with manager level < 8 or ≤ employee level is removed and the employee marked `unresolved`.

**Statuses:** `root`, `department_root`, `ambiguous_root`, `ambiguous`, `unresolved`, `invalid_record`.

#### Worked example (real engine output)

Company: CEO *Cleo*, head *Vic* (VP Engineering, level 9), manager *Mia* (Engineering Manager, team Backend Services), contributor *Ian* (Software Engineer, same team), all in the same city, no skills data.

| Line | Calculation | Confidence |
|---|---|---|
| Vic → Cleo | fixed head → CEO | **92.0** |
| Mia → Vic | base 85 + 5 (only one head) = 90; soft = title overlap 1.0 × 8 + same city 4 = 12; 90 + 12 × 0.35 | **94.2** |
| Ian → Mia | base 90 (only one manager on the team); soft = same city 4 (no title/skill overlap); 90 + 4 × 0.35 | **91.4** |

---

### 6.4 Ground-truth comparison (read-only)

Present in the *Synthetic Data App* (inside `App.py`) and the *True Compare App* (inside its `inference_engine.py`). It never influences inference; it only grades the result.

**Matching people.** The truth file's people must be linked to the engine's records.
- *Synthetic App* (`build_comparison`): key = (company key, e-mail), falling back to (company key, name + designation).
- *True Compare App* (`compare_to_truth`): by e-mail (if several, narrowed by name signature); otherwise by (company key, name signature), narrowed by title. A person is matched only if exactly one candidate remains.

**Outcome for each truth person**

| Truth | Inferred | Outcome |
|---|---|---|
| has a manager | same manager | `correct` |
| has a manager | different manager | `wrong` |
| has a manager | none | `missed` |
| no manager (CEO) | none | `correct (no manager)` |
| no manager (CEO) | some manager | `false line (truth: top of company)` |

People (or managers) that can't be matched are counted separately and **not scored**.

**Metrics** (`score_rows`)

```
precision = correct / (correct + wrong + false lines)      # of the lines I drew, how many are right
recall    = correct / (correct + wrong + missed)           # of the true lines, how many did I get right
```

The True Compare tab also breaks precision down **by decision type** and **by confidence band** (`<70`, `70–79`, `80–89`, `90+`) — useful for choosing `min_confidence` and deciding whether to switch off skip-level lines.