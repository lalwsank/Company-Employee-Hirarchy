import html
import io
import re
import threading
from collections import defaultdict
import pandas as pd
import streamlit as st
import inference_engine as engine

_INFERENCE_LOCK = threading.RLock()

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Org Chart Explorer",
    page_icon="🏢",
    layout="wide",
)

st.title("🏢 Organization Chart Explorer")
st.caption(
    "Infer employee-manager relationships, review confidence scores, "
    "and explore company reporting structures."
)

# ============================================================
# 2. OUTPUT SCHEMAS
# ============================================================

EDGE_COLUMNS = [
    "Company Name",
    "Employee Record ID",
    "Employee Name",
    "Employee Designation",
    "Employee Level",
    "Manager Record ID",
    "Manager Name",
    "Manager Designation",
    "Manager Level",
    "Manager Team",
    "Confidence",
    "Runner-up Score",
    "Score Margin",
    "Evidence",
    "Decision",
]

UNRESOLVED_COLUMNS = [
    "Record ID",
    "Company Name",
    "Employee Name",
    "Designation",
    "Status",
    "Reason",
    "Best Guess Manager ID",
    "Best Guess Score",
]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=False, max_entries=3)
def run_inference(file_bytes, min_confidence, min_margin, allow_shared_teams):
    """
    Run inference for every company in the uploaded dataset.
    Applies UI configuration for the duration of the run, then restores
    the original module settings.
    """
    with _INFERENCE_LOCK:
        original_min_confidence = engine.MIN_CONFIDENCE
        original_min_score_margin = engine.MIN_SCORE_MARGIN
        original_allow_shared = engine.ALLOW_SHARED_TEAM_ASSIGNMENT

        try:
            engine.MIN_CONFIDENCE = int(min_confidence)
            engine.MIN_SCORE_MARGIN = int(min_margin)
            engine.ALLOW_SHARED_TEAM_ASSIGNMENT = bool(allow_shared_teams)

            employees_df = engine.load_data(io.BytesIO(file_bytes))

            all_edges = []
            all_unresolved = []

            for _, company_df in employees_df.groupby("Company Key", sort=False):
                edges, unresolved = engine.infer_company(company_df)
                all_edges.extend(edges)
                all_unresolved.extend(unresolved)

        finally:
            engine.MIN_CONFIDENCE = original_min_confidence
            engine.MIN_SCORE_MARGIN = original_min_score_margin
            engine.ALLOW_SHARED_TEAM_ASSIGNMENT = original_allow_shared

    edges_df = pd.DataFrame(all_edges, columns=EDGE_COLUMNS)
    unresolved_df = pd.DataFrame(all_unresolved, columns=UNRESOLVED_COLUMNS)

    csv_downloads = {
        "employees": employees_df.to_csv(index=False).encode("utf-8"),
        "edges": edges_df.to_csv(index=False).encode("utf-8"),
        "unresolved": unresolved_df.to_csv(index=False).encode("utf-8"),
    }

    return employees_df, edges_df, unresolved_df, csv_downloads


def safe_value(value):
    """Convert values to safe display strings."""
    if pd.isna(value):
        return ""
    return str(value)


# Comparison markers (only used when ground truth is supplied).
MARKER_SYMBOLS = {"ok": "\u2713", "bad": "\u2717", "miss": "\u2013"}
MARKER_TITLES = {
    "ok": "Manager matches ground truth",
    "bad": "Manager differs from ground truth",
    "miss": "Ground truth has a manager but none was inferred",
}

def build_company_options(employees_df):
    """
    Map Company Key -> display label. Companies that share a name (but are
    different organisations, separated by industry) get the industry added.
    """
    firsts = employees_df.drop_duplicates("Company Key")[
        ["Company Key", "Company Name", "Industry"]
    ]
    name_counts = firsts["Company Name"].value_counts()
    options = {}
    for key, name, industry in zip(
        firsts["Company Key"], firsts["Company Name"], firsts["Industry"]
    ):
        label = name
        if name_counts[name] > 1 and str(industry).strip():
            label = f"{name} ({industry})"
        options[key] = label
    return dict(sorted(options.items(), key=lambda item: item[1].lower()))


STATUS_LABELS = {
    "possible_root": "Possible root",
    "ambiguous_root": "Ambiguous root",
    "department_root": "Department root",
    "ambiguous": "Ambiguous",
    "unresolved": "Unassigned",
    "invalid_record": "Invalid record",
}


def build_employee_tree(
    employee_id,
    employee_lookup,
    children_lookup,
    confidence_lookup,
    unresolved_lookup,
    visited=None,
    depth=0,
    parts=None,
    marker_lookup=None,
):
    """
    Recursively construct an expandable HTML organization tree.
    Every fragment is emitted WITHOUT indentation or blank lines so that
    st.markdown() does not treat the HTML as a Markdown code block.
    """
    is_top_level = parts is None
    if parts is None:
        parts = []
    if visited is None:
        visited = set()

    if employee_id in visited:
        parts.append(
            '<div class="tree-warning">'
            "Circular reporting relationship detected."
            "</div>"
        )
        return "".join(parts) if is_top_level else None

    if depth > 100:
        parts.append(
            '<div class="tree-warning">'
            "Maximum display depth reached."
            "</div>"
        )
        return "".join(parts) if is_top_level else None

    employee = employee_lookup.get(employee_id)
    if employee is None:
        return "" if is_top_level else None

    visited.add(employee_id)
    esc = html.escape

    name = esc(safe_value(employee.get("Employee Name")))
    designation = esc(safe_value(employee.get("Designation")))
    department = esc(safe_value(employee.get("Department"))) or "N/A"
    city = esc(safe_value(employee.get("City"))) or "N/A"
    country = esc(safe_value(employee.get("Country"))) or "N/A"
    team = esc(safe_value(employee.get("Team/Business Unit"))) or "N/A"
    level = esc(safe_value(employee.get("Seniority Level"))) or "N/A"

    confidence = confidence_lookup.get(employee_id)
    if confidence is not None:
        confidence_class = (
            "high" if confidence >= 85
            else "medium" if confidence >= 70
            else "low"
        )
        confidence_badge = (
            f'<span class="conf {confidence_class}" '
            f'title="Confidence of this reporting line">'
            f'{confidence:.0f}%</span>'
        )
    else:
        confidence_badge = ""

    status_text = STATUS_LABELS.get(unresolved_lookup.get(employee_id), "")
    status_badge = (
        f'<span class="tag">{esc(status_text)}</span>' if status_text else ""
    )

    direct_reports = children_lookup.get(employee_id, ())
    tooltip = (
        f"{department} \u00b7 {team} \u00b7 level {level} \u00b7 {city}, {country}"
    )

    marker = (marker_lookup or {}).get(employee_id)
    marker_badge = (
        f'<span class="mk {marker}" title="{MARKER_TITLES[marker]}">'
        f'{MARKER_SYMBOLS[marker]}</span>'
        if marker in MARKER_SYMBOLS else ""
    )

    label = (
        f'<span class="emp-name" title="{tooltip}">{name}</span>'
        f'<span class="emp-title">{designation}</span>'
        f'{confidence_badge}{status_badge}{marker_badge}'
    )

    # Leaves are plain rows (no collapsible wrapper) to keep the tree light.
    if not direct_reports:
        parts.append(f'<div class="emp-leaf">{label}</div>')
        return "".join(parts) if is_top_level else None

    report_badge = f'<span class="reports" title="Direct reports">{len(direct_reports)}</span>'
    # Only the first two levels start expanded.
    open_attr = " open" if depth < 2 else ""

    parts.append(
        f'<details class="emp-node"{open_attr}>'
        f'<summary>{label}{report_badge}</summary>'
    )

    for child_id in direct_reports:
        build_employee_tree(
            employee_id=child_id,
            employee_lookup=employee_lookup,
            children_lookup=children_lookup,
            confidence_lookup=confidence_lookup,
            unresolved_lookup=unresolved_lookup,
            visited=visited,
            depth=depth + 1,
            parts=parts,
            marker_lookup=marker_lookup,
        )

    parts.append("</details>")
    return "".join(parts) if is_top_level else None


def _build_lookups(employees_df, edges_df, unresolved_df):
    """Shared lookups used by both the CEO tree and the unmatched trees."""
    employee_lookup = employees_df.set_index("Record ID", drop=False).to_dict("index")

    sort_name = {
        employee_id: safe_value(row["Employee Name"]).lower()
        for employee_id, row in employee_lookup.items()
    }

    children_lookup = defaultdict(list)
    confidence_lookup = {}
    employee_to_manager = {}

    for employee_id, manager_id, confidence in zip(
        edges_df["Employee Record ID"],
        edges_df["Manager Record ID"],
        edges_df["Confidence"],
    ):
        if employee_id not in employee_lookup:
            continue
        if manager_id not in employee_lookup:
            continue
        children_lookup[manager_id].append(employee_id)
        confidence_lookup[employee_id] = float(confidence)
        employee_to_manager[employee_id] = manager_id

    unresolved_lookup = dict(
        zip(
            unresolved_df["Record ID"],
            unresolved_df["Status"].astype(str),
        )
    )

    for children in children_lookup.values():
        children.sort(key=sort_name.__getitem__)

    return (
        employee_lookup,
        sort_name,
        children_lookup,
        confidence_lookup,
        employee_to_manager,
        unresolved_lookup,
    )


def create_company_trees(employees_df, edges_df, unresolved_df, marker_lookup=None):
    """
    Split the organization into two sets of trees:

    1. CEO-connected hierarchy – every employee whose reporting chain
       reaches a CEO / possible_root.
    2. Unmatched relationships – confident reporting hierarchies that
       exist but do NOT roll up to the CEO (e.g. department trees when
       no unique CEO was found, or any other disconnected subtree).

    Completely unresolved people (no edges at all) are excluded from
    both trees; they appear only in the unresolved data tables.
    """
    (
        employee_lookup,
        sort_name,
        children_lookup,
        confidence_lookup,
        employee_to_manager,
        unresolved_lookup,
    ) = _build_lookups(employees_df, edges_df, unresolved_df)

    # Identify CEO / possible-root IDs.
    ceo_root_ids = [
        eid
        for eid, status in unresolved_lookup.items()
        if status in {"possible_root", "ambiguous_root"}
        and eid in employee_lookup
    ]

    # Walk downward from every CEO root to collect everyone under the CEO.
    under_ceo = set()
    stack = list(ceo_root_ids)
    while stack:
        current = stack.pop()
        if current in under_ceo:
            continue
        under_ceo.add(current)
        stack.extend(children_lookup.get(current, ()))

    # Any root that is itself under the CEO is already covered.
    # Roots that are NOT under the CEO form the unmatched hierarchies.
    all_root_ids = [
        eid
        for eid in employee_lookup
        if eid not in employee_to_manager
    ]

    unmatched_root_ids = [
        eid
        for eid in all_root_ids
        if eid not in under_ceo
        # Only include roots that actually have descendants (i.e. real
        # relationships). Lone unresolved people stay out of this section.
        and children_lookup.get(eid)
    ]

    # Also treat any employee who has a manager but whose chain never
    # reaches a CEO-root as part of an unmatched subtree. Their highest
    # ancestor that is not under the CEO becomes an unmatched root.
    for eid in employee_lookup:
        if eid in under_ceo or eid in employee_to_manager:
            continue
        # already considered as a root
    for eid, manager_id in employee_to_manager.items():
        if eid in under_ceo:
            continue
        # climb to the top of this chain
        current = eid
        seen = set()
        while current in employee_to_manager and current not in seen:
            seen.add(current)
            current = employee_to_manager[current]
        if current not in under_ceo and current not in unmatched_root_ids:
            if children_lookup.get(current) or current in employee_to_manager:
                unmatched_root_ids.append(current)

    unmatched_root_ids = sorted(
        set(unmatched_root_ids),
        key=lambda eid: (
            -int(employee_lookup[eid].get("Seniority", 0)),
            sort_name[eid],
        ),
    )

    def render_tree(root_list):
        visited = set()
        parts = []
        for root_id in root_list:
            if root_id in visited:
                continue
            parts.append(
                build_employee_tree(
                    employee_id=root_id,
                    employee_lookup=employee_lookup,
                    children_lookup=children_lookup,
                    confidence_lookup=confidence_lookup,
                    unresolved_lookup=unresolved_lookup,
                    visited=visited,
                    marker_lookup=marker_lookup,
                )
            )
        return "".join(parts), visited

    ceo_html, ceo_visited = render_tree(ceo_root_ids)
    unmatched_html, unmatched_visited = render_tree(unmatched_root_ids)

    return ceo_html, unmatched_html, under_ceo, unmatched_visited


# ============================================================
# 3b. GROUND TRUTH (comparison only - never used by inference)
# ============================================================

TRUTH_REQUIRED = ["Employee ID", "Manager ID", "Company Name", "Employee Name"]
TRUTH_OPTIONAL = [
    "Designation", "Department", "City", "Country", "Email",
    "Team/Business Unit", "Seniority Level",
]


def _clean_id(value):
    value = "" if pd.isna(value) else str(value).strip()
    if value.lower() in {"nan", "none", "null"}:
        return ""
    return re.sub(r"\.0$", "", value)


@st.cache_data(show_spinner=False, max_entries=3)
def load_truth_data(file_bytes):
    """Parse the ground-truth CSV. Independent of the inference pipeline."""
    df = pd.read_csv(io.BytesIO(file_bytes), dtype=str)
    lookup = {str(c).strip().lower(): c for c in df.columns}
    rename = {}
    for col in TRUTH_REQUIRED:
        if col.lower() not in lookup:
            raise ValueError(f"Ground truth is missing column: {col}")
        rename[lookup[col.lower()]] = col
    for col in TRUTH_OPTIONAL:
        if col.lower() in lookup:
            rename[lookup[col.lower()]] = col
    df = df.rename(columns=rename)
    for col in TRUTH_OPTIONAL:
        if col not in df.columns:
            df[col] = ""
    df = df[TRUTH_REQUIRED + TRUTH_OPTIONAL].copy()
    for col in ["Company Name", "Employee Name"] + TRUTH_OPTIONAL:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df["Employee ID"] = df["Employee ID"].map(_clean_id)
    df["Manager ID"] = df["Manager ID"].map(_clean_id)
    df = df[df["Employee ID"] != ""].drop_duplicates("Employee ID")
    df["Company Key"] = df["Company Name"].map(engine.clean_text)
    return df.reset_index(drop=True)


def _match_key(company_key, email, name, designation):
    email = str(email).strip().lower()
    if email:
        return (company_key, email)
    return (company_key, f"n|{str(name).strip().lower()}|{str(designation).strip().lower()}")


@st.cache_data(show_spinner=False, max_entries=3)
def build_comparison(truth_df, employees_df, edges_df):
    """
    Read-only join of ground truth against inferred edges.
    People are matched by (company name, email), falling back to
    name + designation. Nothing here feeds back into the inference.
    Statistics are keyed by the engine's Company Key, so same-named
    companies in different industries are scored separately.
    """
    base_keys = employees_df["Company Name"].map(engine.clean_text)
    record_company = dict(zip(employees_df["Record ID"], employees_df["Company Key"]))

    inferred_index = defaultdict(list)
    for rid, ck, email, name, desig in zip(
        employees_df["Record ID"],
        base_keys,
        employees_df["Email"],
        employees_df["Employee Name"],
        employees_df["Designation"],
    ):
        inferred_index[_match_key(ck, email, name, desig)].append(rid)

    truth_to_inferred, used = {}, defaultdict(int)
    for tid, ck, email, name, desig in zip(
        truth_df["Employee ID"],
        truth_df["Company Key"],
        truth_df["Email"],
        truth_df["Employee Name"],
        truth_df["Designation"],
    ):
        key = _match_key(ck, email, name, desig)
        candidates = inferred_index.get(key, [])
        i = used[key]
        used[key] += 1
        if i < len(candidates):
            truth_to_inferred[tid] = candidates[i]

    inferred_edge = dict(
        zip(edges_df["Employee Record ID"], edges_df["Manager Record ID"])
    )

    truth_manager_inferred_id = {}   # inferred emp id -> inferred id of TRUE manager (or None)
    stats = defaultdict(lambda: defaultdict(int))
    for tid, tmid, ck in zip(
        truth_df["Employee ID"], truth_df["Manager ID"], truth_df["Company Key"]
    ):
        emp_inf = truth_to_inferred.get(tid)
        skey = record_company.get(emp_inf, ck) if emp_inf is not None else ck
        stats[skey]["people"] += 1
        if emp_inf is None:
            stats[skey]["unmatched_people"] += 1
            continue
        mgr_inf = truth_to_inferred.get(tmid) if tmid else None
        truth_manager_inferred_id[emp_inf] = mgr_inf
        inf_mgr = inferred_edge.get(emp_inf)
        if mgr_inf is not None:
            stats[skey]["truth_edges"] += 1
            if inf_mgr is None:
                stats[skey]["missed"] += 1
        if inf_mgr is not None:
            stats[skey]["inferred_edges"] += 1
            if inf_mgr == mgr_inf:
                stats[skey]["correct"] += 1

    return {
        "truth_to_inferred": truth_to_inferred,
        "inferred_edge": inferred_edge,
        "truth_manager_inferred_id": truth_manager_inferred_id,
        "stats": {k: dict(v) for k, v in stats.items()},
    }


def create_truth_tree(truth_company_df, marker_lookup=None):
    """Full ground-truth hierarchy for one company (HTML)."""
    lookup = truth_company_df.set_index("Employee ID", drop=False).to_dict("index")
    names = {e: safe_value(r["Employee Name"]).lower() for e, r in lookup.items()}
    children = defaultdict(list)
    for eid, mid in zip(truth_company_df["Employee ID"], truth_company_df["Manager ID"]):
        if mid and mid in lookup and mid != eid:
            children[mid].append(eid)
    for kids in children.values():
        kids.sort(key=names.__getitem__)
    has_parent = {c for kids in children.values() for c in kids}
    roots = sorted(
        (e for e in lookup if e not in has_parent),
        key=lambda e: (-len(children.get(e, ())), names[e]),
    )
    visited, parts = set(), []
    for root in roots:
        if root in visited:
            continue
        parts.append(
            build_employee_tree(
                employee_id=root,
                employee_lookup=lookup,
                children_lookup=children,
                confidence_lookup={},
                unresolved_lookup={},
                visited=visited,
                marker_lookup=marker_lookup,
            )
        )
    return "".join(parts), len(roots)


# ============================================================
# 4. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .org-tree {
        font-size: 13px;
        line-height: 1.35;
        max-height: 560px;
        overflow: auto;
        padding: 8px 10px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 6px;
    }

    .org-tree details,
    .org-tree .emp-leaf {
        margin-left: 16px;
        padding-left: 8px;
        border-left: 1px solid rgba(128, 128, 128, 0.35);
    }

    .org-tree > details,
    .org-tree > .emp-leaf {
        margin-left: 0;
        padding-left: 0;
        border-left: 0;
    }

    .org-tree summary {
        cursor: pointer;
        padding: 2px 4px;
        border-radius: 4px;
        white-space: nowrap;
    }

    .org-tree summary:hover,
    .org-tree .emp-leaf:hover {
        background: rgba(128, 128, 128, 0.14);
    }

    .org-tree .emp-leaf {
        padding-top: 2px;
        padding-bottom: 2px;
        white-space: nowrap;
    }

    .emp-name {
        font-weight: 600;
    }

    .emp-title {
        opacity: 0.65;
        margin-left: 6px;
    }

    .conf, .tag, .reports {
        display: inline-block;
        margin-left: 7px;
        padding: 0 6px;
        border-radius: 8px;
        border: 1px solid currentColor;
        font-size: 10.5px;
        line-height: 1.5;
        vertical-align: 1px;
    }

    .conf.high { color: #1f9d63; }
    .conf.medium { color: #c58a1a; }
    .conf.low { color: #d1453b; }
    .tag { color: #6366f1; }

    .mk {
        display: inline-block;
        margin-left: 7px;
        font-size: 12px;
        font-weight: 700;
    }

    .mk.ok { color: #1f9d63; }
    .mk.bad { color: #d1453b; }
    .mk.miss { color: #8a8f8c; }

    .reports {
        border-color: transparent;
        background: rgba(128, 128, 128, 0.22);
    }

    .tree-warning {
        margin: 8px;
        padding: 8px;
        background: #fee2e2;
        color: #991b1b;
        border-radius: 6px;
    }

    .unmatched-section {
        margin-top: 1.5rem;
        padding: 1rem;
        border: 1px solid #fbbf24;
        border-radius: 8px;
        background: #fffbeb;
    }

    .unmatched-section h3 {
        color: #92400e;
        margin-top: 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 5. SIDEBAR: INPUT AND CONFIGURATION
# ============================================================

st.sidebar.header("⚙️ Inference Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload employee CSV",
    type=["csv"],
    help=(
        "Upload a CSV containing Company Name, Industry, Designation, "
        "Seniority Level, Department, Team/Business Unit, Employee Name, "
        "Skills/Specialization, City, Country and Email."
    ),
)

st.sidebar.markdown("**Ground truth (optional)**")
truth_file = st.sidebar.file_uploader(
    "Upload ground-truth CSV",
    type=["csv"],
    key="truth_uploader",
    help=(
        "Columns: Employee ID, Manager ID, Company Name, Designation, "
        "Department, Employee Name, City, Country, Email. Used ONLY to "
        "compare against the inferred chart - it never influences inference."
    ),
)
st.sidebar.caption("Comparison only. Not used by the inference engine.")

st.sidebar.divider()

st.sidebar.subheader("Inference Parameters")

min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    min_value=0,
    max_value=100,
    value=int(engine.MIN_CONFIDENCE),
    step=1,
    help=(
        "A candidate manager must meet this minimum score to be "
        "accepted. Higher values produce fewer but safer edges."
    ),
)

min_score_margin = st.sidebar.slider(
    "Minimum Score Margin",
    min_value=0,
    max_value=100,
    value=int(engine.MIN_SCORE_MARGIN),
    step=1,
    help=(
        "Managers scoring within this many points of the best candidate "
        "are treated as a tie. Ties are left unresolved unless shared-team "
        "assignment is enabled below."
    ),
)

st.sidebar.divider()

st.sidebar.subheader("Hierarchy Rules")

st.sidebar.caption(
    "Seniority comes from the **Seniority Level** column "
    "(10 = CEO, 9 = department head, 8 = manager; 6 / 5 / 3 / 1 = "
    "individual contributors). Only levels "
    f"**{engine.MANAGER_MIN_LEVEL} and above** can have direct reports, so "
    "Lead, Senior, Mid and Junior employees never have anyone under them. "
    "If the column is missing, levels are derived from job titles."
)

allow_shared_teams = st.sidebar.checkbox(
    "Guess between managers who share a team",
    value=bool(engine.ALLOW_SHARED_TEAM_ASSIGNMENT),
    help=(
        "When two managers lead the same team they cannot be told apart. "
        "Off (recommended): those employees stay unresolved. "
        "On: employees are split evenly between the managers, which is a "
        "guess and is often wrong."
    ),
)

st.sidebar.divider()

run_button = st.sidebar.button(
    "🚀 Run Inference",
    type="primary",
    use_container_width=True,
    disabled=uploaded_file is None,
)


# ============================================================
# 6. RUN INFERENCE
# ============================================================

if run_button:
    try:
        with st.spinner(
            "Analyzing employees and inferring reporting relationships..."
        ):
            (
                employees_df,
                edges_df,
                unresolved_df,
                csv_downloads,
            ) = run_inference(
                file_bytes=uploaded_file.getvalue(),
                min_confidence=min_confidence,
                min_margin=min_score_margin,
                allow_shared_teams=allow_shared_teams,
            )

        st.session_state["employees_df"] = employees_df
        st.session_state["edges_df"] = edges_df
        st.session_state["unresolved_df"] = unresolved_df
        st.session_state["csv_downloads"] = csv_downloads
        st.session_state["company_options"] = build_company_options(employees_df)
        st.session_state["run_config"] = {
            "min_confidence": min_confidence,
            "min_score_margin": min_score_margin,
            "allow_shared_teams": allow_shared_teams,
        }
        st.session_state["source_filename"] = uploaded_file.name

        st.success("Inference completed successfully!")

    except Exception as error:
        st.error(f"Inference failed: {error}")


# ============================================================
# 7. CHECK WHETHER RESULTS EXIST
# ============================================================

if "employees_df" not in st.session_state:
    st.info(
        "Upload your employee CSV, adjust the inference parameters "
        "if required, and click **Run Inference**."
    )
    st.markdown("### Expected CSV columns")
    st.code(
        "Company Name, Industry, Designation, Seniority Level, Department, "
        "Team/Business Unit, Employee Name, Skills/Specialization, "
        "City, Country, Email",
        language="text",
    )
    st.caption(
        "Only Company Name, Designation, Department, Employee Name, City, "
        "Country and Email are mandatory. Industry, Seniority Level, "
        "Team/Business Unit and Skills/Specialization improve accuracy; "
        "Team/Business Unit matters most."
    )
    st.stop()


employees_df = st.session_state["employees_df"]
edges_df = st.session_state["edges_df"]
unresolved_df = st.session_state["unresolved_df"]
run_config = st.session_state["run_config"]
csv_downloads = st.session_state["csv_downloads"]


# ============================================================
# 8. SUMMARY METRICS
# ============================================================

total_employees = len(employees_df)
total_companies = employees_df["Company Key"].nunique()
total_edges = len(edges_df)
total_unresolved = len(unresolved_df)
average_confidence = (
    edges_df["Confidence"].mean() if not edges_df.empty else 0
)

st.markdown("## Inference Summary")

metric1, metric2, metric3, metric4, metric5 = st.columns(5)

metric1.metric("Employees", f"{total_employees:,}")
metric2.metric("Companies", f"{total_companies:,}")
metric3.metric("Assigned Relationships", f"{total_edges:,}")
metric4.metric("Unresolved Employees", f"{total_unresolved:,}")
metric5.metric(
    "Average Confidence",
    f"{average_confidence:.1f}%" if total_edges else "N/A",
)

st.caption(
    f"Last run: Minimum confidence = "
    f"{run_config['min_confidence']} | "
    f"Minimum score margin = "
    f"{run_config['min_score_margin']} | "
    f"Shared-team guessing = "
    f"{'on' if run_config.get('allow_shared_teams') else 'off'} | "
    f"Source file = {st.session_state.get('source_filename', 'N/A')}"
)

low_level_managers = (
    int((edges_df["Manager Level"] < engine.MANAGER_MIN_LEVEL).sum())
    if not edges_df.empty else 0
)
if low_level_managers == 0:
    st.success(
        f"Hierarchy rule check passed: no one below level "
        f"{engine.MANAGER_MIN_LEVEL} has direct reports."
    )
else:
    st.error(
        f"{low_level_managers:,} reporting lines have a manager below "
        f"level {engine.MANAGER_MIN_LEVEL}. Please review."
    )

st.info(
    "Confidence is a heuristic score based on seniority level, department, "
    "team, skills and location evidence. It is not a calibrated "
    "probability. **A wrong edge is worse than no edge** – ambiguous "
    "cases are left unresolved, with a best guess shown for review only."
)


# ============================================================
# 9. COMPANY SELECTION
# ============================================================

st.divider()
st.markdown("## Explore Organization")

company_options = st.session_state["company_options"]

if not company_options:
    st.warning("No companies were found in the uploaded dataset.")
    st.stop()

selected_company_key = st.selectbox(
    "Select Company",
    options=list(company_options.keys()),
    format_func=company_options.get,
    index=0,
)
selected_company = company_options[selected_company_key]

company_employees = employees_df[
    employees_df["Company Key"] == selected_company_key
].copy()

company_record_ids = set(company_employees["Record ID"])

company_edges = edges_df[
    edges_df["Employee Record ID"].isin(company_record_ids)
    & edges_df["Manager Record ID"].isin(company_record_ids)
].copy()

company_unresolved = unresolved_df[
    unresolved_df["Record ID"].isin(company_record_ids)
].copy()


# ============================================================
# 10. COMPANY METRICS
# ============================================================

company_metric1, company_metric2, company_metric3 = st.columns(3)

company_metric1.metric("Employees", f"{len(company_employees):,}")
company_metric2.metric("Reporting Relationships", f"{len(company_edges):,}")
company_metric3.metric("Unresolved", f"{len(company_unresolved):,}")


# ============================================================
# 11. ORGANIZATION TREE – CEO-connected hierarchy
#     (optionally side by side with the ground truth)
# ============================================================

# ---- optional ground-truth comparison (read-only) ----
truth_df = None
comparison = None
if truth_file is not None:
    try:
        truth_df = load_truth_data(truth_file.getvalue())
        comparison = build_comparison(truth_df, employees_df, edges_df)
    except Exception as error:
        st.sidebar.error(f"Ground truth could not be loaded: {error}")
        truth_df = None
        comparison = None

company_key = selected_company_key          # key used for inferred statistics
truth_company = None
if truth_df is not None and not company_employees.empty:
    base_key = engine.clean_text(company_employees["Company Name"].iloc[0])
    truth_company = truth_df[truth_df["Company Key"] == base_key].copy()
    if " | " in selected_company_key and comparison is not None:
        # Same name, different industries: keep only the people that matched
        # an employee of THIS company.
        t2i_all = comparison["truth_to_inferred"]
        truth_company = truth_company[
            truth_company["Employee ID"].map(
                lambda tid: t2i_all.get(tid) in company_record_ids
            )
        ]
has_truth = truth_company is not None and not truth_company.empty

inferred_markers = {}
truth_markers = {}
if has_truth:
    t2i = comparison["truth_to_inferred"]
    inf_edge = comparison["inferred_edge"]
    truth_mgr = comparison["truth_manager_inferred_id"]

    for emp_id, mgr_id in zip(
        company_edges["Employee Record ID"], company_edges["Manager Record ID"]
    ):
        if emp_id in truth_mgr:
            inferred_markers[emp_id] = "ok" if truth_mgr[emp_id] == mgr_id else "bad"

    for tid, tmid in zip(truth_company["Employee ID"], truth_company["Manager ID"]):
        emp_inf = t2i.get(tid)
        if emp_inf is None:
            continue
        inf_mgr = inf_edge.get(emp_inf)
        if tmid and truth_mgr.get(emp_inf) is not None:
            if inf_mgr is None:
                truth_markers[tid] = "miss"
            else:
                truth_markers[tid] = "ok" if inf_mgr == truth_mgr[emp_inf] else "bad"
        elif not tmid and inf_mgr is not None:
            truth_markers[tid] = "bad"   # truth says root, we gave a manager

st.markdown("### Organization Chart")

if company_employees.empty:
    st.warning("No employee records found for this company.")
    ceo_html = ""
    unmatched_html = ""
    under_ceo = set()
    unmatched_visited = set()
else:
    (
        ceo_html,
        unmatched_html,
        under_ceo,
        unmatched_visited,
    ) = create_company_trees(
        employees_df=company_employees,
        edges_df=company_edges,
        unresolved_df=company_unresolved,
        marker_lookup=inferred_markers or None,
    )

if truth_df is not None and not has_truth:
    st.warning(
        f"The ground-truth file has no records for \u201c{selected_company}\u201d. "
        "Showing the inferred chart only."
    )

if has_truth:
    # ----- accuracy for this company -----
    cs = comparison["stats"].get(company_key, {})
    correct = cs.get("correct", 0)
    inferred_n = cs.get("inferred_edges", 0)
    truth_n = cs.get("truth_edges", 0)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Precision", f"{correct / inferred_n:.0%}" if inferred_n else "N/A",
              help="Of the inferred reporting lines, how many match the truth.")
    m2.metric("Recall", f"{correct / truth_n:.0%}" if truth_n else "N/A",
              help="Of the true reporting lines, how many were inferred correctly.")
    m3.metric("Correct lines", f"{correct:,}")
    m4.metric("Wrong lines", f"{inferred_n - correct:,}")
    m5.metric("Missed lines", f"{cs.get('missed', 0):,}",
              help="True lines for which no manager was inferred.")
    if cs.get("unmatched_people"):
        st.caption(
            f"{cs['unmatched_people']:,} ground-truth people could not be matched "
            "to the uploaded employee CSV (matched on email, else name + designation)."
        )

    st.caption(
        "\u2713 manager matches the truth \u00b7 \u2717 manager differs \u00b7 "
        "\u2013 truth has a manager but none was inferred. "
        "Hover a name for department and location."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Inferred")
        if ceo_html:
            st.markdown(f'<div class="org-tree">{ceo_html}</div>', unsafe_allow_html=True)
        else:
            st.info("No CEO (or unique CEO) was identified in the inferred data.")
    with right:
        st.markdown("#### Ground truth")
        truth_html, truth_root_count = create_truth_tree(
            truth_company, marker_lookup=truth_markers or None
        )
        st.markdown(f'<div class="org-tree">{truth_html}</div>', unsafe_allow_html=True)
        if truth_root_count > 1:
            st.caption(f"{truth_root_count} top-level people in the ground truth.")

    # ----- wrong lines table -----
    emp_info = company_employees.set_index("Record ID")
    wrong_rows = []
    for emp_id, mgr_id in zip(
        company_edges["Employee Record ID"], company_edges["Manager Record ID"]
    ):
        if emp_id not in truth_mgr or truth_mgr[emp_id] == mgr_id:
            continue
        true_id = truth_mgr[emp_id]
        wrong_rows.append({
            "Employee": emp_info.at[emp_id, "Employee Name"],
            "Employee Designation": emp_info.at[emp_id, "Designation"],
            "Inferred Manager": emp_info.at[mgr_id, "Employee Name"],
            "Inferred Manager Designation": emp_info.at[mgr_id, "Designation"],
            "True Manager": emp_info.at[true_id, "Employee Name"] if true_id in emp_info.index else "(none / top level)",
            "True Manager Designation": emp_info.at[true_id, "Designation"] if true_id in emp_info.index else "",
        })
    with st.expander(f"Wrong reporting lines ({len(wrong_rows):,})"):
        if wrong_rows:
            st.dataframe(pd.DataFrame(wrong_rows), use_container_width=True, hide_index=True)
        else:
            st.success("No inferred line in this company disagrees with the ground truth.")

    # ----- all companies -----
    with st.expander("Accuracy across all companies in the ground truth"):
        tot = defaultdict(int)
        for stat in comparison["stats"].values():
            for k, v in stat.items():
                tot[k] += v
        a1, a2, a3 = st.columns(3)
        a1.metric("Precision", f"{tot['correct'] / tot['inferred_edges']:.1%}" if tot["inferred_edges"] else "N/A")
        a2.metric("Recall", f"{tot['correct'] / tot['truth_edges']:.1%}" if tot["truth_edges"] else "N/A")
        a3.metric("People matched", f"{tot['people'] - tot['unmatched_people']:,} / {tot['people']:,}")
else:
    st.caption(
        "Hierarchy that rolls up to the CEO. "
        "Expand an employee to see their direct reports. "
        "The badge beside a person is the confidence of that person\u2019s own "
        "inferred manager relationship; the grey number is their direct reports."
    )
    if ceo_html:
        st.markdown(f'<div class="org-tree">{ceo_html}</div>', unsafe_allow_html=True)
    elif not company_employees.empty:
        st.info(
            "No CEO (or unique CEO) was identified, so there is no "
            "company-wide hierarchy under a CEO."
        )


# ============================================================
# 12. UNMATCHED RELATIONSHIPS
#     Confident reporting lines that exist but do NOT reach the CEO
# ============================================================

st.divider()
st.markdown("### [Unmatched Relationships]")

st.caption(
    "These are real, confident reporting relationships that form "
    "hierarchies, but the hierarchy does **not** roll up to the CEO. "
    "Typical cases: department trees when no unique CEO was found, "
    "or any other disconnected confident subtree."
)

if unmatched_html:
    st.markdown(f'<div class="org-tree">{unmatched_html}</div>', unsafe_allow_html=True)

    # Also show a flat table of the edges that belong only to these trees
    # (employee is in the unmatched tree and not under the CEO).
    if not company_edges.empty and unmatched_visited:
        unmatched_edge_mask = (
            company_edges["Employee Record ID"].isin(unmatched_visited)
            & ~company_edges["Employee Record ID"].isin(under_ceo)
        )
        unmatched_edges = company_edges[unmatched_edge_mask]
        if not unmatched_edges.empty:
            with st.expander("Flat list of unmatched reporting lines"):
                st.dataframe(
                    unmatched_edges[
                        [
                            "Employee Name",
                            "Employee Designation",
                            "Manager Name",
                            "Manager Designation",
                            "Confidence",
                            "Evidence",
                        ]
                    ].sort_values(by="Confidence", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )
else:
    st.success(
        "Every confident reporting relationship rolls up to the CEO. "
        "There are no unmatched hierarchies."
    )


# ============================================================
# 13. EMPLOYEE TABLE
# ============================================================

st.divider()
st.markdown("### Employee Details")

employee_table = company_employees.copy()

if not company_edges.empty:
    manager_columns = company_edges[
        [
            "Employee Record ID",
            "Manager Name",
            "Manager Designation",
            "Manager Level",
            "Confidence",
            "Evidence",
        ]
    ].rename(columns={"Employee Record ID": "Record ID"})
    employee_table = employee_table.merge(
        manager_columns, on="Record ID", how="left"
    )
else:
    employee_table["Manager Name"] = None
    employee_table["Manager Designation"] = None
    employee_table["Manager Level"] = None
    employee_table["Confidence"] = None
    employee_table["Evidence"] = None

if not company_unresolved.empty:
    unresolved_columns = company_unresolved[["Record ID", "Status", "Reason"]]
    employee_table = employee_table.merge(
        unresolved_columns, on="Record ID", how="left"
    )
else:
    employee_table["Status"] = None
    employee_table["Reason"] = None

display_columns = [
    "Employee Name",
    "Designation",
    "Seniority Level",
    "Department",
    "Team/Business Unit",
    "Skills/Specialization",
    "Manager Name",
    "Manager Designation",
    "Confidence",
    "Status",
    "City",
    "Country",
    "Evidence",
    "Reason",
]

st.dataframe(
    employee_table[
        [c for c in display_columns if c in employee_table.columns]
    ].sort_values(by="Employee Name", ascending=True),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# 14. INFERRED REPORTING LINES
# ============================================================

st.divider()

with st.expander("View Inferred Reporting Lines"):
    if company_edges.empty:
        st.warning("No reporting relationships were inferred for this company.")
    else:
        st.dataframe(
            company_edges[
                [
                    "Employee Name",
                    "Employee Designation",
                    "Employee Level",
                    "Manager Name",
                    "Manager Designation",
                    "Manager Level",
                    "Manager Team",
                    "Confidence",
                    "Runner-up Score",
                    "Score Margin",
                    "Decision",
                    "Evidence",
                ]
            ].sort_values(by="Confidence", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# 15. UNRESOLVED EMPLOYEES (full list)
# ============================================================

with st.expander("View All Unresolved Employees"):
    if company_unresolved.empty:
        st.success("There are no unresolved employees for this company.")
    else:
        st.caption(
            "A best-guess manager is shown for review only. It is NOT "
            "an accepted reporting line and is not drawn in the chart."
        )
        name_by_id = dict(
            zip(company_employees["Record ID"], company_employees["Employee Name"])
        )
        unresolved_view = company_unresolved.copy()
        unresolved_view["Best Guess Manager"] = unresolved_view[
            "Best Guess Manager ID"
        ].map(name_by_id)
        st.dataframe(
            unresolved_view[
                [
                    "Employee Name",
                    "Designation",
                    "Status",
                    "Reason",
                    "Best Guess Manager",
                    "Best Guess Score",
                ]
            ].sort_values(by="Employee Name", ascending=True),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# 16. DOWNLOAD OUTPUT FILES
# ============================================================

st.divider()
st.markdown("## Download Results")

download1, download2, download3 = st.columns(3)

with download1:
    st.download_button(
        label="⬇️ Normalized Employees",
        data=csv_downloads["employees"],
        file_name="normalized_employees.csv",
        mime="text/csv",
        use_container_width=True,
    )

with download2:
    st.download_button(
        label="⬇️ Reporting Lines",
        data=csv_downloads["edges"],
        file_name="inferred_reporting_lines.csv",
        mime="text/csv",
        use_container_width=True,
    )

with download3:
    st.download_button(
        label="⬇️ Unresolved Employees",
        data=csv_downloads["unresolved"],
        file_name="unresolved_employees.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# 17. METHOD AND LIMITATIONS
# ============================================================

with st.expander("Methodology and Limitations"):
    st.markdown(
        """
        **How the inference works**

        1. **Seniority ladder** – Each person has a Seniority Level
           (10 CEO, 9 department head, 8 manager, 6/5/3/1 individual
           contributors). A manager must always be more senior than the
           employee, so reporting loops are impossible.
        2. **Only levels 8+ manage people** – Lead, Senior, Mid and Junior
           employees are never chosen as a manager. This is enforced before
           scoring and re-checked at the end.
        3. **Nearest tier first** – Employees look for a manager at the
           nearest level above them in their own department (employee →
           manager → department head → CEO). A level is skipped only when
           nobody there belongs to the department, and that costs points.
        4. **Team match** – Within a department, the employee's
           Team/Business Unit must match the manager's team. Skills overlap,
           title overlap, city and email domain only break ties.
        5. **Industry** – Companies with the same name but different
           industries are kept apart.
        6. **No guessing** – If a manager cannot be resolved with sufficient
           confidence and margin, the employee stays unresolved. A wrong
           edge is worse than no edge. Two managers leading the same team
           cannot be told apart, so those employees stay unresolved unless
           shared-team guessing is switched on.

        **Ground-truth comparison (optional)**

        The ground-truth file is used only to score the result. Precision is
        the share of inferred lines that are correct; recall is the share of
        true lines that were inferred correctly. It never influences the
        inference.

        **Important limitations**

        - Confidence is a heuristic score, not a probability.
        - Without the Team/Business Unit column, accuracy drops sharply.
        - Root employees and unresolved employees do not have an accepted
          manager relationship, so their confidence is shown as N/A.
        - Evaluate the results against known reporting relationships
          before treating inferred relationships as ground truth.
        """
    )

st.caption(
    "Org Chart Explorer | Heuristic employee-manager inference "
    "(no-guess policy)"
)
