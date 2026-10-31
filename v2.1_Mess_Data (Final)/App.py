"""
Org Chart Explorer - rewritten from scratch.

Run with:   streamlit run App.py

The employee table is the only input.  inference_engine.prepare() cleans it
(one company = one e-mail domain, one canonical industry, duplicates merged,
levels / departments / teams normalised) and inference_engine.infer_all()
works out who reports to whom.
"""

from __future__ import annotations

import html
import io
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st

import inference_engine as engine

BUNDLED_FILE = Path(__file__).with_name("Employees_Database.csv")

st.set_page_config(page_title="Org Chart Explorer", page_icon="🏢", layout="wide")


# ============================================================
# 1. CACHED PIPELINE
# ============================================================

@st.cache_data(show_spinner=False, max_entries=2)
def load_prepared(file_bytes: bytes, use_skills: bool):
    return engine.prepare(io.BytesIO(file_bytes), infer_from_skills=use_skills)


@st.cache_data(show_spinner=False, max_entries=4)
def run_inference(file_bytes: bytes, use_skills: bool, min_confidence: int,
                  min_margin: int, skip_level: bool, guess_ties: bool):
    prepared = load_prepared(file_bytes, use_skills)
    cfg = engine.InferenceConfig(
        min_confidence=min_confidence, min_margin=min_margin,
        allow_skip_level=skip_level, guess_between_ties=guess_ties,
    )
    return engine.infer_all(prepared.employees, cfg)


@st.cache_data(show_spinner=False, max_entries=2)
def run_comparison(file_bytes, use_skills, min_confidence, min_margin, skip_level, guess_ties, truth_bytes):
    prepared = load_prepared(file_bytes, use_skills)
    edges, _ = run_inference(file_bytes, use_skills, min_confidence, min_margin, skip_level, guess_ties)
    truth = engine.load_truth(io.BytesIO(truth_bytes))
    return engine.compare_to_truth(truth, prepared.employees, edges)


# ============================================================
# 2. SMALL HELPERS
# ============================================================

LEVEL_NAMES = {10: "CEO", 9: "Department head", 8: "Manager", 6: "Lead",
               5: "Senior", 3: "Mid-level", 1: "Junior"}
DECISION_LABELS = {
    "head_to_ceo": "Head → CEO",
    "manager_to_head": "Manager → department head",
    "team_manager": "Same team as manager",
    "department_manager": "Only manager in department",
    "skip_to_head": "Skip level (no manager on team)",
    "manager_skip_to_ceo": "Skip level (no department head)",
    "tie_guess": "Guess between equally likely managers",
}


def show_table(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, hide_index=True, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, hide_index=True, use_container_width=True, **kwargs)


def esc(value) -> str:
    return html.escape("" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value))


def conf_class(c: float) -> str:
    return "high" if c >= 85 else "mid" if c >= 70 else "low"


def to_csv(df: pd.DataFrame) -> bytes:
    return df.drop(columns=[c for c in df.columns if c.startswith("_")]).to_csv(index=False).encode("utf-8")


# ============================================================
# 3. ORG CHART (HTML tree)
# ============================================================

MARK = {"correct": ("ok", "\u2713", "Matches the ground truth"),
        "wrong": ("bad", "\u2717", "Differs from the ground truth"),
        "missed": ("miss", "\u2013", "Truth has a manager, none inferred")}


def build_tree_html(root_ids, people, children, edge_info, open_depth: int, markers=None) -> tuple[str, int]:
    """Nested <details> tree. Returns (html, number_of_people_drawn)."""
    parts: list[str] = []
    drawn = 0
    seen: set[str] = set()

    def label(pid: str) -> str:
        p = people[pid]
        e = edge_info.get(pid)
        badges = ""
        if e is not None:
            badges += (f'<span class="badge {conf_class(e["Confidence"])}" '
                       f'title="{esc(e["Evidence"])}">{e["Confidence"]:.0f}%</span>')
            if e["Decision"] in ("skip_to_head", "manager_skip_to_ceo"):
                badges += '<span class="badge skip" title="No manager on the team / no department head">skip level</span>'
            if e["Decision"] == "tie_guess":
                badges += '<span class="badge skip">guess</span>'
        mk = (markers or {}).get(pid)
        if mk in MARK:
            badges += f'<span class="mk {MARK[mk][0]}" title="{MARK[mk][2]}">{MARK[mk][1]}</span>'
        tip = (f'{p["Department"] or "n/a"} · {p["Team/Business Unit"] or "n/a"} · '
               f'level {p["Seniority Level"]} · {p["City"] or "n/a"}, {p["Country"] or "n/a"}')
        return (f'<span class="nm" title="{esc(tip)}">{esc(p["Employee Name"])}</span>'
                f'<span class="ti">{esc(p["Designation"]) or "&mdash;"}</span>'
                f'<span class="dp">{esc(p["Department"])}'
                f'{" · " + esc(p["Team/Business Unit"]) if p["Team/Business Unit"] else ""}</span>{badges}')

    def walk(pid: str, depth: int) -> None:
        nonlocal drawn
        if pid in seen or depth > 40:
            return
        seen.add(pid)
        drawn += 1
        kids = children.get(pid, [])
        if not kids:
            parts.append(f'<div class="leaf">{label(pid)}</div>')
            return
        opened = " open" if depth < open_depth else ""
        parts.append(f'<details{opened}><summary>{label(pid)}'
                     f'<span class="cnt" title="Direct reports">{len(kids)}</span></summary>')
        for kid in kids:
            walk(kid, depth + 1)
        parts.append("</details>")

    for rid in root_ids:
        walk(rid, 0)
    return "".join(parts), drawn


def chain_to_top(pid: str, parent: dict) -> list[str]:
    out, seen = [pid], {pid}
    while pid in parent and parent[pid] not in seen:
        pid = parent[pid]
        out.append(pid)
        seen.add(pid)
    return out


# ============================================================
# 4. STYLES
# ============================================================

st.markdown(
    """
<style>
.org { font-size: 13px; line-height: 1.4; max-height: 640px; overflow: auto;
       padding: 10px 12px; border: 1px solid rgba(128,128,128,.28); border-radius: 8px; }
.org details, .org .leaf { margin-left: 18px; padding-left: 8px;
       border-left: 1px solid rgba(128,128,128,.35); }
.org > details, .org > .leaf, .org > .virtual { margin-left: 0; padding-left: 0; border-left: 0; }
.org summary { cursor: pointer; padding: 2px 4px; border-radius: 4px; white-space: nowrap; }
.org summary:hover, .org .leaf:hover { background: rgba(128,128,128,.14); }
.org .leaf { padding: 2px 4px; white-space: nowrap; }
.nm { font-weight: 600; }
.ti { opacity: .75; margin-left: 8px; }
.dp { opacity: .45; margin-left: 8px; font-size: 11.5px; }
.badge, .cnt { display: inline-block; margin-left: 7px; padding: 0 6px; border-radius: 9px;
       font-size: 10.5px; line-height: 1.55; border: 1px solid currentColor; vertical-align: 1px; }
.badge.high { color: #1f9d63; } .badge.mid { color: #c58a1a; } .badge.low { color: #d1453b; }
.badge.skip { color: #6366f1; }
.mk { margin-left: 7px; font-weight: 700; }
.mk.ok { color: #1f9d63; } .mk.bad { color: #d1453b; } .mk.miss { color: #8a8f8c; }
.cnt { border-color: transparent; background: rgba(128,128,128,.25); }
.virtual { padding: 4px 6px 8px 6px; opacity: .85; font-style: italic; }
.pill { display: inline-block; padding: 1px 10px; border-radius: 12px; font-size: 12px;
        background: rgba(99,102,241,.16); margin-right: 6px; }
.crumb { font-size: 13px; }
.crumb b { font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🏢 Organization Chart Explorer")
st.caption("Reporting relationships inferred from the employee table: company, industry, "
           "department, team, seniority and skills.")


# ============================================================
# 5. SIDEBAR - DATA + PARAMETERS
# ============================================================

st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Employee CSV", type=["csv"])
if uploaded is not None:
    file_bytes, source_name = uploaded.getvalue(), uploaded.name
elif BUNDLED_FILE.exists():
    file_bytes, source_name = BUNDLED_FILE.read_bytes(), BUNDLED_FILE.name
    st.sidebar.caption(f"Using **{BUNDLED_FILE.name}** from the app folder. Upload a file to replace it.")
else:
    file_bytes, source_name = None, ""

truth_file = st.sidebar.file_uploader(
    "Ground truth CSV (optional)", type=["csv"], key="truth",
    help="Needs Employee ID and Manager ID, plus Email or Employee Name (Company Name helps). "
         "Used only to score the result; it never influences the inference.")

st.sidebar.header("Cleaning")
use_skills = st.sidebar.checkbox(
    "Fill blank team / department from skills", value=True,
    help="Skills are learned from the table: if a skill almost always appears in one team, "
         "a person with a blank team and that skill is placed in it. Such lines carry a "
         "confidence penalty.")

st.sidebar.header("Inference")
min_conf = st.sidebar.slider(
    "Minimum confidence", 0, 100, engine.InferenceConfig.min_confidence, 1,
    help="Reporting lines scoring below this are left unresolved.")
min_margin = st.sidebar.slider(
    "Minimum margin between candidates", 0, 30, engine.InferenceConfig.min_margin, 1,
    help="When several managers fit, the best must beat the runner-up by this many points; "
         "otherwise the person is marked ambiguous.")
skip_level = st.sidebar.checkbox(
    "Allow skip-level lines", value=True,
    help="If a team has no manager in the data, attach the person to their department head "
         "(and a manager with no department head to the CEO). These lines are tagged.")
guess_ties = st.sidebar.checkbox(
    "Guess between equally likely managers", value=False,
    help="Off (recommended): ties stay unresolved. On: people are spread across the tied "
         "managers, which is a guess.")

if file_bytes is None:
    st.info("Upload the employee CSV in the sidebar to begin.")
    st.stop()


# ============================================================
# 6. RUN
# ============================================================

try:
    with st.spinner("Cleaning the employee table (first run only)…"):
        prepared = load_prepared(file_bytes, use_skills)
    with st.spinner("Inferring reporting lines…"):
        edges_df, unresolved_df = run_inference(
            file_bytes, use_skills, min_conf, min_margin, skip_level, guess_ties)
except Exception as error:  # noqa: BLE001
    st.error(f"Could not process the file: {error}")
    st.stop()

comparison = None
if truth_file is not None:
    try:
        with st.spinner("Comparing with the ground truth…"):
            comparison = run_comparison(file_bytes, use_skills, min_conf, min_margin, skip_level,
                                        guess_ties, truth_file.getvalue())
    except Exception as error:  # noqa: BLE001
        st.sidebar.error(f"Ground truth could not be used: {error}")

employees = prepared.employees
companies = prepared.companies
report = prepared.report

# ============================================================
# 7. SUMMARY
# ============================================================

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("People", f"{len(employees):,}")
c2.metric("Companies", f"{len(companies):,}")
c3.metric("Industries", f"{report['industries']:,}")
c4.metric("Reporting lines", f"{len(edges_df):,}")
needs_review = int(unresolved_df["Status"].isin(["ambiguous", "unresolved"]).sum())
c5.metric("Need review", f"{needs_review:,}")
c6.metric("Avg confidence", f"{edges_df['Confidence'].mean():.0f}%" if len(edges_df) else "n/a")
st.caption(
    f"Source: {source_name} · {report['raw_rows']:,} rows → {len(employees):,} people "
    f"({report['duplicates_merged']:,} duplicate records merged, "
    f"{report['unassigned_rows']:,} rows could not be tied to a company)."
)

# ============================================================
# 8. SELECT COMPANY
# ============================================================

st.divider()
st.subheader("Select company")

f1, f2 = st.columns([1, 2])
industry_choices = ["All industries"] + sorted(i for i in companies["Industry"].unique() if i)
industry_pick = f1.selectbox("Industry", industry_choices)
pool = companies if industry_pick == "All industries" else companies[companies["Industry"] == industry_pick]
if pool.empty:
    st.warning("No companies in this industry.")
    st.stop()
label_to_key = dict(zip(pool["Label"], pool["Company Key"]))
company_label = f2.selectbox(
    f"Company ({len(pool):,})", list(label_to_key), help="Type to search.")
company_key = label_to_key[company_label]
company = companies[companies["Company Key"] == company_key].iloc[0]

st.markdown(
    f'<span class="pill">{esc(company["Industry"] or "Industry unknown")}</span>'
    f'<span class="pill">{esc(company["Domain"])}</span>'
    f'<span class="pill">{int(company["Employees"]):,} people</span>',
    unsafe_allow_html=True,
)

emp = employees[employees["Company Key"] == company_key]
ed = edges_df[edges_df["Company Key"] == company_key]
un = unresolved_df[unresolved_df["Company Key"] == company_key]

people = {r["Record ID"]: r for r in emp.to_dict("records")}
children: dict[str, list[str]] = defaultdict(list)
parent: dict[str, str] = {}
edge_info: dict[str, dict] = {}
for r in ed.to_dict("records"):
    children[r["Manager ID"]].append(r["Employee ID"])
    parent[r["Employee ID"]] = r["Manager ID"]
    edge_info[r["Employee ID"]] = r
for kids in children.values():
    kids.sort(key=lambda i: (-int(people[i]["Seniority Level"]), people[i]["Employee Name"].lower()))
status = dict(zip(un["Record ID"], un["Status"]))

m1, m2, m3, m4 = st.columns(4)
m1.metric("People", f"{len(emp):,}")
m2.metric("Reporting lines", f"{len(ed):,}")
m3.metric("Need review", f"{int(un['Status'].isin(['ambiguous', 'unresolved']).sum()):,}")
m4.metric("Avg confidence", f"{ed['Confidence'].mean():.0f}%" if len(ed) else "n/a")

tab_chart, tab_truth, tab_people, tab_lines, tab_review, tab_clean, tab_dl = st.tabs(
    ["Org chart", "Ground truth", "People", "Reporting lines", "Needs review", "Data cleaning", "Downloads"])

# outcome markers for this company (only when a ground truth was supplied)
inferred_markers: dict[str, str] = {}
truth_markers: dict[str, str] = {}
company_cmp = None
if comparison is not None:
    cmp_rows = comparison["rows"]
    company_cmp = cmp_rows[cmp_rows["Company Key"] == company_key]
    for rid, outcome in zip(company_cmp["Record ID"], company_cmp["Outcome"]):
        if outcome in ("correct", "wrong", "missed"):
            truth_markers[rid] = outcome
            if outcome != "missed":
                inferred_markers[rid] = outcome
        elif outcome.startswith("false line"):
            inferred_markers[rid] = "wrong"
            truth_markers[rid] = "wrong"

# ------------------------------------------------------------
# ORG CHART
# ------------------------------------------------------------
with tab_chart:
    ceo_ids = [i for i, s in status.items() if s == "root"]
    dept_roots = [i for i, s in status.items() if s == "department_root" and children.get(i)]
    top_ids = ceo_ids or sorted(dept_roots, key=lambda i: people[i]["Department"])
    depth = st.slider("Levels expanded", 1, 5, 2, key="depth")

    if not top_ids:
        st.info("No hierarchy could be built for this company.")
    else:
        html_tree, drawn = build_tree_html(top_ids, people, children, edge_info, depth, inferred_markers or None)
        prefix = ""
        if not ceo_ids:
            prefix = ('<div class="virtual">No CEO record in the data. '
                      'Each department head below is the top of their department.</div>')
        st.markdown(f'<div class="org">{prefix}{html_tree}</div>', unsafe_allow_html=True)
        st.caption(f"{drawn:,} of {len(emp):,} people are in the chart. "
                   "Hover a name for department, team and location; hover a % for the evidence. "
                   "Grey number = direct reports.")

    # groups not connected to the top
    in_chart = set()
    stack = list(top_ids)
    while stack:
        cur = stack.pop()
        if cur not in in_chart:
            in_chart.add(cur)
            stack.extend(children.get(cur, []))
    detached = [i for i in children if i not in in_chart and i not in parent]
    if detached:
        with st.expander(f"Teams not connected to the top ({len(detached)})"):
            st.caption("These managers have a team but no confident line upwards.")
            html_tree2, _ = build_tree_html(detached, people, children, edge_info, 1)
            st.markdown(f'<div class="org">{html_tree2}</div>', unsafe_allow_html=True)

    st.markdown("#### Find an employee")
    options = sorted(people, key=lambda i: people[i]["Employee Name"].lower())
    pick = st.selectbox(
        "Employee", options, index=None, placeholder="Type a name…",
        format_func=lambda i: f'{people[i]["Employee Name"] or "(no name)"} — {people[i]["Designation"] or "no designation"}')
    if pick:
        p = people[pick]
        chain = chain_to_top(pick, parent)
        st.markdown(
            '<div class="crumb"><b>Reporting chain:</b> ' +
            " &rarr; ".join(esc(people[i]["Employee Name"]) for i in chain) + "</div>",
            unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            st.markdown(
                f"**{esc(p['Employee Name'])}** · {esc(p['Designation']) or 'no designation'}  \n"
                f"Level {p['Seniority Level']} ({LEVEL_NAMES.get(int(p['Seniority Level']), '')}, "
                f"from {p['Level Source']})  \n"
                f"Department: {p['Department'] or 'n/a'} · Team: {p['Team/Business Unit'] or 'n/a'}  \n"
                f"{p['City'] or 'n/a'}, {p['Country'] or 'n/a'} · {p['Email'] or 'no email'}  \n"
                f"Source row(s) in the file: {p['Source Rows']}")
            e = edge_info.get(pick)
            if e:
                st.markdown(
                    f"**Manager:** {esc(e['Manager Name'])} ({esc(e['Manager Designation'])}) · "
                    f"confidence {e['Confidence']:.0f}%  \n"
                    f"*{DECISION_LABELS.get(e['Decision'], e['Decision'])}.* {esc(e['Evidence'])}.")
            else:
                st.markdown(f"**No manager assigned** - {engine.STATUS_LABELS.get(status.get(pick, ''), 'not resolved')}.")
        with b:
            reports = children.get(pick, [])
            st.markdown(f"**Direct reports ({len(reports)})**")
            if reports:
                show_table(pd.DataFrame([{
                    "Name": people[i]["Employee Name"], "Designation": people[i]["Designation"],
                    "Team": people[i]["Team/Business Unit"],
                    "Confidence": edge_info[i]["Confidence"]} for i in reports]))

# ------------------------------------------------------------
# GROUND TRUTH
# ------------------------------------------------------------
with tab_truth:
    if comparison is None:
        st.info("Upload a ground-truth CSV in the sidebar (Employee ID, Manager ID, plus Email or "
                "Employee Name; Company Name helps) to score the inferred chart. It is used only "
                "for comparison and never influences the inference.")
    else:
        all_rows = comparison["rows"]
        total = engine.score_rows(all_rows)
        pct = lambda v: f"{v:.1%}" if v is not None else "n/a"
        st.markdown("#### Overall accuracy")
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("Precision", pct(total["precision"]), help="Of the inferred lines, how many match the truth.")
        t2.metric("Recall", pct(total["recall"]), help="Of the true lines, how many were inferred correctly.")
        t3.metric("Correct lines", f"{total['correct']:,}")
        t4.metric("Wrong lines", f"{total['wrong']:,}")
        t5.metric("Missed lines", f"{total['missed']:,}", help="Truth has a manager, none was inferred.")
        st.caption(
            f"{comparison['matched_people']:,} of {comparison['truth_people']:,} ground-truth people were "
            f"matched to the cleaned table (email, else company + name). "
            f"{comparison['unmatched_people']:,} could not be matched and "
            f"{comparison['unmatched_manager']:,} had a manager who could not be matched; those are not scored.")

        a, b = st.columns(2)
        with a:
            st.markdown("**Accuracy by type of line**")
            lines = all_rows[all_rows["Decision"] != ""]
            by_dec = []
            for dec, g in lines.groupby("Decision"):
                sc = engine.score_rows(g)
                by_dec.append({"Type": DECISION_LABELS.get(dec, dec), "Lines": len(g),
                               "Correct": sc["correct"], "Precision": sc["precision"]})
            show_table(pd.DataFrame(by_dec).sort_values("Lines", ascending=False))
        with b:
            st.markdown("**Accuracy by confidence band**")
            bands = pd.cut(lines["Confidence"], [0, 70, 80, 90, 101], right=False,
                           labels=["< 70", "70-79", "80-89", "90+"])
            by_band = []
            for band, g in lines.groupby(bands, observed=True):
                sc = engine.score_rows(g)
                by_band.append({"Confidence": str(band), "Lines": len(g),
                                "Correct": sc["correct"], "Precision": sc["precision"]})
            show_table(pd.DataFrame(by_band))
        st.caption("If precision is high only in the top bands, raise *Minimum confidence*; "
                   "if a line type is weak, switch it off (e.g. skip-level lines) in the sidebar.")

        st.markdown("#### Accuracy by company")
        names_by_key = dict(zip(companies["Company Key"], companies["Label"]))
        per_company = []
        for ck, g in all_rows.groupby("Company Key"):
            sc = engine.score_rows(g)
            per_company.append({"Company": names_by_key.get(ck, ck), "Scored people": len(g),
                                "Precision": sc["precision"], "Recall": sc["recall"],
                                "Wrong": sc["wrong"], "Missed": sc["missed"]})
        if per_company:
            show_table(pd.DataFrame(per_company).sort_values("Precision", na_position="last"))

        st.markdown(f"#### {esc(company_label)}: inferred vs ground truth")
        if company_cmp is None or company_cmp.empty:
            st.warning("The ground truth has no people matched to this company.")
        else:
            cs = engine.score_rows(company_cmp)
            n1, n2, n3, n4 = st.columns(4)
            n1.metric("Precision", pct(cs["precision"]))
            n2.metric("Recall", pct(cs["recall"]))
            n3.metric("Wrong lines", f"{cs['wrong']:,}")
            n4.metric("Missed lines", f"{cs['missed']:,}")
            st.caption("\u2713 matches the truth  \u00b7  \u2717 differs  \u00b7  \u2013 truth has a manager, none inferred.")

            t_children: dict[str, list[str]] = defaultdict(list)
            for rid, tm in comparison["truth_parent"].items():
                if rid in people and tm in people:
                    t_children[tm].append(rid)
            for kids in t_children.values():
                kids.sort(key=lambda i: (-int(people[i]["Seniority Level"]), people[i]["Employee Name"].lower()))
            scored_ids = set(company_cmp["Record ID"])
            t_roots = sorted(
                (i for i in scored_ids if i not in comparison["truth_parent"] or comparison["truth_parent"][i] not in people),
                key=lambda i: (-len(t_children.get(i, [])), people[i]["Employee Name"].lower()))
            t_roots = [i for i in t_roots if t_children.get(i) or i in scored_ids]
            left, right = st.columns(2)
            with left:
                st.markdown("**Inferred**")
                if top_ids:
                    h, _ = build_tree_html(top_ids, people, children, edge_info, 2, inferred_markers or None)
                    st.markdown(f'<div class="org">{h}</div>', unsafe_allow_html=True)
                else:
                    st.info("No hierarchy inferred.")
            with right:
                st.markdown("**Ground truth**")
                h, _ = build_tree_html(t_roots, people, t_children, {}, 2, truth_markers or None)
                st.markdown(f'<div class="org">{h}</div>', unsafe_allow_html=True)

            wrong = company_cmp[company_cmp["Outcome"].isin(["wrong", "missed"]) |
                                company_cmp["Outcome"].str.startswith("false line")]
            with st.expander(f"Wrong and missed lines ({len(wrong):,})", expanded=len(wrong) > 0):
                def nm(i):
                    return people[i]["Employee Name"] if i in people else ""
                def dg(i):
                    return people[i]["Designation"] if i in people else ""
                if wrong.empty:
                    st.success("Every scored line in this company matches the ground truth.")
                else:
                    show_table(pd.DataFrame({
                        "Employee": wrong["Record ID"].map(nm),
                        "Designation": wrong["Record ID"].map(dg),
                        "Inferred manager": wrong["Inferred Manager ID"].map(nm),
                        "Inferred designation": wrong["Inferred Manager ID"].map(dg),
                        "True manager": wrong["Truth Manager ID"].map(nm).replace("", "(top of company)"),
                        "True designation": wrong["Truth Manager ID"].map(dg),
                        "Outcome": wrong["Outcome"],
                        "Type of line": wrong["Decision"].map(lambda d: DECISION_LABELS.get(d, d)),
                        "Confidence": wrong["Confidence"],
                    }))

# ------------------------------------------------------------
# PEOPLE
# ------------------------------------------------------------
with tab_people:
    table = emp.merge(
        ed[["Employee ID", "Manager Name", "Manager Designation", "Confidence", "Decision"]]
        .rename(columns={"Employee ID": "Record ID"}), on="Record ID", how="left")
    table = table.merge(un[["Record ID", "Status"]], on="Record ID", how="left")
    table["Status"] = table["Status"].map(lambda s: engine.STATUS_LABELS.get(s, s) if isinstance(s, str) else "")
    table["Decision"] = table["Decision"].map(lambda s: DECISION_LABELS.get(s, s) if isinstance(s, str) else "")

    q1, q2, q3 = st.columns([2, 1, 1])
    text = q1.text_input("Search name / designation / email")
    depts = q2.multiselect("Department", sorted(d for d in table["Department"].unique() if d))
    lvls = q3.multiselect("Level", sorted(table["Seniority Level"].unique(), reverse=True))
    if text:
        needle = text.lower()
        table = table[table[["Employee Name", "Designation", "Email"]].apply(
            lambda c: c.astype(str).str.lower().str.contains(needle, regex=False)).any(axis=1)]
    if depts:
        table = table[table["Department"].isin(depts)]
    if lvls:
        table = table[table["Seniority Level"].isin(lvls)]
    cols = ["Employee Name", "Designation", "Seniority Level", "Department", "Team/Business Unit",
            "Manager Name", "Manager Designation", "Confidence", "Decision", "Status",
            "Skills/Specialization", "City", "Country", "Email"]
    show_table(table[cols].sort_values("Employee Name"))

# ------------------------------------------------------------
# REPORTING LINES
# ------------------------------------------------------------
with tab_lines:
    if ed.empty:
        st.warning("No reporting lines were inferred for this company.")
    else:
        l1, l2 = st.columns([1, 2])
        floor = l1.slider("Show lines with confidence ≥", 0, 100, 0, key="floor")
        kinds = l2.multiselect("Type of line", sorted(ed["Decision"].unique()),
                               format_func=lambda d: DECISION_LABELS.get(d, d))
        view = ed[ed["Confidence"] >= floor]
        if kinds:
            view = view[view["Decision"].isin(kinds)]
        view = view.assign(Decision=view["Decision"].map(lambda d: DECISION_LABELS.get(d, d)))
        show_table(view[["Employee Name", "Employee Designation", "Employee Team", "Manager Name",
                         "Manager Designation", "Manager Team", "Confidence", "Decision", "Evidence"]]
                   .sort_values("Confidence", ascending=False))

# ------------------------------------------------------------
# NEEDS REVIEW
# ------------------------------------------------------------
with tab_review:
    review = un[~un["Status"].isin(["root"])].copy()
    if review.empty:
        st.success("Everyone in this company has a reporting line.")
    else:
        names = {i: p["Employee Name"] for i, p in people.items()}
        review["Best guess manager"] = review["Best Guess Manager ID"].map(names)
        review["Status"] = review["Status"].map(lambda s: engine.STATUS_LABELS.get(s, s))
        st.caption("A best guess is shown for review only. It is not a reporting line and is not drawn in the chart.")
        show_table(review[["Employee Name", "Designation", "Level", "Department", "Team", "Status",
                           "Reason", "Best guess manager", "Best Guess Score"]].sort_values("Employee Name"))

# ------------------------------------------------------------
# DATA CLEANING
# ------------------------------------------------------------
with tab_clean:
    st.markdown("#### What the engine did to your table")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Industry spellings", f"{report['raw_industry_spellings']:,} → {report['industries']}")
    k2.metric("Company-name spellings", f"{report['raw_company_spellings']:,} → {report['companies']}")
    k3.metric("Duplicates merged", f"{report['duplicates_merged']:,}")
    k4.metric("Rows without a company", f"{report['unassigned_rows']:,}")
    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Level from column", f"{report['levels_from_column']:,}")
    k6.metric("Level from designation", f"{report['levels_learned'] + report['levels_keywords']:,}")
    k7.metric("Team filled from skills", f"{report['team_from_skills']:,}")
    k8.metric("Invalid records", f"{report['invalid_records']:,}")
    st.markdown(
        "- **Company** = one e-mail domain. Rows with a typo in the domain or name are attached "
        "by name + industry + country; if that is not unique the row is left out rather than guessed.\n"
        "- **Industry** is cleaned to one spelling and shown only to tell apart different "
        "companies that share a name.\n"
        "- **Level** comes from *Seniority Level*; when blank it is learned from the designation "
        "(what level people with that title have elsewhere in the table), then from keywords.\n"
        "- **Department / team** are normalised; blanks are filled from the team, designation "
        "or skills, each learned from the table.")
    with st.expander("Industry spellings → cleaned industry"):
        show_table(report["industry_map"])
    with st.expander(f"Rows that could not be tied to a company ({len(prepared.unassigned):,})"):
        show_table(prepared.unassigned)

# ------------------------------------------------------------
# DOWNLOADS
# ------------------------------------------------------------
with tab_dl:
    st.markdown("#### Whole dataset")
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Cleaned employees", to_csv(employees), "cleaned_employees.csv", "text/csv")
    d2.download_button("Reporting lines", to_csv(edges_df), "reporting_lines.csv", "text/csv")
    d3.download_button("Needs review", to_csv(unresolved_df), "unresolved.csv", "text/csv")
    d4.download_button("Company list", to_csv(companies), "companies.csv", "text/csv")
    st.markdown(f"#### {esc(company_label)} only")
    e1, e2 = st.columns(2)
    e1.download_button("Reporting lines", to_csv(ed), "company_reporting_lines.csv", "text/csv", key="dl_ed")
    e2.download_button("People", to_csv(emp), "company_people.csv", "text/csv", key="dl_emp")

with st.expander("How the reporting lines are inferred"):
    st.markdown(
        """
1. **Levels**: 10 = CEO, 9 = department head, 8 = manager, 6/5/3/1 = individual contributors.
   Only level 8 and above can have direct reports.
2. **Heads → CEO.** **Managers → head of their department.** **Everyone else → manager of their team.**
3. If several managers fit, skills, designation words and city break the tie. If they can't be
   separated by at least the margin, the person is marked *ambiguous*, not guessed.
4. If a team has no manager in the data, the person is attached to the department head and tagged
   *skip level* (switch off in the sidebar).
5. **Confidence** is a heuristic score of how strong the evidence is, not a probability.
   A wrong line is worse than no line.
"""
    )
