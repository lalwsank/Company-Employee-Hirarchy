import logging
import sys
from datetime import datetime
import random
import re
import math
import pandas as pd
from faker import Faker

# Generate timestamped log filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"log_{timestamp}.txt"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

# Redirect print() output to the logger
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

def get_input(prompt, default, minimum=None):
    while True:
        try:
            value = int(
                input(f"{prompt} [Default: {default}]: ") or default
            )

            if minimum is None or value >= minimum:
                return value

            print(f"Value must be >= {minimum}.")

        except ValueError:
            print("Please enter a valid integer.")


N = get_input("Total Companies (>= 500)", 500, 500)

MIN_EMPLOYEES = get_input(
    "Minimum Employees (>= 100)",
    100,
    100
)

MAX_EMPLOYEES = get_input(
    "Maximum Employees",
    300,
    MIN_EMPLOYEES
)

SEED = get_input("Random Seed", 42)

random.seed(SEED)

# Separate RNG for the newly added attributes (Industry, Team, Skills).
# This keeps names, cities, designations etc. identical to earlier runs
# with the same seed.
attr_rng = random.Random(SEED)

company_fake = Faker("en_US")
company_fake.seed_instance(SEED)


# ==========================================
# 2. COUNTRY AND FAKER CONFIGURATION
# ==========================================

fake_generators = {
    "USA": Faker("en_US"),
    "India": Faker("en_IN"),
    "UK": Faker("en_GB"),
    "Australia": Faker("en_AU"),
    "Germany": Faker("de_DE"),
    "Canada": Faker("en_CA")
}

for index, fake in enumerate(fake_generators.values()):
    fake.seed_instance(SEED + index)


# ==========================================
# 3. ORGANIZATION STRUCTURE
# ==========================================

organization = {
    "Engineering": {
        "head": "VP Engineering",
        "managers": [ "Engineering Manager", "Software Development Manager", "QA Manager", "Infrastructure Manager"],
        "employees": [ "Technical Lead", "Senior Software Engineer", "Software Engineer", "Junior Software Engineer", "Associate Engineer"]},
        
    "Sales": {
        "head": "VP Sales",
        "managers": ["Sales Operations Manager",  "Senior Sales Manager", "Sales Manager" , "Regional Sales Manager"],
        "employees": ["Account Executive", "Sales Representative", "Business Development Representative", "Client Success Associate", "Sales Operations Associate"]},

    "Human Resources": {
        "head": "HR Director",
        "managers": [ "HR Manager", "Talent Acquisition Manager", "HR Operations Manager", "Compensation and Benefits Manager"],
        "employees": [ "HR Business Partner", "HR Executive", "Recruiter"]},

    "Finance": {
        "head": "Finance Director",
        "managers": [ "Finance Manager", "Accounting Manager", "Tax Manager", "Financial Planning Manager"],
        "employees": [ "Senior Financial Analyst", "Financial Analyst", "Accountant"]},

    "Marketing": {
        "head": "Marketing Director",
        "managers": [ "Marketing Manager", "Digital Marketing Manager", "Brand Manager", "Content Marketing Manager", "PR Manager"],
        "employees": [ "Marketing Analyst", "Marketing Executive", "Content Specialist", "Social Media Intern"]},

    "Operations": {
        "head": "Operations Director",
        "managers": [ "Operations Manager", "Regional Operations Manager", "Process Manager", "Supply Chain Manager"],
        "employees": [ "Operations Analyst", "Operations Executive", "Process Associate"]}
}

departments = list(organization.keys())


# ==========================================
# 3B. INDUSTRY, TEAM AND SKILL REFERENCE DATA
# ==========================================

INDUSTRIES = [
    "Information Technology", "Financial Services", "Healthcare",
    "Manufacturing", "Retail & E-Commerce", "Telecommunications",
    "Energy & Utilities", "Education", "Logistics & Transportation",
    "Media & Entertainment", "Consulting", "Real Estate",
    "Automotive", "Pharmaceuticals", "Hospitality"
]

# Business units / teams available in each department
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

CEO_SKILLS = [
    "Corporate Strategy", "Executive Leadership", "Investor Relations",
    "M&A", "Organizational Design", "Board Governance", "P&L Ownership"
]
HEAD_SKILLS = [
    "Strategic Planning", "Budget Ownership", "Stakeholder Management",
    "Cross-Functional Leadership", "Talent Strategy", "Executive Reporting"
]
MANAGER_SKILLS = [
    "People Management", "Coaching & Mentoring", "Resource Allocation",
    "Performance Reviews", "Project Delivery", "Team Planning"
]

# Seniority is derived from the designation
HEAD_DESIGNATIONS = {cfg["head"] for cfg in organization.values()}
MANAGER_DESIGNATIONS = {m for cfg in organization.values() for m in cfg["managers"]}
EMPLOYEE_DESIGNATIONS = {e for cfg in organization.values() for e in cfg["employees"]}


def get_seniority(designation):
    if designation == "CEO":
        return "10"
    if designation in HEAD_DESIGNATIONS:
        return "9"
    if designation in MANAGER_DESIGNATIONS:
        return "8"

    lowered = designation.lower()
    if "lead" in lowered:
        return "6"
    if "senior" in lowered:
        return "5"
    if any(word in lowered for word in ("junior", "associate", "intern")):
        return "1"
    return "3"


def get_skills(designation, department, team):
    """Pick 2-3 skills relevant to the person's role and team."""
    if designation == "CEO":
        return "; ".join(attr_rng.sample(CEO_SKILLS, 3))

    if designation in HEAD_DESIGNATIONS:
        return "; ".join(attr_rng.sample(HEAD_SKILLS, 3))

    team_skills = TEAMS[department][team]

    if designation in MANAGER_DESIGNATIONS:
        chosen = (
            attr_rng.sample(MANAGER_SKILLS, 1)
            + attr_rng.sample(team_skills, 2)
        )
        return "; ".join(chosen)

    return "; ".join(
        attr_rng.sample(team_skills, attr_rng.randint(2, 3))
    )


# ==========================================
# 4. GENERATE UNIQUE COMPANY NAMES
# ==========================================

company_names = set()

while len(company_names) < N:
    company_names.add(company_fake.company())

company_names = sorted(company_names)


# ==========================================
# 5. EMPLOYEE GENERATION FUNCTION
# ==========================================

records = []


def create_employee(
    company_id,
    company_name,
    employee_number,
    manager_id,
    designation,
    department,
    fake,
    country,
    city,
    company_domain,
    industry,
    team
):

    # Unique Employee ID
    employee_id = (
        f"C{company_id:04d}-E{employee_number:05d}"
    )

    # Generate employee name
    employee_name = fake.name()

    name_parts = employee_name.lower().split()

    first_name = re.sub(
        r"[^a-z0-9]", "", name_parts[0]
    )

    last_name = re.sub(
        r"[^a-z0-9]", "", name_parts[-1]
    )

    # Generate unique email
    email = (
        f"{first_name}.{last_name}."
        f"{employee_number}@{company_domain}"
    )

    # New attributes
    seniority_level = get_seniority(designation)

    skills = get_skills(designation, department, team)

    return {
        "Employee ID": employee_id,
        "Manager ID": manager_id,
        "Company Name": company_name,
        "Industry": industry,
        "Designation": designation,
        "Seniority Level": seniority_level,
        "Department": department,
        "Team/Business Unit": team,
        "Employee Name": employee_name,
        "Skills/Specialization": skills,
        "City": city,
        "Country": country,
        "Email": email
    }


# ==========================================
# 6. GENERATE COMPANY EMPLOYEES
# ==========================================

for company_index, company in enumerate(company_names, 1):

    print(f"Generating Company {company_index}/{N}: {company}")

    # Random employee count
    employee_count = random.randint(
        MIN_EMPLOYEES,
        MAX_EMPLOYEES
    )

    # Assign country
    country = random.choice(
        list(fake_generators.keys())
    )

    fake = fake_generators[country]

    # Generate 10 cities for the company
    cities = sorted({
        fake.city()
        for _ in range(10)
    })

    # Assign one industry per company
    industry = attr_rng.choice(INDUSTRIES)

    # Synthetic company email domain
    company_domain = (
        f"company{company_index:04d}.example"
    )

    employee_number = 1

    # --------------------------------------
    # CEO
    # --------------------------------------

    ceo = create_employee(
        company_index,
        company,
        employee_number,
        None,
        "CEO",
        "Executive",
        fake,
        country,
        random.choice(cities),
        company_domain,
        industry,
        EXECUTIVE_TEAM
    )

    records.append(ceo)

    ceo_id = ceo["Employee ID"]

    employee_number += 1

    # --------------------------------------
    # ALLOCATE EMPLOYEES ACROSS DEPARTMENTS
    # --------------------------------------

    remaining = employee_count - 1

    # Each department gets at least 8 employees
    department_sizes = [8] * len(departments)

    # Distribute remaining employees randomly
    for _ in range(remaining - sum(department_sizes)):

        department_sizes[
            random.randrange(len(departments))
        ] += 1

    # --------------------------------------
    # GENERATE DEPARTMENT HIERARCHY
    # --------------------------------------

    for department, dept_size in zip(
        departments,
        department_sizes
    ):

        config = organization[department]

        # ----------------------------------
        # DEPARTMENT HEAD
        # ----------------------------------

        head = create_employee(
            company_index,
            company,
            employee_number,
            ceo_id,
            config["head"],
            department,
            fake,
            country,
            random.choice(cities),
            company_domain,
            industry,
            LEADERSHIP_TEAM
        )

        records.append(head)

        head_id = head["Employee ID"]

        employee_number += 1

        # ----------------------------------
        # GENERATE MULTIPLE MANAGERS
        # ----------------------------------

        manager_count = max(
            1,
            math.ceil((dept_size - 1) / 10)
        )

        manager_ids = []
        manager_teams = []

        # Each manager leads one team (teams are reused if managers > teams)
        department_teams = attr_rng.sample(
            list(TEAMS[department].keys()),
            len(TEAMS[department])
        )

        for manager_index in range(manager_count):

            team = department_teams[
                manager_index % len(department_teams)
            ]

            # Randomly select manager designation
            manager_designation = random.choice(
                config["managers"]
            )

            manager = create_employee(
                company_index,
                company,
                employee_number,
                head_id,
                manager_designation,
                department,
                fake,
                country,
                random.choice(cities),
                company_domain,
                industry,
                team
            )

            records.append(manager)

            manager_ids.append(
                manager["Employee ID"]
            )

            manager_teams.append(team)

            employee_number += 1

        # ----------------------------------
        # GENERATE INDIVIDUAL CONTRIBUTORS
        # ----------------------------------

        individual_count = (
            dept_size - 1 - manager_count
        )

        for i in range(individual_count):

            # Distribute employees among managers
            manager_id = manager_ids[
                i % manager_count
            ]

            # Employees belong to the same team as their manager
            team = manager_teams[
                i % manager_count
            ]

            designation = random.choice(
                config["employees"]
            )

            employee = create_employee(
                company_index,
                company,
                employee_number,
                manager_id,
                designation,
                department,
                fake,
                country,
                random.choice(cities),
                company_domain,
                industry,
                team
            )

            records.append(employee)

            employee_number += 1


# ==========================================
# 7. CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(records)

# Arrange columns
df = df[
    [
        "Employee ID",
        "Manager ID",
        "Company Name",
        "Industry",
        "Designation",
        "Seniority Level",
        "Department",
        "Team/Business Unit",
        "Employee Name",
        "Skills/Specialization",
        "City",
        "Country",
        "Email"
    ]
]


# ==========================================
# 8. VALIDATE DATA
# ==========================================

print("\nRunning validation checks...")

company_counts = df.groupby("Company Name").size()

# 1. Unique Employee IDs
assert df["Employee ID"].is_unique, ("Duplicate Employee IDs found!")
# 2. Unique Emails
assert df["Email"].is_unique, ("Duplicate Email addresses found!")
# 3. Minimum employee count
assert company_counts.min() >= MIN_EMPLOYEES, ("Some companies have fewer employees than required!")
# 4. Maximum employee count
assert company_counts.max() <= MAX_EMPLOYEES, ("Some companies exceed the maximum employee count!")
# 5. All Manager IDs must exist
employee_ids = set(df["Employee ID"])
manager_ids = set(df["Manager ID"].dropna())
assert manager_ids.issubset(employee_ids), ("Invalid Manager IDs found!")
# 6. Every company must have exactly one CEO
ceo_counts = (df[df["Designation"] == "CEO"].groupby("Company Name").size())
assert len(ceo_counts) == N
assert (ceo_counts == 1).all(), ("Each company must have exactly one CEO!")
# 7. CEO must not have a manager
assert df.loc[df["Designation"] == "CEO", "Manager ID"].isna().all(), ("CEO should not have a Manager ID!")
# 8. Every non-CEO must have a manager
assert df.loc[df["Designation"] != "CEO", "Manager ID"].notna().all(), ("Some employees do not have a Manager ID!")
# 9. Manager and employee must belong to same company
manager_company = df.set_index("Employee ID")["Company Name"]
for _, row in df[df["Manager ID"].notna()].iterrows():
     assert (
    manager_company[row["Manager ID"]] == row["Company Name"]
    ), (
    f"Cross-company manager relationship found: "
    f"{row['Employee ID']}"
    )
# 10. No employee can report to themselves
assert (
    df.loc[df["Manager ID"].notna(),"Employee ID"].values != df.loc[df["Manager ID"].notna(), "Manager ID"].values).all(), (
    "Self-reporting relationship found!"
)

# 11. New columns must be fully populated
new_columns = [
    "Industry", "Seniority Level",
    "Team/Business Unit", "Skills/Specialization"
]
for col in new_columns:
    assert df[col].notna().all() and (df[col].astype(str).str.strip() != "").all(), (
        f"Missing values found in '{col}'!"
    )
# 12. Each company must belong to exactly one industry
assert (df.groupby("Company Name")["Industry"].nunique() == 1).all(), (
    "A company has more than one industry!"
)
# 13. Individual contributors must share their manager's team
ic_rows = df[df["Designation"].isin(EMPLOYEE_DESIGNATIONS)]
manager_team = df.set_index("Employee ID")["Team/Business Unit"]
assert (
    ic_rows["Manager ID"].map(manager_team) == ic_rows["Team/Business Unit"]
).all(), "Employee team does not match manager team!"

print("All validation checks passed!")

# ==========================================
# 9. EXPORT CSV
# ==========================================
output_file1 = "Synthetic_Company_Employees.csv"
df.to_csv(output_file1, index=False)

# ==========================================
# 10. CREATING JUMBLED DATASET
# ==========================================
columns = [
    "Company Name", "Industry", "Designation", "Seniority Level",
    "Department", "Team/Business Unit", "Employee Name",
    "Skills/Specialization", "City", "Country", "Email"
]
jumbled_df = df[columns].copy()

# Shuffle rows while preserving all employee records
jumbled_df = jumbled_df.sample(
    frac=1,
    random_state=None
).reset_index(drop=True)

output_file2 = "Employees_Database.csv"
jumbled_df.to_csv(output_file2, index=False)

# ==========================================
# 11. SUMMARY
# ==========================================
print("\n" + "=" * 50)
print("DATASET GENERATED SUCCESSFULLY!")
print("=" * 50)
print(f"Total Companies       : {df['Company Name'].nunique():,}")
print(f"Total Employees       : {len(df):,}")
print(f"Minimum Employees     : {company_counts.min():,}")
print(f"Maximum Employees     : {company_counts.max():,}")
print(f"Average Employees     : {company_counts.mean():.2f}")
print(f"Total Departments     : {df['Department'].nunique():,}")
print(f"Unique Designations   : {df['Designation'].nunique():,}")
print(f"Unique Countries      : {df['Country'].nunique():,}")
print(f"Unique Industries     : {df['Industry'].nunique():,}")
print(f"Unique Teams          : {df['Team/Business Unit'].nunique():,}")
print("\nDepartment Distribution:")
print(df["Department"].value_counts())
print("\nDesignation Distribution:")
print(df["Designation"].value_counts())
print("\nSeniority Distribution:")
print(df["Seniority Level"].value_counts())
print("\nIndustry Distribution (by company):")
print(df.drop_duplicates("Company Name")["Industry"].value_counts())
print("\nSample Records:")
print(df.head(15).to_string(index=False))
print(f"\nOutput File: {output_file1}")
print("=" * 50)
print(f"\nOutput File: {output_file2}")
print("=" * 50)