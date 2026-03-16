import pandas as pd
import re 
import numpy as np
import ast

pd.read_csv("super_dirty_students.csv")

students = pd.read_csv("super_dirty_students.csv").dropna(
    subset=["name", "score", "gpa"],
    how="all"
)

#Age ustuni
students.loc[students["age"].str.lower() == "twenty", "age"] = 20
students["age"] = students["age"].astype(str).str.extract(r"(\d+)")  # faqat raqamni oladi
students["age"] = students["age"].astype("Int64")  # Int64 dtype, NaN saqlanadi

#Score ustuni
students.loc[~students["score"].str.contains(r"\d", na=False), "score"] = pd.NA
students["score"] = pd.to_numeric(students["score"], errors="coerce")

#Attendance ustuni
students["attendance"] = pd.to_numeric(students["attendance"], errors="coerce")
students.loc[students["attendance"] % 1 != 0, "attendance"] = pd.NA
students.loc[~students["attendance"].between(1, 100), "attendance"] = pd.NA

#GPA ustuni
students.loc[~students["gpa"].str.contains(r"\d", na=False), "gpa"] = pd.NA
students["gpa"] = pd.to_numeric(students["gpa"], errors="coerce")
students.loc[~students["gpa"].between(0, 5), "gpa"] = pd.NA
students.head(20)

#Money_spent ustuni
students["money_spent"] = students["money_spent"].str.replace(",", ".")
students["money_spent"] = students["money_spent"].str.replace(r"[^0-9.]", "", regex=True).astype(float)

#Name ustuni
students["name"] = students["name"].fillna("Unknown")

#Gender ustuni
students["gender"] = students["gender"].str.lower()
students.loc[students["gender"].str.startswith("f", na=False), "gender"] = "Female"
students.loc[students["gender"].str.startswith("m", na=False), "gender"] = "Male"
students["gender"] = students["gender"].fillna("Unknown")
students["gender"] = students["gender"].astype("category")

#Course ustuni
students["course"] = students["course"].str.lower()
students.loc[students["course"].str.startswith("d", na=False), "course"] = "Data Science"
students.loc[students["course"].str.startswith("p", na=False), "course"] = "Python"
students["course"] = students["course"].fillna("Unknown")
students["course"] = students["course"].astype("category")

#Status ustuni
students["status"] = students["status"].astype("category")

#Remarks ustuni
students["remarks"] = students["remarks"].fillna("unknown")
students["remarks"] = students["remarks"].astype("category")

#Date_of_join ustuni
students["date_of_join"] = pd.to_datetime(
    students["date_of_join"], errors="coerce", dayfirst=True
    ).dt.normalize()

#Event_time ustuni
students["event_time"] = pd.to_datetime(
    students["event_time"], errors="coerce", dayfirst=True
).dt.normalize()

#Phone ustuni
students["phone"] = (
    students["phone"]
    .astype(str)
    .str.replace(r"\D", "", regex=True)
    .str[-10:]
    .str.replace(r"(\d{3})(\d{3})(\d{4})", r"\1-\2-\3", regex=True)
    .str.strip()
)
students.loc[students["phone"].str.len() != 12, "phone"] = "NA"

# Email ustuni 
# Regex pattern for valid email
pattern = r"^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$"

# Step 1: Normalize emails
students["email"] = (
    students["email"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"@{2,}", "@", regex=True)   # ko‘p @ → 1 ta
    .str.replace(r"\.{2,}", ".", regex=True)  # ketma-ket nuqtalarni 1 ga kamaytirish
    .str.replace(r"^\.+", "", regex=True)     # string boshidagi nuqtalarni olib tashlash
    .str.replace(r"\.+@", "@", regex=True)    # @ oldidagi nuqtalarni olib tashlash
    .str.replace(r"@\.+", "@", regex=True)    # @ keyingi nuqtalarni olib tashlash
)

# Step 2: Replace invalid emails with "NA" string
students["email"] = students["email"].where(
    students["email"].str.match(pattern),
    "NA"
)
students.loc[students["email"].duplicated(keep=False), "email"] = "NA"

# Yangi ustunlar
students['addr_city'] = 'Tashkent'  # doimiy qiymat
students['addr_district'] = None
students['addr_postal'] = None

for i, row in students.iterrows():
    addr = str(row['address_raw'])

    # Postal code (oxirgi 5-6 raqam)
    postal_match = re.search(r'\b\d{5,6}\b', addr)
    if postal_match:
        students.at[i, 'addr_postal'] = postal_match.group()

    # District (district so'zi bilan)
    district_match = re.search(r'([A-Za-z\s]+district)', addr)
    if district_match:
        students.at[i, 'addr_district'] = district_match.group().strip()

result= students[['addr_city', 'addr_district', 'addr_postal']]

students.head(20)

# Step 6: Address parsing

students['addr_city'] = 'Tashkent'
students['addr_district'] = None
students['addr_postal'] = None

for i, row in students.iterrows():
    addr = str(row['address_raw'])
    addr_clean = re.sub(r'\s*,\s*', ',', addr)
    addr_clean = re.sub(r'\s+', ' ', addr_clean)
    parts = [p.strip() for p in re.split(r',', addr_clean) if p.strip()]

    try:
        t_idx = next(idx for idx, p in enumerate(parts) if 'Tashkent' in p)
    except StopIteration:
        t_idx = None

    district = None
    postal = None

    for p in parts:
        m = re.search(r'\b\d{5,6}\b', p)
        if m:
            postal = m.group()
            break

    if t_idx is not None:
        for p in parts[:t_idx][::-1]:
            if not re.search(r'\b\d{5,6}\b', p):
                district = p
                break
        if district is None and t_idx + 1 < len(parts):
            for p in parts[t_idx + 1:]:
                if not re.search(r'\b\d{5,6}\b', p):
                    district = p
                    break

    students.at[i, 'addr_district'] = district
    students.at[i, 'addr_postal'] = postal

# Step 5: JSON parsing safe

def safe_eval(x):
    if pd.isna(x):
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return {}
    return {}

students['profile_dict'] = students['profile_json'].apply(safe_eval)

# Top-level keys
students['hobbies'] = students['profile_dict'].apply(lambda x: x.get('hobbies', []))
students['skills'] = students['profile_dict'].apply(lambda x: x.get('skills', {}))
students['family'] = students['profile_dict'].apply(lambda x: x.get('family', {}))
students['devices'] = students['profile_dict'].apply(lambda x: x.get('devices', []))

# Skills flatten
def flatten_skills(sk):
    result = {}
    if 'tech' in sk:
        for k, v in sk['tech'].items():
            result[f"skill_tech_{k}"] = v
    if 'soft' in sk:
        result['skill_soft'] = sk['soft']
    return result

skills_flat = students['skills'].apply(flatten_skills).apply(pd.Series)
students = pd.concat([students, skills_flat], axis=1)

# Family flatten
def flatten_family(fam):
    result = {}
    result['siblings'] = fam.get('siblings')
    income = fam.get('income', {})
    result['income_father'] = income.get('father')
    result['income_mother'] = income.get('mother')
    return result

family_flat = students['family'].apply(flatten_family).apply(pd.Series)
students = pd.concat([students, family_flat], axis=1)

# Devices flatten
def flatten_devices(dev_list):
    result = {}
    for i, d in enumerate(dev_list, start=1):
        for k, v in d.items():
            result[f"device{i}_{k}"] = v
    return result

devices_flat = students['devices'].apply(flatten_devices).apply(pd.Series)
students = pd.concat([students, devices_flat], axis=1)

# Keraksiz columnlar
students = students.drop(columns=['profile_dict', 'skills', 'family', 'devices'])

# Hamma ustunlarga string bo‘lsa strip qilish
for col in students.columns:
    if students[col].dtype == object:  # faqat string ustunlar
        students[col] = students[col].str.strip()

# 1️⃣ Original va cleaned row sonini solishtirish
original_rows = len(students)
cleaned_rows = len(students.drop_duplicates())
print(f"Original rows: {original_rows}, Cleaned rows (after drop duplicates): {cleaned_rows}")

# 2️⃣ Missing email va phone sonini tekshirish
missing_email = students['email'].isna().sum() if 'email' in students.columns else "Column not found"
missing_phone = students['phone'].isna().sum() if 'phone' in students.columns else "Column not found"
print(f"Missing emails: {missing_email}, Missing phones: {missing_phone}")

# 3️⃣ Numeric columnlar qiymatlari diapazonini tekshirish
numeric_cols = ['GPA', 'attendance', 'score']
for col in numeric_cols:
    if col in students.columns:
        min_val = students[col].min()
        max_val = students[col].max()
        print(f"{col}: min={min_val}, max={max_val}")
        # misol uchun shartlar, masalan GPA 0-4, attendance 0-100, score 0-100
        if col == 'GPA' and ((min_val < 0) or (max_val > 4)):
            print(f"⚠ {col} values out of range!")
        if col in ['attendance','score'] and ((min_val < 0) or (max_val > 100)):
            print(f"⚠ {col} values out of range!")

# 4️⃣ Duplicate rowlar yo‘qligini tasdiqlash
dup_rows = students.duplicated().sum()
print(f"Duplicate rows: {dup_rows}")

#students.to_csv("super_dirty_students_cleaned.csv", index=False)

students.info()
