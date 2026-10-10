# -*- coding: utf-8 -*-
"""
تدريب نموذج تصنيف نصي حقيقي (TF-IDF + Naive Bayes) يطابق شخصية المستخدم بأحد
التصنيفات العشرة، بنفس التقنيات المستخدمة في البرنامج التأسيسي بالذكاء الاصطناعي
(تنظيف النص -> TF-IDF -> Naive Bayes متعدد الفئات)، ثم تصدير النموذج المدرَّب
كملف JSON يُستخدم للاستدلال مباشرة داخل المتصفح (بدون أي سيرفر أو مكتبة خارجية).
"""
import json
import re
import sys
import importlib.util

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- 1) دالة تنظيف النص (مطابقة لنسخة اليوم الثاني من البرنامج التأسيسي) ---
def clean_text(text):
    text = re.sub(r"[ً-ٰٟۖ-ۭ]", "", text)  # حذف التشكيل
    text = re.sub(r"[أإآ]", "ا", text)  # توحيد الألف
    text = re.sub(r"[^ء-ي\s]", " ", text)  # إبقاء الحروف العربية فقط
    text = re.sub(r"\bوال", "ال", text)  # فصل واو العطف عن "ال" التعريف
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- 2) تحميل بيانات مشروع "استكشف" (الشخصيات + التصنيفات + الصفات) ---
sys.path.insert(0, "/home/claude/culture_site")
figures_mod = load_module("/home/claude/culture_site/figures_data.py", "figures_data")
quiz_mod = load_module("/home/claude/culture_site/quiz_logic.py", "quiz_logic")
traits_mod = load_module("/home/claude/culture_site/traits_data.py", "traits_data")

FIGURES = figures_mod.FIGURES
CATEGORIES = quiz_mod.CATEGORIES
CATEGORY_DESCRIPTIONS = quiz_mod.CATEGORY_DESCRIPTIONS
TRAITS = traits_mod.TRAITS

# --- 3) بناء بيانات التدريب: كل شخصية وكل صفة تصير مثالا تدريبيا موسوما بتصنيفها ---
texts, labels = [], []

for fig in FIGURES:
    texts.append(clean_text(fig["bio"]))
    labels.append(fig["category"])

for t in TRAITS:
    texts.append(clean_text(t["name"]))
    labels.append(t["category"])

for cat, desc in CATEGORY_DESCRIPTIONS.items():
    texts.append(clean_text(desc))
    labels.append(cat)

print(f"إجمالي أمثلة التدريب: {len(texts)}")
from collections import Counter

print("توزيع الأمثلة على التصنيفات:", Counter(labels))

# --- 4) تقسيم تدريب/اختبار (مطابق لأسلوب اليوم الثالث) ---
X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# --- 5) TF-IDF ثم Naive Bayes (نفس تسلسل اليوم الثاني والثالث) ---
# النص معالَج مسبقا (حروف عربية ومسافات فقط)، فنستخدم الفراغ كفاصل توكنز بسيط
vectorizer = TfidfVectorizer(token_pattern=r"\S+")
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nدقة النموذج على بيانات الاختبار: {accuracy:.2%}")
print(classification_report(y_test, y_pred, zero_division=0))

# --- 6) إعادة التدريب على كامل البيانات (تدريب/اختبار معا) للنشر النهائي ---
# (شائع في المشاريع التعليمية الصغيرة: بعد قياس الدقة بأمانة، ندرب على كل البيانات
#  المتاحة قبل النشر حتى يستفيد النموذج المنشور من أكبر قدر من الأمثلة)
vectorizer_final = TfidfVectorizer(token_pattern=r"\S+")
X_all = vectorizer_final.fit_transform(texts)
model_final = MultinomialNB()
model_final.fit(X_all, labels)

vocab = vectorizer_final.vocabulary_  # token -> index
idf = vectorizer_final.idf_.tolist()
classes = model_final.classes_.tolist()
class_log_prior = model_final.class_log_prior_.tolist()
feature_log_prob = model_final.feature_log_prob_.tolist()

print(f"\nحجم المفردات (vocabulary) النهائي: {len(vocab)} كلمة")

# --- 7) تمثيل TF-IDF لكل شخصية (لاختيار أقرب شخصية داخل التصنيف لاحقا) ---
figure_vectors = []
bio_texts_clean = [clean_text(f["bio"]) for f in FIGURES]
X_figures = vectorizer_final.transform(bio_texts_clean)
for i, fig in enumerate(FIGURES):
    row = X_figures.getrow(i)
    indices = row.indices.tolist()
    values = row.data.tolist()
    figure_vectors.append({"id": fig["id"], "idx": indices, "val": values})

model_export = {
    "vocabulary": vocab,
    "idf": idf,
    "classes": classes,
    "class_log_prior": class_log_prior,
    "feature_log_prob": feature_log_prob,
    "figure_vectors": figure_vectors,
    "meta": {
        "algorithm": "TF-IDF + Multinomial Naive Bayes",
        "train_examples": len(texts),
        "test_accuracy": round(accuracy, 4),
        "vocab_size": len(vocab),
    },
}

out_path = "/home/claude/culture_site_static/assets/js/model_data.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(model_export, f, ensure_ascii=False)

import os

print(f"\nتم تصدير النموذج إلى {out_path} (الحجم: {os.path.getsize(out_path) / 1024:.1f} KB)")
