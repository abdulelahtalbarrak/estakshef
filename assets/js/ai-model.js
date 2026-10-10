// محرك استدلال TF-IDF + Naive Bayes — نفس التقنيات المُتعلَّمة في مواد البرنامج
// (تنظيف النص -> TF-IDF -> Multinomial Naive Bayes)، لكن النموذج مُدرَّب مسبقاً
// بـ scikit-learn في Python (انظر train_model.py)، ومُصدَّر كأرقام جاهزة في
// model_data.json. هذا الملف يعيد تطبيق نفس المعادلات الرياضية بجافاسكربت
// خالص، بدون أي مكتبة خارجية وبدون اتصال إنترنت، فيعمل فوراً كموقع ثابت.

let _model = null;

async function loadModel() {
  if (_model) return _model;
  const res = await fetch("assets/js/model_data.json");
  if (!res.ok) throw new Error("تعذر تحميل ملف النموذج model_data.json");
  _model = await res.json();
  return _model;
}

// نفس دالة تنظيف النص المستخدمة في تدريب النموذج (train_model.py)
function cleanTextJS(text) {
  text = text.replace(/[ً-ٰٟۖ-ۭ]/g, ""); // حذف التشكيل
  text = text.replace(/[أإآ]/g, "ا"); // توحيد الألف
  text = text.replace(/[^ء-ي\s]/g, " "); // إبقاء الحروف العربية فقط
  text = text.replace(/\bوال/g, "ال"); // فصل واو العطف عن "ال" التعريف
  text = text.replace(/\s+/g, " ").trim();
  return text;
}

// يحول نصاً إلى متجه TF-IDF متناثر {index: value}، بنفس معادلة TfidfVectorizer
// الافتراضية في scikit-learn: tf خام × idf، ثم تطبيع L2 لكامل المتجه.
function tfidfVectorize(text, model) {
  const cleaned = cleanTextJS(text);
  const tokens = cleaned.split(" ").filter(Boolean);

  const counts = {};
  tokens.forEach((tok) => {
    const idx = model.vocabulary[tok];
    if (idx !== undefined) {
      counts[idx] = (counts[idx] || 0) + 1;
    }
  });

  const vector = {};
  let sumSquares = 0;
  Object.entries(counts).forEach(([idxStr, count]) => {
    const idx = Number(idxStr);
    const val = count * model.idf[idx];
    vector[idx] = val;
    sumSquares += val * val;
  });

  const norm = Math.sqrt(sumSquares);
  if (norm > 0) {
    Object.keys(vector).forEach((idx) => {
      vector[idx] = vector[idx] / norm;
    });
  }
  return vector;
}

// Multinomial Naive Bayes: نفس معادلة sklearn
// joint_log_likelihood[c] = class_log_prior[c] + Σ (x_i × feature_log_prob[c][i])
function predictCategory(vector, model) {
  const entries = Object.entries(vector); // [[idx, val], ...] فقط غير الصفرية
  const scores = model.classes.map((_, c) => {
    let score = model.class_log_prior[c];
    const rowLogProb = model.feature_log_prob[c];
    for (const [idxStr, val] of entries) {
      score += val * rowLogProb[Number(idxStr)];
    }
    return score;
  });

  // softmax لتحويل درجات log-likelihood إلى نسب مئوية مفهومة (يطابق predict_proba)
  const maxScore = Math.max(...scores);
  const expScores = scores.map((s) => Math.exp(s - maxScore));
  const sumExp = expScores.reduce((a, b) => a + b, 0);
  const probs = expScores.map((e) => e / sumExp);

  const order = model.classes
    .map((cat, i) => ({ category: cat, prob: probs[i] }))
    .sort((a, b) => b.prob - a.prob);

  return order; // مرتبة تنازلياً حسب الاحتمال
}

function cosineSimSparse(vecA, vecB) {
  // كلا المتجهين مُطبَّعان (L2-normalized) أصلاً، فحاصل الضرب النقطي = تشابه الجيب التمام
  let dot = 0;
  for (const idx of Object.keys(vecA)) {
    if (vecB[idx] !== undefined) dot += vecA[idx] * vecB[idx];
  }
  return dot;
}

// أقرب شخصية داخل تصنيف معيّن، بالاعتماد على تشابه TF-IDF مع سيرتها الذاتية.
// إن تساوت كل الدرجات عند صفر (لا تقاطع كلمات بين وصف المستخدم وأي سيرة)، نختار عشوائياً.
function bestFigureInCategory(userVector, model, category) {
  const candidates = model.figure_vectors
    .map((fv) => {
      const figure = getFigure(fv.id);
      if (!figure || figure.category !== category) return null;
      const figVector = {};
      fv.idx.forEach((idx, i) => {
        figVector[idx] = fv.val[i];
      });
      return { figure, score: cosineSimSparse(userVector, figVector) };
    })
    .filter(Boolean);

  const maxScore = Math.max(...candidates.map((c) => c.score));
  const top = maxScore > 0 ? candidates.filter((c) => c.score === maxScore) : candidates;
  const picked = top[Math.floor(Math.random() * top.length)];
  return picked;
}

/**
 * الدالة الرئيسية: تحلل نصاً حراً وترجع التصنيف المتوقع (بالاحتمالات) + شخصية مطابقة
 * من داخل ذلك التصنيف + أقرب تصنيفين بديلين كبدائل.
 */
async function matchByText(userText) {
  const model = await loadModel();
  const vector = tfidfVectorize(userText, model);

  const categoryRanking = predictCategory(vector, model); // مرتبة تنازلياً حسب الاحتمال
  const topCategory = categoryRanking[0];

  const best = bestFigureInCategory(vector, model, topCategory.category);

  const alternatives = categoryRanking.slice(1, 4).map((entry) => {
    const alt = bestFigureInCategory(vector, model, entry.category);
    return { figure: alt.figure, score: entry.prob, category: entry.category };
  });

  return {
    category: topCategory.category,
    categoryConfidence: topCategory.prob,
    categoryRanking,
    bestFigure: best.figure,
    bestFigureScore: best.score,
    alternatives,
    modelMeta: model.meta,
  };
}
