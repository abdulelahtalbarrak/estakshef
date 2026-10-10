// منطق المطابقة المشترك (منقول من quiz_logic.py الأصلي + منطق قاعدة البيانات)
// يعتمد على FIGURES / CATEGORIES / CATEGORY_DESCRIPTIONS / QUESTIONS المعرّفة في data.js

function calculateResult(answerIndexes) {
  // answerIndexes: مصفوفة من 7 أرقام (0-3) تمثل الخيار المختار بكل سؤال
  const scores = {};
  CATEGORIES.forEach((c) => (scores[c] = 0));

  QUESTIONS.forEach((question, i) => {
    const idx = answerIndexes[i];
    if (idx === undefined || idx === null) return;
    const option = question.options[idx];
    Object.entries(option.points).forEach(([cat, pts]) => {
      scores[cat] = (scores[cat] || 0) + pts;
    });
  });

  const maxScore = Math.max(...Object.values(scores));
  const top = Object.keys(scores).filter((c) => scores[c] === maxScore);
  const winner = top[Math.floor(Math.random() * top.length)];
  return { winner, scores };
}

function calculateResultFromTraits(selectedTraitNames) {
  const scores = {};
  CATEGORIES.forEach((c) => (scores[c] = 0));
  const selected = new Set(selectedTraitNames);

  TRAITS.forEach((t) => {
    if (selected.has(t.name)) {
      scores[t.category] = (scores[t.category] || 0) + 1;
    }
  });

  const total = Object.values(scores).reduce((a, b) => a + b, 0);
  let winner;
  if (total === 0) {
    winner = CATEGORIES[Math.floor(Math.random() * CATEGORIES.length)];
  } else {
    const maxScore = Math.max(...Object.values(scores));
    const top = Object.keys(scores).filter((c) => scores[c] === maxScore);
    winner = top[Math.floor(Math.random() * top.length)];
  }
  return { winner, scores };
}

function getAllFigures(category) {
  if (!category) return FIGURES;
  return FIGURES.filter((f) => f.category === category);
}

function getFigure(id) {
  return FIGURES.find((f) => f.id === Number(id)) || null;
}

function getRandomFigureByCategory(category) {
  const list = getAllFigures(category);
  if (list.length === 0) return null;
  return list[Math.floor(Math.random() * list.length)];
}
