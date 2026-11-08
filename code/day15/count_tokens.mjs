// Ders 15 — commands.csv'deki her komutun OpenVLA prompt'unu LLaMA-2 tokenizer'ıyla sayar (model GEREKMEZ).
// LLaMA-2 BPE'nin JS uyarlaması: npm install llama-tokenizer-js   (Ders 14'te 29871 = "▁" doğrulandı)
//   node count_tokens.mjs ../../datasets/01-libero-spatial-tr-mini/data/commands.csv token_counts.json
// Çıktı: her satır için prompt token sayısı (BOS dahil), komutun kendi token sayısı, parça listesi;
// ayrıca mekânsal ifadelerin ve "kase/kâse/kaldır" kelimelerinin tek başına parçalanışı.
import fs from "node:fs";
import llamaTokenizer from "llama-tokenizer-js";

const [csvPath, outPath = "token_counts.json"] = process.argv.slice(2);
const text = fs.readFileSync(csvPath, "utf8");

// basit CSV ayrıştırıcı (tırnaklı alanlar dahil)
function parseCSV(s) {
  const rows = []; let row = [], field = "", q = false;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (q) { if (c === '"') { if (s[i + 1] === '"') { field += '"'; i++; } else q = false; } else field += c; }
    else if (c === '"') q = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
    else if (c !== "\r") field += c;
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  return rows;
}
const [header, ...data] = parseCSV(text).filter(r => r.length > 1);
const col = Object.fromEntries(header.map((h, i) => [h, i]));

const prompt = (ins) => `In: What action should the robot take to ${ins.toLowerCase()}?\nOut:`;
const pieces = (s) => llamaTokenizer.decode ? llamaTokenizer.encode(s, false, false).map(id => llamaTokenizer.decode([id], false, false)) : [];

const rows = data.map(r => {
  const ins = r[col.instruction];
  const p = prompt(ins);
  const idsPrompt = llamaTokenizer.encode(p);            // BOS dahil
  const idsIns = llamaTokenizer.encode(ins, false, false); // BOS/EOS yok
  return {
    id: r[col.id], task_id: +r[col.task_id], lang: r[col.lang], condition: r[col.condition], instruction: ins,
    prompt_tokens_with_bos: idsPrompt.length, instruction_tokens: idsIns.length,
    last_prompt_token: idsPrompt[idsPrompt.length - 1],
    instruction_pieces: pieces(ins),
  };
});

// mekânsal ifadeler ve tek kelimeler
const phrases = {
  "between the plate and the ramekin": "en", "tabak ile ramekinin arasındaki": "tr", "tabakla ramekin arasında": "tr2",
  "next to the ramekin": "en", "ramekinin yanındaki": "tr", "ramekinin yanında": "tr2",
  "from table center": "en", "masanın ortasındaki": "tr", "masanın ortasında": "tr2",
  "on the cookie box": "en", "kurabiye kutusunun üstündeki": "tr", "kurabiye kutusunun üzerinde": "tr2",
  "in the top drawer of the wooden cabinet": "en", "ahşap dolabın üst çekmecesindeki": "tr", "ahşap dolabın üst çekmecesinde": "tr2",
  "kase": "w", "kâse": "w", "kaseyi": "w", "kâseyi": "w", "kaldır": "w", "al": "w", "koy": "w", "yerleştir": "w", "duran": "w",
};
const phraseRows = Object.entries(phrases).map(([ph, kind]) => ({ phrase: ph, kind, n: llamaTokenizer.encode(ph, false, false).length, pieces: pieces(ph) }));

fs.writeFileSync(outPath, JSON.stringify({ rows, phrases: phraseRows }, null, 1));
for (const r of rows) console.log(`${r.id.padEnd(16)} ${String(r.prompt_tokens_with_bos).padStart(3)} prompt  ${String(r.instruction_tokens).padStart(3)} komut  son=${r.last_prompt_token}`);
console.log("\n-- ifadeler --");
for (const p of phraseRows) console.log(`${String(p.n).padStart(3)}  ${p.phrase.padEnd(42)} ${p.pieces.join("|")}`);
console.log(`\nyazıldı: ${outPath}`);
