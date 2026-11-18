// Ders 22 — Ders 15'in açık hipotezini kapatmak için: llama-tokenizer-js (Ders 14–15'te kullandığımız JS uyarlaması)
// ile HF `tokenizers` (Rust, transformers'ın kullandığı) aynı LLaMA-2 id dizisini veriyor mu?
//   npm install llama-tokenizer-js
//   node count_js_ids.mjs ../../datasets/04-tr-spatial-commands/data/commands.csv js_ids.json
//   python tokenize_set.py --jsids js_ids.json ...   → crosscheck.md
import fs from "node:fs";
import llamaTokenizer from "llama-tokenizer-js";

const [csvPath, outPath = "js_ids.json"] = process.argv.slice(2);
const text = fs.readFileSync(csvPath, "utf8");

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
const out = {};
for (const r of data) out[r[col.id]] = llamaTokenizer.encode(r[col.instruction], false, true); // BOS yok, öndeki boşluk (dummy prefix ▁) VAR — prompt içindeki hâl
fs.writeFileSync(outPath, JSON.stringify(out));
console.log(`${Object.keys(out).length} komut → ${outPath}`);
