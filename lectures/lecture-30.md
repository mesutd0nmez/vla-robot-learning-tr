# Ders 30 — Danışman Toplantısı: Repodaki 11 Açık Karar, Etki × Aciliyet, 30 Dakikalık Gündem, 7 Slayt ve Karar Kaydı

**Faz 4 / Gün 30** · **Süre:** ~55 dk (okuma 40 + görev 15)
**Ön koşul:** Ders 21 (Webots karar notu), Ders 25 (S1–S10), Ders 26 (RQ-a/b/c), Ders 27 (deney matrisi, bütçe), Ders 29 (`thesis/proposal-v0.1.md`) · **Kod:** `code/day30/` (`collect_questions.py` repoyu tarar, `plot_day30.py` öncelik ve gündem — ikisi de bu oturumda koştu) · **Çıktı:** `slides/advisor-meeting-01.md` (gündem + 7 slayt + soru listesi + karar kaydı şablonu)

> **Bugünün tek cümlesi:** 29 derste biriken "danışmanla konuşulacak" notlarının hepsi repoda; `collect_questions.py` Ders 27'nin sonundaki repoda bunları 37 dosyada 184 satırda buldu, biz 11 soruya indirdik ve etki × aciliyetle sıraladık — 30 dakikalık toplantının 17 dakikası beş karara gidiyor (Q1 suite, Q2 iddia birimi, Q3 zayıf/güçlü okuma, Q4 LoRA/GPU, Q11 takvim), çünkü bu beşi cevaplanmadan deney matrisinin yarısı başlayamaz ve bütçe 108 ile 258 T4 saati arasında asılı kalır; bir soru, cevabı hangi hücreyi değiştireceği yazılmamışsa toplantıya girmez.

---

## 1. Neden ayrı bir ders: danışman zamanı en kıt kaynak

Bu noktaya kadar her ders bir şeyi "danışmanla netleşecek" diye bıraktı: Ders 21 Webots'u, Ders 24 LoRA bütçesini, Ders 25 S7–S10'u, Ders 26 RQ sıralamasını, Ders 27 iddia birimini ve 4-bit OpenVLA'yı. Bunlar dağınık kalırsa iki şey olur: ya toplantıda hatırlanmaz, ya da toplantı "neler yaptın" anlatımına gider ve karar çıkmaz. Ders 27'nin hesabı bu kararların fiyatını gösterdi: iddia birimi tek başına bütçeyi 2,4 kat değiştiriyor. Danışmanın bir cevabı haftalarca T4 zamanı demek.

Bugünün yöntemi üç adım: **topla** (repoda iz bırakan her açık karar), **süz** (cevabı bir hücreyi, RQ'yu ya da tez cümlesini değiştirmeyen soru listeden çıkar), **sırala** (etki × aciliyet). Sonuç tek dosya: `slides/advisor-meeting-01.md`. Danışmana 2–3 gün önce e-postayla gider, yanında Ders 29'un bir sayfalık önerisi (`thesis/proposal-v0.1.md`).

Danışman toplantısı üzerine yazılmış rehberler (University Affairs'in "six steps" yazısı, UW CSE doktora el kitabının danışman ilişkisi bölümü, Imperial'in gündem tavsiye notu, MIT OCW'nin "effective meetings with advisors" dersi; Kaynaklar [1]) bu oturumda yalnızca arama sonucu olarak görüldü, içerikleri açılamadı. Aşağıdaki üç kural (gündemi önceden göndermek, kararları toplantıda yazmak, aynı gün özet e-postası) bu dersin önerisi; o kaynaklarla karşılaştırmak senin okuman.

Tez bağlantısı: Türkçe mekânsal grounding tezinin çerçevesi (hangi suite, hangi birim, hangi katkı düzeyi) teknik değil, **akademik** bir karar. Hangi sonucun jüri için "katkı" sayılacağını sen değil danışmanın bilir; Q3 (zayıf mı güçlü okuma mı) bunun en net örneği.

---

## 2. Topla: repodaki her açık karar, dosya:satır izli

`collect_questions.py` repodaki bütün `.md` dosyalarını sekiz anahtarla tarıyor ("danışman", "S7", "S8", "S9", "S10", "karar", "iddia birimi", "4-bit") ve her eşleşmeyi kaynağıyla CSV'ye yazıyor (`results/questions_raw.csv`). Bu oturumdaki gerçek çıktı (Ders 27 sonundaki repo, Ders 28–31 yazılmadan önce):

```
184 satır, 37 dosya
anahtar başına: karar 85, danışman 55, 4-bit 43, S10 17, iddia birimi 10, S9 9, S7 7, S8 5
klasör başına: lectures 96, plan 36, code 18, thesis 13, resources 7, README 5, home 5, …
'danışman' + karar/soru içeren aday satır: 20
```

Ders 28–31 eklendikten sonra aynı script 290 satır / 45 dosya buluyor (`results/collect_stdout.txt`), çünkü yeni derslerin metni de bu kararlardan söz ediyor (bu ders dahil). Sayı büyüyor ama soru kümesi aynı kalıyor; süzmenin değeri burada. 184 satırın çoğu tekrar: aynı S7 hem `plan/faz3-open-questions.md`'de hem Ders 25'te hem README'de geçiyor. Süzme elle yapıldı ve kural tek: **cevabı neyi değiştiriyor?** Değiştirdiği şeyi yazamadığın soru, merak sorusudur; e-postaya ya da sonraki toplantıya kalır. Örneğin "Qwen3-VL Türkçe'yi Çince tarafında mı okuyor?" (Ders 24 hipotezi) iyi bir soru ama danışmanın cevabı onu kapatmaz, S1 koşusu kapatır. Bu yüzden listede yok. `plan/faz3-open-questions.md`'deki S1–S6 da yok, çünkü onlar **senin koşun**: danışmana "S1'i koştum, örüntü B" diye götürülür, sorulmaz.

Kaynağı ayrı tutmanın amacı uydurma soruyu engellemek. `code/day30/questions.csv`'deki her sorunun `source` sütunu bir repo dosyasını gösteriyor. Toplantıda "bunu nereden çıkardın?" diye sorulursa cevap bir satır numarası.

---

## 3. Süz ve sırala: 11 soru, etki × aciliyet

Her soru altı alanla yazıldı (Ders 26'nın RQ şablonunun küçük kardeşi): bağlam (tek cümle), seçenekler, **önerimiz**, cevabın neyi değiştirdiği (hücre / RQ / tez cümlesi), etki (1–5), aciliyet (1–5). Etki ve aciliyet **öznel puanlar**, bu dersin önerisi; toplantıdan önce sen değiştirebilirsin, `plot_day30.py` yeniden çizer.

![Öncelik matrisi](images/lecture-30/02-priority-matrix.png)

| sıra | soru | etki × aciliyet | yeri |
|---|---|---|---|
| 1 | **Q1** Ana suite: Spatial / Goal / ikisi (S7) | 5 × 5 = 25 | toplantı |
| 2 | **Q2** İddia birimi: görev / blok | 5 × 5 = 25 | toplantı |
| 3 | **Q3** Katkı ölçütü: zayıf / güçlü okuma (S10) | 5 × 4 = 20 | toplantı |
| 4 | **Q11** Tez önerisi takvimi ve formu | 4 × 4 = 16 | toplantı |
| 5 | **Q4** Türkçe LoRA, GPU erişimi (S9) | 4 × 3 = 12 | toplantı |
| 6 | **Q5** OpenVLA 4-bit kabul | 3 × 4 = 12 | hızlı onay |
| 7 | **Q6** Türkçe set için ikinci okuyucu | 3 × 3 = 9 | hızlı onay |
| 8 | **Q10** Tekrar yalnız pilotta (ön kayıt değişikliği) | 2 × 3 = 6 | hızlı onay |
| 9 | **Q7** Webots rolü (S8) | 3 × 2 = 6 | e-posta |
| 10 | **Q8** Tatar vd. bildirisiyle ilişki | 3 × 2 = 6 | e-posta |
| 11 | **Q9** Cohn hattı tezin parçası mı | 2 × 2 = 4 | e-posta |

Aciliyetin tanımı "cevap gelmeden hangi hücre başlayamaz". Q1 ve Q2 olmadan A1–A4 (RQ-a çekirdeği, 152 saat orta senaryo) planlanamaz, bu yüzden 5. Webots (Q7) etkili bir soru, çünkü cevap "evet" olursa haftalar ekler. Ama Ders 21'in karar notu LIBERO'da kalmayı öneriyor ve hiçbir çekirdek hücre ona bağlı değil, bu yüzden aciliyeti 2. Yazılı sorulmasının nedeni de bu: danışman Ders 21'in üç sorusunu kendi zamanında okuyup cevaplayabilir. Q8 (Tatar vd.) ve Q9 (Cohn) için de aynı mantık geçerli. İkisi tezin *konumlanışını* belirliyor (giriş ve ilgili çalışmalar bölümü), deneyin akışını belirlemiyor.

Q11 senin repondan çıkmayan tek soru. Enstitünün tez önerisi takvimi ve formu repoda yazılı değil, Ders 29'un taslağı da belirli bir formu varsaymadan yazıldı. Bu bilgiyi kendi kendine tahmin etmek yerine sormak en ucuz yol.

---

## 4. Karar ağacı: danışmanın cevapları hangi RQ'yu birincil yapar?

Ders 26 RQ sıralamasını S1 örüntüsüne bağlamıştı (veriden gelen koşul). Danışmanın cevapları ikinci bir koşul: **çerçeve**. İkisi birlikte okunur. S1 "Türkçe okunmuyor" (örüntü B) derken danışman "zayıf okuma yeter" derse, tez MT/parser'la kapanan bir ölçüm tezidir. Aynı S1 ile "güçlü okuma şart, GPU yok" cevabı gelirse tezin birincil sorusu RQ-b'ye kayar: güçlü okuma ölçülür ama kapatılamaz.

![Karar ağacı](images/lecture-30/01-decision-tree.png)

Ağacın sırası bilinçli: önce Q3 (katkı ölçütü), çünkü tezin başlık cümlesini o belirliyor. Sonra Q4 (yalnızca "güçlü okuma şart" dalında anlamlı). Sonra Q1 (suite) ve en sonda Q2 (birim). Q2 bütçeyi belirliyor ama hiçbir RQ'yu elemiyor. Toplantıda soruları bu sırayla sorarsan her cevap bir sonrakinin seçeneklerini daraltır.

Üç "önerimiz" dalı (zayıf taban + güçlü hedef; ikisi; blok) Ders 24–27'nin gerekçeleriyle seçildi. Gerekçenin kendisi slaytta tek satır, ayrıntı `slides/advisor-meeting-01.md`'nin soru tablosunda. Önerisi olmayan soru toplantıyı danışmana iş yükleyen bir oturuma çevirir. Önerisi olan soru ise "onaylıyor musunuz, yoksa neden değil?" sorusuna dönüşür ve daha hızlı karara bağlanır.

---

## 5. Gündem ve 7 slayt

![Gündem](images/lecture-30/03-agenda.png)

Gündemin oranı kasıtlı: **7 dakika bağlam, 17 dakika karar, 3 dakika hızlı onay, 3 dakika özet.** Bağlamı uzatma isteği güçlü olur, çünkü 30 ders birikmiş durumda. Ama danışman ayrıntıyı e-postadaki bir sayfalık öneriden okuyabilir, karar ise yalnızca toplantıda çıkar.

`slides/advisor-meeting-01.md`'deki iskelet, her slayt için başlık, üç madde ve görsel içeriyor. Görseller önceki derslerden alındı, yeni çizim gerekmiyor:

| slayt | başlık | görsel |
|---|---|---|
| 1 | Tez ekseni ve neredeyiz (S1 henüz yok, dürüstçe) | Ders 25 / 01 |
| 2 | Üç araştırma sorusu | Ders 26 / 03 |
| 3 | Deney matrisi ve bütçe (108 / 258 sa) | Ders 27 / 03 |
| 4 | Karar: suite + iddia birimi (Q1, Q2) | Ders 27 / 04 |
| 5 | Karar: katkı ölçütü + LoRA (Q3, Q4) | Ders 30 / 01 |
| 6 | Riskler ve hızlı onaylar (Q5, Q6, Q10) | Ders 27 / 05 |
| 7 | Takvim ve sonraki 4 hafta (Q11) | Ders 29 Gantt |
| yedek | Literatürdeki konum | Ders 26 / 05 |

Slayt 1'deki "S1 henüz yok" satırı bilerek orada. Toplantıyı S1'i (Ders 18 koşusu, ~2 T4 saati) koştuktan sonra yapmak çok daha güçlü olur: karar ağacının iki koşulu (veri + çerçeve) aynı anda masada olur. Takvim izin vermezse slayt 1 bunu açıkça söyler ve RQ sıralamasının koşullu olduğunu gösterir (Ders 26 §6 tablosu yedek slayt olabilir).

---

## 6. Toplantıdan sonra: karar kaydı ve dosyaların güncellenmesi

![Toplantı döngüsü](images/lecture-30/04-meeting-cycle.png)

Toplantının çıktısı bir **karar kaydı** (ADR, Architecture Decision Record tarzı). `slides/advisor-meeting-01.md`'nin sonundaki şablonda her karar için şu alanlar var: başlık, tarih, durum (kabul / ret / ertelendi), tek cümlelik karar, danışmanın gerekçesi, sonuç (hangi dosya ve hücre değişiyor), sahip, son tarih. Şablonu toplantı bitmeden doldurmak gerekiyor; ertesi gün hatırlanan karar, danışmanın söylediği karar değildir.

Aynı gün üç dosya güncellenir, aksi hâlde karar repoya girmemiş sayılır:

- `plan/faz3-open-questions.md`: S7–S10 satırlarının **Durum** sütunu (tarih + tek satır sonuç; satır silinmez).
- `thesis/rq-candidates.md` → v0.2: sıralama tablosuna danışman koşulu, Q10 onaylandıysa tekrar politikası. Ön kayıt mantığıyla değişiklik tarih ve gerekçeyle sürüm bölümüne yazılır.
- `thesis/experiment-matrix.csv`: Q1/Q2'ye göre `n_init` ve suite sütunları, Q4/Q5'e göre `status`. Sonra `budget.py` yeniden çalıştırılır ve yeni toplam ADR'nin "Sonuç" alanına yazılır.

Son adım özet e-postası: kararların birer satırlık listesi, "yanlış anladığım bir şey varsa düzeltir misiniz?" ve bir sonraki toplantı için tarih önerisi. Bu e-posta yazılı bir onay işlevi görür; altı ay sonra "bunu birlikte kararlaştırmıştık" demenin kanıtı odur. Cevapsız kalan soru (danışman "düşüneyim" dediyse) sonraki toplantının listesine `ertelendi` durumuyla taşınır.

---

## 7. Yazılımcı için köprüler

- **Danışman toplantısı = architecture review board.** Sunuma kod yürüyüşü götürmezsin; seçenekleri, önerini ve her seçeneğin maliyetini götürürsün. Kurul onaylar ya da gerekçeyle reddeder. "Ne yapmalıyım?" diye sormak yerine "A'yı öneriyorum, çünkü X; B'nin bedeli Y. Onaylıyor musunuz?" diye sormak toplantıyı kısaltır.
- **ADR = karar kaydı, commit mesajı değil.** `docs/adr/0001-use-postgres.md` gibi: bağlam, karar, sonuçlar, durum. Altı ay sonra "neden Goal ağırlıklı?" sorusunun cevabı danışmanın hafızasında değil `ADR-001`'de durur. Durumu `superseded` olan ADR silinmez, yenisine bağlanır.
- **Etki × aciliyet = backlog önceliklendirme.** Jira'daki priority × severity ya da WSJF (cost of delay / job size) ile aynı fikir. "Hangi hücre bekliyor?" sorusu cost of delay'in kendisi: Q2 cevapsız kaldıkça 150 saatlik iş bloke.
- **`collect_questions.py` = TODO/FIXME tarayıcısı.** Kod tabanında `// TODO(arch-review)` satırlarını toplayan bir analyzer gibi. Ham çıktı gürültülü (184 satır), değer süzmede (11 soru). Her öğenin dosya:satır izi olması, issue'nun koda bağlı olması gibi bir şey.
- **Hızlı onay = consent agenda.** Tartışma gerektirmeyen maddeler (Q5, Q6, Q10) toplu sunulur, itiraz yoksa geçer. Toplantı süresi yalnızca gerçekten tartışılacak maddelere harcanır.
- **Özet e-postası = PR açıklaması + "LGTM?"** Toplantıda konuşulan şey merge edilmeden önce yazılı olarak gözden geçirilir; sessizlik onay sayılmaz, cevap beklenir.

---

## 8. Bugünün görevi (15 dk)

`code/day30/questions.csv`'yi aç ve **etki** ile **aciliyet** sütunlarını kendi yargınla yeniden puanla. Özellikle Q4 (GPU erişimini zaten biliyor olabilirsin) ve Q11'e (enstitü takvimini biliyorsan aciliyeti değişir) bak. Sonra `cd code/day30 && python plot_day30.py` çalıştır ve sıralamanın değişip değişmediğine bak. `slides/advisor-meeting-01.md`'de "önerimiz" sütununda katılmadığın bir satır varsa kendi önerinle değiştir; danışmanın karşısında savunacak olan sensin. Son olarak toplantı tarihini dosyanın başına yaz ve gündem e-postasının ilk paragrafını (3 cümle: neredeyim, beş karar, ekler) dosyanın altına taslak olarak ekle. Commit: `Ders 30: danışman toplantısı 01 hazırlığı`.

## Takılırsan

| Belirti | Çözüm |
|---|---|
| `plot_day30.py`: `ValueError: invalid literal for int()` | `questions.csv`'de virgül içeren bir hücre tırnaksız kalmış (Türkçe ondalık "16,8" gibi); hücreyi `"…"` içine al |
| `collect_questions.py` çok fazla satır buluyor | Normal (ham tarama). Daraltmak için `KEYS` sözlüğünden `karar`'ı çıkar ya da çıktıyı `keys` sütununda `danisman` içerenlere süz |
| Öncelik matrisinde daireler üst üste biniyor | Aynı (etki, aciliyet) hücresinde 3'ten fazla soru var demektir; puanlardan birini değiştir ya da `dx` çarpanını büyüt |
| Danışman toplantıda yeni bir konu açtı | ADR şablonuna `ertelendi` durumuyla ekle; gündemi kaydırma, son 3 dakikayı özet için koru |
| S1 toplantıdan önce bitmeyecek | Slayt 1'de "koşullu" de, Ders 26 §6 sıralama tablosunu yedek slayt yap; Q1–Q3'ü yine sor, çünkü çerçeve kararı veriden bağımsız |

---

## Sonraki ders (Ders 31)
**Final: "bir proje yapalım" simülasyonu ve Aşama 0'ın kapanışı — son ders.** Planın son satırı: Claude bir proje önerir, sen tasarlarsın. Bu sefer ne RQ ne matris hazır verilir; 30 derste kurulan araç kutusunun (literatür → sınanabilir soru → deney matrisi → bütçe → baseline/ablation → danışman kararı) yeni bir soru üzerinde uçtan uca ne kadar hızlı ve doğru çalıştığı ölçülür. Ders 31 aynı zamanda Aşama 0'ın envanterini çıkarır: hangi script gerçekten koştu, hangi veri kartı var, neyin hipotez kaldığı. Bu derslerden sonra yeni ders yazılmaz; sonraki aşamanın planı sende.

---

## Kaynaklar
- Çıktı: `slides/advisor-meeting-01.md` (gündem, 7 slayt + yedek, 11 soru, ADR şablonu; v0.1). Girdiler: `plan/faz3-open-questions.md` (S7–S10), `thesis/rq-candidates.md` (Ders 26), `thesis/experiment-matrix.md` (Ders 27 açık kararları), `thesis/proposal-v0.1.md` (Ders 29), `lectures/lecture-21.md` §6 (Webots karar notu + üç soru), `lectures/lecture-05.md` (Tatar vd., SIU 2026), `lectures/lecture-23.md` (Cohn hattı, yedi makale), `datasets/04-tr-spatial-commands/dataset-card.md` (anadil okuması v0.2'ye bırakıldı), `plan/lecture-plan.md` "Açık kararlar".
- **[1]** Danışman toplantısı rehberleri. Bu oturumda yalnızca arama sonucu olarak görüldüler, içerikleri açılamadı; ileri okuma olarak listeleniyor: [University Affairs — Six steps for stellar student-supervisor meetings](https://universityaffairs.ca/career-advice/six-steps-for-stellar-student-supervisor-meetings/) · [UW CSE PhD handbook — Building a relationship (advising guide)](https://www.cs.washington.edu/academics/phd/handbook/advising-guide/building-a-relationship) · [MIT OCW RES.16-9960 — Effective meetings with advisors (transkript)](https://ocw.mit.edu/courses/res-16-9960-the-engineering-students-guide-to-effective-technical-communication-spring-2026/mitres16-9960-effective_meetings_with_advisors_transcript.pdf) · [Imperial College — Effective PhD supervision meeting agendas (advice sheet)](https://t4awstest.cc.ic.ac.uk/media/imperial-college/administration-and-support-services/staff-development/public/pfdc/Download-C.4.9---Advice-sheet---Effective-PhD-supervision-meeting-agendas.pdf). Bu dersteki üç kural (önceden gündem, toplantıda karar kaydı, aynı gün özet e-postası) ve ADR uyarlaması bu dersin önerisidir.
- **[2]** ADR: Michael Nygard, *Documenting Architecture Decisions* (2011) — yazılım mühendisliğindeki karar kaydı biçimi; burada akademik karar kaydına uyarlandı. Önceliklendirme: etki × aciliyet matrisi (Eisenhower tipi), puanlar öznel.
- Dayanak sayılar: Ders 27 (`code/day27/results/budget_stdout.txt`: pilot + çekirdek 258 sa, yalın 108 sa, görev MDE 23 / blok n=120 → 13), Ders 24 (`qlora_budget.py` 65–98 T4 sa, tahmin), Ders 19 (komutsuz tavan Spatial 9/10, Goal 1/10), OpenVLA Tablo 2 (bf16 16,8 GB) ve VLAQuantBench Tablo 1 (OFT W4 Goal 96 → 82), Ders 27 kaynaklarında.
- Kod: `code/day30/` — `collect_questions.py` (bu oturumda koştu: Ders 27 sonundaki repoda 184 satır / 37 dosya; Ders 28–31 eklendikten sonra 290 / 45 — `results/questions_raw.csv` ve `collect_stdout.txt` son koşu), `questions.csv` (11 soru, elle süzüldü; her satırda `source`), `plot_day30.py` (koştu; `results/priority.csv`, `priority_stdout.txt`).
- Görseller: 01 ve 04 Mermaid (`.mmd` yanında); 02 öncelik matrisi (puanlar **öznel**, başlıkta yazılı); 03 gündem (bu dersin önerisi). Model koşusu yok. Toplantı simülasyonu değil, hazırlık dosyası: gerçek cevaplar danışmandan gelecek.
