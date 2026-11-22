# Ders 22 — Tokenizasyon Analizi: Aynı 100 Komut, İki Tokenizer — LLaMA-2 (OpenVLA) ve Qwen3 (VLANeXt) Türkçe Mekânsal Komutu Nasıl Böler?

**Faz 3 / Gün 22** · **Süre:** ~55 dk (okuma 40 + görev 15)
**Ön koşul:** Ders 5 (Tatar vd.'nin `Kal|dır` iddiası), Ders 13–14 (komutun VLANeXt'te chat şablonuna, OpenVLA'da `In: … Out:` prompt'una girişi), Ders 15 (ilk LLaMA-2 token sayımı; açık hipotez), Ders 20 (`datasets/04` seti, `tr_a_seg` / `rel_seg` ek ayrımı) · **Kod:** `code/day22/` · **Veri:** `datasets/05-tokenization-tr-commands/`

> **Bugünün tek cümlesi:** Türkçe komut modele cümle olarak değil parça dizisi olarak girer ve iki tokenizer'da da ilişkiyi taşıyan ek (*-ndeki*, *-ne*) kararlı bir parça değildir: LLaMA-2 Türkçeyi İngilizceye göre 1,6× şişirir, *ç/ş/ğ/ı* ile başlayan sözcüklerin önüne boş bir `▁` koyar ve *soldaki*'yi İngilizce *sold* + *aki* diye okur; Qwen3 1,4× şişirir, *arasındaki*'yi tek parça yapar ama aynı sözcüğü cümle başında üçe böler — "Türkçe okunmuyor" hipotezinin modelden bağımsız, bugün ölçülebilen ilk katmanı budur ve ölçtük.

---

## 0. Ne yapıldı, ne yapılamadı

Ders 20'nin vaadi, Ders 21'in Webots'a kayması yüzünden bir gün gecikmeyle: `datasets/04`'ün 100 satırı (20 EN–TR çift × 5 yüzey) iki tokenizer'dan geçti, çıktılar `datasets/05-tokenization-tr-commands/data/` altında, hepsi `code/day22/tokenize_set.py` ile yeniden üretilebilir. **Model çalıştırılmadı ve gerekmedi:** tokenizer, modelin önündeki saf metin işlemcisidir.

---

## 1. İki tokenizer, iki felsefe

![İki boru hattı](images/lecture-22/01-pipeline.png)

Tokenizer, metni modelin embedding tablosundaki satır numaralarına çeviren **deterministik** bir fonksiyondur; öğrenilmiş tek şeyi, eğitim külliyatında sık görülen karakter dizilerinin birleştirme (merge) listesidir. İkisi de BPE ama farklı tabanlarda:

| | **LLaMA-2 (OpenVLA)** | **Qwen3 (VLANeXt'in Qwen3-VL-2B'si)** |
|---|---|---|
| Tür | SentencePiece BPE, karakter tabanlı; bilinmeyen karakter → 256 bayt token'ı (`<0xCC>`) | byte-level BPE (GPT-2/tiktoken soyu): her UTF-8 baytı başlangıç sembolü, bilinmeyen diye bir şey yok |
| Sözlük | **32.000** parça (58.980 merge) | **151.643** parça (151.387 merge) + 26 özel token |
| Ön işlem | başa `▁` ekle (*dummy prefix*), boşlukları `▁` yap; ön-bölme yok | NFC normalizasyonu; regex ile sözcük/rakam/noktalama ön-bölmesi (`' ?harfler'` kalıbı: boşluk sözcüğe yapışık) |
| Eğitim verisi dili | LLaMA-2 raporu Tablo 10: İngilizce %89,7, "bilinmeyen" %8,4, diğer her dil ≤ %0,17; **Türkçe listede yok** (eşik %0,005) | Qwen3 raporu: 36 trilyon token, **119 dil ve lehçe**; Türkçe (`tur_Latn`) Belebele'nin Turkic grubunda değerlendiriliyor |
| Komutun girişi | `In: What action should the robot take to {komut.lower()}?\nOut:` + BOS (Ders 14) | `<|im_start|>user\n<|vision_start|>…<|vision_end|>{komut}<|im_end|>…` — komut görüntü token'larının **hemen ardından, boşluksuz** (Ders 13; şablon string'i hipotez) |
| Action'ın yeri | aynı vocab'ın **son 256 id'si** (31744–31999) action bin'leri (OpenVLA: "the 256 least used tokens … the last 256 tokens") | metin vocab'ında değil; policy başı flow-matching ile üretir (Ders 13) |

İki fark bugünkü bütün sayıların arkasında. **Birincisi sözlük büyüklüğü ve dili:** 151 bin parçalı, 119 dille eğitilmiş bir BPE Türkçe sözcüklerin bütününü (*arasındaki*, *yakın*, *önüne*, *kaldır*) tek parça olarak barındırabiliyor; 32 binlik, neredeyse yalnızca İngilizce görmüş bir BPE'de Türkçe karakterler bile tek tek (`ç` = id 30019, `ğ` = 30200, `ı` = 30130) ama onlarla başlayan hiçbir birleşme yok. **İkincisi boşluğun yeri:** SentencePiece boşluğu ayrı bir `▁` sembolü sayıp sonra birleştirir, bu yüzden `▁ç` diye bir parça yoksa `▁` tek başına kalır; Qwen boşluğu sözcüğe yapıştırıp öyle böler, bu yüzden aynı sözcük cümle başında (boşluksuz) ve ortasında (boşluklu) **farklı** parçalanır. Her ikisinin sonucunu §2–4'te göreceksin.

Tez eksenine bağ: Ders 8'de "VLANeXt'in Qwen omurgası OpenVLA'nın LLaMA-2'sine karşı doğal bir Türkçe tokenizer kontrolü sunar" demiştik. Bugün o kontrolün **ne kadar** kontrol olduğunu sayıyla görüyorsun: fark var ama "Qwen Türkçeyi düzgün böler" kadar temiz değil.

---

## 2. Sayılar: şişme, fertility, Türkçe harflerin bedeli

![Token sayıları](images/lecture-22/02-token-counts.png)

Yalnızca komut metni (şablon ve BOS hariç), 20 çiftin ortalaması (`summary.md`):

| yüzey | LLaMA-2 | Qwen3 | L/Q | fertility L | fertility Q |
|---|---|---|---|---|---|
| `en` | 13,6 (7–21) | 12,3 (6–19) | 1,11 | 1,14 | 1,01 |
| `tr_a` | **22,4** (12–36) | **16,9** (8–28) | 1,33 | **3,09** | **2,36** |
| `tr_b` | 21,9 | 16,1 | 1,37 | 3,35 | 2,44 |
| `tr_ascii` | 17,6 | 16,0 | 1,10 | 2,35 | 2,20 |
| `tr_bare` | 14,9 | 12,2 | 1,22 | 2,02 | 1,68 |

Okuma sırası: **(1) İngilizce'de iki tokenizer neredeyse aynı** — sözcük başına ~1 token; LLaMA-2'nin fazlası *bowl* → `▁bow|l`, *drawer* → `▁dra|wer`, *stove* → `▁st|ove` gibi kırılmalardan (OpenVLA'nın kendi eğitim komutlarında da böyle; kırılma tek başına felaket değil, model bu parçaları LIBERO fine-tune'unda görmüş). **(2) Türkçe'de şişme ikisinde de gerçek:** aynı çiftte `tr_a/en` oranı LLaMA-2'de ortalama **1,63** (1,20–2,00), Qwen3'te **1,41** (0,89–2,00). Sağdaki saçılım köşegenin altında yığılıyor: her çiftte LLaMA daha çok şişiriyor; tek istisna Goal çiftleri (C13/C14) — orada EN komut o kadar kısa (*put the bowl on the plate*, 6–7 token) ki oran paydadan büyüyor; C18 (*tabağı ocağın önüne it*) Qwen'de **0,89** çünkü `▁önüne` ve `▁it` tek parça, EN karşılığı 9 token. **(3) Fertility** (token/sözcük) Türkçe için LLaMA-2'de 3,1, Qwen3'te 2,4: Türkçe bir sözcük ortalama üç parçaya bölünüyor. Petrov vd. (NeurIPS 2023) aynı metnin diller arası token uzunluğunun 15 kata kadar açıldığını raporlar; bizim 1,6× ılımlı bir vaka, ama fertility 3 demek, modelin "bir sözcük" için üç konumda dikkat kurması demek. **(4) `tr_ascii` sütunu iki tokenizer'ı ayırıyor:** Türkçe harfleri atınca LLaMA-2 ortalama **4,9 token** kazanıyor, Qwen3 yalnızca **0,9** (görsel 04 sağ panel). Yani LLaMA-2'deki şişmenin yarısından fazlası *ç ğ ı ö ş ü*'nün kendisinden; Qwen3'te harfler zaten birleşmelerin içinde. Qwen'de A01'in işareti ters (−1): *arasındaki* tek token, *arasindaki* üç — ASCII'leştirmek Qwen için **bozulma**dır, LLaMA için rahatlama. Ders 18'in koşul listesine bir aday koşul daha: `tr_ascii`, iki modelde zıt yönde çalışması beklenen tek yüzey.

Şablon içinde toplam uzunluk (OpenVLA prompt'u, BOS dahil): EN ortalama 27,6, `tr_a` 36,5, en uzun satır A04 (*ahşap dolabın üst çekmecesindeki…*) **50 token**; Ders 15'teki 29–35 / 42–50 aralıklarıyla tutarlı. Qwen tarafında metin 23 → 28, ama asıl kalabalık görüntü: 256 px başına 64 token × 2 kamera (Ders 13); komut dizinin %15–20'si. Hiçbir satır bağlam sınırına yaklaşmıyor — şişme bir **kapasite** sorunu değil, **temsil** sorunu.

---

## 3. İlişki sözcüğü: ek nerede, token nerede?

Ders 20 her çiftin ilişkiyi taşıyan sözcüğünü (`rel_span`: *arasındaki, soldaki, üstüne, sağına…*) ve ek ayrımını (`rel_seg`: *ara-sı-nda-ki*) vermişti. Soru şuydu: tokenizer, Türkçenin ilişki bilgisini koyduğu yere — **sözcüğün sonundaki ek zincirine** — bir sınır koyuyor mu?

![Parça şeritleri](images/lecture-22/03-segmentation.png)

Kırmızı kesik çizgiler biçimbirim sınırları, kutular token'lar. Üç gözlem:

**(a) Ek kararlı bir token değil.** Aynı `-ki` eki LLaMA-2'de üç biçimde karşına çıkıyor (`aki` ×5, `i` ×3, `ki` ×1), Qwen3'te beş biçimde (`eki`, `ındaki`, `aki`, `▁arasındaki`, `ki`); yönelme eki (`-nA/-A`) 5 / 6 biçimde (`suffix_forms.csv`, görsel 04 orta). *Üstündeki*'de `-ki`, `ek|i` diye ikiye bile bölünüyor (L). Yani "modele `-ki` = seçim/konum eki" diye tek bir sinyal gitmiyor; dokuz sözcükte dokuz farklı kuyruk gidiyor. Bu, İngilizce'nin *on / in / next to / between* sözcüklerinin **her zaman aynı token** olmasıyla taban tabana zıt: EN'de ilişki = sabit id, TR'de ilişki = bağlama göre değişen kuyruk. Ders 6'nın "dil dünyanın sıkıştırmasıdır" cümlesine bir ek: Türkçe'de o sıkıştırma eke gidiyor, tokenizer ise eki açmıyor.

**(b) Sınırlar tutmuyor; LLaMA-2 çok keserek tutturuyor.** 20 `tr_a` komutundaki 111 biçimbirim sınırının LLaMA-2'de 87'si (recall 0,78), Qwen3'te 78'i (0,70) bir token sınırına denk geliyor — ama LLaMA-2 bunu **296** sözcük-içi kesimle yapıyor (precision 0,29), Qwen3 **189** kesimle (0,41). Birincisi neredeyse harf harf kestiği için biçimbirim sınırlarını "tesadüfen" buluyor; ikincisi daha az kesiyor ve kestiklerinin %41'i gerçek sınır. Hiçbiri bir morfolojik çözümleyici değil; ikisinin de amacı sıkıştırma. (`seg_align.csv`; görsel 04 sol.)

![Sınır örtüşmesi, ek biçimleri, Türkçe harflerin bedeli](images/lecture-22/04-alignment.png)

**(c) Minimal çiftler token tarafında çift değil.** Ders 20'nin 11 minimal çiftinden ikisi: *soldaki / sağdaki* ve *soluna / sağına*. LLaMA-2: `▁sold|aki` / `▁sa|ğ|d|aki`; `▁sol|una` / `▁sa|ğ|ı|na`. Qwen3 (cümle başında): `sold|aki` / `sa|ğ|d|aki`; (ortada) `▁sol|una` / `▁sa|ğı|na`. Dört sözcükte ortak olan şey ek (*-da-ki*, *-ı/u-na*), ama paylaşılan token yalnızca `aki`; *sağ* hiçbir yerde kendi başına parça değil (`sa|ğ`), *sol* ise **soldaki**'de İngilizce ***sold*** (id 5239, "satıldı") ile birleşiyor. Model *soldaki*'nin içinde *sol*'u görmüyor; *sold*'u görüyor. Tatar vd.'nin "yanlış ipucu" argümanı (Ders 5) için bu, `Kal|dır`'dan çok daha iyi bir örnek — ve gerçek.

Bir de teselli: *üstüne* (C13–C15) her iki tokenizer'da da `…ü|ne` diye bitiyor, yönelme eki **ayrı** ve sınırlar birebir (recall 1,0). Goal suite'in "kaseyi X'in üstüne koy" komutları token tarafında en temiz Türkçe satırlar; Spatial'ın `-ndeki` seçim komutları en kirlileri. Ders 19'un "Goal'da dil taşıyıcı, Spatial'da imza" bulgusuyla yan yana koyunca ilginç bir öngörü çıkıyor: Türkçe'nin **en iyi parçalandığı** suite, dilin **en çok işe yaradığı** suite. Hipotez; Ders 18'in Goal koşusu söyleyecek.

---

## 4. Beş tuzak ve bir rahatlama

![Tuzaklar](images/lecture-22/05-traps.png)

`probe_edge_cases.py` çıktısından (`probe_edge_cases.txt`):

1. **Tek başına `▁` (yalnızca LLaMA-2).** 80 Türkçe satırda **14** kez `▁` kendi başına token: `▁|ş` ×6 (*şarap, şişesini*), `▁|ön` ×5 (*önüne, öndeki*), `▁|ç` ×3 (*çekmecesindeki*). Sebep: `▁ç`, `▁ş`, `▁ğ`, `▁ı` vocab'da yok (`▁ö`, `▁ü`, `▁â` var). *Çekmecesindeki* böylece **8** parça: `▁|ç|ek|m|ec|es|inde|ki`. Boş `▁` token'ı (id 29871) OpenVLA'nın prompt'unda zaten en sonda duruyor (Ders 14: "Out:" + `▁`); Türkçe komut aynı id'yi cümlenin ortasına da serpiştiriyor.
2. **Konum etkisi (yalnızca Qwen3).** Setin 31 Türkçe sözcüğü cümle başında ve ortasında farklı bölünüyor: *arasındaki* ortada `▁arasındaki` (1), başta `aras|ındaki` (2); *kaldır* ortada `▁kaldır` (1), başta `k|ald|ır` (3). B07–B09'un ilişki sözcükleri (*soldaki, sağdaki, öndeki*) komutun **ilk** sözcüğü ve VLANeXt şablonunda komut `<|vision_end|>`'den hemen sonra geliyorsa boşluksuz tokenize ediliyor. Tasarım dersi: Türkçe komutlarda ilişki sözcüğü başta olduğu için (Ders 20: "ayırt edici tarif TR'de en başta") bu etki tam da ölçmek istediğimiz sözcüğe düşüyor. Karşı önlem basit: komutun başına bir boşluk ya da şablona `\n` koymak — ama bunun eğitim dağılımına uymadığını da bil (hipotez: VLANeXt'in eğitim komutları da boşluksuz).
3. **`lower()` tuzağı (yalnızca OpenVLA).** Python'da `"İ".lower()` = `i` + U+0307 (birleştirici nokta); Ders 16'da BDDL için görmüştük, tokenizer'da karşılığı: *İçine koy* → `▁i|<0xCC>|<0x87>|ç|ine|▁ko|y` — iki **bayt kırıntısı**. Setin 100 satırı küçük harf olduğu için etkilenmedi (`n_byte_fragments` = 0), ama büyük harfle başlayan bir Türkçe komut (ya da *İ* içeren *İstanbul*, *İleri*) OpenVLA'ya iki anlamsız bayt token'ı gönderir. Qwen'in NFC normalizer'ı da bunu birleştirmez (NFC, `i`+U+0307'yi `İ`'ye çevirmez).
4. **Yanlış arkadaşlar.** *it* (itmek, C18) = İngilizce *it* (LLaMA id **372**, Qwen 432) — LIBERO-Spatial'ın her komutundaki *place **it** on the plate*'in *it*'i. *soldaki* → *sold*; *al* → *al* (*et al.*'ın *al*'ı, id 394); *duran* → `▁dur|an`. Bunlar "yanlış ipucu" değil "**aynı ipucu, farklı anlam**": model *it* görünce eğitimden bildiği şey nesne zamiridir, Türkçe'de fiil. Tatar'ın `Kal|dır` örneğine gelince: gerçek LLaMA-2 tokenizer *kaldır*'ı `▁k|ald|ı|r`, *Kaldır*'ı `▁K|ald|ı|r` yapıyor; `Kal` ya da `dır` diye bir parça çıkmıyor. Makaledeki örnek bu tokenizer'dan gelmiyor (hipotez: şematik ya da başka bir tokenizer); argümanın özü — kökün anlamsız yerinden kırılması — yine de doğru, yalnızca örnek yanlış. Qwen3 ise *kaldır*'ı cümle ortasında **tek token** yapıyor.
5. **Şapkalı â.** *kâseyi* iki tokenizer'da da `k|â|sey|i` (4), *kaseyi* `k|ase|yi` (3); `tr_b` yüzeyinde (*kâse*) bu, çift başına bir token ve kökün farklı kırılması demek. Ders 20'nin "anadil okuması" sorusuna token tarafından cevap: imla tercihi parçalanışı değiştiriyor, `tr_b`'yi "daha doğal ama daha farklı tokenize" diye okumak gerek.

**Rahatlama:** OpenVLA'nın action bin'lerine ayırdığı son 256 id'ye (31744–31999) Türkçe komutlardan **hiçbir token düşmüyor** (`action_collision.csv`); Korece, Japonca, Çince, Rusça ve Arapça birer deneme cümlesinde de sıfır — o aralık nadir CJK karakterleri ve `ķ`, `ṯ` gibi parçalardan oluşuyor. Dong vd.'nin (Ders 15) Korece'de gördüğü düşüş (97,6 → 49,2) action-token çakışmasıyla açıklanamaz; başka yerde aranacak.

---

## 5. Bu sayılar tez için ne söyler, ne söylemez

**Söyledikleri.** (i) "Türkçe okunmuyor" hipotezinin (Ders 19'un B dalı) modelden bağımsız bir ön katmanı var ve iki modelde farklı büyüklükte: LLaMA-2'de şişme + harf başı boşlukları + kökün İngilizce sözcüklerle birleşmesi; Qwen3'te daha az şişme, bazı tam sözcükler, ama ekin yine kararsız oluşu ve konum etkisi. (ii) Ders 18 analizinde `n_tokens`, `fertility`, `n_space_alone` ve "ilişki sözcüğü kaç parça" sütunları artık her komut için **hazır ortak değişken** (`tokens.csv`); success rate'i komut başına bunlarla çaprazlamak (örneğin: parçalanma arttıkça `tr` – `empty` farkı küçülüyor mu?) üç satır pandas. (iii) `tr_ascii` koşulu iki modeli **zıt** yönde etkilemesi beklenen tek yüzey; Ders 24'teki "çeviri katmanı / çok dilli omurga / Türkçe LoRA" karşılaştırmasına girmeden önce ucuz bir ayrıştırıcı.

**Söylemedikleri.** Token sayısı anlama değildir. OpenVLA'nın kendi İngilizce komutlarında da *bowl* iki parçadır ve model %97 başarır — çünkü o parçaları **eğitimde gördü**. Türkçe parçaların sorunu uzun olmaları değil, LLaMA-2'nin ön eğitiminde Türkçe'nin %0,005'in altında kalması ve LIBERO fine-tune'unda hiç olmaması. Qwen3-VL Türkçe gördü (119 dil), ama VLANeXt'in **policy başı** yalnızca İngilizce komut-eylem çiftleriyle eğitildi (Ders 13: 29 katmanın her birine cross-attention). Yani Qwen tarafında temiz tokenizasyon, komut vektörlerinin İngilizce karşılıklarına **yakın düşmesini** garanti etmez; bunun ölçüsü Ders 15'in `d(en, tr)` vs `d(en, para)` tasarımıdır, tokenizer tablosu değil. Üçüncü uyarı: bugünkü sayılar `datasets/04` v0.1 üstünde; `human_check.csv` düzeltmelerin `tr_b`'yi değiştirirse tablo yeniden koşar (bir komut, 1 dk).

Tez metninde bu dersin yeri muhtemelen bir alt bölüm: "Türkçe komutların iki tokenizer'daki görünümü" — bir tablo (yukarıdaki), bir şerit görseli (03) ve iki paragraf. Danışmana götürülecek cümle: *"Qwen tabanlı modelin Türkçe'de daha iyi çıkması beklenir; ama tokenizer farkı tek başına bunu açıklamaz, çünkü ilişki eki orada da kararsız."*

---

## 6. Yazılımcı için köprüler

- **Tokenizer = lexer; vocab = keyword tablosu.** Deterministik, öğrenmesiz (çalışma anında), girdiyi id dizisine çevirir. BPE merge listesi, lexer'ın "en uzun eşleşme" kuralının öğrenilmiş hâli. İki modelin farkı iki farklı lexer'ın aynı kaynak dosyayı farklı token akışına çevirmesi — parser (model) ondan sonra gelir.
- **Byte fallback = `Encoding.GetBytes` ile kurtarma.** SentencePiece bilmediği karakteri baytlara düşürür (`<0xCC><0x87>`); byte-level BPE zaten bayttan başlar, "bilinmeyen" kavramı yoktur. `lower()`'ın İ'yi iki code point'e açması, .NET'te `ToLowerInvariant()` yerine `ToLower(tr-TR)` kullanmamanın klasik hatasıdır; burada tersine, Python'un `str.lower()`'ı kültürden bağımsız çalışıp Türkçe İ'yi bozuyor.
- **`▁` = boşluğun escape'i.** SentencePiece boşluğu bir sembole (U+2581) çevirir ki birleştirme boşluğu da kapsasın; `▁ç` yoksa `▁` tek kalır. Qwen'in `Ġ`'si aynı fikrin byte-level versiyonu, ama regex ön-bölme boşluğu sözcüğe baştan yapıştırdığı için konum (cümle başı/ortası) tokenizasyonu değiştirir — `Trim()` yapıp yapmamanın hash'i değiştirmesi gibi.
- **Action bin'lerinin son 256 id'ye yazılması = enum'un üst değerlerini ödünç almak.** Mevcut tabloyu büyütmek yerine en az kullanılan 256 satırı yeniden anlamlandırmak; bugünkü kontrol, "mevcut girdilerden biri o aralığa düşüyor mu" testidir (CI'da reserved-range çakışma testi). Türkçe için düşmüyor.
- **Fertility = sıkıştırma oranı; recall/precision = tokenizer'ın morfolojiye "uyumu".** İkisi farklı amaç: BPE'nin hedef fonksiyonu sıkıştırmadır, bizim istediğimiz anlam sınırıdır. Yüksek recall + düşük precision (LLaMA-2) = çok küçük parçalara bölünce sınırı kaçırmamak; `split('')` her sınırı bulur ama hiçbir şey söylemez.
- **`crosscheck.md` = iki implementasyonun golden-file testi.** JS uyarlaması ile referans C++ kütüphanesi aynı 100 girdide aynı çıktıyı verdi; Ders 15'teki "uyarlama Türkçe'de ayrışıyor olabilir" şüphesi kapandı. Üçüncü implementasyon (OpenVLA'nın HF `LlamaTokenizerFast`'i) senin görevin.

---

## 7. Bugünün görevi (15 dk)

Mac'te, `vla` ortamında, model indirmeden: `code/day22/README.md`'deki komutları sırayla koş — `tokenizer.model`'i indirip sha256'sını kontrol et, `tokenize_set.py`'yi **iki kez** çalıştır: bir kez `--llama tokenizer.model --qwen Qwen/Qwen3-VL-2B-Instruct` (HF'den yalnızca tokenizer dosyaları iner), bir kez `--llama openvla/openvla-7b --qwen Qwen/Qwen3-VL-2B-Instruct --out /tmp/d22_hf`. Sonra `diff <(cut -d, -f1-7 datasets/05-tokenization-tr-commands/data/tokens.csv) <(cut -d, -f1-7 /tmp/d22_hf/tokens.csv)` — boşsa iki hipotez birden kapanır (Qwen3-VL tokenizer'ı = bugünkü dosya; OpenVLA'nın HF tokenizer'ı = Meta SentencePiece). Fark varsa farklı satırları `datasets/05-tokenization-tr-commands/dataset-card.md`'nin "Bilinen sınırlılıklar" bölümüne yaz ve kartı v0.2 yap. Ek 3 dakika: Ders 13'ün `show_prompt.py`'sini çalıştırıp `<|vision_end|>` ile komut arasında boşluk/`\n` olup olmadığını kesinleştir — §4'ün 2. tuzağı buna bağlı; sonucu `plan/daily-log.md`'ye bir satır not düş.

## Takılırsan

| Belirti | Çözüm |
|---|---|
| `tokenize_set.py`: `immutable_proto is deprecated` ya da `return_type` hatası | sentencepiece sürüm farkı; script ikisini de dener — `pip install -U sentencepiece` (0.2.x) |
| `Tokenizer.from_pretrained` 401/ağ hatası | HF erişimi yok: `tokenizer.json`'u başka makinede indirip `--qwen /yol/tokenizer.json` ver; `hf download Qwen/Qwen3-VL-2B-Instruct tokenizer.json tokenizer_config.json` |
| `shasum` çıktısı `9e556afd…` değil | Dosya LLaMA-2 tokenizer'ı değil (ör. npm `@lenml/tokenizer-llama2` farklı vocab); `karpathy/llama2.c` yolunu kullan ya da `openvla/openvla-7b`'den `tokenizer.model` indir |
| HF `openvla/openvla-7b` ile `tokens.csv` farklı çıkıyor | Önce `ids` sütununu karşılaştır; fark yalnızca ilk token'daysa `legacy`/`add_prefix_space` ayarı — tokenizer_config'e bak, karta yaz |
| `count_js_ids.mjs`: `Cannot find package 'llama-tokenizer-js'` | `npm install llama-tokenizer-js` script'in bulunduğu klasörde |
| Qwen parçalarında `<Ġ…>` gibi kırıntılar | Tek başına geçerli UTF-8 olmayan bayt parçaları; setteki 100 satırda yok, kendi eklediğin komutta varsa `n_byte_fragments` sütununa bakıp hangi karakterden geldiğini bul |
| `plot_day22.py`: Türkçe harfler kutuda `□` | Monospace font (DejaVu Sans Mono) yok — `pip install -U matplotlib` ya da `family=` satırını kaldır |

---

## Sonraki ders (Ders 23)
**Cohn makaleleri, abstract düzeyinde: 2401.03991 (AAAI-24), 2405.15064 (IJCAI-24), 2406.16528 (COSIT-24), 2410.03492, 2507.12059 (QR-25), 2511.02627 (TMLR), 2605.18380.** Danışmanın Hallac'a gönderdiği yedi makale; her biri için iki cümle — ne soruyor, bizim LIBERO/Türkçe eksenine hangi ilişki türü ya da değerlendirme fikriyle dokunuyor. Bugünkü dersin bir ucu oraya uzanıyor: Cohn'un LLM'lerde niteliksel mekânsal akıl yürütme çalışmaları "dil ≠ dünya" tezinin metin tarafı; bizim bugün gördüğümüz şey ise dilin daha modele ulaşmadan, token tarafında nasıl inceldiği.

---

## Kaynaklar
- OpenVLA: Kim vd. (2024). *OpenVLA: An Open-Source Vision-Language-Action Model.* [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) — §3.2: 256 bin, 1.–99. yüzdelik; "overwriting the 256 least used tokens in the Llama tokenizer's vocabulary (which corresponds to the last 256 tokens)"; Llama tokenizer'ın yalnızca 100 özel token ayırması.
- LLaMA-2: Touvron vd. (2023). *Llama 2: Open Foundation and Fine-Tuned Chat Models.* [arXiv:2307.09288](https://arxiv.org/abs/2307.09288) — tokenizer (SentencePiece BPE, 32k, rakamlar tek tek, bayt fallback), Tablo 10 dil dağılımı (İngilizce %89,70, bilinmeyen %8,38; Türkçe %0,005 eşiğinin üstünde listelenmiyor — tabloyu kendin doğrula). Tokenizer dosyası: [karpathy/llama2.c `tokenizer.model`](https://github.com/karpathy/llama2.c) (sha256 `9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347`, Meta checklist ile aynı).
- Qwen3: Yang vd. (2025). *Qwen3 Technical Report.* [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) — byte-level BPE, vocab 151.669; 36T token, 119 dil ve lehçe; Tablo 36 Belebele Turkic: `kaz_Cyrl, azj_Latn, tur_Latn, uzn_Latn`. Qwen3-VL: [arXiv:2511.21631](https://arxiv.org/abs/2511.21631) — Qwen3 omurgası, 2B/4B/8B/32B + MoE, SigLIP2 (2B/4B'de Large-300M), 2×2 görsel birleştirme, 39 dilde OCR. Tokenizer: [huggingface.co/Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) (bu oturumda erişilemedi; dosya npm `@lenml/tokenizer-qwen3@3.7.2`'den, bilinen id'lerle doğrulandı — kimlik hipotezi görevde kapanır).
- Diller arası token eşitsizliği: Petrov vd. (2023). *Language Model Tokenizers Introduce Unfairness Between Languages.* NeurIPS 2023, [arXiv:2305.15425](https://arxiv.org/abs/2305.15425) — aynı metnin diller arası token uzunluğu 15 kata kadar açılıyor; çok dilli tokenizer'larda da sürüyor.
- Kütüphaneler: [google/sentencepiece](https://github.com/google/sentencepiece) (0.2.3; `return_type="proto"` ile bayt offset'leri), [huggingface/tokenizers](https://github.com/huggingface/tokenizers) (Rust; `offsets`), [belladoreai/llama-tokenizer-js](https://github.com/belladoreai/llama-tokenizer-js) (1.2.2; Ders 14–15'teki sayım — bugün 100/100 SentencePiece ile aynı).
- Önceki dersler: Ders 5 (Tatar vd., `Kal|dır`), Ders 13 (VLANeXt chat şablonu, 64 görüntü token'ı/kare, policy başı), Ders 14 (OpenVLA prompt'u, 29871 = `▁`), Ders 15 (ilk token sayımı; `k|ald|ı|r` hipotezi; Dong vd. ko 49,2), Ders 16 (`İ`.lower() tuzağı BDDL'de), Ders 19 (B dalı: "Türkçe okunmuyor"), Ders 20 (`datasets/04`, `rel_seg`, minimal çiftler).
- Veri: `datasets/05-tokenization-tr-commands/` (kart + `tokens.csv`, `rel_span.csv`, `seg_align.csv`, `words.csv`, `suffix_forms.csv`, `action_collision.csv`, `crosscheck.md`, `probe_edge_cases.txt`, `summary.md`). Kod: `code/day22/` — `tokenize_set.py`, `count_js_ids.mjs`, `probe_edge_cases.py`, `plot_day22.py` (hepsi çalıştırıldı). Görseller: 01 ve 05 Mermaid (`.mmd` yanında); 02–04 `plot_day22.py` — **gerçek tokenizer çıktısı**, temsili veri yok.
