# Ders yazım rehberi (otomatik görev ve Claude için)

Bu dosya, `lectures/lecture-NN.md` dosyalarının nasıl yazılacağını tanımlar. Her yeni ders bu kurallara uyar.

## Sıra ve konu
- Yeni dersin numarası = `lectures/` altındaki en büyük `lecture-NN` + 1.
- Konu: `plan/lecture-plan.md` tablosunda aynı numaralı "Gün" satırı. 30'dan sonra: `README.md`'deki aşamalar (literatür → deney altyapısı → deneyler → tez yazımı) ve `plan/daily-log.md`'deki notlar/sorular doğrultusunda, bir önceki dersin "Sonraki ders" bölümünde vaat edilen konu.
- Önceki dersin "Sonraki ders" bölümüne sadık kal; `plan/daily-log.md`'de Mesut'un bıraktığı not/soru varsa dersin başında bir paragrafla cevapla.

## Format (lecture-01…11 ile birebir aynı iskelet)
1. `# Ders N — Başlık` (başlıkta iki noktadan önce kısa ad; menüde o kullanılır)
2. Meta satırı: `**Faz X / Gün N** · **Okuma süresi:** ~40 dk` · ön koşul · kaynak/kod yolu
3. `> **Bugünün tek cümlesi:** …` (tek cümle, anahtar fikir)
4. `---` ile ayrılmış 5–8 numaralı bölüm; her bölümde en az bir somut örnek; toplam okuma ≤ 40 dk, uygulama günlerinde ≤ 60 dk toplam
5. `## N. Yazılımcı için köprüler` (Mesut 10 yıllık C#/.NET backend geliştirici; benzetmeler buradan)
6. `## N. Kendini sına (cevaplar aşağıda)` — 5 açık uçlu soru
7. `## N. Bugünün görevi (15 dk)` — tek somut görev; "çıktı doldurma" bölümü KOYMA
8. `## Çoktan seçmeli (kutuları işaretle, sonra cevabı aç)` — 3 tek seçimli + 2 çok seçimli; şıklar `- [ ] A) …`; cevap `<details><summary>Cevap</summary>\n\n**A.** gerekçe\n\n</details>`
9. Uygulama günlerinde `## Takılırsan` tablosu (belirti → çözüm)
10. `## Sonraki ders (Ders N+1)` — bir paragraf
11. `## Cevaplar` — açık uçlu soruların cevapları
12. `## Kaynaklar` — arXiv/GitHub bağlantılarıyla; görsellerin nasıl üretildiği notu

## Görseller (zorunlu, en az 3)
- Klasör: `lectures/images/lecture-NN/`; derste `![alt](images/lecture-NN/01-ad.png)`
- Diyagramlar Mermaid ile (`mmdc -i x.mmd -o x.png -b white -w 1500 -s 2`), `.mmd` kaynağı yanına konur. Mermaid sequenceDiagram'da özel karakter (→, ≠, parantez içinde ok) kullanma.
- Grafikler matplotlib ile; gerçek makale değerleri kullanılıyorsa kaynağı başlıkta yaz; temsili/sentetik veri ise altında belirt.
- Renk dili: sarı `#ffd166` = model/policy/anahtar fikir, mavi `#cfe8ff` = gözlem/girdi, yeşil `#d4f5d4` = action/çıktı, kırmızı `#ffd6d6` = sorun.
- Çalıştırılabilen her şeyi gerçekten çalıştır (MuJoCo, robosuite, scriptler) ve gerçek çıktıyı görsel olarak kullan; çalıştırılamıyorsa derste açıkça söyle.

## Kod
- Uygulama günlerinde script'ler `code/dayNN/` altına; dersteki komutlar `curl -O https://raw.githubusercontent.com/mesutd0nmez/vla-robot-learning-tr/main/code/dayNN/x.py` ile indirilebilir olmalı.
- Her `code/dayNN/` klasörüne README.md: her script için `## ad.py` + ```python bloğu (site bunu gösterir).
- Donanım: MacBook Pro M2 32 GB (CUDA yok, MPS var) + ücretsiz Colab/Kaggle T4. Mac'te çalışmayan şey (flash-attn, bitsandbytes, VLANeXt eval) Colab'a; Colab notebook'ları `code/dayNN/*.ipynb`.
- Sürüm pinleri: `numpy<2`, `mujoco==2.3.7`, `robosuite==1.4.1`, Python 3.10.

## Dil ve ton
- Türkçe; teknik terimler İngilizce bırakılır (policy, action, success rate). Mesut'a "sen" diye hitap. Pazarlama dili yok. Her iddia kaynaklı; emin olunmayan şey "hipotez" diye işaretlenir.
- Tez ekseni (aday, danışmanla kesinleşmedi): Türkçe mekânsal komutların VLA'larda grounding'i; OpenVLA (LLaMA-2 tokenizer) vs VLANeXt (Qwen3-VL) karşılaştırması; LIBERO/LIBERO-Plus; boş-komut tabanı. Her ders bu eksene en az bir cümleyle bağlanır.

## Bitirirken
1. `plan/daily-log.md` sonuna `## Ders N — Konu` girişi (Durum / Ders / Görev / Notlar).
2. `python tools/gen_sidebar.py`
3. `home.md`'deki ders ızgarasına yeni kart; README'deki ders tablosuna satır.
4. Commit mesajı: `Ders N: <konu>`; push.
