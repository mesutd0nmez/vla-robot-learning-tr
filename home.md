<div class="home-hero">

# Robot Learning ve VLA — tez günlüğü

Bilgisayar Mühendisliği yüksek lisans tezimin açık kaydı: Vision-Language-Action modelleri, LIBERO benchmark'ı ve Türkçe mekânsal komutların robot davranışına bağlanması. Günde bir saat, her ders bir konu; notlar Türkçe ve herkesin kullanımına açık.

</div>

## Dersler

<div class="home-grid">
<a href="#/lectures/lecture-01"><b>01</b><span>Robot learning döngüsü ve kaynak haritası<small>Faz 1 · Kavramlar</small></span></a>
<a href="#/lectures/lecture-02"><b>02</b><span>OpenVLA'nın içi: görüntü + dil → action<small>Faz 1 · Kavramlar</small></span></a>
<a href="#/lectures/lecture-03"><b>03</b><span>Benchmark ≠ simülatör: LIBERO, MuJoCo, LIBERO-Plus<small>Faz 1 · Kavramlar</small></span></a>
<a href="#/lectures/lecture-04"><b>04</b><span>Language grounding ve shortcut learning<small>Faz 1 · Kavramlar</small></span></a>
<a href="#/lectures/lecture-05"><b>05</b><span>Makale derin okuma: Tatar vd. (2026)<small>Faz 1 · Kavramlar</small></span></a>
<a href="#/lectures/lecture-06"><b>06</b><span>Mekânsal akıl yürütmenin temelleri<small>Faz 1 · Kavramlar</small></span></a>
<a href="#/lectures/lecture-07"><b>07</b><span>RoomSpace → LIBERO: Türkçe komut seti<small>Faz 1 · Kavramlar</small></span></a>
<a href="#/lectures/lecture-08"><b>08</b><span>Hafta 1 sentezi ve VLA model aileleri<small>Faz 1 · Kavramlar</small></span></a>
<a href="#/lectures/lecture-09"><b>09</b><span>Mac'te conda, MuJoCo ve ilk sahne<small>Faz 2 · Klavye</small></span></a>
<a href="#/lectures/lecture-10"><b>10</b><span>LIBERO kurulumu ve ilk episode<small>Faz 2 · Klavye</small></span></a>
<a href="#/lectures/lecture-11"><b>11</b><span>LIBERO demo verisi: HDF5'in içi<small>Faz 2 · Klavye</small></span></a>
<a href="#/lectures/lecture-12"><b>12</b><span>Colab T4'te VLANeXt: ilk success rate<small>Faz 2 · Klavye</small></span></a>
<a href="#/lectures/lecture-13"><b>13</b><span>Kod okuma: komut → action → success<small>Faz 2 · Kod okuma</small></span></a>
<a href="#/lectures/lecture-14"><b>14</b><span>OpenVLA-7B Mac'te: görüntü + komut → 7 token → action<small>Faz 2 · Klavye</small></span></a>
<a href="#/lectures/lecture-15"><b>15</b><span>Tatar mini-reprodüksiyonu: 5 kare × 6 koşul, L2 + cosine<small>Faz 2 · Klavye</small></span></a>
<a href="#/lectures/lecture-16"><b>16</b><span>Faz 2 sentezi + LIBERO-Plus: ayrı runtime, komutun kod yolu, boş komut<small>Faz 2 · Sentez + Klavye</small></span></a>
<a href="#/lectures/lecture-17"><b>17</b><span>LIBERO-Plus'ın yedi bozulması kodda: üç taşıyıcı, kamera, 500 Panda, R1/R2, Türkçe varyant<small>Faz 3 · Kod okuma + Klavye</small></span></a>
<a href="#/lectures/lecture-18"><b>18</b><span>Aynı görev, dört komut: clean / empty / wrong / tr × 10 init state — deney tasarımı, yama, Wilson ve güç<small>Faz 3 · Deney + Klavye</small></span></a>
<a href="#/lectures/lecture-19"><b>19</b><span>Sonuç analizi: shortcut var mı, hangi suite'te? — okuma protokolü, BDDL'den yapısal üst sınır (Spatial 9/10, Goal 1/10), 50 init state'in geometrisi<small>Faz 3 · Analiz + Klavye</small></span></a>
<a href="#/lectures/lecture-20"><b>20</b><span>Türkçe mekânsal komut seti: 20 EN–TR çift — ilişki adı + üç ek, "soldaki" kime göre (üç çerçeve), 50 init state'ten referans, karşı-olgusal BDDL<small>Faz 3 · Tasarım + Veri</small></span></a>
<a href="#/lectures/lecture-21"><b>21</b><span>Webots: danışmanın ikinci simülatörü — yığındaki yeri (benchmark değil, simülatör), Mac'te kurulum + conda vla, UR10e ile ilk sahne, tez için karar notu<small>Faz 3 · Araç + Karar</small></span></a>
<a href="#/lectures/lecture-22"><b>22</b><span>Tokenizasyon analizi: aynı 100 komut, iki tokenizer — LLaMA-2 1,6× / Qwen3 1,4× şişme, ilişki eki kararsız, "soldaki" = sold+aki, cümle başı tuzağı, action aralığı temiz<small>Faz 3 · Analiz + Veri</small></span></a>
<a href="#/lectures/lecture-23"><b>23</b><span>Cohn hattı: yedi makale, abstract düzeyinde — StepGame düzeltmesi, RoomSpace, kardinal yön (2024→2025), skor belirsizliği (PI), DecompSR, QSTRBench; derinlikle çöküş (LIBERO'da k=1), yüzey değişince düşüş, tek skor rapor etme<small>Faz 3 · Literatür + Metodoloji</small></span></a>
<a href="#/lectures/lecture-24"><b>24</b><span>Çözüm yaklaşımları: MT çeviri / parser + sembolik / çok dilli omurga / Türkçe LoRA / hidden-state hizalama — Dong & Chen (Haz 2026) sayılarıyla artı-eksi, Mac–T4 bütçesi, Ders 18 örüntüsüne göre karar ağacı; parser gerçekten koştu (set içi 79/80, dışı 5/12)<small>Faz 3 · Karar + Klavye</small></span></a>
<a href="#/lectures/lecture-25"><b>25</b><span>Faz 3 sentezi: sekiz ders tek sayfada — bilanço 21 ölçüm = 11 gerçek / 6 Mesut'ta / 4 hipotez; phase3_smoke.py 30 script 28 OK (Ders 22 tokens.csv birebir yeniden üretildi); LangGap (aynı sahne, Spatial 97 → 5,9); açık sorular S1–S10 → üç RQ tohumu<small>Faz 3 · Sentez</small></span></a>
<a href="#/lectures/lecture-26"><b>26</b><span>Üç aday araştırma sorusu: tohumdan sınanabilir soruya — 6 alan (null + δ, değişkenler, sayısal eşik, suite+blok+n, sistemler, evet/hayır); rq_power.py: n=50 ancak ≥23 puanı yakalar, TOST δ=15 → n≈25; RQ-a/b/c, S1'e göre koşullu sıralama; thesis/rq-candidates.md<small>Faz 4 · RQ + Ön kayıt</small></span></a>
</div>

## VLA ve Çok Dilli VLA Literatürü

<div class="home-grid">
<a href="#/literature/makaleler"><b>01</b><span>Makale listesi: 15 hakemli çalışma ve 2 ön baskı<small>Literatür · Liste</small></span></a>
<a href="#/literature/rapor"><b>02</b><span>Rapor: VLA ve çok dilli VLA'da mevcut durum<small>Literatür · 2 sayfa</small></span></a>
</div>

## Nasıl çalışıyorum

Her ders aynı iskelette: 35 dakikalık okuma (görsellerle) ve 15 dakikalık bir görev. Plan ve günlük ilerleme `Plan` bölümünde; kurulum ve deney script'leri `Kod` bölümünde.

## Aşamalar

| Aşama | Kapsam | Durum |
|---|---|---|
| 0. Hızlandırılmış giriş (Ders 1–30) | Kavramlar → ortam kurulumu → ilk deneyler → proje üzerine akıl yürütme | Ders 1–26 hazır |
| 1. Literatür ve konu seçimi | Tez ekseninin danışmanla netleştirilmesi, kaynak taraması | — |
| 2. Deney altyapısı | Benchmark, model, metrik ve veri setinin kurulması | — |
| 3. Deneyler | Ana deneyler, ablasyonlar, sonuç analizi | — |
| 4. Tez yazımı ve savunma | Tez metni, makale, sunum | — |

Kaynak kodu ve ham markdown: [github.com/mesutd0nmez/vla-robot-learning-tr](https://github.com/mesutd0nmez/vla-robot-learning-tr)
