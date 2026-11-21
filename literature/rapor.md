# Rapor: VLA ve çok dilli VLA'da mevcut durum

**Kapsam:** 15 hakemli makale ([makale listesi](makaleler.md)) · **Tarih:** 21 Kasım 2026 · Köşeli parantezli numaralar kaynakçaya ve makale listesine karşılık gelir.

## 1. VLA nedir, neden önemli?

Vision-Language-Action (VLA) modeli, kamera görüntüsünü ve doğal dil komutunu girdi alıp doğrudan robot action'ı üreten tek bir politikadır. Önemi, internet ölçeğinde eğitilmiş vision-language modellerinin (VLM) bilgisini robot kontrolüne taşımasındadır. Terimi ortaya koyan RT-2, action'ları metin token'ı olarak ifade edip modeli robot ve web verisiyle birlikte eğitmiş, bunun yeni nesnelere genellemeyi iyileştirdiğini göstermiştir [2]. Komut arayüzü dil olduğu için, modelin hangi dilleri gerçekten anladığı doğrudan bir kullanılabilirlik sorusudur.

## 2. Alanın gelişimi

RT-2'nin ardından OpenVLA, 970 bin robot gösterimiyle eğitilmiş, 7 milyar parametreli açık kaynak bir model sundu; Llama 2 üzerine kuruludur ve tüketici GPU'larında ince ayarlanabilir [3]. π0, önceden eğitilmiş bir VLM'in üzerine flow matching mimarisi kurar [4]. π0.5 ise heterojen verilerle birlikte eğitim sayesinde daha önce görmediği evlerde uzun görevleri yerine getirir [5]. Paralel bir hat verimliliğe odaklanır: OpenVLA-OFT ince ayar tarifini değiştirerek LIBERO ortalamasını %76,5'ten %97,1'e çıkarır [6], FAST sıkıştırma tabanlı bir action tokenizer'ı önerir [7], TinyVLA [8] ve Evo-1 [14] robot verisiyle büyük ön eğitim gerektirmeyen küçük modeller hedefler. Diğer çalışmalar yeni girdi ve akıl yürütme biçimleri ekler: konuşma komutu [9], görsel düşünce zinciri [10], VLM yeteneklerini koruyan eğitim [11], 3B dünya modeli [12] ve dört ayaklı robotlar [13]. Ko'nun derlemesi, mevcut benchmark'ların RGB ağırlıklı girdilere ve tek bir başarı oranı metriğine yaslandığını saptar [15].

## 3. Çok dilli VLA'da mevcut durum

On beş makaleden yalnızca biri çok dilliliği konu alır [1]. Dong vd., LIBERO komutlarını Google Translate ile dokuz dile çevirir (Çince, Japonca, Korece, Fransızca, İspanyolca, Arapça, Tayca, Portekizce, Vietnamca; İngilizce ile on dil) ve çevirileri insan kontrolü ile geri çeviriyle denetler. Test edilen iki model OpenVLA-OFT ve π0.5'tir. İngilizce ortalama başarı %97,1 ve %96,9 iken diğer dillerde OpenVLA-OFT %50,8–65,3, π0.5 %55,4–61,2 aralığına düşer. Kayıp 32–46 puandır ve en ağır hâliyle Goal görevlerinde görülür (%6–17). İkinci bulgu, dilin etkisinin adımlara eşit dağılmadığıdır: az sayıda "dil-kritik" adım başarısızlığı belirler. Yazarlar bu adımlarda gizli temsilleri İngilizce referanslara hizalayan, yeniden eğitim gerektirmeyen bir müdahale önerir. Ortalama başarı OpenVLA-OFT'de %62,6–70,9'a, π0.5'te %79,8–82,1'e çıkar, ancak İngilizce düzeyine ulaşmaz. Tüm deneyler simülasyondadır.

## 4. Literatür boşluğu ve açık araştırma soruları

Boşluk açıktır: hakemli mecrada doğrudan çok dilli VLA'yı inceleyen tek çalışma [1]'dir ve diğer on dört makalenin özetinde İngilizce dışı dillerden söz edilmez. Aynı yöne işaret eden iki ön baskı vardır (arXiv:2606.15714, arXiv:2609.07470), ancak henüz hakem denetiminden geçmemişlerdir. Tek hakemli çalışma da sınırlıdır: iki model, tek benchmark, makine çevirisi ve dokuz dil. Türkçe bunların arasında yoktur. Bu yüzden boşluğu "alan çok dilliliği çözemedi" diye değil, "henüz yeterince ölçmedi" diye okumak gerekir. Açık sorular şunlardır. Bulgular gerçek robotlarda ve LIBERO dışındaki benchmark'larda geçerli mi? Türkçe gibi sondan eklemeli diller aynı örüntüyü gösterir mi? Kayıp dilin kendisinden mi, çeviri yüzeyinden mi, yoksa modelin komutu zaten az kullanmasından mı kaynaklanır? Bunu ayırmak için boş ve yanlış komut gibi kontrol koşulları gerekir. Farklı dil omurgaları farklı mı davranır? Kalıcı çözüm çıkarım zamanı müdahale mi, yoksa çok dilli eğitim mi? Bu sorular, bu projenin Türkçe mekânsal komut eksenini doğrudan gerekçelendirir.

## Kaynakça

- [1] Dong, X., Han, Z., Niu, T., Zhu, Q. ve Che, W. (2026). When Does Language Matter? Multilingual Instructions Reveal Step-wise Language Sensitivity in Vision-Language-Action Models. *ACL 2026*, 44615–44629.
- [2] Zitkovich, B. vd. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. *CoRL 2023*, PMLR 229, 2165–2183.
- [3] Kim, M. J. vd. (2025). OpenVLA: An Open-Source Vision-Language-Action Model. *CoRL 2024*, PMLR 270, 2679–2713.
- [4] Black, K. vd. (2025). π0: A Vision-Language-Action Flow Model for General Robot Control. *RSS 2025*.
- [5] Black, K. vd. (2025). π0.5: a Vision-Language-Action Model with Open-World Generalization. *CoRL 2025*, PMLR 305, 17–40.
- [6] Kim, M. J., Finn, C. ve Liang, P. (2025). Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success. *RSS 2025*.
- [7] Pertsch, K. vd. (2025). FAST: Efficient Action Tokenization for Vision-Language-Action Models. *RSS 2025*.
- [8] Wen, J. vd. (2025). TinyVLA: Toward Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation. *IEEE Robotics and Automation Letters*, 10(4), 3988–3995.
- [9] Zhao, W. vd. (2025). VLAS: Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation. *ICLR 2025*.
- [10] Zhao, Q. vd. (2025). CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models. *CVPR 2025*, 1702–1713.
- [11] Yang, S. vd. (2026). Vision-Language-Action Instruction Tuning: From Understanding to Manipulation. *ICLR 2026*.
- [12] Zhen, H. vd. (2024). 3D-VLA: A 3D Vision-Language-Action Generative World Model. *ICML 2024*, PMLR 235, 61229–61245.
- [13] Ding, P. vd. (2024). QUAR-VLA: Vision-Language-Action Model for Quadruped Robots. *ECCV 2024*, LNCS 15063, 352–367.
- [14] Lin, T. vd. (2026). Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment. *CVPR 2026*, 13397–13406.
- [15] Ko, B. C. (2026). Sensing the Action: Rethinking Sensor Modalities and Multi-Modal Fusion in Vision–Language–Action Models for Robotic Manipulation. *Sensors*, 26(11), 3541.
