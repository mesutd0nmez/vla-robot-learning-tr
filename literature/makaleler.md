# Makale listesi: 15 hakemli çalışma ve 2 ön baskı

Bu bölüm, Vision-Language-Action (VLA) modelleri ve çok dilli (multilingual) VLA üzerine önceden taranıp seçilmiş 15 hakemli makaleyi listeler. Künyeler ve özetler 21 Kasım 2026'da her makalenin kendi yayın sayfasından okundu; bir bilgi sayfada yoksa "doğrulanamadı" diye işaretlendi. Bu makalelere dayanan iki sayfalık değerlendirme: [Rapor](rapor.md).

**Ana bulgu:** 15 makaleden yalnızca biri (1 numara) çok dilli VLA'yı doğrudan konu alır. Diğer 14 makalenin özetinde İngilizce dışı dillerden söz edilmez; aşağıdaki "çok dillilikle ilgisi" satırları bunu makale makale gösterir.

| Grup | Makaleler |
|---|---|
| (a) Çok dilli VLA | 1 |
| (b) Temel VLA modelleri | 2, 3, 4, 5 |
| (c) Verimlilik ve ince ayar | 6, 7, 8, 14 |
| (d) Akıl yürütme ve yeni modaliteler | 9, 10, 11, 12, 13 |
| (e) Derleme | 15 |

---

## (a) Çok dilli VLA

### 1. When Does Language Matter? Multilingual Instructions Reveal Step-wise Language Sensitivity in Vision-Language-Action Models
- **Yazarlar:** Xuan Dong, Zhe Han, Tianhao Niu, Qingfu Zhu, Wanxiang Che
- **Mecra ve yıl:** ACL 2026 (64th Annual Meeting of the ACL, Volume 1: Long Papers), s. 44615–44629, Temmuz 2026
- **Bağlantı:** [aclanthology.org/2026.acl-long.2066](https://aclanthology.org/2026.acl-long.2066/) · DOI [10.18653/v1/2026.acl-long.2066](https://doi.org/10.18653/v1/2026.acl-long.2066)
- **Özet:** LIBERO komutları Google Translate ile dokuz dile çevrilir (İngilizce ile birlikte on dil) ve OpenVLA-OFT ile π0.5 bu komutlarla değerlendirilir; İngilizce dışı komutlarda ortalama başarı 32–46 puan düşer. Yazarlar dilin etkisinin adımlara eşit dağılmadığını gösterir ve yalnızca "dil-kritik" adımlarda gizli temsilleri İngilizce referanslara hizalayan, yeniden eğitim gerektirmeyen bir müdahale önerir.
- **Çok dillilikle ilgisi:** Ana konu. Diller: Çince, Japonca, Korece, Fransızca, İspanyolca, Arapça, Tayca, Portekizce, Vietnamca. Türkçe yok.

Makalenin Tablo 1 ve Tablo 2'sinden ortalama başarı oranları (%; dört LIBERO suite'inin ortalaması):

| Model | İngilizce | Diğer 9 dil (aralık) | Adım bazlı müdahaleyle |
|---|---|---|---|
| OpenVLA-OFT | 97,1 | 50,8–65,3 | 62,6–70,9 |
| π0.5 | 96,9 | 55,4–61,2 | 79,8–82,1 |

En ağır kayıp Goal suite'indedir: İngilizce 97,9 ve 97,6 iken diğer dillerde 6,4–16,8. Çeviriler insan kontrolü ve geri çeviriyle denetlenmiştir. Yazarlar ayrıca OpenVLA-OFT'yi Çince komutlarla ince ayarlar (Tablo 4): ortalama 51,7'den 86,8'e çıkar; yalnızca dil-kritik adımlarla 85,5, rastgele seçilmiş adımlarla 44,9 elde edilir. Tüm deneyler simülasyondadır; yazarlar gerçek robotlara aktarımı açık soru olarak bırakır.

---

## (b) Temel VLA modelleri

### 2. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
- **Yazarlar:** Brianna Zitkovich, Tianhe Yu, Sichun Xu vd. (54 yazar; tam liste yayın sayfasında)
- **Mecra ve yıl:** CoRL 2023 (Proceedings of The 7th Conference on Robot Learning), PMLR 229:2165–2183, 2023
- **Bağlantı:** [proceedings.mlr.press/v229/zitkovich23a.html](https://proceedings.mlr.press/v229/zitkovich23a.html)
- **Özet:** Vision-language modelleri, robot yörüngeleri ve web ölçeğindeki görsel-dil görevleriyle birlikte ince ayarlanır; robot action'ları metin token'ı olarak ifade edilir. Özet bu model sınıfını "vision-language-action models (VLA)" diye adlandırır ve yaklaşık 6 bin deneme ile genellemenin iyileştiğini bildirir.
- **Çok dillilikle ilgisi:** Ana konu değil. Özette İngilizce dışı dillerden söz edilmiyor.

### 3. OpenVLA: An Open-Source Vision-Language-Action Model
- **Yazarlar:** Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti vd. (17 yazar)
- **Mecra ve yıl:** CoRL 2024 (Proceedings of The 8th Conference on Robot Learning), PMLR 270:2679–2713, 2025
- **Bağlantı:** [proceedings.mlr.press/v270/kim25c.html](https://proceedings.mlr.press/v270/kim25c.html) · [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)
- **Özet:** 970 bin robot gösterimiyle eğitilmiş, 7 milyar parametreli açık kaynak bir VLA; Llama 2 dil modelini DINOv2 ve SigLIP görsel özellikleriyle birleştirir. Özet, 55 milyar parametreli RT-2-X'i mutlak başarıda 16,5 puan geçtiğini ve LoRA ile tüketici GPU'larında ince ayarlanabildiğini bildirir.
- **Çok dillilikle ilgisi:** Yok. Özette İngilizce dışı dillerden söz edilmiyor.

### 4. π0: A Vision-Language-Action Flow Model for General Robot Control
- **Yazarlar:** Kevin Black, Noah Brown, Danny Driess vd. (25 yazar)
- **Mecra ve yıl:** RSS 2025 (Robotics: Science and Systems XXI), Haziran 2025
- **Bağlantı:** [roboticsproceedings.org/rss21/p010.html](https://www.roboticsproceedings.org/rss21/p010.html) · DOI [10.15607/RSS.2025.XXI.010](https://doi.org/10.15607/RSS.2025.XXI.010)
- **Özet:** Önceden eğitilmiş bir vision-language modelinin üzerine bir flow matching mimarisi kurar; böylece modelin internet ölçeğindeki anlamsal bilgisini devralır. Birden çok robot platformunun verisiyle eğitilir; çamaşır katlama, masa temizleme ve kutu montajı gibi görevlerde değerlendirilir.
- **Çok dillilikle ilgisi:** Yok. Özette İngilizce dışı dillerden söz edilmiyor.

### 5. π0.5: a Vision-Language-Action Model with Open-World Generalization
- **Yazarlar:** Kevin Black, Noah Brown, James Darpinian vd. (35 yazar)
- **Mecra ve yıl:** CoRL 2025 (Proceedings of The 9th Conference on Robot Learning), PMLR 305:17–40, 2025
- **Bağlantı:** [proceedings.mlr.press/v305/black25a.html](https://proceedings.mlr.press/v305/black25a.html)
- **Özet:** π0 üzerine kurulur; farklı robotlar, üst düzey anlamsal tahmin ve web verisi gibi heterojen kaynaklarla birlikte eğitim (co-training) kullanır. Daha önce görmediği evlerde mutfak ya da yatak odası toplamak gibi uzun görevleri yerine getirdiği bildirilir.
- **Çok dillilikle ilgisi:** Ana konu değil. Özette İngilizce dışı dillerden söz edilmiyor; 1 numaralı makalede çok dilli komutlarla test edilen iki modelden biridir.

---

## (c) Verimlilik ve ince ayar

### 6. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (OpenVLA-OFT)
- **Yazarlar:** Moo Jin Kim, Chelsea Finn, Percy Liang
- **Mecra ve yıl:** RSS 2025 (Robotics: Science and Systems XXI), Haziran 2025
- **Bağlantı:** [roboticsproceedings.org/rss21/p017.html](https://www.roboticsproceedings.org/rss21/p017.html) · DOI [10.15607/RSS.2025.XXI.017](https://doi.org/10.15607/RSS.2025.XXI.017)
- **Özet:** OpenVLA'yı temel alıp ince ayarın tasarım kararlarını (action decoding, action temsili, öğrenme hedefi) inceler ve Optimized Fine-Tuning (OFT) tarifini çıkarır. OpenVLA-OFT, LIBERO ortalamasını %76,5'ten %97,1'e yükseltir ve action üretim hızını 26 kat artırır.
- **Çok dillilikle ilgisi:** Kendi içinde yok. 1 numaralı makalede çok dilli komutlarla test edilen iki modelden biridir.

### 7. FAST: Efficient Action Tokenization for Vision-Language-Action Models
- **Yazarlar:** Karl Pertsch, Kyle Stachowicz, Brian Ichter vd. (9 yazar)
- **Mecra ve yıl:** RSS 2025 (Robotics: Science and Systems XXI), Haziran 2025
- **Bağlantı:** [roboticsproceedings.org/rss21/p012.html](https://www.roboticsproceedings.org/rss21/p012.html) · DOI [10.15607/RSS.2025.XXI.012](https://doi.org/10.15607/RSS.2025.XXI.012) · [arXiv:2501.09747](https://arxiv.org/abs/2501.09747)
- **Özet:** Boyut ve zaman adımı başına kutulama yapan action tokenizer'ların yüksek frekanslı, becerikli görevlerde zayıf kaldığını savunur ve DCT tabanlı bir sıkıştırma tokenizer'ı (FAST) önerir. 1 milyon yörüngeyle eğitilmiş evrensel FAST+ tokenizer'ını yayımlar; eğitim süresinin 5 kata kadar kısaldığını bildirir.
- **Çok dillilikle ilgisi:** Yok. Özette İngilizce dışı dillerden söz edilmiyor.

### 8. TinyVLA: Toward Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation
- **Yazarlar:** Junjie Wen, Yichen Zhu, Jinming Li vd. (dergi kaydında 13 yazar)
- **Mecra ve yıl:** IEEE Robotics and Automation Letters, cilt 10, sayı 4, s. 3988–3995, Nisan 2025
- **Bağlantı:** DOI [10.1109/LRA.2025.3544909](https://doi.org/10.1109/LRA.2025.3544909) · [arXiv:2409.12514](https://arxiv.org/abs/2409.12514)
- **Özet:** Büyük ölçekli robot verisiyle ön eğitim gerektirmeyen, daha hızlı çıkarım yapan kompakt bir VLA ailesi sunar. Önceden eğitilmiş çok modlu bir omurgaya ince ayar sırasında diffusion policy decoder ekler; hız ve veri verimliliğinde OpenVLA'yı geçtiğini bildirir.
- **Çok dillilikle ilgisi:** Yok. Özette İngilizce dışı dillerden söz edilmiyor.

### 14. Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment
- **Yazarlar:** Tao Lin, Yilei Zhong, Yuxin Du vd. (14 yazar)
- **Mecra ve yıl:** CVPR 2026 (Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition), s. 13397–13406
- **Bağlantı:** [openaccess.thecvf.com — CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Lin_Evo-1_Lightweight_Vision-Language-Action_Model_with_Preserved_Semantic_Alignment_CVPR_2026_paper.html) · [arXiv:2511.04555](https://arxiv.org/abs/2511.04555)
- **Özet:** Robot verisiyle ön eğitim olmadan çalışan, 0,77 milyar parametreli hafif bir VLA; action öğrenimini algıyla aşamalı hizalayan iki aşamalı eğitimle VLM temsillerini korur. Özet, Meta-World ve RoboTwin'de önceki en iyi sonuçları geçtiğini, LIBERO'da %94,8 ve gerçek robotta %78 başarı elde ettiğini bildirir.
- **Çok dillilikle ilgisi:** Yok. Özette İngilizce dışı dillerden söz edilmiyor.

---

## (d) Akıl yürütme ve yeni modaliteler

### 9. VLAS: Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation
- **Yazarlar:** Wei Zhao, Pengxiang Ding, Min Zhang, Zhefei Gong, Shuanghao Bai, Han Zhao, Donglin Wang
- **Mecra ve yıl:** ICLR 2025 (poster)
- **Bağlantı:** [iclr.cc/virtual/2025/poster/30076](https://iclr.cc/virtual/2025/poster/30076)
- **Özet:** Konuşmayı ayrı bir konuşma tanıma katmanı olmadan doğrudan girdi alan uçtan uca bir VLA; böylece ses izi gibi anlam dışı ipuçları korunur. Üç aşamalı konuşma komutu eğitimi ve kişiselleştirilmiş görevler için sesli retrieval-augmented generation kullanır.
- **Çok dillilikle ilgisi:** Dolaylı: girdi modalitesi metin yerine konuşmadır. Konuşma komutlarının hangi dilde olduğu yayın sayfasında yazmıyor (doğrulanamadı).

### 10. CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models
- **Yazarlar:** Qingqing Zhao, Yao Lu, Moo Jin Kim vd. (15 yazar)
- **Mecra ve yıl:** CVPR 2025, s. 1702–1713
- **Bağlantı:** [openaccess.thecvf.com — CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_CoT-VLA_Visual_Chain-of-Thought_Reasoning_for_Vision-Language-Action_Models_CVPR_2025_paper.html)
- **Özet:** Action üretmeden önce gelecekteki görüntü karelerini görsel hedef olarak tahmin eden, 7 milyar parametreli bir VLA. Özet, önceki en iyi VLA'yı gerçek robot görevlerinde %17, simülasyonda %6 geçtiğini bildirir.
- **Çok dillilikle ilgisi:** Yok. Özette İngilizce dışı dillerden söz edilmiyor.

### 11. Vision-Language-Action Instruction Tuning: From Understanding to Manipulation (InstructVLA)
- **Yazarlar:** Shuai Yang, Hao Li, Bin Wang, Yilun Chen, Yang Tian, Tai Wang, Hanqing Wang, Feng Zhao, Yiyi Liao, Jiangmiao Pang
- **Mecra ve yıl:** ICLR 2026
- **Bağlantı:** [proceedings.iclr.cc — özet sayfası](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f81332c0e757b169c66d436ec2fff624-Abstract-Conference.html) · [PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/f81332c0e757b169c66d436ec2fff624-Paper-Conference.pdf)
- **Özet:** InstructVLA modelini ve Vision-Language-Action Instruction Tuning (VLA-IT) eğitim yaklaşımını sunar: standart VLM verisi ile 650 bin örneklik VLA-IT veri seti birlikte kullanılır, böylece VLM'in akıl yürütme yeteneği action eğitimi sırasında korunur. 80 görevlik SimplerEnv-Instruct benchmark'ını tanıtır.
- **Çok dillilikle ilgisi:** Yok. Özette İngilizce dışı dillerden söz edilmiyor.

### 12. 3D-VLA: A 3D Vision-Language-Action Generative World Model
- **Yazarlar:** Haoyu Zhen, Xiaowen Qiu, Peihao Chen, Jincheng Yang, Xin Yan, Yilun Du, Yining Hong, Chuang Gan
- **Mecra ve yıl:** ICML 2024 (Proceedings of the 41st International Conference on Machine Learning), PMLR 235:61229–61245, 2024
- **Bağlantı:** [proceedings.mlr.press/v235/zhen24a.html](https://proceedings.mlr.press/v235/zhen24a.html)
- **Özet:** 3B tabanlı bir dil modelinin üzerine kurulur; 3B algı, akıl yürütme ve action'ı üretken bir dünya modeliyle birbirine bağlar. Hedef görüntüleri ve nokta bulutlarını tahmin eden diffusion modelleri eğitilip dil modeline hizalanır.
- **Çok dillilikle ilgisi:** Yok. Özette İngilizce dışı dillerden söz edilmiyor.

### 13. QUAR-VLA: Vision-Language-Action Model for Quadruped Robots
- **Yazarlar:** Pengxiang Ding, Han Zhao, Wenjie Zhang, Wenxuan Song, Min Zhang, Siteng Huang, Ningxi Yang, Donglin Wang
- **Mecra ve yıl:** ECCV 2024 (Computer Vision – ECCV 2024, LNCS 15063), s. 352–367, Springer
- **Bağlantı:** DOI [10.1007/978-3-031-72652-1_21](https://doi.org/10.1007/978-3-031-72652-1_21) · [arXiv:2312.14457](https://arxiv.org/abs/2312.14457)
- **Özet:** Dört ayaklı robotlar için görsel bilgi ve komutlardan çalıştırılabilir action üreten QUART modelini ve algı, navigasyon ile tüm gövde manipülasyonu görevlerini içeren QUARD veri setini sunar.
- **Çok dillilikle ilgisi:** Yok. Özette İngilizce dışı dillerden söz edilmiyor.

---

## (e) Derleme

### 15. Sensing the Action: Rethinking Sensor Modalities and Multi-Modal Fusion in Vision–Language–Action Models for Robotic Manipulation
- **Yazarlar:** Byoung Chul Ko
- **Mecra ve yıl:** Sensors, 26(11), 3541, 2026 (derleme makalesi; 3 Haziran 2026)
- **Bağlantı:** [mdpi.com/1424-8220/26/11/3541](https://www.mdpi.com/1424-8220/26/11/3541) · DOI [10.3390/s26113541](https://doi.org/10.3390/s26113541)
- **Özet:** VLA'yı sensör, füzyon ve action hattı olarak yeniden ele alır; RGB, derinlik, dokunma, kuvvet/tork gibi sensör modalitelerini ve füzyon tasarımlarını sınıflandırır. Mevcut benchmark'ların RGB ağırlıklı girdilere ve tek bir başarı oranı metriğine yaslandığını saptar.
- **Çok dillilikle ilgisi:** Yok. Özette İngilizce dışı dillerden söz edilmiyor.

---

## İlgili ön baskılar

Aşağıdaki iki çalışma **hakemli değildir**: arXiv ön baskısıdır ve 21 Kasım 2026 itibarıyla arXiv sayfalarında bir yayın mecrası belirtilmemiştir. Yukarıdaki 15 makalelik listeye dahil değildir.

### Beyond English: Uncovering the Multilingual Gap in Vision-Language-Action Models
- **Yazarlar:** Hanyang Chen, Hongliang Li, Jiarui Cao, Yang Li, Yang Jiang, Haonan Wen, Kaiyu Huang, Shengnan Guo, Huaiyu Wan
- **Durum:** arXiv ön baskısı, 14 Haziran 2026 (v1)
- **Bağlantı:** [arXiv:2606.15714](https://arxiv.org/abs/2606.15714)
- **Özet:** Ağırlıklı olarak İngilizce komutlarla eğitilen VLA'ların, dil omurgası çok dilli olsa bile diğer dillerde belirgin biçimde zayıfladığını bildirir. Çok dilli temsilleri hizalayan bir ince ayar yöntemi (Multilingual Principal Component Alignment) önerir.
- **Çok dillilikle ilgisi:** Ana konu.

### Measuring Language Transfer in Robot Policies: Adding Greek to a Cosmos3 Vision-Language-Action Policy
- **Yazarlar:** Ayoub Kirouane, Georgios Giaples, Christos Petrocheilos
- **Durum:** arXiv ön baskısı, 7 Eylül 2026 (v1)
- **Bağlantı:** [arXiv:2609.07470](https://arxiv.org/abs/2609.07470)
- **Özet:** Açık bir VLA politikasına yalnızca makineyle yeniden yazılmış komutlarla Yunanca ekler. Özete göre iki dilli eğitim kontrol koşulunun 6,7–7,1 puan üstüne çıkar ve İngilizce performansın yaklaşık beşte ikisine ulaşır; yazarlar bir metriğe güvenmeden önce yanlış komutla ölçülen bir taban gerektiğini vurgular.
- **Çok dillilikle ilgisi:** Ana konu (Yunanca).
