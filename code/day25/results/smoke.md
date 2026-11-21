# Faz 3 kuru koşu — 28 OK / 0 HATA / 2 ATLANDI (toplam 30)

| Ders | Script | Bağımlılık | Sonuç | Süre | Son satır |
|---|---|---|---|---|---|
| 17 | `perturbation_table.py` | LIBERO-plus klonu | OK | 0.1 | / Objects Layout / 385 / BDDL dosyası: `_add_K` (yeni nesne + `add_object_region_K`) ve `_levelL_sampleM` (böl |
| 17 | `camera_geometry.py` | LIBERO-plus klonu | OK | 1.3 | yazıldı: /tmp/faz3smoke_6b2b61hm/cam |
| 17 | `robot_init_spread.py` | LIBERO-plus klonu | OK | 0.9 | yazıldı: /tmp/faz3smoke_6b2b61hm/rob |
| 17 | `layout_shift.py` | LIBERO-plus klonu | OK | 0.9 | yazıldı: /tmp/faz3smoke_6b2b61hm/lay |
| 17 | `classify_rewrites.py` | LIBERO-plus klonu | OK | 0.9 | yazıldı: /tmp/faz3smoke_6b2b61hm/cls |
| 17 | `make_tr_variant.py` | LIBERO-plus klonu + bddl | OK | 0.1 | # Sonra: VLANEXT_TASK_IDS olarak yeni görevlerin 0 tabanlı indeksleri = eski liste uzunluğu, +1, …  (Ders 16:  |
| 18 | `make_conditions.py` | yok | OK | 0.0 | Colab: os.environ["VLANEXT_CONDITIONS"] = open("cond.json").read() |
| 18 | `spec_order_demo.py` | yok | OK | 0.0 | Log'da trial=<init>, condition=<ad>; video adı '<ad>__<komut>' — episode numarasından koşulu çıkarmak gerekmez |
| 18 | `analyze_4cond.py` | yok | OK | 0.7 | selftest OK (sayılar TEMSİLİ; gerçek log'la çalıştır) |
| 18 | `power_day18.py` | yok | OK | 15.1 | McNemar en küçük p, n=10, b uyumsuz: b=3: 0.2500, b=4: 0.1250, b=5: 0.0625, b=6: 0.0312, b=10: 0.0020 |
| 19 | `read_patterns.py` | yok | OK | 0.0 | selftest OK |
| 19 | `read_patterns.py --demo` | yok | OK | 0.0 | demo OK: dört temsili örüntü doğru sınıflandı |
| 19 | `suite_structure.py` | LIBERO klonu | OK | 0.0 |   libero_10: LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket → (And (In alpha |
| 19 | `init_state_geometry.py` | LIBERO klonu | OK | 0.1 |   kâse–kâse mesafesi 17.6 cm (min 14.4, maks 21.1); hedef kâse–tabak 11.7 cm (min 9.7, maks 13.7) |
| 20 | `tr_morph.py` | yok | OK | 0.0 | tr_morph selftest: 42/42 doğru |
| 20 | `make_tr_set.py` | yok | OK | 0.0 |       kase-yi tabağ-ın sol-u-na koy |
| 20 | `validate_set.py` | yok | OK | 0.0 | SONUÇ: TEMİZ — hata yok |
| 20 | `scene_frames.py` | LIBERO klonu | OK | 0.1 |    higher      none    → bowl_1   min fark   22.2 cm |
| 20 | `make_cf_bddl.py` | LIBERO klonu + bddl | OK | 0.0 | 5 karşı-olgusal BDDL + index.csv → /tmp/faz3smoke_6b2b61hm/cf |
| 21 | `check_webots_env.py` | Webots | ATLANDI |  |  |
| 21 | `ur10e_fk.py` | yok | OK | 0.1 | 03_plate      0.220    0.021    1.000  True       0.921 |
| 22 | `tokenize_set.py` | sentencepiece+tokenizer | OK | 0.5 | - 100 komutta action aralığına düşen token: **0**; aralığın örnek içeriği: 忠, 错, 단, 면, ķ, 桥 |
| 22 | `tokens.csv == datasets/05?` | sentencepiece+tokenizer | OK | 0.0 | 201 vs 201 satır, ilk 7 sütunda farklı satır: 0 → BİREBİR AYNI |
| 22 | `probe_edge_cases.py` | sentencepiece+tokenizer | OK | 0.5 |   [qwen3 ]  3  et / ▁al / . |
| 23 | `prediction_interval.py` | yok | OK | 0.7 | selftest OK: PI formülü (t·s·sqrt(1/n+1/n')), sıfır genişlik, n' etkisi, tek koşu = inf, 0,01 eşiği |
| 23 | `prediction_interval.py --scores` | yok | OK | 0.7 | n=3  x̄=0.867  PI95 = 0.867 ± 0.203  [0.664, 1.069]  genişlik=0.406  → tekrar ekle (makale eşiği 0,01) |
| 24 | `tr_parse.py --selftest` | yok | OK | 0.0 | selftest: 4/4 |
| 24 | `tr_parse.py` | yok | OK | 0.0 |   held-out: 5/12 |
| 24 | `qlora_budget.py` | yok | OK | 0.0 | Uyarı: bu bir hesap; gerçek sayı Mesut'un T4'te `finetune.py --use_quantization True --batch_size 1` ile ölçec |
| 24 | `mt_layer_mac.py` | HF+GPU | ATLANDI |  |  |
