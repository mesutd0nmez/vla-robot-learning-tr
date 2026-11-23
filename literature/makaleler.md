# Paper list: 15 peer-reviewed works and 2 preprints

This section lists 15 peer-reviewed papers on Vision-Language-Action (VLA) models and multilingual VLA, screened and selected in advance. The citations and summaries were read on 21 November 2026 from each paper's own publication page; where a piece of information is not on the page it is marked "could not be verified". A two-page assessment based on these papers: [Report](rapor.md).

**Main finding:** of the 15 papers, only one (number 1) deals directly with multilingual VLA. The abstracts of the other 14 do not mention languages other than English; the "relevance to multilinguality" lines below show this paper by paper.

| Group | Papers |
|---|---|
| (a) Multilingual VLA | 1 |
| (b) Foundational VLA models | 2, 3, 4, 5 |
| (c) Efficiency and fine-tuning | 6, 7, 8, 14 |
| (d) Reasoning and new modalities | 9, 10, 11, 12, 13 |
| (e) Survey | 15 |

---

## (a) Multilingual VLA

### 1. When Does Language Matter? Multilingual Instructions Reveal Step-wise Language Sensitivity in Vision-Language-Action Models
- **Authors:** Xuan Dong, Zhe Han, Tianhao Niu, Qingfu Zhu, Wanxiang Che
- **Venue and year:** ACL 2026 (64th Annual Meeting of the ACL, Volume 1: Long Papers), pp. 44615–44629, July 2026
- **Link:** [aclanthology.org/2026.acl-long.2066](https://aclanthology.org/2026.acl-long.2066/) · DOI [10.18653/v1/2026.acl-long.2066](https://doi.org/10.18653/v1/2026.acl-long.2066)
- **Summary:** LIBERO commands are translated into nine languages with Google Translate (ten languages with English) and OpenVLA-OFT and π0.5 are evaluated with them; with non-English commands the average success drops by 32–46 points. The authors show that the effect of language is not evenly distributed over steps, and propose a retraining-free intervention that aligns hidden representations to English references only at "language-critical" steps.
- **Relevance to multilinguality:** Main topic. Languages: Chinese, Japanese, Korean, French, Spanish, Arabic, Thai, Portuguese, Vietnamese. No Turkish.

Average success rates from Tables 1 and 2 of the paper (%; average of the four LIBERO suites):

| Model | English | Other 9 languages (range) | With the step-wise intervention |
|---|---|---|---|
| OpenVLA-OFT | 97.1 | 50.8–65.3 | 62.6–70.9 |
| π0.5 | 96.9 | 55.4–61.2 | 79.8–82.1 |

The heaviest loss is in the Goal suite: 97.9 and 97.6 in English versus 6.4–16.8 in the other languages. The translations were checked by human inspection and back-translation. The authors also fine-tune OpenVLA-OFT with Chinese commands (Table 4): the average rises from 51.7 to 86.8; 85.5 is obtained with language-critical steps only and 44.9 with randomly chosen steps. All experiments are in simulation; the authors leave transfer to real robots as an open question.

---

## (b) Foundational VLA models

### 2. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
- **Authors:** Brianna Zitkovich, Tianhe Yu, Sichun Xu et al. (54 authors; full list on the publication page)
- **Venue and year:** CoRL 2023 (Proceedings of The 7th Conference on Robot Learning), PMLR 229:2165–2183, 2023
- **Link:** [proceedings.mlr.press/v229/zitkovich23a.html](https://proceedings.mlr.press/v229/zitkovich23a.html)
- **Summary:** Vision-language models are fine-tuned jointly on robot trajectories and web-scale vision-language tasks; robot actions are expressed as text tokens. The abstract names this model class "vision-language-action models (VLA)" and reports improved generalization across about 6 thousand trials.
- **Relevance to multilinguality:** Not the main topic. The abstract does not mention languages other than English.

### 3. OpenVLA: An Open-Source Vision-Language-Action Model
- **Authors:** Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti et al. (17 authors)
- **Venue and year:** CoRL 2024 (Proceedings of The 8th Conference on Robot Learning), PMLR 270:2679–2713, 2025
- **Link:** [proceedings.mlr.press/v270/kim25c.html](https://proceedings.mlr.press/v270/kim25c.html) · [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)
- **Summary:** An open-source VLA with 7 billion parameters trained on 970 thousand robot demonstrations; it combines the Llama 2 language model with DINOv2 and SigLIP visual features. The abstract reports that it beats the 55-billion-parameter RT-2-X by 16.5 points in absolute success and can be fine-tuned on consumer GPUs with LoRA.
- **Relevance to multilinguality:** None. The abstract does not mention languages other than English.

### 4. π0: A Vision-Language-Action Flow Model for General Robot Control
- **Authors:** Kevin Black, Noah Brown, Danny Driess et al. (25 authors)
- **Venue and year:** RSS 2025 (Robotics: Science and Systems XXI), June 2025
- **Link:** [roboticsproceedings.org/rss21/p010.html](https://www.roboticsproceedings.org/rss21/p010.html) · DOI [10.15607/RSS.2025.XXI.010](https://doi.org/10.15607/RSS.2025.XXI.010)
- **Summary:** Builds a flow-matching architecture on top of a pre-trained vision-language model, thereby inheriting the model's internet-scale semantic knowledge. It is trained with data from multiple robot platforms and evaluated on tasks such as laundry folding, table clearing and box assembly.
- **Relevance to multilinguality:** None. The abstract does not mention languages other than English.

### 5. π0.5: a Vision-Language-Action Model with Open-World Generalization
- **Authors:** Kevin Black, Noah Brown, James Darpinian et al. (35 authors)
- **Venue and year:** CoRL 2025 (Proceedings of The 9th Conference on Robot Learning), PMLR 305:17–40, 2025
- **Link:** [proceedings.mlr.press/v305/black25a.html](https://proceedings.mlr.press/v305/black25a.html)
- **Summary:** Built on π0; it uses co-training with heterogeneous sources such as different robots, high-level semantic prediction and web data. It is reported to carry out long tasks, such as cleaning a kitchen or a bedroom, in homes it has not seen before.
- **Relevance to multilinguality:** Not the main topic. The abstract does not mention languages other than English; it is one of the two models tested with multilingual commands in paper number 1.

---

## (c) Efficiency and fine-tuning

### 6. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (OpenVLA-OFT)
- **Authors:** Moo Jin Kim, Chelsea Finn, Percy Liang
- **Venue and year:** RSS 2025 (Robotics: Science and Systems XXI), June 2025
- **Link:** [roboticsproceedings.org/rss21/p017.html](https://www.roboticsproceedings.org/rss21/p017.html) · DOI [10.15607/RSS.2025.XXI.017](https://doi.org/10.15607/RSS.2025.XXI.017)
- **Summary:** Taking OpenVLA as the base, it examines the design decisions of fine-tuning (action decoding, action representation, learning objective) and derives the Optimized Fine-Tuning (OFT) recipe. OpenVLA-OFT raises the LIBERO average from 76.5% to 97.1% and increases action generation speed by 26 times.
- **Relevance to multilinguality:** None on its own. It is one of the two models tested with multilingual commands in paper number 1.

### 7. FAST: Efficient Action Tokenization for Vision-Language-Action Models
- **Authors:** Karl Pertsch, Kyle Stachowicz, Brian Ichter et al. (9 authors)
- **Venue and year:** RSS 2025 (Robotics: Science and Systems XXI), June 2025
- **Link:** [roboticsproceedings.org/rss21/p012.html](https://www.roboticsproceedings.org/rss21/p012.html) · DOI [10.15607/RSS.2025.XXI.012](https://doi.org/10.15607/RSS.2025.XXI.012) · [arXiv:2501.09747](https://arxiv.org/abs/2501.09747)
- **Summary:** Argues that action tokenizers that bin per dimension and per time step fall short on high-frequency, dexterous tasks, and proposes a DCT-based compression tokenizer (FAST). It releases the universal FAST+ tokenizer trained on 1 million trajectories; it reports training time cut by up to 5 times.
- **Relevance to multilinguality:** None. The abstract does not mention languages other than English.

### 8. TinyVLA: Toward Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation
- **Authors:** Junjie Wen, Yichen Zhu, Jinming Li et al. (13 authors in the journal record)
- **Venue and year:** IEEE Robotics and Automation Letters, vol. 10, no. 4, pp. 3988–3995, April 2025
- **Link:** DOI [10.1109/LRA.2025.3544909](https://doi.org/10.1109/LRA.2025.3544909) · [arXiv:2409.12514](https://arxiv.org/abs/2409.12514)
- **Summary:** Presents a compact family of VLAs that need no pre-training on large-scale robot data and run inference faster. It adds a diffusion policy decoder to a pre-trained multimodal backbone during fine-tuning; it reports surpassing OpenVLA in speed and data efficiency.
- **Relevance to multilinguality:** None. The abstract does not mention languages other than English.

### 14. Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment
- **Authors:** Tao Lin, Yilei Zhong, Yuxin Du et al. (14 authors)
- **Venue and year:** CVPR 2026 (Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition), pp. 13397–13406
- **Link:** [openaccess.thecvf.com — CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Lin_Evo-1_Lightweight_Vision-Language-Action_Model_with_Preserved_Semantic_Alignment_CVPR_2026_paper.html) · [arXiv:2511.04555](https://arxiv.org/abs/2511.04555)
- **Summary:** A lightweight VLA with 0.77 billion parameters that works without pre-training on robot data; it preserves VLM representations with a two-stage training that progressively aligns action learning with perception. The abstract reports that it surpasses the previous best results on Meta-World and RoboTwin, reaching 94.8% on LIBERO and 78% success on a real robot.
- **Relevance to multilinguality:** None. The abstract does not mention languages other than English.

---

## (d) Reasoning and new modalities

### 9. VLAS: Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation
- **Authors:** Wei Zhao, Pengxiang Ding, Min Zhang, Zhefei Gong, Shuanghao Bai, Han Zhao, Donglin Wang
- **Venue and year:** ICLR 2025 (poster)
- **Link:** [iclr.cc/virtual/2025/poster/30076](https://iclr.cc/virtual/2025/poster/30076)
- **Summary:** An end-to-end VLA that takes speech directly as input, without a separate speech recognition layer; thus non-semantic cues such as the voice track are preserved. It uses three-stage speech-command training and voice retrieval-augmented generation for personalized tasks.
- **Relevance to multilinguality:** Indirect: the input modality is speech instead of text. The language of the speech commands is not stated on the publication page (could not be verified).

### 10. CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models
- **Authors:** Qingqing Zhao, Yao Lu, Moo Jin Kim et al. (15 authors)
- **Venue and year:** CVPR 2025, pp. 1702–1713
- **Link:** [openaccess.thecvf.com — CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_CoT-VLA_Visual_Chain-of-Thought_Reasoning_for_Vision-Language-Action_Models_CVPR_2025_paper.html)
- **Summary:** A VLA with 7 billion parameters that predicts future image frames as visual goals before generating actions. The abstract reports that it surpasses the previous best VLA by 17% on real-robot tasks and 6% in simulation.
- **Relevance to multilinguality:** None. The abstract does not mention languages other than English.

### 11. Vision-Language-Action Instruction Tuning: From Understanding to Manipulation (InstructVLA)
- **Authors:** Shuai Yang, Hao Li, Bin Wang, Yilun Chen, Yang Tian, Tai Wang, Hanqing Wang, Feng Zhao, Yiyi Liao, Jiangmiao Pang
- **Venue and year:** ICLR 2026
- **Link:** [proceedings.iclr.cc — abstract page](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f81332c0e757b169c66d436ec2fff624-Abstract-Conference.html) · [PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/f81332c0e757b169c66d436ec2fff624-Paper-Conference.pdf)
- **Summary:** Presents the InstructVLA model and the Vision-Language-Action Instruction Tuning (VLA-IT) training approach: standard VLM data and a 650-thousand-sample VLA-IT dataset are used together, so the VLM's reasoning ability is preserved during action training. It introduces the 80-task SimplerEnv-Instruct benchmark.
- **Relevance to multilinguality:** None. The abstract does not mention languages other than English.

### 12. 3D-VLA: A 3D Vision-Language-Action Generative World Model
- **Authors:** Haoyu Zhen, Xiaowen Qiu, Peihao Chen, Jincheng Yang, Xin Yan, Yilun Du, Yining Hong, Chuang Gan
- **Venue and year:** ICML 2024 (Proceedings of the 41st International Conference on Machine Learning), PMLR 235:61229–61245, 2024
- **Link:** [proceedings.mlr.press/v235/zhen24a.html](https://proceedings.mlr.press/v235/zhen24a.html)
- **Summary:** Built on a 3D-based language model; it connects 3D perception, reasoning and action through a generative world model. Diffusion models that predict goal images and point clouds are trained and aligned to the language model.
- **Relevance to multilinguality:** None. The abstract does not mention languages other than English.

### 13. QUAR-VLA: Vision-Language-Action Model for Quadruped Robots
- **Authors:** Pengxiang Ding, Han Zhao, Wenjie Zhang, Wenxuan Song, Min Zhang, Siteng Huang, Ningxi Yang, Donglin Wang
- **Venue and year:** ECCV 2024 (Computer Vision – ECCV 2024, LNCS 15063), pp. 352–367, Springer
- **Link:** DOI [10.1007/978-3-031-72652-1_21](https://doi.org/10.1007/978-3-031-72652-1_21) · [arXiv:2312.14457](https://arxiv.org/abs/2312.14457)
- **Summary:** Presents the QUART model, which produces executable actions for quadruped robots from visual information and commands, and the QUARD dataset containing perception, navigation and whole-body manipulation tasks.
- **Relevance to multilinguality:** None. The abstract does not mention languages other than English.

---

## (e) Survey

### 15. Sensing the Action: Rethinking Sensor Modalities and Multi-Modal Fusion in Vision–Language–Action Models for Robotic Manipulation
- **Authors:** Byoung Chul Ko
- **Venue and year:** Sensors, 26(11), 3541, 2026 (review article; 3 June 2026)
- **Link:** [mdpi.com/1424-8220/26/11/3541](https://www.mdpi.com/1424-8220/26/11/3541) · DOI [10.3390/s26113541](https://doi.org/10.3390/s26113541)
- **Summary:** Reconsiders VLA as a sensor, fusion and action pipeline; classifies sensor modalities such as RGB, depth, touch and force/torque, and fusion designs. It finds that current benchmarks lean on RGB-heavy inputs and a single success-rate metric.
- **Relevance to multilinguality:** None. The abstract does not mention languages other than English.

---

## Related preprints

The two works below are **not peer-reviewed**: they are arXiv preprints, and as of 21 November 2026 no publication venue is stated on their arXiv pages. They are not included in the 15-paper list above.

### Beyond English: Uncovering the Multilingual Gap in Vision-Language-Action Models
- **Authors:** Hanyang Chen, Hongliang Li, Jiarui Cao, Yang Li, Yang Jiang, Haonan Wen, Kaiyu Huang, Shengnan Guo, Huaiyu Wan
- **Status:** arXiv preprint, 14 June 2026 (v1)
- **Link:** [arXiv:2606.15714](https://arxiv.org/abs/2606.15714)
- **Summary:** Reports that VLAs trained mainly with English commands weaken markedly in other languages, even when the language backbone is multilingual. Proposes a fine-tuning method that aligns multilingual representations (Multilingual Principal Component Alignment).
- **Relevance to multilinguality:** Main topic.

### Measuring Language Transfer in Robot Policies: Adding Greek to a Cosmos3 Vision-Language-Action Policy
- **Authors:** Ayoub Kirouane, Georgios Giaples, Christos Petrocheilos
- **Status:** arXiv preprint, 7 September 2026 (v1)
- **Link:** [arXiv:2609.07470](https://arxiv.org/abs/2609.07470)
- **Summary:** Adds Greek to an open VLA policy using only machine-rewritten commands. According to the abstract, bilingual training rises 6.7–7.1 points above the control condition and reaches about two fifths of English performance; the authors stress that a baseline measured with a wrong command is needed before trusting a metric.
- **Relevance to multilinguality:** Main topic (Greek).
