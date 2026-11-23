# Report: the current state of VLA and multilingual VLA

**Scope:** 15 peer-reviewed papers ([paper list](makaleler.md)) · **Date:** 21 November 2026 · Bracketed numbers refer to the reference list and the paper list.

## 1. What is a VLA, and why does it matter?

A Vision-Language-Action (VLA) model is a single policy that takes a camera image and a natural-language command as input and directly produces a robot action. Its importance is that it carries the knowledge of vision-language models (VLMs) trained at internet scale into robot control. RT-2, which introduced the term, expressed actions as text tokens and trained the model jointly on robot and web data, showing that this improves generalization to new objects [2]. Since the command interface is language, which languages a model really understands is a direct usability question.

## 2. Development of the field

After RT-2, OpenVLA offered an open-source model with 7 billion parameters trained on 970 thousand robot demonstrations; it is built on Llama 2 and can be fine-tuned on consumer GPUs [3]. π0 builds a flow-matching architecture on top of a pre-trained VLM [4]. π0.5, thanks to co-training with heterogeneous data, carries out long tasks in homes it has not seen before [5]. A parallel line focuses on efficiency: OpenVLA-OFT changes the fine-tuning recipe and raises the LIBERO average from 76.5% to 97.1% [6], FAST proposes a compression-based action tokenizer [7], and TinyVLA [8] and Evo-1 [14] target small models that do not need large pre-training on robot data. Other works add new input and reasoning forms: speech commands [9], visual chain of thought [10], training that preserves VLM abilities [11], a 3B world model [12] and quadruped robots [13]. Ko's survey notes that current benchmarks lean on RGB-heavy inputs and a single success-rate metric [15].

## 3. The current state of multilingual VLA

Of the fifteen papers, only one deals with multilinguality [1]. Dong et al. translate LIBERO commands into nine languages with Google Translate (Chinese, Japanese, Korean, French, Spanish, Arabic, Thai, Portuguese, Vietnamese; ten languages with English) and check the translations by human inspection and back-translation. The two models tested are OpenVLA-OFT and π0.5. While the English average success is 97.1% and 96.9%, in the other languages it falls to 50.8–65.3% for OpenVLA-OFT and 55.4–61.2% for π0.5. The loss is 32–46 points and is heaviest in Goal tasks (6–17%). The second finding is that the effect of language is not evenly distributed over steps: a small number of "language-critical" steps determine failure. The authors propose a retraining-free intervention that aligns hidden representations to English references at these steps. Average success rises to 62.6–70.9% for OpenVLA-OFT and 79.8–82.1% for π0.5, but does not reach the English level. All experiments are in simulation.

## 4. The literature gap and open research questions

The gap is clear: the only study that directly examines multilingual VLA in a peer-reviewed venue is [1], and the abstracts of the other fourteen papers do not mention languages other than English. There are two preprints pointing in the same direction (arXiv:2606.15714, arXiv:2609.07470), but they have not yet passed peer review. The single peer-reviewed study is also limited: two models, one benchmark, machine translation and nine languages. Turkish is not among them. So the gap should be read not as "the field has failed to solve multilinguality" but as "the field has not yet measured it enough". The open questions are these. Do the findings hold on real robots and on benchmarks other than LIBERO? Do agglutinative languages such as Turkish show the same pattern? Does the loss come from the language itself, from the translation surface, or from the model already using the command little? Separating this needs control conditions such as empty and wrong commands. Do different language backbones behave differently? Is the lasting solution an inference-time intervention or multilingual training? These questions directly justify this project's axis of Turkish spatial commands.

## References

- [1] Dong, X., Han, Z., Niu, T., Zhu, Q. and Che, W. (2026). When Does Language Matter? Multilingual Instructions Reveal Step-wise Language Sensitivity in Vision-Language-Action Models. *ACL 2026*, 44615–44629.
- [2] Zitkovich, B. et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. *CoRL 2023*, PMLR 229, 2165–2183.
- [3] Kim, M. J. et al. (2025). OpenVLA: An Open-Source Vision-Language-Action Model. *CoRL 2024*, PMLR 270, 2679–2713.
- [4] Black, K. et al. (2025). π0: A Vision-Language-Action Flow Model for General Robot Control. *RSS 2025*.
- [5] Black, K. et al. (2025). π0.5: a Vision-Language-Action Model with Open-World Generalization. *CoRL 2025*, PMLR 305, 17–40.
- [6] Kim, M. J., Finn, C. and Liang, P. (2025). Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success. *RSS 2025*.
- [7] Pertsch, K. et al. (2025). FAST: Efficient Action Tokenization for Vision-Language-Action Models. *RSS 2025*.
- [8] Wen, J. et al. (2025). TinyVLA: Toward Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation. *IEEE Robotics and Automation Letters*, 10(4), 3988–3995.
- [9] Zhao, W. et al. (2025). VLAS: Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation. *ICLR 2025*.
- [10] Zhao, Q. et al. (2025). CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models. *CVPR 2025*, 1702–1713.
- [11] Yang, S. et al. (2026). Vision-Language-Action Instruction Tuning: From Understanding to Manipulation. *ICLR 2026*.
- [12] Zhen, H. et al. (2024). 3D-VLA: A 3D Vision-Language-Action Generative World Model. *ICML 2024*, PMLR 235, 61229–61245.
- [13] Ding, P. et al. (2024). QUAR-VLA: Vision-Language-Action Model for Quadruped Robots. *ECCV 2024*, LNCS 15063, 352–367.
- [14] Lin, T. et al. (2026). Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment. *CVPR 2026*, 13397–13406.
- [15] Ko, B. C. (2026). Sensing the Action: Rethinking Sensor Modalities and Multi-Modal Fusion in Vision–Language–Action Models for Robotic Manipulation. *Sensors*, 26(11), 3541.
