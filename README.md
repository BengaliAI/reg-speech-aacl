## **Are ASR foundation models generalized enough to capture features of regional dialects for low-resource languages?**

Accepted at **AACL 2025** | [Paper (Coming Soon)](#) | [Poster](AACL_2025_Poster.pdf) | [Dataset](https://huggingface.co/datasets/bengaliAI/Ben-10) | [Model](https://huggingface.co/bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium/tree/main) | [Demo](https://huggingface.co/spaces/bengaliAI/regional_bengali-asr_tugstugi_whisper-medium)

---

## Abstract

Conventional research on speech recognition modeling relies on the canonical form for most low-resource languages while automatic speech recognition (ASR) for regional dialects is treated as a fine-tuning task. To investigate the effects of dialectal variations on ASR we develop a **78-hour** annotated Bengali Speech-to-Text (STT) corpus named **Ben-10**. 

Investigation from linguistic and data-driven perspectives shows that speech foundation models struggle heavily in regional dialect ASR, both in zero-shot and fine-tuned settings. We observe that all deep learning methods struggle to model speech data under dialectal variations but dialect-specific model training alleviates the issue. Our dataset also serves as an out-of-distribution (OOD) resource for ASR modeling under constrained resources in ASR algorithms. The dataset and code developed for this project are publicly available.

---

## Competitions & Reports
A competition was organized on Kaggle based on this dataset.
- **Kaggle Competition:** [Ben-10 Competition](https://www.kaggle.com/competitions/ben10/discussion/491012)
- **Competition Model Details Report:** [Google Drive Report Folder](https://drive.google.com/drive/u/2/folders/1V1ZdRUImUqZ5Jiv9AiGBXoFY7p0WW15M)

---

## Dataset Details

- **Size:** 78 hours of annotated Bengali speech
- **Dialects:** 10 regional dialects of Bengali
- **Format:** Speech-to-Text (STT) corpus
- **Use Case:** Out-of-distribution (OOD) resource for ASR modeling under constrained resources

---

## Repository Structure

```
reg-speech-aacl/
├── finetuning/          # Fine-tuning scripts and notebooks
├── result_analysis/     # Analysis notebooks and results
├── AACL_2025_Poster.pdf # Conference poster
└── README.md           # This file
```
## Usage

The repository contains:
- **Finetuning scripts** (`finetuning/`): Code for fine-tuning ASR models on regional dialects
- **Result analysis** (`result_analysis/`): Analysis notebooks for different regions

---

## Citation

If you use this dataset or code in your research, please cite:

```bibtex
@inproceedings{ben10-2025,
  title={Are ASR foundation models generalized enough to capture features of regional dialects for low-resource languages?},
  author={...},
  booktitle={Proceedings of AACL 2025},
  year={2025}
}
```

*Citation will be updated once the paper is published.*

