# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

- **Authors:** Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
- **Year:** 2020
- **Venue:** ICLR 2021
- **Link:** https://arxiv.org/abs/2010.11929
- **Key contributions:**
  - Pure transformer applied directly to sequences of image patches for image classification
  - No CNN — patches projected linearly to token embeddings
  - CLS token for classification; learnable 1D position embeddings
  - Scales well with data: ViT-H/14 surpasses CNN SOTA when pretrained on JFT-300M
- **Implementation scope:** Core ViT architecture (patch embedding, transformer encoder, classification head) with ViT-B/16 configuration. No pretraining — unit tests verify shapes and forward pass.
