# Image Prediction & Tagging

**Machine Learning Project**

**Author:** Kelly Lwin  
**Course:** CS 4200 Artificial Intelligence, CPP Spring 2025  
**Assignment:** Course Project  
**Last Updated:** April 2025  

---
## Demo Video

<iframe width="560" height="315" src="https://www.youtube.com/embed/QIf7X2DXDBE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

[Youtube Link](https://youtu.be/QIf7X2DXDBE?si=aH_GTB3ARBp735x8)
---
## Language

- Python

[Download models here.](https://livecsupomona-my.sharepoint.com/:f:/g/personal/klwin_cpp_edu/Eq2qKShAYGRCpFzuoglnmuEBPv6kZzKLQ0zSH1RELokbWg?e=mjuYI1)
---

## Requirements

Install dependencies with:

```bash
pip install tensorflow numpy matplotlib nltk opencv-python fiftyone pillow
```

---

## Dataset References & Citations

### Common Objects in Context COCO 80-Category
A large-scale object detection, segmentation, and captioning dataset licensed under CC BY 4.0.  
[COCO Dataset](https://cocodataset.org/#home)

### Food-101  
Bossard, L., Guillaumin, M., & Van Gool, L. (2014). _Food-101 – Mining Discriminative Components with Random Forests_. European Conference on Computer Vision.

```bibtex
@inproceedings{bossard14,
  title     = {Food-101 -- Mining Discriminative Components with Random Forests},
  author    = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision},
  year      = {2014}
}
```

### ImageNet-R (200 Classes)
Hendrycks, D., Basart, S., Mu, N., Kadavath, S., Wang, F., Dorundo, E., Desai, R., Zhu, T., Parajuli, S., Guo, M., Song, D., Steinhardt, J., & Gilmer, J. (2021).  
_The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization_. ICCV.

```bibtex
@article{hendrycks2021many,
  title   = {The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization},
  author  = {Hendrycks, Dan and Basart, Steven and Mu, Norman and Kadavath, Saurav and Wang, Frank and Dorundo, Evan and Desai, Rahul and Zhu, Tyler and Parajuli, Samyak and Guo, Mike and Song, Dawn and Steinhardt, Jacob and Gilmer, Justin},
  journal = {ICCV},
  year    = {2021}
}
```

### Hugging Face Furniture-Dataset (7 Classes)
[Arkan0ID/furniture-dataset](https://huggingface.co/datasets/Arkan0ID/furniture-dataset)

---

## Tools Used

- **Programming Language:** Python 3.12.10  
- **IDE/Notebook:** Jupyter Notebook, VS Code  
- **Framework:** TensorFlow 2.19.0  

---

## References

- [Voxel51 COCO Integration](https://docs.voxel51.com/integrations/coco.html)  
- [Voxel51 Dataset Zoo](https://docs.voxel51.com/dataset_zoo/index.html)  
- [RAPIDS cuML Documentation](https://docs.rapids.ai/api/cuml/stable/)  
