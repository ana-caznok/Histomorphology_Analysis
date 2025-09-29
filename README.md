<div align="center">   

# Gene expression and Histomorphology in the Uterosacral Ligament
</div>

<h3 align="center">
  <a href="https://arxiv.org/abs/2212.10156">Paper Link*</a> |
  <a href="sources/cvpr23_uniad_poster.png">Poster</a> |
  <a href="https://opendrivelab.com/e2ead/UniAD_plenary_talk_slides.pdf">Slides</a>
</h3>

<br><br>

## Table of Contents:
1. [Research Overview](#overview)
2. [Methodology Overview](#methodology)
3. [Getting Started](#start)

## Research Overview <a name="overview"></a>
This is the github repository for article: Principal Component Analysis of Uterosacral Ligaments and HOXA-11 Protein Quantification in Women with Pelvic Organ Prolapse. 

Pelvic organ prolapse (POP) results from the failure of the support mechanisms of the pelvic viscera, resulting in vaginal protrusion and the fall of the pelvic organs through the vaginal canal. The pathophysiology of genital prolapse is multifactorial, in which environmental factors such as lifestyle, parity, and childbirth interact with molecular, endocrine, and genetic factors. The uterosacral ligaments (USL) are composed of collagen, smooth muscle, elastin and nerve bundles. On POP there are histomorphological changes characterized by decreased smooth muscle content, decreased cellularity, alterations in the extracellular matrix (ECM), increased apoptosis, increased inflammation and increased adipocytes. The ECM dysfunction is characterized by alterations in metabolism and distribution of the main proteins, like changes in the proportions of collagen subtypes. 
Hoxa11 is responsible for development of the female reproductive system and formation of uterosacral ligaments, lower uterine segment, and cervix, and also acts in collagen type III synthesis and matrix metalloproteinase 2 (MMP2) synthesis. HOXA11 regulates morphology and integrity of USLs by promoting cell proliferation and attenuating apoptosis and also regulating extracellular matrix homeostasis. Studies have shown reduced expression of HOXA11 associated with reduced expression of collagens, ECM disfunction and low cellularity in women with POP. 
It is our hypothesis that due to the low expression of HOXA11 in women with prolapse, there are changes in the uterosacral ligament, as well as the organization of the connective tissue, proportion of smooth muscle and cellularity. Therefore, the objective of this study is to analyze the expression of the HOXA11 gene and its association with the histomorphological alterations in the uterosacral ligament in women with pelvic organs. 

## Methodology Overview <a name="methodology"></a>
The computational methodology was divided into three parts: a histomorphological quantification, a immunohistochemical quantification and joining both information into a single dataframe that allows PCA analysis. 
(INSERIR IMAGEM AQUI) 

### Histomorphology
HE histopathological images were annotated and masks were created with QuPath. QuPath measurements were exported to an excel table that is than read and analysed by python scripts available in this repository. 
 
### Gene expression 
Immunohistochemitry images were analysed through an ImageJ macro .ijm available in this repository.

### Joining information
To join information in both datasets we also used python scripts available in the repository

## Getting Started <a name="start"></a>
1. Clone this repository: 

2. Then install al requirements, in your linux or windows terminal write the following command:
pip install -r requirements.txt
3. Use QuPath to annotate images and extract measurements
4. Paste measurements into an excel spreadsheet
5. Organize your immunohistochemistry images
6. Run .ijm imagej macro
7. Run extract_area_proportion_final.py script to get proper measurements from masks
8. Run df_compilation_plot_final.py to join histomorphological and immunohistochemical measurements 
