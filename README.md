# Official implementation of paper "Retinal Image Restoration and Vessel Segmentation using Modified Cycle-CBAM and CBAM-UNet"
## Cycle-consistent Generative Adversarial Network (CycleGAN) with Convolutional Block Attention Module (CBAM) - Cycle-CBAM. Modified UNet with CBAM - CBAM-UNet.

1. Create anaconda environment with python=3.9.

2. Download DRIVE, STARE and CHASE DB1 in UNet folder.

3. Run aug_drive.py, aug_stare.py, aug_chase.py in UNet folder.

4. Run train.py in UNet folder selecting the dataset.

5. Run test.py in UNet folder selecting the dataset.

6. Download EyeQ dataset and place in datasets folder according to the qualitites: 0, 1, and 2.

7. Run train.py in main folder.

8. Run test.py in main folder.

## <a name="Citing"></a>Citing 

If you use this code, please use the following BibTeX entry.

```
@inproceedings{alimanov2022retinal,
  title={Retinal Image Restoration and Vessel Segmentation using Modified Cycle-CBAM and CBAM-UNet},
  author={Alimanov, Alnur and Islam, Md Baharul},
  booktitle={2022 Innovations in Intelligent Systems and Applications Conference (ASYU)},
  pages={1--6},
  year={2022},
  organization={IEEE}
}

```
