## FaceFormer

The source code for the paper:

**FaceFormer: Speech-Driven 3D Facial Animation with Transformers**. ***CVPR 2022*** [[PDF]](https://arxiv.org/pdf/2112.05329.pdf)

<p align="center">
<img src="framework.jpg" width="80%" />
</p>

## Environment

- Ubuntu 18.04.1
- Python 3.7
- Pytorch 1.9.0

## Dependencies

Check the required packages in `requirements.txt`.
- transformers
- librosa
- trimesh
- opencv-python
- pyrender
- [MPI-IS/mesh](https://github.com/MPI-IS/mesh)

## Training and Testing on VOCASET

###  Data
 
Request the VOCASET data from [https://voca.is.tue.mpg.de/](https://voca.is.tue.mpg.de/). Place the downloaded files `data_verts.npy`, `raw_audio_fixed.pkl`, `templates.pkl` and `subj_seq_to_idx.pkl` in `vocaset/VOCASET/`. Download "FLAME_sample.ply" from [voca](https://github.com/TimoBolkart/voca/tree/master/template) and put it in `VOCASET/templates`.

### Demo

- To animate a mesh given an audio signal, download the [pretrained model](https://drive.google.com/file/d/1GUQBk9FqUimoT6UNgU0gyQnjGv-2_Lyp/view?usp=sharing) and put it in the folder `vocaset/VOCASET`, run: 

	```
	cd vocaset
	python demo.py --wav_path "VOCASET/demo/wav/test.wav"
	```

###  Data Preparation

- Read the vertices/audio data and convert them to .npy/.wav files stored in `vocaset/VOCASET/vertices_npy` and `vocaset/VOCASET/wav`:

	```
	cd vocaset
	python process_voca_data.py
	```

### Training and Testing

- To train the model and obtain the results on the testing set, run:

	```
	cd vocaset
	python main.py
	```
	The results will be available in the `vocaset/VOCASET/result` folder, and the models will be stored in the `vocaset/VOCASET/save` folder.

### Visualization

- To visualize the results, run:

	```
	cd vocaset
	python render.py
	```
	The rendered videos will be available in the `vocaset/VOCASET/output` folder.

## Training and Testing on BIWI

### Data
 
Request the dataset from [Biwi 3D Audiovisual Corpus of Affective Communication](https://data.vision.ee.ethz.ch/cvl/datasets/b3dac2.en.html). The dataset contains the following subfolders:

- 'faces' contains the binary (.vl) files for the tracked facial geometries. 
- 'rigid_scans' contains the templates stored as .obj files. 
- 'audio' contains audio signals stored as .wav files. 

Place the folders 'faces' and 'rigid_scans' in `BIWI_data` and place the wav files in `BIWI_data/wav`.

### Demo

- To animate a mesh given an audio signal, download the [pretrained model](https://drive.google.com/file/d/1WR1P25EE7Aj1nDZ4MeRsqdyGnGzmkbPX/view?usp=sharing) and put it in the folder `biwi/BIWI_data/`, run: 

	```
	cd biwi
	python demo.py --wav_path "BIWI_data/demo/wav/test.wav"
	```

###  Data Preparation

- (to do) Read the geometry data and convert them to .npy files stored in `biwi/BIWI_data/vertices_npy`.

### Training and Testing

- To train the model and obtain the results on testing set, run:

	```
	cd biwi
	python main.py
	```
	The results will be available in the `biwi/BIWI_data/result` folder, and the models will be stored in the `biwi/BIWI_data/save` folder.

### Visualization

- To visualize the results, run:

	```
	cd biwi
	python render.py
	```
	The rendered videos will be available in the `biwi/BIWI_data/output` folder.

## Citation

If you find this code useful for your work, please consider citing:
```
@inproceedings{faceformer2022,
title={FaceFormer: Speech-Driven 3D Facial Animation with Transformers},
author={Fan, Yingruo and Lin, Zhaojiang and Saito, Jun and Wang, Wenping and Komura, Taku},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2022}
}
```
## Acknowledgement

We gratefully acknowledge ETHZ-CVL for providing the [B3D(AC)2](https://data.vision.ee.ethz.ch/cvl/datasets/b3dac2.en.html) database and MPI-IS for releasing the [VOCASET](https://voca.is.tue.mpg.de/) dataset. The implementation of wav2vec2 is built upon [huggingface-transformers](https://github.com/huggingface/transformers/blob/master/src/transformers/models/wav2vec2/modeling_wav2vec2.py), and the temporal bias is modified from [ALiBi](https://github.com/ofirpress/attention_with_linear_biases). We use [MPI-IS/mesh](https://github.com/MPI-IS/mesh) for mesh processing and [VOCA/rendering](https://github.com/TimoBolkart/voca) for rendering. We thank the authors for their great works.

Any third-party packages and data are owned by their respective authors and must be used under their respective licenses.

