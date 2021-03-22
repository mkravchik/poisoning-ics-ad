# poisoning-ics-ad
Official implementation of "Poisoning Attacks on Cyber Attack Detectors for Industrial Control Systems" by Moshe Kravchik, Battista Biggio, and Asaf Shabtai, SAC 2021.
# UNDER CONSTRUCTION
## Requirements
 * Python 3.6
 * tensorflow==1.15
 * Keras==2.2.4
 * munch==2.5.0
 * numpy==1.18.1
 * scikit-image==0.16.2
 * scikit-learn==0.22.1
 * scipy==1.4.1

For running the SWaT tests the code is expecting to have the dataset that can be requested at https://itrust.sutd.edu.sg/itrust-labs_datasets/.
The dataset train and test files should be subsampled at the 5 seconds rate and saved locally in the files named SWaT_Dataset_Normal_sub5.csv and SWaT_Dataset_Attack_sub5.csv, correspondingly.
  
## Usage
```blockquote
python Poisoning.py [-h] [-c CONFIGURATION] [-a {3,7,16,31,32,33,36}] {syn,swat}
```
For training and poisoning using the synthetic data run:
```blockquote
python Poisoning.py syn
```

For training and poisoning using the SWaT data run:
```blockquote
python Poisoning.py swat [-a {3,7,16,31,32,33,36}]
```

The are multiple configuration parameters defined in the conf_syn.py and conf_swat.py files. You can tweak them to experiment with different settings.
The meaning of the non-standard ones is explained in the article.

#### Parameters (in the alphabetic order)
* activate_last {True, False} - Use activation function in the last layer. Using it increases the model's robustness, but might also limit the maximal possible attack on the model. 
* activation - The activation function used in the model.
* adv_it_count - The number of iterations of a single back-gradient algorithm run.
* adv_lr - The learning rate for the poison updates used in the back-gradient algorithm.
* att_len - The attack duration (in time steps).
* att_magn - The attack magnitude.
* att_point - {"CUSTOM_FIXED"|"CUSTOM_LINEAR"|"SIN_TOP"|"SIN_BOTTOM"|"SIN_SIDE"}. The attack location on the signal's period.
* batches - The number of batches in the model's training.
* code_ratio - The proportion between the input and the model's bottleneck dimensions.
* find_poison {True, False} - Use optimization to find the maximal starting poison that does not trigger an alert.
* generator {SinGenerator|DoubleSinGenerator} - The synthetic signal generator. Defined in generators.py. 
* inflate_factor - The factor of the model's inflation layer.
* it_count - Model's training iteration count.
* layers - The number of layers in the model (encoder and decoders separately).
* lr - The model's training learning rate.
* max_adv_iter - The maximal number of poisoning iterations.
* max_clean_poison - Not used.
* naive {True, False} - The algorithm to use: interpolation(True) or back-gradient.
* optimizer - The model's training optimizer.
* partial_attacked - The indices of features to attack.
* periods - The number of periods to include in the synthesized signal. 
* randomize {True, False} - Use random shuffling when training the model.
* retrain - Not used.
* retrain_points - Not used.
* sec_seq_len - If present, the second model's sequence length.
* seq_len - The model sequence length. 
* signal_func - The signal function used by the generators.
* silent {True|False} - The output verbosity.
* single_sequence {False|True} - Model the entire signal at ones or as short overlapping sequences.
* threshold - The attack detection threshold.
* total_len - Don't set, is overwritten by the code.
* train_points - The number of training batches for the synthetic data; for the SWaT attacks, controls how long is the singla used for the model's training (measured in the signal's period). 
* window - The attack detection window.
