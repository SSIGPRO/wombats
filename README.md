# WOMBATS: Wide Open Model-Based Anomaly Test Suite
Yes, we liked the acronym :)
This repository contains the code to a recently published paper:

A. Enttsel, S. Onofri, A. Marchioni, M. Mangia, G. Setti and R. Rovatti, "A General Framework for the Assessment of Detectors of Anomalies in Time Series," in IEEE Transactions on Industrial Informatics, [doi: 10.1109/TII.2024.3413359](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10583949).

Where we propose a test suite for anomaly detectors working with **univariate time-series** data on a window basis. The core of the methodology is a set of **12 abstract anomalies** that simulate most of the effect real-world anomalies have on the normal signal. These abstract behaviors are well characterized from the mathematical point of view and all share a common parameter controlling their intensity. 
The code in this repository allows to **synthetically inject** these abstract anomalies into the normal test data to produce its anomalous counterpart. With this, a detector can be analyzed with respect to different effects and different intensities.

## Citation
If you find this code useful to your project, please cite the original article:

    @ARTICLE{10583949,
      author={Enttsel, Andriy and Onofri, Silvia and Marchioni, Alex and Mangia, Mauro and Setti, Gianluca and Rovatti, Riccardo},
      journal={IEEE Transactions on Industrial Informatics}, 
      title={A General Framework for the Assessment of Detectors of Anomalies in Time Series}, 
      year={2024},
      volume={},
      number={},
      pages={1-11},
      keywords={Detectors;Time series analysis;Monitoring;Vectors;Informatics;Time measurement;Testing;Model selection;outlier detection;second-order statistics;sensor faults;synthetic anomalies;time series},
      doi={10.1109/TII.2024.3413359}}
