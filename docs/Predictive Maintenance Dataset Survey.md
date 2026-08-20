# **Analysis and Survey of Predictive Maintenance Datasets for Industrial Intelligence and Machine Learning**

## **Conceptual Evolution of Industrial Maintenance**

Industrial maintenance strategies have undergone a fundamental shift over the past several decades, driven by advancements in sensing technology, computational capacity, and artificial intelligence1. The evolutionary trajectory spans from reactive, run-to-failure approaches to preventive schedules, and finally to modern predictive and condition-based maintenance systems1.  
Reactive maintenance relies on operating assets until failure occurs before performing corrective actions2. While this minimizes routine maintenance interventions, it introduces substantial risks of irreversible equipment damage, costly operational downtime, and severe safety hazards2.  
To mitigate these unexpected disruptions, industries adopted time-based or preventive maintenance3. This strategy schedules interventions based on elapsed time or operational cycles, leveraging historical mean-time-between-failures calculations3. However, because components degrade at highly variable rates depending on localized stress, environment, and manufacturing variations, time-based maintenance frequently triggers premature, unnecessary parts replacement and generates high prevention costs1.  
Predictive maintenance leverages real-time multi-sensor telemetry to continuously monitor equipment health indicators and identify impending faults before they cause failures3. By employing machine learning and deep learning algorithms, predictive systems optimize maintenance schedules dynamically2.  
This shift relies on two core tasks: fault diagnostics (identifying, classifying, and isolating anomalies) and fault prognostics (estimating the Remaining Useful Life (![][image1]) of the system)2. Through continuous deterioration analysis and service effects modeling, these data-driven architectures narrow uncertainty bounds, maximize asset reliability, and reduce total production overhead1.

## **Exhaustive Technical Survey of Key Datasets**

A systematic categorization of the premier open-source industrial predictive maintenance datasets is presented below, utilizing structured specifications to outline their features, limitations, and standard benchmarks.

### **1\. NASA N-CMAPSS Dataset**

| Metadata Field | Technical Specification / Characterization |
| :---- | :---- |
| **Dataset Name** | New Commercial Modular Aero-Propulsion System Simulation (N-CMAPSS)8 |
| **Industry / Use Case** | Aerospace / Turbofan Jet Propulsion Prognostics8 |
| **Link / Source** | [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) \[cite: 10\] |
| **Number of Samples** | Multi-million data points across sub-datasets (e.g., DS02 has files with 0.6M to 1.0M samples per engine unit)11 |
| **Number of Machines / Devices** | Fleet of 9 to 20 turbofan engines across different sub-datasets11 |
| **Sensor Modalities Available** | Total temperatures (T24, T30, T50), total/static pressures (P30, Ps30), physical fan/core speeds (Nf, Nc), corrected fan/core speeds (NRf, NRc), bypass ratio (BPR), bleed enthalpy (htBleed), coolant bleeds (W31, W32), and flight condition settings12 (Vibration: No, Acoustic: No, Electrical: No, Logs: No, Images/Video: No) |
| **Labels Available** | Continuous Remaining Useful Life (![][image1]) and multiclass fault indicators (High-Pressure Compressor, Low-Pressure Compressor, High-Pressure Turbine, Low-Pressure Turbine)9 |
| **Time Resolution / Frequency** | 1 Hz high-frequency sampling across 90-minute flight profiles7 |
| **Multivariate Time-Series Structure** | Yes, 3D time-series tensor formatted as \[samples, time steps, features\] \[cite: 8\] |
| **Supported Analytics Tasks** | Regression (RUL estimation), Multiclass Classification (fault localization), and Survival Analysis7 |
| **Data Quality Assessment** | Exceptionally high-fidelity thermodynamic and aerodynamic simulations under variable flight envelopes (altitude, Mach number, ambient temperature)8 |
| **Known Issues** | Extreme data size creates heavy computational bottlenecks; models typically require downsampling to 0.02 Hz for local model training7 |
| **Research Popularity** | High and rapidly growing; replaces the legacy C-MAPSS dataset as the modern deep learning benchmark8 |
| **Common Benchmark Models** | LSTMs, Transformers, Gated Convolutional Networks, and Temporal Heterogeneous Graph Neural Networks (THGNN)7 |
| **Generation Paradigm** | High-fidelity simulation under real flight conditions using MATLAB Simulink9 |
| **Difficulty Level** | Advanced8 |
| **Best Suited For** | Advanced deep learning research and high-performance system simulation8 |
| **Gold Standard Status** | Yes. It is the premier benchmark for multi-regime propulsion prognostics8 |
| **Status as Outdated / Unrealistic** | No, it represents the modern, expanded iteration of C-MAPSS8 |
| **Proximity to Real Deployment** | High; captures structural flight transitions but operates within simulated limits9 |

The NASA N-CMAPSS dataset addresses the limitations of its predecessor by introducing dynamic operating regimes, representing realistic commercial flight profiles with distinct climb, cruise, and descent phases8. The integration of multiple operating conditions (altitude, Mach number, throttle resolver angle) directly influences the thermodynamic behavior of the turbofan engine, producing highly non-stationary sensor signals8. This complexity forces models to learn the underlying system physics rather than relying on simple temporal correlations8.

### **2\. Petrobras 3W Dataset**

| Metadata Field | Technical Specification / Characterization |
| :---- | :---- |
| **Dataset Name** | Petrobras 3W Dataset (Version 1.0.0 & Version 2.0.0)16 |
| **Industry / Use Case** | Petrochemical / Subsea Petroleum Production Wells16 |
| **Link / Source** | [Petrobras GitHub Repository](https://github.com/petrobras/3W) \[cite: 17\] |
| **Number of Samples** | v1.0.0 has 1,984 instances; v2.0.0 has 2,228 instances16 |
| **Number of Machines / Devices** | 21 to 42 real offshore naturally flowing wells, augmented with simulated instances16 |
| **Sensor Modalities Available** | Pressure (P-PDG, P-TPT, P-MON-CKP), Temperature (T-TPT, T-JUS-CKP), and digital states (DHSV valve trigger, choke openings)18 |
| **Labels Available** | Binary anomaly, anomaly labels, fault category (8 events: slugging, hydrate formation, productivity loss, etc.), transient and steady-state markers16 |
| **Time Resolution / Frequency** | 1 Hz sampling rate16 |
| **Multivariate Time-Series Structure** | Yes, 8 process variables21 |
| **Supported Analytics Tasks** | Supervised classification, unsupervised anomaly detection, and multivariate sequence segmentation16 |
| **Data Quality Assessment** | Highly realistic, preserving genuine industrial noise, outliers, and raw telemetry gaps16 |
| **Known Issues** | High missingness, extreme class imbalance, sensor freeze anomalies, and operational drift across distinct wells16 |
| **Research Popularity** | High in petrochemical AI and upstream research16 |
| **Common Benchmark Models** | Random Forests, SVMs, LSTMs, SDG-Former, CHARM17 |
| **Generation Paradigm** | Hybrid (combines real operational data with OLGA simulator runs and expert-drawn trends)16 |
| **Difficulty Level** | Very Advanced16 |
| **Best Suited For** | Research and production simulation16 |
| **Gold Standard Status** | Yes, the definitive open benchmark for upstream oil/gas anomaly detection16 |
| **Status as Outdated / Unrealistic** | No, it is a highly active and modern repository with a 2025 version 2.0.0 release16 |
| **Proximity to Real Deployment** | Exceptional; directly collected from operational SCADA pipelines16 |

The Petrobras 3W dataset provides an exceptional resource for studying rare, high-impact events in subsea oil wells, where unplanned downtime can result in massive financial losses and environmental hazards16. The dataset's hybrid structure—combining real operational data with high-fidelity simulations—allows researchers to validate models against the actual physical constraints of multiphase flow dynamics16. Modeling these events requires architectures that can handle highly imbalanced classes and learn robust representations from noisy, non-stationary industrial streams16.

### **3\. Paderborn University (PU) Bearing Dataset**

| Metadata Field | Technical Specification / Characterization |
| :---- | :---- |
| **Dataset Name** | Paderborn University (PU) Bearing Dataset26 |
| **Industry / Use Case** | Electric Drives / Rotational Transmission System Monitoring27 |
| **Link / Source** | [Paderborn University Archive](https://ei.uni-paderborn.de/) \[cite: 29\] |
| **Number of Samples** | Millions of data points (trials of 4-second continuous signals at 64 kHz)26 |
| **Number of Machines / Devices** | 32 deep groove ball bearings (6203 type) under varying speeds/loads26 |
| **Sensor Modalities Available** | High-frequency radial/axial vibration accelerometers and motor phase currents27 (Temperature, pressure, acoustic: No) |
| **Labels Available** | Multiclass component damage (inner race, outer race, mixed), healthy baseline, and damage severity tiers30 |
| **Time Resolution / Frequency** | 64 kHz sampling rate30 |
| **Multivariate Time-Series Structure** | Yes, parallel vibration and motor phase-current channels27 |
| **Supported Analytics Tasks** | Multi-class Classification, Anomaly Detection, Out-of-Distribution Domain Adaptation27 |
| **Data Quality Assessment** | High-quality lab dataset combining artificially machined notches with natural wear fatigue26 |
| **Known Issues** | Signal non-stationarity under variable loads and noise, making features transfer-dependent26 |
| **Research Popularity** | Extremely High26 |
| **Common Benchmark Models** | Multimodal 1D/2D CNNs, Random Forests, XGBoost, and Physics-Informed Neural Networks27 |
| **Generation Paradigm** | Real-world physical testbed experiments26 |
| **Difficulty Level** | Advanced26 |
| **Best Suited For** | Research on domain adaptation and non-intrusive current diagnostics27 |
| **Gold Standard Status** | Yes, the standard benchmark for electromechanical transfer learning26 |
| **Status as Outdated / Unrealistic** | No, active development and usage continue across top diagnostics groups26 |
| **Proximity to Real Deployment** | High; current monitoring allows sensorless retrofitting in existing frequency inverters28 |

The Paderborn University bearing dataset stands out due to its inclusion of naturally occurring fatigue defects generated through accelerated life tests26. These natural defects generate much weaker and more complex fault impulses than artificially machined notches, reflecting the actual wear patterns encountered in industrial electric motors26. Furthermore, the simultaneous recording of motor phase currents allows researchers to develop sensorless diagnostics, reducing the hardware cost of predictive maintenance systems28.

### **4\. Case Western Reserve University (CWRU) Bearing Dataset**

| Metadata Field | Technical Specification / Characterization |
| :---- | :---- |
| **Dataset Name** | Case Western Reserve University (CWRU) Bearing Dataset35 |
| **Industry / Use Case** | Rotational Components / Electric Motor Defect Diagnostic Validation34 |
| **Link / Source** | [CWRU Bearing Data Center](https://engineering.case.edu/) \[cite: 35\] |
| **Number of Samples** | Thousands of segmented samples26 |
| **Number of Machines / Devices** | Single test motor under variable loads (0 to 3 HP) and speeds (1730 to 1797 RPM)26 |
| **Sensor Modalities Available** | Vibration accelerometers at the drive-end, fan-end, and base34 (Current, temp, pressure: No) |
| **Labels Available** | Multiclass defect location (inner race, outer race, rolling ball), healthy baseline, and artificial fault diameters26 |
| **Time Resolution / Frequency** | 12 kHz and 48 kHz sampling frequencies26 |
| **Multivariate Time-Series Structure** | Yes, multi-point accelerometer configurations8 |
| **Supported Analytics Tasks** | Binary and multiclass classification, baseline anomaly detection36 |
| **Data Quality Assessment** | High, clean laboratory conditions with minimal background noise26 |
| **Known Issues** | High clean-signal ratio yields "similarity bias"; models easily overfit to artificial EDM notches and achieve \~100% accuracy, failing to generalize to real wear38 |
| **Research Popularity** | Historically the most widely cited dataset in bearing diagnostics26 |
| **Common Benchmark Models** | 1D CNNs, MLPs, SVMs, Random Forests, classical signal transforms (FFT/STFT)26 |
| **Generation Paradigm** | Real physical test motor with EDM-seeded artificial defects26 |
| **Difficulty Level** | Beginner26 |
| **Best Suited For** | Beginners, educational exercises, and baseline software tests26 |
| **Gold Standard Status** | Historically yes, but currently viewed as a basic "sandbox" benchmark rather than a modern research challenge26 |
| **Status as Outdated / Unrealistic** | Outdated and unrealistic; artificially machined defects produce highly predictable, clean impulses that do not reflect real industrial degradation26 |
| **Proximity to Real Deployment** | Low; lacks natural decay kinetics and industrial background noise26 |

While the CWRU bearing dataset has served as the baseline standard for validating rotational fault detection models for decades, it is now considered mostly solved26. The artificially introduced single-point defects generate highly deterministic cyclic shock impulses that are easily classified by basic machine learning models26. Consequently, models trained on CWRU struggle to generalize to real-world industrial environments, where faults develop progressively alongside complex background noise and operational transitions26.

### **5\. Center for Intelligent Maintenance Systems (IMS) Bearing Dataset**

| Metadata Field | Technical Specification / Characterization |
| :---- | :---- |
| **Dataset Name** | IMS Bearing Dataset (University of Cincinnati)10 |
| **Industry / Use Case** | Rotational Components / Wind Turbine and Heavy Industrial Bearings35 |
| **Link / Source** | [NASA Prognostics Repository (IMS Bearings)](https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip) \[cite: 10\] |
| **Number of Samples** | 3 test runs with 2156, 984, and 6324 continuous files35 |
| **Number of Machines / Devices** | 4 bearings loaded on a single shaft35 |
| **Sensor Modalities Available** | Radial high-frequency vibration accelerometers35 (Temperature, current, pressure: No) |
| **Labels Available** | Natural run-to-failure progression (culminating in inner race, outer race, or roller defects)35 |
| **Time Resolution / Frequency** | 20,480 Hz sampling rate, recorded in 1-second bursts every 10 minutes35 |
| **Multivariate Time-Series Structure** | Yes, 4 to 8 accelerometer channels42 |
| **Supported Analytics Tasks** | Regression (![][image1]), Survival Analysis, Anomaly Detection13 |
| **Data Quality Assessment** | High-quality physical degradation data mapping the entire life cycle35 |
| **Known Issues** | Heavy background noise and absence of continuous intermediate fault logs; requires computing statistical health indicators35 |
| **Research Popularity** | Very High45 |
| **Common Benchmark Models** | Stacked Autoencoders (SAE), BiLSTMs, Hidden Markov Models, and Particle Filters46 |
| **Generation Paradigm** | Real physical test rig degradation35 |
| **Difficulty Level** | Intermediate to Advanced40 |
| **Best Suited For** | Deep learning research and continuous degradation modeling40 |
| **Gold Standard Status** | Yes, the definitive physical run-to-failure bearing dataset45 |
| **Status as Outdated / Unrealistic** | No, remains a standard, highly active prognostic benchmark45 |
| **Proximity to Real Deployment** | High; directly mirrors heavy industrial rotary diagnostics40 |

The IMS Bearing dataset provides continuous tracking of natural physical degradation, capturing the progressive transition from a pristine state to structural failure over several weeks of operation35. Unlike datasets with seeded, static faults, the gradual wear in the IMS experiments produces stochastic, non-stationary signal trends35. This makes it an ideal testbed for developing continuous health indicators and validating remaining useful life projections44.

### **6\. ZeMA Hydraulic System Dataset**

| Metadata Field | Technical Specification / Characterization |
| :---- | :---- |
| **Dataset Name** | Condition Monitoring of Hydraulic Systems (ZeMA gGmbH)49 |
| **Industry / Use Case** | Fluid Power / Multi-Component Hydraulic Plant Monitoring50 |
| **Link / Source** | Included in UCI Repository as "Condition Monitoring of Hydraulic Systems"52 |
| **Number of Samples** | 2,205 complete 60-second cycles51 |
| **Number of Machines / Devices** | Multi-component hydraulic test rig (primary and secondary circuits)51 |
| **Sensor Modalities Available** | Pressure (PS1-PS6), Flow (FS1, FS2), Temperature (TS1-TS4), Motor Power (EPS1), Vibration (VS1), and virtual sensors (CE, CP, SE)51 |
| **Labels Available** | Multi-task conditions: Cooler status, Valve status, Pump status, Accumulator status, and Stable operation flag51 |
| **Time Resolution / Frequency** | Multirate: Pressure/Power at 100 Hz, Flow at 10 Hz, Temperature/Vibration at 1 Hz51 |
| **Multivariate Time-Series Structure** | Yes51 |
| **Supported Analytics Tasks** | Multitask Classification, Regression, and Multimodal Feature Fusion49 |
| **Data Quality Assessment** | Very clean, high-precision industrial data with systematically induced reversible degradation levels49 |
| **Known Issues** | Non-synchronized sampling frequencies require interpolation or parallel multi-scale feature extractors54 |
| **Research Popularity** | High in multi-task learning and sensor fusion circles49 |
| **Common Benchmark Models** | Random Forests, CatBoost, LSTMs, and Variational Autoencoders (VAE)49 |
| **Generation Paradigm** | Real-world physical test stand data51 |
| **Difficulty Level** | Intermediate53 |
| **Best Suited For** | Research in sensor fusion, domain adaptation, and multi-task diagnostics50 |
| **Gold Standard Status** | Yes, the standard benchmark for multi-component hydraulic monitoring49 |
| **Status as Outdated / Unrealistic** | No49 |
| **Proximity to Real Deployment** | High; directly maps to machinery such as hydraulic presses and reach trucks50 |

The ZeMA Hydraulic System dataset provides a highly structured environment for validating multi-task diagnostics49. By simultaneously tracking the degradation states of four distinct hydraulic components (cooler, valve, pump, accumulator), the dataset challenges models to perform joint feature extraction and multi-label classification51. Additionally, its multirate sensor configurations mirror the diverse sampling speeds typical of real-world industrial control networks51.

### **7\. AI4I 2020 Predictive Maintenance Dataset**

| Metadata Field | Technical Specification / Characterization |
| :---- | :---- |
| **Dataset Name** | AI4I 2020 Predictive Maintenance Dataset57 |
| **Industry / Use Case** | Machine Tools / CNC Milling Machine Snapshots59 |
| **Link / Source** | [UCI Machine Learning Repository (AI4I 2020\)](http://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) \[cite: 58\] |
| **Number of Samples** | 10,000 data points57 |
| **Number of Machines / Devices** | Single simulated milling machine59 |
| **Sensor Modalities Available** | Air temperature, process temperature, RPM, torque, tool wear, and product quality variants58 (No raw high-frequency vibration, acoustics, or current waveforms) |
| **Labels Available** | Binary machine failure flag, and 5 independent failure modes (TWF, HDF, PWF, OSF, RNF)58 |
| **Time Resolution / Frequency** | Tabular static snapshots58 |
| **Multivariate Time-Series Structure** | Yes, 14 feature columns58 |
| **Supported Analytics Tasks** | Binary/Multiclass Classification, Explainable AI (XAI)57 |
| **Data Quality Assessment** | Extremely clean, zero missing values58 |
| **Known Issues** | Lacks raw continuous waveforms or complex time-series transitions, reducing its utility for deep sequence learning58 |
| **Research Popularity** | High for education, introductory tutorials, and quick ML model tests59 |
| **Common Benchmark Models** | LightGBM, XGBoost, CatBoost, and Random Forests62 |
| **Generation Paradigm** | Synthetically generated based on physical equations57 |
| **Difficulty Level** | Beginner59 |
| **Best Suited For** | Beginners, introductory portfolio projects, and rapid ML algorithm benchmarking59 |
| **Gold Standard Status** | No, too simplistic for advanced research59 |
| **Status as Outdated / Unrealistic** | Unrealistic for time-series modeling due to its tabularized static nature58 |
| **Proximity to Real Deployment** | Low; does not capture raw high-frequency sensor streams or environmental noise58 |

While the AI4I 2020 dataset does not capture the high-frequency dynamics of real sensor streams, its clean, tabular structure makes it highly valuable for educational purposes and rapid prototyping58. By framing predictive maintenance as a classic classification task with five distinct failure modes, it allows developers to quickly train baseline models, explore feature importance metrics, and evaluate explainable AI techniques59.

### **8\. Microsoft Azure Predictive Maintenance Dataset**

| Metadata Field | Technical Specification / Characterization |
| :---- | :---- |
| **Dataset Name** | Microsoft Azure Predictive Maintenance Dataset (Fidan Synthetic)43 |
| **Industry / Use Case** | Enterprise Assets / Heavy Rotating Plant Fleet Management65 |
| **Link / Source** | [Kaggle (Azure Predictive Maintenance)](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance) \[cite: 43, 66\] |
| **Number of Samples** | 876,100 hourly telemetry rows64 |
| **Number of Machines / Devices** | Fleet of 100 identical machines tracked over a 1-year period66 |
| **Sensor Modalities Available** | Hourly sensor averages (voltage, rotation, pressure, vibration), coupled with error logs, maintenance records, and machine metadata66 |
| **Labels Available** | Multiclass failures (component 1 to 4 failures), error codes, and maintenance logs64 |
| **Time Resolution / Frequency** | Hourly averaged records66 |
| **Multivariate Time-Series Structure** | Yes, relational time-series66 |
| **Supported Analytics Tasks** | Multi-class classification, RUL regression, and Survival Analysis13 |
| **Data Quality Assessment** | Highly structured, formatted relational database64 |
| **Known Issues** | Synthetic telemetry sequences; extreme class imbalance (\~0.09% failure rate) requires structured sliding-window aggregation to capture transient features64 |
| **Research Popularity** | High for MLOps, relational data pipeline tests, and enterprise engineering portfolios64 |
| **Common Benchmark Models** | LightGBM, CatBoost, LSTMs, Random Survival Forests (RSF), and DeepHit13 |
| **Generation Paradigm** | Synthetically generated based on predefined asset behavior rules64 |
| **Difficulty Level** | Intermediate64 |
| **Best Suited For** | Relational data engineering, production pipeline modeling, and portfolio projects64 |
| **Gold Standard Status** | Yes, specifically for testing relational data integration in enterprise PdM65 |
| **Status as Outdated / Unrealistic** | No, remains highly useful for testing relational schemas64 |
| **Proximity to Real Deployment** | Medium-High; mirrors standard enterprise asset management (EAM) and SCADA databases66 |

The Microsoft Azure dataset stands out because of its relational database structure66. By packaging raw telemetry alongside error logs, routine maintenance write-ups, and failure records, it represents the typical multi-table schemas encountered in modern industrial IT systems66. This setup challenges data engineers to develop robust feature extraction pipelines, aggregating disparate, asynchronous logs into structured historical tables suitable for machine learning64.

### **9\. APS Failure at Scania Trucks Dataset**

| Metadata Field | Technical Specification / Characterization |
| :---- | :---- |
| **Dataset Name** | Air Pressure System (APS) Failure at Scania Trucks Dataset67 |
| **Industry / Use Case** | Automotive Fleet Operations / Heavy Truck Pneumatic Air Brake Diagnostics67 |
| **Link / Source** | [UCI Machine Learning Repository (Scania Trucks)](https://www.mdpi.com/2079-9292/14/24/4957) \[cite: 67\] |
| **Number of Samples** | 76,000 instances (60,000 train, 16,000 test)67 |
| **Number of Machines / Devices** | Fleet of heavy Scania trucks in active daily service67 |
| **Sensor Modalities Available** | 171 numerical features and histograms representing pneumatic, thermal, electrical, and mechanical states67 (Vibration, acoustic: No, Logs: No) |
| **Labels Available** | Highly cost-sensitive binary classification: Positive (APS component failure) vs. Negative (non-APS component failure)67 |
| **Time Resolution / Frequency** | Event-based static snapshots67 |
| **Multivariate Time-Series Structure** | Yes, high-dimensional tabular data67 |
| **Supported Analytics Tasks** | Binary Classification, Cost-Sensitive Optimization, Anomaly Detection67 |
| **Data Quality Assessment** | High-quality real-world fleet data, but heavily anonymized and containing massive missing values67 |
| **Known Issues** | High missingness (\>50% missingness in several features), extreme class imbalance (1.66% positive rate)67 |
| **Research Popularity** | Very High in cost-sensitive and matrix-completion studies71 |
| **Common Benchmark Models** | LightGBM, XGBoost, CatBoost, DNNs with Focal Loss, and SMOTE63 |
| **Generation Paradigm** | Real-world operational fleet data67 |
| **Difficulty Level** | Advanced67 |
| **Best Suited For** | Research on imbalanced classifiers, tabular imputation, and cost-sensitive optimization67 |
| **Gold Standard Status** | Yes, the primary benchmark for heavy fleet cost-sensitive analytics71 |
| **Status as Outdated / Unrealistic** | No, remains highly relevant and active67 |
| **Proximity to Real Deployment** | Extremely High; directly collected from operational vehicle telematics67 |

The Scania Trucks APS dataset provides a highly realistic, challenging testbed for cost-sensitive learning67. Because missing an actual brake system failure has critical safety implications, the cost of a false negative is significantly higher than that of a false positive68. This asymmetry forces developers to design and evaluate models using business-centric metrics rather than standard accuracy scores, aligning directly with real-world fleet operational constraints70.

### **10\. Metro PT Dataset**

| Metadata Field | Technical Specification / Characterization |
| :---- | :---- |
| **Dataset Name** | Metro PT Dataset (Metro do Porto, Portugal)72 |
| **Industry / Use Case** | Rail Transit / Railway Air Compressor Predictive Maintenance72 |
| **Link / Source** | [Kaggle / Zenodo (Metro PT)](https://www.researchgate.net/publication/394396672_An_Explainable_Machine_Learning_Framework_for_Railway_Predictive_Maintenance_using_Data_Streams_from_the_Metro_Operator_of_Portugal) \[cite: 72\] |
| **Number of Samples** | Millions of records spanning 30 days69 |
| **Number of Machines / Devices** | Fleet of active metro trains (Metro do Porto)72 |
| **Sensor Modalities Available** | Pressures (air pressure, barometric), temperature (oil temp), motor current, and digital valve triggers72 (No raw high-frequency acoustics or images) |
| **Labels Available** | Binary anomalies, operational control states, and explicit failures72 |
| **Time Resolution / Frequency** | High-frequency continuous telemetry streams69 |
| **Multivariate Time-Series Structure** | Yes, multivariate data streams69 |
| **Supported Analytics Tasks** | Streaming Anomaly Detection, Real-time Classification, and Explainable AI (XAI)72 |
| **Data Quality Assessment** | High, clean, raw continuous data representing actual public service72 |
| **Known Issues** | Non-stationary noise from varied routes, and subtle failure precursors that are easily masked72 |
| **Research Popularity** | High and growing; extensively used in stream classification and XAI research72 |
| **Common Benchmark Models** | Hoeffding Stream Trees, MSCRED, MOMENT, and Joint Embedding Predictive Architectures (JEPA)24 |
| **Generation Paradigm** | Real-world public transit operations72 |
| **Difficulty Level** | Advanced74 |
| **Best Suited For** | Research on stream analytics, real-time diagnostics, and XAI72 |
| **Gold Standard Status** | Yes, the premier dataset for real-time railway stream prognostics72 |
| **Status as Outdated / Unrealistic** | No, highly modern and active72 |
| **Proximity to Real Deployment** | Exceptionally High; directly collected from onboard vehicle diagnostic buses72 |

The Metro PT dataset provides an invaluable open resource for real-time streaming analytics72. Collected from active passenger rail operations, the telemetry signals are subject to continuous environment transitions (e.g., elevation changes and speed adjustments) that produce highly dynamic signal behavior72. Modeling these data streams requires sliding-window architectures and incremental learning techniques designed to handle continuous distribution shifts and isolate genuine degradation from transient operational noise72.

## **Second and Third-Order Analytical Syntheses**

### **The Sim-to-Real Generalization Barrier and Notch-Induced Bias**

A fundamental challenge in condition monitoring and machine fault diagnostics is the performance drop that models experience when transitioning from laboratory benchmarks to active production environments26. This barrier is highly evident when analyzing bearing diagnostics26. Models trained on the Case Western Reserve University (CWRU) dataset consistently achieve classification accuracies of 99% to 100% using lightweight 1D CNNs or basic Support Vector Machines38.  
However, when these exact models are evaluated on the Paderborn University (PU) dataset or applied to active machinery in industrial plants, their diagnostic accuracy drops significantly26.  
This diagnostic discrepancy stems from the difference in how faults are generated26. The CWRU dataset relies on artificially introduced defects created via electro-discharge machining (EDM)26. These machined notches produce clean, deterministic, high-amplitude impact signatures at precise geometric frequencies (e.g., Inner and Outer Race Ball Pass Frequencies, ![][image2] and ![][image3])26. These distinct patterns make it easy for machine learning models to identify faults, but can lead to a "similarity bias" where models overfit to these crisp notches rather than learning actual degradation dynamics39.  
In contrast, the Paderborn University dataset utilizes both artificial notches and natural fatigue defects generated through accelerated lifetime tests26. Naturally occurring pitting and fatigue develop progressively, producing diffuse, stochastic, and weak acoustic emissions26. When these signals are combined with real background noise, varying torque loads, and rotational speed changes, the clean impulses disappear, causing standard diagnostic classifiers to fail26.  
Addressing this generalization barrier requires using domain adaptation, transfer learning, or physics-informed loss functions27. For example, integrating characteristic bearing fault equations directly into neural network loss formulations allows models to generalize better across different environments and operating conditions33:  
![][image4]  
![][image5]

### **Multi-Sensor Data Fusion Architectures vs. Edge Constraints**

To improve failure detection and Remaining Useful Life (![][image1]) estimation, industrial architectures are increasingly integrating multi-sensor arrays that capture diverse modalities, including vibration, temperature, and electrical current27. This multi-modal approach helps capture physical changes across multiple regimes76. For example, in the Paderborn University dataset, integrating high-speed vibration sensors with motor phase-current measurements helps detect structural motor anomalies more reliably than relying on a single modality27.  
However, streaming high-frequency multi-sensor signals creates significant challenges for edge processing and real-time inference on resource-constrained hardware27. Acquiring multi-channel vibration at 64 kHz (as in the Paderborn dataset) or 20.48 kHz (as in the IMS dataset) generates massive data volumes that quickly exhaust edge storage, bus bandwidth, and memory allocation32. This has driven the development of several model designs to balance diagnostic accuracy with computational efficiency:

1. **Early vs. Late Multi-Sensor Fusion:** Early fusion concatenates raw, high-frequency waveforms at the input layer27. While this preserves fine-grained phase relationships, it requires large model architectures and creates latency bottlenecks27. Late fusion extracts compact, low-frequency features from each sensor independently (e.g., time-domain statistics like Root Mean Square or kurtosis, and frequency-domain spectral energy bands) before fusing them in a shared latent space27. This reduces edge computational requirements and helps models handle missing data when a sensor fails27.  
2. **Phase-Current Non-Intrusive Diagnostics:** To avoid the cost and installation challenges of mounting external accelerometers in tight industrial environments, research is focusing on Motor Current Signature Analysis (MCSA)28. MCSA reads current signals directly from existing frequency inverters28. While current signals have a lower signal-to-noise ratio than direct vibration measurements, training deep learning models on phase-current data can identify internal rotor and bearing anomalies without requiring additional sensors28.  
3. **Optimized Quantization and Sliding Windows:** To run deep learning models on microcontrollers, raw time-series inputs are typically processed using sliding windows (e.g., 25 to 60 cycles)79. These inputs are then passed to optimized, low-bitwidth models (such as INT8-quantized TensorFlow Lite configurations)62. This temporal slicing maintains a balance between capturing longer-term degradation patterns and keeping inference latency low62.

### **Non-Stationarity, Survival Analysis, and Stochastic Asset Decay**

Many academic studies treat Remaining Useful Life (![][image1]) estimation as a straightforward regression task with linear target labeling7. However, this approach can struggle in real industrial deployments3. In practice, assets rarely experience a clean, linear decay3. Instead, physical degradation is stochastic, non-stationary, and highly dependent on changing operating regimes, environment conditions, and maintenance events6.  
This is clearly illustrated in the NASA N-CMAPSS and Petrobras 3W datasets8. Turbofan engines operate across distinct flight regimes (climb, cruise, descent) that subject components to varying thermal and aerodynamic stresses8. Similarly, offshore oil wells experience varying flow velocities, choke valve adjustments, and gas lift interventions16. This operational variability means that elapsed time alone is a poor indicator of asset condition3.  
To address this, modern predictive maintenance architectures are adopting more robust probabilistic and survival analysis models13:

1. **Piecewise Degradation Target Formulation:** Rather than predicting linear decay from day one, models typically use a piecewise linear target7. This approach assumes the asset remains in a normal, healthy state with a constant ![][image1] until a degradation onset threshold is reached, after which the ![][image1] decays linearly7. This prevents models from trying to learn early-stage degradation when no physical wear has occurred7.  
2. **Survival Analysis and Hazard Rates:** Survival analysis models the *probability* of an asset surviving past a given time ![][image6] rather than predicting a single deterministic failure point13. This uses hazard function modeling to represent the instantaneous rate of failure conditioned on survival up to ![][image6]13. This provides maintenance planners with a confidence-bound curve of failure probabilities, helping them make more informed scheduling decisions under operational uncertainty13:

![][image7]

3. **Stochastic Process Formulation:** Advanced approaches treat continuous wear-and-tear using stochastic models like Gamma processes or Inverse Gaussian processes6. These methods capture the variability of progressive degradation and help identify subtle premonitory indicators of failure even in noisy, non-stationary industrial telemetry streams6.

## **Shortlists and Strategic Recommendations**

### **The Definitive Global Shortlist**

A curated evaluation of the top five predictive maintenance datasets is presented below, ranked by their data quality, completeness, relevance to industrial settings, and benchmarking value.

| Rank | Dataset Name | Dominant Modalities | Prime Analytical Paradigm | Suitability for Machine Learning Paradigms | Deployment Realism |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **1** | **Petrobras 3W** \[cite: 16\] | Multi-point pressures, temperatures, and valve control state signals16 | Multiclass time-series classification, streaming anomaly detection16 | High-capacity Recurrent Networks (LSTMs, GRUs), Transformers, and Self-Supervised Embeddings (JEPA)23 | **Exceptional**; captures actual raw control-room sensor errors and missing values16 |
| **2** | **NASA N-CMAPSS** \[cite: 8\] | Aerodynamic/thermodynamic pressures, temperatures, and rotational speeds12 | Piecewise linear ![][image1] regression, operational regime classification7 | Temporal Convolutional Networks (TCNs), deep LSTMs, Transformers, and Graph Neural Networks (GNNs)7 | **High**; models complex thermodynamic interactions under dynamic flight conditions8 |
| **3** | **Paderborn Bearing** \[cite: 26\] | High-frequency radial/axial vibration and two-phase motor currents27 | Multiclass diagnostics, domain generalization, vibration-current fusion27 | Multi-branch 1D/2D CNNs, Wavelet-based architectures, and gradient boosted trees (XGBoost)27 | **High**; captures realistic fatigue-induced failures and sensorless current patterns26 |
| **4** | **Metro PT Stream** \[cite: 72\] | Pneumatic pressures, oil temperatures, current, and digital valve triggers72 | Continuous streaming classification, explainability, online anomaly detection72 | Incremental classifiers (Hoeffding Trees), real-time sequential models, and XAI frameworks72 | **Exceptional**; collected directly from passenger rail transport operations72 |
| **5** | **ZeMA Hydraulics** \[cite: 49\] | Multirate pressures, fluid flows, temperatures, power, and vibration51 | Multitask classification, multirate fusion, transfer learning49 | Parallel multi-scale CNNs, classical ensembles (Random Forest, CatBoost), and VAEs49 | **High**; physical testbed with highly verified multi-component failure points51 |

### **Engineering Target Recommendations**

#### **1\. Educational and Prototyping Projects (Beginner / Intermediate)**

For developers entering the industrial AI space, the **AI4I 2020 Predictive Maintenance Dataset** is highly recommended59. Its clean, tabular structure with zero missing values allows rapid prototyping of classification pipelines without requiring complex temporal feature engineering58. Traditional gradient boosted tree algorithms—including LightGBM and XGBoost—easily scale on this dataset to benchmark binary and multiclass failures62.  
To transition toward time-series and signal diagnostics, the **Case Western Reserve University (CWRU) Bearing Dataset** provides an accessible baseline for applying fast Fourier transforms and training lightweight 1D CNNs26.

#### **2\. Advanced Research Projects**

Researchers focused on state-of-the-art architectures should utilize **NASA N-CMAPSS** and the **Paderborn University Bearing Dataset**8. N-CMAPSS is ideal for evaluating sequential deep learning models like multi-head self-attention Transformers, Temporal Convolutional Networks, and Graph Neural Networks under highly dynamic operating conditions7.  
The Paderborn dataset provides a rigorous testbed for domain generalization, transfer learning, and physics-informed neural networks27. These models must generalize across distinct load torques and radial forces using a mix of vibration and motor phase-current signals27.

#### **3\. Deep Learning Architectures (LSTMs, GRUs, Transformers)**

For deep sequential architectures, **IMS Bearings** and **NASA N-CMAPSS** are the premier choices8. These datasets capture continuous degradation over long sequences, which is ideal for testing the gating mechanisms of LSTMs and the self-attention matrices of Transformers7.  
Additionally, the **Metro PT Dataset** and the **3W Dataset** are well suited for advanced self-supervised representation architectures, such as Joint Embedding Predictive Architectures (JEPA) and masked autoencoders, which learn robust industrial representations from noisy temporal signals16.

#### **4\. Real-World Deployment and Realism**

Projects targeting production-ready predictive maintenance should prioritize the **Petrobras 3W Dataset**, **APS Failure at Scania Trucks**, and the **Metro PT Dataset**16. These datasets avoid clean, idealized telemetry, exposing models to realistic data challenges:

* **Petrobras 3W** contains real operational missingness, noisy signals, and severe class imbalance16.  
* **Scania Trucks APS** provides a high-dimensional, highly imbalanced (\~1.66% failures) environment where missing an anomaly has critical cost implications67.  
* **Metro PT** simulates active data streaming environments, which is ideal for testing real-time inference, model drift, and explainable AI pipelines72.

## **8-Week Solo Engineering Project Roadmap**

The following roadmap outlines an end-to-end predictive maintenance project designed for a solo engineering student to complete over an 8-week period. The project builds a complete, production-grade diagnostic pipeline, transitioning from baseline machine learning models to advanced deep learning architectures and edge deployment.

### **Deployed Edge-Enabled Predictive Maintenance Pipeline**

 \[Phase 1: Week 1-2\] ───► \[Phase 2: Week 3-4\] ───► \[Phase 3: Week 5-6\] ───► \[Phase 4: Week 7-8\]  
  \- Multi-sensor EDA      \- Wavelet Denoising      \- PyTorch LSTMs/TCNs     \- INT8 Quantization  
  \- Linear/Tree Baselines  \- Feature Extraction     \- Attention/XAI (SHAP)   \- FastAPI & React UI

### **Project Timeline and Milestones**

| Phase / Weeks | Analytical Focus & Operational Tasks | Primary Datasets | Deliverables & Tech Stack |
| :---- | :---- | :---- | :---- |
| **Phase 1: Foundations & Exploratory Analysis** (Weeks 1–2) | Data ingestion, handling missing values, temporal alignment, and establishing baseline statistical profiles3. | **AI4I 2020** & **CWRU Bearings** \[cite: 35, 58\] | • Interactive EDA notebooks. • Baseline models (Logistic Regression, Random Forest, XGBoost)5. • Dockerized development environment with Pandas, Scikit-learn, and Optuna63. |
| **Phase 2: Signal Processing & Classical ML** (Weeks 3–4) | Advanced feature engineering on raw waveforms, frequency-domain transformations, and sliding-window feature aggregation35. | **ZeMA Hydraulics** & **IMS Bearings** \[cite: 35, 51\] | • Data prep pipelines with Hilbert and Wavelet transforms22. • Feature-based models (LightGBM) optimized with Bayesian search63. • SHAP/LIME feature importance evaluations62. |
| **Phase 3: Deep Learning & Temporal Fusion** (Weeks 5–6) | Building deep temporal architectures to process raw, multi-sensor sequences directly without manual feature engineering38. | **NASA C-MAPSS** (FD001/FD004)15 | • Custom PyTorch architectures (BiLSTM, Gated TCN, and multi-head attention blocks)7. • Asymmetric loss optimization tracking ![][image1]76. |
| **Phase 4: Model Quantization & Edge Deployment** (Weeks 7–8) | Optimizing models for resource-constrained edge systems and building a real-time monitoring interface62. | **Metro PT Stream** or **Petrobras 3W** \[cite: 16, 72\] | • Quantized model checkpoints (TFLite/ONNX INT8)62. • High-performance FastAPI backend63. • Interactive React dashboard with real-time anomaly alerts84. |

### **Phase 1: Weeks 1–2 — Analytical Foundations, Exploratory Data Analysis, and Tabular Baselines**

The project begins by setting up a reproducible development environment inside a Docker container using Python, PyTorch, and CUDA. The initial analytical focus is on handling class imbalance and exploratory data analysis using the **AI4I 2020 Predictive Maintenance Dataset**58. The student implements data pipelines to manage severe class imbalances (e.g., using SMOTE or custom class weights) and visualizes features across the dataset's five failure modes58.  
Using Scikit-learn, the student trains baseline classifiers (Logistic Regression, Random Forest, and XGBoost) to classify machine failures5. Key hyperparameter tuning is automated using Optuna with Bayesian optimization (Tree-structured Parzen Estimator) to maximize the Macro F1-score63. To explore high-frequency waveforms, the student processes the **CWRU Bearing Dataset**, loading raw vibration signals and analyzing the spectral profiles of healthy versus damaged bearings under varying motor loads26.

### **Phase 2: Weeks 3–4 — Digital Signal Processing, Feature Engineering, and Robust Ensembles**

During this phase, the project moves from static tabular data to high-frequency physical waveforms using the **ZeMA Hydraulics** and **IMS Bearing** datasets35.  
The student implements digital signal processing pipelines:

1. **Denoising:** Applying bandpass filters and Continuous Wavelet Transforms to isolate weak fault frequencies from background operating noise22.  
2. **Envelope Analysis:** Computing the Hilbert transform to extract the signal envelope, making physical impact peaks easier to identify22.  
3. **Statistical Feature Extraction:** Building a sliding-window feature extractor that computes time-domain metrics (Root Mean Square, kurtosis, skewness, crest factor) and frequency-domain metrics (spectral centroid, dominant peak frequencies)19.

Python  
import numpy as np  
import scipy.stats as stats

def extract\_time\_domain\_features(signal\_window: np.ndarray) \-\> dict:  
    """  
    Extracts key time-domain statistical indicators from a high-frequency  
    vibration signal window to track structural degradation.  
    """  
    rms \= np.sqrt(np.mean(signal\_window\*\*2))  
    peak \= np.max(np.abs(signal\_window))  
    crest\_factor \= peak / rms if rms \> 0 else 0  
    kurt \= stats.kurtosis(signal\_window)  
    skew \= stats.skew(signal\_window)  
    p2p \= np.max(signal\_window) \- np.min(signal\_window)  
      
    return {  
        "rms": rms,  
        "crest\_factor": crest\_factor,  
        "kurtosis": kurt,  
        "skewness": skew,  
        "peak\_to\_peak": p2p  
    }

The student handles the multirate sensor structures of the ZeMA dataset by upsampling lower-frequency channels to align with the high-frequency pressure channels53. Using these extracted feature vectors, the student trains a multi-output LightGBM model to predict conditions across all four hydraulic components simultaneously49. Model predictions are evaluated using SHAP (Shapley Additive exPlanations) to identify which sensors contribute most to predicting component failures62.

### **Phase 3: Weeks 5–6 — Deep Sequential Networks, Attention Mechanisms, and RUL Estimation**

This phase focuses on training deep learning models to process raw sequential telemetry directly, using **NASA C-MAPSS (FD001 & FD004)**15. The student implements a PyTorch dataset pipeline using sliding windows with configurable lengths and strides to prepare inputs for sequence models79.  
The student builds and trains three deep learning architectures in PyTorch:

1. **1D CNN-LSTM:** A 1D convolutional network extracts spatial features from the multi-sensor stream, which are then passed to LSTM layers to capture temporal dependencies80.  
2. **Temporal Convolutional Network (TCN):** A network utilizing dilated causal convolutions to model long-range temporal patterns efficiently without vanishing gradient issues41.  
3. **Transformer Encoder:** A multi-head self-attention network designed to capture dynamic interactions across sensors and time steps7.

To optimize model predictions for industrial maintenance schedules, the student implements a custom asymmetric loss function76. This function penalizes overestimating the Remaining Useful Life (![][image1]) more heavily than underestimating it, since overestimating ![][image1] carries a higher risk of unexpected catastrophic failure76:  
![][image8]  
where ![][image9] represents the estimation error76. The student monitors training performance across models using TensorBoard, tracking Root Mean Square Error (![][image10]) alongside this asymmetric loss76.

### **Phase 4: Weeks 7–8 — MLOps Optimization, Edge Inversion, and Live Dashboard Deployment**

The final phase focuses on optimization, deployment, and building the user interface, using the **Metro PT Dataset** or **Petrobras 3W** stream16.  
The student optimizes the PyTorch models for edge execution:

1. **Quantization:** Applying Post-Training Quantization (PTQ) to convert model weights from FP32 to INT8, significantly reducing the model footprint and inference latency62.  
2. **ONNX Export:** Exporting the quantized checkpoints to the Open Neural Network Exchange (ONNX) format for optimized runtime execution63.

Python  
import torch  
import torch.quantization

def apply\_static\_quantization(model: torch.nn.Module, calibration\_loader) \-\> torch.nn.Module:  
    """  
    Applies post-training static INT8 quantization to optimize deep learning  
    models for high-speed inference on resource-constrained edge systems.  
    """  
    model.eval()  
    model.qconfig \= torch.quantization.get\_default\_qconfig('fbgemm')  
    prepared\_model \= torch.quantization.prepare(model, inplace=False)  
      
    \# Calibration pass with representative operational data  
    with torch.no\_grad():  
        for inputs, \_ in calibration\_loader:  
            prepared\_model(inputs)  
              
    quantized\_model \= torch.quantization.convert(prepared\_model, inplace=False)  
    return quantized\_model

The student builds a high-performance, asynchronous FastAPI backend in Python to ingest telemetry streams, execute model inference, and serve predictions63. An alert manager triggers maintenance warnings when predicted failure probabilities cross a specified decision threshold63.  
Finally, the student develops a React frontend dashboard utilizing Recharts to visualize real-time sensor streams, display component health scores, and show remaining useful life projections84. This completes the end-to-end solo engineering project, providing a production-ready predictive maintenance pipeline.

## **Conclusions**

This research report outlines the key datasets and analytical techniques shaping the field of predictive maintenance and industrial diagnostics. Transitioning from basic machine learning models to deep sequential networks (such as Transformers and LSTMs) and self-supervised representations allows systems to detect anomalies and predict Remaining Useful Life more accurately4.  
However, deploying these models successfully requires addressing the sim-to-real generalization gap, handling highly imbalanced datasets, and optimizing architectures to run within edge hardware constraints26.  
By building end-to-end pipelines that integrate robust signal processing, physics-informed constraints, and optimized edge deployment, engineers can deliver reliable decision support systems that minimize unplanned downtime and optimize asset lifecycles in industrial environments33.

#### **Works cited**

1. A Survey of Predictive Maintenance: Systems, Purposes and Approaches \- arXiv, [https://arxiv.org/html/1912.07383v2](https://arxiv.org/html/1912.07383v2)  
2. A Survey of Predictive Maintenance Methods: An Analysis of Prognostics via Classification and Regression \- ResearchGate, [https://www.researchgate.net/publication/393022844\_A\_Survey\_of\_Predictive\_Maintenance\_Methods\_An\_Analysis\_of\_Prognostics\_via\_Classification\_and\_Regression](https://www.researchgate.net/publication/393022844_A_Survey_of_Predictive_Maintenance_Methods_An_Analysis_of_Prognostics_via_Classification_and_Regression)  
3. Remaining Useful Life Prediction Based on Deep Learning: A Survey \- PMC, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11174398/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11174398/)  
4. Predictive maintenance in Industrial Systems Using Machine Learning : A Review, [https://iapress.org/index.php/soic/article/view/3058](https://iapress.org/index.php/soic/article/view/3058)  
5. Automated Machine Learning for Remaining Useful Life Predictions \- arXiv, [https://arxiv.org/html/2306.12215v2](https://arxiv.org/html/2306.12215v2)  
6. Predictive maintenance in Industry 4.0: a survey of planning models and machine learning techniques \- PMC, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11157603/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11157603/)  
7. A decentralized model for fault classification and remaining useful life estimation of physical assets | Journal of Quality in Maintenance Engineering \- Emerald Insight, [https://www.emerald.com/jqme/article/doi/10.1108/JQME-10-2025-0124/1363158/A-decentralized-model-for-fault-classification-and](https://www.emerald.com/jqme/article/doi/10.1108/JQME-10-2025-0124/1363158/A-decentralized-model-for-fault-classification-and)  
8. A Matrix-Statistics-Aware Attention Mechanism for Robust RUL Estimation in Aero-Engines, [https://www.mdpi.com/2076-3417/16/1/169](https://www.mdpi.com/2076-3417/16/1/169)  
9. C-MAPSS Aircraft Engine Simulator Data \- Dataset \- NASA Open Data Portal, [https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data](https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data)  
10. Prognostics Center of Excellence Data Set Repository \- NASA, [https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)  
11. Evolutionary Optimization of Spiking Neural P Systems for Remaining Useful Life Prediction \- iris@unitn, [https://iris.unitn.it/retrieve/handle/11572/335935/537170/algorithms-15-00098.pdf](https://iris.unitn.it/retrieve/handle/11572/335935/537170/algorithms-15-00098.pdf)  
12. Temporal and Heterogeneous Graph Neural Network for Remaining Useful Life Prediction \- arXiv, [https://arxiv.org/html/2405.04336v3](https://arxiv.org/html/2405.04336v3)  
13. Unified Evaluation of Predictive Models for Failure Prediction \- CEUR-WS.org, [https://ceur-ws.org/Vol-4192/XAI4Science-paper8.pdf](https://ceur-ws.org/Vol-4192/XAI4Science-paper8.pdf)  
14. predictive-maintenance · GitHub Topics, [https://github.com/topics/predictive-maintenance](https://github.com/topics/predictive-maintenance)  
15. CMAPSS Jet Engine Simulated Data \- Dataset \- NASA Open Data Portal, [https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)  
16. A Machine Learning-Centric Taxonomy and Structured Characterization of Public Datasets for Upstream Oil and Gas \- MDPI, [https://www.mdpi.com/2504-2289/10/6/188](https://www.mdpi.com/2504-2289/10/6/188)  
17. (PDF) OTC-36194-MS Exploratory Analysis of the 3W Dataset for Detecting Operational Failures in Oil Wells Using Machine Learning Techniques \- ResearchGate, [https://www.researchgate.net/publication/397205628\_OTC-36194-MS\_Exploratory\_Analysis\_of\_the\_3W\_Dataset\_for\_Detecting\_Operational\_Failures\_in\_Oil\_Wells\_Using\_Machine\_Learning\_Techniques](https://www.researchgate.net/publication/397205628_OTC-36194-MS_Exploratory_Analysis_of_the_3W_Dataset_for_Detecting_Operational_Failures_in_Oil_Wells_Using_Machine_Learning_Techniques)  
18. (PDF) An Lstm-Based Anomaly Detection on Subsea Oil-Producing Well \- ResearchGate, [https://www.researchgate.net/publication/399179142\_An\_Lstm-Based\_Anomaly\_Detection\_on\_Subsea\_Oil-Producing\_Well](https://www.researchgate.net/publication/399179142_An_Lstm-Based_Anomaly_Detection_on_Subsea_Oil-Producing_Well)  
19. Classification of undesirable events in oil well operation \- Johannes Jäschke, [https://jaschke.folk.ntnu.no/preprints/2021/TuranClassification\_PC/009.pdf](https://jaschke.folk.ntnu.no/preprints/2021/TuranClassification_PC/009.pdf)  
20. Data-driven Detection and Identification of Undesirable Events in Subsea Oil Wells \- UPV, [https://personales.upv.es/thinkmind/dl/conferences/sensordevices/sensordevices\_2021/sensordevices\_2021\_1\_10\_28039.pdf](https://personales.upv.es/thinkmind/dl/conferences/sensordevices/sensordevices_2021/sensordevices_2021_1_10_28039.pdf)  
21. A Review of Data-Driven Prediction of Undesirable Events in Offshore Oil Wells—Based on the Public 3W Benchmark and Time-Series Deep Learning \- SCIRP, [https://www.scirp.org/journal/paperinformation?paperid=151428](https://www.scirp.org/journal/paperinformation?paperid=151428)  
22. A realistic and public dataset with rare undesirable real events in oil wells \- ResearchGate, [https://www.researchgate.net/publication/334148489\_A\_realistic\_and\_public\_dataset\_with\_rare\_undesirable\_real\_events\_in\_oil\_wells](https://www.researchgate.net/publication/334148489_A_realistic_and_public_dataset_with_rare_undesirable_real_events_in_oil_wells)  
23. An LSTM-Based Anomaly Detection on Subsea Oil-Producing Well, [https://journal.lemigas.esdm.go.id/index.php/SCOG/article/download/1819/1614/7252](https://journal.lemigas.esdm.go.id/index.php/SCOG/article/download/1819/1614/7252)  
24. Giving Sensors a Voice: Multimodal JEPA for Semantic Time-Series Embeddings \- arXiv, [https://arxiv.org/html/2605.31580v1](https://arxiv.org/html/2605.31580v1)  
25. Time to Embed: Unlocking Foundation Models for Time Series with Channel Descriptions \- arXiv, [https://arxiv.org/html/2505.14543v1](https://arxiv.org/html/2505.14543v1)  
26. Rolling Element Bearing Fault Detection and Diagnosis with One- Dimensional Convolutional Neural Network \- arXiv, [https://arxiv.org/pdf/2602.09699](https://arxiv.org/pdf/2602.09699)  
27. IoT-Driven Robust Bearing Fault Diagnosis for Induction Motors Under Operating-Condition Shift \- MDPI, [https://www.mdpi.com/1424-8220/26/12/3829](https://www.mdpi.com/1424-8220/26/12/3829)  
28. Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors \- PHM Papers, [https://papers.phmsociety.org/index.php/phme/article/download/1577/542](https://papers.phmsociety.org/index.php/phme/article/download/1577/542)  
29. Bearing Fault Diagnosis via Incremental Learning Based on the Repeated Replay Using Memory Indexing (R-REMIND) Method \- MDPI, [https://www.mdpi.com/2075-1702/10/5/338](https://www.mdpi.com/2075-1702/10/5/338)  
30. Computational intelligence to detect bearing faults using optimal features from motor current signals \- Taylor & Francis, [https://www.tandfonline.com/doi/full/10.1080/21642583.2024.2437157](https://www.tandfonline.com/doi/full/10.1080/21642583.2024.2437157)  
31. Bearing parameters of type 6203 | Download Scientific Diagram \- ResearchGate, [https://www.researchgate.net/figure/Bearing-parameters-of-type-6203\_tbl1\_390377609](https://www.researchgate.net/figure/Bearing-parameters-of-type-6203_tbl1_390377609)  
32. Research on Semisupervised Bearing Fault Diagnosis Combining Swin Transformer With Adaptive Pseudo-Labeling \- ASME Digital Collection, [https://asmedigitalcollection.asme.org/risk/article/12/4/041105/1228220/Research-on-Semisupervised-Bearing-Fault-Diagnosis](https://asmedigitalcollection.asme.org/risk/article/12/4/041105/1228220/Research-on-Semisupervised-Bearing-Fault-Diagnosis)  
33. Physics-Informed Multimodal Bearing Fault Classification under Variable Operating Conditions using Transfer Learning \- arXiv, [https://arxiv.org/html/2508.07536v1](https://arxiv.org/html/2508.07536v1)  
34. Multi-Scale Temporal Coordinate Attention Network with Peak-Aware Mechanism for Rolling Bearing Fault Diagnosis Under Low Signal-to-Noise Ratio Conditions \- MDPI, [https://www.mdpi.com/1424-8220/26/9/2904](https://www.mdpi.com/1424-8220/26/9/2904)  
35. A Tutorial for Feature Engineering in the Prognostics and Health Management of Gears and Bearings \- MDPI, [https://www.mdpi.com/2076-3417/10/16/5639](https://www.mdpi.com/2076-3417/10/16/5639)  
36. Belahcen, Anouar Multi-Rate Vibration Signal Analysis for Bearing Fault Detection \- acris, [https://acris.aalto.fi/ws/portalfiles/portal/133530919/machines-12-00017-v2.pdf](https://acris.aalto.fi/ws/portalfiles/portal/133530919/machines-12-00017-v2.pdf)  
37. Similarity-aware VAE with wavelet-convolutional 1D-CNN for rolling bearing fault diagnosis, [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0338388](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0338388)  
38. Fault Diagnosis from Raw Sensor Data Using Deep Neural Networks Considering Temporal Coherence \- PMC, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5375835/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5375835/)  
39. Towards a more realistic evaluation of machine learning models for bearing fault diagnosis, [https://arxiv.org/html/2509.22267v3](https://arxiv.org/html/2509.22267v3)  
40. Deep transfer learning strategy for efficient domain generalisation in machine fault diagnosis, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10125977/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10125977/)  
41. Rolling Bearing Fault Diagnosis Considering Long-Term Dependence and Time-Frequency Feature Fusion, [https://media.sciltp.com/articles/2507001019/2507001019.pdf](https://media.sciltp.com/articles/2507001019/2507001019.pdf)  
42. A Smart System for an Assessment of the Remaining Useful Life of Ball Bearings by Applying Chaos-Based Health Indicators and a Self-Selective Regression Model \- PMC, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9920053/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9920053/)  
43. kokikwbt/predictive-maintenance: Datasets for Predictive Maintenance \- GitHub, [https://github.com/kokikwbt/predictive-maintenance](https://github.com/kokikwbt/predictive-maintenance)  
44. Three Ways to Estimate Remaining Useful Life for Predictive Maintenance \- MathWorks, [https://www.mathworks.com/company/technical-articles/three-ways-to-estimate-remaining-useful-life-for-predictive-maintenance.html](https://www.mathworks.com/company/technical-articles/three-ways-to-estimate-remaining-useful-life-for-predictive-maintenance.html)  
45. Bearing Prognostics Using the Pronostia Data: A Comparative Study \- ResearchGate, [https://www.researchgate.net/publication/389935546\_Bearing\_Prognostics\_Using\_the\_Pronostia\_Data\_A\_Comparative\_Study](https://www.researchgate.net/publication/389935546_Bearing_Prognostics_Using_the_Pronostia_Data_A_Comparative_Study)  
46. A Robust Health Prognostics Technique for Failure Diagnosis and the Remaining Useful Lifetime Predictions of Bearings in Electric Motors, [https://digibuo.uniovi.es/dspace/bitstream/handle/10651/68547/applsci-13-02220.pdf?sequence=1\&isAllowed=y](https://digibuo.uniovi.es/dspace/bitstream/handle/10651/68547/applsci-13-02220.pdf?sequence=1&isAllowed=y)  
47. Bearings Fault Detection Using Hidden Markov Models and Principal Component Analysis Enhanced Features \- PHM Papers, [https://papers.phmsociety.org/index.php/phme/article/download/2947/1761](https://papers.phmsociety.org/index.php/phme/article/download/2947/1761)  
48. Prognosis of remaining bearing life with vibration signals using a sequential Monte Carlo framework | The Journal of the Acoustical Society of America | AIP Publishing, [https://pubs.aip.org/asa/jasa/article/146/4/EL358/995163/Prognosis-of-remaining-bearing-life-with-vibration](https://pubs.aip.org/asa/jasa/article/146/4/EL358/995163/Prognosis-of-remaining-bearing-life-with-vibration)  
49. Condition monitoring of a complex hydraulic system using multivariate statistics | Request PDF \- ResearchGate, [https://www.researchgate.net/publication/283428892\_Condition\_monitoring\_of\_a\_complex\_hydraulic\_system\_using\_multivariate\_statistics](https://www.researchgate.net/publication/283428892_Condition_monitoring_of_a_complex_hydraulic_system_using_multivariate_statistics)  
50. Process Independent Condition Monitoring of Hydraulic Systems via Data-Centric Transfer Learning \- RWTH Publications, [https://publications.rwth-aachen.de/record/1028325/files/1028325.pdf](https://publications.rwth-aachen.de/record/1028325/files/1028325.pdf)  
51. Comprehensive Analysis for Sensor-based Hydraulic System Condition Monitoring \- The Science and Information (SAI) Organization, [https://thesai.org/Downloads/Volume12No6/Paper\_15-Comprehensive\_Analysis\_for\_Sensor\_based\_Hydraulic\_System.pdf](https://thesai.org/Downloads/Volume12No6/Paper_15-Comprehensive_Analysis_for_Sensor_based_Hydraulic_System.pdf)  
52. Predictive model for the degradation state of a hydraulic system with dimensionality reduction \- ResearchGate, [https://www.researchgate.net/publication/340457265\_Predictive\_model\_for\_the\_degradation\_state\_of\_a\_hydraulic\_system\_with\_dimensionality\_reduction](https://www.researchgate.net/publication/340457265_Predictive_model_for_the_degradation_state_of_a_hydraulic_system_with_dimensionality_reduction)  
53. (PDF) Comprehensive Analysis for Sensor-based Hydraulic System Condition Monitoring, [https://www.researchgate.net/publication/374157684\_Comprehensive\_Analysis\_for\_Sensor-based\_Hydraulic\_System\_Condition\_Monitoring](https://www.researchgate.net/publication/374157684_Comprehensive_Analysis_for_Sensor-based_Hydraulic_System_Condition_Monitoring)  
54. (PDF) A Multirate Sensor Information Fusion Strategy for Multitask Fault Diagnosis Based on Convolutional Neural Network \- ResearchGate, [https://www.researchgate.net/publication/352059489\_A\_Multirate\_Sensor\_Information\_Fusion\_Strategy\_for\_Multitask\_Fault\_Diagnosis\_Based\_on\_Convolutional\_Neural\_Network](https://www.researchgate.net/publication/352059489_A_Multirate_Sensor_Information_Fusion_Strategy_for_Multitask_Fault_Diagnosis_Based_on_Convolutional_Neural_Network)  
55. Implementation of a Variational Autoencoder for Dimension Reduction of a Hydraulic System \- Faras Brumand-Poor\*; Faried Makansi \- River Publishers, [https://www.riverpublishers.com/downloadchapter.php?file=RP\_9788770047975C37.pdf](https://www.riverpublishers.com/downloadchapter.php?file=RP_9788770047975C37.pdf)  
56. A Semi-Supervised Learning Approach for Fault Detection and Diagnosis in Complex Mechanical Systems | Request PDF \- ResearchGate, [https://www.researchgate.net/publication/374268590\_A\_Semi-Supervised\_Learning\_Approach\_for\_Fault\_Detection\_and\_Diagnosis\_in\_Complex\_Mechanical\_Systems](https://www.researchgate.net/publication/374268590_A_Semi-Supervised_Learning_Approach_for_Fault_Detection_and_Diagnosis_in_Complex_Mechanical_Systems)  
57. AI4I 2020 Predictive Maintenance Dataset \- UCI Machine Learning Repository, [https://archive-beta.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset](https://archive-beta.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset)  
58. AI4I 2020 Predictive Maintenance Dataset \- UCI Machine Learning Repository, [http://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset](http://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)  
59. Predictive maintenance dataset \- Mendeley Data, [https://data.mendeley.com/datasets/5ww3zv87y7](https://data.mendeley.com/datasets/5ww3zv87y7)  
60. Predictive Maintenance AI4I 2020 UCI \- Kaggle, [https://www.kaggle.com/datasets/abdulbasit551/predictive-maintenance-ai4i-2020-uci](https://www.kaggle.com/datasets/abdulbasit551/predictive-maintenance-ai4i-2020-uci)  
61. Datasets \- UCI Machine Learning Repository, [https://archive.ics.uci.edu/datasets?skip=0\&take=10\&sort=desc\&orderBy=NumHits\&search=\&Types=Time-Series](https://archive.ics.uci.edu/datasets?skip=0&take=10&sort=desc&orderBy=NumHits&search&Types=Time-Series)  
62. AI-Driven Predictive Maintenance with Real-Time Contextual Data Fusion for Connected Vehicles: A Multi-Dataset Evaluation \- arXiv, [https://arxiv.org/html/2603.13343v1](https://arxiv.org/html/2603.13343v1)  
63. predictive-maintenance · GitHub Topics, [https://github.com/topics/predictive-maintenance?o=asc\&s=forks](https://github.com/topics/predictive-maintenance?o=asc&s=forks)  
64. (PDF) Temporally rigorous and traceable predictive maintenance via joint labeler-model optimization \- ResearchGate, [https://www.researchgate.net/publication/405008092\_Temporally\_rigorous\_and\_traceable\_predictive\_maintenance\_via\_joint\_labeler-model\_optimization](https://www.researchgate.net/publication/405008092_Temporally_rigorous_and_traceable_predictive_maintenance_via_joint_labeler-model_optimization)  
65. IndustryAssetEQA: A Neurosymbolic Operational Intelligence System for Embodied Question Answering in Industrial Asset Maintenance \- arXiv, [https://arxiv.org/html/2604.23446v1](https://arxiv.org/html/2604.23446v1)  
66. predictive-maintenance \- Kaggle, [https://www.kaggle.com/code/brandonmcmahon/predictive-maintenance](https://www.kaggle.com/code/brandonmcmahon/predictive-maintenance)  
67. Efficient Failure Prediction: A Transfer Learning-Based Solution for Imbalanced Data Classification \- MDPI, [https://www.mdpi.com/2079-9292/14/24/4957](https://www.mdpi.com/2079-9292/14/24/4957)  
68. A Real-Time Explainable Hybrid AI Framework for Predictive Brake Fault Diagnosis with Multilingual Dashboard Alerts in Heavy Transport Vehicles \- FMDB, [https://www.fmdbpub.com/uploads/articles/178211768093174.%20FTSCS-687-2026.pdf](https://www.fmdbpub.com/uploads/articles/178211768093174.%20FTSCS-687-2026.pdf)  
69. Evaluating the Role of Data Enrichment Approaches towards Rare Event Analysis in Manufacturing \- PMC, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11315056/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11315056/)  
70. Predicting a Failure in Scania's Air Pressure System | by Rithwik Shetty \- Medium, [https://medium.com/data-science/predicting-a-failure-in-scanias-air-pressure-system-aps-c260bcc4d038](https://medium.com/data-science/predicting-a-failure-in-scanias-air-pressure-system-aps-c260bcc4d038)  
71. Neural Networks for Predictive Maintenance on Highly Imbalanced Industrial Data \- Diva-Portal.org, [https://www.diva-portal.org/smash/get/diva2:1784433/FULLTEXT01.pdf](https://www.diva-portal.org/smash/get/diva2:1784433/FULLTEXT01.pdf)  
72. (PDF) An explainable machine learning framework for railway predictive maintenance using data streams from the metro operator of Portugal \- ResearchGate, [https://www.researchgate.net/publication/394078628\_An\_explainable\_machine\_learning\_framework\_for\_railway\_predictive\_maintenance\_using\_data\_streams\_from\_the\_metro\_operator\_of\_Portugal](https://www.researchgate.net/publication/394078628_An_explainable_machine_learning_framework_for_railway_predictive_maintenance_using_data_streams_from_the_metro_operator_of_Portugal)  
73. (PDF) An Explainable Machine Learning Framework for Railway Predictive Maintenance using Data Streams from the Metro Operator of Portugal \- ResearchGate, [https://www.researchgate.net/publication/394396672\_An\_Explainable\_Machine\_Learning\_Framework\_for\_Railway\_Predictive\_Maintenance\_using\_Data\_Streams\_from\_the\_Metro\_Operator\_of\_Portugal](https://www.researchgate.net/publication/394396672_An_Explainable_Machine_Learning_Framework_for_Railway_Predictive_Maintenance_using_Data_Streams_from_the_Metro_Operator_of_Portugal)  
74. Fault Detection and Explanation through Big Data Analysis on Sensor Streams | Request PDF \- ResearchGate, [https://www.researchgate.net/publication/317575563\_Fault\_Detection\_and\_Explanation\_through\_Big\_Data\_Analysis\_on\_Sensor\_Streams](https://www.researchgate.net/publication/317575563_Fault_Detection_and_Explanation_through_Big_Data_Analysis_on_Sensor_Streams)  
75. Details of Paderborn University (PU) bearing dataset. \- ResearchGate, [https://www.researchgate.net/figure/Details-of-Paderborn-University-PU-bearing-dataset\_tbl1\_346842544](https://www.researchgate.net/figure/Details-of-Paderborn-University-PU-bearing-dataset_tbl1_346842544)  
76. Linear Methods for Predictive Maintenance: The Case of NASA C-MAPSS Datasets \- MDPI, [https://www.mdpi.com/2076-3417/15/18/9945](https://www.mdpi.com/2076-3417/15/18/9945)  
77. Linear Methods for Predictive Maintenance: The Case of NASA C-MAPSS Datasets, [https://www.researchgate.net/publication/395435090\_Linear\_Methods\_for\_Predictive\_Maintenance\_The\_Case\_of\_NASA\_C-MAPSS\_Datasets](https://www.researchgate.net/publication/395435090_Linear_Methods_for_Predictive_Maintenance_The_Case_of_NASA_C-MAPSS_Datasets)  
78. A Convolutional Autoencoder for Time-Series Data Anomaly Detection \- Edge Impulse, [https://www.edgeimpulse.com/blog/a-convolutional-autoencoder-for-time-series-data-anomaly-detection/](https://www.edgeimpulse.com/blog/a-convolutional-autoencoder-for-time-series-data-anomaly-detection/)  
79. A Deep-Learning Method for Remaining Useful Life Prediction of Power Machinery via Dual-Attention Mechanism \- MDPI, [https://www.mdpi.com/1424-8220/25/2/497](https://www.mdpi.com/1424-8220/25/2/497)  
80. Deep Learning Strategies for Predictive Maintenance | by Alberto Moccardi | Medium, [https://medium.com/@albertomoccardi/deep-learning-strategies-for-predictive-maintenance-9f1f40d8958a](https://medium.com/@albertomoccardi/deep-learning-strategies-for-predictive-maintenance-9f1f40d8958a)  
81. Remaining useful life prediction methods of equipment components based on deep learning for sustainable manufacturing: a literature review | AI EDAM | Cambridge Core, [https://www.cambridge.org/core/journals/ai-edam/article/remaining-useful-life-prediction-methods-of-equipment-components-based-on-deep-learning-for-sustainable-manufacturing-a-literature-review/C3FFF4402D1EF1EC9BDD1F5C84198BC1](https://www.cambridge.org/core/journals/ai-edam/article/remaining-useful-life-prediction-methods-of-equipment-components-based-on-deep-learning-for-sustainable-manufacturing-a-literature-review/C3FFF4402D1EF1EC9BDD1F5C84198BC1)  
82. GIVING SENSORS A VOICE: MULTIMODAL JEPA FOR SEMANTIC TIME-SERIES EMBEDDINGS \- OpenReview, [https://openreview.net/pdf/15d3025cc81e19cf40ba72973fa704f3886196e5.pdf](https://openreview.net/pdf/15d3025cc81e19cf40ba72973fa704f3886196e5.pdf)  
83. Predicting a Failure in Scania's Air Pressure System (APS) \- Towards Data Science, [https://towardsdatascience.com/predicting-a-failure-in-scanias-air-pressure-system-aps-c260bcc4d038/](https://towardsdatascience.com/predicting-a-failure-in-scanias-air-pressure-system-aps-c260bcc4d038/)  
84. predictive-maintenance · GitHub Topics, [https://github.com/topics/predictive-maintenance?l=javascript](https://github.com/topics/predictive-maintenance?l=javascript)  
85. CWRU data sampling system used by CWRU. \- ResearchGate, [https://www.researchgate.net/figure/CWRU-data-sampling-system-used-by-CWRU\_fig3\_330112231](https://www.researchgate.net/figure/CWRU-data-sampling-system-used-by-CWRU_fig3_330112231)  
86. Data-driven Models for Remaining Useful Life Estimation of Aircraft Engines and Hard Disk Drives \- Murray State's Digital Commons, [https://digitalcommons.murraystate.edu/cgi/viewcontent.cgi?article=1118\&context=honorstheses](https://digitalcommons.murraystate.edu/cgi/viewcontent.cgi?article=1118&context=honorstheses)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAaCAYAAADMp76xAAACJUlEQVR4Xu2Wz0tWQRSGT1RQGKQVRRREJkEYtGil2M5tblr6BxS1a6EitHThtmUE0cIicOOiXf+BWwMRBAWlVQhBgkY/zuOc0evb3NvXt/y4D7zgfc/cOzNnzpxPs5aW3uK064rrqqiOE64Lagrn1KjQr4Y1j/8LBrPAbddv1614RqPh/bS0UDjpuux6F7GnMZaNZ1jUYsTvWRqfYexIxNbjuSlBtexa+ogyb8l/IP7H8K+Ln5l2LasZjFl6d04DnUL2+MCWBpznlmJvxf8Vfh1kb1JN55Sl7H9xDUqsYwYsTU7WlHy0j8XH+yFelU3XHTWdS67PluY6I7GOIRNkbFz885YWtiI+ZYDPZkqQxZdqBmycd3Wu/+K1pXJgIblrcJnIEjXcdzT0ACZjUsqlBFkslQPkE+u6HLjF31yzFY9OQGfYr3gZjpHj3HBdOx46hCxSZiUoo6ba/ydkonREb8K/KD6nwGmQKY5e4QIvqFmBb1J+JZir2gKL0Hr4iF6AfHR6cXLX0A1maFmUUglKhXc/acDS5t+rWeK7lY+ItoSvTZ3+in9f/Az3AZVo6r/0+bqNHtLUf/FR/tnkJIZdNy2Nnwi/Cll/pGbQ1H9z16m7qAcLvet6YmngquuGHf9N34tYvjxLrrPxN4tacw3FM8y4vlaeM8xFbT5z7Viai80y30PXB0sXnCSQjK5hotuuV64XEgNaHRMSn7Kj/zVaWlpaWnqMP0HbcXmA/nPJAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAaCAYAAADi4p8jAAACcElEQVR4Xu2WTaiOURDHRyhE+ShS6payUMpCKYXuQlmIIkXdhZUiViyU7GRjKSspWUjJ1s7ilrJAodhRdyOrmygL3+bXnOk975jnPs+9WSjnV//e3pkz53POnEek0Wj8SyxVbVBtTLSuapdBbBazqG70l8j6XRH+p6wUm9g71S/VpvLf9bzYD3hAYbFqveqymP+EjMfdLfbzYhNDTPKs6rvqgWqiar9TdanE3JY/oc0uMf+L8r/vAMb4JBYc2ab6oHopeYd3ZLQxNSzouuqzakdlnxJrz2/GF9XFaCzsFou9Eh1DIPB1NCprVE9Vs6qtwQcsINsYYCL4LlQ2+sJGvxnTqoPRqCxR3Ve9V20Ovl4YjEHpILJHbFfvid25CHE/o1FZJpaG+I9X9ixTSFXnmYyfuEOdeCvWJ33PC9Llm1gK1JB2TOZxsDu+MQwa8bvJ3arBRjY4TDzLnMg5sdh90TGEa2L37KTqaNG06qtYcaGgZGwXGzTemeXF/jHYfUNeyai4PBFLuz7ILmLnnZ7cK3aUHYpsEev0RnQUboqlL4WGNuiqdO+yF5jTMlogd3hI0SDDYmoPwgeN6engyzr24vNItSr4Mqiqt8TuICfvzKgOVf+7YA7ZXe/FqxpVKqNrgb4xR6KjAwoHp8V1qNkrefGq4Z4y1sPoGALpmS0AWHTXAjkN7NnTkdH3/s3Fgt8/0obAmWCH/WKFhxOOjzjM9f5lUClp3/X+dbGg94+F8YVySmxQPscoKFx6fnm3/OTWlhhnQuwhdv+k2CdbF/gmVT/E2h8W66OrMjv+KXhGbKPfqI6pVteNGo1Go9Fo/Gf8BmUUkDYfMWnnAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADoAAAAaCAYAAADmF08eAAACvUlEQVR4Xu2WT6jNURDHRyiKhCJFF7EgZUFW6C1IEkUWysJKKCsWIjvZWOllISlJUrK1s3hRFigp9i+RlURZ+G8+zRl33jjn3fuKxavfp769+2bmnN/5M2fOEeno6JhOzFYtVS2raHGIq0HbWpsZMegfsVJ1WvVa9UC1YoJ3COaJDfCt6pdqefnf9bzY93iDwkzVEtUFMf8RmdjuTrEzOCaOWISTqu+q+6peiN+iOl/a3JQ+d1U/xcYVuSUWuyDZB/JJrGFmg+qD6oXUd/i29BcowsSuqD6rNgX7YbF4/tb4ojpXfq8Riz3bd/+BsTxRXZYpZhAdvspGZaHqqeq9al3yAROpLRBcFPOdCTb6wka/NcZUe8XSc9CObRWLOZYdLfgoDe5lh7JNbJVJIc5khnakVmaOWHriPxTstczphd/PVJvFYn4Eew0yhbjr2dGCNPomtkIR0pGOHie74wvEhDJ+djl7EWxkh0MhjJnkKU9cK70ddp64eKYnZVTsHB5VHSwaU30VK0IUnhobxT7kZ8qZW+wfk90X5qX0ixDn7F2IWaV6U2yrg70G36U/xj8Qzh0rfCo7lLViHV3LjgIpQ1pTkIhBl1Q7YlDAC9EJ6U+UM85ZdjjPwwyeRWAxWrXjL/zjOW0dfCjjReqRan7y1SAlb4idUTLBGVftC/+ThnwvFrAaPm6usVnJV8WrYCu4NVH/0IHsaEDhYPfyTm2XiUXOd3TQ+aQAEjf01cLW1yYCTL41UXYH+1BpI4PvT2e3WFw+9xEeKsQ8zI4WrAYNxpMddokVKHY8PwZgsvuzBpWV+Nb9GTkudrXszA7lqkxhgZkgLx46pBHPPAoPxYG/3Hu+k4tKG6cn/bKORsRWuAW+EbGBE79frI9WJQdSmVjetZH1xT70dTJd4NFBFaeas7i1J2hHR0dHR0fHf+A3vK2pcd/xDswAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,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>

[image5]: <data:image/png;base64,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>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAdCAYAAABmH3YuAAAAj0lEQVR4XmNgGMqAEYjDgJgVXQIEooF4DxBzo0uAdM0H4lZ0CRAQBOLTQOyHLAgy6j8WzAOS5ABiSQaIkSBBEBuE4YAXiA9DJTEAyB6QBMhODAByIUhyDroECID8BpIEOQ4DgCSuArEIA8S/zeiSa4CYBYjNgPgiuuRSBogukKOCkSWZgVgZiI2QBUcBEAAAahobe+sHnnAAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,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>

[image8]: <data:image/png;base64,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>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMEAAAAaCAYAAAAOuzhwAAAHDklEQVR4Xu2ae6hmUxTAl1De7zxCMx4ZGlIYjJA/PAZ5pEShKI/xKM9JCc384Q8pyR+KZBpC5FlEkm7jD0KKiDRyyaM8EhkZ8li/WXvN2XfZ+3zfZe73nds9v1rde9Y+5/v2Xnvttdde5xOZW5yislzlA5WfVZ5WuVllgcrmzW09PbOXQ1SuFHPuz1T+zmRS5UGVW1QWqmyrco7KTSovq/ya3fub2PMrVW5X2UV6ejrKfJVzVV5T+VwaJ/5JzInvElsUW6T7BzFf5WKVx1Tel6mL6BuxRXFJurenZ6TsqPKMNA75rsr9KktVTsrum2n2FlsofC8L5Vux/nyv8lZzW0/PxmcrlctUTlA5cGrTWNlamt1j2ZSWnhoHqbwdlT09c4lHZRYtgk1U9lA5UuWo0DZuqOTsJta/XGpw/05RmcFYd47KjB2iQtkmKjpCtAnSdibCLoy/BnZpex5K7dgnt5HP2cdiZytvO1blcLFd9WRp+sL3npr+B57fPrsG7j1b7PkZYVOVM1T+EOtol8CATO5XYjn5fukaOSbp/pTGoEwSbZOp7eh07e2MdVeVT1V+V5knUx2fexeLPcs9gxbdOKFflILp68PpGlmi8orKWpXTN9xtbYeJ3U9xges8IGCX08TsQqWsFhCwzw9i9iFo1hYf95ASAYuCyturYmnlDSpXqGypsr9Y9Y7PBXYPqnvOLyrnpf8PFltEMwIrEePsGRs6gpc1I3eK6Y8LeiYAfS3yTajcEZUJAgHP1tq7BM5DXy8IeqIp+q+DHqdET6pSAicfNG7sQ8Bsuw+748wUOuAIlb1UvkzXB4g5MwUI+vq4NNGf9zo+Hhbmm6mNxfJk0s8IrNCSk3UBDErf3IA5RBTaVgU9OiJhDZxj36hUNlN5SqxMWmrvEjjshMrrYu9CItggzum9YnappRUEw0Hjxj58btt9i+TfC5P3NqU5PFNlXXb9oTTva9gtyFJGApP+V1R2BKIJRn8xNkgzIRjLwSHQtR3K+KzSFo7xmYRae5fACZk3HDviNsvnFLuwYPIIHSG6Dxo39uGz2+5jPg5N/5+V/mLT0hzyJp/0DOgXu9R2YmcYdrq4YGu7+7R5Q2z1rRH7aUHNyboAEYXJPDHo2SLpN/3PqaUIDs4TP8th8ni21t4lcFj66nm3M0/lE5WXZOoBcxi7uDO2ERdXCVKmJ8SClPOjWOk7Qn/fE0uJGBMHajITBx9lp39H5cZM/7/gsLRaZXexg4lHU7ar6eIHTT+UDSNtEaQEhyS2UXJKrzxcJTZhnAniQYnISF7pkSiCg9e28mG2+q7gKaynQtiWgyNv0RFslYNduL/NLoMWAekin8EONAj8Av9wiPK1KM59fhCPVSqf89qz04a3nQwi/0AG706WwyJBxgkThkPnCxSDURGiihEhKnIoXillo2Hcth2PAx/2GZbrxQ5544B+5n0loK1Q+U7sIBoZVCzALm2HXZhNRYMqDICJzmFApRyYCDzutICtmz7HfuDk6GPNf9CWz0KntFeDZ2tbPd9FdMvh/vw8EsHh6MsDQ8rd9thQ8N04dg6HZfTPBj3ERRPBLtHOEcZaSk0B+/yXbGKksF1ihLw85oer0qC6AIc4+hwXqKctMR/2FKH2a1Bf8CV4hmfzfNQhDSBn7QrMG32Nh2I/EHN4zSFlLOlzSoEwx4sG2CemoIB9rsmuCQC84KrtPGPBFwFlRQfnZ2VjVDrreeTxUs8dc4is1HG/mIZctP7J4Vgr5ehF1ELPmHImkp6IWIJJZLco0bbV8x4iz5dJyaha5DnvKKGsSF/jWJgz9BNB7ztE1DvU6+NnRbBP2/sB7MOZxPFUtm1hjRw3RF53pePuZGzbRDy2PF6aTSb9uGBR0jfOKxH0ubOzYyxUOT/p40s/FvdysUpECcbN7sKuGA/FLHQ+M0+xbhUr41EBGfVCYMwTUn4/QLmRvq5K14ul2RXRT6b/c5arfBSVBWpFA+zD7lBLQTsHg1iZ/l8kzSLAsK+IOcNtYs7EW9pxgPMTUZaK9Y2SGc6bR/d1qc3r3c+LHQxxdnJrzjNcO/w8mzJchO8il71arITHdxFl+T6CBU7OIZyFuE96BohyOBh2GhVeiVsmNv7VYr/BySOtBwEclrHdk/4CdqEttwtjwi614ADYh9+VYR+exz44PvbB8bEP+tw+wIs3AkUnwWh5+QrHYaB57saBK8/vugZ9pSrDxLJoI2zv7Gj3qVwY2jYGnh/nNusKpGkrVK6VshNiF+yGXWJhYWOAbz0ktivFn23MKvzHT5fHhp71LBEr0S4QSyd6GijPXip2VuxSMWHarFF5Lip7NkBKQIp0nXRzN+gCnDN4yTVr4e1drcLSY/T2qYPzU7711LunZ87BwfkFlUdiQ0/PXILiwajLxz09PW38Azr2hvwetrluAAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEEAAAAaCAYAAADovjFxAAADQklEQVR4Xu2XS8hNURTHl1DkHSVRPo+UiORVYsZAMiChmBkQZUAhZSQDEwMZyXOkJI+BiQxulJSJAZFSnyJFUoo88li/u/dq77PuPuf7iEzOr/517lr7vfdae1+RlpaWlnqGqyarpjgNhtGSyk9UDam6exgvzX0MVS1XdVRnVWOj/YhqVPz+J9hEXql+qmbF32hFtH2X3gnym4kfVf2QUG5tpUSVjuqz6r5qrvQuwjrVHdW0zHZGQrs7MxtQd1P0PYy/TfNVl6JvtVUYLJ8kVPQcl2Bf5R0RBnpNyoM1WLALqjeqw1VXlz7VcwmLmjND9VK11NlhpYQ+j3mHhP5OqRZ7RxNUokE69OyT4LvoHZHHqkVSX2akhJ1hsJSZWnV36Vd99MYIbfpTOEx1RfVaNdP5jPWqMd7YxAQJA7zpHRI6a9rlB5Lqd6quLttVJyS0TZkRVXcXFoCQ8lC2NCZyGCfntlRzxXXV7Pi9Ucp91bJNwiB8DI2TMPBHzm6w0tQFyrEzOezYlvhN+6WJAieQ+ohF7ZPe3c+x05nnIMZ6XprrNUJcMxCSkt0Wu1UvJOSEusxMgkPwTUJeyVkjoT1g0P3JVeGApEUw0Ra3RQk7neSi0xJChnxjG/LbLFR9kGrConNuhK+ZrcTJ7JsFY2DGIdX++E3c4juY3EWWSErEiI0hOXpY8Lwvdp96hOUfwerRoA8FjhZ2n7ENQuFu9pvvfGBXJRxRoO1SuNXBieS6pL3StYvdh1a+iYy5dAvVQgzSqE8iduTsuHtIPNQ17E7n3bFLUmwSSiQwEpzvg0W552wG1xs7zlWYw+1CP6WEadDmVm9sgsyc76BB9sXuHzUGocDEDY465XlP3MrshAIJsxQK2DreGGHyXL+TCnb6Kb0PDOqR1wZF0/sAu+0ssOvzkrvnEcPg2LnLkpIhDPQ+8McalqneSwonw67MuvfBAtUTCa/GAWHyFOTYMsCnqumSJgxfos+SzQ0JDx8GuDf6zkmqQ9i8k7RQ/E+gDztRmyX0kWPX5gZJNwFje6t6ZoWijTjfI2FxGC/vD3sq84TeIWnjuJr/CnQ8R8IVxB+YfwEJEDi6TIq+mExLS0tLS0tLy3/jF1CIvsA8/TzVAAAAAElFTkSuQmCC>