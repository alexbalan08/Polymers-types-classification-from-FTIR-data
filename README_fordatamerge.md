This repository contains the aligned and merged FTIR datasets
Two independent FTIR datasets were processed:FTIR_classification(main dataset) and Yan2023Data(external dataset)
Both datasets originally had different wavenumber ranges, resolutions, and metadata formats.
To make the datasets compatible: convertion to numeric format, common overlap region was detected, both data sets interpolated, metadata columns hormonized (For example: Substance,Plastic/other, etc.,)
The two datasets were merged into unified FTIR dataset
To increase the size and variability of the FTIR dataset and improve the robustness classification models, several augmentation methods were applied to the merged FTIR spectra
