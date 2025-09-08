# RxRx1-WILDS

RxRx1-WILDS is a modified version of RxRx1, the first dataset released by
[Recursion][recursion] in the [RxRx.ai][rxrx] series, and was the topic of the
[NeurIPS 2019 CellSignal competition][kaggle] hosted on Kaggle. This modified
dataset contains 125,510 3-channel fluorescent cellular microscopy images, taken
from four cell types perturbed by 1,138 siRNAs. The goal of the competition was
to train models that identify which siRNA was used in test images taken from
experimental batches not seen in the training set.  For more information about
RxRx1 please visit [RxRx.ai][rxrx1].

RxRx1 is part of a larger set of Recursion datasets that can be found at [RxRx.ai][rxrx] and on
[GitHub][github]. For questions about this dataset and others please email
[info@rxrx.ai](mailto:info@rxrx.ai).

## Metadata
The metadata can be found in `metadata.csv`. The schema of the metadata is as follows:

| Attribute      | Description                                                                     |
|----------------|---------------------------------------------------------------------------------|
| site_id        | Unique identifier of a given site                                               |
| well_id        | Unique identifier of a given well                                               |
| cell_type      | Cell type tested                                                                |
| dataset        | The split that this site belongs to; `train`, `val` or `test`                   |
| experiment     | The experiment name, same as explained above                                    |
| plate          | Plate number within the experiment                                              |
| well           | Location on the plate                                                           |
| site           | Indication of the location in the well where image was taken (1 or 2)           |
| well_type      | Indicates if the well is a treatment, `negative_control`, or `positive_control` |
| sirna          | The siRNA (ThermoFisher ID) that was introduced into the well                   |
| sirna_id       | The siRNAs mapped to integers for ease of classification tasks                  |

We note that for `RxRx1-WILDS`, the original `test` dataset is split into `val`
and `test` datasets.  We followed the split used in the [Kaggle
competition][kaggle]: `val` corresponds to the public test set, and `test` to
the private test set.

## Images
The images are located in `images/*` .  These images are modified from the
original RxRx1 images in the following ways:
- only the center 256x256 crops of each image were used, and
- only the first three channels of each image were used (and combined into a single RGB image).
Thus RxRx1-WILDS consists of 256x256x3 8-bit `png` files.  The image paths, such as
`HUVEC-01/Plate1/M23_s2.png`, can be read as follows:

- Experiment Name: Cell type and experiment number (HUVEC experiment 1)
- Plate Number: 1
- Well location on plate: column M, row 23
- Site: 2

## Changelog
- March 2021: original release of RxRx1-WILDS

## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>

This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.


[github]: https://github.com/recursionpharma/rxrx-datasets
[recursion]: http://recursion.com
[rxrx]: http://rxrx.ai
[rxrx1]: https://rxrx.ai/rxrx1
[kaggle]: https://www.kaggle.com/c/recursion-cellular-image-classification
