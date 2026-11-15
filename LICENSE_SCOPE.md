# License scope

Copyright 2026 Hongtao Xu, Peng Xiong, and Haoran Zheng.

## Our original material

Our original source code, configurations, and documentation in this repository are licensed under [Apache-2.0](LICENSE), to the extent we are authorized to license them. Third-party material is excluded from this grant. Apache-2.0 does not impose a research-only or noncommercial restriction.

## Third-party material

Third-party code retains its original copyright notices, licenses, and applicable NOTICE files. Dependencies have their own licenses. The root LICENSE does not replace these terms. The [NovoBench MIT notice](licenses/NovoBench-MIT.txt) is preserved from [the publisher's repository](https://github.com/jingbo02/NovoBench/blob/main/LICENSE); this notice does not assert that every evaluation implementation originates from NovoBench.

Third-party spectra, peptide annotations, and converted copies are not relicensed under Apache-2.0. Their source terms are:

| Source | Published terms | Source and attribution |
| --- | --- | --- |
| Nine-species V1 | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Tran et al., DeepNovo; [MassIVE MSV000081382](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000081382) |
| Nine-species V2, including retained subsets | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Casanovo nine-species benchmark; [MassIVE MSV000090982](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000090982) |
| NovoBench HC-PT, Nine-species, Seven-species release | MIT, as declared by the publisher | Jingbo Zhou et al. (2024); [dataset card](https://huggingface.co/datasets/jingbo02/NovoBench); [preserved MIT text](licenses/NovoBench-MIT.txt) |
| MassIVE-KB v1 30 million PSMs training release | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Wout Bittremieux (2025), v1; [DOI: 10.5281/zenodo.14973856](https://doi.org/10.5281/zenodo.14973856) |

RocNovo converts peptide-modification notation and file formats; the V2 retained evaluation subset also applies the documented exclusions. No new raw spectra were collected. The MassIVE-KB training corpus is not bundled here. The source repository and model repository do not bundle the evaluation HDF5 collection; it is distributed separately in the dataset repository.

CC0 permits reuse, modification, and redistribution, including commercial reuse; citing its source is a scholarly request, not an additional CC0 condition. MIT redistribution requires preservation of the copyright and permission notice. Redistribution of CC BY 4.0 data or adaptations requires appropriate attribution, a license link, indication of changes, and no additional restrictions on licensed uses. The specific Zenodo release is CC BY 4.0; do not infer its terms from the CC0 statement for MassIVE-KB reference libraries.

NovoBench's MIT designation is the direct publisher's statement. Its card traces constituent datasets to earlier studies; this release does not claim to have independently established every upstream right or to override upstream notices. Preserve applicable third-party notices when redistributing data or code. Reference labels copied into prediction CSV files remain subject to their source terms.

## Citation and warranties

Please cite RocNovo and the original datasets when using these resources in research. This request adds no restriction to Apache-2.0, MIT, CC0, or CC BY permissions. The applicable licenses contain their own warranty disclaimers. No endorsement by the original data authors is implied.
