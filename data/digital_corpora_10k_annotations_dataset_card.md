## Dataset Description:
The Digital Corpora Annotations (DC10k) dataset is a collection of question/answer pairs for evaluation of retrieval systems. The question/answer pairs also have document IDs and corresponding page numbers where the answers can be found.

The PDFs are sourced from [Digital Corpora](https://digitalcorpora.org/corpora/file-corpora/cc-main-2021-31-pdf-untruncated/) and are not provided as part of this dataset.

This dataset is ready for commercial/non-commercial use.

## Dataset Owner(s):
NVIDIA Corporation

## Dataset Creation Date:
Created: August 29, 2025

## License/Terms of Use: 
[Apache 2.0](https://github.com/NVIDIA/nv-ingest/blob/main/LICENSE)

## Intended Usage:
Evaluation of PDF retrieval systems.

## Dataset Characterization
** Data Collection Method: Human <br>
** Labeling Method: Human <br>

## Dataset Format
The dataset is formatted as a CSV file with the format noted in the following section.

## Dataset Quantification
**Row count**: 1347
**Dataset size**: <1MB

| Header   | Description                                                 |
| ------ | --------------------------------------------------------- |
| modality | The modality of the query (e.g., text, chart, table, infographic). |
| query    | The question asked.                                         |
| answer   | The answer to the query.                                    |
| pdf      | The document ID of the PDF where the answer can be found.   |
| page     | The page number within the PDF where the answer is located. |

## Reference(s):
https://github.com/NVIDIA/nv-ingest

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.   

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
