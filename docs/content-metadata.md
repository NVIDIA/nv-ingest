**Definitions**:  
Source: The knowledge base file from which content and metadata is extracted  
Content: Data extracted from a source; generally Text or Image  
Metadata: Descriptive data which can be associated with Sources, Content(Image or Text); metadata can be extracted from Source/Content, or generated using models, heuristics, etc

|  | Field | Description | Method |
| ----- | :---- | :---- | :---- | 
| Content | Content | Content extracted from Source  | Extracted |
| Source Metadata | Source Name | Name of source | Extracted |
|  | Source ID | ID of source  | Extracted |
|  | Source location | URL, URI, pointer to storage location | N/A  |
|  | Source Type | PDF, HTML, Docx, TXT, PPTx | Extracted |
|  | Collection ID | Collection in which the source is contained | N/A |
|  | Date Created | Date source was created | Extracted |   |
|  | Last Modified | Date source was last modified | Extracted |   |
|  | Summary | Summarization of Source Doc | Generated | Pending Research |
|  | Partition ID | Offset of this data fragment within a larger set of fragments | Generated |
|  | Access Level | Dictates RBAC | N/A |
| Content Metadata (applicable to all content types) | Type | Text, Image, Structured, Table, Chart | Generated |
|  | Description | Text Description of the content object (Image/Table) | Generated |
|  | Page \# | Page \# where content is contained in source | Extracted |
|  | Hierarchy | Location/order of content within the source document  | Extracted |
|  | Subtype | For structured data subtypes \- table, chart, etc.. |  |  |
| Text Metadata | Text Type | Header, body, etc | Extracted |
|  | Summary | Abbreviated Summary of content | Generated | Pending Research |
|  | Keywords | Keywords, Named Entities, or other phrases  | Extracted | N |
|  | Language |  | Generated | N |
| Image Metadata | Image Type | Structured, Natural,Hybrid,  etc | Generated (Classifier) | Y(needs to be developed) |
|  | Structured Image Type | Bar Chart, Pie Chart, etc | Generated (Classifier) | Y(needs to be developed) |
|  | Caption | Any caption or subheader associated with Image | Extracted |
|  | Text | Extracted text from a structured chart | Extracted | Pending Research |
|  | Image location | Location (x,y) of chart within an image | Extracted |  |
|  | Image location max dimensions | Max dimensions (x\_max,y\_max) of location (x,y) | Extracted |  |
|  | uploaded\_image\_uri | Mirrors source\_metadata.source\_location |  |  |
| Table Metadata (tables within documents) | Table format | Structured (dataframe / lists of rows and columns), or serialized as markdown, html, latex, simple (cells separated just as spaces) | Extracted |
|  | Table content | Extracted text content, formatted according to table\_metadata.table\_format. Important: Tables should not be chunked | Extracted |  |
|  | Table location | Bounding box of the table | Extracted |  |
|  | Table location max dimensions | Max dimensions (x\_max,y\_max) of bounding box of the table  | Extracted |  |
|  | Caption | Detected captions for the table/chart | Extracted |  |
|  | Title | TODO | Extracted |  |
|  | Subtitle | TODO | Extracted |  |
|  | Axis | TODO | Extracted |  |
|  | uploaded\_image\_uri | Mirrors source\_metadata.source\_location | Generated |  |

## Example text extracts for multimodal_test.pdf:
1. [text](example_processed_docs/text/multimodal_test.pdf.metadata.json)
2. [images](example_processed_docs/image/multimodal_test.pdf.metadata.json)
3. [charts and tables](example_processed_docs/structured/multimodal_test.pdf.metadata.json)
