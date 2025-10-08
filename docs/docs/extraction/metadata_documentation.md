# Metadata Schema Documentation

This document provides a detailed explanation of the `MetadataSchema` and its constituent sub-schemas used within the NVIDIA Ingest Framework. This schema defines the structure for metadata associated with ingested content.

## Main Schema: `MetadataSchema`

The `MetadataSchema` is the primary container for all metadata. It includes the core content, its URL, embedding, and various specialized metadata blocks.

| Field                 | Type                                  | Default Value/Behavior | Description                                                                                                |
|-----------------------|---------------------------------------|------------------------|------------------------------------------------------------------------------------------------------------|
| `content`             | `str`                                 | `""`                   | The actual textual content extracted from the source.                                                      |
| `content_url`         | `str`                                 | `""`                   | URL pointing to the location of the content, if applicable.                                                |
| `embedding`           | `Optional[List[float]]`               | `None`                 | Optional numerical vector representation (embedding) of the content.                                       |
| `source_metadata`     | `Optional[SourceMetadataSchema]`      | `None`                 | Metadata about the original source of the content. See [SourceMetadataSchema](#sourcemetadataschema).       |
| `content_metadata`    | `Optional[ContentMetadataSchema]`     | `None`                 | General metadata about the extracted content itself. See [ContentMetadataSchema](#contentmetadataschema).    |
| `audio_metadata`      | `Optional[AudioMetadataSchema]`       | `None`                 | Specific metadata for audio content. Automatically set to `None` if `content_metadata.type` is not `AUDIO`. See [AudioMetadataSchema](#audiometadataschema). |
| `text_metadata`       | `Optional[TextMetadataSchema]`        | `None`                 | Specific metadata for text content. Automatically set to `None` if `content_metadata.type` is not `TEXT`. See [TextMetadataSchema](#textmetadataschema). |
| `image_metadata`      | `Optional[ImageMetadataSchema]`       | `None`                 | Specific metadata for image content. Automatically set to `None` if `content_metadata.type` is not `IMAGE`. See [ImageMetadataSchema](#imagemetadataschema). |
| `table_metadata`      | `Optional[TableMetadataSchema]`       | `None`                 | Specific metadata for tabular content. Automatically set to `None` if `content_metadata.type` is not `STRUCTURED`. See [TableMetadataSchema](#tablemetadataschema). |
| `chart_metadata`      | `Optional[ChartMetadataSchema]`       | `None`                 | Specific metadata for chart content. See [ChartMetadataSchema](#chartmetadataschema).                      |
| `error_metadata`      | `Optional[ErrorMetadataSchema]`       | `None`                 | Metadata describing any errors encountered during processing. See [ErrorMetadataSchema](#errormetadataschema). |
| `info_message_metadata` | `Optional[InfoMessageMetadataSchema]` | `None`                 | Informational messages related to the processing. See [InfoMessageMetadataSchema](#infomessagemetadataschema). |
| `debug_metadata`      | `Optional[Dict[str, Any]]`            | `None`                 | A dictionary for storing any arbitrary debug information.                                                  |
| `raise_on_failure`    | `bool`                                | `False`                | If `True`, indicates that processing should halt on failure.                                               |

**Note:** A `model_validator` ensures that type-specific metadata fields (`audio_metadata`, `image_metadata`, `text_metadata`, `table_metadata`) are set to `None` if the `content_metadata.type` does not match the respective content type.

## Sub-Schemas

### `SourceMetadataSchema`
Describes the origin of the ingested content.

| Field             | Type                               | Default Value                     | Description                                                                                             |
|-------------------|------------------------------------|-----------------------------------|---------------------------------------------------------------------------------------------------------|
| `source_name`     | `str`                              | *Required*                        | Name of the source (e.g., filename, URL).                                                               |
| `source_id`       | `str`                              | *Required*                        | Unique identifier for the source.                                                                       |
| `source_location` | `str`                              | `""`                              | Physical or logical location of the source (e.g., path, database table).                                |
| `source_type`     | `Union[DocumentTypeEnum, str]`     | *Required*                        | Type of the source document (e.g., `pdf`, `docx`, `url`). Uses `DocumentTypeEnum`.                      |
| `collection_id`   | `str`                              | `""`                              | Identifier for any collection this source belongs to.                                                   |
| `date_created`    | `str`                              | `datetime.now().isoformat()`      | ISO 8601 timestamp of when the source was created. Validated to be in ISO 8601 format.                |
| `last_modified`   | `str`                              | `datetime.now().isoformat()`      | ISO 8601 timestamp of when the source was last modified. Validated to be in ISO 8601 format.           |
| `summary`         | `str`                              | `""`                              | A brief summary of the source content.                                                                  |
| `partition_id`    | `int`                              | `-1`                              | Identifier for a partition if the source is part of a larger, partitioned dataset.                      |
| `access_level`    | `Union[AccessLevelEnum, int]`      | `AccessLevelEnum.UNKNOWN`         | Access level associated with the source. Uses `AccessLevelEnum`.                                        |

### `ContentMetadataSchema`
General metadata about the extracted content.

| Field           | Type                                  | Default Value                     | Description                                                                                                |
|-----------------|---------------------------------------|-----------------------------------|------------------------------------------------------------------------------------------------------------|
| `type`          | `ContentTypeEnum`                     | *Required*                        | The type of the extracted content (e.g., `TEXT`, `IMAGE`, `AUDIO`). Uses `ContentTypeEnum`.                |
| `description`   | `str`                                 | `""`                              | A description of the extracted content.                                                                    |
| `page_number`   | `int`                                 | `-1`                              | Page number from which the content was extracted, if applicable (e.g., for PDFs).                        |
| `hierarchy`     | `ContentHierarchySchema`              | `ContentHierarchySchema()`        | Hierarchical information about the content's location within the source. See [ContentHierarchySchema](#contenthierarchyschema). |
| `subtype`       | `Union[ContentTypeEnum, str]`         | `""`                              | A more specific subtype for the content (e.g., if `type` is `IMAGE`, `subtype` could be `diagram`).      |
| `start_time`    | `int`                                 | `-1`                              | Start time in milliseconds for time-based media (e.g., audio, video).                                    |
| `end_time`      | `int`                                 | `-1`                              | End time in milliseconds for time-based media.                                                           |

### `ContentHierarchySchema`
Describes the structural location of content within a document.

| Field            | Type                  | Default Value              | Description                                                                                             |
|------------------|-----------------------|----------------------------|---------------------------------------------------------------------------------------------------------|
| `page_count`     | `int`                 | `-1`                       | Total number of pages in the document, if applicable.                                                   |
| `page`           | `int`                 | `-1`                       | The specific page number where the content resides.                                                       |
| `block`          | `int`                 | `-1`                       | Identifier for a block of content (e.g., paragraph, section).                                         |
| `line`           | `int`                 | `-1`                       | Line number within a block, if applicable.                                                              |
| `span`           | `int`                 | `-1`                       | Span identifier within a line, for finer granularity.                                                   |
| `nearby_objects` | `NearbyObjectsSchema` | `NearbyObjectsSchema()`    | Information about objects (text, images, structured data) near the current content. See [NearbyObjectsSchema](#nearbyobjectsschema). |

### `NearbyObjectsSchema` (Currently Unused)
Container for different types of nearby objects.

| Field        | Type                   | Default Value                | Description                                                              |
|--------------|------------------------|------------------------------|--------------------------------------------------------------------------|
| `text`       | `NearbyObjectsSubSchema` | `NearbyObjectsSubSchema()`   | Nearby textual objects. See [NearbyObjectsSubSchema](#nearbyobjectssubschema). |
| `images`     | `NearbyObjectsSubSchema` | `NearbyObjectsSubSchema()`   | Nearby image objects.                                                    |
| `structured` | `NearbyObjectsSubSchema` | `NearbyObjectsSubSchema()`   | Nearby structured data objects (e.g., tables).                           |

### `NearbyObjectsSubSchema`
Describes a list of nearby objects of a specific type.

| Field     | Type          | Default Value        | Description                                                              |
|-----------|---------------|----------------------|--------------------------------------------------------------------------|
| `content` | `List[str]`   | `default_factory=list` | List of content strings for the nearby objects.                            |
| `bbox`    | `List[tuple]` | `default_factory=list` | List of bounding boxes (e.g., coordinates) for the nearby objects.       |
| `type`    | `List[str]`   | `default_factory=list` | List of types for the nearby objects.                                    |

### `TextMetadataSchema`
Specific metadata for textual content.

| Field                          | Type                             | Default Value       | Description                                                                                             |
|--------------------------------|----------------------------------|---------------------|---------------------------------------------------------------------------------------------------------|
| `text_type`                    | `TextTypeEnum`                   | *Required*          | Type of text (e.g., `document`, `title`, `ocr`). Uses `TextTypeEnum`.                                   |
| `summary`                      | `str`                            | `""`                | A summary of this specific text segment.                                                                |
| `keywords`                     | `Union[str, List[str], Dict]`    | `""`                | Keywords extracted from or associated with the text. Can be a single string, list of strings, or a dictionary. |
| `language`                     | `LanguageEnum`                   | `"en"`              | Detected or specified language of the text. Uses `LanguageEnum`. Defaults to English.                   |
| `text_location`                | `tuple`                          | `(0, 0, 0, 0)`      | Bounding box or coordinates of the text within its source (e.g., on a page).                            |
| `text_location_max_dimensions` | `tuple`                          | `(0, 0, 0, 0)`      | Maximum dimensions of the space where `text_location` is defined (e.g., page width/height).           |

### `ImageMetadataSchema`
Specific metadata for image content.

| Field                             | Type                               | Default Value              | Description                                                                                             |
|-----------------------------------|------------------------------------|----------------------------|---------------------------------------------------------------------------------------------------------|
| `image_type`                      | `Union[DocumentTypeEnum, str]`     | *Required*                 | Type of the image document (e.g., `png`, `jpeg`). Uses `DocumentTypeEnum` or a string.                  |
| `structured_image_type`           | `ContentTypeEnum`                  | `ContentTypeEnum.NONE`     | If the image represents structured data (e.g., a table or chart), its `ContentTypeEnum`.                |
| `caption`                         | `str`                              | `""`                       | Caption associated with the image.                                                                      |
| `text`                            | `str`                              | `""`                       | Text extracted from the image (e.g., via OCR).                                                          |
| `image_location`                  | `tuple`                            | `(0, 0, 0, 0)`             | Bounding box or coordinates of the image within its source.                                             |
| `image_location_max_dimensions`   | `tuple`                            | `(0, 0)`                   | Maximum dimensions of the space where `image_location` is defined.                                      |
| `uploaded_image_url`              | `str`                              | `""`                       | URL of the image if it has been uploaded to a separate storage location.                                |
| `width`                           | `int`                              | `0`                        | Width of the image in pixels. Clamped to be non-negative.                                               |
| `height`                          | `int`                              | `0`                        | Height of the image in pixels. Clamped to be non-negative.                                              |

### `TableMetadataSchema`
Specific metadata for tabular content.

| Field                             | Type                                  | Default Value       | Description                                                                                             |
|-----------------------------------|---------------------------------------|---------------------|---------------------------------------------------------------------------------------------------------|
| `caption`                         | `str`                                 | `""`                | Caption associated with the table.                                                                      |
| `table_format`                    | `TableFormatEnum`                     | *Required*          | Format of the table (e.g., `csv`, `html`). Uses `TableFormatEnum`.                                      |
| `table_content`                   | `str`                                 | `""`                | String representation of the table's content (e.g., CSV string, HTML markup).                           |
| `table_content_format`            | `Union[TableFormatEnum, str]`         | `""`                | Specific format of `table_content`.                                                                     |
| `table_location`                  | `tuple`                               | `(0, 0, 0, 0)`      | Bounding box or coordinates of the table within its source.                                             |
| `table_location_max_dimensions`   | `tuple`                               | `(0, 0)`            | Maximum dimensions of the space where `table_location` is defined.                                      |
| `uploaded_image_uri`              | `str`                                 | `""`                | URI of an image representation of the table, if applicable.                                             |

### `ChartMetadataSchema`
Specific metadata for chart content. (Currently identical in structure to `TableMetadataSchema` but semantically distinct).
**Note:** The entries refer to tables despite being in the ChartMetadataSchema as charts and tables were originally grouped into a single output in the metadata spec.

| Field                             | Type                                  | Default Value       | Description                                                                                             |
|-----------------------------------|---------------------------------------|---------------------|---------------------------------------------------------------------------------------------------------|
| `caption`                         | `str`                                 | `""`                | Caption associated with the chart.                                                                      |
| `table_format`                    | `TableFormatEnum`                     | *Required*          | Underlying data format of the chart (e.g., data might be in `csv` format). Uses `TableFormatEnum`.      |
| `table_content`                   | `str`                                 | `""`                | String representation of the chart's underlying data.                                                   |
| `table_content_format`            | `Union[TableFormatEnum, str]`         | `""`                | Specific format of `table_content`.                                                                     |
| `table_location`                  | `tuple`                               | `(0, 0, 0, 0)`      | Bounding box or coordinates of the chart within its source.                                             |
| `table_location_max_dimensions`   | `tuple`                               | `(0, 0)`            | Maximum dimensions of the space where `table_location` is defined.                                      |
| `uploaded_image_uri`              | `str`                                 | `""`                | URI of an image representation of the chart, if applicable.                                             |

### `AudioMetadataSchema`
Specific metadata for audio content.

| Field              | Type  | Default Value | Description                                     |
|--------------------|-------|---------------|-------------------------------------------------|
| `audio_transcript` | `str` | `""`          | Transcript of the audio content.                |
| `audio_type`       | `str` | `""`          | Type or format of the audio (e.g., `mp3`, `wav`). |

### `ErrorMetadataSchema` (Currently Unused)
Metadata describing errors encountered during processing.

| Field       | Type           | Default Value | Description                                                              |
|-------------|----------------|---------------|--------------------------------------------------------------------------|
| `task`      | `TaskTypeEnum` | *Required*    | The task that was being performed when the error occurred. Uses `TaskTypeEnum`. |
| `status`    | `StatusEnum`   | *Required*    | The status indicating failure. Uses `StatusEnum`.                          |
| `source_id` | `str`          | `""`          | Identifier of the source item that caused the error, if applicable.        |
| `error_msg` | `str`          | *Required*    | The error message.                                                       |

### `InfoMessageMetadataSchema` (Currently Unused)
Informational messages related to processing.

| Field     | Type           | Default Value | Description                                                              |
|-----------|----------------|---------------|--------------------------------------------------------------------------|
| `task`    | `TaskTypeEnum` | *Required*    | The task associated with this informational message. Uses `TaskTypeEnum`.  |
| `status`  | `StatusEnum`   | *Required*    | The status related to this message (e.g., `INFO`, `WARNING`). Uses `StatusEnum`. |
| `message` | `str`          | *Required*    | The informational message content.                                       |
| `filter`  | `bool`         | *Required*    | A flag indicating if this message should be used for filtering purposes.   |

## Enums Used

This schema relies on several enums defined in `nv_ingest_api.internal.enums.common`:

*   `AccessLevelEnum`: Defines access levels (e.g., `PUBLIC`, `CONFIDENTIAL`, `UNKNOWN`).
*   `ContentTypeEnum`: Defines types of content (e.g., `TEXT`, `IMAGE`, `AUDIO`, `STRUCTURED`, `NONE`).
*   `TextTypeEnum`: Defines types of text (e.g., `DOCUMENT`, `TITLE`, `OCR`, `CAPTION`).
*   `LanguageEnum`: Defines languages (e.g., `ENGLISH` (`en`), `SPANISH` (`es`)).
*   `TableFormatEnum`: Defines table formats (e.g., `CSV`, `HTML`, `TEXT`).
*   `StatusEnum`: Defines processing statuses (e.g., `SUCCESS`, `FAILURE`, `PROCESSING`, `INFO`, `WARNING`).
*   `DocumentTypeEnum`: Defines types of source documents (e.g., `PDF`, `DOCX`, `TXT`, `URL`, `PNG`, `MP3`).
*   `TaskTypeEnum`: Defines types of processing tasks (e.g., `EXTRACTION`, `EMBEDDING`, `STORAGE`).
