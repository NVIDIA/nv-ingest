import json
import pandas as pd
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def cosine_sim(text1, text2):
  vectorizer = TfidfVectorizer()
  vectors = vectorizer.fit_transform([text1, text2])
  similarity = cosine_similarity(vectors)
  return similarity[0][1]

def bo20_validate(extract_fn, chart_validation_fn, table_validation_fn, dump_dir):
  print(f"Validating bo20 extracts from {extract_fn}")
  extract_results = json.loads(open(extract_fn).read())

  chart_df = pd.read_csv(chart_validation_fn)
  table_df = pd.read_csv(table_validation_fn)

  result_lines = []
  for pdf_extract in extract_results:
    pdf_path = pdf_extract[0]["metadata"]["source_metadata"]["source_name"]
    fn = pdf_path.split("/")[-1]
    pdf_id = fn.split(".")[0]
    chart_rows = chart_df[chart_df["pdf"] == int(pdf_id)]
    bo20_charts = chart_rows.to_dict(orient='records')
    table_rows = table_df[table_df["pdf"] == int(pdf_id)]
    bo20_tables = table_rows.to_dict(orient='records')

    text_extracts = [x for x in pdf_extract if x["document_type"] == "text"]
    text_extracts = [{
      "page_num": x["metadata"]["content_metadata"]["page_number"],
      "ingest_content": x["metadata"]["content"]
      } for x in text_extracts]

    table_extracts = [x for x in pdf_extract if x["document_type"] == "structured"]
    table_extracts = [{
      "page_num": x["metadata"]["content_metadata"]["page_number"],
      "ingest_content": x["metadata"]["table_metadata"]["table_content"]
      } for x in table_extracts]

    table_matches = []
    table_no_matches = []
    for table in bo20_tables:
      page_num = table["page"]
      bo20_content = table["paddle_ocr"]
      try:
        possible_matches = [x for x in table_extracts if x["page_num"] == page_num]
        similarities = [cosine_sim(bo20_content, poss_match["ingest_content"]) for poss_match in possible_matches]
        max_sim = max(similarities)
        match_id = similarities.index(max_sim)
        found_match = any(x > .8 for x in similarities)
        if found_match:
          table_matches.append({
            "fn": fn, "page_num": page_num, "bo20_content": bo20_content, "ingest_content": possible_matches[match_id]["ingest_content"], "cos_sim": max_sim
          })
        else:
          table_no_matches.append({
            "fn": fn, "page_num": page_num, "bo20_content": bo20_content
          })
      except Exception as e:
        print(f"TABLE_SIM: {fn}: {page_num} {bo20_content}, {e}")

    with open(f"{dump_dir}/table_matches_{fn}", "w") as fp:
      fp.write(json.dumps(table_matches))
    if len(table_no_matches) > 0:
      with open(f"{dump_dir}/table_no_matches_{fn}", "w") as fp:
        fp.write(json.dumps(table_no_matches))

    chart_matches = []
    chart_no_matches = []
    for chart in bo20_charts:
      page_num = chart["page"]
      bo20_content = chart["input"]
      try:
        possible_matches = [x for x in table_extracts if x["page_num"] == page_num]
        similarities = [cosine_sim(bo20_content, poss_match["ingest_content"]) for poss_match in possible_matches]
        found_match = False
        if len(similarities) > 0:
          max_sim = max(similarities)
          match_id = similarities.index(max_sim)
          found_match = any(x > .5 for x in similarities)
        if found_match:
          chart_matches.append({
            "fn": fn, "page_num": page_num, "bo20_content": bo20_content, "ingest_content": possible_matches[match_id]["ingest_content"], "cos_sim": max_sim
          })
        else:
          chart_no_matches.append({
            "fn": fn, "page_num": page_num, "bo20_content": bo20_content
          })
      except Exception as e:
        print(f"CHART_SIM {fn}: {page_num} \n {bo20_content}, \n{e}")

    with open(f"{dump_dir}/chart_matches_{fn}", "w") as fp:
      fp.write(json.dumps(chart_matches))
    if len(chart_no_matches) > 0:
      with open(f"{dump_dir}/chart_no_matches_{fn}", "w") as fp:
        fp.write(json.dumps(chart_no_matches))

    if len(table_no_matches) > 0 or len(chart_no_matches) > 0:
      result_lines.append(f"{fn}: {len(table_no_matches)} unmatched tables, {len(chart_no_matches)} unmatched charts")

  return str(len(extract_results)) + "/20 files processed:\n" + "\n".join(sorted(result_lines))
