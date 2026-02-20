#!/bin/bash

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - Edit these values to match your deployment
# ═══════════════════════════════════════════════════════════════════════════════

# Model names for each NIM service
# Each service has an inference model and a pipeline model (for ensemble metrics)
# Use "pipeline" if your NIM exposes generic pipeline metrics, or the ensemble name

# Page elements (YOLOX-based) - uses generic "pipeline" model
PAGE_ELEMENTS_MODEL="yolox"
PAGE_ELEMENTS_PIPELINE="pipeline"

# Graphic elements (YOLOX-based) - uses yolox_ensemble
GRAPHIC_ELEMENTS_MODEL="yolox"
GRAPHIC_ELEMENTS_PIPELINE="pipeline"

# Table structure (YOLOX-based) - uses yolox_ensemble
TABLE_STRUCTURE_MODEL="yolox"
TABLE_STRUCTURE_PIPELINE="pipeline"

# OCR models - uses scene_text_ensemble
OCR_DETECTION_MODEL="scene_text_det"
OCR_RECOGNITION_MODEL="scene_text_rec"
OCR_PIPELINE="pipeline"

# Embedding model
EMBEDDING_MODEL="nvidia_llama_3_2_nv_embedqa_1b_v2"
EMBEDDING_PIPELINE="pipeline"

# NIM service ports
PORT_PAGE_ELEMENTS=8002
PORT_GRAPHIC_ELEMENTS=8005
PORT_TABLE_STRUCTURE=8008
PORT_OCR=8011
PORT_EMBEDDING=8014

# Redis container name
REDIS_CONTAINER="nv-ingest-redis-1"

# ═══════════════════════════════════════════════════════════════════════════════

# Colors
Y='\e[33m'; C='\e[36m'; G='\e[32m'; M='\e[35m'; RED='\e[31m'; R='\e[0m'

echo "══════════════════════════════════════════════════════════════════════════════"
echo "  $(date '+%Y-%m-%d %H:%M:%S')  NV-INGEST STATUS"
echo "══════════════════════════════════════════════════════════════════════════════"

format_num() {
    local num=$1
    if [ "${num:-0}" -ge 1000000 ]; then printf "%.1fM" "$(echo "${num:-0}/1000000" | bc -l 2>/dev/null)"
    elif [ "${num:-0}" -ge 1000 ]; then printf "%.1fK" "$(echo "${num:-0}/1000" | bc -l 2>/dev/null)"
    else echo "${num:-0}"; fi
}

# ═══ JOB STATUS ═══
echo -e "\n${Y}▸ JOBS${R} (Queued=waiting in Redis, InFlight=processing in Ray, Done=completed & retrieved)"
job_states=$(docker exec $REDIS_CONTAINER redis-cli --no-raw EVAL "
local keys = redis.call('KEYS', 'job_state:*')
local counts = {}
for i, key in ipairs(keys) do
    local state = redis.call('GET', key)
    counts[state] = (counts[state] or 0) + 1
end
local result = {}
for state, count in pairs(counts) do
    table.insert(result, state .. ':' .. count)
end
return result
" 0 2>/dev/null)

submitted=0; completed=0; failed=0
while IFS= read -r line; do
    [[ "$line" =~ SUBMITTED:([0-9]+) ]] && submitted=${BASH_REMATCH[1]}
    [[ "$line" =~ RETRIEVED.*:([0-9]+) ]] && completed=$((completed + ${BASH_REMATCH[1]}))
    [[ "$line" =~ FAILED:([0-9]+) ]] && failed=${BASH_REMATCH[1]}
done <<< "$job_states"

inflight=$(curl -s "http://localhost:8265/logical/actors" 2>/dev/null | python3 -c "
import sys,json
raw=sys.stdin.read()
if raw.startswith('<!'):raw=raw[raw.find('{'):]
try:
    d=json.loads(raw);t=0
    for a in d.get('data',{}).get('actors',{}).values():
        if a.get('state')=='ALIVE':t+=a.get('numPendingTasks',0)+a.get('taskQueueLength',0)
    print(t)
except:print(0)
" 2>/dev/null)
inflight=${inflight:-0}

total=$((submitted + inflight + completed + failed))
printf "  %-10s %10s %10s %10s %10s\n" "Status" "Queued" "InFlight" "Completed" "Failed"
if [ "$submitted" -gt 0 ] || [ "$inflight" -gt 0 ]; then
    printf "  %-10s ${Y}%10s${R} ${C}%10s${R} %10s %10s\n" "Count" "$submitted" "$inflight" "$completed" "$failed"
else
    printf "  %-10s %10s %10s %10s %10s\n" "Count" "$submitted" "$inflight" "$completed" "$failed"
fi

# ═══ REDIS QUEUES ═══
echo -e "\n${Y}▸ REDIS QUEUES${R} (incoming document queues by size priority)"
printf "  %-10s %10s %10s %10s %10s %10s\n" "Queue" "main" "immediate" "small" "medium" "large"
printf "  %-10s" "Depth"
for q in ingest_task_queue ingest_task_queue_immediate ingest_task_queue_small ingest_task_queue_medium ingest_task_queue_large; do
    depth=$(docker exec $REDIS_CONTAINER redis-cli LLEN $q 2>/dev/null || echo "?")
    if [ "$depth" != "?" ] && [ "$depth" -gt 0 ]; then
        printf " ${Y}%10s${R}" "$depth"
    else
        printf " %10s" "$depth"
    fi
done
echo ""

# ═══ REDIS POOL ═══
echo -e "\n${Y}▸ REDIS POOL${R} (connection pool utilization)"
# Fetch metrics from nv-ingest API endpoint
POOL_METRICS=$(curl -s "http://localhost:7670/v1/metrics" 2>/dev/null)
if [ -n "$POOL_METRICS" ]; then
    # Parse pool metrics using Python for reliable parsing
    echo "$POOL_METRICS" | python3 -c "
import sys
import re

metrics_text = sys.stdin.read()

# Parse each metric type
def get_metric(name, label_filter=''):
    pattern = rf'{name}\{{[^}}]*{label_filter}[^}}]*\}}\s+([\d.]+)'
    match = re.search(pattern, metrics_text)
    return float(match.group(1)) if match else 0.0

# Extract pool metrics (look for pool_name=\"ingest\")
in_use = get_metric('redis_pool_connections_in_use', 'pool_name=\"ingest\"')
available = get_metric('redis_pool_connections_available', 'pool_name=\"ingest\"')
max_conn = get_metric('redis_pool_connections_max', 'pool_name=\"ingest\"')
timeouts = get_metric('redis_pool_timeout_total', 'pool_name=\"ingest\"')

# Calculate utilization
util_pct = (in_use / max_conn * 100) if max_conn > 0 else 0

# Parse histogram buckets for wait time distribution
wait_buckets = {}
for match in re.finditer(r'redis_pool_wait_seconds_bucket\{[^}]*pool_name=\"ingest\"[^}]*le=\"([^\"]+)\"\}\s+([\d.]+)', metrics_text):
    le, count = match.groups()
    wait_buckets[le] = float(count)

# Calculate wait time distribution
prev_count = 0
le_50ms = wait_buckets.get('0.05', 0) - prev_count
prev_count = wait_buckets.get('0.05', 0)
le_1s = wait_buckets.get('1.0', 0) - prev_count
prev_count = wait_buckets.get('1.0', 0)
le_5s = wait_buckets.get('5.0', 0) - prev_count

# Color codes
Y = '\033[33m'
G = '\033[32m'
RED = '\033[31m'
R = '\033[0m'

# Determine utilization color
if util_pct >= 80:
    util_color = RED
elif util_pct >= 50:
    util_color = Y
else:
    util_color = G

# Timeout color
timeout_color = RED if timeouts > 0 else ''
timeout_reset = R if timeouts > 0 else ''

print(f'  {\"Pool\":<12} {\"InUse\":>6} {\"Avail\":>6} {\"Max\":>6} {\"Util%\":>8} {\"Timeouts\":>9}')
print(f'  {\"ingest\":<12} {int(in_use):>6} {int(available):>6} {int(max_conn):>6} {util_color}{util_pct:>7.0f}%{R} {timeout_color}{int(timeouts):>9}{timeout_reset}')
if wait_buckets:
    print(f'  Wait time: <50ms: {int(le_50ms)} | <1s: {int(le_1s)} | <5s: {int(le_5s)}')
" 2>/dev/null || echo "  (metrics not available - check if nv-ingest is running)"
else
    echo "  (unable to fetch metrics from http://localhost:7670/v1/metrics)"
fi

# ═══ NIM METRICS ═══
echo -e "\n${Y}▸ NIM METRICS${R} (PipeQ=Pipeline Queue, PipeCp=Pipeline Compute, InfQ=Inference Queue, InfCp=GPU Compute)"
printf "  %-11s %7s %5s %4s %5s %7s %7s %7s %7s %4s %6s\n" "Model" "Items" "Batch" "Pend" "Fail" "PipeQ" "PipeCp" "InfQ" "InfCp" "GPU%" "GPUMem"

get_metrics() {
    local name=$1 port=$2 model=$3 pipe_model=$4
    metrics=$(curl -s "http://localhost:${port}/metrics" 2>/dev/null)

    inputs=$(echo "$metrics" | grep "nv_inference_count{" | grep "model=\"${model}\"" | grep -oE '[0-9.]+$' | awk '{s+=$1}END{print s+0}')
    execs=$(echo "$metrics" | grep "nv_inference_exec_count{" | grep "model=\"${model}\"" | grep -oE '[0-9.]+$' | awk '{s+=$1}END{print s+0}')
    pipe_execs=$(echo "$metrics" | grep "nv_inference_exec_count{" | grep "model=\"${pipe_model}\"" | grep -oE '[0-9.]+$' | awk '{s+=$1}END{print s+0}')

    pending_pipe=$(echo "$metrics" | grep "nv_inference_pending_request_count{" | grep "model=\"${pipe_model}\"" | grep -oE '[0-9.]+$' | awk '{s+=$1}END{print s+0}')
    pending_inf=$(echo "$metrics" | grep "nv_inference_pending_request_count{" | grep "model=\"${model}\"" | grep -oE '[0-9.]+$' | awk '{s+=$1}END{print s+0}')
    pending=$((${pending_pipe:-0} + ${pending_inf:-0}))

    # Failed requests
    failed_pipe=$(echo "$metrics" | grep "nv_inference_request_failure{" | grep "model=\"${pipe_model}\"" | grep -oE '[0-9.]+$' | awk '{s+=$1}END{print s+0}')
    failed_inf=$(echo "$metrics" | grep "nv_inference_request_failure{" | grep "model=\"${model}\"" | grep -oE '[0-9.]+$' | awk '{s+=$1}END{print s+0}')
    failed=$((${failed_pipe:-0} + ${failed_inf:-0}))

    pipe_queue_us=$(echo "$metrics" | grep "nv_inference_queue_duration_us{" | grep "model=\"${pipe_model}\"" | grep -oE '[0-9.]+$' | awk '{s+=$1}END{print s+0}')
    pipe_compute_us=$(echo "$metrics" | grep "nv_inference_compute_infer_duration_us{" | grep "model=\"${pipe_model}\"" | grep -oE '[0-9.]+$' | awk '{s+=$1}END{print s+0}')
    inf_queue_us=$(echo "$metrics" | grep "nv_inference_queue_duration_us{" | grep "model=\"${model}\"" | grep -oE '[0-9.]+$' | awk '{s+=$1}END{print s+0}')
    inf_compute_us=$(echo "$metrics" | grep "nv_inference_compute_infer_duration_us{" | grep "model=\"${model}\"" | grep -oE '[0-9.]+$' | awk '{s+=$1}END{print s+0}')

    gpu_util=$(echo "$metrics" | grep "^nv_gpu_utilization" | head -1 | grep -oE '[0-9.]+$')
    gpu_mem=$(echo "$metrics" | grep "^nv_gpu_memory_used_bytes" | head -1 | awk '{printf "%.1f", $2/1024/1024/1024}')

    avg_batch="0"; avg_pipe_q="0"; avg_pipe_c="0"; avg_inf_q="0"; avg_inf_c="0"
    [ "${execs:-0}" -gt 0 ] && {
        avg_batch=$(echo "scale=1; ${inputs:-0}/$execs" | bc 2>/dev/null)
        avg_inf_q=$(echo "scale=0; $inf_queue_us/$execs/1000" | bc 2>/dev/null)
        avg_inf_c=$(echo "scale=0; $inf_compute_us/$execs/1000" | bc 2>/dev/null)
    }
    [ "${pipe_execs:-0}" -gt 0 ] && {
        avg_pipe_q=$(echo "scale=0; $pipe_queue_us/$pipe_execs/1000" | bc 2>/dev/null)
        avg_pipe_c=$(echo "scale=0; $pipe_compute_us/$pipe_execs/1000" | bc 2>/dev/null)
    }
    gpu_pct=$(echo "scale=0; ${gpu_util:-0}*100/1" | bc 2>/dev/null || echo "0")

    # Find bottleneck
    max=$avg_pipe_q; bn="PQ"
    [ "${avg_pipe_c:-0}" -gt "$max" ] && { max=$avg_pipe_c; bn="PC"; }
    [ "${avg_inf_q:-0}" -gt "$max" ] && { max=$avg_inf_q; bn="IQ"; }
    [ "${avg_inf_c:-0}" -gt "$max" ] && { max=$avg_inf_c; bn="IC"; }
    [ "$max" -eq 0 ] && bn="-"

    # Color GPU% based on utilization: Green>=80% (busy), Yellow>=50%, Red<50% (idle)
    if [ "${gpu_pct:-0}" -ge 80 ]; then
        gpu_color="${G}%4s%%${R}"
    elif [ "${gpu_pct:-0}" -ge 50 ]; then
        gpu_color="${Y}%4s%%${R}"
    else
        gpu_color="${RED}%4s%%${R}"
    fi

    # Format failed count with color (red if > 0)
    if [ "${failed:-0}" -gt 0 ]; then
        fail_fmt="${RED}%5s${R}"
    else
        fail_fmt="%5s"
    fi

    # Color the bottleneck column
    case $bn in
        PQ) printf "  %-11s %7s %5s %4s $fail_fmt ${Y}%5sms${R} %5sms %5sms %5sms $gpu_color %5sG\n" "$name" "$(format_num $inputs)" "$avg_batch" "$pending" "$(format_num $failed)" "$avg_pipe_q" "$avg_pipe_c" "$avg_inf_q" "$avg_inf_c" "$gpu_pct" "${gpu_mem:-?}" ;;
        PC) printf "  %-11s %7s %5s %4s $fail_fmt %5sms ${M}%5sms${R} %5sms %5sms $gpu_color %5sG\n" "$name" "$(format_num $inputs)" "$avg_batch" "$pending" "$(format_num $failed)" "$avg_pipe_q" "$avg_pipe_c" "$avg_inf_q" "$avg_inf_c" "$gpu_pct" "${gpu_mem:-?}" ;;
        IQ) printf "  %-11s %7s %5s %4s $fail_fmt %5sms %5sms ${C}%5sms${R} %5sms $gpu_color %5sG\n" "$name" "$(format_num $inputs)" "$avg_batch" "$pending" "$(format_num $failed)" "$avg_pipe_q" "$avg_pipe_c" "$avg_inf_q" "$avg_inf_c" "$gpu_pct" "${gpu_mem:-?}" ;;
        IC) printf "  %-11s %7s %5s %4s $fail_fmt %5sms %5sms %5sms ${G}%5sms${R} $gpu_color %5sG\n" "$name" "$(format_num $inputs)" "$avg_batch" "$pending" "$(format_num $failed)" "$avg_pipe_q" "$avg_pipe_c" "$avg_inf_q" "$avg_inf_c" "$gpu_pct" "${gpu_mem:-?}" ;;
        *)  printf "  %-11s %7s %5s %4s $fail_fmt %5sms %5sms %5sms %5sms $gpu_color %5sG\n" "$name" "$(format_num $inputs)" "$avg_batch" "$pending" "$(format_num $failed)" "$avg_pipe_q" "$avg_pipe_c" "$avg_inf_q" "$avg_inf_c" "$gpu_pct" "${gpu_mem:-?}" ;;
    esac
}

get_metrics "page-elem" $PORT_PAGE_ELEMENTS "$PAGE_ELEMENTS_MODEL" "$PAGE_ELEMENTS_PIPELINE"
get_metrics "graphic-el" $PORT_GRAPHIC_ELEMENTS "$GRAPHIC_ELEMENTS_MODEL" "$GRAPHIC_ELEMENTS_PIPELINE"
get_metrics "table-str" $PORT_TABLE_STRUCTURE "$TABLE_STRUCTURE_MODEL" "$TABLE_STRUCTURE_PIPELINE"
get_metrics "ocr-det" $PORT_OCR "$OCR_DETECTION_MODEL" "$OCR_PIPELINE"
get_metrics "ocr-rec" $PORT_OCR "$OCR_RECOGNITION_MODEL" "$OCR_PIPELINE"
get_metrics "embedding" $PORT_EMBEDDING "$EMBEDDING_MODEL" "$EMBEDDING_PIPELINE"

# ═══ BOTTLENECK SUMMARY ═══
echo -e "\n${Y}▸ BOTTLENECK SUMMARY${R} (which component dominates latency)"
printf "  %-11s %8s %9s %8s %8s   %-20s\n" "Model" "PipeQ" "PipeCp" "InfQ" "InfCp" "Bottleneck"

show_bottleneck() {
    local name=$1 port=$2 model=$3 pipe_model=$4
    metrics=$(curl -s "http://localhost:${port}/metrics" 2>/dev/null)

    execs=$(echo "$metrics" | grep "nv_inference_exec_count{" | grep "model=\"${model}\"" | grep -oE '[0-9.]+$' | awk '{s+=$1}END{print s+0}')
    pipe_execs=$(echo "$metrics" | grep "nv_inference_exec_count{" | grep "model=\"${pipe_model}\"" | grep -oE '[0-9.]+$' | awk '{s+=$1}END{print s+0}')

    pipe_queue_us=$(echo "$metrics" | grep "nv_inference_queue_duration_us{" | grep "model=\"${pipe_model}\"" | grep -oE '[0-9.]+$' | awk '{s+=$1}END{print s+0}')
    pipe_compute_us=$(echo "$metrics" | grep "nv_inference_compute_infer_duration_us{" | grep "model=\"${pipe_model}\"" | grep -oE '[0-9.]+$' | awk '{s+=$1}END{print s+0}')
    inf_queue_us=$(echo "$metrics" | grep "nv_inference_queue_duration_us{" | grep "model=\"${model}\"" | grep -oE '[0-9.]+$' | awk '{s+=$1}END{print s+0}')
    inf_compute_us=$(echo "$metrics" | grep "nv_inference_compute_infer_duration_us{" | grep "model=\"${model}\"" | grep -oE '[0-9.]+$' | awk '{s+=$1}END{print s+0}')

    avg_pipe_q="0"; avg_pipe_c="0"; avg_inf_q="0"; avg_inf_c="0"
    [ "${execs:-0}" -gt 0 ] && {
        avg_inf_q=$(echo "scale=0; $inf_queue_us/$execs/1000" | bc 2>/dev/null)
        avg_inf_c=$(echo "scale=0; $inf_compute_us/$execs/1000" | bc 2>/dev/null)
    }
    [ "${pipe_execs:-0}" -gt 0 ] && {
        avg_pipe_q=$(echo "scale=0; $pipe_queue_us/$pipe_execs/1000" | bc 2>/dev/null)
        avg_pipe_c=$(echo "scale=0; $pipe_compute_us/$pipe_execs/1000" | bc 2>/dev/null)
    }

    max=${avg_pipe_q:-0}; bn="${Y}PipeQ (upstream)${R}"
    [ "${avg_pipe_c:-0}" -gt "$max" ] && { max=$avg_pipe_c; bn="${M}PipeCp (pre/post)${R}"; }
    [ "${avg_inf_q:-0}" -gt "$max" ] && { max=$avg_inf_q; bn="${C}InfQ (GPU busy)${R}"; }
    [ "${avg_inf_c:-0}" -gt "$max" ] && { max=$avg_inf_c; bn="${G}InfCp (compute)${R}"; }
    [ "$max" -eq 0 ] && bn="-"

    printf "  %-11s %7sms %8sms %7sms %7sms   %b\n" "$name" "$avg_pipe_q" "$avg_pipe_c" "$avg_inf_q" "$avg_inf_c" "$bn"
}

show_bottleneck "page-elem" $PORT_PAGE_ELEMENTS "$PAGE_ELEMENTS_MODEL" "$PAGE_ELEMENTS_PIPELINE"
show_bottleneck "graphic-el" $PORT_GRAPHIC_ELEMENTS "$GRAPHIC_ELEMENTS_MODEL" "$GRAPHIC_ELEMENTS_PIPELINE"
show_bottleneck "table-str" $PORT_TABLE_STRUCTURE "$TABLE_STRUCTURE_MODEL" "$TABLE_STRUCTURE_PIPELINE"
show_bottleneck "ocr-det" $PORT_OCR "$OCR_DETECTION_MODEL" "$OCR_PIPELINE"
show_bottleneck "ocr-rec" $PORT_OCR "$OCR_RECOGNITION_MODEL" "$OCR_PIPELINE"
show_bottleneck "embedding" $PORT_EMBEDDING "$EMBEDDING_MODEL" "$EMBEDDING_PIPELINE"

# ═══ BATCHING EFFICIENCY ═══
echo -e "\n${Y}▸ BATCHING EFFICIENCY${R} (images per request to NIM)"
printf "  %-11s %10s %10s %8s\n" "Model" "Requests" "Items" "Ratio"

show_batching() {
    local name=$1 port=$2 model=$3 pipe_model=$4
    metrics=$(curl -s "http://localhost:${port}/metrics" 2>/dev/null)

    pipeline_req=$(echo "$metrics" | grep "nv_inference_request_success{" | grep "model=\"${pipe_model}\"" | grep -oE '[0-9.]+$' | awk '{s+=$1}END{print s+0}')
    inference_items=$(echo "$metrics" | grep "nv_inference_count{" | grep "model=\"${model}\"" | grep -oE '[0-9.]+$' | awk '{s+=$1}END{print s+0}')

    if [ "${pipeline_req:-0}" -gt 0 ]; then
        ratio=$(echo "scale=1; ${inference_items:-0}/$pipeline_req" | bc 2>/dev/null || echo "0")
    else
        ratio="0"
    fi

    printf "  %-11s %10s %10s %7sx\n" "$name" "$(format_num $pipeline_req)" "$(format_num $inference_items)" "$ratio"
}

show_batching "page-elem" $PORT_PAGE_ELEMENTS "$PAGE_ELEMENTS_MODEL" "$PAGE_ELEMENTS_PIPELINE"
show_batching "graphic-el" $PORT_GRAPHIC_ELEMENTS "$GRAPHIC_ELEMENTS_MODEL" "$GRAPHIC_ELEMENTS_PIPELINE"
show_batching "table-str" $PORT_TABLE_STRUCTURE "$TABLE_STRUCTURE_MODEL" "$TABLE_STRUCTURE_PIPELINE"
show_batching "ocr" $PORT_OCR "$OCR_DETECTION_MODEL" "$OCR_PIPELINE"

# ═══ RAY PIPELINE ═══
echo -e "\n${Y}▸ RAY PIPELINE${R} (Repl=Replicas, Pend=Pending tasks, Queue=Task queue depth)"
TMPFILE=$(mktemp)
curl -s "http://localhost:9411/api/v2/traces?serviceName=nv-ingest&limit=50&minDuration=1000" 2>/dev/null > "$TMPFILE"
curl -s "http://localhost:8265/logical/actors" 2>/dev/null | ZIPKIN_FILE="$TMPFILE" python3 -c "
import sys,json,os
from collections import defaultdict
raw=sys.stdin.read()
if raw.startswith('<!'):raw=raw[raw.find('{'):]

stage_times=defaultdict(list)
try:
    with open(os.environ.get('ZIPKIN_FILE',''),'r') as f:traces=json.load(f)
    for t in traces:
        for s in t:
            n,d=s.get('name',''),s.get('duration',0)/1000
            if d<100000 and len(n)!=36 and not n.endswith('-v2') and '_channel_in' not in n:stage_times[n].append(d)
except:pass

m={'PDFExtractorStage':'pdf_extractor','TableExtractorStage':'table_extractor','ChartExtractorStage':'chart_extractor',
   'InfographicExtractorStage':'infographic_extractor','TextEmbeddingTransformStage':'text_embedder'}
try:
    d=json.loads(raw);actors=d.get('data',{}).get('actors',{})
    stats=defaultdict(lambda:{'p':0,'q':0,'e':0,'c':0})
    for a in actors.values():
        if a.get('state')!='ALIVE':continue
        c=a.get('actorClass','')
        stats[c]['p']+=a.get('numPendingTasks',0);stats[c]['q']+=a.get('taskQueueLength',0)
        stats[c]['e']+=a.get('numExecutedTasks',0);stats[c]['c']+=1

    print(f'  {\"Stage\":<12} {\"Repl\":>4} {\"Pend\":>4} {\"Queue\":>5} {\"Done\":>6} {\"AvgTime\":>8} {\"P95\":>8}')
    tp,tq=0,0
    for cls,nm in [('PDFExtractorStage','PDF'),('TableExtractorStage','Table'),('ChartExtractorStage','Chart'),
                   ('InfographicExtractorStage','Infograph'),('TextEmbeddingTransformStage','Embed')]:
        if cls in stats:
            s=stats[cls];tp+=s['p'];tq+=s['q']
            ts=sorted(stage_times.get(m.get(cls,''),[]))
            avg=f'{sum(ts)/len(ts):>6.0f}ms' if ts else '      -'
            p95=f'{ts[int(len(ts)*0.95)] if ts else 0:>6.0f}ms' if ts else '      -'
            w=s['p']>0 or s['q']>0
            if w:print(f'  \033[33m{nm:<12} {s[\"c\"]:>4} {s[\"p\"]:>4} {s[\"q\"]:>5} {s[\"e\"]:>6} {avg} {p95}\033[0m')
            else:print(f'  {nm:<12} {s[\"c\"]:>4} {s[\"p\"]:>4} {s[\"q\"]:>5} {s[\"e\"]:>6} {avg} {p95}')
    if tp>0 or tq>0:print(f'  \033[33mBackpressure: {tp} pending, {tq} queued\033[0m')
except Exception as e:print(f'  Error: {e}')
"
rm -f "$TMPFILE"

# ═══ REPLICAS & CPU ═══
echo -e "\n${Y}▸ REPLICAS${R} (alive/total actors per stage)"
curl -s "http://localhost:8265/logical/actors" 2>/dev/null | python3 -c "
import sys,json
from collections import Counter
raw=sys.stdin.read()
if raw.startswith('<!'):raw=raw[raw.find('{'):]
try:
    d=json.loads(raw);actors=d.get('data',{}).get('actors',{})
    alive,total=Counter(),Counter()
    for a in actors.values():
        c=a.get('actorClass','?');total[c]+=1
        if a.get('state')=='ALIVE':alive[c]+=1
    parts=[]
    for cls,nm in [('PDFExtractorStage','PDF'),('TableExtractorStage','Table'),('ChartExtractorStage','Chart'),
                   ('InfographicExtractorStage','Infograph'),('TextEmbeddingTransformStage','Embed')]:
        if cls in total:parts.append(f'{nm}:{alive[cls]}/{total[cls]}')
    print('  '+' | '.join(parts)+f' | Total:{sum(alive.values())}/{len(actors)}')
except Exception as e:print(f'  Error:{e}')
"

echo -e "\n${Y}▸ CPU (top 3)${R}"
docker stats --no-stream --format "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null | grep "^nv-ingest" | sort -t$'\t' -k2 -rn | head -3 | while read line; do
    name=$(echo "$line" | awk -F'\t' '{print $1}' | sed 's/nv-ingest-//' | sed 's/-1$//')
    cpu=$(echo "$line" | awk -F'\t' '{print $2}')
    mem=$(echo "$line" | awk -F'\t' '{print $3}' | cut -d'/' -f1)
    printf "  %-25s %8s %12s\n" "$name" "$cpu" "$mem"
done
echo ""
