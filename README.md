# RAG_chunking_embed_vector_store

Cắt tài liệu Markdown thành chunk, tạo vector BGE-M3 (dense + sparse) và BM25, lưu vào Qdrant, pgvector hoặc Milvus, tìm hybrid và chấm độ đúng của đoạn lấy ra.

Dự án dừng ở retrieval. Không xếp hạng lại và không sinh câu trả lời.

Mọi lệnh chạy từ thư mục gốc của repo. Tham số nằm trong file YAML, không viết trong lệnh.

## Dự án chạy thế nào

```text
Markdown -> chuẩn hóa -> cắt chunk -> BM25 + BGE-M3 -> collection -> search / eval
```

| Việc | File |
| --- | --- |
| Tham số mẫu, 4 bài trong `data/sample` | `config.yaml` |
| Tham số 20 bài trong `data/raw` và `data/eval/testset.jsonl` | `config.corpus.yaml` |
| Cắt chữ, embed, ghi kho, tìm, chấm | `rag/` |
| Lệnh `ingest`, `search`, `eval`, `compare` | `python -m rag.cli` |
| Chạy trên Colab | `notebooks/colab_demo.ipynb` |

`ingest` đọc Markdown, cắt, encode và ghi kho. `search` nhận một câu. `eval` chấm cả bộ câu hỏi. `compare` chấm lại bằng RRF phía client, để đặt cạnh kết quả gộp của Qdrant.

Chunk đúng khi chứa mọi cụm trong `answer_keys`, sau khi chuẩn hóa cả hai phía. Câu `text` và câu `table` được in riêng. Trong `testset.jsonl`, `type` là `paraphrase` hoặc `entity_number` thì tính là text, `table` thì tính là table.

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Trên Linux hoặc macOS, kích hoạt bằng `source .venv/bin/activate`.

Qdrant:

```bash
docker compose up -d qdrant
```

## Chạy trên dữ liệu mẫu

`config.yaml` trỏ `data/sample`, bộ câu `data/eval/questions.jsonl`, collection `medical_chunks_v1`.

Chưa có GPU hoặc chưa tải model thì thêm `--fake-embedder` vào từng lệnh. Cờ này tạo vector giả từ hash của token để kiểm tra luồng. Tokenizer lúc đó tách theo khoảng trắng, không phải tokenizer của BGE-M3. Điểm recall lúc này không dùng để kết luận chất lượng model.

```bash
python scripts/scan_corpus.py data/sample --chunk-size 512
python -m rag.cli ingest --config config.yaml --fake-embedder
python -m rag.cli search "liều paracetamol cho trẻ sốt" --branches dense,sparse --fake-embedder
python -m rag.cli eval --configs dense,sparse,bm25,dense+sparse --fake-embedder
python -m rag.cli compare --configs dense,sparse,dense+sparse --fake-embedder
```

Ba lệnh `ingest`, `search`, `eval` phải cùng kiểu. Đã nạp bằng `--fake-embedder` thì lúc tìm cũng phải có cờ đó. Đã nạp bằng BGE-M3 thì lúc tìm bỏ cờ.

## Chạy bằng BGE-M3

Tải model một lần vào đúng thư mục trong config:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', local_dir='models/bge-m3')"
```

Rồi bỏ `--fake-embedder`:

```bash
python -m rag.cli ingest --config config.yaml --recreate
python -m rag.cli search "liều paracetamol cho trẻ sốt" --config config.yaml --branches dense,sparse
python -m rag.cli eval --config config.yaml --configs dense,sparse,bm25,dense+sparse
```

`--recreate` cần khi collection đã chứa vector của lần nạp trước (kể cả FakeEmbedder). `use_fp16: true` chỉ có tác dụng khi máy có GPU. Trên CPU chương trình tự tắt fp16.

## Chạy 20 bài và testset.jsonl

20 file `data/raw/DOC_001.md` … `DOC_020.md` không nằm trong git. Bộ câu hỏi là `data/eval/testset.jsonl`. Cả hai đường dẫn đã ghi trong `config.corpus.yaml`, collection là `medical_corpus_v1`. Giữ nguyên `config.yaml` cho bản demo mẫu.

Thử luồng:

```bash
python -m rag.cli ingest --config config.corpus.yaml --fake-embedder --recreate
python -m rag.cli eval --config config.corpus.yaml --configs dense,sparse,bm25,dense+sparse --fake-embedder
```

Đo bằng BGE-M3, sau khi đã tải `models/bge-m3`:

```bash
python -m rag.cli ingest --config config.corpus.yaml --recreate
python -m rag.cli search "câu hỏi thử" --config config.corpus.yaml --branches dense,sparse
python -m rag.cli eval --config config.corpus.yaml --configs dense,sparse,bm25,dense+sparse
```

`search` và `eval` phải dùng cùng `--config` với lần `ingest`, vì từ điển BM25 và tên model nằm theo file đó.

## Đọc kết quả

`search` in hạng, điểm, `chunk_id`, loại `text` hoặc `table`, và đầu đoạn. Dùng lệnh này để xem một câu lấy đúng chunk chưa.

`eval` chấm cả bộ câu hỏi rồi đo thời gian. Dòng in ra có:

- `text`: tỷ lệ câu văn bản có ít nhất một chunk đúng trong top k
- `table`: tỷ lệ câu tra bảng
- `encode p50`: thời gian embed câu hỏi, đơn vị mili giây
- `query p50`: thời gian hỏi kho, đơn vị mili giây

Một câu được tính là đúng khi có ít nhất một chunk trong top k chứa mọi cụm của `answer_keys`. Đây là success@k trên từng câu, ghi trong báo cáo dưới tên `text_recall_at_k` và `table_recall_at_k`. Câu bảng cần cả cụm nằm trong cùng một chunk.

Giữ nguyên tên và cách tính các metric cũ. K lấy từ `search.top_k`, mặc định **10** trong cả hai file cấu hình. Màn hình in thêm Recall@K và nDCG@K bên cạnh kết quả cũ và P50. File `reports/<thời điểm>_<cấu hình>.json` thêm:

| Khóa | Ý nghĩa |
| --- | --- |
| `text_count`, `table_count` | Số câu mỗi loại |
| `text_retrieval_recall_at_k`, `table_retrieval_recall_at_k` | Recall@K: số chunk phù hợp khác nhau trong top K / tổng số chunk phù hợp trong corpus, trung bình theo câu hỏi |
| `text_ndcg_at_k`, `table_ndcg_at_k` | nDCG@K với nhãn nhị phân: tổng điểm phù hợp giảm theo `1/log2(rank+1)`, chia cho điểm của thứ hạng lý tưởng tại cùng K, trung bình theo câu hỏi |
| `text_no_relevant_count`, `table_no_relevant_count` | Số câu không có chunk nào phù hợp trong corpus; các câu này nhận 0 cho hai metric mới |
| `text_mrr`, `table_mrr` | Trung bình 1/hạng của chunk đúng đầu tiên. Không có chunk đúng thì hạng đó là 0 |
| `encode_ms`, `query_ms` | P50, P95, P99 |
| `warmup_runs`, `runs` | Số lượt bỏ đi và số lượt được tính |
| `model_name`, `library_versions`, `config`, `created_at` | Model, phiên bản thư viện, toàn bộ tham số và thời điểm chạy |

`compare` ghi `client_rrf` với cùng các khóa chất lượng, không đo độ trễ. Mỗi nhóm nhánh được tìm riêng với `prefetch_limit`, rồi gộp bằng `rrf_k` và cắt còn `top_k`.

Hai metric mới dò `answer_keys` trên **toàn bộ chunk đã ingest trong cache** để xác định tập chunk phù hợp, theo cùng quy tắc chứa đủ cụm chữ của metric cũ. Vì vậy đây là đánh giá relevance nhị phân dựa vào từ khóa, chưa phải nhãn do người đánh giá. Chunk lặp trong kết quả không được cộng điểm lần nữa. Ví dụ corpus có 3 chunk phù hợp nhưng top K chỉ tìm được 1: Success@K là 1, Recall@K là 1/3. nDCG còn xét vị trí các chunk phù hợp.

CLI và notebook truyền corpus từ cache vào `Evaluator(service, questions, chunks)`. Dùng cache của đúng collection đang chấm; CLI kiểm tra model và số chunk, evaluator kiểm tra ID/nội dung kết quả với corpus. Khi gọi API cũ `Evaluator(service, questions)` không có corpus, chỉ các metric cũ được trả về, không suy mẫu số Recall từ top K. Notebook giữ riêng corpus cho từng lần ingest để chấm dữ liệu mẫu và dữ liệu đầy đủ.

Đo độ trễ bằng `time.perf_counter` bên trong `SearchService`: một đồng hồ cho encode, một đồng hồ cho truy vấn. `warmup_runs` lượt đầu bị bỏ. `latency_runs` lượt sau xoay vòng bộ câu hỏi. Mặc định 10 và 500. Trên CPU với BGE-M3, hạ `eval.latency_runs` khi chỉ cần xem điểm.

Điểm từ `--fake-embedder` chỉ chứng minh lệnh chạy. Muốn kết luận chất lượng thì nạp và chấm bằng BGE-M3 trên `config.corpus.yaml`. Tám câu trong `questions.jsonl` quá ít để so model.

## Đo từng trường hợp

Giữ nguyên bộ câu hỏi, `search.top_k` và model trong suốt một bảng so sánh. Mỗi lần `eval` ghi một file riêng trong `reports/`. Đặt cạnh nhau bằng cách mở `config` và `model_name` trong JSON, rồi so `text_recall_at_k`, `table_recall_at_k`, `text_mrr`, `table_mrr`. So thời gian thì dùng P50 làm mức thường gặp, P95 và P99 làm mức chậm.

`encode_ms` là chi phí của model trên câu hỏi. `query_ms` là chi phí của kho. Đổi kho thì nhìn `query_ms`. Đổi model thì nhìn cả recall và `encode_ms`.

### Nhánh tìm

Một lần `ingest`, một lần `eval`. Bốn nhóm ra bốn khối trong cùng một báo cáo, cùng collection, cùng chunk, cùng model.

```bash
python -m rag.cli eval --config config.corpus.yaml --configs dense,sparse,bm25,dense+sparse
```

`dense` và `sparse` là hai đầu ra của cùng BGE-M3. `bm25` không dùng model embedding. `dense+sparse` là hybrid. Trên Qdrant, hybrid gộp ở server. Trên pgvector và Milvus, hybrid gộp ở client.

```bash
python -m rag.cli compare --config config.corpus.yaml --configs dense+sparse,dense+sparse+bm25
```

`compare` luôn gộp ở client. Dùng nó khi muốn đặt công thức RRF chung cạnh kết quả hybrid của Qdrant. Hai số chỉ so được khi cùng `top_k`, `prefetch_limit` và `rrf_k`.

### Top k, prefetch và RRF

Đổi `search.top_k` rồi `eval` lại, không cần `ingest`. Recall@10 và recall@5 là hai thước khác nhau: cùng một lần tìm có thể đúng ở k=10 và sai ở k=5. Ghi `top_k` trong báo cáo cạnh mỗi số.

`prefetch_limit` và `rrf_k` chỉ đổi hybrid và `compare`. Nhánh đơn `dense`, `sparse`, `bm25` không đọc hai số này. Muốn xem RRF kéo hạng thế nào, chấm `dense`, `sparse` và `dense+sparse` trong cùng một lệnh `eval`, rồi nhìn MRR: recall bằng nhau mà MRR cao hơn nghĩa là chunk đúng lên gần hạng 1.

### Cỡ chunk

Sửa `chunking.chunk_size` và `chunking.chunk_overlap`, `ingest --recreate`, rồi `eval` cùng `--configs`. So riêng `text_*` và `table_*`. Chunk nhỏ dễ giữ một hàng bảng trong một đoạn. Chunk lớn dễ giữ đủ câu dài nhưng có thể nuốt nhiều mục. `scripts/scan_corpus.py data/raw --chunk-size 512` in trước độ dài bài và số bảng, chạy trước khi đổi `chunk_size`.

### BM25 syllable và word

Sửa `bm25.mode`, `ingest --recreate`, rồi `eval` với `--configs bm25,dense+sparse+bm25`. Mode `word` tách từ bằng pyvi. Mode `syllable` tách theo khoảng trắng và chữ thường. So nhánh `bm25` trước. Nhánh `dense` không đổi theo mode này nếu model embedding và cách cắt giữ nguyên.

### Ba kho

Cùng model, cùng cách cắt, cùng `m`, `ef_construct`, `hnsw_ef`, `rrf_k`, `top_k`. Đổi `store`, bật service, `ingest --recreate`, `eval` cùng `--configs`. Lặp cho `qdrant`, `pgvector`, `milvus`.

```bash
docker compose up -d qdrant
python -m rag.cli ingest --config config.corpus.yaml --recreate
python -m rag.cli eval --config config.corpus.yaml --configs dense,sparse,bm25,dense+sparse
```

So `dense` và `sparse` giữa các kho để xem chỉ mục. Nhánh `bm25` và hybrid trên Qdrant có thêm IDF của server. pgvector và Milvus dùng đúng trọng số BM25 đã lưu rồi mới RRF, nên hai nhánh đó chênh với Qdrant dù vector đầu vào giống nhau.

Đo HNSW trên server Docker. Qdrant trong bộ nhớ của notebook tìm chính xác, bỏ qua `hnsw_ef`, nên `query_ms` ở notebook không phải số của HNSW. `full_scan_threshold` để 10 vì Qdrant 1.19.2 không nhận 0.

### Model embedding

Mỗi model một collection và một thư mục cache. Giữ `chunking`, `search.top_k` và file câu hỏi. Nạp và tìm bằng cùng model.

```bash
python -m rag.cli ingest --config config.corpus.yaml --recreate
python -m rag.cli eval --config config.corpus.yaml --configs dense,sparse,bm25,dense+sparse
```

Đổi `embedding.model_name`, `embedding.model_path`, `paths.cache_dir`, `bm25.vocab_path` và tên collection, rồi chạy lại hai lệnh. `text_recall_at_k` và `text_mrr` so chất lượng câu chữ. `table_recall_at_k` so tra bảng. `encode_ms` so tốc độ embed. `bm25` là mốc chung vì không phụ thuộc model. Model sparse khác từ điển BGE-M3 thì sửa `sparse_dimension` cho đúng từ điển đó trước khi nạp.

## Thử các trường hợp sau

Mỗi lần đổi dữ liệu, model, cách cắt hoặc kho, chạy lại `ingest`. Thêm `--recreate` khi collection đã có dữ liệu. Câu hỏi chấm theo cụm chữ, nên đổi cách cắt chunk vẫn chấm được. Cách đọc số và cách so từng dòng trong bảng nằm ở mục Đo từng trường hợp.

| Muốn xem | Sửa | Lệnh |
| --- | --- | --- |
| Dense, sparse, BM25, hoặc gộp | Không sửa file. Đổi `--branches` hoặc `--configs` | `dense`, `sparse`, `bm25`, `dense,sparse`, `dense+sparse` |
| Top k, số ứng viên trước RRF, hệ số RRF | `search.top_k`, `search.prefetch_limit`, `search.rrf_k` | `ingest` lại nếu đổi dữ liệu đã lưu; `search` và `eval` đọc các số này lúc chạy |
| Cỡ chunk và phần chồng | `chunking.chunk_size`, `chunking.chunk_overlap` | `ingest --recreate` |
| BM25 theo âm tiết hoặc theo từ | `bm25.mode`: `syllable` hoặc `word` | `ingest --recreate`. Mode `word` dùng pyvi |
| Kho khác | `store`: `qdrant`, `pgvector` hoặc `milvus` | Bật đúng service, rồi `ingest --recreate` |
| Model embedding khác | `embedding.model_name`, `embedding.model_path`, và tên collection riêng | Tải model vào `model_path`, `ingest --recreate`, rồi `search` / `eval` cùng file config |

Nhánh tìm:

```bash
python -m rag.cli search "liều paracetamol cho trẻ sốt" --config config.yaml --branches dense
python -m rag.cli search "liều paracetamol cho trẻ sốt" --config config.yaml --branches bm25
python -m rag.cli eval --config config.yaml --configs dense,sparse,bm25,dense+sparse
```

`dense+sparse` trong `--configs` là một nhóm hybrid. Dấu phẩy tách các nhóm. `eval` đo từng nhóm. `compare` đo RRF gộp ở client trên cùng các nhóm đó.

Ba kho dùng chung `qdrant.hnsw.m`, `qdrant.hnsw.ef_construct`, `qdrant.hnsw_ef`, `qdrant.upsert_batch_size` và `search.rrf_k`. Qdrant gộp RRF trên server và tính thêm IDF cho nhánh `bm25`. pgvector và Milvus gộp bằng hàm `rrf()` trong `rag/fusion.py`, dùng đúng vector BM25 đã lưu.

```bash
docker compose up -d pgvector
docker compose up -d etcd minio milvus
```

Trong config đang dùng, đặt `store: pgvector` hoặc `store: milvus`, rồi:

```bash
python -m rag.cli ingest --config config.corpus.yaml --recreate
python -m rag.cli eval --config config.corpus.yaml --configs dense,sparse,bm25,dense+sparse
```

Đổi model embedding thì nạp và tìm bằng chính model đó. Đặt collection riêng, ví dụ `medical_e5_v1`, để khỏi ghi đè `medical_corpus_v1`. Vector của model A không tìm được bằng model B: hai model không cùng không gian vector, kể cả khi số chiều bằng nhau. Lệnh tìm sẽ dừng nếu tên model trong collection khác model đang dùng.

`hnsw_ef` là `ef` lúc truy vấn, mặc định 128. `full_scan_threshold` để 10 vì Qdrant 1.19.2 không nhận 0. `sparse_dimension` là 250002, bằng kích thước từ điển BGE-M3. Model sparse khác từ điển này thì sửa số đó cho đúng từ điển của model, rồi `ingest --recreate`.

Qdrant ở chế độ `:memory:` (notebook) tìm chính xác, không dựng HNSW. Muốn đo HNSW thì chạy server `docker compose up -d qdrant`.

## Sửa code

| Muốn sửa | File |
| --- | --- |
| Chuẩn hóa chữ | `rag/text/normalizer.py` |
| Cách cắt Markdown và bảng | `rag/text/chunker.py` |
| Model embedding mới | lớp con của `Embedder` trong `rag/encode/`, rồi nối ở `rag/cli.py` |
| Công thức BM25 | `rag/encode/bm25.py` |
| Kho vector mới | lớp con của `VectorStore` trong `rag/store/` |
| Công thức RRF dùng cho pgvector, Milvus và `compare` | `rag/fusion.py` |
| Cách tính recall | `rag/evaluate.py` |
| Câu hỏi chấm | `data/eval/questions.jsonl` hoặc `data/eval/testset.jsonl` |

Một dòng câu hỏi có `qid`, `question`, `answer_keys`. Bộ mẫu thêm `kind` (`text` hoặc `table`). Bộ testset thêm `type`, `doc_id`, `source`. Cụm trong `answer_keys` phải xuất hiện nguyên trong chunk đúng. Câu bảng nên lấy vài cụm của cùng một hàng.

Sau khi sửa hàm thuần, chạy:

```bash
ruff check .
ruff format --check .
pytest -q
```

## Colab

Mở `notebooks/colab_demo.ipynb`. Nén cả thư mục repo thành `RAG.zip`, gồm `rag/`, `config.yaml`, `data/sample`, `data/eval`. Muốn chấm 20 bài thì zip phải có thêm `data/raw` và `data/eval/testset.jsonl`, vì hai chỗ này không nằm trong git.

Upload notebook, upload `RAG.zip` vào `/content`, chọn Runtime → T4 GPU, chạy từ ô đầu. Notebook tải `BAAI/bge-m3`, tự cài Postgres với pgvector, mở Milvus Lite, và so với Qdrant trong bộ nhớ. Lần đầu tải model khoảng 2 GB, cài Postgres khoảng 2 phút.

Một kho lỗi thì ô in dòng lỗi của kho đó và chạy tiếp các kho còn lại. Milvus Lite đi kèm `pymilvus==2.5.18` có thể từ chối collection có cả vector dense và hai vector sparse. Bản Milvus đầy đủ trên máy có Docker là `docker compose up -d etcd minio milvus`.

## Chọn dense BGE-M3 hoặc multilingual E5

Mặc định vẫn dùng BGE-M3 cho cả dense và sparse. Để thử E5, thêm các khóa sau
vào nhóm `embedding` trong YAML đang dùng, giữ các tham số BGE-M3 hiện có:

```yaml
embedding:
  dense_model: e5-base
  e5_model_path: models/multilingual-e5-base
  e5_max_length: 512
  e5_batch_size: 16
```

Tải `intfloat/multilingual-e5-base` vào `e5_model_path` trước khi chạy.
Chọn `paths.cache_dir`, `bm25.vocab_path` và collection của kho đang dùng riêng
cho thử nghiệm E5. `config.corpus.e5.yaml` đã đặt các đường dẫn riêng cho 20 paper.
Đổi `dense_model` về `bge-m3` để trở lại baseline; `--fake-embedder` bỏ qua E5.

Lớp `DenseOverrideEmbedder` trong `pipeline.py` thay dense bằng E5 (768 chiều),
giữ sparse và tokenizer cắt chunk của BGE-M3. BGE vẫn tính dense rồi bỏ qua.
Batch ngoài theo `embedding.batch_size`; E5 chia tiếp theo `e5_batch_size`.
E5 thêm `passage: ` và truncate theo `e5_max_length` (tối đa 512 token, gồm
prefix, header và special tokens); cần tính giới hạn này khi chọn kích thước chunk.

Lựa chọn này áp dụng cho `ingest`, `search`, `eval` và `compare`. Dense query E5
dùng `encode_queries()` với prefix `query: `, sparse query vẫn dùng BGE-M3.
Tên model trong collection và báo cáo ghi cả hai encoder. Tokenizer cắt chunk
vẫn là BGE-M3 để hai lựa chọn dùng cùng các chunk và ground truth.

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('intfloat/multilingual-e5-base', local_dir='models/multilingual-e5-base')"
python -m rag.cli ingest --config config.corpus.e5.yaml
python -m rag.cli search "câu hỏi thử" --config config.corpus.e5.yaml --branches dense,sparse
python -m scripts.score_ground_truth --config config.corpus.e5.yaml --configs dense,sparse,bm25,dense+sparse
```

`config.corpus.e5.yaml` dùng ground truth dạng JSON bên dưới. Muốn dùng lệnh
`rag.cli eval` chấm theo cụm chữ như trước, đặt `eval.questions_path` thành
`data/eval/testset.jsonl` trong một bản sao config đó. Luôn dùng cùng dense model
cho tài liệu và câu hỏi; đổi model cần ingest vào collection/cache riêng.

## Đánh giá bằng ground_truth có nhãn chunk

`extra/ground_truth.json` có 30 câu (18 text, 12 table), gồm 20 câu một tài liệu
và 10 câu nhiều tài liệu. Nhãn gắn với snapshot 432 chunk trong `extra/chunks.json`.
Đây là nhãn do assistant rà soát, chưa phải bộ nhãn được chuyên gia đánh giá độc lập.
Script kiểm tra SHA-256 của toàn bộ chunk và từ chối chấm khi snapshot khác.
Không thay nhãn bằng kết quả truy xuất hay dò `reference_answer` trong chunk.

```bash
python -m rag.cli ingest --config config.corpus.ground-truth.yaml
python -m scripts.score_ground_truth --config config.corpus.ground-truth.yaml --configs dense,sparse,bm25,dense+sparse
```

Config này dùng BGE-M3 thật, batch 2 cho máy ít RAM, 3 lượt warmup và 30 lượt đo
độ trễ mỗi nhánh. Toàn bộ 30 câu đều được chấm tại K=1, 5 và `search.top_k`
(mặc định 10). Có thể tăng `eval.latency_runs` lên 500 để đo độ trễ ổn định hơn;
p95/p99 từ 30 mẫu chỉ mang tính tham khảo.

| Metric | Cách tính |
| --- | --- |
| `success_at_k` | Tỷ lệ câu có ít nhất một chunk được gán nhãn relevant trong top K |
| `mrr_at_k` | Trung bình nghịch đảo hạng chunk relevant đầu tiên; không có trong top K thì bằng 0 |
| `recall_at_k` | Số chunk relevant khác nhau lấy được / tổng số chunk relevant được gán nhãn, trung bình theo câu |
| `ndcg_at_k` | nDCG nhị phân, có chiết khấu theo hạng, không cộng điểm chunk lặp |
| `evidence_coverage_at_k` | Trung bình tỷ lệ nhóm bằng chứng bắt buộc đã lấy được |
| `all_evidence_success_at_k` | Tỷ lệ câu lấy được ít nhất một chunk của **mọi** nhóm bằng chứng bắt buộc |

JSON trong `reports/` chứa kết quả tổng, theo text/table, theo câu một/nhiều tài liệu
và từng câu với danh sách chunk truy xuất. `latency.encode_ms`, `query_ms`, `total_ms`
đều có p50, p95, p99 (mili giây), bỏ các lượt warmup. `total_ms` đo thời gian thực
của một lần `search_timed`, gồm chuẩn hóa, encode và truy vấn; không cộng các phân vị
của encode/query. Đo tuần tự, không phải benchmark tải đồng thời hay throughput.

Lệnh `rag.cli eval` cũ cũng ghi thêm `text_success_at_k`, `table_success_at_k`,
`text_mrr_at_k`, `table_mrr_at_k` và `latency.total_ms`; vẫn giữ các khóa cũ.

Để kiểm tra spec khi cần tạo lại nhãn, ghi ra một file mới:

```bash
python -m scripts.build_ground_truth --chunks extra/chunks.json --spec extra/ground_truth_spec.json --output reports/ground_truth_rebuilt.json
```
