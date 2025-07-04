import os
import json
import chromadb
import requests

# --- Constants and Configuration ---
# Thư mục để lưu trữ cơ sở dữ liệu vector ChromaDB
CHROMA_DIR = "chroma_db_cars" # Change directory name to avoid conflicts with other projects
# Tên của bộ sưu tập (collection) trong ChromaDB để lưu trữ car data
COLLECTION_NAME = "cars"
# Tên của tệp JSON chứa dữ liệu ban đầu về car
JSON_FILE = "cars.json"
# Tên của mô hình nhúng (embedding model) được sử dụng thông qua Ollama
EMBED_MODEL = "mxbai-embed-large"
# Tên của mô hình ngôn ngữ lớn (LLM) được sử dụng thông qua Ollama để tạo câu trả lời
LLM_MODEL = "llama3.2"

# --- Load initial data ---
# Mở tệp JSON chứa car data và tải nội dung vào biến car_data
with open(JSON_FILE, "r", encoding="utf-8") as f:
    car_data = json.load(f)

# --- ChromaDB Setup ---
# Khởi tạo một ChromaDB client. PersistentClient đảm bảo dữ liệu được lưu trữ trên đĩa.
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
# Lấy hoặc tạo một collection trong ChromaDB.
# Nếu collection 'cars' không tồn tại, nó sẽ được tạo.
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# --- Hàm tạo Embeddings với Ollama ---
# Hàm này giao tiếp với API Ollama để tạo bản nhúng (embedding) cho một đoạn văn bản.
def get_embedding(text):
    try:
        # Gửi một POST request đến endpoint /api/embeddings của Ollama
        response = requests.post("http://localhost:11434/api/embeddings", json={
            "model": EMBED_MODEL,  # Chỉ định embedding model để sử dụng
            "prompt": text         # Văn bản để tạo embedding
        })
        response.raise_for_status() # Kiểm tra lỗi HTTP
        data = response.json()
        # print("Embedding API response:", data)  # Uncomment this line for debugging if needed
        if "embedding" not in data:
            raise ValueError(f"Embedding API error: 'embedding' key missing in response: {data}")
        # Return the embedding vector
        return data["embedding"]
    except requests.exceptions.ConnectionError:
        print("🚫 Connection error to Ollama. Please ensure Ollama is running.")
        print("   You can check by opening Terminal/Command Prompt and typing: ollama run llama3.2")
        exit() # Exit the program if connection fails
    except Exception as e:
        print(f"🚫 Error getting embedding: {e}")
        exit() # Exit the program if another error occurs

# --- Add new data to ChromaDB (if any) ---
# Lấy tất cả các ID tài liệu hiện có trong ChromaDB collection
# Sử dụng try-except để xử lý các trường hợp collection trống hoặc có lỗi với get()
try:
    existing_ids = set(collection.get(ids=[item['id'] for item in car_data])['ids'])
except:
    existing_ids = set() # Nếu có lỗi hoặc collection trống, giả sử không có ID nào tồn tại

# Lọc các mục mới từ car_data chưa có trong ChromaDB
new_items = [item for item in car_data if item['id'] not in existing_ids]

# Kiểm tra xem có bất kỳ mục mới nào để thêm không
if new_items:
    print(f"🆕 Adding {len(new_items)} new documents to ChromaDB...")
    # Lặp lại từng mục mới
    for item in new_items:
        # Bắt đầu với mô tả văn bản gốc của car
        enriched_text = item["text"]
        # Làm phong phú văn bản bằng cách thêm thông tin make nếu có
        if "make" in item:
            enriched_text += f" This car is made by {item['make']}."
        # Làm phong phú văn bản bằng cách thêm thông tin car type nếu có
        if "type" in item:
            enriched_text += f" It is a type of {item['type']}."
        # Thêm thông tin số chỗ ngồi nếu có
        if "seat" in item:
            enriched_text += f" It has {item['seat']} seats."
        # Thêm thông tin loại nhiên liệu nếu có
        if "fuel_type" in item:
            enriched_text += f" It uses {item['fuel_type']} fuel."

        # Tạo một embedding cho văn bản đã được làm phong phú
        emb = get_embedding(enriched_text)

        # Thêm tài liệu vào ChromaDB collection
        # documents: Văn bản thực tế sẽ được truy xuất làm context
        # embeddings: Embedding được sử dụng để tìm kiếm sự tương đồng
        # ids: ID duy nhất của tài liệu
        collection.add(
            documents=[item["text"]],  # Sử dụng văn bản gốc làm context có thể truy xuất
            embeddings=[emb],          # Sử dụng embedding của văn bản đã được làm phong phú để tìm kiếm
            ids=[item["id"]]           # ID của car
        )
    print("✅ New documents added successfully.")
else:
    print("✅ All documents already in ChromaDB.")

# --- RAG Query Function (Retrieval-Augmented Generation) ---
# Hàm này thực hiện toàn bộ quy trình RAG để trả lời một câu hỏi
def rag_query(question):
    # Step 1: Embed the user question
    # Chuyển đổi câu hỏi của người dùng thành một embedding vector
    q_emb = get_embedding(question)

    # Step 2: Query the vector database (ChromaDB)
    # Tìm kiếm các tài liệu tương tự nhất với embedding của câu hỏi.
    # n_results=3: Trả về 3 tài liệu liên quan nhất.
    results = collection.query(query_embeddings=[q_emb], n_results=3)

    # Step 3: Extract relevant documents
    # Lấy nội dung văn bản của các tài liệu hàng đầu
    top_docs = results['documents'][0]
    # Lấy ID của các tài liệu hàng đầu
    top_ids = results['ids'][0]

    # Step 4: Display friendly explanation of retrieved documents
    print("\n🧠 Retrieving relevant information to answer your question...\n")
    for i, doc in enumerate(top_docs):
        print(f"🔹 Source {i + 1} (ID: {top_ids[i]}):")
        print(f"    \"{doc}\"\n")
    print("📚 These seem to be the most relevant pieces of information to answer your question.\n")

    # Step 5: Build the prompt from the retrieved context
    # Nối các tài liệu đã truy xuất thành một chuỗi context duy nhất
    context = "\n".join(top_docs)

    # Tạo prompt cho LLM. Prompt bao gồm context và câu hỏi,
    # hướng dẫn LLM chỉ sử dụng context được cung cấp để trả lời.
    prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""

    # Step 6: Generate answer with Ollama (using the LLM)
    try:
        # Gửi một POST request đến endpoint /api/generate của Ollama
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": LLM_MODEL,  # Chỉ định LLM model để sử dụng
            "prompt": prompt,    # Prompt đã được tăng cường context
            "stream": False      # Không stream phản hồi, đợi phản hồi đầy đủ
        })
        response.raise_for_status() # Kiểm tra lỗi HTTP
        # Step 7: Return the final result
        # Trích xuất câu trả lời từ phản hồi JSON và loại bỏ khoảng trắng đầu/cuối
        return response.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        return "🚫 Connection error to Ollama when generating answer. Please ensure Ollama is running."
    except Exception as e:
        return f"🚫 Error generating answer: {e}"


# --- Interactive Loop ---
print("\n🧠 Car RAG is ready. Ask a question (type 'exit' to quit):\n")
while True:
    # Lấy câu hỏi từ người dùng
    question = input("You: ")
    # Thoát nếu người dùng nhập 'exit' hoặc 'quit'
    if question.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break
    # Gọi hàm rag_query để lấy câu trả lời
    answer = rag_query(question)
    # In câu trả lời từ LLM
    print("🤖:", answer)
