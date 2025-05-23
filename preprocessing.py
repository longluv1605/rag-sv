import os
from dotenv import load_dotenv
import google.generativeai as genai
from rich.console import Console
from rich.markdown import Markdown
import json


def load_done_file(path='done_files.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        done_files = set(line.strip() for line in f if line.strip())
    return done_files

done_files_path = 'done_files.txt'
done_files = load_done_file(done_files_path)

console = Console()

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL = os.getenv("MODEL_GEMINI_2_FLASH")

genai.configure(api_key=API_KEY)

generation_config = {
    'temperature': 0.5,
    'response_mime_type': 'text/plain',
}

gemini = genai.GenerativeModel(
    model_name=MODEL,
    generation_config=generation_config,
    system_instruction="Answer in Vietnamese. Đưa ra câu trả lời trực tiếp, không rườm rà. Câu trả lời ở định dạng dành cho file txt/plain chứ không phải markdown. Và lưu ý là không bỏ sót thông tin."
)

data_dir = 'source_data'
cln_data_dir = 'source_data'
qa = []
for fname in os.listdir(data_dir):
    try:
        if fname in done_files:
            continue
        else:
            done_files.add(fname)
            
        fpath = os.path.join(data_dir, fname)
        cln_fpath = f'{cln_data_dir}/{fname}'
        
        with open(fpath, 'r', encoding='utf-8') as f:
            text = f.readlines()
            
            prompt = f"Dưới đây là dữ liệu đã được thu thập nhưng ở dạng thô, hãy làm sạch (chỉnh sửa lại, chứ không phải gợi ý code) để tôi có thể lấy câu trả lời của bạn và lưu vào txt và tiến hành embedding cho RAG với LLMs và nếu dữ liệu không có giá trị, hãy trả về NULL (lưu ý, ký tự | thể hiện cho bảng):\n{text}"
            response = gemini.generate_content(prompt)
            # console.print(Markdown(response.text))
            if response.text != 'NULL\n':
                with open(cln_fpath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
            
            prompt = '''Dưới đây là dữ liệu mà tôi có, hãy tạo cặp question/answer từ dữ liệu, ở định dạng json. Hãy đưa ra câu trả lời trực tiếp. Lưu ý, câu hỏi và câu trả lời không nên quá dài, khoảng dưới 50 từ:\n''' + response.text
            response = gemini.generate_content(prompt)
            # console.print(Markdown(response.text))
            # print(response.text)
            qa.extend(json.loads(response.text.strip().removeprefix("```json").removesuffix("```")))
        with open("qa.json", "w", encoding="utf-8") as f:
            json.dump(qa, f, ensure_ascii=False, indent=2)
        
        with open(done_files_path, 'w', encoding='utf-8') as f:
            for done_file in done_files:
                f.write(f'{done_file}\n')
        print(f'Done {fname}')
    except Exception as e:
        print(f'Error at {fname}: {e}')