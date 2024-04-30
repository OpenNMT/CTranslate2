import os
import sys
import ctranslate2
from transformers import AutoTokenizer
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton, QWidget
from PySide6.QtGui import QTextCursor, QCloseEvent
from PySide6.QtCore import Signal, QObject, QThread

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MODEL_LOCATION = [specify local path or huggingface repo id] # local path to ct2 model or huggingface repo id
SYSTEM_PROMPT = "Your are a helpful and courteous assistant who tries to help the person you're speaking with." # change to your liking
DEVICE = "cuda" # set to cpu if you want
COMPUTE_TYPE = "int8" # set to float16 etc.
INTRA_THREADS = 4 # set number of cpu threads; ok to keep this parameter even when using CUDA
FLASH_ATTENTION = True # Can only use "True" on Ampere or later GPUs


class ChatLlama3:
    def __init__(self):
        print("Loading the model...")
        self.generator = ctranslate2.Generator(MODEL_LOCATION, device=DEVICE, compute_type=COMPUTE_TYPE, intra_threads=INTRA_THREADS, flash_attention=FLASH_ATTENTION)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCATION)
        self.context_length = 4096
        self.max_generation_length = 512
        self.max_prompt_length = self.context_length - self.max_generation_length
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eot_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        self.end_tokens = [self.eos_token_id, self.eot_token_id]
        self.messages = []

        if SYSTEM_PROMPT:
            self.messages.append({"role": "system", "content": SYSTEM_PROMPT})

    def generate_response(self, user_prompt, response_callback=None):
        self.messages.append({"role": "user", "content": user_prompt})

        while True:
            input_ids = self.tokenizer.apply_chat_template(
                self.messages,
                add_generation_prompt=True,
                return_tensors="np"
            )

            if len(input_ids[0]) <= self.max_prompt_length:
                break

            # Remove old conversations when prompt size becomes too large
            if SYSTEM_PROMPT:
                self.messages = [self.messages[0]] + self.messages[3:]
            else:
                self.messages = self.messages[2:]

        prompt_tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids.tolist()]

        step_results = self.generator.generate_tokens(
            prompt_tokens,
            max_length=self.max_generation_length,
            sampling_temperature=0.6,
            sampling_topk=20,
            sampling_topp=1,
            end_token=self.end_tokens,
        )

        text_output = ""
        for word in self.generate_words(step_results):
            text_output += word
            if response_callback:
                response_callback(word)
            else:
                print(word, end="", flush=True)

        self.messages.append({"role": "assistant", "content": text_output.strip()})
        return text_output.strip()

    def generate_words(self, step_results):
        tokens_buffer = []

        for step_result in step_results:
            is_new_word = step_result.token.startswith("Ġ")
            if is_new_word and tokens_buffer:
                word = self.tokenizer.decode(tokens_buffer)
                if word:
                    yield word
                tokens_buffer = []
            tokens_buffer.append(step_result.token_id)

        if tokens_buffer:
            word = self.tokenizer.decode(tokens_buffer)
            if word:
                yield word

class ChatWorker(QObject):
    response_generated = Signal(str)

    def __init__(self, chat_llama3):
        super().__init__()
        self.chat_llama3 = chat_llama3

    def generate_response(self, user_prompt):
        self.chat_llama3.generate_response(user_prompt, self.response_callback)

    def response_callback(self, word):
        self.response_generated.emit(word)

class ChatLlama3GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chat with Llama3")
        self.setGeometry(100, 100, 600, 800)

        self.chat_llama3 = ChatLlama3()
        self.chat_worker = ChatWorker(self.chat_llama3)
        self.thread = QThread()
        self.chat_worker.moveToThread(self.thread)
        self.thread.start()

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.chat_history = QTextEdit(self)
        self.chat_history.setReadOnly(True)
        layout.addWidget(self.chat_history)

        input_layout = QHBoxLayout()

        self.user_input = QLineEdit(self)
        self.user_input.setPlaceholderText("Type your message...")
        self.user_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.user_input)

        send_button = QPushButton("Send", self)
        send_button.clicked.connect(self.send_message)
        input_layout.addWidget(send_button)

        layout.addLayout(input_layout)

        self.chat_worker.response_generated.connect(self.update_chat_history)

    def send_message(self):
        user_prompt = self.user_input.text()
        self.user_input.clear()
        self.chat_history.append(f"You: {user_prompt}")
        self.chat_history.append("Llama3: ")
        self.chat_worker.generate_response(user_prompt)

    def update_chat_history(self, word):
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chat_history.setTextCursor(cursor)
        self.chat_history.insertPlainText(word)
        QApplication.processEvents()

    def closeEvent(self, event: QCloseEvent):
        self.thread.quit()
        self.thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    chat_gui = ChatLlama3GUI()
    chat_gui.show()
    sys.exit(app.exec())