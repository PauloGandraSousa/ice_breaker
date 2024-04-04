from dotenv import load_dotenv
import os

print(os.environ["OPENAI_API_KEY"])

print("\n**** system ****\n")
for key, value in os.environ.items():
    print(f"{key}: {value}")

load_dotenv()
