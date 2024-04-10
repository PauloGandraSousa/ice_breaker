from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredFileLoader

print("before creating the loader")
loader = UnstructuredFileLoader("../docs/examples/simple.csv", mode="elements")
print(f"loader {loader}")
docs = loader.load()
print(f"docs {docs}")
