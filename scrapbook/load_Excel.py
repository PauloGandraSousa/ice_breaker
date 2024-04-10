from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredFileLoader

print("before creating the loader")
# loader = UnstructuredExcelLoader("docs/examples/simple.xlsx", mode="elements")
loader = UnstructuredFileLoader("docs/examples/simple.xlsx", mode="elements")
print(f"loader {loader}")
docs = loader.load()
print(f"docs {docs}")
